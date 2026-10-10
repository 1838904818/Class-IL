"""Fail-closed aggregation for the full Exp 3 measurement matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DATASETS = ["NSL-KDD", "UNSW-NB15", "CIC-IDS-2017", "CSE-CIC-IDS2018"]
METHODS = ["naive", "ewc", "lwf", "replay", "replay-herding", "replay-dpmeans", "icarl"]
SEEDS = [1, 2, 3, 4, 42]
SILENCE_TOLERANCES = [0.0, 0.02, 0.05, 0.10]
NCD_Q_THRESHOLDS = [0.01, 0.025, 0.05, 0.10]
TOP_K_VALUES = [5, 10, 15, 20]
JACCARD_THRESHOLDS = [0.50, 0.60, 0.70, 0.80]


def identity(row: dict[str, object]) -> tuple[str, str, int]:
    return str(row["dataset"]), str(row["method"]), int(row["seed"])


def expected_identities() -> set[tuple[str, str, int]]:
    return {(dataset, method, seed) for dataset in DATASETS for method in METHODS for seed in SEEDS}


def mean_std(values: Iterable[float | int | None]) -> tuple[float | None, float | None]:
    cleaned = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not cleaned:
        return None, None
    return statistics.fmean(cleaned), statistics.stdev(cleaned) if len(cleaned) > 1 else 0.0


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_artifact(result_path: Path, result: dict[str, object], key: str) -> Path:
    recorded = Path(str(result["artifacts"][key]))
    candidates = [recorded, result_path.parent / recorded.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"cannot locate {key} artifact for {result_path}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def check_coverage(
    drift: dict[tuple[str, str, int], dict[str, object]],
    performance: dict[tuple[str, str, int], dict[str, object]],
    allow_partial: bool,
) -> dict[str, object]:
    expected = expected_identities()
    drift_keys = set(drift)
    performance_keys = set(performance)
    if drift_keys != performance_keys:
        raise ValueError(
            "drift/performance identity mismatch: "
            f"drift_only={sorted(drift_keys - performance_keys)} "
            f"performance_only={sorted(performance_keys - drift_keys)}"
        )
    missing = expected - drift_keys
    extra = drift_keys - expected
    if extra:
        raise ValueError(f"unexpected Exp 3 identities: {sorted(extra)}")
    if missing and not allow_partial:
        raise ValueError(f"incomplete Exp 3 matrix: missing {len(missing)} of {len(expected)} runs")
    return {
        "expected_runs": len(expected),
        "observed_runs": len(drift_keys),
        "complete": not missing,
        "missing": [list(item) for item in sorted(missing)],
    }


def weighted_rate(rows: list[dict[str, object]], rate_key: str, weight_key: str) -> float | None:
    weight = sum(int(row[weight_key]) for row in rows)
    if not weight:
        return None
    return sum(float(row[rate_key]) * int(row[weight_key]) for row in rows) / weight


def group_explainer_units(
    explainer_rows: list[dict[str, str]],
) -> dict[tuple[str, str, int], list[tuple[float, list[dict[str, str]]]]]:
    units: dict[tuple[str, str, int, int, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in explainer_rows:
        key = (
            row["dataset"],
            row["method"],
            int(row["seed"]),
            int(row["checkpoint_before"]),
            int(row["checkpoint_after"]),
            int(row["class_id"]),
        )
        units[key].append(row)

    by_run: dict[tuple[str, str, int], list[tuple[float, list[dict[str, str]]]]] = defaultdict(list)
    for key, rows in units.items():
        explainers = {row["explainer"] for row in rows}
        if len(rows) != 3 or len(explainers) != 3:
            raise ValueError(f"incomplete three-explainer unit for {key}: {sorted(explainers)}")
        recall_deltas = {float(row["recall_delta"]) for row in rows}
        if len(recall_deltas) != 1:
            raise ValueError(f"inconsistent recall delta for {key}")
        by_run[key[:3]].append((recall_deltas.pop(), rows))
    return by_run


def ncd_q_sensitivity_rows(explainer_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Recompute cross-explainer NCD consensus over registered FDR thresholds."""
    by_run = group_explainer_units(explainer_rows)

    output: list[dict[str, object]] = []
    for (dataset, method, seed), run_units in sorted(by_run.items()):
        for tolerance in SILENCE_TOLERANCES:
            silent = [rows for recall_delta, rows in run_units if recall_delta > -tolerance]
            for threshold in NCD_Q_THRESHOLDS:
                votes = [
                    sum(
                        float(row["q_value"]) < threshold
                        and float(row["ncd_effect"]) > 0.0
                        for row in rows
                    )
                    for rows in silent
                ]
                drift_units = sum(vote_count >= 2 for vote_count in votes)
                agreement_units = sum(vote_count in (0, 3) for vote_count in votes)
                output.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "silence_tolerance": tolerance,
                        "q_threshold": threshold,
                        "silent_units": len(silent),
                        "ncd_consensus_drift_units": drift_units,
                        "ncd_consensus_drift_rate": (
                            drift_units / len(silent) if silent else None
                        ),
                        "ncd_all_explainer_agreement_rate": (
                            agreement_units / len(silent) if silent else None
                        ),
                    }
                )
    return output


def fixed_consensus_sensitivity_rows(
    explainer_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    """Recompute B1/B2 consensus sensitivity over top-k and Jaccard thresholds."""
    by_run = group_explainer_units(explainer_rows)
    output: list[dict[str, object]] = []
    for (dataset, method, seed), run_units in sorted(by_run.items()):
        for tolerance in SILENCE_TOLERANCES:
            silent = [rows for recall_delta, rows in run_units if recall_delta > -tolerance]
            for top_k in TOP_K_VALUES:
                for threshold in JACCARD_THRESHOLDS:
                    b1_votes = [
                        sum(float(row[f"b1_jaccard_k{top_k}"]) < threshold for row in rows)
                        for rows in silent
                    ]
                    b2_votes = [
                        sum(float(row[f"b2_jaccard_k{top_k}"]) < threshold for row in rows)
                        for rows in silent
                    ]
                    b1_drift = sum(votes >= 2 for votes in b1_votes)
                    b2_drift = sum(votes >= 2 for votes in b2_votes)
                    output.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "seed": seed,
                            "silence_tolerance": tolerance,
                            "top_k": top_k,
                            "jaccard_threshold": threshold,
                            "silent_units": len(silent),
                            "b1_consensus_drift_units": b1_drift,
                            "b1_consensus_drift_rate": (
                                b1_drift / len(silent) if silent else None
                            ),
                            "b2_consensus_drift_units": b2_drift,
                            "b2_consensus_drift_rate": (
                                b2_drift / len(silent) if silent else None
                            ),
                            "b1_all_explainer_agreement_rate": (
                                sum(votes in (0, 3) for votes in b1_votes) / len(silent)
                                if silent
                                else None
                            ),
                            "b2_all_explainer_agreement_rate": (
                                sum(votes in (0, 3) for votes in b2_votes) / len(silent)
                                if silent
                                else None
                            ),
                        }
                    )
    return output


def summarize_group(
    drift_rows: list[dict[str, object]],
    performance_rows: list[dict[str, object]],
    consensus_rows: list[dict[str, str]],
    cost_rows: list[dict[str, object]],
) -> dict[str, object]:
    silent = sum(int(row["silent_units_5pp"]) for row in drift_rows)
    b1 = sum(int(row["b1_consensus_drift_units"]) for row in drift_rows)
    b2 = sum(int(row["b2_consensus_drift_units"]) for row in drift_rows)
    ncd = sum(int(row["ncd_consensus_drift_units"]) for row in drift_rows)
    future_positive: list[float] = []
    future_negative: list[float] = []
    for row in consensus_rows:
        if str(row.get("silent_at_5pp", "")).lower() != "true":
            continue
        value = row.get("future_recall_delta", "")
        if value in (None, ""):
            continue
        target = future_positive if str(row.get("ncd_consensus_drift", "")).lower() == "true" else future_negative
        target.append(float(value))

    summary: dict[str, object] = {
        "seeds": sorted(int(row["seed"]) for row in drift_rows),
        "runs": len(drift_rows),
        "silent_units_5pp": silent,
        "b1_drift_units": b1,
        "b1_drift_rate": b1 / silent if silent else None,
        "b2_drift_units": b2,
        "b2_drift_rate": b2 / silent if silent else None,
        "ncd_drift_units": ncd,
        "ncd_drift_rate": ncd / silent if silent else None,
        "b1_all_explainer_agreement_rate": weighted_rate(
            drift_rows, "b1_all_explainer_agreement_rate", "silent_units_5pp"
        ),
        "b2_all_explainer_agreement_rate": weighted_rate(
            drift_rows, "b2_all_explainer_agreement_rate", "silent_units_5pp"
        ),
        "ncd_all_explainer_agreement_rate": weighted_rate(
            drift_rows, "ncd_all_explainer_agreement_rate", "silent_units_5pp"
        ),
        "future_units_after_ncd_drift": len(future_positive),
        "future_units_without_ncd_drift": len(future_negative),
        "median_future_recall_delta_after_ncd_drift": (
            statistics.median(future_positive) if future_positive else None
        ),
        "median_future_recall_delta_without_ncd_drift": (
            statistics.median(future_negative) if future_negative else None
        ),
    }
    for field in (
        "final_accuracy_seen",
        "final_balanced_accuracy_seen",
        "final_macro_f1_seen",
        "final_normal_recall",
        "average_forgetting_prior_classes",
    ):
        mean, standard_deviation = mean_std(row.get(field) for row in performance_rows)
        summary[f"{field}_mean"] = mean
        summary[f"{field}_std"] = standard_deviation
    for field in (
        "training_seconds",
        "explanation_seconds",
        "fixed_rule_seconds",
        "ncd_permutation_seconds",
    ):
        mean, standard_deviation = mean_std(row.get(field) for row in cost_rows)
        summary[f"{field}_mean"] = mean
        summary[f"{field}_std"] = standard_deviation
        summary[f"{field}_total"] = sum(float(row.get(field, 0.0) or 0.0) for row in cost_rows)
    fixed_total = float(summary["fixed_rule_seconds_total"])
    summary["ncd_to_fixed_rule_time_ratio"] = (
        float(summary["ncd_permutation_seconds_total"]) / fixed_total if fixed_total else None
    )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(
    drift_paths: list[Path],
    performance_paths: list[Path],
    allow_partial: bool,
) -> dict[str, object]:
    drift: dict[tuple[str, str, int], dict[str, object]] = {}
    performance: dict[tuple[str, str, int], dict[str, object]] = {}
    costs: dict[tuple[str, str, int], dict[str, object]] = {}
    consensus: list[dict[str, str]] = []
    sensitivity: list[dict[str, str]] = []
    explainer_units: list[dict[str, str]] = []
    inputs: list[dict[str, str]] = []

    for path in drift_paths:
        resolved = path.resolve(strict=True)
        result = read_json(resolved)
        inputs.append({"role": "drift", "path": str(resolved), "sha256": sha256(resolved)})
        for row in result["run_summaries"]:
            key = identity(row)
            if key in drift:
                raise ValueError(f"duplicate drift result for {key}")
            drift[key] = row
        for row in result["runs"]:
            key = identity(row)
            if key in costs:
                raise ValueError(f"duplicate drift cost result for {key}")
            costs[key] = row
        consensus.extend(read_csv(locate_artifact(resolved, result, "consensus_units")))
        sensitivity.extend(read_csv(locate_artifact(resolved, result, "threshold_sensitivity")))
        explainer_units.extend(read_csv(locate_artifact(resolved, result, "explainer_units")))

    for path in performance_paths:
        resolved = path.resolve(strict=True)
        result = read_json(resolved)
        inputs.append({"role": "performance", "path": str(resolved), "sha256": sha256(resolved)})
        dataset = str(result["dataset"])
        for raw in result["runs"]:
            row = {"dataset": dataset, **raw}
            key = identity(row)
            if key in performance:
                raise ValueError(f"duplicate performance result for {key}")
            performance[key] = row

    coverage = check_coverage(drift, performance, allow_partial)
    ncd_sensitivity = ncd_q_sensitivity_rows(explainer_units)
    fixed_consensus_sensitivity = fixed_consensus_sensitivity_rows(explainer_units)
    summaries: list[dict[str, object]] = []
    observed_groups = sorted(
        {(dataset, method) for dataset, method, _ in drift},
        key=lambda item: (DATASETS.index(item[0]), METHODS.index(item[1])),
    )
    for dataset, method in observed_groups:
        keys = sorted(
            [key for key in drift if key[:2] == (dataset, method)], key=lambda key: key[2]
        )
        group_consensus = [
            row for row in consensus
            if row["dataset"] == dataset and row["method"] == method
        ]
        summaries.append(
            {
                "dataset": dataset,
                "method": method,
                **summarize_group(
                    [drift[key] for key in keys],
                    [performance[key] for key in keys],
                    group_consensus,
                    [costs[key] for key in keys],
                ),
            }
        )

    return {
        "analysis": "Exp 3 full-matrix performance, forgetting, drift, agreement, sensitivity, and cost aggregation",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "coverage": coverage,
        "inputs": inputs,
        "method_summaries": summaries,
        "consensus_units": consensus,
        "threshold_sensitivity_rows": sensitivity,
        "ncd_q_sensitivity_rows": ncd_sensitivity,
        "fixed_consensus_sensitivity_rows": fixed_consensus_sensitivity,
        "limitations": [
            "Drift-to-future-recall comparisons are descriptive and do not establish prediction or causality.",
            "NCD significance or agreement does not by itself establish detector validity or substantive innovation.",
            "Exp 2 remains the decisive source for known-positive and known-negative validity claims.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drift-result", type=Path, action="append", required=True)
    parser.add_argument("--performance-result", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    result = aggregate(args.drift_result, args.performance_result, args.allow_partial)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "exp3_full_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "exp3_method_summary.csv", result["method_summaries"])
    write_csv(args.output_dir / "exp3_consensus_units.csv", result["consensus_units"])
    write_csv(args.output_dir / "exp3_threshold_sensitivity.csv", result["threshold_sensitivity_rows"])
    write_csv(args.output_dir / "exp3_ncd_q_sensitivity.csv", result["ncd_q_sensitivity_rows"])
    write_csv(
        args.output_dir / "exp3_fixed_consensus_sensitivity.csv",
        result["fixed_consensus_sensitivity_rows"],
    )
    print(
        "EXP3_AGGREGATION_COMPLETE "
        f"observed_runs={result['coverage']['observed_runs']} "
        f"expected_runs={result['coverage']['expected_runs']} "
        f"complete={str(result['coverage']['complete']).lower()} output={result_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
