"""Aggregate the preregistered Section 12 CRR ground-truth benchmarks.

This script deliberately leaves the real-sequence gate pending.  Section 12.8
places that evidence in the day-10 deliverable, so the day-5 benchmark matrix
cannot by itself establish the full Section 12.7 claim.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

REGISTERED_SEEDS = (1, 2, 3, 4, 42)
REGISTERED_METHODS = (
    "replay",
    "icarl",
    "replay_lwf",
    "kd_perturbed",
    "rcl_tab",
    "global_ir",
    "oldpair_kd",
    "crr",
)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _one_sided_sign_flip_p(differences: list[float]) -> float:
    """Exact paired randomization p-value for H1: mean(difference) < 0."""

    observed = _mean(differences)
    permuted = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        permuted.append(_mean([sign * value for sign, value in zip(signs, differences)]))
    return sum(value <= observed + 1e-15 for value in permuted) / len(permuted)


def _load(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
    return rows


def aggregate(rows: list[dict]) -> dict:
    lookup: dict[tuple, dict] = {}
    for row in rows:
        strength = row.get("rho") if row["benchmark"] == "hijack" else row.get("k")
        key = (row["dataset"], int(row["seed"]), row["method"], row["benchmark"], float(strength))
        if key in lookup:
            raise ValueError(f"duplicate result cell: {key}")
        lookup[key] = row

    datasets = sorted({str(row["dataset"]) for row in rows})
    missing = []
    for dataset in datasets:
        for seed in REGISTERED_SEEDS:
            for method in REGISTERED_METHODS:
                for rho in (0.5, 0.7, 0.9, 1.0):
                    if (dataset, seed, method, "hijack", rho) not in lookup:
                        missing.append((dataset, seed, method, "hijack", rho))
                for k in (1.0, 2.0, 3.0):
                    if (dataset, seed, method, "necessity", k) not in lookup:
                        missing.append((dataset, seed, method, "necessity", k))
    if missing:
        raise ValueError(f"missing {len(missing)} registered cells; first={missing[0]}")

    criterion_1_details = []
    criterion_1_pass = True
    for dataset in datasets:
        for rho in (0.7, 0.9, 1.0):
            means = {}
            for method in ("crr", "rcl_tab", "global_ir", "oldpair_kd", "kd_perturbed"):
                means[method] = _mean(
                    [
                        max(0.0, float(lookup[(dataset, seed, method, "hijack", rho)]["marker_recall_drop"]))
                        for seed in REGISTERED_SEEDS
                    ]
                )
            comparable = all(means["crr"] <= 1.2 * means[name] + 1e-12 for name in ("rcl_tab", "global_ir"))
            lower = all(means["crr"] <= 0.5 * means[name] + 1e-12 for name in ("oldpair_kd", "kd_perturbed"))
            passed = comparable and lower
            criterion_1_pass = criterion_1_pass and passed
            criterion_1_details.append({"dataset": dataset, "rho": rho, "means": means, "passed": passed})

    criterion_2_details = []
    criterion_2_pass = True
    for dataset in datasets:
        crr_by_seed = {
            seed: _mean(
                [float(lookup[(dataset, seed, "crr", "necessity", k)]["pair_confusion_rate"]) for k in (2.0, 3.0)]
            )
            for seed in REGISTERED_SEEDS
        }
        tests = {}
        for baseline in ("rcl_tab", "global_ir"):
            differences = []
            for seed in REGISTERED_SEEDS:
                baseline_value = _mean(
                    [
                        float(lookup[(dataset, seed, baseline, "necessity", k)]["pair_confusion_rate"])
                        for k in (2.0, 3.0)
                    ]
                )
                differences.append(crr_by_seed[seed] - baseline_value)
            tests[baseline] = {
                "mean_difference": _mean(differences),
                "one_sided_exact_p": _one_sided_sign_flip_p(differences),
                "passed": _mean(differences) < 0.0 and _one_sided_sign_flip_p(differences) < 0.05,
            }
        recall_checks = {}
        for label in ("c_recall", "d_recall"):
            crr_mean = _mean(
                [
                    float(lookup[(dataset, seed, "crr", "necessity", k)][label])
                    for seed in REGISTERED_SEEDS
                    for k in (2.0, 3.0)
                ]
            )
            replay_mean = _mean(
                [
                    float(lookup[(dataset, seed, "replay", "necessity", k)][label])
                    for seed in REGISTERED_SEEDS
                    for k in (2.0, 3.0)
                ]
            )
            recall_checks[label] = {
                "crr": crr_mean,
                "replay": replay_mean,
                "passed": crr_mean + 1e-12 >= replay_mean,
            }
        passed = all(item["passed"] for item in tests.values()) and all(
            item["passed"] for item in recall_checks.values()
        )
        criterion_2_pass = criterion_2_pass and passed
        criterion_2_details.append(
            {"dataset": dataset, "paired_tests": tests, "recall_checks": recall_checks, "passed": passed}
        )

    costs = defaultdict(list)
    for row in rows:
        costs[(row["dataset"], row["method"])].append(float(row["cost_vs_replay"]))
    criterion_4_details = [
        {
            "dataset": dataset,
            "crr_mean_cost_vs_replay": _mean(costs[(dataset, "crr")]),
            "passed": _mean(costs[(dataset, "crr")]) <= 3.0,
        }
        for dataset in datasets
    ]
    criterion_4_pass = all(item["passed"] for item in criterion_4_details)

    benchmark_pass = criterion_1_pass and criterion_2_pass and criterion_4_pass
    return {
        "registered_datasets": datasets,
        "registered_seeds": list(REGISTERED_SEEDS),
        "registered_methods": list(REGISTERED_METHODS),
        "criterion_1_hijack": {"passed": criterion_1_pass, "details": criterion_1_details},
        "criterion_2_necessity": {"passed": criterion_2_pass, "details": criterion_2_details},
        "criterion_3_real_sequence": {
            "status": "pending_day_10",
            "passed": None,
            "reason": "Section 12.8 schedules real-sequence evidence for day 10.",
        },
        "criterion_4_cost": {"passed": criterion_4_pass, "details": criterion_4_details},
        "day_5_benchmark_verdict": "PASS" if benchmark_pass else "FAIL",
        "full_section_12_7_verdict": "PENDING_REAL_SEQUENCE" if benchmark_pass else "FAIL",
    }


def _write_summary_csv(path: Path, result: dict) -> None:
    rows = []
    for item in result["criterion_1_hijack"]["details"]:
        rows.append(
            {
                "criterion": "hijack",
                "dataset": item["dataset"],
                "strength": item["rho"],
                "passed": item["passed"],
                **item["means"],
            }
        )
    for item in result["criterion_2_necessity"]["details"]:
        rows.append(
            {
                "criterion": "necessity",
                "dataset": item["dataset"],
                "strength": "k>=2",
                "passed": item["passed"],
                "crr_minus_rcl": item["paired_tests"]["rcl_tab"]["mean_difference"],
                "p_rcl": item["paired_tests"]["rcl_tab"]["one_sided_exact_p"],
                "crr_minus_global": item["paired_tests"]["global_ir"]["mean_difference"],
                "p_global": item["paired_tests"]["global_ir"]["one_sided_exact_p"],
            }
        )
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = aggregate(_load(args.inputs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_summary_csv(args.output.with_suffix(".csv"), result)
    print(json.dumps({"day_5": result["day_5_benchmark_verdict"], "section_12_7": result["full_section_12_7_verdict"]}))


if __name__ == "__main__":
    main()
