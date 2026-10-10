"""Exp 3 analysis of real adjacent Class-IL checkpoints.

The analysis consumes completed ``ncd.explain`` runs.  It compares the legacy
fixed top-k rule, its repeat-mean B2 control, and the permutation/FDR NCD test
on real (non-intervened) class-incremental transitions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ncd.groundtruth import benjamini_hochberg, jaccard, ncd_permutation_test, normalize
from ncd.source_commit import source_commit
from ncd.train_cil import Tee


EXPLAINERS = ("expected_gradients", "feature_ablation", "grad_x_input")
TOP_K_VALUES = (5, 10, 15, 20)
JACCARD_THRESHOLDS = (0.5, 0.6, 0.7, 0.8)
SILENCE_TOLERANCES = (0.0, 0.02, 0.05, 0.10)


def git_commit() -> str:
    return source_commit(REPO_ROOT)


def resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


def resolve_training_path(value: str, explanation_path: Path) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return raw.resolve()
    repository_candidate = (REPO_ROOT / raw).resolve()
    if repository_candidate.is_file():
        return repository_candidate
    relocated_candidate = explanation_path.parent / raw.name
    if relocated_candidate.is_file():
        return relocated_candidate.resolve()
    return repository_candidate


def state_inputs(state_path: Path, run_repo: Path) -> list[str]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    completed = state.get("completed_runs", [])
    if not state.get("complete") or len(completed) != state.get("expected_runs"):
        raise RuntimeError("dataset state is incomplete")
    inputs = [(run_repo / item["explanation_result"]).resolve() for item in completed]
    if not all(path.is_file() for path in inputs):
        missing = [str(path) for path in inputs if not path.is_file()]
        raise FileNotFoundError(f"missing explanation results: {missing[:3]}")
    return [str(path) for path in inputs]


def cauchy_combination(p_values: list[float]) -> float:
    clipped = np.clip(np.asarray(p_values, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    statistic = float(np.mean(np.tan((0.5 - clipped) * math.pi)))
    combined = 0.5 - math.atan(statistic) / math.pi
    return float(np.clip(combined, 0.0, 1.0))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_run(explanation_path: Path, permutations: int) -> tuple[list[dict], dict]:
    explanation = json.loads(explanation_path.read_text(encoding="utf-8"))
    run_dir = explanation_path.parent
    training_path = resolve_training_path(explanation["training_result"], explanation_path)
    training = json.loads(training_path.read_text(encoding="utf-8"))
    recalls = {
        int(item["checkpoint"]): {
            int(class_id): float(value)
            for class_id, value in item["per_class_recall"].items()
        }
        for item in training["checkpoints"]
    }
    artifacts: dict[tuple[int, int, str], dict[str, np.ndarray]] = {}
    explanation_seconds = 0.0
    for record in explanation["records"]:
        explanation_seconds += float(record["elapsed_seconds"])
        with np.load(run_dir / record["path"], allow_pickle=False) as artifact:
            attributions = artifact["attributions"].astype(np.float64)
            estimates = artifact["estimates"].astype(np.float64)
        single = normalize(np.abs(attributions[0]).mean(axis=0))
        repeat_mean = normalize(np.abs(attributions).mean(axis=(0, 1)))
        artifacts[
            (int(record["checkpoint"]), int(record["class_id"]), str(record["explainer"]))
        ] = {
            "estimates": estimates,
            "single": single,
            "repeat_mean": repeat_mean,
        }

    checkpoints = sorted(recalls)
    rows: list[dict] = []
    fixed_rule_seconds = 0.0
    ncd_permutation_seconds = 0.0
    run_seed = int(explanation["seed"])
    for transition_index, (before, after) in enumerate(zip(checkpoints, checkpoints[1:])):
        shared = sorted(set(recalls[before]) & set(recalls[after]))
        for class_id in shared:
            recall_delta = recalls[after][class_id] - recalls[before][class_id]
            future_checkpoint = (
                checkpoints[transition_index + 2]
                if transition_index + 2 < len(checkpoints)
                else None
            )
            future_recall_delta = None
            if future_checkpoint is not None and class_id in recalls[future_checkpoint]:
                future_recall_delta = recalls[future_checkpoint][class_id] - recalls[after][class_id]
            for explainer_index, explainer in enumerate(EXPLAINERS):
                left = artifacts[(before, class_id, explainer)]
                right = artifacts[(after, class_id, explainer)]
                ncd_started = time.perf_counter()
                test = ncd_permutation_test(
                    left["estimates"],
                    right["estimates"],
                    permutations=permutations,
                    seed=run_seed * 1_000_000 + before * 10_000 + class_id * 100 + explainer_index,
                )
                ncd_permutation_seconds += time.perf_counter() - ncd_started
                row = {
                    "dataset": explanation["dataset"],
                    "method": explanation["method"],
                    "seed": run_seed,
                    "checkpoint_before": before,
                    "checkpoint_after": after,
                    "class_id": class_id,
                    "class_name": training["class_order"][class_id],
                    "recall_before": recalls[before][class_id],
                    "recall_after": recalls[after][class_id],
                    "recall_delta": recall_delta,
                    "future_recall_delta": future_recall_delta,
                    "silent_at_0pp": recall_delta > 0.0,
                    "silent_at_2pp": recall_delta > -0.02,
                    "silent_at_5pp": recall_delta > -0.05,
                    "silent_at_10pp": recall_delta > -0.10,
                    "explainer": explainer,
                    **test,
                }
                fixed_started = time.perf_counter()
                for top_k in TOP_K_VALUES:
                    row[f"b1_jaccard_k{top_k}"] = jaccard(
                        left["single"], right["single"], top_k
                    )
                    row[f"b2_jaccard_k{top_k}"] = jaccard(
                        left["repeat_mean"], right["repeat_mean"], top_k
                    )
                row["b1_fixed_drift"] = row["b1_jaccard_k15"] < 0.70
                row["b2_fixed_drift"] = row["b2_jaccard_k15"] < 0.70
                fixed_rule_seconds += time.perf_counter() - fixed_started
                rows.append(row)

    for explainer in EXPLAINERS:
        indices = [index for index, row in enumerate(rows) if row["explainer"] == explainer]
        adjusted = benjamini_hochberg([float(rows[index]["p_value"]) for index in indices])
        for index, q_value in zip(indices, adjusted):
            rows[index]["q_value"] = q_value
            rows[index]["ncd_drift"] = q_value < 0.05 and rows[index]["ncd_effect"] > 0.0

    run_info = {
        "dataset": explanation["dataset"],
        "method": explanation["method"],
        "seed": run_seed,
        "training_result": relative(training_path),
        "explanation_result": relative(explanation_path),
        "training_seconds": float(training["elapsed_seconds"]),
        "explanation_seconds": float(explanation["elapsed_seconds"]),
        "record_explanation_seconds": explanation_seconds,
        "fixed_rule_seconds": fixed_rule_seconds,
        "ncd_permutation_seconds": ncd_permutation_seconds,
        "decision_score": explanation["decision_score"],
    }
    return rows, run_info


def consensus_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"], row["method"], row["seed"], row["checkpoint_before"],
            row["checkpoint_after"], row["class_id"],
        )
        grouped[key].append(row)
    output: list[dict] = []
    for key, items in sorted(grouped.items()):
        if {item["explainer"] for item in items} != set(EXPLAINERS):
            raise RuntimeError(f"incomplete explainer set for {key}")
        ncd_votes = int(sum(bool(item["ncd_drift"]) for item in items))
        b1_votes = int(sum(bool(item["b1_fixed_drift"]) for item in items))
        b2_votes = int(sum(bool(item["b2_fixed_drift"]) for item in items))
        p_values = [float(item["p_value"]) for item in items]
        effects = [float(item["ncd_effect"]) for item in items]
        cauchy_p = cauchy_combination(p_values)
        output.append(
            {
                "dataset": key[0],
                "method": key[1],
                "seed": key[2],
                "checkpoint_before": key[3],
                "checkpoint_after": key[4],
                "class_id": key[5],
                "class_name": items[0]["class_name"],
                "recall_delta": items[0]["recall_delta"],
                "future_recall_delta": items[0]["future_recall_delta"],
                "silent_at_0pp": items[0]["silent_at_0pp"],
                "silent_at_2pp": items[0]["silent_at_2pp"],
                "silent_at_5pp": items[0]["silent_at_5pp"],
                "silent_at_10pp": items[0]["silent_at_10pp"],
                "b1_drift_votes": b1_votes,
                "b1_consensus_drift": b1_votes >= 2,
                "b2_drift_votes": b2_votes,
                "b2_consensus_drift": b2_votes >= 2,
                "ncd_drift_votes": ncd_votes,
                "ncd_consensus_drift": ncd_votes >= 2,
                "ncd_cauchy_p": cauchy_p,
                "ncd_median_effect": float(np.median(effects)),
                "b1_all_explainers_agree": b1_votes in (0, 3),
                "b2_all_explainers_agree": b2_votes in (0, 3),
                "ncd_all_explainers_agree": ncd_votes in (0, 3),
            }
        )
    run_keys = sorted({(row["dataset"], row["method"], row["seed"]) for row in output})
    for run_key in run_keys:
        indices = [
            index for index, row in enumerate(output)
            if (row["dataset"], row["method"], row["seed"]) == run_key
        ]
        adjusted = benjamini_hochberg(
            [float(output[index]["ncd_cauchy_p"]) for index in indices]
        )
        for index, q_value in zip(indices, adjusted):
            output[index]["ncd_cauchy_q"] = q_value
            output[index]["ncd_cauchy_drift"] = (
                q_value < 0.05 and output[index]["ncd_median_effect"] > 0.0
            )
    return output


def sensitivity_rows(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    run_keys = sorted({(row["dataset"], row["method"], row["seed"]) for row in rows})
    for dataset, method, seed in run_keys:
        run_rows = [
            row for row in rows
            if (row["dataset"], row["method"], row["seed"]) == (dataset, method, seed)
        ]
        for explainer in EXPLAINERS:
            exp_rows = [row for row in run_rows if row["explainer"] == explainer]
            for tolerance in SILENCE_TOLERANCES:
                silent = [row for row in exp_rows if row["recall_delta"] > -tolerance]
                for top_k in TOP_K_VALUES:
                    for threshold in JACCARD_THRESHOLDS:
                        output.append(
                            {
                                "dataset": dataset,
                                "method": method,
                                "seed": seed,
                                "explainer": explainer,
                                "silence_tolerance": tolerance,
                                "top_k": top_k,
                                "jaccard_threshold": threshold,
                                "silent_units": len(silent),
                                "b1_drift_units": sum(
                                    row[f"b1_jaccard_k{top_k}"] < threshold for row in silent
                                ),
                                "b1_drift_rate": (
                                    float(np.mean([
                                        row[f"b1_jaccard_k{top_k}"] < threshold for row in silent
                                    ])) if silent else None
                                ),
                                "b2_drift_units": sum(
                                    row[f"b2_jaccard_k{top_k}"] < threshold for row in silent
                                ),
                                "b2_drift_rate": (
                                    float(np.mean([
                                        row[f"b2_jaccard_k{top_k}"] < threshold for row in silent
                                    ])) if silent else None
                                ),
                            }
                        )
    return output


def run_summaries(consensus: list[dict], run_info: list[dict]) -> list[dict]:
    output: list[dict] = []
    for info in run_info:
        subset = [
            row for row in consensus
            if (row["dataset"], row["method"], row["seed"])
            == (info["dataset"], info["method"], info["seed"])
            and row["silent_at_5pp"]
        ]
        future = [row for row in subset if row["future_recall_delta"] is not None]
        ncd_future = [float(row["future_recall_delta"]) for row in future if row["ncd_consensus_drift"]]
        no_ncd_future = [float(row["future_recall_delta"]) for row in future if not row["ncd_consensus_drift"]]
        output.append(
            {
                "dataset": info["dataset"],
                "method": info["method"],
                "seed": info["seed"],
                "silent_units_5pp": len(subset),
                "b1_consensus_drift_units": sum(row["b1_consensus_drift"] for row in subset),
                "b1_consensus_drift_rate": float(np.mean([row["b1_consensus_drift"] for row in subset])) if subset else None,
                "b2_consensus_drift_units": sum(row["b2_consensus_drift"] for row in subset),
                "b2_consensus_drift_rate": float(np.mean([row["b2_consensus_drift"] for row in subset])) if subset else None,
                "ncd_consensus_drift_units": sum(row["ncd_consensus_drift"] for row in subset),
                "ncd_consensus_drift_rate": float(np.mean([row["ncd_consensus_drift"] for row in subset])) if subset else None,
                "b1_all_explainer_agreement_rate": float(np.mean([row["b1_all_explainers_agree"] for row in subset])) if subset else None,
                "b2_all_explainer_agreement_rate": float(np.mean([row["b2_all_explainers_agree"] for row in subset])) if subset else None,
                "ncd_all_explainer_agreement_rate": float(np.mean([row["ncd_all_explainers_agree"] for row in subset])) if subset else None,
                "future_units": len(future),
                "median_future_recall_delta_after_ncd_drift": float(np.median(ncd_future)) if ncd_future else None,
                "median_future_recall_delta_without_ncd_drift": float(np.median(no_ncd_future)) if no_ncd_future else None,
                "training_seconds": info["training_seconds"],
                "explanation_seconds": info["explanation_seconds"],
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--explanation-result", action="append")
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--run-repo", type=Path)
    parser.add_argument("--output-dir", default="ncd/results/exp3/real_seed1_first_tranche")
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--log-path", required=True)
    args = parser.parse_args()
    input_values = list(args.explanation_result or [])
    if args.state_path is not None:
        if args.run_repo is None:
            parser.error("--run-repo is required with --state-path")
        state_path = args.state_path if args.state_path.is_absolute() else args.run_repo / args.state_path
        input_values.extend(state_inputs(state_path.resolve(), args.run_repo.resolve()))
    if not input_values:
        parser.error("provide --explanation-result or --state-path with --run-repo")
    log_path = resolve(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        with redirect_stdout(Tee(sys.__stdout__, log_file)), redirect_stderr(
            Tee(sys.__stderr__, log_file)
        ):
            analysis_started = time.perf_counter()
            output_dir = resolve(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            rows: list[dict] = []
            run_info: list[dict] = []
            for value in input_values:
                run_rows, info = load_run(resolve(value), args.permutations)
                rows.extend(run_rows)
                run_info.append(info)
            consensus = consensus_rows(rows)
            sensitivity = sensitivity_rows(rows)
            summaries = run_summaries(consensus, run_info)
            unit_path = output_dir / "exp3_explainer_units.csv"
            consensus_path = output_dir / "exp3_consensus_units.csv"
            sensitivity_path = output_dir / "exp3_threshold_sensitivity.csv"
            summary_path = output_dir / "exp3_run_summary.csv"
            write_csv(unit_path, rows)
            write_csv(consensus_path, consensus)
            write_csv(sensitivity_path, sensitivity)
            write_csv(summary_path, summaries)
            scope_rows = [
                {
                    "dataset": dataset,
                    "method": method,
                    "seeds": sorted(
                        int(info["seed"])
                        for info in run_info
                        if info["dataset"] == dataset and info["method"] == method
                    ),
                }
                for dataset, method in sorted(
                    {(info["dataset"], info["method"]) for info in run_info}
                )
            ]
            scope = {
                "run_count": len(run_info),
                "datasets": sorted({info["dataset"] for info in run_info}),
                "methods": sorted({info["method"] for info in run_info}),
                "seeds": sorted({int(info["seed"]) for info in run_info}),
                "run_matrix": scope_rows,
            }
            result = {
                "analysis": "Exp 3 real adjacent-checkpoint analysis",
                "git_commit": git_commit(),
                "command": " ".join(sys.argv),
                "log_path": relative(log_path),
                "inputs": [relative(resolve(value)) for value in input_values],
                "scope": scope,
                "permutations": args.permutations,
                "fdr_q": 0.05,
                "fixed_rule": "Jaccard@15 < 0.70",
                "silence_primary": "recall_after - recall_before > -0.05",
                "analysis_seconds": time.perf_counter() - analysis_started,
                "runs": run_info,
                "run_summaries": summaries,
                "artifacts": {
                    "explainer_units": relative(unit_path),
                    "consensus_units": relative(consensus_path),
                    "threshold_sensitivity": relative(sensitivity_path),
                    "run_summary": relative(summary_path),
                },
                "limitations": [
                    (
                        f"This result contains {scope['run_count']} runs across "
                        f"{len(scope['datasets'])} dataset(s), {len(scope['methods'])} method(s), "
                        f"and seeds {scope['seeds']}; it does not cover unlisted datasets or methods."
                    ),
                    "NCD thresholds are estimated within each run; no external calibration split is used.",
                    "Drift-to-future-recall comparisons are descriptive and do not establish prediction or causality.",
                    "Exp 2 showed no NCD superiority over B2, so NCD is not treated as an established innovation.",
                ],
            }
            result_path = output_dir / "exp3_result.json"
            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps({"result": relative(result_path), "runs": len(run_info)}, indent=2))


if __name__ == "__main__":
    main()
