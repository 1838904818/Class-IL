"""Section21 G1-G7 descriptive summaries; incomplete matrices never rank problems.

The best CIL method is selected once per dataset/metric by its five-seed mean,
not separately for each seed. Per-class winners are explicitly a classwise
envelope, not one deployable classifier. No mechanism is proposed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

SEEDS = (1, 2, 3, 4, 42)
DATASETS = ("NSL-KDD", "UNSW-NB15", "CIC-IDS-2017", "CSE-CIC-IDS2018")
CORE_METHODS = ("joint", "replay", "icarl", "replay_lwf", "rcl_tab", "derpp")
G7_MEASUREMENT_LIMITS = {
    "training_seconds_total": "Recorded cumulative successful checkpoint training-loop intervals; includes in-loop DER++ memory updates, not total cell walltime.",
    "evaluation_seconds_total": "Sum of saved checkpoint evaluation intervals, including scorer construction and ID/available shift evaluation; not repeated or failed-attempt evaluation cost.",
    "buffer_selection_seconds_total": "Sum of saved post-training selection intervals, including iCaRL herding; DER++ updates memory inside training, so its near-zero selection timer is not zero memory-management cost.",
    "walltime": "Training, evaluation and selection do not exhaust walltime. Data loading, construction, serialization and other overhead, plus failed/cancelled attempts, require separate logs; never add overlapping runner and Slurm durations.",
    "cuda_memory": "Maximum observed PyTorch training-phase allocated/reserved bytes, captured before selection/evaluation; not device-wide usage or a whole-cell peak. Null means unmeasured; missing checkpoint observations can underestimate the peak.",
    "replay_memory": "Maximum saved post-checkpoint payload and allocated replay-storage bytes are distinct; neither includes model/optimizer memory, temporary herding arrays, host RSS or whole-process memory.",
    "comparability": "Stage costs are descriptive for each recorded execution environment; local and HPC timings are not hardware-matched speed comparisons.",
}


def finite(value):
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def summary(values):
    values = list(values)
    if len(values) != len(SEEDS) or not all(finite(v) for v in values):
        return None
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values),
            "by_seed": dict(zip(map(str, SEEDS), values))}


def best_comparison(method_values, higher_is_better=True):
    """Return a fixed mean-best CIL comparison, refusing incomplete seed vectors."""
    joint = method_values.get("joint", {})
    if set(joint) != set(SEEDS) or not all(finite(v) for v in joint.values()):
        return None
    complete = {m: v for m, v in method_values.items() if m != "joint"
                and set(v) == set(SEEDS) and all(finite(x) for x in v.values())}
    if not complete:
        return None
    sign = 1 if higher_is_better else -1
    winner = min(complete, key=lambda m: (-sign * statistics.mean(complete[m].values()), m))
    j = [joint[s] for s in SEEDS]
    c = [complete[winner][s] for s in SEEDS]
    gaps = [sign * (a - b) for a, b in zip(j, c)]
    return {"best_cil_method": winner, "joint": summary(j), "best_cil": summary(c),
            "gap_pp": summary(100 * x for x in gaps),
            "gap_positive_all_five_seeds": all(x > 0 for x in gaps),
            "mean_gap_at_least_5pp": statistics.mean(gaps) >= .05,
            "direction": "Joint minus CIL" if higher_is_better else "CIL minus Joint"}


def count_bucket(count):
    # The two stated intervals overlap at 1k/10k; explicit inclusive middle.
    return "<1k" if count < 1000 else "1k-10k" if count <= 10000 else ">10k"


def error_flow(matrix, class_id, tasks, normal_id=0):
    """Mutually exclusive bins, using learning time relative to the true class."""
    own = next(t for t, group in enumerate(tasks) if class_id in group)
    born = {c: t for t, group in enumerate(tasks) for c in group}
    row = matrix[class_id]
    bins = {"normal": 0, "newer_classes": 0, "other_old_or_same_task": 0}
    for predicted, n in enumerate(row):
        if predicted == class_id:
            continue
        key = "normal" if predicted == normal_id else (
            "newer_classes" if born[predicted] > own else "other_old_or_same_task")
        bins[key] += int(n)
    errors = sum(bins.values())
    support = sum(row)
    return {"support": support, "errors": errors, "counts": bins,
            "fraction_of_true_class": {k: v / support if support else None for k, v in bins.items()},
            "fraction_of_errors": {k: v / errors if errors else None for k, v in bins.items()}}


def class_recall(metrics, class_id):
    matrix = metrics["confusion_matrix"]
    total = sum(matrix[class_id])
    return matrix[class_id][class_id] / total if total else None


def validate_confusion(metrics):
    matrix = metrics["confusion_matrix"]
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("Confusion matrix must use the complete class_order axis")
    if any(not isinstance(x, int) or x < 0 for row in matrix for x in row):
        raise ValueError("Confusion counts must be nonnegative integers")
    rows = metrics["per_class"]
    labels = [int(r["class_id"]) for r in rows]
    if len(labels) != len(set(labels)) or any(c < 0 or c >= len(matrix) for c in labels):
        raise ValueError("Per-class axes are invalid")
    if sum(map(sum, matrix)) != metrics["evaluated_rows"]:
        raise ValueError("Confusion support differs from evaluated_rows")

    def close(actual, expected):
        return actual is None and expected is None or (finite(actual) and finite(expected) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12))

    f1s, recalls = [], []
    for row in rows:
        c = int(row["class_id"])
        support, predicted, tp = sum(matrix[c]), sum(r[c] for r in matrix), matrix[c][c]
        recall = tp / support if support else None
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        if row["support"] != support or not close(row["recall"], recall) or not close(row["f1"], f1):
            raise ValueError("Per-class metrics disagree with confusion counts")
        f1s.append(f1)
        if recall is not None:
            recalls.append(recall)
    attack = [c for c in labels if c != 0]
    attack_n = sum(sum(matrix[c]) for c in attack)
    attack_detected = sum(matrix[c][p] for c in attack for p in attack)
    normal_n = sum(matrix[0])
    expected = {"macro_f1": statistics.mean(f1s) if f1s else None,
                "balanced_accuracy": statistics.mean(recalls) if recalls else None,
                "attack_recall": attack_detected / attack_n if attack_n else None,
                "normal_fpr": (normal_n - matrix[0][0]) / normal_n if normal_n else None}
    for key, value in expected.items():
        if not close(metrics[key], value):
            raise ValueError(f"{key} disagrees with independent confusion recomputation")


def checkpoint_costs(cell):
    """Keep absent/invalid costs distinct from measured zero and unmeasured CUDA."""
    identity = cell.get("identity", {})
    context = "/".join(str(identity.get(k, "unknown")) for k in ("dataset", "method", "seed"))

    def required(record, field, scope, nullable=False):
        if field not in record:
            raise ValueError(f"Missing G7 cost field {field} in {context} {scope}")
        value = record[field]
        if nullable and value is None:
            return None
        if isinstance(value, bool) or not finite(value) or value < 0:
            raise ValueError(f"Invalid G7 cost field {field} in {context} {scope}: expected a finite nonnegative number")
        return value

    checkpoints = cell["checkpoints"]
    if not checkpoints:
        raise ValueError(f"No successful checkpoints for G7 costs in {context}")
    training = required(cell, "training_seconds_total", "cell")
    fields = ("training_seconds", "evaluation_seconds", "buffer_selection_seconds",
              "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes",
              "replay_bytes_after", "replay_allocated_bytes_after")
    recorded = {field: [] for field in fields}
    for index, checkpoint in enumerate(checkpoints):
        scope = f"checkpoint {index}"
        for field in fields:
            recorded[field].append(required(checkpoint, field, scope, nullable=field.startswith("peak_cuda_")))
    totals = {}
    for field in ("training_seconds", "evaluation_seconds", "buffer_selection_seconds"):
        total = sum(recorded[field])
        totals[field] = required({field: total}, field, "checkpoint sum")
    if not math.isclose(training, totals["training_seconds"], rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(f"G7 training_seconds_total differs from successful checkpoint training sum in {context}")
    return {
        "training_seconds_total": training,
        "evaluation_seconds_total": totals["evaluation_seconds"],
        "buffer_selection_seconds_total": totals["buffer_selection_seconds"],
        "peak_cuda_allocated_bytes": max((v for v in recorded["peak_cuda_allocated_bytes"] if v is not None), default=None),
        "peak_cuda_reserved_bytes": max((v for v in recorded["peak_cuda_reserved_bytes"] if v is not None), default=None),
        "maximum_replay_bytes": max(recorded["replay_bytes_after"]),
        "maximum_replay_allocated_bytes": max(recorded["replay_allocated_bytes_after"]),
        "cost_scope": "Cumulative successful checkpoint training with separate evaluation/selection totals; failed-attempt walltime requires separate logs",
    }


def analyze(cells, methods=CORE_METHODS):
    keyed = {}
    for cell in cells:
        identity = cell["identity"]
        key = (identity["dataset"], identity["method"], int(identity["seed"]))
        if key in keyed:
            raise ValueError(f"Duplicate cell {key}")
        if not cell.get("completed", False):
            continue
        for checkpoint in cell["checkpoints"]:
            validate_confusion(checkpoint["id"])
            if checkpoint.get("shift") is not None:
                validate_confusion(checkpoint["shift"])
        keyed[key] = cell
    expected = {(d, m, s) for d in DATASETS for m in methods for s in SEEDS}
    missing = sorted(expected - set(keyed))
    smoke_flags = {bool(c["identity"].get("effective_config", {}).get("smoke", False)) for c in keyed.values()}
    if len(smoke_flags) > 1:
        raise ValueError("Never mix smoke and formal cells")
    smoke = True in smoke_flags
    out = {"schema_version": 1, "execution_status": "smoke_only" if smoke else ("partial" if missing else "core_matrix_complete"),
           "complete_cells": len(keyed), "expected_cells": len(expected), "missing_cells": missing,
           "smoke": smoke, "scientific_comparisons_final": not missing and not smoke,
           "coverage_caveat": "Partial summaries are provisional; missing methods may change the best CIL method. Additional NIDS baseline eligibility is recorded separately.",
           "selection_policy": "Fixed method per dataset/metric using five-seed mean; classwise envelope for G2 only",
           "G7_measurement_limits": dict(G7_MEASUREMENT_LIMITS),
           "candidate_ranking_status": "pending_quantitative_matrix_and_targeted_literature_evidence",
           "candidates": [], "G1": [], "G2": [], "G3": [], "G4": [], "G5": [], "G6": [], "G7": []}
    for dataset in DATASETS:
        subset = {(m, s): c for (d, m, s), c in keyed.items() if d == dataset}
        if not subset:
            continue
        exemplar = next(iter(subset.values()))
        classes, tasks = exemplar["class_order"], exemplar["tasks"]
        for cell in subset.values():
            if cell["class_order"] != classes or cell["tasks"] != tasks:
                raise ValueError(f"Class/task mismatch in {dataset}")
        def values(metric, mode="id", class_id=None):
            result = defaultdict(dict)
            for (m, s), cell in subset.items():
                metrics = cell["checkpoints"][-1].get(mode)
                if metrics is not None:
                    result[m][s] = class_recall(metrics, class_id) if class_id is not None else metrics.get(metric)
            return result
        for metric in ("macro_f1", "balanced_accuracy"):
            out["G1"].append({"dataset": dataset, "metric": metric,
                              "comparison": best_comparison(values(metric))})
        for metric, higher in (("attack_recall", True), ("normal_fpr", False)):
            out["G5"].append({"dataset": dataset, "metric": metric,
                              "comparison": best_comparison(values(metric), higher)})
        for cid, name in enumerate(classes):
            train_counts = exemplar["train_counts"]
            n = int(train_counts.get(name, train_counts.get(str(cid))) if isinstance(train_counts, dict) else train_counts[cid])
            out["G2"].append({"dataset": dataset, "class_id": cid, "class_name": name,
                              "train_count": n, "train_count_bucket": count_bucket(n),
                              "comparison": best_comparison(values("recall", class_id=cid))})
            if dataset == "CIC-IDS-2017":
                out["G6"].append({"dataset": dataset, "class_id": cid, "class_name": name,
                                  "comparison": best_comparison(values("recall", mode="shift", class_id=cid)),
                                  "qualification": "One source dataset only; cannot alone satisfy the two-dataset candidate gate"})
        for (method, seed), cell in subset.items():
            final = cell["checkpoints"][-1]
            for cid, name in enumerate(classes):
                own_task = next(i for i, group in enumerate(tasks) if cid in group)
                own = next(cp for cp in cell["checkpoints"] if cp["task_index"] == own_task)
                learned = class_recall(own["id"], cid)
                retained = class_recall(final["id"], cid)
                base = {"dataset": dataset, "method": method, "seed": seed, "class_id": cid, "class_name": name}
                out["G3"].append({**base, "own_task_index": own_task, "own_task_recall": learned,
                                  "final_recall": retained,
                                  "own_minus_final_recall_pp": 100 * (learned - retained) if finite(learned) and finite(retained) else None})
                out["G4"].append({**base, "is_old_at_final": own_task < len(tasks) - 1,
                                  **error_flow(final["id"]["confusion_matrix"], cid, tasks)})
            out["G7"].append({"dataset": dataset, "method": method, "seed": seed,
                              **checkpoint_costs(cell)})
    return out


def load_cells(root, filename="result.json"):
    """Allow multiple matrix roots while never treating their indexes as cells."""
    paths, cells = [], []
    for path in sorted(root.rglob(filename)):
        value = json.loads(path.read_text(encoding="utf-8"))
        if "identity" not in value:
            if "cells" in value and "expected" in value:
                continue
            raise ValueError(f"Unrecognized result schema: {path}")
        paths.append(path)
        cells.append(value)
    return paths, cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cell-name", default="result.json")
    parser.add_argument("--methods", nargs="+", default=list(CORE_METHODS))
    args = parser.parse_args()
    paths, cells = load_cells(args.input_root, args.cell_name)
    result = analyze(cells, args.methods)
    result["inputs"] = [{"file": p.relative_to(args.input_root).as_posix(),
                         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
    log_path = args.output.with_suffix(".log")
    result["analysis_provenance"] = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": {"methods": args.methods, "seeds": list(SEEDS), "datasets": list(DATASETS)},
        "seed": None, "seed_note": "Deterministic descriptive aggregation, no bootstrap or random sampling",
        "command": subprocess.list2cmdline(["python", "-I", "ncd/analyze_problem_scout.py", *sys.argv[1:]]),
        "log_path": log_path.as_posix(),
        "experiment_source_commits": sorted({c["identity"]["git_commit"] for c in cells})}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    message = json.dumps({k: result[k] for k in ("execution_status", "complete_cells", "expected_cells", "candidate_ranking_status")})
    log_path.write_text(message + "\n", encoding="utf-8")
    print(message)


if __name__ == "__main__":
    main()
