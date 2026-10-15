"""Section22 step-floor comparison, using final ID confusion counts as evidence.

Descriptive, paired five-seed comparisons only. Incomplete coverage cannot issue
a Section22.3 decision. Baseline Joint is reused, never retrained or relabelled.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

DATASETS = ("NSL-KDD", "UNSW-NB15", "CIC-IDS-2017")
METHODS = ("replay", "replay_lwf", "rcl_tab", "icarl", "derpp")
SEEDS = (1, 2, 3, 4, 42)
BUDGETS = ("original", "500", "2000")
NAMED_CLASSES = (("NSL-KDD", "U2R"), ("UNSW-NB15", "Shellcode"),
                 ("UNSW-NB15", "Worms"), ("CIC-IDS-2017", "Botnet"),
                 ("CIC-IDS-2017", "Infiltration"))
OPERATIONAL_CONFIG = {"min_steps_per_task", "cpu_threads", "device", "datasets",
                      "methods", "seeds", "problem_scout_data_root", "eval_batch_size",
                      "herding_mmap_threshold_bytes"}


def close(actual, expected):
    if actual is None or expected is None:
        return actual is expected
    return (not isinstance(actual, bool) and isinstance(actual, (int, float))
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12))


def summarize(values):
    """Keep missing seeds/null support visible; no imputation or pooled counts."""
    by_seed = {str(s): values.get(s) for s in SEEDS}
    observed = [v for v in by_seed.values() if v is not None]
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
           for v in observed):
        raise ValueError("Summary values must be finite numbers or null")
    return {"n": len(observed), "complete": len(observed) == len(SEEDS),
            "mean": statistics.mean(observed) if observed else None,
            "sample_sd": statistics.stdev(observed) if len(observed) > 1 else None,
            "by_seed": by_seed}


def paired_delta(left, right, scale=100):
    return summarize({s: scale * (left[s] - right[s]) for s in SEEDS
                      if left.get(s) is not None and right.get(s) is not None})


def at_least(value, threshold):
    # Ratio division can represent an exact .2 boundary as .19999999999999998.
    return value > threshold or close(value, threshold)


def strictly_below(value, threshold):
    return value < threshold and not close(value, threshold)


def recompute(metrics, classes, seen):
    """Validate saved summaries, then return independently recomputed metrics."""
    n = len(classes)
    cm = metrics["confusion_matrix"]
    if (len(cm) != n or any(len(row) != n for row in cm)
            or any(type(x) is not int or x < 0 for row in cm for x in row)):
        raise ValueError("Confusion matrix must have full class axes and integer counts")
    if metrics.get("confusion_labels") != list(range(n)):
        raise ValueError("Confusion label axis mismatch")
    if metrics["evaluated_rows"] != sum(map(sum, cm)):
        raise ValueError("Confusion total differs from evaluated_rows")
    unseen = set(range(n)) - set(seen)
    if any(sum(cm[c]) or sum(row[c] for row in cm) for c in unseen):
        raise ValueError("Unseen class has confusion counts")
    saved = {r["class_id"]: r for r in metrics["per_class"]}
    if len(saved) != len(metrics["per_class"]) or set(saved) != set(seen):
        raise ValueError("Per-class summary has missing, extra or duplicate labels")
    recalls, f1s, supports = {}, [], {}
    for c in seen:
        support, predicted, tp = sum(cm[c]), sum(r[c] for r in cm), cm[c][c]
        recall = tp / support if support else None
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        row = saved[c]
        if (row["class_name"] != classes[c] or row["support"] != support
                or not close(row["recall"], recall) or not close(row["f1"], f1)):
            raise ValueError("Per-class metrics disagree with confusion counts/axis")
        recalls[c], supports[c] = recall, support
        f1s.append(f1)
    supported_recalls = [v for v in recalls.values() if v is not None]
    result = {"recall": recalls, "support": supports,
              "macro_f1": statistics.mean(f1s) if f1s else None,
              "balanced_accuracy": statistics.mean(supported_recalls) if supported_recalls else None}
    for name in ("macro_f1", "balanced_accuracy"):
        if not close(metrics[name], result[name]):
            raise ValueError(f"{name} disagrees with confusion recomputation")
    return result


def validate_schedule(cell, budget):
    config = cell["identity"]["effective_config"]
    if config.get("min_steps_per_task", 0) != int(budget):
        raise ValueError(f"Wrong min_steps_per_task for {budget} arm")
    if config["epochs_per_task"] != 8:
        raise ValueError("Section22 requires eight baseline epochs")
    if budget == "0":
        return
    for cp in cell["checkpoints"]:
        schedule = cp["training_schedule"]
        rows, batch = cp["train_rows"], config["batch_size"]
        batches = math.ceil(rows / batch)
        target = max(8 * batches, int(budget))
        expected = {"rows_per_epoch": rows, "batches_per_epoch": batches,
                    "baseline_steps": 8 * batches, "target_steps": target,
                    "optimizer_steps": target, "min_steps_per_task": int(budget),
                    "requested_epochs": 8,
                    "completed_epochs": target // batches,
                    "partial_epoch_batches": target % batches,
                    "partial_epoch_examples": min(target % batches * batch, rows)}
        expected["examples_presented"] = (expected["completed_epochs"] * rows
                                           + expected["partial_epoch_examples"])
        for name, value in expected.items():
            if type(schedule.get(name)) is not int or schedule[name] != value:
                raise ValueError(f"Training schedule {name} does not match step-floor rule")
        if not close(schedule.get("effective_epochs"), expected["examples_presented"] / rows):
            raise ValueError("Training schedule effective_epochs does not match exposure")


def validate_cell(cell, budget):
    classes, tasks = cell["class_order"], cell["tasks"]
    if len(set(classes)) != len(classes) or not tasks or any(not task for task in tasks):
        raise ValueError("Invalid class/task metadata")
    if sorted(c for task in tasks for c in task) != list(range(len(classes))):
        raise ValueError("Each class must be introduced exactly once")
    identity = cell["identity"]
    if identity.get("class_order") != classes or identity.get("tasks") != tasks:
        raise ValueError("Identity and cell class/task axes differ")
    checkpoints = cell["checkpoints"]
    if [cp["task_index"] for cp in checkpoints] != list(range(len(tasks))):
        raise ValueError("Completed cell must contain every checkpoint in task order")
    for t, cp in enumerate(checkpoints):
        if (cp["new_classes"] != tasks[t]
                or cp["seen_classes"] != sorted(c for task in tasks[:t + 1] for c in task)):
            raise ValueError("Checkpoint class/task axes differ")
    validate_schedule(cell, "0" if budget == "original" else budget)
    final = recompute(checkpoints[-1]["id"], classes, list(range(len(classes))))
    previous = recompute(checkpoints[-2]["id"], classes, checkpoints[-2]["seen_classes"]) if len(tasks) > 1 else None
    return {"cell": cell, "final": final, "previous": previous}


def index_cells(cells, budget):
    indexed, ignored, unfinished = {}, [], []
    for cell in cells:
        identity = cell["identity"]
        key = (identity["dataset"], identity["method"], int(identity["seed"]))
        if budget == "original" and key[0] == "CSE-CIC-IDS2018":
            ignored.append(key)
            continue
        allowed_methods = ("joint", *METHODS) if budget == "original" else METHODS
        if key[0] not in DATASETS or key[1] not in allowed_methods or key[2] not in SEEDS:
            raise ValueError(f"Unexpected cell in {budget}: {key}")
        if key in indexed or key in unfinished:
            raise ValueError(f"Duplicate cell in {budget}: {key}")
        if cell.get("completed") is not True:
            unfinished.append(key)
            continue
        indexed[key] = validate_cell(cell, budget)
    expected = {(d, m, s) for d in DATASETS for m in (("joint", *METHODS) if budget == "original" else METHODS) for s in SEEDS}
    return indexed, {"expected_cells": len(expected), "complete_cells": len(indexed),
                     "missing_cells": sorted(expected - set(indexed)),
                     "unfinished_cells": sorted(unfinished), "ignored_cells": sorted(ignored)}


def validate_matching(indexes):
    references = {}
    for budget, indexed in indexes.items():
        for (dataset, method, seed), record in indexed.items():
            cell, metrics = record["cell"], record["final"]
            identity = cell["identity"]
            common = {"classes": cell["class_order"], "tasks": cell["tasks"],
                      "train_counts": cell["train_counts"], "support": metrics["support"],
                      "data_binding": identity["data_binding"]}
            if dataset in references and references[dataset] != common:
                raise ValueError(f"Data/class/task/support mismatch for {dataset}")
            references[dataset] = common
            baseline = indexes["original"].get((dataset, method, seed))
            if budget != "original" and baseline:
                old_identity = baseline["cell"]["identity"]
                config = {k: v for k, v in identity["effective_config"].items() if k not in OPERATIONAL_CONFIG}
                old_config = {k: v for k, v in old_identity["effective_config"].items() if k not in OPERATIONAL_CONFIG}
                if config != old_config or identity["regularization"] != old_identity["regularization"]:
                    raise ValueError(f"Scientific config changed beyond step floor: {dataset}/{method}/{seed}")
    return references


def fixed_best(values):
    """Do not let an incomplete method arm masquerade as the best of all five."""
    required = ("joint", *METHODS)
    if any(set(values.get(m, {})) != set(SEEDS) or any(v is None for v in values[m].values()) for m in required):
        return {"complete": False, "best_cil_method": None, "gap_pp": None}
    winner = min(METHODS, key=lambda m: (-statistics.mean(values[m].values()), m))
    delta = paired_delta(values["joint"], values[winner])
    return {"complete": True, "best_cil_method": winner,
            "joint": summarize(values["joint"]), "best_cil": summarize(values[winner]),
            "gap_pp": delta, "all_five_positive": all(v > 0 for v in delta["by_seed"].values()),
            "mean_gap_at_least_5pp": delta["mean"] >= 5}


def decide(named_rows, complete, smoke=False):
    """Thresholds apply separately to all five classes, not pooled datasets."""
    rows = {(r["dataset"], r["class_name"]): r for r in named_rows}
    if smoke or not complete or len(named_rows) != len(NAMED_CLASSES) or set(rows) != set(NAMED_CLASSES):
        return {"decision": "pending", "reason": "Full formal comparison coverage is required"}
    ratios = [rows[k]["ratio_of_five_seed_means"] for k in NAMED_CLASSES]
    if any(v is None for v in ratios):
        return {"decision": "pending", "reason": "At least one class lacks a defined five-seed Replay/Joint ratio"}
    if all(at_least(v, .5) for v in ratios):
        return {"decision": "A", "reason": "Replay reaches at least 50% of Joint mean recall for every named class"}
    if all(strictly_below(v, .2) for v in ratios):
        return {"decision": "B", "reason": "Replay remains below 20% of Joint mean recall for every named class"}
    return {"decision": "intermediate", "reason": "The five separate class ratios do not jointly satisfy A or B; researcher decision required"}


def analyze(baseline, steps_500, steps_2000):
    indexes, coverage = {}, {}
    for budget, cells in zip(BUDGETS, (baseline, steps_500, steps_2000)):
        indexes[budget], coverage[budget] = index_cells(cells, budget)
    references = validate_matching(indexes)
    all_records = [r for arm in indexes.values() for r in arm.values()]
    smoke_flags = {r["cell"]["identity"]["effective_config"].get("smoke", False) for r in all_records}
    if len(smoke_flags) > 1:
        raise ValueError("Never mix smoke and formal cells")
    smoke = True in smoke_flags
    complete = all(not c["missing_cells"] for c in coverage.values())

    def vector(budget, dataset, method, metric, cid=None):
        arm = indexes["original" if method == "joint" else budget]
        result = {}
        for seed in SEEDS:
            record = arm.get((dataset, method, seed))
            if record:
                result[seed] = record["final"][metric][cid] if cid is not None else record["final"][metric]
        return result

    out = {"schema_version": 1, "execution_status": "smoke_only" if smoke else "complete" if complete else "partial",
           "scientific_comparisons_final": complete and not smoke, "coverage": coverage,
           "interpretation": {
               "units": "Recall, Macro-F1 and balanced accuracy are fractions; differences marked pp are percentage points",
               "aggregation": "Arithmetic mean and sample standard deviation across seeds; exact seed-paired differences; no pooled confusion matrices",
               "partial": "Partial summaries explicitly retain n, complete and missing seed values; no final verdict before complete coverage",
               "selection": "G1 fixed CIL method chosen by five-seed mean per dataset/metric/budget; G2 is a per-class envelope, not one classifier",
               "joint": "Original Section21 eight-epoch Joint is reused unchanged, not a universal empirical upper bound",
               "decision": "Five separate ratios of five-seed Replay mean recall to original Joint mean recall at S=2000; A all >=0.5, B all <0.2; mixed is intermediate",
               "threshold_precision": "Numerical equality at a ratio threshold uses 1e-12 absolute/relative tolerance; this preserves the inclusive 0.5 and exclusive 0.2 boundaries",
               "zero_joint": "A zero Joint denominator yields null, never infinity or a qualifying ratio",
               "runtime": "Original sources may mix HPC/local software and hardware; timings are not hardware-matched comparisons",
               "research_scope": "Section22 is a training-protocol diagnostic under RQ2/RO2, not a mechanism, innovation claim, or overall research completion"},
           "recall": [], "aggregate_metrics": [], "G1": [], "G2": [],
           "named_class_decision_evidence": []}
    for dataset in DATASETS:
        if dataset not in references:
            continue
        classes = references[dataset]["classes"]
        tasks = references[dataset]["tasks"]
        for budget in BUDGETS:
            for method in ("joint", *METHODS):
                if method == "joint" and budget != "original":
                    continue
                for cid, name in enumerate(classes):
                    vals = vector(budget, dataset, method, "recall", cid)
                    original = vector("original", dataset, method, "recall", cid)
                    joint = vector("original", dataset, "joint", "recall", cid)
                    before_final = {}
                    arm = indexes[budget]
                    if cid not in tasks[-1]:
                        for s in SEEDS:
                            record = arm.get((dataset, method, s))
                            if record and record["previous"]:
                                before_final[s] = record["previous"]["recall"].get(cid)
                    out["recall"].append({"dataset": dataset, "budget": budget, "method": method,
                                          "class_id": cid, "class_name": name,
                                          "role": "new_final_task" if cid in tasks[-1] else "old",
                                          "train_count": references[dataset]["train_counts"][str(cid)],
                                          "test_support": references[dataset]["support"][cid],
                                          "recall": summarize(vals),
                                          "delta_from_original_pp": paired_delta(vals, original),
                                          "joint_minus_method_pp": paired_delta(joint, vals),
                                          "old_recall_before_final_task": summarize(before_final) if cid not in tasks[-1] else None,
                                          "last_task_recall_change_pp": paired_delta(vals, before_final) if cid not in tasks[-1] else None})
                for metric in ("macro_f1", "balanced_accuracy"):
                    vals = vector(budget, dataset, method, metric)
                    out["aggregate_metrics"].append({"dataset": dataset, "budget": budget, "method": method,
                                                     "metric": metric, "value": summarize(vals),
                                                     "delta_from_original_pp": paired_delta(vals, vector("original", dataset, method, metric))})
            for metric in ("macro_f1", "balanced_accuracy"):
                result = fixed_best({m: vector(budget, dataset, m, metric) for m in ("joint", *METHODS)})
                out["G1"].append({"dataset": dataset, "budget": budget, "metric": metric, **result})
            for cid, name in enumerate(classes):
                result = fixed_best({m: vector(budget, dataset, m, "recall", cid) for m in ("joint", *METHODS)})
                out["G2"].append({"dataset": dataset, "budget": budget, "class_name": name,
                                  "class_id": cid, "classwise_envelope_not_one_classifier": True, **result})
    for dataset, name in NAMED_CLASSES:
        cid = references[dataset]["classes"].index(name) if dataset in references else None
        replay = vector("2000", dataset, "replay", "recall", cid) if cid is not None else {}
        joint = vector("original", dataset, "joint", "recall", cid) if cid is not None else {}
        rs, js = summarize(replay), summarize(joint)
        ratio = rs["mean"] / js["mean"] if rs["complete"] and js["complete"] and js["mean"] > 0 else None
        out["named_class_decision_evidence"].append({"dataset": dataset, "class_name": name,
                                                    "replay_2000": rs, "joint_original": js,
                                                    "ratio_of_five_seed_means": ratio,
                                                    "paired_ratio_by_seed": {str(s): replay[s] / joint[s] if replay.get(s) is not None and joint.get(s) is not None and joint[s] > 0 else None for s in SEEDS},
                                                    "replay_minus_joint_pp": paired_delta(replay, joint),
                                                    "passes_A_threshold": at_least(ratio, .5) if ratio is not None else None,
                                                    "passes_B_threshold": strictly_below(ratio, .2) if ratio is not None else None})
    out["section22_3"] = decide(out["named_class_decision_evidence"], complete, smoke)
    return out


def load_cells(root):
    if not root.exists():
        return [], []
    if not root.is_dir():
        raise ValueError("Input root must be a directory")
    paths, cells = [], []
    for path in sorted(root.rglob("result.json")):
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if "identity" not in value:
            if "cells" in value:
                continue
            raise ValueError(f"Unrecognized result schema: {path.name}")
        paths.append(path)
        cells.append(value)
    return paths, cells


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def export_tables(out, output_dir):
    recall_rows, metric_rows = [], []
    for row in out["recall"]:
        for s in SEEDS:
            recall_rows.append({k: row[k] for k in ("dataset", "budget", "method", "class_name", "role", "train_count", "test_support")} | {
                "seed": s, "recall": row["recall"]["by_seed"][str(s)],
                "delta_from_original_pp": row["delta_from_original_pp"]["by_seed"][str(s)],
                "joint_minus_method_pp": row["joint_minus_method_pp"]["by_seed"][str(s)],
                "old_recall_before_final_task": row["old_recall_before_final_task"]["by_seed"][str(s)] if row["role"] == "old" else None,
                "last_task_recall_change_pp": row["last_task_recall_change_pp"]["by_seed"][str(s)] if row["role"] == "old" else None})
    for row in out["aggregate_metrics"]:
        for s in SEEDS:
            metric_rows.append({k: row[k] for k in ("dataset", "budget", "method", "metric")} | {
                "seed": s, "value": row["value"]["by_seed"][str(s)],
                "delta_from_original_pp": row["delta_from_original_pp"]["by_seed"][str(s)]})
    recall_columns = ["dataset", "budget", "method", "class_name", "role", "train_count", "test_support", "seed", "recall", "delta_from_original_pp", "joint_minus_method_pp", "old_recall_before_final_task", "last_task_recall_change_pp"]
    write_csv(output_dir / "recall_by_seed.csv", recall_rows, recall_columns)
    write_csv(output_dir / "metrics_by_seed.csv", metric_rows, ["dataset", "budget", "method", "metric", "seed", "value", "delta_from_original_pp"])
    recall_summary = [{k: row[k] for k in ("dataset", "budget", "method", "class_name", "role")} | {
        "n": row["recall"]["n"], "complete": row["recall"]["complete"],
        "mean": row["recall"]["mean"], "sample_sd": row["recall"]["sample_sd"],
        "delta_from_original_pp_mean": row["delta_from_original_pp"]["mean"],
        "delta_from_original_pp_sample_sd": row["delta_from_original_pp"]["sample_sd"]} for row in out["recall"]]
    write_csv(output_dir / "recall_summary.csv", recall_summary, ["dataset", "budget", "method", "class_name", "role", "n", "complete", "mean", "sample_sd", "delta_from_original_pp_mean", "delta_from_original_pp_sample_sd"])
    metric_summary = [{k: row[k] for k in ("dataset", "budget", "method", "metric")} | {
        "n": row["value"]["n"], "complete": row["value"]["complete"],
        "mean": row["value"]["mean"], "sample_sd": row["value"]["sample_sd"],
        "delta_from_original_pp_mean": row["delta_from_original_pp"]["mean"],
        "delta_from_original_pp_sample_sd": row["delta_from_original_pp"]["sample_sd"]} for row in out["aggregate_metrics"]]
    write_csv(output_dir / "metrics_summary.csv", metric_summary, ["dataset", "budget", "method", "metric", "n", "complete", "mean", "sample_sd", "delta_from_original_pp_mean", "delta_from_original_pp_sample_sd"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--steps-500", required=True, type=Path)
    parser.add_argument("--steps-2000", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    roots = (args.baseline, args.steps_500, args.steps_2000)
    if any(args.output_dir.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(args.output_dir.resolve()) for root in roots):
        raise ValueError("Analysis output must be separate from all immutable raw input roots")
    loaded = [load_cells(root) for root in roots]
    out = analyze(*(cells for _, cells in loaded))
    out["inputs"] = [{"budget": budget, "files": [
        {"file": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in paths]} for budget, root, (paths, _) in zip(BUDGETS, roots, loaded)]
    repo = Path(__file__).resolve().parents[1]
    def portable(path):
        try:
            return path.resolve().relative_to(repo).as_posix()
        except ValueError:
            return path.name
    command = ["python", "-I", "ncd/analyze_step_budget.py", "--baseline", portable(args.baseline),
               "--steps-500", portable(args.steps_500), "--steps-2000", portable(args.steps_2000),
               "--output-dir", portable(args.output_dir)]
    out["analysis_provenance"] = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": {"datasets": DATASETS, "methods": METHODS, "seeds": SEEDS, "budgets": BUDGETS},
        "seed": None, "seed_note": "Deterministic paired descriptive aggregation; no random sampling",
        "command": subprocess.list2cmdline(command), "log_path": portable(args.output_dir / "analysis.log"),
        "experiment_source_commits": sorted({c["identity"]["git_commit"] for _, cells in loaded for c in cells})}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    export_tables(out, args.output_dir)
    message = json.dumps({"execution_status": out["execution_status"], "coverage": {b: c["complete_cells"] for b, c in out["coverage"].items()}, "section22_3": out["section22_3"]})
    (args.output_dir / "analysis.log").write_text(message + "\n", encoding="utf-8")
    print(message)


if __name__ == "__main__":
    main()
