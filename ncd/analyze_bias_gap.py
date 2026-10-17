"""Section23.2: descriptive S2000 versus original Joint error decomposition.

Reads existing small result files only. No training, tuning, weight or data reads.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ncd.analyze_problem_scout import error_flow
from ncd.analyze_step_budget import (
    DATASETS, METHODS, SEEDS, OPERATIONAL_CONFIG, close, paired_delta,
    summarize, validate_cell,
)

EXPECTED_BEST = {"NSL-KDD": "replay_lwf", "UNSW-NB15": "rcl_tab", "CIC-IDS-2017": "derpp"}
BINS = ("normal", "newer_classes", "other_old_or_same_task")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def load_arm(root, baseline=False):
    """Select only 15 Joint inputs from the historical 120-cell baseline."""
    root = root.resolve()
    index_path = root / "result.json"
    index = json.loads(index_path.read_text(encoding="utf-8-sig"))
    require(index.get("execution_status") == "complete" and not index.get("smoke"), "Input index is not complete formal evidence")
    expected = {(d, m, s) for d in DATASETS for m in (("joint",) if baseline else METHODS) for s in SEEDS}
    records, evidence, seen, configs = {}, [], set(), {}
    for row in index["cells"]:
        key = (row["dataset"], row["method"], row["seed"])
        require(key not in seen, "Duplicate input index cell")
        seen.add(key)
        if baseline and key not in expected:
            continue
        require(key in expected, "Unexpected S2000 cell")
        path = (root / row["result"]).resolve()
        require(root in path.parents and path.name == "result.json", "Input result escapes source root")
        digest = sha(path)
        require(digest == row["result_sha256"], "Indexed result hash mismatch")
        cell = json.loads(path.read_text(encoding="utf-8-sig"))
        identity = cell["identity"]
        require(tuple(identity[k] for k in ("dataset", "method", "seed")) == key, "Index/identity mismatch")
        require(cell.get("completed") is True, "Incomplete cell")
        config = identity["effective_config"]
        require(config.get("smoke") is False and config["batch_size"] == 256, "Wrong formal/batch configuration")
        require(canonical(config) == identity["effective_config_sha256"], "Effective config hash mismatch")
        configs[identity["effective_config_sha256"]] = config
        require(len(cell["checkpoints"]) == row["checkpoints"], "Checkpoint count mismatch")
        for cp in cell["checkpoints"]:
            require(cp["identity_hash"] == canonical(identity), "Checkpoint identity hash mismatch")
        records[key] = validate_cell(cell, "original" if baseline else "2000")
        evidence.append({"dataset": key[0], "method": key[1], "seed": key[2],
                         "result": row["result"], "sha256": digest,
                         "git_commit": identity["git_commit"],
                         "config_sha256": identity["config_sha256"],
                         "effective_config_sha256": identity["effective_config_sha256"],
                         "data_metadata_sha256": identity["data_metadata_sha256"],
                         "source_hashes": identity["source_hashes"],
                         "runtime_versions": identity.get("runtime_versions"),
                         "checkpoints": len(cell["checkpoints"])})
    require(set(records) == expected, "Required input coverage is incomplete")
    return records, {"index_sha256": sha(index_path), "selected_cells": len(records),
                     "selected_checkpoints": sum(len(r["cell"]["checkpoints"]) for r in records.values()),
                     "effective_configs_by_sha256": configs, "files": evidence}


def class_metrics(metrics, classes):
    cm = metrics["confusion_matrix"]
    saved = {r["class_id"]: r for r in metrics["per_class"]}
    result = []
    for c, name in enumerate(classes):
        support, predicted, tp = sum(cm[c]), sum(row[c] for row in cm), cm[c][c]
        values = {"support": support, "predicted": predicted, "tp": tp,
                  "fp": predicted - tp, "fn": support - tp,
                  "precision": ratio(tp, predicted), "recall": ratio(tp, support),
                  "f1": ratio(2 * tp, support + predicted)}
        # Preserve undefined class F1 as null; the original macro definition
        # includes a zero term for a fully empty class, separately named here.
        values["f1_zero_division"] = values["f1"] is None
        values["macro_f1_term"] = values["f1"] if values["f1"] is not None else 0.0
        for metric in ("precision", "recall", "f1"):
            computed = values["macro_f1_term"] if metric == "f1" else values[metric]
            require(close(computed, saved[c][metric]), f"Stored {name} {metric} disagrees with confusion counts")
        result.append(values)
    return result


def validate_comparability(joint, cil):
    references, configs = {}, []
    for records in (joint, cil):
        for (dataset, _, _), record in records.items():
            cell, identity = record["cell"], record["cell"]["identity"]
            signature = {"class_order": cell["class_order"], "tasks": cell["tasks"],
                         "train_counts": cell["train_counts"], "support": record["final"]["support"],
                         "data_binding": identity["data_binding"], "feature_names": identity["feature_names"],
                         "data_metadata_sha256": identity["data_metadata_sha256"]}
            require(dataset not in references or references[dataset] == signature, "Class/data/test-support binding mismatch")
            references[dataset] = signature
            configs.append({k: v for k, v in identity["effective_config"].items() if k not in OPERATIONAL_CONFIG})
    require(all(c == configs[0] for c in configs), "Scientific configuration differs beyond step floor/recorded runtime fields")
    return references, configs[0]


def summarize_columns(rows, columns):
    return {name: summarize({r["seed"]: r[name] for r in rows}) for name in columns}


def analyze(joint, cil):
    references, config = validate_comparability(joint, cil)
    out = {"schema_version": 1, "execution_status": "complete", "scientific_config": config,
           "definitions": {
               "selection": "One S2000 method per dataset: highest five-seed mean final ID Macro-F1 among all five methods; tie broken lexicographically. No per-seed oracle or tuning.",
               "units": "Metric levels are fractions; Joint-minus-CIL differences and additive contributions are percentage points.",
               "contribution": "100*(Joint class F1-CIL class F1)/K; signed contributions sum to the paired Macro-F1 gap. This is an arithmetic decomposition, not a causal attribution to precision or recall.",
               "aggregation": "Arithmetic mean and sample SD (ddof=1) of exact paired seeds 1,2,3,4,42. Pooled-count ratios are separately labelled and are not means of seed ratios.",
               "zero_denominator": "Undefined precision, recall, F1 and count ratios are null, never zero-filled. A fully empty class has f1_zero_division=true; its separately named macro_f1_term is zero to preserve the original macro averaging convention.",
               "g4": "Old true classes exclude the final task. Errors to Normal take precedence; otherwise newer means introduced strictly later than the true class; remaining errors are other-old-or-same-task. Same-task attacks are not newer. Report all-old and old-attacks-only populations separately.",
               "normal_fpr": "Normal rows predicted as any attack / all Normal rows; paired excess is CIL minus Joint.",
               "scope": "Section23.2 final ID decomposition only; original eight-epoch Joint reused unchanged, not a universal empirical upper bound. No S1 shift, training, hyperparameter selection, causal mechanism or innovation conclusion.",
               "runtime": "Historical Joint mixes local/HPC runtime versions; no hardware-matched timing inference."},
           "selection": [], "datasets": [], "class_by_seed": [], "class_summary": [],
           "fp_sources_by_seed": [], "fp_sources_summary": [], "fp_sources_paired": [], "g4_class_by_seed": [],
           "g4_by_seed": [], "g4_summary": []}
    for dataset in DATASETS:
        reference = references[dataset]
        classes, tasks = reference["class_order"], reference["tasks"]
        n, normal = len(classes), classes.index("Normal")
        candidates = {m: summarize({s: cil[dataset, m, s]["final"]["macro_f1"] for s in SEEDS}) for m in METHODS}
        best = min(METHODS, key=lambda m: (-candidates[m]["mean"], m))
        require(best == EXPECTED_BEST[dataset], "Selected best differs from declared Section22 result")
        out["selection"].append({"dataset": dataset, "best_cil_method": best, "candidates_macro_f1": candidates})
        arm_metrics = {"joint": {}, "cil": {}}
        class_rows, fp_rows, flow_rows = [], [], []
        for seed in SEEDS:
            records = {"joint": joint[dataset, "joint", seed], "cil": cil[dataset, best, seed]}
            matrices, metrics = {}, {}
            for arm, record in records.items():
                matrices[arm] = record["cell"]["checkpoints"][-1]["id"]["confusion_matrix"]
                metrics[arm] = class_metrics(record["cell"]["checkpoints"][-1]["id"], classes)
                arm_metrics[arm][seed] = record["final"] | {"normal_fpr": ratio(metrics[arm][normal]["fn"], metrics[arm][normal]["support"])}
                require(close(arm_metrics[arm][seed]["normal_fpr"], record["cell"]["checkpoints"][-1]["id"]["normal_fpr"]), "Normal FPR mismatch")
                total_errors = sum(row["fp"] for row in metrics[arm])
                for predicted, name in enumerate(classes):
                    for source, source_name in enumerate(classes):
                        if source == predicted:
                            continue
                        count = matrices[arm][source][predicted]
                        fp_rows.append({"dataset": dataset, "best_cil_method": best, "arm": arm, "seed": seed,
                                        "predicted_class": name, "true_class": source_name, "count": count,
                                        "receiver_fp": metrics[arm][predicted]["fp"],
                                        "fraction_of_receiver_fp": ratio(count, metrics[arm][predicted]["fp"]),
                                        "fraction_of_all_errors": ratio(count, total_errors),
                                        "fraction_of_source_support": ratio(count, metrics[arm][source]["support"])})
                old_classes = [c for group in tasks[:-1] for c in group]
                flows = {c: error_flow(matrices[arm], c, tasks, normal) for c in old_classes}
                saved_classes = {r["class_id"]: r for r in record["cell"]["checkpoints"][-1]["id"]["per_class"]}
                for c, flow in flows.items():
                    saved = saved_classes[c]
                    for name, saved_name in (("normal", "error_to_normal"), ("newer_classes", "error_to_newer_class"), ("other_old_or_same_task", "error_to_other_old")):
                        require(flow["counts"][name] == saved[saved_name], "Stored G4 counts disagree")
                    out["g4_class_by_seed"].append({"dataset": dataset, "best_cil_method": best, "arm": arm, "seed": seed, "true_class": classes[c], "support": flow["support"], "errors": flow["errors"], **flow["counts"]})
                for population in ("all_old", "old_attacks_only"):
                    chosen = [flow for c, flow in flows.items() if population == "all_old" or c != normal]
                    support, errors = sum(f["support"] for f in chosen), sum(f["errors"] for f in chosen)
                    row = {"dataset": dataset, "best_cil_method": best, "arm": arm, "seed": seed, "population": population, "support": support, "errors": errors}
                    for name in BINS:
                        count = sum(f["counts"][name] for f in chosen)
                        row[name] = count
                        row[name + "_fraction_of_errors"] = ratio(count, errors)
                        row[name + "_fraction_of_support"] = ratio(count, support)
                    require(sum(row[b] for b in BINS) == errors, "G4 bins do not exhaust errors")
                    flow_rows.append(row)
            contributions = []
            for c, name in enumerate(classes):
                row = {"dataset": dataset, "best_cil_method": best, "seed": seed, "class_name": name,
                       "class_id": c, "role": "new" if c in tasks[-1] else "old", "train_count": reference["train_counts"][str(c)]}
                for arm in ("joint", "cil"):
                    row.update({arm + "_" + k: v for k, v in metrics[arm][c].items()})
                for metric in ("precision", "recall", "f1"):
                    j, k = metrics["joint"][c][metric], metrics["cil"][c][metric]
                    row[metric + "_gap_pp"] = 100 * (j - k) if j is not None and k is not None else None
                for count in ("tp", "fp", "fn"):
                    row[count + "_gap"] = metrics["joint"][c][count] - metrics["cil"][c][count]
                row["macro_f1_contribution_pp"] = 100 * (metrics["joint"][c]["macro_f1_term"] - metrics["cil"][c]["macro_f1_term"]) / n
                contributions.append(row["macro_f1_contribution_pp"])
                class_rows.append(row)
            gap = 100 * (arm_metrics["joint"][seed]["macro_f1"] - arm_metrics["cil"][seed]["macro_f1"])
            require(math.isclose(math.fsum(contributions), gap, abs_tol=1e-12, rel_tol=1e-12), "Class contributions do not reconcile to Macro-F1 gap")
        dataset_row = {"dataset": dataset, "best_cil_method": best, "classes": classes, "tasks": tasks, "data_binding": reference["data_binding"]}
        for metric in ("macro_f1", "balanced_accuracy", "normal_fpr"):
            j = {s: arm_metrics["joint"][s][metric] for s in SEEDS}
            k = {s: arm_metrics["cil"][s][metric] for s in SEEDS}
            dataset_row[metric] = {"joint": summarize(j), "cil": summarize(k), "joint_minus_cil_pp": paired_delta(j, k)}
            if metric == "normal_fpr":
                dataset_row[metric]["cil_minus_joint_pp"] = paired_delta(k, j)
        out["datasets"].append(dataset_row)
        out["class_by_seed"].extend(class_rows)
        numeric = [k for k in class_rows[0] if k not in ("dataset", "best_cil_method", "seed", "class_name", "class_id", "role", "train_count") and not k.endswith("zero_division")]
        for name in classes:
            rows = [r for r in class_rows if r["class_name"] == name]
            out["class_summary"].append({"dataset": dataset, "best_cil_method": best, "class_name": name, "role": rows[0]["role"], **summarize_columns(rows, numeric)})
        out["fp_sources_by_seed"].extend(fp_rows)
        for arm in ("joint", "cil"):
            for predicted in classes:
                for source in classes:
                    if source == predicted:
                        continue
                    rows = [r for r in fp_rows if r["arm"] == arm and r["predicted_class"] == predicted and r["true_class"] == source]
                    out["fp_sources_summary"].append({"dataset": dataset, "best_cil_method": best, "arm": arm, "predicted_class": predicted, "true_class": source,
                        **summarize_columns(rows, ("count", "receiver_fp", "fraction_of_receiver_fp", "fraction_of_all_errors", "fraction_of_source_support")),
                        "pooled_count": sum(r["count"] for r in rows), "pooled_receiver_fp": sum(r["receiver_fp"] for r in rows),
                        "pooled_fraction_of_receiver_fp": ratio(sum(r["count"] for r in rows), sum(r["receiver_fp"] for r in rows))})
        for predicted in classes:
            for source in classes:
                if source == predicted:
                    continue
                selected = {arm: {r["seed"]: r for r in fp_rows if r["arm"] == arm and r["predicted_class"] == predicted and r["true_class"] == source} for arm in ("joint", "cil")}
                row = {"dataset": dataset, "best_cil_method": best, "predicted_class": predicted, "true_class": source,
                       "direction": "Joint minus CIL; counts in examples, fraction differences in fraction units"}
                for name in ("count", "receiver_fp", "fraction_of_receiver_fp", "fraction_of_all_errors", "fraction_of_source_support"):
                    row[name] = paired_delta({s: selected["joint"][s][name] for s in SEEDS}, {s: selected["cil"][s][name] for s in SEEDS}, scale=1)
                out["fp_sources_paired"].append(row)
        out["g4_by_seed"].extend(flow_rows)
        for population in ("all_old", "old_attacks_only"):
            row = {"dataset": dataset, "best_cil_method": best, "population": population, "arms": {}, "joint_minus_cil": {}}
            columns = [k for k in flow_rows[0] if k not in ("dataset", "best_cil_method", "arm", "seed", "population")]
            for arm in ("joint", "cil"):
                rows = [r for r in flow_rows if r["arm"] == arm and r["population"] == population]
                pooled = {k: sum(r[k] for r in rows) for k in ("support", "errors", *BINS)}
                pooled.update({b + "_fraction_of_errors": ratio(pooled[b], pooled["errors"]) for b in BINS})
                pooled.update({b + "_fraction_of_support": ratio(pooled[b], pooled["support"]) for b in BINS})
                row["arms"][arm] = {"seed_summary": summarize_columns(rows, columns), "pooled_counts": pooled}
            for name in columns:
                j = {s: row["arms"]["joint"]["seed_summary"][name]["by_seed"][str(s)] for s in SEEDS}
                k = {s: row["arms"]["cil"]["seed_summary"][name]["by_seed"][str(s)] for s in SEEDS}
                row["joint_minus_cil"][name] = paired_delta(j, k, scale=1)
            out["g4_summary"].append(row)
    return out


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        elif isinstance(item, list):
            result[name] = json.dumps(item, separators=(",", ":"))
        else:
            result[name] = item
    return result


def write_outputs(out, directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "analysis.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name in ("datasets", "class_by_seed", "class_summary", "fp_sources_by_seed", "fp_sources_summary", "fp_sources_paired", "g4_class_by_seed", "g4_by_seed", "g4_summary"):
        rows = [flatten(row) for row in out[name]]
        columns = list(dict.fromkeys(k for r in rows for k in r))
        with (directory / (name + ".csv")).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    message = {"execution_status": "complete", "selected_cells": {k: v["selected_cells"] for k, v in out["provenance"]["inputs"].items()},
               "gap_pp": {r["dataset"]: r["macro_f1"]["joint_minus_cil_pp"]["mean"] for r in out["datasets"]},
               "additive_reconciliation": "PASS for each of 15 paired seeds at 1e-12 tolerance"}
    (directory / "analysis.log").write_text(json.dumps(message, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(message, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--steps-2000", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    paths = [p.resolve() for p in (args.baseline, args.steps_2000, args.output_dir)]
    require(all(repo in p.parents for p in paths), "All inputs/outputs must be repository-relative for portable provenance")
    require(all(paths[2] != p and p not in paths[2].parents and paths[2] not in p.parents for p in paths[:2]), "Output cannot overlap input roots")
    joint, old_evidence = load_arm(paths[0], baseline=True)
    cil, new_evidence = load_arm(paths[1])
    out = analyze(joint, cil)
    old_evidence["root"], new_evidence["root"] = (p.relative_to(repo).as_posix() for p in paths[:2])
    out["provenance"] = {"created_utc": datetime.now(timezone.utc).isoformat(),
                         "analysis_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                         "analysis_source_sha256": sha(Path(__file__)),
                         "helper_sha256": {str(p.relative_to(repo).as_posix()): sha(p) for p in (repo / "ncd/analyze_step_budget.py", repo / "ncd/analyze_problem_scout.py")},
                         "command": "python -I ncd/analyze_bias_gap.py --baseline " + old_evidence["root"] + " --steps-2000 " + new_evidence["root"] + " --output-dir " + paths[2].relative_to(repo).as_posix(),
                         "inputs": {"original_joint": old_evidence, "s2000": new_evidence},
                         "validation_scope": "Selected 15 original Joint and all 75 S2000 small results, index/file/config/identity hashes, source coverage, task schedules, final and previous confusion summaries; no dataset or weight payload audit."}
    write_outputs(out, paths[2])


if __name__ == "__main__":
    main()
