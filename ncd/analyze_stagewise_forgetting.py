"""Section28.6--28.7 stagewise alpha=2% CIL analysis; saved checkpoints only.

Reuse already calibrated stage CMs and valid final Section27.9 caches. Only
missing baseline stages need CPU logits. No training, weight/hash audit, or
test-selected threshold. Mean incremental accuracy means seen-class BAcc.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from ncd.model import MLP
from ncd.calibrate_aacil_baselines import select_taus, CALIBRATION_PROTOCOL, shared_read
from ncd.run_problem_scout import save_json

BASE = ROOT / "ncd/results/problem_scout"
SEEDS = [1, 2, 3, 4, 42]
METHODS = {"replay", "derpp", "wacb", "freq", "farq", "farq_all", "joint",
           "aacil", "ts_mem", "aa_noattmem", "replay_bs", "aacil_der", "aacil_kd", "aacil_nogkd", "aacil_iso",
           "aacil_wa", "aacil_cbf", "aacil_ncm"}
SECTION31_METHODS = {"aacil_kd_weak", "aacil_kd_wa", "aacil_kd_cbf", "aacil_ema_kd", "aacil_dual"}
METHODS |= SECTION31_METHODS
SECTION31_ROOT = BASE / "section31/formal_local_v1"
SECTION31_CONFIG_KEYS = ("section31", "section31_schema_version", "lambda_g",
    "derpp_alpha", "derpp_beta", "cbf_epochs", "cbf_lr_multiplier", "ema_decay", "ema_consistency",
    "family_parameter_constraint", "future_output_policy", "gate_step_policy", "calibration_policy")
SECTION29_METHODS = {"aacil_kd", "aacil_nogkd", "aacil_iso", "aacil_wa", "aacil_cbf", "aacil_ncm"}
SECTION29_CONFIG_KEYS = ("section298", "section298_schema_version", "lambda_g",
                         "derpp_alpha", "derpp_beta", "cbf_epochs", "cbf_lr_multiplier")
DEFAULT_ROOTS = [BASE / p for p in (
    "section26/local_v2", "section26/local_cic_v2", "section27/local_v1",
    "section28/replay_bs_local_v1", "section28/cic_local_queue_v1",
    "section28/cic_aacil_fallback_v1", "section28/hpc_cic_section28", "section29/local_v1",
    "section29/retention_local_v1", "section29/section299_cic_kd_local_v1",
    "section29/section2911_cic_kd_resume_v1", "section27/hpc_cic_section2911")]
PROTOCOL = {
    "section": "28.6--28.7/29.3", "version": 1, "alpha": .02,
    "average_incremental_accuracy": "Unweighted mean of stage seen-class balanced accuracies",
    "forgetting": "For each class introduced before the final task: maximum recall over preceding stages minus final recall; unweighted class mean; signed values retained",
    "primary_old_class_scope": "All old classes, including Normal",
    "secondary_old_class_scope": "Old attack classes only, excluding Normal",
    "selection": "Earliest completed physical source per dataset/method/seed, never by metric",
    "calibration": "Common permanent TRAIN Normal holdout at each stage; Section27.9 Normal-only tau extension for multiclass models; stored strict gate threshold for AA variants",
    "prior": "Original prepared TRAIN class frequencies, including permanently heldout rows; no TEST prior",
    "seen_labels": "Mask model outputs and TEST rows to the current seen classes",
    "decomposition": "For an attack class at its earliest maximum-recall preceding stage: change in unconditional Normal-miss rate plus change in other-attack-family rate equals signed recall loss",
    "no_counterfactual_family_inference": True,
    "registered_section27_8_unchanged": True,
}


def now():
    return datetime.now().astimezone().isoformat()


def label(path):
    return path.resolve().relative_to(ROOT).as_posix()


def group_role(method):
    if method == "joint":
        return "offline_reference_not_deployment_matched"
    if method in {"farq", "farq_all"}:
        return "supplementary_FARQ"
    if method in {"replay_bs", "aacil_der"}:
        return "post_hoc_control"
    if method in SECTION29_METHODS:
        return "section29_candidate_or_ablation"
    if method in SECTION31_METHODS:
        return "compact_candidate_formal_results_on_development_datasets"
    return "primary_method_or_registered_comparator"


def stage_metrics(value, seen, names):
    cm = np.asarray(value, dtype=np.int64)
    n, normal = len(names), names.index("Normal")
    if cm.shape != (n, n) or np.any(cm < 0):
        raise ValueError("Invalid stage confusion matrix")
    support, predicted, tp = cm.sum(1), cm.sum(0), cm.diagonal()
    if any(support[c] <= 0 for c in seen):
        raise ValueError("A seen class is absent from the stage test support")
    unseen = sorted(set(range(n))-set(seen))
    if unseen and (cm[unseen, :].sum() or cm[:, unseen].sum()):
        raise ValueError("Stage evaluation includes unseen labels")
    per_class = []
    for c in seen:
        miss = None if c == normal else float(cm[c, normal]/support[c])
        wrong_family = None if c == normal else float((support[c]-tp[c]-cm[c, normal])/support[c])
        per_class.append({"class_id": c, "class_name": names[c], "support": int(support[c]),
            "precision": float(tp[c]/predicted[c]) if predicted[c] else None,
            "recall": float(tp[c]/support[c]), "f1": float(2*tp[c]/(support[c]+predicted[c])),
            "gate_normal_miss_rate": miss, "other_attack_family_rate": wrong_family,
            "family_error_given_attack_alert": float((support[c]-tp[c]-cm[c,normal])/(support[c]-cm[c,normal]))
                if c != normal and support[c] > cm[c,normal] else None})
    attacks = [c for c in seen if c != normal]
    return {"confusion_matrix": cm.tolist(), "seen_classes": seen, "per_class": per_class,
            "balanced_accuracy": float(np.mean([tp[c]/support[c] for c in seen])),
            "macro_f1": float(np.mean([2*tp[c]/(support[c]+predicted[c]) for c in seen])),
            "normal_fpr": float(1-tp[normal]/support[normal]),
            "attack_detection": float(1-cm[attacks,normal].sum()/support[attacks].sum()) if attacks else None,
            "evaluated_test_rows": int(support.sum())}


def aggregate_trajectory(stages, tasks, names):
    old = [c for task in tasks[:-1] for c in task]
    normal = names.index("Normal")
    final = {r["class_id"]: r for r in stages[-1]["metrics"]["per_class"]}
    records = []
    for c in old:
        candidates = [(s["task_index"], r) for s in stages[:-1]
                      for r in s["metrics"]["per_class"] if r["class_id"] == c]
        best_task, best = max(candidates, key=lambda pair: (pair[1]["recall"], -pair[0]))
        last = final[c]
        row = {"class_id": c, "class_name": names[c], "best_pre_final_task": best_task,
               "best_pre_final_recall": best["recall"], "final_recall": last["recall"],
               "forgetting": best["recall"]-last["recall"], "included_primary": True,
               "included_attack_only": c != normal}
        if c != normal:
            row.update(gate_normal_miss_increase=last["gate_normal_miss_rate"]-best["gate_normal_miss_rate"],
                       other_attack_family_error_increase=last["other_attack_family_rate"]-best["other_attack_family_rate"])
        else:
            row["normal_false_alarm_increase"] = best["recall"]-last["recall"]
        records.append(row)
    attack = [r for r in records if r["included_attack_only"]]
    return {"average_incremental_accuracy": statistics.mean(s["metrics"]["balanced_accuracy"] for s in stages),
            "final_balanced_accuracy": stages[-1]["metrics"]["balanced_accuracy"],
            "final_macro_f1": stages[-1]["metrics"]["macro_f1"],
            "final_normal_fpr": stages[-1]["metrics"]["normal_fpr"],
            "final_attack_detection": stages[-1]["metrics"]["attack_detection"],
            "forgetting_all_old_classes": statistics.mean(r["forgetting"] for r in records) if records else None,
            "forgetting_old_attack_classes": statistics.mean(r["forgetting"] for r in attack) if attack else None,
            "old_class_count_including_normal": len(records), "old_attack_class_count": len(attack),
            "per_old_class": sorted(records, key=lambda r: (-r["forgetting"], r["class_id"])),
            "attack_decomposition_mean": {k: statistics.mean(r[k] for r in attack) if attack else None
                for k in ("gate_normal_miss_increase", "other_attack_family_error_increase")}}


def prepare_data(identity, override, cache):
    dataset = identity["dataset"]
    if dataset not in cache:
        parent = override or ROOT / identity["effective_config"]["problem_scout_data_root"]
        folder = parent / dataset.lower()
        if not (folder / "test_x.npy").exists() and override is None:
            cfg = shared_read(ROOT / "ncd/configs/problem_scout_section21.json")
            folder = ROOT / cfg["problem_scout_data_root"] / dataset.lower()
        cache[dataset] = {name: np.load(folder/file, mmap_mode="r") for name, file in (
            ("x_train", "train_x.npy"), ("y_train", "train_y.npy"),
            ("x_test", "test_x.npy"), ("y_test", "test_y.npy"))}
    return cache[dataset]


def logits(model, x, rows, seen, batch):
    parts = []
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            xx = torch.from_numpy(np.array(x[rows[start:start+batch]], copy=True))
            parts.append(model(xx)[:,seen])
    out = torch.cat(parts)
    if not torch.isfinite(out).all():
        raise ValueError("Nonfinite stage logits")
    return out


def stage_binding(path, source, checkpoint):
    identity = source["identity"]
    binding = {"source_result": label(path), "source_completed_at": source["completed_at"],
            "training_source_commit": identity["git_commit"], "config_sha256": identity["config_sha256"],
            "identity_hash": checkpoint.get("identity_hash"),
            "checkpoint_file": checkpoint["checkpoint_file"], "checkpoint_sha256_recorded": checkpoint["checkpoint_sha256"],
            "task_index": checkpoint["task_index"], "seen_classes": checkpoint["seen_classes"], "protocol": PROTOCOL}
    if identity["method"] in SECTION31_METHODS:
        # Bind the saved evaluation and effective CLI configuration explicitly;
        # no compact model checkpoint is loaded or forwarded by this analyzer.
        canonical = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        binding.update(compact_saved_alpha2_sha256=canonical(checkpoint["operating_points"]["0.02"]),
            compact_effective_config_sha256=canonical(identity["effective_config"]),
            compact_source_hashes=identity["source_hashes"],
            compact_source_record_sha256=canonical(source))
    if source.get("section39_execution"):
        binding["section39_original_result_sha256"] = source["publication_derivation"]["original_result_sha256"]
        binding["section39_invocation_sha256"] = source["section39_execution"]["invocation_sha256"]
    return binding


def get_stage(path, source, cp, target, args, data_cache):
    started = time.perf_counter()
    binding = stage_binding(path, source, cp)
    if target.exists():
        row = shared_read(target)
        if row.get("completed") and row["binding"] == binding:
            return row, "reused_stage_cache"
        raise ValueError("Existing stage cache differs; preserve it and select a new output namespace")
    ident, names, seen = source["identity"], source["class_order"], cp["seen_classes"]
    calibration, cm, mode, costs = None, None, None, {}
    if "0.02" in cp.get("operating_points", {}):
        cm = cp["operating_points"]["0.02"]["confusion_matrix"]
        if "calibration" in cp:
            calibration = cp["calibration"]["0.02"]
        else:
            cal = cp["normal_holdout_calibration"]
            calibration = {"operating_point": cal["operating_points"]["0.02"],
                           "protocol": cal["protocol"], "original_train_counts": cal["original_train_counts"],
                           "calibration_rows": cal["calibration_rows"]}
        mode = "reused_saved_stage_alpha2_no_forward"
        costs["existing_stage_calibration_seconds_not_recharged"] = cp.get("calibration_seconds")
        costs["existing_stage_evaluation_seconds_not_recharged"] = cp.get("evaluation_seconds")
    if cm is None and (source.get("section39_execution") or ident["method"] in SECTION29_METHODS | SECTION31_METHODS):
        raise ValueError("Formal retention/compact stages require their saved calibrated alpha=2% CM; no checkpoint inference fallback is permitted")
    if cm is None and cp["task_index"] == len(source["tasks"])-1:
        cached = args.final_cache / ident["dataset"].lower().replace("-", "_") / f"{ident['method']}_seed{ident['seed']}" / "result.json"
        if cached.exists():
            final = shared_read(cached)
            b = final.get("binding", {})
            if (final.get("completed") and b.get("source_result") == label(path)
                    and b.get("checkpoint_sha256") == cp["checkpoint_sha256"]
                    and b.get("training_source_commit") == ident["git_commit"]
                    and final.get("verification", {}).get("all_requested_checks_passed")):
                cal = final["normal_holdout_calibration"]
                point = cal["operating_points"]["0.02"]
                if cal["protocol"] == CALIBRATION_PROTOCOL and point.get("status") == "available":
                    cm = point["test"]["confusion_matrix"]
                    calibration = {"operating_point": {k:v for k,v in point.items() if k != "test"},
                                   "pool": cal["pool"], "protocol": cal["protocol"]}
                    mode = "reused_valid_final_calibration_no_forward"
                    costs["existing_final_cache_cost_seconds_not_recharged"] = final.get("cost_seconds")
    if cm is None:
        data = prepare_data(ident, args.data_root, data_cache)
        normal = names.index("Normal")
        normal_rows = np.flatnonzero(data["y_train"] == normal)
        count = max(5, math.ceil(.1*len(normal_rows)))
        ids = np.random.default_rng(ident["seed"]*100003+normal*1009+17).permutation(normal_rows)[:count]
        if count != ident["section26_holdout"][str(normal)]["validation_count"]:
            raise ValueError("Permanent Normal holdout count does not match the recorded identity")
        state = torch.load(path.parent / cp["checkpoint_file"], map_location="cpu", weights_only=False)
        model = MLP(data["x_train"].shape[1], int(ident["effective_config"]["hidden"]), len(names))
        model.load_state_dict(state["model_state"]); model.eval()
        counts = np.bincount(data["y_train"], minlength=len(names))
        prior = torch.from_numpy(np.log(counts/counts.sum()).astype(np.float32)[seen])
        t = time.perf_counter()
        holdout_logits = logits(model, data["x_train"], ids, seen, args.batch_size)
        costs["new_normal_holdout_forward_seconds"] = time.perf_counter()-t
        t = time.perf_counter()
        grid, selected, queries, gap = select_taus(holdout_logits, prior, seen.index(normal))
        costs["new_normal_only_search_seconds"] = time.perf_counter()-t
        point = selected["0.02"]
        calibration = {"operating_point": point, "grid": grid, "search_evaluations": queries,
                       "protocol": CALIBRATION_PROTOCOL, "calibration_rows": len(ids),
                       "original_train_counts": counts.tolist(), "seen_class_columns": seen,
                       "minimum_normal_log_prior_gap": gap,
                       "helper_scope": "Existing helper also searches 1% and 5%; only 2% test CM is evaluated and reported here"}
        t = time.perf_counter()
        test_rows = np.flatnonzero(np.isin(data["y_test"], seen))
        values = logits(model, data["x_test"], test_rows, seen, args.batch_size)
        pred = np.asarray(seen)[(values + point["total_tau"]*prior).argmax(1).numpy()]
        n = len(names)
        cm = np.bincount(data["y_test"][test_rows]*n+pred, minlength=n*n).reshape(n,n)
        costs["new_test_forward_and_selected_evaluation_seconds"] = time.perf_counter()-t
        mode = "new_CPU_stage_calibration"
    metrics = stage_metrics(cm, seen, names)
    row = {"completed": True, "completed_at": now(), "task_index": cp["task_index"],
           "binding": binding, "mode": mode, "calibration": calibration, "metrics": metrics,
           "cost_seconds": costs | {"new_stage_elapsed": time.perf_counter()-started}}
    save_json(target, row)
    return row, mode


def process_cell(path, source, args, data_cache, analysis_commit):
    started = time.perf_counter()
    identity = source["identity"]
    dataset, method, seed = (identity[k] for k in ("dataset", "method", "seed"))
    folder = args.output_dir / dataset.lower().replace("-", "_") / f"{method}_seed{seed}"
    target = folder / "result.json"
    bindings = [stage_binding(path, source, cp) for cp in source["checkpoints"]]
    if len(bindings) != len(source["tasks"]):
        raise ValueError("Completed source has an incomplete saved checkpoint trajectory")
    if target.exists():
        old = shared_read(target)
        if old.get("completed") and old["stage_bindings"] == bindings and old["protocol"] == PROTOCOL:
            return old, True, CounterLike()
        raise ValueError("Existing cell cache differs; preserve it and select a new namespace")
    stages, modes = [], CounterLike()
    for cp in source["checkpoints"]:
        row, mode = get_stage(path, source, cp, folder / "stages" / f"task_{cp['task_index']:03d}.json", args, data_cache)
        stages.append(row); modes[mode] += 1
        print(f"STAGE {dataset} {method} seed={seed} task={cp['task_index']} {mode}", flush=True)
    configuration = {k:identity["effective_config"].get(k) for k in (
        "hidden", "batch_size", "epochs_per_task", "min_steps_per_task", "learning_rate",
        "buffer_per_class", "gate_step_policy", "calibration_policy", "normal_training_rows",
        "section29", "section29_schema_version", "section298", "section298_schema_version",
        "selection_only", "lambda_g", "derpp_alpha", "derpp_beta", "cbf_epochs", "cbf_lr_multiplier")}
    if method in SECTION31_METHODS:
        configuration.update({k: identity["effective_config"].get(k) for k in SECTION31_CONFIG_KEYS})
    row = {"completed": True, "completed_at": now(), "dataset": dataset, "method": method, "seed": seed,
           "group_role": group_role(method), "protocol": PROTOCOL, "source_result": label(path),
           "stage_bindings": bindings, "class_order": source["class_order"], "tasks": source["tasks"],
           "training_git_commit": identity["git_commit"], "training_configuration": configuration,
           "analysis": {"entrypoint": "ncd/analyze_stagewise_forgetting.py", "git_commit": analysis_commit,
                        "device": "cpu", "cpu_threads": 2, "batch_size": args.batch_size},
           "stages": stages, "metrics": aggregate_trajectory(stages, source["tasks"], source["class_order"]),
           "cost_seconds": {"new_cell_elapsed": time.perf_counter()-started}, "stage_modes": dict(modes),
           "error_component_interpretation": "Actual G rejection versus F wrong-family alert" if method in
               {"aacil", "ts_mem", "aa_noattmem", "aacil_der"} | SECTION29_METHODS | SECTION31_METHODS else
               "Normal versus other-attack confusion categories only; this single classifier has no separate G and F",
           "limitations": ["Post-hoc unified stagewise analysis; original Section27.8 decision unchanged.",
               "G misses and wrong-family alerts are observable CM error components; F behavior on gate-rejected rows is not inferred.",
               "Signed forgetting can be negative; no class or seed is dropped due to its sign.",
               "Saved stage/final calibration results retain their original evaluation hardware; newly needed stages use CPU float32.",
               "Joint uses offline historical training data and is a separate reference; FARQ is supplementary."]}
    if method in SECTION31_METHODS:
        row["architecture_capacity"] = source["architecture_capacity"]
        row["original_execution_cost_seconds"] = {k: v for k, v in source.items() if k.endswith("_seconds_total")}
        row["original_cost_scope"] = source["cost_scope"]
        row["stage_resource_accounting"] = [{
            "task_index": cp["task_index"], "family_metadata": cp["family_metadata"],
            "allocation": {k: v for k, v in cp["allocation"].items()
                           if k not in {"training_row_ids", "validation_row_ids"}},
            **{key: cp[key] for key in (
                "calibration_pool_storage", "calibration_pool_rows", "calibration_pool_budget",
                "calibration_additional_unique_rows_outside_M", "selection_attack_holdout_storage",
                "ephemeral_normal_training_feature_and_id_bytes", "gate_training_schedule",
                "family_training_schedule", "family_postprocess", "peak_cuda_allocated_bytes",
                "peak_cuda_reserved_bytes") if key in cp},
            "recorded_stage_seconds": {k: v for k, v in cp.items() if k.endswith("_seconds")},
        } for cp in source["checkpoints"]]
        row["resource_accounting_scope"] = "Copied from immutable source records; original successful execution costs are not added again to new analysis time. Training=G+F; base-family/CBF nested within F; peak CUDA is not host or exclusive GPU usage."
        row["limitations"].append("Compact candidate selection used UNSW/NSL validation seeds5/6; formal seeds1/2/3/4/42 are results on development datasets, not blind confirmation.")
    if source.get("section39_execution"):
        row["section39_execution"] = source["section39_execution"]
        row["publication_derivation"] = source["publication_derivation"]
        row["original_execution_cost_seconds"] = {k: v for k, v in source.items() if k.endswith("_seconds_total")}
        row["original_cost_scope"] = source["cost_scope"]
        row["limitations"].append("Section39 CIC17 is a fixed descriptive completion on a development dataset, not blind confirmation. The legacy loader validated shift arrays but did not evaluate them; CSE18 model metrics remain sealed.")
    save_json(target, row)
    return row, False, modes


class CounterLike(defaultdict):
    def __init__(self):
        super().__init__(int)


def mean_sd(values):
    values = list(values)
    defined = [v for v in values if v is not None]
    return {"n": len(values), "defined_n": len(defined),
            "mean": statistics.mean(defined) if defined else None,
            "sample_sd": statistics.stdev(defined) if len(defined)>1 else None}


def write_summary(args, attempt):
    path = args.output_dir / "result.json"
    previous = shared_read(path) if path.exists() else {}
    changed_groups = {("CIC-IDS-2017", m) for m in ("aacil_kd_wa", "replay_bs")} if args.section39_publication else None
    paths = ([args.output_dir / "cic_ids_2017" / f"{m}_seed{s}/result.json"
              for _, m in sorted(changed_groups) for s in SEEDS] if changed_groups
             else args.output_dir.glob("*/*/result.json"))
    rows = [shared_read(p) for p in paths]
    rows = [r for r in rows if r.get("completed") and r.get("protocol") == PROTOCOL]
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset"],row["method"]].append(row)
    metrics = ["average_incremental_accuracy", "final_balanced_accuracy", "final_macro_f1",
               "final_normal_fpr", "final_attack_detection", "forgetting_all_old_classes", "forgetting_old_attack_classes"]
    summaries = []
    for (dataset, method), cells in sorted(groups.items()):
        seeds = sorted(r["seed"] for r in cells)
        perclass = defaultdict(list)
        stage_groups = defaultdict(list)
        for r in cells:
            for c in r["metrics"]["per_old_class"]:
                perclass[c["class_id"]].append(c)
            for stage in r["stages"]:
                stage_groups[stage["task_index"]].append(stage["metrics"])
        stage_summaries = []
        for task_index, values in sorted(stage_groups.items()):
            stage_classes = defaultdict(list)
            for value in values:
                for c in value["per_class"]:
                    stage_classes[c["class_id"]].append(c)
            stage_summaries.append({"task_index": task_index, "n": len(values),
                "metrics": {k: mean_sd(v[k] for v in values) for k in
                    ("balanced_accuracy", "macro_f1", "normal_fpr", "attack_detection")},
                "per_class": [{"class_id": c, "class_name": points[0]["class_name"],
                    "support": points[0]["support"],
                    **{k: mean_sd(p[k] for p in points) for k in ("precision", "recall", "f1")}}
                    for c, points in sorted(stage_classes.items())]})
        summaries.append({"dataset": dataset, "method": method, "group_role": group_role(method),
            "seeds": seeds, "n": len(cells), "complete_five_seed": seeds == sorted(SEEDS),
            "metrics": {key: mean_sd(r["metrics"][key] for r in cells) for key in metrics},
            "stages": stage_summaries, "final_per_class": stage_summaries[-1]["per_class"],
            "old_class_forgetting": [{"class_id": c, "class_name": values[0]["class_name"],
                "forgetting": mean_sd(v["forgetting"] for v in values),
                "gate_normal_miss_increase": mean_sd(v["gate_normal_miss_increase"] for v in values if "gate_normal_miss_increase" in v),
                "other_attack_family_error_increase": mean_sd(v["other_attack_family_error_increase"] for v in values if "other_attack_family_error_increase" in v)}
                for c, values in sorted(perclass.items())]})
    cell_index = [{"dataset": r["dataset"], "method": r["method"], "seed": r["seed"],
                   "source_result": r["source_result"], "completed_at": r["completed_at"]} for r in rows]
    if changed_groups:
        # Append the new candidate and complete BS42 without recalculating any
        # other previously delivered group, stage, comparison or statistical test.
        summaries += [g for g in previous["groups"] if (g["dataset"], g["method"]) not in changed_groups]
        cell_index += [r for r in previous["cells"] if (r["dataset"], r["method"]) not in changed_groups]
        summaries.sort(key=lambda g: (g["dataset"], g["method"]))
    save_json(path, {"completed_at": now(), "protocol": PROTOCOL,
        "units": "Metrics are fractions; multiply by 100 for percentages or forgetting/error-change percentage points",
        "undefined_value_aggregation": "Section29.7: retain null per seed; aggregate only defined seeds and disclose defined_n alongside total n",
        "groups": summaries, "completed_logical_cells": len(cell_index),
        "execution_attempts": previous.get("execution_attempts", [])+[attempt],
        "cost_accounting": "Each execution attempt includes cache-reading/resume time. Nested stage/cell timings are not additive to attempt elapsed. Replay+LA is an alias and adds no forward cost.",
        "cells": cell_index})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["UNSW-NB15", "NSL-KDD", "CIC-IDS-2017"], default=["UNSW-NB15", "NSL-KDD", "CIC-IDS-2017"])
    parser.add_argument("--input", type=Path, action="append")
    parser.add_argument("--section31", action="store_true",
                        help="Append compact formal UNSW/NSL root; never development, queue, or CSE18 roots")
    parser.add_argument("--section39-publication", action="store_true",
                        help="Only six validated Section39 CIC publication cells; saved alpha2 only, update only the two affected summary groups")
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS),
                        help="Optional one-version processing filter; existing summary caches remain included")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--final-cache", type=Path, default=BASE/"section27/baseline_calibration_v2")
    parser.add_argument("--output-dir", type=Path, default=BASE/"section28/forgetting_v1")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--source-commit")
    parser.add_argument("--complete-retention-groups-only", action="store_true",
                        help="Process Section29.8 groups only after all five formal seeds are complete")
    parser.add_argument("--complete-compact-groups-only", action="store_true",
                        help="Process Section31 groups only after all five formal seeds are complete")
    args = parser.parse_args()
    if args.section39_publication:
        if args.input or args.section31 or args.methods or args.complete_compact_groups_only or args.complete_retention_groups_only:
            parser.error("Section39 has an exact six-cell assignment; do not combine other input/method modes")
        args.datasets = ["CIC-IDS-2017"]
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    args.output_dir = args.output_dir.resolve(); args.final_cache = args.final_cache.resolve()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    started, began = time.perf_counter(), now()
    analysis_commit = args.source_commit or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    selected = {}
    roots = ([BASE / "section28/publication_v1/cic_ids_2017"] if args.section39_publication
             else list(args.input or DEFAULT_ROOTS))
    if args.section31 and SECTION31_ROOT not in roots:
        roots.append(SECTION31_ROOT)
    for root in roots:
        paths = ([root / f"{m}_seed{s}/result.json" for m, s in
                  [("aacil_kd_wa", s) for s in SEEDS] + [("replay_bs", 42)]]
                 if args.section39_publication else sorted(root.resolve().rglob("result.json")))
        for path in paths:
            if root.resolve() == SECTION31_ROOT.resolve() and any("cse" in part.lower() for part in path.relative_to(root.resolve()).parts):
                continue  # CSE18 blindness/release is managed separately; never open its result files here.
            source = shared_read(path)
            identity = source.get("identity", {})
            config = identity.get("effective_config", {})
            if args.section39_publication:
                execution = source["section39_execution"]
                invocation = execution["invocation"]
                if (invocation["launcher_commit"] != "a4d55cd8e8d4087ea67f33a26535c721547483c3"
                        or invocation["launcher_sha256"] != "b0a76a0dbb8ef7d2acd488e584eedd4219990b917b10bf576e4bb82bbdedded5"
                        or invocation["dataset"] != "CIC-IDS-2017" or invocation["smoke"]
                        or execution["cell_attempt"]["exit_code"] != 0
                        or invocation["selection_changed"] or invocation["CSE18_model_metrics_accessed"]
                        or (identity["method"], identity["seed"]) != (invocation["method"], invocation["seed"])
                        or any("0.02" not in cp.get("operating_points", {}) for cp in source["checkpoints"])):
                    raise ValueError("Invalid Section39 publication provenance or saved stage evaluation")
            if (not source.get("completed") or not source.get("checkpoints") or
                    identity.get("dataset") not in args.datasets or identity.get("method") not in METHODS or
                    identity.get("seed") not in SEEDS or
                    source.get("smoke") or config.get("smoke") or
                    source.get("selection_only") or config.get("selection_only") or
                    (args.methods and identity.get("method") not in args.methods)):
                continue
            if identity["method"] in SECTION31_METHODS and (
                    identity["dataset"] not in ({"CIC-IDS-2017"} if args.section39_publication else {"UNSW-NB15", "NSL-KDD"}) or
                    config.get("section31") is not True or config.get("section31_schema_version") != 1 or
                    not source.get("architecture_capacity")):
                continue
            if identity["method"] in SECTION29_METHODS | SECTION31_METHODS and (
                    source.get("test_evaluation_performed") is not True or
                    len(source["checkpoints"]) != len(source.get("tasks", [])) or
                    any(cp.get("test_evaluation_performed") is not True or
                        "0.02" not in cp.get("operating_points", {}) for cp in source["checkpoints"])):
                continue
            if root.resolve() in {(BASE / "section29/retention_local_v1").resolve(),
                                  (BASE / "section29/section299_cic_kd_local_v1").resolve(),
                                  (BASE / "section29/section2911_cic_kd_resume_v1").resolve()} and (
                    config.get("section298") is not True or config.get("section298_schema_version") != 1):
                continue
            key = tuple(identity[k] for k in ("dataset", "method", "seed"))
            if identity["method"] in SECTION29_METHODS and key in selected:
                previous_config = selected[key][2]["identity"]["effective_config"]
                if any(config.get(k) != previous_config.get(k) for k in SECTION29_CONFIG_KEYS):
                    raise ValueError("Conflicting Section29 protocols/configurations for one formal cell; use explicit inputs and a separate output namespace without changing old caches")
            if identity["method"] in SECTION31_METHODS and key in selected:
                previous_identity = selected[key][2]["identity"]
                previous_config = previous_identity["effective_config"]
                if (any(config.get(k) != previous_config.get(k) for k in SECTION31_CONFIG_KEYS)
                        or identity.get("source_hashes") != previous_identity.get("source_hashes")):
                    raise ValueError("Conflicting compact source/configurations for one formal cell; use separate namespaces and preserve old caches")
            stamp = datetime.fromisoformat(source["completed_at"].replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                raise ValueError("Source completion timestamp has no timezone")
            order = stamp, label(path)
            if key not in selected or order < selected[key][0]:
                selected[key] = (order, path, source)
    if args.complete_retention_groups_only:
        retention_keys = {key for key, value in selected.items()
                          if value[2]["identity"]["effective_config"].get("section298") is True}
        retention_seeds = {}
        for dataset, method, seed in retention_keys:
            retention_seeds.setdefault((dataset, method), set()).add(seed)
        selected = {key: value for key, value in selected.items()
                    if key not in retention_keys or retention_seeds[key[:2]] == set(SEEDS)}
    if args.complete_compact_groups_only:
        compact_keys = {key for key, value in selected.items()
                        if value[2]["identity"]["effective_config"].get("section31") is True}
        compact_seeds = {}
        for dataset, method, seed in compact_keys:
            compact_seeds.setdefault((dataset, method), set()).add(seed)
        selected = {key: value for key, value in selected.items()
                    if key not in compact_keys or compact_seeds[key[:2]] == set(SEEDS)}
    new, reused, modes, cache = 0, 0, CounterLike(), {}
    if args.section39_publication and set(selected) != {
            ("CIC-IDS-2017", m, s) for m, s in
            [("aacil_kd_wa", s) for s in SEEDS] + [("replay_bs", 42)]}:
        raise ValueError("Section39 requires all six complete publication cells")
    order = {dataset:i for i,dataset in enumerate(args.datasets)}
    for key, (_, path, source) in sorted(selected.items(), key=lambda pair:(order[pair[0][0]],pair[0][1],SEEDS.index(pair[0][2]))):
        row, was_reused, cell_modes = process_cell(path, source, args, cache, analysis_commit)
        new += int(not was_reused); reused += int(was_reused)
        for mode, value in cell_modes.items():modes[mode] += value
        if key[1] == "replay":
            alias = row | {"method": "replay_la", "logical_alias_of": "replay",
                           "alias_rule": "Same weights and total stage tau; initial LA tau=1 plus delta; no extra optimization or inference",
                           "cost_seconds": {"new_cell_elapsed": 0., "physical_cost_counted_under": "replay"}}
            save_json(args.output_dir/key[0].lower().replace("-", "_")/f"replay_la_seed{key[2]}"/"result.json", alias)
        print(f"CELL_COMPLETE {key} {'cache_reused' if was_reused else 'new'}", flush=True)
    attempt = {"started_at": began, "completed_at": now(), "elapsed_seconds": time.perf_counter()-started,
               "datasets": args.datasets, "analysis_git_commit": analysis_commit,
               "new_physical_cells": new, "reused_physical_cells": reused, "stage_modes": dict(modes),
               "structured_command": {"entrypoint": "ncd/analyze_stagewise_forgetting.py", "datasets": args.datasets,
                                      "device": "cpu", "cpu_threads": 2, "batch_size": args.batch_size,
                                      "complete_retention_groups_only": args.complete_retention_groups_only,
                                      "complete_compact_groups_only": args.complete_compact_groups_only,
                                      "include_section31": args.section31, "methods": args.methods,
                                      "section39_publication": args.section39_publication}}
    write_summary(args, attempt)
    print(json.dumps({"status":"complete", **attempt}), flush=True)


if __name__ == "__main__":
    main()
