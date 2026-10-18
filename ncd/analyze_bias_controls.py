"""Section23 controls: portable, seed-paired analysis of existing small results.

No training or candidate tuning. The frozen selection is independently checked
against development seeds 5/6; formal seeds only describe correction efficacy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ncd.analyze_bias_gap import EXPECTED_BEST, canonical, load_arm, ratio, require
from ncd.analyze_step_budget import DATASETS, METHODS, SEEDS, OPERATIONAL_CONFIG, close, paired_delta, recompute, summarize, validate_schedule

CORRECTIONS = ("wa", "bic", "ssil", "cbf")
DEVELOPMENT_SEEDS = (5, 6)
ROLES = {"development": "training_holdout_not_official_ID_test", "formal": "official_ID_test_not_used_for_selection"}
TARGETS = ("UNSW-NB15", "CIC-IDS-2017")
ID_BINDINGS = ("x_train", "y_train", "x_test", "y_test")
SOURCE_PATHS = {"ncd/run_bias_correction.py", "ncd/section23_bias.py", "ncd/run_bias_control.py",
                "ncd/run_problem_scout.py", "ncd/crr_benchmarks.py", "ncd/model.py",
                "ncd/prepare_problem_scout_data.py", "ncd/train_cil.py", "ncd/data.py",
                "ncd/source_commit.py", "ncd/run_crr_real_sequence.py"}
COST_TIMERS = ("training_seconds", "correction_seconds", "buffer_selection_seconds", "evaluation_seconds")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    finite(value)
    return value


def finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), "Nonfinite numeric value")
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def hashed(value, length=64):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{" + str(length) + r"}", value) is not None


def contained(root, relative):
    require(isinstance(relative, str) and not re.match(r"^[A-Za-z]:|^[\\/]", relative), "Absolute provenance path is not portable")
    path = (root / relative).resolve()
    require(root.resolve() in path.parents, "Path escapes source root")
    return path


def groups():
    return [(d, b, c) for d in DATASETS for b in ("replay", EXPECTED_BEST[d]) for c in CORRECTIONS]


def base_role(dataset, base):
    return "replay" if base == "replay" else "s2000_best"


def validate_spec(spec):
    require(spec["datasets"] == list(DATASETS) and spec["best_methods"] == EXPECTED_BEST, "Wrong dataset/base grid")
    require(spec["development_seeds"] == [5, 6] and spec["reporting_seeds"] == list(SEEDS), "Seed roles differ from Section23")
    require((spec["epochs_per_task"], spec["min_steps_per_task"], spec["batch_size"]) == (8, 2000, 256), "Wrong Section23 training budget")
    require({k: len(v) for k, v in spec["candidates"].items()} == {"wa": 1, "bic": 2, "ssil": 2, "cbf": 2}, "Candidate grid differs")
    require(spec["bic_training_memory_per_class"] + spec["bic_holdout_per_class_max"] <= 200, "BiC exceeds memory budget")


def checkpoint_metrics(cp, classes):
    """Recompute all seen-class quantities from complete-axis integer counts."""
    seen, new = cp["seen_classes"], cp["new_classes"]
    result = recompute(cp["id"], classes, seen)
    cm = cp["id"]["confusion_matrix"]
    saved = {r["class_id"]: r for r in cp["id"]["per_class"]}
    per_class = {}
    for c in seen:
        support, predicted, tp = sum(cm[c]), sum(r[c] for r in cm), cm[c][c]
        precision = ratio(tp, predicted)
        f1 = ratio(2 * tp, support + predicted)
        require(close(precision, saved[c]["precision"]), "Saved precision disagrees with confusion counts")
        per_class[c] = {"support": support, "predicted": predicted, "tp": tp,
                        "fp": predicted - tp, "fn": support - tp, "precision": precision,
                        "recall": result["recall"][c], "f1": f1,
                        "macro_f1_term": 0.0 if f1 is None else f1}
    normal = classes.index("Normal")
    normal_fpr = ratio(sum(cm[normal]) - cm[normal][normal], sum(cm[normal]))
    require(close(normal_fpr, cp["id"]["normal_fpr"]), "Saved Normal FPR disagrees with confusion counts")
    aggregates = {k: result[k] for k in ("macro_f1", "balanced_accuracy")}
    aggregates["normal_fpr"] = normal_fpr
    for role, ids in (("new", new), ("old", [c for c in seen if c not in new])):
        values = [per_class[c]["recall"] for c in ids]
        # No old class in the first task is undefined, not zero retention.
        aggregates[role + "_macro_recall"] = statistics.mean(values) if values and all(v is not None for v in values) else None
    return {"aggregate": aggregates, "per_class": per_class}


def posthoc_steps(cp, correction):
    # The closed-form WA record has no optimizer counter after the first task.
    steps = cp["correction"].get("optimizer_steps", 0 if correction == "wa" else None)
    require(type(steps) is int and steps >= 0, "Invalid post-hoc optimizer step count")
    require(correction not in ("wa", "ssil") or steps == 0, "Unexpected post-hoc optimizer steps for WA/SS-IL")
    return steps


def validate_cell(path, row, bound, spec, s2000, weight_mode):
    cell = read(path)
    identity, expected = cell["identity"], row["identity"]
    require(cell.get("completed") is True, "Indexed cell is incomplete")
    for key, value in expected.items():
        require(identity.get("base_method" if key == "base" else key) == value, "Index/child identity mismatch")
    d, base, correction, seed, phase = (identity[k] for k in ("dataset", "base_method", "correction", "seed", "phase"))
    require(identity.get("smoke") is False and identity["effective_config"].get("smoke") is False, "Smoke is not formal/development evidence")
    require((d, base, correction) in groups(), "Unexpected control group")
    require(type(seed) is int and phase in ROLES and seed in (DEVELOPMENT_SEEDS if phase == "development" else SEEDS), "Wrong seed/phase")
    require(identity["selection_seeds"] == [5, 6] and identity["evaluation_role"] == ROLES[phase], "Wrong selection/evaluation role")
    candidate = identity["candidate"]
    require(type(candidate) is int and 0 <= candidate < len(spec["candidates"][correction]), "Unknown candidate")
    require(identity["hyperparameters"] == spec["candidates"][correction][candidate], "Candidate parameters differ from frozen grid")
    for key in ("git_commit", "config_sha256", "source_hashes", "config_input_hashes"):
        require(identity[key] == bound[key], "Child/controller provenance differs")
    classes, tasks = cell["class_order"], cell["tasks"]
    require(identity["class_order"] == classes and identity["tasks"] == tasks, "Child axes differ from identity")
    reference = s2000[d, base, SEEDS[0]]["cell"]
    require(classes == reference["class_order"] and tasks == reference["tasks"], "Control/reference class/task axes differ")
    config = identity["effective_config"]
    require(set(identity["data_binding"]) == set(ID_BINDINGS), "Control result must bind exactly the ID training/test arrays")
    bindings = identity["data_binding"]
    for name in ID_BINDINGS:
        array = bindings[name]
        require(hashed(array["sha256"]) and isinstance(array["shape"], list) and all(type(n) is int and n > 0 for n in array["shape"]), "Invalid bound array hash/shape")
        require(array["dtype"] == reference["identity"]["data_binding"][name]["dtype"], "Bound array dtype differs")
    require(bindings["x_train"]["shape"][0] == bindings["y_train"]["shape"][0] == sum(cell["train_counts"].values()), "Training array row counts disagree")
    require(bindings["x_test"]["shape"][0] == bindings["y_test"]["shape"][0], "Test feature/label rows disagree")
    require(bindings["x_train"]["shape"][1:] == bindings["x_test"]["shape"][1:] == reference["identity"]["data_binding"]["x_train"]["shape"][1:], "Feature dimensions differ")
    allowed = OPERATIONAL_CONFIG | {"ssil_old_replay_batch_size", "buffer_per_class"}
    require({k: v for k, v in config.items() if k not in allowed} == {k: v for k, v in reference["identity"]["effective_config"].items() if k not in allowed}, "Scientific base config changed")
    require(config["ssil_old_replay_batch_size"] == spec["ssil_old_replay_batch_size"], "Extra SSIL replay configuration differs")
    expected_memory = spec["bic_training_memory_per_class"] if correction == "bic" else 200
    require(config["buffer_per_class"] == expected_memory, "Training memory configuration differs")
    require(identity["regularization"] == reference["identity"]["regularization"], "Historical base regularization changed")
    split = identity["development_split"]
    require(split["role"] == ROLES[phase], "Data role mismatch")
    if phase == "formal":
        require(identity["data_binding"] == {k: reference["identity"]["data_binding"][k] for k in ID_BINDINGS}, "Formal ID data binding differs from S2000")
        require(cell["train_counts"] == reference["train_counts"], "Formal train counts differ")
    else:
        require(split["seed"] == seed and split["disjoint_indices"] is True and split["fraction"] == spec["development_validation_fraction"], "Development holdout contract differs")
        require(split["training_rows"] + split["validation_rows"] == reference["identity"]["data_binding"]["x_train"]["shape"][0], "Development split does not partition original training pool")
        require(split["training_rows"] == sum(cell["train_counts"].values()), "Development training counts differ")
        require(split["validation_rows"] == bindings["y_test"]["shape"][0], "Development validation rows differ")
        require(hashed(split["train_indices_sha256"]) and hashed(split["validation_indices_sha256"]), "Missing development index provenance")
        require(identity.get("frozen_selection") is None, "Development consumed formal selection")
    cps = cell["checkpoints"]
    require(len(cps) == len(tasks) == row["checkpoint_count"], "Complete child checkpoint coverage differs")
    require([cp["task_index"] for cp in cps] == list(range(len(tasks))), "Checkpoint order differs")
    validate_schedule(cell, "2000")
    checked_weights, computed = 0, []
    for t, cp in enumerate(cps):
        require(cp["identity_hash"] == canonical(identity), "Checkpoint identity hash differs")
        require(cp["correction_complete"] is True and cp["evaluation_role"] == ROLES[phase], "Checkpoint correction/evaluation incomplete")
        require(cp["new_classes"] == tasks[t] and cp["seen_classes"] == sorted(c for task in tasks[:t + 1] for c in task), "Checkpoint task axes differ")
        weight = contained(path.parent, cp["checkpoint_file"])
        require(weight.suffix == ".pt" and hashed(cp["checkpoint_sha256"]), "Missing checkpoint provenance")
        require(read(weight.with_suffix(".json")) == cp, "Metrics sidecar differs from result")
        if weight_mode == "verify":
            require(weight.is_file() and digest(weight) == cp["checkpoint_sha256"], "Checkpoint payload missing or changed")
            checked_weights += 1
        for key in COST_TIMERS:
            require(type(cp[key]) in (int, float) and cp[key] >= 0, "Invalid measured cost")
        posthoc_steps(cp, correction)
        for c in cp["seen_classes"]:
            memory = cp["training_memory_counts"].get(str(c), 0)
            held = cp["calibration_holdout_counts"].get(str(c), 0)
            require(type(memory) is int and type(held) is int and 0 <= memory <= expected_memory and 0 <= held <= spec["bic_holdout_per_class_max"] and memory + held <= 200, "Recorded memory/holdout budget exceeded")
            require(correction == "bic" or held == 0, "Unexpected calibration holdout")
        metrics = checkpoint_metrics(cp, classes)
        if phase == "formal":
            ref = reference["checkpoints"][t]["id"]["confusion_matrix"]
            require([sum(r) for r in cp["id"]["confusion_matrix"]] == [sum(r) for r in ref], "Formal test support differs from reference")
        computed.append(metrics)
    require(sum(sum(r) for r in cps[-1]["id"]["confusion_matrix"]) == identity["data_binding"]["y_test"]["shape"][0], "Final confusion count differs from bound test array length")
    require(close(computed[-1]["aggregate"]["macro_f1"], row["macro_f1"]), "Index final metric differs")
    return {"cell": cell, "computed": computed, "result_sha256": digest(path), "result": row["result"], "weights_verified": checked_weights}


def validate_selection(selection, selection_sha, bound, spec, development):
    require(selection["provenance"] == bound and selection["selection_seeds"] == [5, 6], "Frozen selection provenance/seeds differ")
    require(selection["selection_metric"] == "final training-held-out validation Macro-F1" and selection["tie_break"] == "lower candidate index", "Selection metric/tie break differs")
    rows = selection["selections"]
    keys = [(r["dataset"], r["base"], r["correction"]) for r in rows]
    require(len(keys) == len(set(keys)) and set(keys) == set(groups()), "Frozen selection lacks exactly 24 groups")
    selected = {}
    for key, row in zip(keys, rows):
        candidates = row["candidates"]
        require([c["candidate"] for c in candidates] == list(range(len(spec["candidates"][key[2]]))), "Selection candidate coverage differs")
        for candidate in candidates:
            n = candidate["candidate"]
            require(candidate["parameters"] == spec["candidates"][key[2]][n], "Selected parameter definition differs")
            proofs = candidate["development_results"]
            require([p["seed"] for p in proofs] == [5, 6], "Selection evidence includes reporting seeds or lacks development seed")
            scores = []
            for proof in proofs:
                record = development.get((*key, n, proof["seed"]))
                require(record is not None, "Frozen selection lacks development source")
                require(proof["result"] == record["result"] and proof["result_sha256"] == record["result_sha256"], "Frozen development result/hash differs")
                expected_identity = {"dataset": key[0], "base": key[1], "correction": key[2], "candidate": n, "seed": proof["seed"], "phase": "development", "smoke": False}
                require(proof["identity"] == expected_identity and proof["checkpoint_count"] == len(record["computed"]), "Development proof identity differs")
                score = record["computed"][-1]["aggregate"]["macro_f1"]
                require(close(proof["macro_f1"], score), "Development proof metric differs")
                scores.append(score)
            require(close(candidate["mean_validation_macro_f1"], statistics.mean(scores)), "Development selection mean differs")
        winner = max(candidates, key=lambda c: (c["mean_validation_macro_f1"], -c["candidate"]))
        require(row["selected_candidate"] == winner["candidate"], "Candidate is not the development-only winner")
        selected[key] = winner["candidate"]
    return selected


def summarize_costs(records, events=None):
    """Sum completed-cell counters once; subprocess time is separate elapsed time."""
    starts, exits = {}, {}
    for event in events or []:
        if event["event"] not in ("child_start", "child_exit"):
            continue
        identity = event["identity"]
        key = (canonical(identity), event["attempt"])
        require(type(event["attempt"]) is int and event["attempt"] > 0, "Invalid cost attempt number")
        target = starts if event["event"] == "child_start" else exits
        require(key not in target, "Duplicate cost attempt event")
        target[key] = event
        if target is exits:
            require(type(event["process_seconds"]) in (int, float) and math.isfinite(event["process_seconds"]) and event["process_seconds"] >= 0, "Invalid subprocess elapsed cost")
            require(type(event["exit_code"]) is int, "Invalid subprocess exit code")
    cells, indexed = [], set()
    for record in records:
        cell = record["cell"]
        identity = cell["identity"]
        requested = {k: identity["base_method" if k == "base" else k]
                     for k in ("dataset", "base", "correction", "candidate", "seed", "phase", "smoke")}
        identity_hash = canonical(requested)
        indexed.add(identity_hash)
        started = {k for k in starts if k[0] == identity_hash}
        ended = {k for k in exits if k[0] == identity_hash}
        complete = bool(ended) and started == ended and any(exits[k]["exit_code"] == 0 for k in ended)
        cells.append({"phase": identity["phase"], "dataset": identity["dataset"],
            "base_method": identity["base_method"], "base_role": base_role(identity["dataset"], identity["base_method"]),
            "correction": identity["correction"], "cells": 1, "checkpoints": len(cell["checkpoints"]),
            **{timer: math.fsum(cp[timer] for cp in cell["checkpoints"]) for timer in COST_TIMERS},
            "base_optimizer_steps": sum(cp["training_schedule"]["optimizer_steps"] for cp in cell["checkpoints"]),
            "posthoc_optimizer_steps": sum(posthoc_steps(cp, identity["correction"]) for cp in cell["checkpoints"]),
            "exited_attempts": len(ended), "failed_attempts": sum(exits[k]["exit_code"] != 0 for k in ended),
            "cells_with_complete_process_timing": int(complete),
            "recorded_subprocess_seconds": math.fsum(exits[k]["process_seconds"] for k in sorted(ended))})
    labels = ("phase", "dataset", "base_method", "base_role", "correction")
    numeric = ("cells", "checkpoints", *COST_TIMERS, "base_optimizer_steps", "posthoc_optimizer_steps",
               "exited_attempts", "failed_attempts", "cells_with_complete_process_timing", "recorded_subprocess_seconds")
    rows = []
    for level, keys in (("overall", ()), ("phase", ("phase",)), ("configuration", labels)):
        group_keys = sorted({tuple(cell[k] for k in keys) for cell in cells})
        for values in group_keys:
            subset = [cell for cell in cells if tuple(cell[k] for k in keys) == values]
            row = {"aggregation_level": level, **{k: dict(zip(keys, values)).get(k) for k in labels}}
            row.update({k: math.fsum(cell[k] for cell in subset) if k.endswith("seconds") else sum(cell[k] for cell in subset) for k in numeric})
            row["process_timing_complete"] = row["cells_with_complete_process_timing"] == row["cells"]
            row["subprocess_seconds"] = row["recorded_subprocess_seconds"] if row["process_timing_complete"] else None
            rows.append(row)
    return {"scope": "Indexed completed cells only. Development includes all candidates; formal includes the frozen selected candidate. Unindexed attempts are excluded and counted separately.",
        "definitions": {
            "subprocess_seconds": "Sum of child_exit process_seconds, including failed exited attempts for indexed cells. This is elapsed subprocess time, not GPU utilization-hours or controller elapsed time. Null unless every indexed cell has matched start/exit events and a successful exit; recorded_subprocess_seconds retains the observed exited-attempt sum.",
            "components": "Sum each checkpoint's non-cumulative training, post-hoc correction, buffer selection and evaluation timer once. Components are not added to subprocess elapsed time; startup, data loading, saving and other overhead are not included in these component timers.",
            "optimizer_steps": "Base steps include SS-IL's modified training workload. Post-hoc steps are separate BiC/CBF optimization; closed-form WA (whose records omit this counter) and SS-IL have zero post-hoc optimizer steps. SS-IL extra replay/distillation time is included in training_seconds, not isolated as correction_seconds.",
            "aggregation": "Rows are overlapping views: overall, phase, and phase/dataset/base/correction. Sum within one view only; never sum rows across aggregation levels."},
        "attempt_log_available": events is not None,
        "unindexed_exited_attempts": sum(k[0] not in indexed for k in exits), "rows": rows}


def load_controls(root, spec_path, s2000, weight_mode="portable"):
    require(weight_mode in ("portable", "verify"), "Unknown checkpoint verification mode")
    root, spec_path = root.resolve(), spec_path.resolve()
    spec = read(spec_path)
    validate_spec(spec)
    expected = {(*group, s) for group in groups() for s in SEEDS}
    evidence = {"weight_mode": weight_mode, "checkpoint_payload_status": "not_read_recorded_hashes_only" if weight_mode == "portable" else "actual_payload_sha256_checked",
                "expected_formal_cells": 120, "expected_development_cells": 84, "selection_verified": False,
                "formal_cells": 0, "development_cells": 0, "weights_verified": 0, "input_files": []}
    if not (root / "result.json").exists():
        evidence.update(controller_status="not_available", missing_formal_cells=sorted(expected))
        return {}, evidence, spec
    # The controller is live: bind the exact atomic snapshot bytes read, not a
    # later version of result.json after another cell completes.
    index_bytes = (root / "result.json").read_bytes()
    index = json.loads(index_bytes.decode("utf-8-sig"))
    finite(index)
    index_sha = hashlib.sha256(index_bytes).hexdigest()
    require(index.get("smoke") is False, "Smoke controller cannot be analyzed as formal evidence")
    require(index["expected_cells"] == {"development": 84, "formal": 120}, "Controller expected grid differs")
    bound = index["provenance"]
    require(hashed(bound["git_commit"], 40) and bound["config_sha256"] == digest(spec_path), "Source/config provenance is incomplete or changed")
    require(set(bound["source_hashes"]) == SOURCE_PATHS and all(hashed(h) for h in bound["source_hashes"].values()), "Invalid or incomplete source hashes")
    require(bound["config_input_hashes"] and all(hashed(h) for h in bound["config_input_hashes"].values()), "Invalid base/lambda input hashes")
    first = s2000[DATASETS[0], "replay", SEEDS[0]]["cell"]["identity"]
    expected_inputs = {spec["base_config"], *first["effective_config"].get("lambda_files", {}).values()}
    require(set(bound["config_input_hashes"]) == expected_inputs, "Incomplete base/lambda input provenance")
    development, formal, seen = {}, {}, set()
    for row in index["cells"]:
        identity = row["identity"]
        key = tuple(identity[k] for k in ("dataset", "base", "correction", "candidate", "seed", "phase"))
        require(key not in seen, "Duplicate indexed cell")
        seen.add(key)
        path = contained(root, row["result"])
        require(path.name == "result.json" and digest(path) == row["result_sha256"], "Indexed result hash mismatch")
        record = validate_cell(path, row, bound, spec, s2000, weight_mode)
        group = key[:3]
        if key[-1] == "development":
            development[(*group, key[3], key[4])] = record
        else:
            require((*group, key[4]) not in formal, "Multiple candidates for a formal seed")
            formal[(*group, key[4])] = record
        evidence["weights_verified"] += record["weights_verified"]
        evidence["input_files"].append({"path": row["result"], "sha256": row["result_sha256"], "phase": key[-1], "checkpoints": len(record["computed"])})
    actual_counts = {"development": len(development), "formal": len(formal)}
    require(index["completed_cells"] == actual_counts, "Controller completed counts differ from indexed cells")
    selection_path = root / "selection.json"
    if index.get("selection_sha256"):
        require(selection_path.exists(), "Indexed frozen selection is missing")
        selection_sha = digest(selection_path)
        require(index.get("selection_sha256") == selection_sha, "Controller selection hash differs")
        selected = validate_selection(read(selection_path), selection_sha, bound, spec, development)
        require(len(development) == 84, "Frozen selection is missing candidate coverage")
        for (d, base, correction, seed), record in formal.items():
            identity = record["cell"]["identity"]
            chosen = selected[d, base, correction]
            proof = identity["frozen_selection"]
            require(identity["candidate"] == chosen and proof["selected_candidate"] == chosen and proof["sha256"] == selection_sha and proof["selection_seeds"] == [5, 6], "Formal cell does not use frozen selection")
        evidence.update(selection_verified=True, selection_sha256=selection_sha, selection_groups=len(selected))
    else:
        require(not formal, "Formal cells exist without frozen development selection")
    require(set(formal).issubset(expected), "Unexpected formal cell")
    if index["status"] == "complete":
        require(len(formal) == 120 and len(development) == 84 and evidence["selection_verified"], "Complete controller has missing evidence")
    evidence.update(controller_status=index["status"], controller_sha256=index_sha, provenance=bound,
                    formal_cells=len(formal), development_cells=len(development), missing_formal_cells=sorted(expected - set(formal)))
    attempt_path = root / "driver_attempts.jsonl"
    events = None
    if attempt_path.is_file():
        attempt_bytes = attempt_path.read_bytes()
        events = [json.loads(line) for line in attempt_bytes.decode("utf-8-sig").splitlines() if line.strip()]
        finite(events)
        evidence["cost_attempt_log"] = {"path": "driver_attempts.jsonl", "sha256": hashlib.sha256(attempt_bytes).hexdigest()}
    evidence["costs"] = summarize_costs([*development.values(), *formal.values()], events)
    return formal, evidence, spec


def closure(corrected, best, joint):
    denominator = paired_delta(joint, best)
    gain = paired_delta(corrected, best)
    complete = summarize(corrected)["complete"] and summarize(best)["complete"] and summarize(joint)["complete"]
    ratio_means = gain["mean"] / denominator["mean"] if complete and denominator["mean"] > 0 else None
    paired_ratios = {s: (corrected[s] - best[s]) / (joint[s] - best[s]) if corrected.get(s) is not None and joint.get(s) is not None and best.get(s) is not None and joint[s] > best[s] else None for s in SEEDS}
    return {"complete": complete, "denominator_gap_pp": denominator, "gain_over_fixed_best_pp": gain,
            "remaining_gap_pp": paired_delta(joint, corrected), "ratio_of_five_seed_means": ratio_means,
            "paired_seed_ratios": summarize(paired_ratios), "ratio_meaningful": complete and denominator["mean"] > 0}


def decide(configurations, complete):
    if not complete:
        return {"decision": "pending", "reason": "All 120 formal cells and complete development-only frozen selection are required"}
    rows = {(r["dataset"], r["base_role"], r["correction"]): r for r in configurations}
    require(len(rows) == 24, "Decision requires all 24 corrected configurations")
    def value(d, b, c):
        return rows[d, b, c]["closure"]["ratio_of_five_seed_means"]
    if any(value(d, b, c) is None for d in TARGETS for b in ("replay", "s2000_best") for c in CORRECTIONS):
        return {"decision": "pending", "reason": "Target residual gap or complete closure ratio is undefined"}
    strict = [{"base_role": b, "correction": c} for b in ("replay", "s2000_best") for c in CORRECTIONS if all(value(d, b, c) >= .7 or close(value(d, b, c), .7) for d in TARGETS)]
    cross = [c for c in CORRECTIONS if all(max(value(d, b, c) for b in ("replay", "s2000_best")) >= .7 or close(max(value(d, b, c) for b in ("replay", "s2000_best")), .7) for d in TARGETS)]
    best = {d: max(value(d, b, c) for b in ("replay", "s2000_best") for c in CORRECTIONS) for d in TARGETS}
    result = {"strict_same_correction_same_base_role_70pct": strict, "cross_base_same_correction_70pct_sensitivity": cross, "best_closure_by_dataset": best,
              "confirmed_rule": "User-confirmed solved condition requires the same correction and base role across UNSW/CIC17; dataset-specific fixed-best base methods may differ. Cross-base success alone does not satisfy this criterion."}
    if strict:
        result.update(decision="solved", reason="At least one same correction/base-role configuration closes >=70% on both target datasets")
    elif cross:
        result.update(decision="intermediate", reason="Cross-base sensitivity alone does not meet the confirmed same-base-role criterion; intermediate for owner next-step decision")
    elif all(v < .3 and not close(v, .3) for v in best.values()):
        result.update(decision="unsolved", reason="Even each target dataset's best fixed corrected configuration closes <30%")
    else:
        result.update(decision="intermediate", reason="Neither predeclared threshold is jointly satisfied; owner decision required")
    return result


def analyze(joint, s2000, formal, coverage):
    # Revalidate reference checkpoints once, not repeatedly per correction.
    references = {}
    for dataset in DATASETS:
        means = {method: statistics.mean(s2000[dataset, method, s]["final"]["macro_f1"] for s in SEEDS) for method in METHODS}
        require(min(METHODS, key=lambda m: (-means[m], m)) == EXPECTED_BEST[dataset], "S2000 fixed best differs from declared winner")
        reference_cell = s2000[dataset, EXPECTED_BEST[dataset], SEEDS[0]]["cell"]
        for method in ("joint", "replay", EXPECTED_BEST[dataset]):
            arm = joint if method == "joint" else s2000
            for seed in SEEDS:
                record = arm[dataset, method, seed]
                cell = record["cell"]
                require(all(cell[k] == reference_cell[k] for k in ("class_order", "tasks", "train_counts")), "Reference class/task/train counts differ")
                require({k: cell["identity"]["data_binding"][k] for k in ID_BINDINGS} == {k: reference_cell["identity"]["data_binding"][k] for k in ID_BINDINGS}, "Joint/S2000 ID data bindings differ")
                require([sum(row) for row in cell["checkpoints"][-1]["id"]["confusion_matrix"]] == [sum(row) for row in reference_cell["checkpoints"][-1]["id"]["confusion_matrix"]], "Reference test supports differ")
                references[dataset, method, seed] = [checkpoint_metrics(cp, record["cell"]["class_order"]) for cp in record["cell"]["checkpoints"]]
    complete = len(formal) == 120 and coverage["selection_verified"] and coverage["development_cells"] == 84 and coverage["controller_status"] == "complete"
    out = {"schema_version": 1, "execution_status": "complete" if complete else "partial", "scientific_comparisons_final": complete,
           "coverage": {k: v for k, v in coverage.items() if k != "costs"},
           "costs": coverage.get("costs", summarize_costs([])), "definitions": {
               "levels": "Precision, recall, F1 and aggregate metrics are fractions; paired metric differences are percentage points. Count differences use examples.",
               "aggregation": "Exact paired seeds 1,2,3,4,42; arithmetic mean and sample SD (ddof=1). Missing/undefined quantities remain null; n and complete are explicit.",
               "recall_roles": "Per checkpoint, new classes are that checkpoint's introduced classes; old classes are previously introduced classes. New/old macro recall is the unweighted mean of all role classes, null if empty or unsupported.",
               "empty_class_f1": "Undefined class F1 is null; the separate zero macro_f1_term preserves the original macro convention for fully empty classes.",
               "selection": "Hyperparameters frozen from training-held-out development seeds 5/6 only. Reporting best corrected configuration is one complete five-seed mean, never per-seed maxima, and is descriptive rather than a new hyperparameter selection.",
               "closure": "(corrected mean Macro-F1 - unchanged S2000 fixed-best mean)/(original Joint mean - unchanged S2000 fixed-best mean). Same-base improvement is separately reported and is NOT the closure numerator. No clipping to [0,1].",
               "closure_gate": "Thresholds use ratio of five-seed means, not mean seed ratios. Nonpositive dataset denominator (including NSL) makes closure undefined. Strict main rule uses same correction plus same base role across UNSW/CIC; cross-base alternatives are sensitivity only.",
               "portable": "Portable mode verifies index/result/metrics-sidecar/frozen-selection hashes and recorded identities. Checkpoint hashes retain original weight provenance but no actual .pt bytes are read; this is not fresh payload verification.",
               "research_scope": "Bounded Section23/RQ2 comparison of adapted borrowed methods, not proof of novelty or overall research completion. Original eight-epoch Joint is unchanged; runtime/hardware are not matched."},
           "checkpoint_metrics": [], "checkpoint_classes": [], "configurations": [], "best_corrected": []}
    for d, base, correction in groups():
        original = s2000[d, base, SEEDS[0]]["cell"]
        classes, tasks = original["class_order"], original["tasks"]
        records = {s: formal[d, base, correction, s] for s in SEEDS if (d, base, correction, s) in formal}
        candidates = {r["cell"]["identity"]["candidate"] for r in records.values()}
        require(len(candidates) <= 1, "Multiple reporting candidates in one group")
        label = {"dataset": d, "base_method": base, "base_role": base_role(d, base), "correction": correction, "selected_candidate": next(iter(candidates), None)}
        for t, new in enumerate(tasks):
            seen = sorted(c for group in tasks[:t + 1] for c in group)
            for metric in ("macro_f1", "balanced_accuracy", "normal_fpr", "new_macro_recall", "old_macro_recall"):
                corrected = {s: r["computed"][t]["aggregate"][metric] for s, r in records.items()}
                old = {s: references[d, base, s][t]["aggregate"][metric] for s in SEEDS}
                j = {s: references[d, "joint", s][t]["aggregate"][metric] for s in SEEDS}
                out["checkpoint_metrics"].append(label | {"task_index": t, "is_final": t == len(tasks) - 1, "metric": metric,
                    "corrected": summarize(corrected), "same_base_s2000": summarize(old), "joint": summarize(j),
                    "corrected_minus_same_base_pp": paired_delta(corrected, old), "corrected_minus_joint_pp": paired_delta(corrected, j)})
            for c in seen:
                for metric in ("precision", "recall", "f1", "support", "predicted", "tp", "fp", "fn"):
                    corrected = {s: r["computed"][t]["per_class"][c][metric] for s, r in records.items()}
                    old = {s: references[d, base, s][t]["per_class"][c][metric] for s in SEEDS}
                    j = {s: references[d, "joint", s][t]["per_class"][c][metric] for s in SEEDS}
                    scale = 100 if metric in ("precision", "recall", "f1") else 1
                    out["checkpoint_classes"].append(label | {"task_index": t, "is_final": t == len(tasks) - 1, "class_id": c, "class_name": classes[c], "role": "new" if c in new else "old", "metric": metric,
                        "difference_units": "pp" if scale == 100 else "examples", "corrected": summarize(corrected), "same_base_s2000": summarize(old), "joint": summarize(j),
                        "corrected_minus_same_base": paired_delta(corrected, old, scale), "corrected_minus_joint": paired_delta(corrected, j, scale)})
        corrected = {s: r["computed"][-1]["aggregate"]["macro_f1"] for s, r in records.items()}
        best = {s: references[d, EXPECTED_BEST[d], s][-1]["aggregate"]["macro_f1"] for s in SEEDS}
        j = {s: references[d, "joint", s][-1]["aggregate"]["macro_f1"] for s in SEEDS}
        old = {s: references[d, base, s][-1]["aggregate"]["macro_f1"] for s in SEEDS}
        out["configurations"].append(label | {"final_macro_f1": summarize(corrected), "same_base_improvement_pp": paired_delta(corrected, old), "closure": closure(corrected, best, j)})
    for d in DATASETS:
        rows = [r for r in out["configurations"] if r["dataset"] == d]
        all_complete = all(r["final_macro_f1"]["complete"] for r in rows)
        winner = min(rows, key=lambda r: (-r["final_macro_f1"]["mean"], r["correction"], r["base_role"])) if all_complete else None
        out["best_corrected"].append({"dataset": d, "selection_complete": all_complete,
            "configuration": {k: winner[k] for k in ("base_method", "base_role", "correction", "selected_candidate")} if winner else None,
            "final_macro_f1": winner["final_macro_f1"] if winner else None, "closure": winner["closure"] if winner else None})
    out["section23_4"] = decide(out["configurations"], complete)
    return out


def export(out, directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "analysis.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for table in ("checkpoint_metrics", "checkpoint_classes"):
        summary, per_seed = [], []
        for record in out[table]:
            fixed = {k: v for k, v in record.items() if not isinstance(v, dict)}
            summaries = {k: v for k, v in record.items() if isinstance(v, dict)}
            summary.append(fixed | {k + "_" + name: v[name] for k, v in summaries.items() for name in ("n", "complete", "mean", "sample_sd")})
            for seed in SEEDS:
                per_seed.append(fixed | {"seed": seed} | {k: v["by_seed"][str(seed)] for k, v in summaries.items()})
        for suffix, rows in (("summary", summary), ("by_seed", per_seed)):
            with (directory / (table + "_" + suffix + ".csv")).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
    rows = []
    for r in out["configurations"]:
        rows.append({k: r[k] for k in ("dataset", "base_method", "base_role", "correction", "selected_candidate")} |
            {"n": r["final_macro_f1"]["n"], "macro_f1_mean": r["final_macro_f1"]["mean"], "macro_f1_sample_sd": r["final_macro_f1"]["sample_sd"],
             "same_base_improvement_pp": r["same_base_improvement_pp"]["mean"], "closure": r["closure"]["ratio_of_five_seed_means"],
             "fixed_best_gap_pp": r["closure"]["denominator_gap_pp"]["mean"], "gain_over_fixed_best_pp": r["closure"]["gain_over_fixed_best_pp"]["mean"]})
    with (directory / "configurations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    cost_rows = out["costs"]["rows"]
    with (directory / "costs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cost_rows[0]) if cost_rows else ["aggregation_level", "cells", "subprocess_seconds"])
        writer.writeheader(); writer.writerows(cost_rows)
    message = {"execution_status": out["execution_status"], "formal_cells": out["coverage"]["formal_cells"], "development_cells": out["coverage"]["development_cells"], "section23_4": out["section23_4"]}
    (directory / "analysis.log").write_text(json.dumps(message, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(message, allow_nan=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--steps-2000", type=Path, required=True)
    p.add_argument("--controls", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("ncd/configs/bias_correction_section23.json"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--weights", choices=("portable", "verify"), default="portable")
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    roots = [path.resolve() for path in (args.baseline, args.steps_2000, args.controls)]
    output = args.output_dir.resolve()
    require(all(repo in path.parents for path in [*roots, output, args.config.resolve()]), "Paths must be repository-relative for portable provenance")
    require(all(output != path and path not in output.parents and output not in path.parents for path in roots), "Output overlaps an input root")
    joint, jproof = load_arm(roots[0], baseline=True)
    s2000, sproof = load_arm(roots[1])
    formal, coverage, spec = load_controls(roots[2], args.config, s2000, args.weights)
    out = analyze(joint, s2000, formal, coverage)
    out["provenance"] = {"created_utc": datetime.now(timezone.utc).isoformat(), "analysis_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "analysis_source_sha256": digest(Path(__file__)), "reference_inputs": {"joint": jproof, "s2000": sproof},
        "input_roots": [path.relative_to(repo).as_posix() for path in roots], "config": args.config.resolve().relative_to(repo).as_posix(), "config_sha256": digest(args.config),
        "command": "python -I ncd/analyze_bias_controls.py --baseline " + roots[0].relative_to(repo).as_posix() + " --steps-2000 " + roots[1].relative_to(repo).as_posix() + " --controls " + roots[2].relative_to(repo).as_posix() + " --config " + args.config.resolve().relative_to(repo).as_posix() + " --output-dir " + output.relative_to(repo).as_posix() + " --weights " + args.weights,
        "source_validation_scope": "Recorded frozen training source hashes agree across controller, selection and cells. Historical training source/weight/data payload bytes are not re-read in portable mode; actual small result/sidecar/selection bytes and confusion-derived values are checked."}
    export(out, output)


if __name__ == "__main__":
    main()
