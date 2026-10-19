"""Describe fixed trajectories and discarded one-Adam-step interference probes.

No training, inference, test feedback, hyperparameter selection or significance
test. Aggregate histograms do not identify individual prediction migrations.
Local row traces are inspected but never copied to the publishable JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"NSL-KDD": ("nsl_kdd", "U2R", "R2L", "R2L", 52),
            "CIC-IDS-2017": ("cic_ids_2017", "Infiltration", "WebAttack", "DoS", 30)}
ARMS = ("random", "mir", "mir_rare_ce", "mir_rare_margin")
PROBE_ARMS = ("mir", "mir_rare_ce")
VARIANTS = ("full", "rare_only", "nonrare_only", "zero_gradient", "without_competitor")
EVENT_STEPS = list(range(0, 2000, 50))
CONTRASTS = {
    "full_minus_rare_only_add_nonrare": {"full": 1, "rare_only": -1},
    "full_minus_nonrare_only_add_rare": {"full": 1, "nonrare_only": -1},
    "nonrare_only_minus_zero_gradient": {"nonrare_only": 1, "zero_gradient": -1},
    "rare_only_minus_zero_gradient": {"rare_only": 1, "zero_gradient": -1},
    "full_minus_zero_gradient": {"full": 1, "zero_gradient": -1},
    "fourway_interaction": {"full": 1, "rare_only": -1, "nonrare_only": -1, "zero_gradient": 1},
    "full_minus_without_competitor": {"full": 1, "without_competitor": -1},
    **{f"{v}_minus_pre_no_update": {v: 1, "pre": -1} for v in VARIANTS},
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON constant {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str(value.dtype).encode())
    h.update(np.asarray(value.shape, dtype="<i8").tobytes())
    h.update(value.tobytes())
    return h.hexdigest()


def relative(path, base):
    path = Path(path)
    require(not path.is_absolute(), "Artifact reference must be relative")
    out = (base / path).resolve()
    require(out.is_relative_to(base.resolve()), "Artifact reference escapes root")
    return out


def label(path):
    path = path.resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name


def stats(values):
    values = [float(v) for v in values]
    require(all(math.isfinite(v) for v in values), "Nonfinite summary")
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def counts(value, classes, rows=None):
    require(isinstance(value, list) and len(value) == classes
            and all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in value),
            "Prediction counts must be nonnegative integers in class order")
    require(rows is None or sum(value) == rows, "Prediction histogram row count changed")
    return value


def curve_summary(steps, histograms, class_id, names):
    require(len(steps) == len(histograms) and steps == sorted(set(steps)), "Invalid trace steps")
    rows = sum(histograms[0])
    for hist in histograms:
        counts(hist, len(names), rows)
    correct = [h[class_id] for h in histograms]
    def first(predicate):
        return next((s for s, n in zip(steps, correct) if predicate(n)), None)
    declines = []
    for i in range(1, len(steps)):
        if correct[i] >= correct[i - 1]:
            continue
        recovery = next((j for j in range(i + 1, len(steps)) if correct[j] >= correct[i - 1]), None)
        delta = [b - a for a, b in zip(histograms[i - 1], histograms[i])]
        declines.append({"from_sampled_step": steps[i - 1], "to_sampled_step": steps[i],
                         "correct_before": correct[i - 1], "correct_after": correct[i],
                         "net_prediction_count_change": delta,
                         "net_gaining_competitors": {names[c]: d for c, d in enumerate(delta) if c != class_id and d > 0},
                         "recovery_to_predecline_count_step": steps[recovery] if recovery is not None else None,
                         "still_below_predecline_at_next_sample": correct[i + 1] < correct[i - 1] if i + 1 < len(steps) else None,
                         "unrecovered_through_final_sample": recovery is None})
    return {"rows": rows, "steps": steps, "correct_counts": correct,
            "prediction_counts_by_step": histograms,
            "initial_prediction_counts": histograms[0], "final_prediction_counts": histograms[-1],
            "first_positive_family_recall_step": first(lambda x: x > 0),
            "first_at_least_95_percent_family_recall_step": first(lambda x: x / rows >= .95),
            "first_all_correct_step": first(lambda x: x == rows),
            "never_correct_at_any_sample": not any(correct), "max_correct_count": max(correct),
            "net_correct_change": correct[-1] - correct[0],
            "net_prediction_count_change": [b - a for a, b in zip(histograms[0], histograms[-1])],
            "first_decline": declines[0] if declines else None,
            "first_unrecovered_decline": next((d for d in declines if d["unrecovered_through_final_sample"]), None),
            "declines": declines,
            "interpretation": "Counts are aggregate net changes, not identified row migrations; sampling cannot locate exact updates or exclude between-sample changes."}


def checkpoint_check(base, record, mode):
    require(isinstance(record.get("sha256"), str) and len(record["sha256"]) == 64, "Missing checkpoint SHA")
    path = relative(record["path"], base)
    if mode == "verify":
        require(path.is_file() and sha(path) == record["sha256"], f"Checkpoint payload SHA mismatch: {label(path)}")
    return {"path": label(path), "sha256": record["sha256"], "actual_payload_verified": mode == "verify"}


def pilot_analysis(root, mode):
    curves, sources, weights = [], [], []
    for dataset, (slug, rare, old, _, rare_rows) in DATASETS.items():
        for arm in ARMS:
            for seed in (1, 2):
                path = root / slug / f"{arm}_seed{seed}" / "result.json"
                cell = read(path)
                ident, names = cell["identity"], cell["class_order"]
                require(cell["completed"] and not ident["smoke"] and ident["steps"] == 2000,
                        "Pilot must be completed formal-shape exploratory data")
                require((ident["dataset"], ident["arm"], ident["seed"]) == (dataset, arm, seed), "Pilot cell identity differs")
                require(cell["memory_unchanged"] and cell["old_pool_revisits"] == 0, "Pilot access boundary differs")
                traces = cell["audit"]["traces"]
                require([t["step"] for t in traces] == list(range(0, 2001, 50)), "Pilot trace sampling incomplete")
                for name, role, n in ((rare, "current_rare_train", rare_rows), (old, "resident_memory", 200)):
                    c = names.index(name)
                    rows = [[r for r in t["per_class"] if r["class_id"] == c and r["role"] == role] for t in traces]
                    require(all(len(r) == 1 and r[0]["rows"] == n for r in rows), "Pilot fixed class probe differs")
                    curve = curve_summary([t["step"] for t in traces], [r[0]["prediction_counts"] for r in rows], c, names)
                    curves.append({"dataset": dataset, "arm": arm, "seed": seed, "class_name": name,
                                   "class_id": c, "class_order": names, "role": role, **curve})
                weights.append(checkpoint_check(path.parent, cell["final_checkpoint"], mode))
                sources.append({"result": label(path), "sha256": sha(path), "source_commit": ident["git_commit"]})
    return {"complete": True, "curves": curves, "sources": sources, "checkpoint_verification": weights}


def task_end_analysis(root):
    path = root / "result.json"
    if not path.is_file():
        return {"complete": False, "reason": "Existing task-end diagnosis absent"}
    data = read(path)
    require(data["complete"] and data["completed_cells"] == data["expected_cells"] == 16,
            "Task-end diagnosis is incomplete")
    out = []
    for cell in data["cells"]:
        child_path = relative(cell["result"], root)
        require(sha(child_path) == cell["result_sha256"], "Diagnosis child result hash changed")
        child = read(child_path)
        names = child["class_order"]
        require(cell["full_seen_confusion_exact_matches"] == 4, "Diagnosis full CM reproduction failed")
        for row in cell["post_introduction_transitions"]:
            if row["class_name"] not in ("R2L", "WebAttack"):
                continue
            c = row["class_id"]
            for update in row["updates"]:
                mat = update["prediction_transition_matrix"]
                require(len(mat) == len(names), "Task-end transition shape mismatch")
                for r in mat:
                    counts(r, len(names))
                destinations = {names[k]: mat[c][k] for k in range(len(names)) if k != c}
                require(sum(destinations.values()) == update["correct_family_to_wrong"], "Task-end transition arithmetic mismatch")
                out.append({"dataset": cell["dataset"], "method": cell["method"], "seed": cell["seed"],
                            "class_name": row["class_name"], "from_task": update["from_task_index"],
                            "to_task": update["to_task_index"], "previously_correct_to_wrong": destinations,
                            "originally_correct_rows": sum(mat[c]), "retained_correct_rows": mat[c][c]})
    grouped = []
    for key in sorted({(r["dataset"], r["method"], r["class_name"]) for r in out}):
        rows = [r for r in out if (r["dataset"], r["method"], r["class_name"]) == key]
        grouped.append({"dataset": key[0], "method": key[1], "class_name": key[2],
                        "seeds": [r["seed"] for r in rows],
                        "destination_count_across_seed_summary": {k: stats(r["previously_correct_to_wrong"][k] for r in rows)
                                                                  for k in rows[0]["previously_correct_to_wrong"]}})
    return {"complete": True, "source": label(path), "sha256": sha(path), "rows": out, "groups": grouped,
            "scope": "Existing test diagnostic only, never intervention selection or tuning; task-end individual transitions, no within-task timing."}


def row_onsets(steps, predictions, true_class, names):
    """Per-row first events, summarized without publishing row-level arrays."""
    require(predictions.ndim == 2 and predictions.shape[0] == len(steps), "Row trace shape differs")
    require(predictions.dtype.kind in "iu" and np.all((predictions >= 0) & (predictions < len(names))), "Invalid row predictions")
    first_learn, first_losses, unrecovered, recovered, never, net = [], [], 0, 0, 0, np.zeros(len(names), dtype=int)
    first_loss_step_destinations = {}
    all_transitions = np.zeros((len(names), len(names)), dtype=int)
    loss_destinations = np.zeros(len(names), dtype=int)
    for col in predictions.T:
        hits = np.flatnonzero(col == true_class)
        if not len(hits):
            never += 1
        else:
            first_learn.append(int(steps[hits[0]]))
        losses = np.flatnonzero((col[:-1] == true_class) & (col[1:] != true_class)) + 1
        if len(losses):
            i = int(losses[0])
            first_losses.append(int(steps[i]))
            loss_destinations[col[i]] += 1
            key = str(int(steps[i]))
            first_loss_step_destinations.setdefault(key, [0] * len(names))[int(col[i])] += 1
            if np.any(col[i + 1:] == true_class):
                recovered += 1
            else:
                unrecovered += 1
        np.add.at(all_transitions, (col[:-1], col[1:]), 1)
        net[col[-1]] += 1
    def histogram(values):
        return {str(s): values.count(s) for s in sorted(set(values))}
    correct = predictions == true_class
    return {"rows": predictions.shape[1], "never_learned_rows": never,
            "steps_including_zero": len(steps), "ever_correct_rows": int(correct.any(axis=0).sum()),
            "never_correct_rows": never, "final_correct_rows": int(correct[-1].sum()),
            "ever_correct_but_final_wrong_rows": int((correct.any(axis=0) & ~correct[-1]).sum()),
            "correct_to_wrong_events": int((correct[:-1] & ~correct[1:]).sum()),
            "wrong_to_correct_events": int((~correct[:-1] & correct[1:]).sum()),
            "prediction_change_events": int((predictions[1:] != predictions[:-1]).sum()),
            "first_correct_step_counts": histogram(first_learn), "first_correct_to_wrong_step_counts": histogram(first_losses),
            "first_correct_to_wrong_step_destination_counts": first_loss_step_destinations,
            "first_loss_destination_counts": dict(zip(names, map(int, loss_destinations))),
            "first_loss_recovered_later_rows": recovered, "first_loss_unrecovered_through_final_rows": unrecovered,
            "all_adjacent_prediction_transition_counts": all_transitions.tolist(), "final_prediction_counts": net.tolist(),
            "limit": "Rows and steps are dependent observations, not independent replicates; a first-loss recovery does not preclude later losses."}


def event_contrasts(events, probe_roles, metric_names):
    output = {}
    for role in probe_roles:
        output[role] = {}
        for name, coefficients in CONTRASTS.items():
            metrics = {}
            for metric in metric_names:
                values = [sum(coef * event["variants"][variant][role][metric] for variant, coef in coefficients.items()) for event in events]
                metrics[metric] = {"all_prespecified_events": stats(values),
                                   "positive_events": sum(v > 0 for v in values), "negative_events": sum(v < 0 for v in values),
                                   "fixed_rare_presence_sensitivity": {
                                       "rare_present": stats(v for e, v in zip(events, values) if e["rare_rows"] > 0),
                                       "rare_absent": stats(v for e, v in zip(events, values) if e["rare_rows"] == 0)},
                                   "exploratory_frozen_windows": [{"steps": [lo, lo + 499],
                                       **stats(v for e, v in zip(events, values) if lo <= e["step"] < lo + 500)} for lo in range(0, 2000, 500)]}
            output[role][name] = metrics
    return output


def probe_analysis(root, pilot_root, mode):
    # Missing cells are pending, not failures. Existing incomplete cells are not
    # included in the complete-group causal summaries.
    cells, missing = [], []
    for dataset, (slug, rare, old, competitor, _) in DATASETS.items():
        for arm in PROBE_ARMS:
            for seed in (1, 2):
                path = root / slug / f"{arm}_seed{seed}" / "result.json"
                if not path.is_file():
                    missing.append(f"{slug}/{arm}_seed{seed}")
                    continue
                cell = read(path)
                if not cell.get("completed"):
                    missing.append(f"{slug}/{arm}_seed{seed}")
                    continue
                cells.append(analyze_probe_cell(cell, path, pilot_root, mode, dataset, arm, seed))
    groups = []
    for dataset in DATASETS:
        for arm in PROBE_ARMS:
            rows = [c for c in cells if c["dataset"] == dataset and c["arm"] == arm]
            if {r["seed"] for r in rows} != {1, 2}:
                continue
            group = {"dataset": dataset, "arm": arm, "seeds": [1, 2], "contrasts": {}}
            for role, contrasts in rows[0]["contrasts"].items():
                group["contrasts"][role] = {}
                for contrast, metrics in contrasts.items():
                    group["contrasts"][role][contrast] = {metric: stats(r["contrasts"][role][contrast][metric]["all_prespecified_events"]["mean"] for r in rows) for metric in metrics}
            groups.append(group)
    return {"complete": not missing, "expected_cells": 8, "completed_cells": len(cells), "missing_cells": missing,
            "cells": cells, "complete_two_seed_groups": groups,
            "unit": "One seed-path mean over all 40 prespecified events; across-seed n=2 descriptive mean/sample SD, not 80 independent updates."}


def analyze_probe_cell(cell, path, pilot_root, mode, dataset, arm, seed):
    """Runner schema adapter; strict validation is deliberately centralized."""
    ident, names = cell["identity"], cell["class_order"]
    slug, rare, old, competitor, rare_rows = DATASETS[dataset]
    require((ident["dataset"], ident["arm"], ident["seed"]) == (dataset, arm, seed), "Probe identity differs")
    require(not ident["smoke"] and ident["steps"] == 2000 and ident["pre_step_grid"] == EVENT_STEPS,
            "Scientific probe requires all 2000 real steps and 40 fixed events")
    require(ident["conditions"] == list(VARIANTS), "Clone conditions differ")
    require(ident["competitor_class"] == names.index(competitor)
            and ident["old_probe_class"] == names.index(old)
            and ident["rare_classes"] == [names.index(rare)], "Fixed family definitions differ")
    require(cell["memory_unchanged"] and cell["old_pool_revisits"] == 0, "Probe access boundary differs")
    reference_path = pilot_root / slug / f"{arm}_seed{seed}" / "result.json"
    reference = read(reference_path)
    ref = ident["reference"]
    require(relative(ref["result_path"], ROOT) == reference_path.resolve()
            and sha(reference_path) == ref["result_sha256"], "Original pilot reference changed")
    require(reference["completed"] and not reference["identity"]["smoke"], "Incomplete original reference")
    for key in ("class_order", "tasks", "data_binding", "start_checkpoint", "allowed_pool_x_sha256",
                "allowed_pool_y_sha256", "current_original_row_indices_sha256", "config_sha256"):
        require(ident[key] == reference["identity"][key], f"Probe reference binding differs: {key}")
    require(cell["id"]["confusion_matrix"] == reference["id"]["confusion_matrix"], "Final full CM differs from original path")
    require(cell["training_schedule"] == reference["training_schedule"], "Real schedule differs from original path")
    for key in ("chains", "class_exposures", "current_rows", "replay_rows", "rare_rows", "rare_batches", "step"):
        require(cell["audit"][key] == reference["audit"][key], f"Original real-path audit differs: {key}")
    exact = cell["reference_exact_reproduction"]
    for key in ("required", "model_tensors_equal", "final_confusion_equal", "all_six_audit_chains_equal",
                "class_exposures_equal", "training_schedule_equal", "current_rows_equal", "replay_rows_equal",
                "optimizer_full_state_equal", "resident_memory_equal"):
        require(exact.get(key) is True, f"Original exact endpoint requirement failed: {key}")
    require(exact["model_tensor_sha256"] == exact["reference_model_tensor_sha256"] == ref["model_tensor_sha256"],
            "Recorded endpoint tensor hash differs")
    checkpoint = checkpoint_check(path.parent, cell["final_checkpoint"], mode)
    original_checkpoint = checkpoint_check(reference_path.parent, reference["final_checkpoint"], mode)
    require(relative(ref["checkpoint_path"], ROOT) == relative(reference["final_checkpoint"]["path"], reference_path.parent)
            and ref["checkpoint_sha256"] == reference["final_checkpoint"]["sha256"], "Reference payload binding differs")
    if mode == "verify":
        verify_endpoints(path, cell, reference_path, reference)
    events = cell["interference_audit"]["events"]
    require([e["pre_step"] for e in events] == EVENT_STEPS, "Missing, duplicate or selected clone events")
    require(cell["interference_audit"]["full_clone_actual_update_exact_matches"] == 40, "Full sham clone does not reproduce real updates")
    require(cell["interference_audit"]["full_clone_actual_optimizer_exact_matches"] == 40, "Full sham clone does not reproduce Adam states")
    require(cell["local_only_prestates"] == [e["local_only_prestate"] for e in events], "Prestate inventory differs from events")
    roles = {"current_rare_train": (names.index(rare), rare_rows), "old_rare_resident": (names.index(old), 200)}
    metric_names = ("mean_ce", "recall", "true_minus_max_other_mean", "true_minus_max_other_p05")
    normalized, clone_counts, prestates = [], {v: 0 for v in VARIANTS}, []
    effect_map = {"full_minus_rare_ce": "full_minus_rare_only_add_nonrare", "full_minus_nonrare_ce": "full_minus_nonrare_only_add_rare",
                  "nonrare_minus_zero_ce": "nonrare_only_minus_zero_gradient", "rare_minus_zero_ce": "rare_only_minus_zero_gradient",
                  "interaction_ce": "fourway_interaction", "full_minus_without_competitor_ce": "full_minus_without_competitor"}
    for event in events:
        require(event["update_step"] == event["pre_step"] + 1 and set(event["conditions"]) == set(VARIANTS), "Invalid clone event")
        require(isinstance(event["rare_rows"], int) and 0 <= event["rare_rows"] <= event["batch_rows"]
                and isinstance(event["competitor_rows"], int) and 0 <= event["competitor_rows"] <= event["batch_rows"],
                "Invalid component row counts")
        variants = {"pre": event["pre_probes"]}
        require(event["local_only_prestate"]["publish"] is False and event["local_only_prestate"]["pre_step"] == event["pre_step"],
                "Prestate must be correctly bound and LOCAL ONLY")
        prestates.append(checkpoint_check(path.parent, event["local_only_prestate"], mode))
        if mode == "verify":
            import torch
            state = torch.load(relative(event["local_only_prestate"]["path"], path.parent), map_location="cpu", weights_only=True)
            require(state["identity"] == ident and state["pre_step"] == event["pre_step"], "Actual prestate identity/step differs")
            ids = state["selected_pool_indices"].numpy()
            require(ids.ndim == 1 and ids.dtype.kind in "iu" and len(ids) == event["batch_rows"]
                    and np.all(ids >= 0) and array_sha(ids) == event["local_only_prestate"]["selected_pool_indices_sha256"]
                    == event["selected_indices_sha256"], "Actual prestate selected minibatch differs")
            require(state["model_state"] and state["optimizer_state"], "Prestate lacks model/Adam state")
        for variant in VARIANTS:
            condition = event["conditions"][variant]
            require(condition["virtual_optimizer_steps"] == 1 and condition["probe_forward_rows"] == rare_rows + 200,
                    "Clone step/probe budget differs")
            clone_counts[variant] += condition["virtual_optimizer_steps"]
            variants[variant] = condition["probes"]
            require(set(variants[variant]) == set(roles), "Clone probe populations differ")
            for role, (class_id, n) in roles.items():
                p = variants[variant][role]
                require(p["rows"] == n, "Clone probe rows differ")
                hist = counts(p["prediction_counts"], len(names), n)
                rivals = counts(p["max_other_class_counts"], len(names), n)
                require(rivals[class_id] == 0, "Max-other competitor includes true class")
                require(abs(p["recall"] - hist[class_id] / n) < 1e-7, "Clone recall differs from integer counts")
                require(all(isinstance(p[m], (int, float)) and math.isfinite(p[m]) for m in metric_names), "Nonfinite clone metric")
                require(p["mean_ce"] >= 0, "Negative clone CE")
        require(set(variants["pre"]) == set(roles), "No-update probe populations differ")
        for role, (class_id, n) in roles.items():
            p = variants["pre"][role]
            require(p["rows"] == n and abs(p["recall"] - counts(p["prediction_counts"], len(names), n)[class_id] / n) < 1e-7,
                    "No-update probe counts/recall differ")
            require(all(isinstance(p[m], (int, float)) and math.isfinite(p[m]) for m in metric_names), "Nonfinite no-update probe")
        for role in roles:
            for field, contrast in effect_map.items():
                value = sum(coef * variants[v][role]["mean_ce"] for v, coef in CONTRASTS[contrast].items())
                require(math.isclose(event["effects"][role][field], value, rel_tol=1e-10, abs_tol=1e-12), "Recorded clone contrast differs")
            for variant in VARIANTS:
                value = variants[variant][role]["mean_ce"] - variants["pre"][role]["mean_ce"]
                require(math.isclose(event["effects"][role]["condition_minus_pre_ce"][variant], value, rel_tol=1e-10, abs_tol=1e-12), "Recorded clone-pre effect differs")
        normalized.append({"step": event["pre_step"], "rare_rows": event["rare_rows"], "variants": variants})
    require(cell["interference_audit"]["costs"]["clone_optimizer_steps"] == 200,
            "Five clones times 40 events optimizer count differs")
    traces = new_row_traces(cell, path, mode, roles, names)
    return {"dataset": dataset, "arm": arm, "seed": seed, "source_commit": ident["git_commit"],
            "source": label(path), "sha256": sha(path), "class_order": names, "fixed_competitor": competitor,
            "reference_source_commit": reference["identity"]["git_commit"], "reference_result_sha256": sha(reference_path),
            "reference_exact_reproduction": exact, "endpoint_independently_payload_verified": mode == "verify",
            "checkpoints": [checkpoint, original_checkpoint], "local_only_prestate_checkpoints": prestates,
            "event_steps": EVENT_STEPS, "intervened_updates": [s + 1 for s in EVENT_STEPS],
            "empty_component_events": {"rare_rows_zero": sum(e["rare_rows"] == 0 for e in events),
                                       "competitor_rows_zero": sum(e["competitor_rows"] == 0 for e in events),
                                       "primary_mean_denominator": 40,
                                       "limit": "No zero-row events are excluded. Empty rare-only uses explicit zero gradients with inherited Adam moments."},
            "gradient_diagnostics": gradient_summaries(events),
            "clone_optimizer_steps": clone_counts, "contrasts": event_contrasts(normalized, roles, metric_names),
            "metric_units": {"mean_ce": "natural log cross entropy", "recall": "fraction, positive contrast is better",
                             "true_minus_max_other_mean": "logit units, positive contrast is better",
                             "true_minus_max_other_p05": "logit units; difference of per-condition fifth percentiles, not paired per-row percentile"},
            "contrast_interpretation": {"current_rare_primary": "full_minus_rare_only_add_nonrare",
                                        "old_rare_primary": "full_minus_nonrare_only_add_rare",
                                        "CE_sign": "Positive is increased mean CE (harm); negative is decreased CE.",
                                        "interaction": "Nonadditivity of Adam/probe response; not an additive causal attribution."},
            "training_row_trajectories": traces,
            "costs": {"original_real_path": cell["audit"]["costs"], "added_diagnostics": cell["interference_audit"]["costs"],
                      "evaluation_seconds": cell["evaluation_seconds"],
                      "note": "Instrumented real elapsed contains component/clone work; do not sum that inclusive field again with diagnostic components."}}


def gradient_summaries(events):
    result = {}
    for block in ("features", "head"):
        rows = [e["gradient_blocks"][block] for e in events]
        fields = ("rare_l2", "nonrare_l2", "competitor_l2", "full_l2", "rare_dot_nonrare",
                  "decomposition_residual_l2", "decomposition_residual_max_abs")
        out = {f: stats(r[f] for r in rows) for f in fields}
        defined = [r["rare_nonrare_cosine"] for r in rows if r["rare_nonrare_cosine"] is not None]
        out["cosine_when_both_gradients_nonzero"] = stats(defined)
        out["undefined_zero_norm_cosine_events"] = len(rows) - len(defined)
        for r in rows:
            require((r["rare_nonrare_cosine"] is None) == (r["rare_l2"] * r["nonrare_l2"] == 0), "Undefined gradient cosine differs from zero-norm status")
        out["clone_displacement_l2"] = {v: stats(e["conditions"][v]["displacement"][block]["l2"] for e in events) for v in VARIANTS}
        result[block] = out
    result["limit"] = "Gradient cosine is descriptive, not an Adam or long-run causal guarantee; undefined zero-norm cosines remain null, not zero direction."
    return result


def tensor_hash(state):
    h = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(np.asarray(value.shape, dtype="<i8").tobytes())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def verify_endpoints(path, cell, reference_path, reference):
    import torch
    a = relative(cell["final_checkpoint"]["path"], path.parent)
    b = relative(reference["final_checkpoint"]["path"], reference_path.parent)
    require(read(a.with_suffix(".json")) == cell and read(b.with_suffix(".json")) == reference, "Endpoint sidecar differs")
    actual = torch.load(a, map_location="cpu", weights_only=True)
    original = torch.load(b, map_location="cpu", weights_only=True)
    require(actual["identity"] == cell["identity"] and original["identity"] == reference["identity"], "Payload identity differs")
    require(actual["task_index"] == original["task_index"] == 3, "Wrong endpoint task")
    for state, result in ((actual, cell), (original, reference)):
        require(state["book"]["chains"] == result["audit"]["chains"], "Payload audit chains differ")
    require(set(actual["model_state"]) == set(original["model_state"])
            and all(torch.equal(v, original["model_state"][k]) for k, v in actual["model_state"].items()), "Actual endpoint tensors differ")
    require(tensor_hash(actual["model_state"]) == cell["reference_exact_reproduction"]["model_tensor_sha256"], "Actual tensor hash differs")
    def same(a, b):
        if isinstance(a, torch.Tensor):
            return isinstance(b, torch.Tensor) and torch.equal(a, b)
        if isinstance(a, dict):
            return isinstance(b, dict) and a.keys() == b.keys() and all(same(v, b[k]) for k, v in a.items())
        if isinstance(a, (list, tuple)):
            return isinstance(b, type(a)) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        return a == b
    require(same(actual["optimizer_state"], original["optimizer_state"]), "Actual endpoint Adam states differ")
    require(same(actual["buffer_x"], original["buffer_x"]) and same(actual["buffer_y"], original["buffer_y"]), "Actual resident memory differs")


def new_row_traces(cell, path, mode, roles, names):
    descriptor = cell["local_only_trace"]
    require(descriptor["publish"] is False, "Row trace must remain LOCAL ONLY")
    raw_path = relative(descriptor["path"], path.parent)
    if not raw_path.is_file():
        require(mode != "verify", "Verify mode requires local row trace")
        return {"available": False, "independently_recomputed": False,
                "recorded_summaries": cell.get("training_trace_summaries"),
                "reason": "LOCAL ONLY NPZ not available; portable output does not claim row verification"}
    require(sha(raw_path) == descriptor["sha256"], "Local NPZ SHA differs")
    out = {}
    with np.load(raw_path, allow_pickle=False) as raw:
        require(set(("steps", "true_class_ids", "predictions", "margins", "competitors", "group_offsets")).issubset(raw.files), "Per-step NPZ schema incomplete")
        steps, truth = raw["steps"], raw["true_class_ids"]
        predictions, margins, rivals = raw["predictions"], raw["margins"], raw["competitors"]
        require(np.array_equal(steps, np.arange(2001)), "New per-step trace must include 0 through 2000")
        require(np.array_equal(raw["group_offsets"], [0, roles["current_rare_train"][1], sum(n for _, n in roles.values())]), "Probe column groups differ")
        require(predictions.shape == margins.shape == rivals.shape == (2001, len(truth)), "Per-step array shape differs")
        require(descriptor["array_shape"] == list(predictions.shape)
                and descriptor["group_offsets"] == raw["group_offsets"].tolist()
                and descriptor["group_order"] == list(roles), "NPZ descriptor shape/group order differs")
        for key in ("current_prepared_row_ids", "resident_prefix_indices"):
            ids = raw[key]
            expected_rows = roles["current_rare_train"][1] if key == "current_prepared_row_ids" else roles["old_rare_resident"][1]
            upper = cell["identity"]["data_binding"]["y_train"]["shape"][0] if key == "current_prepared_row_ids" else cell["memory_rows"]
            require(ids.dtype.kind in "iu" and ids.ndim == 1 and len(np.unique(ids)) == len(ids)
                    and len(ids) == expected_rows and np.all((ids >= 0) & (ids < upper))
                    and array_sha(ids) == descriptor[key + "_sha256"], "Fixed probe row identity differs")
        require(np.isfinite(margins).all() and rivals.dtype.kind in "iu"
                and np.all((rivals >= 0) & (rivals < len(names))) and np.all(rivals != truth[None, :]), "Invalid per-row margin/competitor")
        # Strict positive/negative margin must agree with argmax. Exact ties may
        # be won or lost according to the fixed class-index argmax rule.
        require(np.all(predictions[margins > 0] == np.broadcast_to(truth, predictions.shape)[margins > 0])
                and np.all(predictions[margins < 0] != np.broadcast_to(truth, predictions.shape)[margins < 0]), "Margins and predictions disagree")
        offset = 0
        for role, (class_id, n) in roles.items():
            require(np.array_equal(truth[offset:offset + n], np.full(n, class_id)), "Per-row true classes differ")
            values = predictions[:, offset:offset + n]
            out[role] = row_onsets(steps, values, class_id, names)
            out[role]["mean_margin_initial_final"] = [float(margins[0, offset:offset + n].mean()), float(margins[-1, offset:offset + n].mean())]
            recorded = cell["training_trace_summaries"][role]
            for key in ("rows", "steps_including_zero", "ever_correct_rows", "never_correct_rows", "final_correct_rows",
                        "ever_correct_but_final_wrong_rows", "correct_to_wrong_events", "wrong_to_correct_events",
                        "prediction_change_events", "final_prediction_counts"):
                require(out[role][key] == recorded[key], f"Recorded row summary differs: {role}/{key}")
            require(math.isclose(out[role]["mean_margin_initial_final"][-1], recorded["final_margin_mean"], rel_tol=1e-7, abs_tol=1e-7), "Recorded final margin differs")
            for trace in cell["audit"]["traces"]:
                row = [r for r in trace["per_class"] if r["class_id"] == class_id]
                require(len(row) == 1 and np.array_equal(np.bincount(values[trace["step"]], minlength=len(names)), row[0]["prediction_counts"]),
                        "Per-row NPZ disagrees with 50-step histogram")
            offset += n
    return {"available": True, "independently_recomputed": True, "source_sha256": descriptor["sha256"],
            "publish_npz": False, "groups": out,
            "limit": "Newly measured exact per-update training path only, not recovered historical timing or held-out generalization."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = "ncd/results/problem_scout/rare_boundary"
    parser.add_argument("--pilot", type=Path, default=Path(base) / "pilot_v1")
    parser.add_argument("--diagnosis", type=Path, default=Path(base) / "diagnosis_v1")
    parser.add_argument("--probes", type=Path, default=Path(base) / "interference_v1")
    parser.add_argument("--output-dir", type=Path, default=Path(base) / "interference_analysis_v1")
    parser.add_argument("--weights", choices=("portable", "verify"), default="portable")
    args = parser.parse_args()
    for key in ("pilot", "diagnosis", "probes", "output_dir"):
        setattr(args, key, getattr(args, key).resolve())
    require(all(args.output_dir != source and not args.output_dir.is_relative_to(source)
                and not source.is_relative_to(args.output_dir) for source in (args.pilot, args.diagnosis, args.probes)),
            "Analysis output must be separate from immutable input directories")
    result = {"schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
              "command": "python -I ncd/analyze_family_interference.py " + " ".join(
                  ["--pilot", label(args.pilot), "--diagnosis", label(args.diagnosis), "--probes", label(args.probes),
                   "--output-dir", label(args.output_dir), "--weights", args.weights]),
              "analyzer_sha256": sha(Path(__file__)),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "weights_mode": args.weights,
              "pilot_aggregate_trajectories": pilot_analysis(args.pilot, args.weights),
              "existing_task_end_individual_transitions": task_end_analysis(args.diagnosis),
              "discarded_adam_probes": probe_analysis(args.probes, args.pilot, args.weights),
              "limitations": ["Exploratory exposed-data diagnostic, not a new endpoint improvement or innovation claim.",
                  "Original pilot has aggregate 50-step histograms only; no individual migrations or exact update timing can be inferred from them.",
                  "Probe metrics use current rare training rows and resident old-rare memory; test evidence is historical context only and never feedback.",
                  "Zero-gradient Adam can move parameters through inherited moments; it is not a no-op or an untrained reference.",
                  "Fixed competitor is R2L on NSL and DoS on CIC; output destination is not gradient cause.",
                  "Same data access/current order/class quota/real steps plus discarded one-Adam-step probes; not equal loss mass, gradient norms or FLOPs.",
                  "Effects are local one-step counterfactuals conditional on the recorded state and minibatch, not long-run causal mechanisms.",
                  "Each path has same-prestate clones; MIR versus extra-CE paths can have different parameters, Adam moments and selected memory indices at the same step. Cross-arm differences are descriptive, not a same-state extra-CE causal contrast.",
                  "Per-update row flips and fixed intervention updates 1/51/.../1951 are different grids; an observed flip is not necessarily directly intervened upon.",
                  "RQ1/RQ3 and the overall locked research remain unresolved by this bounded RQ2 diagnostic."]}
    result["complete"] = all(result[key]["complete"] for key in (
        "pilot_aggregate_trajectories", "existing_task_end_individual_transitions", "discarded_adam_probes"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "result.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps({"complete": result["complete"], "pilot_curves": len(result["pilot_aggregate_trajectories"]["curves"]),
                      "probe_cells": result["discarded_adam_probes"]["completed_cells"], "output": label(target)}))


if __name__ == "__main__":
    main()
