"""Section26.10 secondary operating-point analysis of saved final checkpoints.

No training or tuning: one CPU logit pass/cell, then 41 fixed TRAIN-prior shifts.
This post-primary-failure analysis does not replace Section26.5 or prove
equivalence outside the sampled curve. Outputs contain no checkpoint payloads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ncd.model import MLP

DATASETS = ("NSL-KDD", "UNSW-NB15", "CIC-IDS-2017")
METHODS = ("replay", "wacb", "freq", "farq", "derpp")
OPTIONAL_REFERENCES = ("joint",)
SEEDS = (1, 2, 3, 4, 42)
TAUS = np.linspace(-1.0, 3.0, 41).tolist()
TARGETS = (.02, .05, .10)
CALIBRATION_TARGETS = (.01, .02, .05)
POLICY = "benign_refresh_attack_memory_only"
TRAINING_SOURCE = "b835a9309d8aa3bfb20e4e30eb426cb8b076e9d1"
SOURCES = ("ncd/run_problem_scout.py", "ncd/prepare_problem_scout_data.py",
           "ncd/crr_benchmarks.py", "ncd/run_crr_real_sequence.py", "ncd/model.py",
           "ncd/train_cil.py", "ncd/data.py", "ncd/source_commit.py", "ncd/groundtruth.py",
           "src/methods/replay_dpmeans.py", "ncd/exp_h_data.py", "ncd/audit_quality_tabular.py")
METRICS = ("normal_fpr", "attack_detection", "macro_f1", "balanced_accuracy",
           "normal_recall", "attack_family_macro_recall", "attack_guard_recall")
PROTOCOL = {
    "section": "26.10", "secondary_after_primary_failure": True,
    "primary_decision_unchanged": True, "access_policy": POLICY,
    "adjustment": "logit + tau * log(original prepared TRAIN class prior)",
    "prior_role": "Original TRAIN labels, including permanent heldout rows; no test labels used to choose priors",
    "taus": TAUS, "target_fprs": list(TARGETS),
    "interpolation": "Per seed; adjacent tau brackets only; smallest tau exact-FPR point, or lowest-tau bracket; no extrapolation",
    "interpolated_macro_f1": "Linear interpolation of endpoint Macro-F1, not a measured single classifier or a metric recomputed from a mixed CM",
    "selection": "First valid completed source per dataset/method/seed, by aware completion timestamp; never by score",
    "scope": "Final full-test checkpoint only; no deployment calibration, retraining, new loss, or network",
}
CALIBRATION_PROTOCOL = {
    "section": "27.9", "targets": list(CALIBRATION_TARGETS), "total_tau_range": [-1., 3.],
    "pool": "Same seed-specific permanent Normal TRAIN holdout for every method; never optimized",
    "rule": "First adjacent tau grid bracket; 50 bisections using only Normal holdout FP counts; retain feasible upper endpoint",
    "finite_sample_limit": "FP <= floor(alpha * calibration_count); no extrapolation outside grid FPR range",
    "test_role": "Evaluate selected total tau only; test metrics never select tau",
    "replay_la": "Same Replay weights and total tau; baseline tau=1 plus delta, no extra training or inference",
    "limitation": "The permanent pool was available to allocation validation; it is not a fresh independent calibration sample",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def label(path):
    path = path.resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    for attempt in range(40):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.05)


def cm_metrics(value, names):
    """Independent integer-CM arithmetic, with undefined precision left null."""
    cm = np.asarray(value)
    n = len(names)
    if cm.shape != (n, n) or not np.issubdtype(cm.dtype, np.integer) or np.any(cm < 0):
        raise ValueError("Invalid full-test integer confusion matrix")
    support, predicted, tp = cm.sum(1), cm.sum(0), cm.diagonal()
    if np.any(support == 0):
        raise ValueError("Final evaluation is missing a class")
    normal = names.index("Normal")
    attack = [c for c in range(n) if c != normal]
    guard = [c for c in attack if support[c] >= 20]
    recall = tp / support
    f1 = 2 * tp / (support + predicted)
    return {"confusion_matrix": cm.tolist(), "support": support.tolist(),
            "normal_fpr": float((support[normal] - tp[normal]) / support[normal]),
            "attack_detection": float(1 - cm[attack, normal].sum() / support[attack].sum()),
            "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall.mean()),
            "normal_recall": float(recall[normal]),
            "attack_family_macro_recall": float(recall[attack].mean()),
            "attack_guard_recall": float(recall[guard].mean()) if guard else None}


def interpolate(points, target):
    """Keep the tau order: do not sort FPR or optimize either reported metric."""
    exact = [i for i, p in enumerate(points) if p["normal_fpr"] == target]
    brackets = [i for i in range(len(points) - 1)
                if (points[i]["normal_fpr"] - target) * (points[i + 1]["normal_fpr"] - target) < 0]
    out = {"target_fpr": target, "exact_indices": exact, "bracket_start_indices": brackets,
           "available_fpr_range": [min(p["normal_fpr"] for p in points), max(p["normal_fpr"] for p in points)]}
    if exact:
        k = min(exact, key=lambda i: points[i]["tau"])
        out.update(status="available", rule="exact_fpr_smallest_tau", selected_indices=[k],
                   selected_taus=[points[k]["tau"]], upper_endpoint_weight=0.,
                   exact_value_ranges={m: [min(points[i][m] for i in exact), max(points[i][m] for i in exact)]
                                       for m in ("attack_detection", "macro_f1")})
        out.update({m: points[k][m] for m in METRICS})
    elif brackets:
        k = min(brackets, key=lambda i: points[i]["tau"])
        a, b = points[k:k + 2]
        weight = (target - a["normal_fpr"]) / (b["normal_fpr"] - a["normal_fpr"])
        out.update(status="available", rule="adjacent_tau_lowest_bracket", selected_indices=[k, k + 1],
                   selected_taus=[a["tau"], b["tau"]], upper_endpoint_weight=weight)
        out.update({m: float(a[m] + weight * (b[m] - a[m])) if a[m] is not None and b[m] is not None else None
                    for m in METRICS})
    else:
        out.update(status="unavailable", reason="Target outside sampled FPR range; no extrapolation",
                   attack_detection=None, macro_f1=None)
    return out


def verify_source(path, expected, source_hashes, config_hash):
    result = read(path)
    if not result.get("completed"):
        return None
    identity = result["identity"]
    if tuple(identity[k] for k in ("dataset", "method", "seed")) != expected:
        raise ValueError(f"Cell identity mismatch: {label(path)}")
    if expected[1] == "joint" and identity.get("offline_joint_reference") is not True:
        raise ValueError("Joint must be explicitly identified as an offline reference")
    cfg = identity["effective_config"]
    if (cfg.get("smoke") or cfg.get("section26") is not True
            or cfg.get("section26_schema_version") != 2
            or cfg.get("section26_access_policy") != POLICY
            or identity.get("section26_access_policy") != POLICY
            or result.get("section26_schema_version") != 2 or result.get("section26_policy") != POLICY):
        raise ValueError("Only completed formal Section26.9 schema2 cells are admitted")
    fixed = {"epochs_per_task": 8, "min_steps_per_task": 2000, "buffer_per_class": 200, "batch_size": 256}
    if any(cfg.get(k) != v for k, v in fixed.items()):
        raise ValueError("Cell is not the prescribed full S2000 protocol")
    if (canonical(cfg) != identity["effective_config_sha256"] or identity["source_hashes"] != source_hashes
            or identity["config_sha256"] != config_hash or identity.get("git_commit") != TRAINING_SOURCE):
        raise ValueError("Current frozen source/config bytes or canonical config differ")
    completed = datetime.fromisoformat(result["completed_at"].replace("Z", "+00:00"))
    if completed.tzinfo is None:
        raise ValueError("Completion timestamp must be timezone-aware")
    checkpoints, tasks, names = result["checkpoints"], identity["tasks"], identity["class_order"]
    if len(checkpoints) != len(tasks):
        raise ValueError("Completed cell has a missing task checkpoint")
    for t, cp in enumerate(checkpoints):
        seen = [c for task in tasks[:t + 1] for c in task]
        schedule = cp["training_schedule"]
        target = max(8 * math.ceil(cp["train_rows"] / 256), 2000)
        if (cp["task_index"] != t or cp["seen_classes"] != seen
                or cp["identity_hash"] != canonical(identity)
                or schedule["optimizer_steps"] != schedule["target_steps"] or schedule["optimizer_steps"] != target):
            raise ValueError("Checkpoint trajectory identity or S2000 schedule differs")
        if expected[1] == "joint" and (cp["allocation"].get("offline_joint_reference") is not True
                                      or cp["allocation"].get("reference_not_replay_budget") is not True):
            raise ValueError("Joint checkpoint must disclose its offline, non-replay-budget access exception")
    final = checkpoints[-1]
    if (final["seen_classes"] != list(range(len(names)))
            or final["id"]["confusion_labels"] != list(range(len(names)))):
        raise ValueError("Final checkpoint does not include all classes in label order")
    payload = (path.parent / final["checkpoint_file"]).resolve()
    if not payload.is_relative_to(path.parent.resolve()):
        raise ValueError("Checkpoint path escapes the cell")
    if read(payload.with_suffix(".json")) != final or digest(payload) != final["checkpoint_sha256"]:
        raise ValueError("Actual final checkpoint/sidecar differs from result")
    cm_metrics(final["id"]["confusion_matrix"], names)
    return {"path": path, "result": result, "identity": identity, "final": final,
            "payload": payload, "completed": completed, "result_sha256": digest(path)}


def load_data(identity, override, cache):
    cfg = identity["effective_config"]
    root = override or Path(cfg["problem_scout_data_root"])
    root = root if root.is_absolute() else ROOT / root
    folder = root.resolve() / identity["dataset"].lower()
    key = (str(folder), canonical(identity["data_binding"]), identity["data_metadata_sha256"])
    if key in cache:
        return cache[key], 0.
    start = time.perf_counter()
    info = read(folder / "data_quality.json")
    if (canonical(info) != identity["data_metadata_sha256"] or info.get("status") != "complete"
            or info["class_order"] != identity["class_order"] or info["tasks"] != identity["tasks"]
            or info["feature_names"] != identity["feature_names"]):
        raise ValueError("Prepared data receipt/class/feature mapping differs")
    arrays = {}
    for split in ("train", "test"):
        for field in ("x", "y"):
            name = f"{field}_{split}"
            a = np.load(folder / f"{split}_{field}.npy", mmap_mode="r", allow_pickle=False)
            binding = identity["data_binding"][name]
            if list(a.shape) != binding["shape"] or a.dtype.str != binding["dtype"]:
                raise ValueError(f"Prepared {name} schema differs")
            h = hashlib.sha256()
            for offset in range(0, len(a), 16384):
                values = np.ascontiguousarray(a[offset:offset + 16384])
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite prepared {name}")
                h.update(memoryview(values).cast("B"))
            if h.hexdigest() != binding["sha256"]:
                raise ValueError(f"Prepared {name} payload differs")
            arrays[name] = a
    n = len(identity["class_order"])
    for split in ("train", "test"):
        y = arrays[f"y_{split}"]
        if y.dtype != np.int64 or len(y) != len(arrays[f"x_{split}"]) or np.any((y < 0) | (y >= n)):
            raise ValueError("Prepared label type/range/rows differ")
    counts = np.bincount(arrays["y_train"], minlength=n)
    if np.any(counts == 0):
        raise ValueError("Original TRAIN prior has an absent class")
    arrays["counts"] = counts
    arrays["log_prior"] = np.log(counts / counts.sum()).astype(np.float32)
    arrays["receipt_sha256"] = digest(folder / "data_quality.json")
    cache[key] = arrays
    return arrays, time.perf_counter() - start


def verify_holdout(identity, y):
    for c in range(len(identity["class_order"])):
        rows = np.flatnonzero(y == c)
        count = max(5, math.ceil(.1 * len(rows)))
        perm = np.random.default_rng(identity["seed"] * 100003 + c * 1009 + 17).permutation(rows)
        expected = {"train_count": len(rows) - count, "validation_count": count,
                    "validation_indices_sha256": hashlib.sha256(perm[:count].tobytes()).hexdigest()}
        if identity["section26_holdout"][str(c)] != expected:
            raise ValueError("Permanent TRAIN holdout differs from canonical split")


def verify_curve(curve, final, names):
    points = curve["points"]
    if [p["tau"] for p in points] != TAUS:
        raise ValueError("Saved curve has a different tau grid")
    for p in points:
        if cm_metrics(p["confusion_matrix"], names) != {k: v for k, v in p.items() if k != "tau"}:
            raise ValueError("Saved curve metrics differ from integer CMs")
    if points[10]["confusion_matrix"] != final["id"]["confusion_matrix"]:
        raise ValueError("tau=0 does not exactly reproduce the original full-test CM")
    if "la_id" in final and points[20]["confusion_matrix"] != final["la_id"]["confusion_matrix"]:
        raise ValueError("Replay tau=1 does not exactly reproduce the original LA CM")
    expected = {str(t): interpolate(points, t) for t in TARGETS}
    if curve["operating_points"] != expected:
        raise ValueError("Saved interpolation differs from the fixed adjacent-tau rule")


def cm_difference(original, reproduced):
    a, b = np.asarray(original, dtype=np.int64), np.asarray(reproduced, dtype=np.int64)
    delta = b - a
    return {"original_confusion_matrix": a.tolist(), "different_entries": int(np.count_nonzero(delta)),
            "absolute_difference_sum": int(np.abs(delta).sum()),
            "maximum_absolute_difference": int(np.abs(delta).max()),
            "changed_entries": [{"true_class": int(i), "predicted_class": int(j),
                                 "original": int(a[i, j]), "reproduced": int(b[i, j])}
                                for i, j in zip(*np.nonzero(delta))]}


def model_logits(model, x, batch, indices=None):
    count = len(x) if indices is None else len(indices)
    logits = None
    with torch.inference_mode():
        for start in range(0, count, batch):
            rows = x[start:start + batch] if indices is None else x[indices[start:start + batch]]
            scores = model(torch.from_numpy(np.array(rows, copy=True)))
            if not torch.isfinite(scores).all():
                raise ValueError("Nonfinite final-model logits")
            if logits is None:
                logits = np.empty((count, scores.shape[1]), dtype=np.float32)
            logits[start:start + len(rows)] = scores.numpy()
    return torch.from_numpy(logits)


def calibrate_normal(model, data, identity, batch, test_logits):
    """Choose tau with Normal holdout only, then report test metrics at that tau."""
    started = time.perf_counter()
    names, seed = identity["class_order"], identity["seed"]
    normal, n = names.index("Normal"), len(names)
    rows = np.flatnonzero(data["y_train"] == normal)
    count = max(5, math.ceil(.1 * len(rows)))
    indices = np.random.default_rng(seed * 100003 + normal * 1009 + 17).permutation(rows)[:count]
    indices_sha = hashlib.sha256(indices.tobytes()).hexdigest()
    if indices_sha != identity["section26_holdout"][str(normal)]["validation_indices_sha256"]:
        raise ValueError("Normal calibration pool differs from permanent holdout")
    prior = torch.from_numpy(data["log_prior"])
    if not bool(torch.all(prior[normal] >= prior)):
        raise ValueError("Normal prior is not maximal; monotone tau calibration is not justified")
    infer_started = time.perf_counter()
    logits = model_logits(model, data["x_train"], batch, indices)
    inference_seconds = time.perf_counter() - infer_started
    def fp(tau):
        return int(((logits + tau * prior).argmax(1) != normal).sum())
    counts = [fp(tau) for tau in TAUS]
    if any(a < b for a, b in zip(counts, counts[1:])):
        raise ValueError("Normal calibration FP counts are not monotone on the fixed grid")
    grid = [{"tau": tau, "false_positives": value, "normal_count": count, "normal_fpr": value / count}
            for tau, value in zip(TAUS, counts)]
    selected = {}
    for alpha in CALIBRATION_TARGETS:
        limit = math.floor(alpha * count)
        item = {"target_fpr": alpha, "maximum_false_positives": limit}
        if not counts[-1] / count <= alpha <= counts[0] / count:
            item.update(status="unavailable", reason="Target outside calibration grid FPR range; no extrapolation")
        else:
            k = next(i for i, value in enumerate(counts) if value <= limit)
            lower, upper = TAUS[max(0, k - 1)], TAUS[k]
            bracket = [lower, upper]
            if k:
                for _ in range(50):
                    mid = (lower + upper) / 2
                    if fp(mid) <= limit:
                        upper = mid
                    else:
                        lower = mid
            actual = fp(upper)
            if actual > limit:
                raise ValueError("Calibrated upper endpoint violates its finite-sample FP limit")
            pred = (test_logits + upper * prior).argmax(1).numpy()
            cm = np.bincount(data["y_test"] * n + pred, minlength=n * n).reshape(n, n)
            item.update(status="available", total_tau=upper, initial_grid_bracket=bracket,
                        final_bisection_interval=[lower, upper], bisection_iterations=50 if k else 0,
                        calibration_false_positives=actual, calibration_count=count,
                        calibration_fpr=actual / count, test=cm_metrics(cm, names))
        selected[str(alpha)] = item
    feature_bytes = count * data["x_train"].shape[1] * data["x_train"].dtype.itemsize
    label_bytes = count * data["y_train"].dtype.itemsize
    out = {"protocol": CALIBRATION_PROTOCOL, "pool": {"count": count, "indices_sha256": indices_sha,
           "feature_bytes": feature_bytes, "label_bytes": label_bytes, "index_bytes": indices.nbytes,
           "feature_and_label_storage_bytes": feature_bytes + label_bytes,
           "outside_attack_replay_memory_budget": True}, "grid": grid, "operating_points": selected,
           "cost_seconds": {"normal_holdout_logit_inference": inference_seconds,
                            "calibration_total": time.perf_counter() - started},
           "numerical_rule": "Float32 logits/prior and first-index argmax ties; bisection returns the feasible upper endpoint, not an exact real-number minimum"}
    if identity["method"] == "replay":
        out["replay_la_alias"] = {"same_checkpoint_and_predictions": True, "initial_tau": 1.,
                                  "extra_training_seconds": 0., "extra_inference_seconds": 0.,
                                  "operating_points": {key: dict(value, delta_tau=value["total_tau"] - 1.)
                                                       if value["status"] == "available" else dict(value)
                                                       for key, value in selected.items()}}
    return out


def verify_calibration(curve, identity):
    calibration = curve["normal_holdout_calibration"]
    names = identity["class_order"]
    expected_pool = identity["section26_holdout"][str(names.index("Normal"))]
    pool, grid = calibration["pool"], calibration["grid"]
    count = pool["count"]
    if (calibration["protocol"] != CALIBRATION_PROTOCOL or count != expected_pool["validation_count"]
            or pool["indices_sha256"] != expected_pool["validation_indices_sha256"]
            or [p["tau"] for p in grid] != TAUS):
        raise ValueError("Saved calibration protocol/pool/grid differs")
    counts = [p["false_positives"] for p in grid]
    if (any(not isinstance(v, int) or not 0 <= v <= count for v in counts)
            or any(a < b for a, b in zip(counts, counts[1:]))
            or any(p["normal_count"] != count or p["normal_fpr"] != p["false_positives"] / count for p in grid)):
        raise ValueError("Saved calibration grid count arithmetic differs")
    ops = calibration["operating_points"]
    if set(ops) != {str(a) for a in CALIBRATION_TARGETS}:
        raise ValueError("Saved calibration alpha grid differs")
    for alpha in CALIBRATION_TARGETS:
        op, limit = ops[str(alpha)], math.floor(alpha * count)
        available = counts[-1] / count <= alpha <= counts[0] / count
        if op["maximum_false_positives"] != limit or op["target_fpr"] != alpha:
            raise ValueError("Saved calibration finite-sample target differs")
        if not available:
            if op["status"] != "unavailable":
                raise ValueError("Out-of-range calibration target was extrapolated")
            continue
        k = next(i for i, value in enumerate(counts) if value <= limit)
        bracket = [TAUS[max(0, k - 1)], TAUS[k]]
        lower, upper = op["final_bisection_interval"]
        actual = op["calibration_false_positives"]
        if (op["status"] != "available" or op["initial_grid_bracket"] != bracket
                or op["bisection_iterations"] != (50 if k else 0)
                or not bracket[0] <= lower <= upper == op["total_tau"] <= bracket[1]
                or not isinstance(actual, int) or not 0 <= actual <= limit
                or op["calibration_count"] != count or op["calibration_fpr"] != actual / count
                or cm_metrics(op["test"]["confusion_matrix"], names) != op["test"]):
            raise ValueError("Saved calibrated operating point arithmetic differs")
    if identity["method"] == "replay":
        expected = {key: dict(value, delta_tau=value["total_tau"] - 1.) if value["status"] == "available" else dict(value)
                    for key, value in ops.items()}
        if calibration["replay_la_alias"]["operating_points"] != expected:
            raise ValueError("Replay+LA calibrated alias differs from Replay total tau")


def run_cell(cell, data, data_seconds, binding, output, execution_args):
    identity, final = cell["identity"], cell["final"]
    names, cfg = identity["class_order"], identity["effective_config"]
    if output.exists():
        previous = read(output)
        if not previous.get("completed") or previous.get("binding") != binding:
            raise ValueError(f"Existing curve has different/incomplete provenance; preserve it and choose a new output: {label(output)}")
        verify_curve(previous, final, names)
        if binding["parameters"]["calibrate_normal_holdout"]:
            if "normal_holdout_calibration" not in previous:
                raise ValueError("Completed curve is missing requested Normal holdout calibration")
            verify_calibration(previous, identity)
        return previous
    started = time.perf_counter()
    state = torch.load(cell["payload"], map_location="cpu", weights_only=True)
    if (state["identity"] != identity or state["task_index"] != final["task_index"]
            or state["seen_classes"] != final["seen_classes"] or state["allocation"] != final["allocation"]):
        raise ValueError("Actual checkpoint canonical identity/task/allocation differs")
    verify_holdout(identity, data["y_train"])
    model = MLP(data["x_test"].shape[1], int(cfg["hidden"]), len(names))
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()
    load_seconds = time.perf_counter() - started
    del state
    inference_start = time.perf_counter()
    batch = binding["parameters"]["batch_size"]
    base = model_logits(model, data["x_test"], batch)
    inference_seconds = time.perf_counter() - inference_start
    sweep_start = time.perf_counter()
    log_prior = torch.from_numpy(data["log_prior"])
    y, n = data["y_test"], len(names)
    points = []
    for tau in TAUS:
        pred = (base + tau * log_prior).argmax(1).numpy()
        cm = np.bincount(y * n + pred, minlength=n * n).reshape(n, n)
        points.append({"tau": tau, **cm_metrics(cm, names)})
    curve = {"schema_version": 1, "completed": False,
             "binding": binding, "dataset": identity["dataset"], "method": identity["method"], "seed": identity["seed"],
             "analysis_role": "section27_9_offline_reference" if identity["method"] == "joint" else "section26_10_main_method",
             "excluded_from_section26_10_judgement": identity["method"] == "joint",
             "protocol": PROTOCOL, "execution_arguments": execution_args,
             "class_order": names, "points": points,
             "operating_points": {str(t): interpolate(points, t) for t in TARGETS},
             "cost_note": "cell_attempt_elapsed includes component timers, and calibration_total includes its inference timer; do not add totals to components. Full run/resume elapsed is in summary.execution.",
             "cost_seconds": {"data_verification_charged": data_seconds, "checkpoint_load_and_holdout": load_seconds,
                              "single_logit_inference": inference_seconds, "41_point_sweep": time.perf_counter() - sweep_start},
             "verification": {"actual_final_payload_sha": True, "final_sidecar_and_canonical_identity": True,
                              "source_config_and_all_four_prepared_array_hashes": True,
                              "curve_verified": False, "all_requested_checks_passed": False,
                              "tau0_exact_cm": points[10]["confusion_matrix"] == final["id"]["confusion_matrix"],
                              "replay_tau1_exact_cm": (points[20]["confusion_matrix"] == final["la_id"]["confusion_matrix"]
                                                       if "la_id" in final else None)},
             "limitations": ["CPU/GPU numerical changes are not tolerated in CM reproduction; a mismatch stops this cell.",
                             "Curve targets are descriptive test-set operating points, not independently calibrated deployment thresholds.",
                             "Only final checkpoint payload is re-read here; trajectory records are identity/schedule checked.",
                             "Equal capacity caps do not imply equal occupied replay memory."]}
    try:
        verify_curve(curve, final, names)
        curve["verification"]["curve_verified"] = True
        if binding["parameters"]["calibrate_normal_holdout"]:
            curve["normal_holdout_calibration"] = calibrate_normal(model, data, identity, batch, base)
            verify_calibration(curve, identity)
    except Exception as exc:
        curve["failed_at"] = datetime.now(timezone.utc).isoformat()
        curve["error"] = "Saved-checkpoint curve or Normal holdout calibration verification failed; this cell is not complete."
        curve["error_type"] = type(exc).__name__
        curve["cm_reproduction_differences"] = {"tau0": cm_difference(final["id"]["confusion_matrix"], points[10]["confusion_matrix"])}
        if "la_id" in final:
            curve["cm_reproduction_differences"]["tau1"] = cm_difference(final["la_id"]["confusion_matrix"], points[20]["confusion_matrix"])
        curve["cost_seconds"]["cell_attempt_elapsed"] = time.perf_counter() - started
        failed = output.with_name("result.failed.json")
        if failed.exists():
            failed = output.with_name("result.failed." + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json")
        save(failed, curve)
        raise
    curve.update(completed=True, completed_at=datetime.now(timezone.utc).isoformat())
    curve["verification"]["all_requested_checks_passed"] = True
    curve["cost_seconds"]["cell_attempt_elapsed"] = time.perf_counter() - started
    save(output, curve)
    return curve


def stats(values):
    return {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}


def summarize(cells, requested, missing):
    lookup = {(r["dataset"], r["method"], r["seed"]): r for r in cells}
    groups = []
    for dataset in DATASETS:
        for method in METHODS:
            rows = [lookup.get((dataset, method, seed)) for seed in SEEDS]
            if not any(rows):
                continue
            group = {"dataset": dataset, "method": method, "completed_seeds": [s for s, r in zip(SEEDS, rows) if r],
                     "status": "complete" if all(rows) else "pending"}
            if all(rows):
                group["mean_tau_curve"] = [{"tau": tau, **{m: stats([r["points"][i][m] for r in rows]) for m in METRICS}}
                                           for i, tau in enumerate(TAUS)]
                group["operating_points"] = {}
                for target in TARGETS:
                    ops = [r["operating_points"][str(target)] for r in rows]
                    group["operating_points"][str(target)] = (
                        {"status": "available", **{m: stats([p[m] for p in ops]) for m in ("attack_detection", "macro_f1")}}
                        if all(p["status"] == "available" for p in ops) else
                        {"status": "unavailable", "unavailable_seeds": [s for s, p in zip(SEEDS, ops) if p["status"] != "available"]})
            groups.append(group)
    candidates = []
    for candidate in ("farq", "wacb"):
        results, unavailable = [], False
        for dataset in ("CIC-IDS-2017", "UNSW-NB15"):
            for target in TARGETS:
                pairs = [(lookup.get((dataset, candidate, s)), lookup.get((dataset, "replay", s))) for s in SEEDS]
                item = {"dataset": dataset, "target_fpr": target}
                if any(a is None or b is None for a, b in pairs):
                    item.update(status="pending", reason="Missing matched seed output; no seeds discarded")
                elif any(r["operating_points"][str(target)]["status"] != "available" for pair in pairs for r in pair):
                    item.update(status="unavailable", reason="At least one paired seed does not bracket the target")
                else:
                    delta = [100 * (a["operating_points"][str(target)]["attack_detection"]
                                    - b["operating_points"][str(target)]["attack_detection"]) for a, b in pairs]
                    mean = float(np.mean(delta))
                    item.update(status="available", seeds=list(SEEDS), paired_adr_change_pp=delta,
                                paired_adr_change_pp_stats=stats(delta),
                                meets_criterion=mean >= 2 if target == .05 else mean > 0)
                unavailable |= item["status"] != "available"
                results.append(item)
        status = "pending_or_unavailable" if unavailable else (
            "criterion_met" if all(r["meets_criterion"] for r in results) else "criterion_not_met")
        candidates.append({"candidate": candidate, "status": status, "paired_results": results})
    overall = ("criterion_met" if any(c["status"] == "criterion_met" for c in candidates) else
               "criterion_not_met" if all(c["status"] == "criterion_not_met" for c in candidates) else "pending_or_unavailable")
    calibration_groups, offline_reference_groups = [], []
    for dataset in DATASETS:
        for method in (*METHODS, "replay_la", *OPTIONAL_REFERENCES):
            rows = [lookup.get((dataset, "replay" if method == "replay_la" else method, seed)) for seed in SEEDS]
            if not any(rows):
                continue
            group = {"dataset": dataset, "method": method, "operating_points": {}}
            for alpha in CALIBRATION_TARGETS:
                ops = [r.get("normal_holdout_calibration", {}).get("operating_points", {}).get(str(alpha)) if r else None for r in rows]
                unavailable = [s for s, p in zip(SEEDS, ops) if not p or p["status"] != "available"]
                group["operating_points"][str(alpha)] = (
                    {"status": "unavailable_or_pending", "missing_or_unavailable_seeds": unavailable}
                    if unavailable else {"status": "available", "seeds": list(SEEDS),
                         "test": {metric: stats([p["test"][metric] for p in ops]) for metric in METRICS},
                         "calibration_fpr": stats([p["calibration_fpr"] for p in ops]),
                         "selected_total_taus": [p["total_tau"] for p in ops]})
            if method == "replay_la":
                group["reuses_replay_total_tau_predictions"] = True
            if method == "joint":
                group.update(role="section27_9_offline_reference", reference_not_replay_budget=True,
                             old_attack_training_pool_access=True,
                             excluded_from_section26_10_main_groups_and_judgement=True)
                offline_reference_groups.append(group)
            else:
                calibration_groups.append(group)
    return {"schema_version": 1, "updated_at": datetime.now(timezone.utc).isoformat(), "protocol": PROTOCOL,
            "requested_cells": requested, "completed_cells": len(cells), "missing_cells": missing,
            "completed_main_method_cells": sum(r["method"] in METHODS for r in cells),
            "completed_offline_reference_cells": sum(r["method"] in OPTIONAL_REFERENCES for r in cells),
            "normal_holdout_calibration": {"protocol": CALIBRATION_PROTOCOL, "groups": calibration_groups,
                                          "offline_reference_groups": offline_reference_groups,
                                          "section26_10_test_interpolation_not_used_for_threshold_selection": True},
            "groups": groups, "secondary_judgement": {"status": overall, "same_candidate_required_in_both_datasets": True,
            "candidates": candidates, "interpretation_limit": "Failure does not establish statistical equivalence; conclusions are limited to the registered finite tau grid and observed seeds."},
            "cell_evidence": [{k: r[k] for k in ("dataset", "method", "seed", "completed_at", "binding", "cost_seconds")} for r in cells]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", dest="inputs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=METHODS + OPTIONAL_REFERENCES, default=list(METHODS),
                        help="Default five Section26.10 methods; optional joint is a separate Section27.9 offline reference")
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, help="Default: checkpoint's original evaluation batch size")
    parser.add_argument("--calibrate-normal-holdout", action=argparse.BooleanOptionalAction, default=True,
                        help="Also select alpha=1/2/5 percent operating points solely on permanent Normal holdout")
    args = parser.parse_args()
    if args.cpu_threads < 1 or (args.batch_size is not None and args.batch_size < 1):
        parser.error("Thread and batch counts must be positive")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    inputs = args.inputs or [ROOT / "ncd/results/problem_scout/section26/local_v2",
                             ROOT / "ncd/results/problem_scout/section26/local_cic_v2"]
    attempt_started, attempt_clock = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    output, cache, cells, missing, paired = args.output_dir.resolve(), {}, [], [], {}
    requested = len(args.datasets) * len(args.methods) * len(args.seeds)
    execution_args = {"entrypoint": "python -I ncd/analyze_farq_operating_curves.py",
                      "inputs": [label(p) for p in inputs], "output_dir": label(output),
                      "data_root": label(args.data_root) if args.data_root else "checkpoint original config",
                      "datasets": args.datasets, "methods": args.methods, "seeds": args.seeds,
                      "device": args.device, "cpu_threads": args.cpu_threads, "batch_size": args.batch_size,
                      "calibrate_normal_holdout": args.calibrate_normal_holdout}
    previous = read(output / "summary.json") if (output / "summary.json").exists() else {}
    history = previous.get("execution", {}).get("attempts", [])
    attempt = {"started_at": attempt_started, "arguments": execution_args,
               "new_inference_cells_attempted": 0, "new_inference_cells_completed": 0,
               "reused_cells": 0, "resume_validation_seconds": 0.,
               "elapsed_includes": "Source/data hashes, inference, sweep, calibration, resume validation and preceding output writes; final summary write itself excluded"}
    def write_summary(status):
        attempt["status"] = status
        attempt["elapsed_seconds"] = time.perf_counter() - attempt_clock
        if status != "running":
            attempt["finished_at"] = datetime.now(timezone.utc).isoformat()
        summary = summarize(cells, requested, missing)
        summary["execution"] = {"attempts": history + [dict(attempt)]}
        save(output / "summary.json", summary)
    try:
        source_hashes = {p: digest(ROOT / p) for p in SOURCES}
        config_hash = digest(ROOT / "ncd/configs/problem_scout_section21.json")
        script_hash = digest(Path(__file__))
        for dataset in args.datasets:
            slug = dataset.lower().replace("-", "_")
            for method in args.methods:
                for seed in args.seeds:
                    expected = (dataset, method, seed)
                    attempt["current_cell"] = dict(zip(("dataset", "method", "seed"), expected))
                    cell_output = output / slug / f"{method}_seed{seed}" / "result.json"
                    reused, check_start = cell_output.exists(), time.perf_counter()
                    try:
                        paths = {root.resolve() / slug / f"{method}_seed{seed}" / "result.json" for root in inputs}
                        candidates = [verify_source(p, expected, source_hashes, config_hash) for p in sorted(paths) if p.exists()]
                        candidates = sorted((c for c in candidates if c), key=lambda c: (c["completed"], str(c["path"])))
                        if not candidates:
                            missing.append({"dataset": dataset, "method": method, "seed": seed})
                            continue
                        cell = candidates[0]
                        identity = cell["identity"]
                        match = {k: identity[k] for k in ("section26_holdout", "data_binding", "class_order", "tasks")}
                        match["matched_config"] = {k: identity["effective_config"][k] for k in
                                                   ("hidden", "learning_rate", "epochs_per_task", "min_steps_per_task",
                                                    "batch_size", "buffer_per_class", "section26_schema_version")}
                        key = (dataset, seed)
                        if key in paired and paired[key] != match:
                            raise ValueError("Methods do not share the same permanent holdout/data/task mapping")
                        paired[key] = match
                        data, seconds = load_data(identity, args.data_root, cache)
                        if cell["result"]["train_counts"] != {str(c): int(v) for c, v in enumerate(data["counts"])}:
                            raise ValueError("Original TRAIN frequencies differ from the source result")
                        binding = {"protocol_sha256": canonical(PROTOCOL), "analysis_script_sha256": script_hash,
                                   "calibration_protocol_sha256": canonical(CALIBRATION_PROTOCOL),
                                   "source_result": label(cell["path"]), "source_result_sha256": cell["result_sha256"],
                                   "source_completed_at": cell["result"]["completed_at"],
                                   "checkpoint_file": cell["final"]["checkpoint_file"], "checkpoint_sha256": cell["final"]["checkpoint_sha256"],
                                   "identity_sha256": canonical(identity), "training_source_commit": identity["git_commit"],
                                   "source_hashes": source_hashes, "config_sha256": config_hash,
                                   "data_binding": identity["data_binding"], "receipt_sha256": data["receipt_sha256"],
                                   "original_train_counts": data["counts"].tolist(), "prior": (data["counts"] / data["counts"].sum()).tolist(),
                                   "log_prior_float32": data["log_prior"].tolist(),
                                   "parameters": {"device": args.device, "cpu_threads": args.cpu_threads,
                                                  "calibrate_normal_holdout": args.calibrate_normal_holdout,
                                                  "batch_size": args.batch_size or identity["effective_config"]["eval_batch_size"],
                                                  "torch": str(torch.__version__), "numpy": np.__version__}}
                        if not reused:
                            attempt["new_inference_cells_attempted"] += 1
                        curve = run_cell(cell, data, seconds, binding, cell_output, execution_args)
                        attempt["reused_cells" if reused else "new_inference_cells_completed"] += 1
                        cells.append(curve)
                    finally:
                        if reused:
                            attempt["resume_validation_seconds"] += time.perf_counter() - check_start
                    write_summary("running")
                    print(json.dumps({"dataset": dataset, "method": method, "seed": seed,
                                      "completed": True, "reused": reused, "tau0_exact_cm": True}), flush=True)
    except BaseException as exc:
        attempt["error"] = "Analysis stopped; completed outputs and any failed curve evidence are preserved."
        attempt["error_type"] = type(exc).__name__
        write_summary("failed")
        raise
    attempt.pop("current_cell", None)
    write_summary("finished")


if __name__ == "__main__":
    main()
