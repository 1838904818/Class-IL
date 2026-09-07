"""CPU score-level R1 experiment mathematics, not checkpoint/SHAP extraction."""
from __future__ import annotations

import hashlib
import json
import math
import time
import tracemalloc

import numpy as np


ARMS = ("random", "periodic", "error-only", "disagreement-only", "expansion-aware",
        "audit-only", "expansion-w0", "expansion-no-attribution", "r1-reachability",
        "expansion-r1-reachability")
ATTRIBUTION_ARMS = {"expansion-aware", "expansion-w0", "expansion-r1-reachability"}
DECOMPOSITION_ARMS = ATTRIBUTION_ARMS | {"expansion-no-attribution"}


class Blocked(ValueError):
    pass


def check(condition, message):
    if not condition:
        raise Blocked(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def finite_matrix(value, shape=None):
    raw = np.asarray(value)
    check(raw.dtype.kind in "fiu", "numeric non-boolean score arrays required")
    x = np.asarray(raw, dtype=np.float64)
    check(x.ndim == 2 and np.isfinite(x).all(), "finite 2D score matrix required")
    check(shape is None or x.shape == shape, "score matrix shape mismatch")
    return x


def sigmoid(x):
    # Stable for finite, arbitrarily large logits, without clipping scores.
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    e = np.exp(x[~positive])
    out[~positive] = e / (1.0 + e)
    return out


def router_normalize(raw, epsilon):
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            out = (raw - raw.mean(axis=1, keepdims=True)) / (raw.std(axis=1, keepdims=True) + epsilon)
        check(np.isfinite(out).all(), "nonfinite router normalization")
        return out
    except FloatingPointError as exc:
        raise Blocked("router normalization overflow/invalid arithmetic") from exc


def fused(head_logits, router_raw, fusion, target=None, params=(1.0, 0.0)):
    h = finite_matrix(head_logits)
    r = finite_matrix(router_raw, h.shape)
    if target is not None:
        check(0 <= target < h.shape[1], "existing target required")
        h = h.copy()
        try:
            with np.errstate(over="raise", invalid="raise"):
                h[:, target] = params[0] * h[:, target] + params[1]
        except FloatingPointError as exc:
            raise Blocked("R1 transformed logit overflow") from exc
    return sigmoid(h) + fusion["router_weight"] * router_normalize(r, fusion["epsilon"])


def margins(scores, target, rivals=None):
    rivals = list(range(scores.shape[1])) if rivals is None else list(rivals)
    rivals = [x for x in rivals if x != target]
    check(bool(rivals), "at least one rival required")
    return scores[:, target] - scores[:, rivals].max(axis=1)


def score_path(partition, classes, old_classes, target_class, fusion):
    check(partition.get("old_head_logits") is not None and partition.get("old_router_raw") is not None,
          "A/B/C/D unavailable: independent old raw scores are absent")
    old_indices = [classes.index(c) for c in old_classes]
    old_target = old_classes.index(target_class)
    target = classes.index(target_class)
    new_h = finite_matrix(partition["new_head_logits"])
    new_r = finite_matrix(partition["new_router_raw"], new_h.shape)
    old_h = finite_matrix(partition["old_head_logits"], (len(new_h), len(old_classes)))
    old_r = finite_matrix(partition["old_router_raw"], old_h.shape)
    a = fused(old_h, old_r, fusion)
    b = fused(new_h[:, old_indices], new_r[:, old_indices], fusion)
    deployed = fused(new_h, new_r, fusion)
    values = {"A": margins(a, old_target), "B": margins(b, old_target),
              "C": margins(deployed, target, old_indices), "D": margins(deployed, target)}
    effects = {"state": values["B"] - values["A"], "normalization": values["C"] - values["B"],
               "new_rival": values["D"] - values["C"]}
    residual = values["D"] - values["A"] - sum(effects.values())
    check(np.max(np.abs(residual)) <= fusion["identity_atol"], "score path does not telescope")
    check(np.max(effects["new_rival"]) <= fusion["identity_atol"], "new-rival monotonicity failure")
    return {"scores": values, "effects": effects, "identity_max_abs": float(np.max(np.abs(residual))),
            "old_predictions": np.asarray(old_classes)[a.argmax(axis=1)], "deployed": deployed}


def group_intervals(metrics, alpha, family_size, min_groups):
    """Paired group-equal means with simultaneous Hoeffding intervals.

    Each metric is (paired row values, corresponding group IDs, known lo, hi).
    Groups are independent sampling units; observations inside each group can
    be arbitrarily dependent. The estimand weights represented groups equally.
    family_size may conservatively include tests not computed in this call.
    """
    check(0 < alpha < 1 and family_size >= len(metrics) and min_groups >= 2, "invalid simultaneous CI policy")
    result = {}
    for name, (values, groups, lower, upper) in metrics.items():
        v, g = np.asarray(values, dtype=float), np.asarray(groups, dtype=str)
        check(v.ndim == 1 and len(v) == len(g) and len(v) > 0 and np.isfinite(v).all(), "invalid metric support")
        check(lower < upper and np.all(v >= lower) and np.all(v <= upper), "metric outside declared range")
        units = sorted(set(g.tolist()))
        check(len(units) >= min_groups, "insufficient independent group support")
        means = np.asarray([v[g == group].mean() for group in units])
        estimate = float(means.mean())
        radius = (upper - lower) * math.sqrt(math.log(2 * family_size / alpha) / (2 * len(units)))
        result[name] = {"estimate": estimate, "lower": max(lower, estimate - radius),
                        "upper": min(upper, estimate + radius), "groups": len(units), "rows": len(v),
                        "estimand": "equal-weight mean across represented independent groups",
                        "method": "paired-group-Hoeffding-union-bound", "family_size": family_size, "alpha": alpha}
    return result


def classification_report(labels, predictions, classes, attack_classes=()):
    y, pred = np.asarray(labels), np.asarray(predictions)
    check(len(y) == len(pred) and len(y) > 0 and set(y).issubset(classes) and set(pred).issubset(classes), "invalid predictions/labels")
    counts = np.asarray([[np.sum((y == a) & (pred == b)) for b in classes] for a in classes], dtype=int)
    support = counts.sum(axis=1)
    check(np.all(support > 0), "missing class support in classification report")
    tp = counts.diagonal()
    precision = np.divide(tp, counts.sum(axis=0), out=np.zeros(len(classes), dtype=float), where=counts.sum(axis=0) > 0)
    recall = tp / support
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(classes)), where=(precision + recall) > 0)
    report = {"class_axis": list(classes), "confusion_counts": counts.tolist(), "accuracy": float(tp.sum() / len(y)),
              "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall.mean()),
              "per_class": {c: {"support": int(support[i]), "recall": float(recall[i]), "precision": float(precision[i]),
                                  "f1": float(f1[i]), "errors": int(support[i] - tp[i])} for i, c in enumerate(classes)}}
    if attack_classes:
        actual_attack, predicted_attack = np.isin(y, attack_classes), np.isin(pred, attack_classes)
        check(actual_attack.any() and (~actual_attack).any(), "attack and benign support required")
        report["attack_recall"] = float(predicted_attack[actual_attack].mean())
        report["false_positive_rate"] = float(predicted_attack[~actual_attack].mean())
        report["missed_attacks"] = int((~predicted_attack[actual_attack]).sum())
        report["false_positives"] = int(predicted_attack[~actual_attack].sum())
    return report


def paired_metrics(labels, baseline_predictions, repaired_predictions, groups, classes, old_classes, attack_classes=()):
    y, before, after, g = map(np.asarray, (labels, baseline_predictions, repaired_predictions, groups))
    delta = (after != y).astype(float) - (before != y).astype(float)
    metrics = {f"class_error_delta:{c}": (delta[y == c], g[y == c], -1, 1) for c in classes}
    for name, class_set in (("old_error_delta", old_classes), ("new_error_delta", [c for c in classes if c not in old_classes])):
        mask = np.isin(y, class_set)
        metrics[name] = (delta[mask], g[mask], -1, 1)
    metrics["overall_error_delta"] = (delta, g, -1, 1)
    if attack_classes:
        attack = np.isin(y, attack_classes)
        before_a, after_a = np.isin(before, attack_classes), np.isin(after, attack_classes)
        miss_delta = (~after_a).astype(float) - (~before_a).astype(float)
        fpr_delta = after_a.astype(float) - before_a.astype(float)
        metrics["missed_attack_rate_delta"] = (miss_delta[attack], g[attack], -1, 1)
        metrics["false_positive_rate_delta"] = (fpr_delta[~attack], g[~attack], -1, 1)
    return metrics


def reachability_bound(partition, target, fusion, r1):
    """Pointwise necessary upper bound. A SINGLE (a,b) must fit all rows later."""
    h = finite_matrix(partition["new_head_logits"])
    r = finite_matrix(partition["new_router_raw"], h.shape)
    scores = fused(h, r, fusion)
    z = h[:, target]
    maximum_logit = np.maximum(r1["scale_min"] * z, r1["scale_max"] * z) + r1["offset_max"]
    maximum_target = sigmoid(maximum_logit) + fusion["router_weight"] * router_normalize(r, fusion["epsilon"])[:, target]
    other = np.delete(scores, target, axis=1).max(axis=1)
    return {"can_cross_necessary_bound": maximum_target > other,
            "maximum_target_score": maximum_target, "rival_score": other,
            "jointly_achievable": False}


def _explanation_signal(partition, target_class, mask):
    info = partition.get("explanations", {}).get(target_class)
    check(isinstance(info, dict) and info.get("independent_draws_by_group") is True, "method-internal attribution evidence missing")
    a, d = np.asarray(info["A"], float), np.asarray(info["D"], float)
    check(a.shape == d.shape and a.ndim == 3 and a.shape[0] >= 4 and a.shape[0] % 2 == 0 and
          a.shape[1] == len(mask) and np.isfinite(a).all() and np.isfinite(d).all(), "invalid repeated attribution arrays")
    check(np.all(np.sum(np.abs(a), axis=2) > 0) and np.all(np.sum(np.abs(d), axis=2) > 0), "zero explanation norm")
    a = a / np.sum(np.abs(a), axis=2, keepdims=True)
    d = d / np.sum(np.abs(d), axis=2, keepdims=True)
    drift = (np.abs(d - a).sum(axis=2) / 2).mean(axis=0)
    noise_a = (np.abs(a[::2] - a[1::2]).sum(axis=2) / 2).mean(axis=0)
    noise_d = (np.abs(d[::2] - d[1::2]).sum(axis=2) / 2).mean(axis=0)
    return drift[mask], np.maximum(noise_a, noise_d)[mask], info["method"], int(a.size + d.size)


def candidate_pool(partition, manifest, policy, arm):
    check(arm in ARMS, "unknown arm")
    start = time.perf_counter()
    classes, old = manifest["classes"], manifest["old_classes"]
    y, groups = np.asarray(partition["labels"]), np.asarray(partition["group_ids"])
    trigger, ci = policy["trigger"], policy["uncertainty"]
    baseline = fused(partition["new_head_logits"], partition["new_router_raw"], policy["fusion"])
    predicted = np.asarray(classes)[baseline.argmax(axis=1)]
    candidates, diagnostic_cells = [], baseline.size
    for c in old:
        target, mask = classes.index(c), y == c
        record = {"class_id": c, "eligible": False, "priority": None, "reason": None,
                  "trigger_rows": sorted(np.asarray(partition["row_ids"])[mask].tolist())}
        try:
            check(mask.sum() >= trigger["min_rows_per_class"] and len(set(groups[mask])) >= ci["min_groups"], "insufficient trigger support")
            if arm in ("random", "periodic", "audit-only"):
                record.update(eligible=arm != "audit-only", priority=0.0)
                candidates.append(record)
                continue
            path = score_path(partition, classes, old, c, policy["fusion"])
            delta = (predicted[mask] != y[mask]).astype(float) - (path["old_predictions"][mask] != y[mask]).astype(float)
            disagreement = (predicted[mask] != path["old_predictions"][mask]).astype(float)
            metrics = {"harm": (delta, groups[mask], -1, 1), "disagreement": (disagreement, groups[mask], 0, 1)}
            if arm in ATTRIBUTION_ARMS:
                drift, noise, method, cells = _explanation_signal(partition, c, mask)
                check(method == trigger["explanation_method"], "within-method uncertainty mismatch")
                metrics.update(drift=(drift, groups[mask], 0, 1), noise=(noise, groups[mask], 0, 1))
                diagnostic_cells += cells
            intervals = group_intervals(metrics, ci["trigger_alpha"], ci["trigger_family_size"], ci["min_groups"])
            record["intervals"] = intervals
            means = {k: float(v[mask].mean()) for k, v in path["effects"].items()}
            record["decomposition"] = {**means, "identity_max_abs": path["identity_max_abs"], "causal": False}
            harm = intervals["harm"]["lower"]
            if arm == "disagreement-only":
                record.update(eligible=intervals["disagreement"]["lower"] > trigger["min_disagreement"],
                              priority=intervals["disagreement"]["lower"])
            else:
                eligible = harm > trigger["min_error_increase"]
                if arm in ATTRIBUTION_ARMS:
                    record["drift_excess"] = intervals["drift"]["lower"] - intervals["noise"]["upper"]
                    eligible = eligible and record["drift_excess"] > trigger["min_drift_excess"]
                mass = sum(abs(v) for v in means.values())
                share = max(0, -means["state"]) / mass if mass else 0.0
                priority = harm
                if arm in ("expansion-aware", "expansion-no-attribution"):
                    priority *= 1 + trigger["state_priority_weight"] * share
                if arm in ("r1-reachability", "expansion-r1-reachability"):
                    bound = reachability_bound(partition, target, policy["fusion"], policy["r1"])
                    wrong = mask & (predicted != y)
                    reachable = float(bound["can_cross_necessary_bound"][wrong].mean()) if wrong.any() else 0.0
                    record["r1_pointwise_reachable_fraction"] = reachable
                    record["r1_bound_is_not_joint_feasibility"] = True
                    priority = harm * reachable
                    eligible = eligible and reachable > 0
                record.update(eligible=bool(eligible), priority=float(priority), state_share=float(share))
            diagnostic_cells += sum(v.size for v in path["scores"].values())
        except Blocked as exc:
            record["reason"] = str(exc)
        candidates.append(record)
    check({x["class_id"] for x in candidates} == set(old), "incomplete candidate pool")
    available = [x for x in candidates if x["eligible"]]
    available.sort(key=lambda x: x["class_id"])
    if arm == "random":
        np.random.default_rng(np.random.SeedSequence([trigger["random_seed"], manifest["increment"]])).shuffle(available)
    elif arm == "periodic":
        if manifest["increment"] % trigger["periodic_every"] != trigger["periodic_phase"]:
            available = []
        elif available:
            offset = manifest["increment"] // trigger["periodic_every"] % len(available)
            available = available[offset:] + available[:offset]
    else:
        available.sort(key=lambda x: (-x["priority"], x["class_id"]))
    return {"arm": arm, "candidates": candidates, "ordered_targets": [x["class_id"] for x in available],
            "trigger_policy_sha256": digest(trigger), "pool_sha256": digest(candidates),
            "diagnostic_cells_processed": diagnostic_cells, "diagnosis_wall_seconds": time.perf_counter() - start}


def fit_r1(partition, classes, target_class, fusion, settings, progress=None):
    """Fixed-step projected full-batch gradient descent on FIT labels only.

    Scores: sigmoid(a*z_target+b) + frozen router term; all other heads frozen.
    Objective: class-balanced multiclass softmax CE on these fused scores plus
    registered L2 distance to identity. There is one shared (a,b), not per row.
    """
    check(partition["role"] == "fit", "R1 optimizer accepts fit partition only")
    h = finite_matrix(partition["new_head_logits"])
    r = finite_matrix(partition["new_router_raw"], h.shape)
    y = np.asarray([classes.index(c) for c in partition["labels"]])
    target = classes.index(target_class)
    check(set(y.tolist()) == set(range(len(classes))), "fit requires all registered classes")
    counts = np.bincount(y, minlength=len(classes))
    weights = 1.0 / (len(classes) * counts[y])
    fixed = fused(h, r, fusion)
    frozen_other = np.delete(fixed, target, axis=1).copy()
    router_term = fusion["router_weight"] * router_normalize(r, fusion["epsilon"])[:, target]
    z, params = h[:, target], np.asarray([1.0, 0.0])
    lower = np.asarray([settings["scale_min"], settings["offset_min"]])
    upper = np.asarray([settings["scale_max"], settings["offset_max"]])
    wall, cpu = time.perf_counter(), time.process_time()
    own_trace = not tracemalloc.is_tracing()
    if own_trace:
        tracemalloc.start()
    history, completed = [], 0
    try:
        for step in range(settings["steps"]):
            check(time.perf_counter() - wall <= settings["max_fit_seconds"], "fit walltime budget exhausted")
            target_head = sigmoid(params[0] * z + params[1])
            scores = fixed.copy()
            scores[:, target] = target_head + router_term
            shifted = scores - scores.max(axis=1, keepdims=True)
            log_partition = np.log(np.exp(shifted).sum(axis=1))
            ce = -shifted[np.arange(len(y)), y] + log_partition
            loss = float(np.dot(weights, ce) + settings["identity_l2"] * np.sum((params - [1, 0]) ** 2))
            softmax = np.exp(shifted - log_partition[:, None])
            derivative = weights * (softmax[:, target] - (y == target)) * target_head * (1 - target_head)
            gradient = np.asarray([np.dot(derivative, z), derivative.sum()]) + 2 * settings["identity_l2"] * (params - [1, 0])
            check(np.isfinite(loss) and np.isfinite(gradient).all(), "nonfinite fit objective/gradient")
            params = np.clip(params - settings["learning_rate"] * gradient, lower, upper)
            completed += 1
            history.append(loss)
            if progress is not None:
                progress(completed, time.perf_counter() - wall, time.process_time() - cpu)
        repaired = fused(h, r, fusion, target, params)
        check(np.array_equal(np.delete(repaired, target, axis=1), frozen_other), "non-target scores changed")
        result = {"operation": "R1-bounded-target-logit-affine-v1", "target": target_class,
                  "scale": float(params[0]), "offset": float(params[1]), "steps_completed": completed,
                  "fit_rows_processed": completed * len(y), "fit_wall_seconds": time.perf_counter() - wall,
                  "fit_cpu_seconds": time.process_time() - cpu, "loss_history": history,
                  "traced_peak_bytes": int(tracemalloc.get_traced_memory()[1]),
                  "explicit_input_array_bytes": int(h.nbytes + r.nbytes + y.nbytes),
                  "memory_scope": "tracemalloc peak; not process RSS or GPU memory", "optimizer": "projected-full-batch-gradient-descent"}
        result["state_sha256"] = digest({"operation": result["operation"], "target": target_class,
                                         "scale": result["scale"], "offset": result["offset"]})
        return result
    finally:
        if own_trace:
            tracemalloc.stop()


def evaluate_pair(partition, classes, old_classes, fusion, state, policy, phase):
    check(phase in ("acceptance", "subsequent") and partition["role"] == phase, "evaluation partition mismatch")
    wall, cpu = time.perf_counter(), time.process_time()
    y, groups = partition["labels"], partition["group_ids"]
    baseline = fused(partition["new_head_logits"], partition["new_router_raw"], fusion)
    params = (state["scale"], state["offset"])
    repaired = fused(partition["new_head_logits"], partition["new_router_raw"], fusion, classes.index(state["target"]), params)
    bp, rp = np.asarray(classes)[baseline.argmax(axis=1)], np.asarray(classes)[repaired.argmax(axis=1)]
    attack = policy["semantics"]["attack_classes"]
    metrics = paired_metrics(y, bp, rp, groups, classes, old_classes, attack)
    ci = policy["uncertainty"]
    intervals = group_intervals(metrics, ci[f"{phase}_alpha"], ci[f"{phase}_family_size"], ci["min_groups"])
    result = {"baseline": classification_report(y, bp, classes, attack), "proposed": classification_report(y, rp, classes, attack),
              "paired_group_intervals": intervals, "row_ids": partition["row_ids"],
              "baseline_predictions": bp.tolist(), "proposed_predictions": rp.tolist(),
              "baseline_scores": baseline.tolist(), "proposed_scores": repaired.tolist(), "raw_alerts_preserved": True}
    if attack:
        result["baseline_attack_alerts"] = np.isin(bp, attack).tolist()
        result["proposed_attack_alerts"] = np.isin(rp, attack).tolist()
    if phase == "acceptance":
        rule, failures = policy["acceptance"], []
        target_delta = intervals[f"class_error_delta:{state['target']}"]
        if -target_delta["upper"] < rule["min_target_improvement"]:
            failures.append("target_improvement")
        for c in classes:
            allowed = rule["max_old_error_increase"] if c in old_classes else rule["max_new_error_increase"]
            if intervals[f"class_error_delta:{c}"]["upper"] > allowed:
                failures.append(f"class_error_delta:{c}")
        for name, key in (("missed_attack_rate_delta", "max_missed_attack_increase"),
                          ("false_positive_rate_delta", "max_fpr_increase")):
            if attack and intervals[name]["upper"] > rule[key]:
                failures.append(name)
        result.update(decision="REJECT" if failures else "ACCEPT_RESEARCH_VARIANT", failed_constraints=failures,
                      effective_state="baseline" if failures else "proposed")
    result.update(scored_rows=2 * len(y), evaluation_wall_seconds=time.perf_counter() - wall,
                  evaluation_cpu_seconds=time.process_time() - cpu)
    return result
