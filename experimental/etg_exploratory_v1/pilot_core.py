"""Pure descriptive selection helpers. No execution driver or safety certificate."""
import hashlib
import json
import numpy as np

ARMS = ("audit-only", "error-only", "raw-drift", "noise-aware", "random-harmful")
ROLES = ("trigger", "fit", "acceptance", "evaluation")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def partition_indices(calibration, fitting, class_id, salt, counts):
    """Partition native calibration identities, never a same-sized substitute."""
    cal, fit = [int(i) for i in calibration], [int(i) for i in fitting]
    require(len(cal) == len(set(cal)) and len(fit) == len(set(fit)), "duplicate source offset")
    require(not set(cal).intersection(fit), "training/calibration overlap")
    require(tuple(counts) == ROLES, "role order must be fixed")
    require(all(type(n) is int and n > 0 for n in counts.values()), "invalid counts")
    require(sum(counts.values()) <= len(cal), "insufficient class support")
    order = sorted(cal, key=lambda i: (digest([salt, class_id, i]), i))
    out, pos = {}, 0
    for role, count in counts.items():
        out[role] = sorted(order[pos:pos + count])
        pos += count
    return out


def explanation_signal(old, new, repeats=8, quantile=0.9):
    """Signed L1-normalized distances; empirical noise floor, not an interval."""
    a, b = np.asarray(old, dtype=float), np.asarray(new, dtype=float)
    require(a.shape == b.shape and a.ndim == 3 and a.shape[0] == repeats
            and repeats >= 4 and repeats % 2 == 0 and min(a.shape[1:]) > 0,
            "expected matched [even repeats, rows, features]")
    require(np.isfinite(a).all() and np.isfinite(b).all(), "nonfinite attribution")
    require(0 <= quantile <= 1, "invalid noise quantile")
    norms_a, norms_b = np.abs(a).sum(-1, keepdims=True), np.abs(b).sum(-1, keepdims=True)
    require(np.isfinite(norms_a).all() and np.isfinite(norms_b).all()
            and np.all(norms_a > 0) and np.all(norms_b > 0), "degenerate attribution")
    a, b = a / norms_a, b / norms_b
    between = np.median(0.5 * np.abs(a - b).sum(-1), axis=0)
    na = np.quantile(0.5 * np.abs(a[::2] - a[1::2]).sum(-1), quantile, axis=0)
    nb = np.quantile(0.5 * np.abs(b[::2] - b[1::2]).sum(-1), quantile, axis=0)
    noise = np.maximum(na, nb)
    return {"between": float(np.median(between)), "noise": float(np.median(noise)),
            "excess": float(np.median(between - noise)), "rows": int(a.shape[1]),
            "statistical_certificate": False}


def choose(candidates, arm, salt, threshold=0.05):
    """Equal harm priority; explanation gates eligibility, not score weighting."""
    require(arm in ARMS and np.isfinite(threshold) and 0 <= threshold <= 1, "invalid arm or threshold")
    require(len({c["class_id"] for c in candidates}) == len(candidates), "duplicate target")
    for c in candidates:
        require(type(c["class_id"]) is int and np.isfinite(c["error_increase"])
                and -1 <= c["error_increase"] <= 1, "invalid performance evidence")
    if arm == "audit-only":
        return {"status": "audit-only", "target": None}
    harmful = [c for c in candidates if c["error_increase"] > 0]
    if arm in ("raw-drift", "noise-aware"):
        # Unavailable evidence is not zero drift or an affirmative abstention.
        if any(not c.get("attribution_validated", False) for c in harmful):
            return {"status": "unavailable-attribution", "target": None}
        key = "between" if arm == "raw-drift" else "excess"
        lower = 0 if arm == "raw-drift" else -1
        require(all(np.isfinite(c[key]) and lower <= c[key] <= 1 for c in harmful),
                "invalid explanation evidence")
        harmful = [c for c in harmful if c[key] > threshold]
    if not harmful:
        return {"status": "abstain", "target": None}
    if arm == "random-harmful":
        chosen = min(harmful, key=lambda c: (digest([salt, "random-control", c["class_id"]]), c["class_id"]))
    else:
        chosen = min(harmful, key=lambda c: (-c["error_increase"], c["class_id"]))
    return {"status": "attempt", "target": chosen["class_id"]}


def empirical_accept(labels, baseline, proposed, target, classes, attack_classes):
    """Small-sample empirical rollback rule; never population risk control."""
    y, a, b = [np.asarray(x) for x in (labels, baseline, proposed)]
    require(y.ndim == 1 and y.shape == a.shape == b.shape and len(y) > 0, "unaligned predictions")
    require(len(set(classes)) == len(classes) and target in classes, "invalid class axis")
    require(set(y.tolist()) == set(classes) and set(a.tolist()) <= set(classes)
            and set(b.tolist()) <= set(classes), "missing or unknown classes")
    require(len(set(attack_classes)) == len(attack_classes) and
            set(attack_classes) < set(classes) and len(attack_classes) == len(classes)-1,
            "explicit one-Benign attack semantics required")
    gains = {int(c): int(np.sum((y == c) & (b == c)) - np.sum((y == c) & (a == c))) for c in classes}
    attack = np.isin(y, attack_classes)
    misses = int(np.sum(attack & ~np.isin(b, attack_classes)) - np.sum(attack & ~np.isin(a, attack_classes)))
    fpr = int(np.sum(~attack & np.isin(b, attack_classes)) - np.sum(~attack & np.isin(a, attack_classes)))
    passed = gains[target] > 0 and min(gains.values()) >= 0 and misses <= 0 and fpr <= 0
    return {"accepted_empirically": bool(passed), "class_correct_count_changes": gains,
            "attack_miss_count_change": misses, "benign_fp_count_change": fpr,
            "statistical_certificate": False}
