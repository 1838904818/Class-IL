"""Native-anchored R1: original float32 scores stay exact at identity."""
import time
import numpy as np
from pilot_core import require
from native_target import validate_native


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1 / (1 + np.exp(-x[positive]))
    e = np.exp(x[~positive]); out[~positive] = e / (1 + e)
    return out


def validate_settings(s):
    require(set(s) == {"scale_min", "scale_max", "offset_min", "offset_max", "steps",
                       "learning_rate", "identity_l2", "max_seconds"}, "R1 settings incomplete")
    require(all(type(v) in (int, float) and np.isfinite(v) for v in s.values()), "invalid settings")
    require(0 < s["scale_min"] <= 1 <= s["scale_max"] and
            s["offset_min"] <= 0 <= s["offset_max"] and type(s["steps"]) is int
            and 0 < s["steps"] <= 10000 and s["learning_rate"] > 0
            and s["identity_l2"] >= 0 and s["max_seconds"] > 0, "R1 bound contract")


def adjust(native, target, params):
    axis = list(np.asarray(native["class_axis"]).tolist())
    rows = len(native["head_scores"])
    validate_native(native, rows, axis)
    require(target in axis and len(params) == 2 and np.isfinite(params).all() and params[0] > 0,
            "invalid R1 state")
    j = axis.index(target)
    logits = np.asarray(native["binary_logits"])
    require(logits.dtype == np.float32 and logits.shape == (rows, len(axis), 2)
            and np.isfinite(logits).all(), "captured logits required")
    # Do not recompute identity through float64 sigmoid or Router normalization.
    if params[0] == 1 and params[1] == 0:
        return native["joint_scores"].copy()
    delta = logits[:, j, 1].astype(np.float64) - logits[:, j, 0].astype(np.float64)
    with np.errstate(over="raise", invalid="raise"):
        p = sigmoid(params[0]*delta + params[1]).astype(np.float32)
    out = native["joint_scores"].copy()
    out[:, j] = p + np.float32(0.5)*native["router_z_scores"][:, j]
    require(np.isfinite(out).all(), "nonfinite R1 score")
    require(np.array_equal(np.delete(out, j, axis=1), np.delete(native["joint_scores"], j, axis=1)),
            "non-target scores changed")
    return out


def fit(native, labels, target, settings, role):
    """Fixed projected steps; smooth float64 optimization, native-quantized output."""
    require(role == "fit", "only fitting rows may optimize R1")
    validate_settings(settings)
    axis = list(np.asarray(native["class_axis"]).tolist())
    y = np.asarray(labels)
    require(y.ndim == 1 and len(y) == len(native["head_scores"]) and set(y.tolist()) == set(axis),
            "fit needs all registered classes")
    # Identity validation and logit shape checks occur before optimizer access.
    fixed = adjust(native, target, [1.0, 0.0]).astype(np.float64)
    j = axis.index(target)
    yi = np.asarray([axis.index(c) for c in y])
    counts = np.bincount(yi, minlength=len(axis))
    weights = 1.0 / (len(axis)*counts[yi])
    logits = native["binary_logits"]
    z = logits[:, j, 1].astype(np.float64) - logits[:, j, 0].astype(np.float64)
    router = (np.float32(0.5)*native["router_z_scores"][:, j]).astype(np.float64)
    theta = np.array([1., 0.])
    low = np.array([settings["scale_min"], settings["offset_min"]])
    high = np.array([settings["scale_max"], settings["offset_max"]])
    start = time.monotonic()
    for _ in range(settings["steps"]):
        require(time.monotonic()-start < settings["max_seconds"], "fit budget exhausted")
        p = sigmoid(theta[0]*z+theta[1])
        s = fixed.copy(); s[:, j] = p + router
        s -= s.max(axis=1, keepdims=True)
        q = np.exp(s); q /= q.sum(axis=1, keepdims=True)
        d = weights*(q[:, j]-(yi == j))*p*(1-p)
        grad = np.array([np.dot(d, z), d.sum()]) + 2*settings["identity_l2"]*(theta-[1, 0])
        require(np.isfinite(grad).all(), "nonfinite fit gradient")
        theta = np.clip(theta-settings["learning_rate"]*grad, low, high)
    require(time.monotonic()-start < settings["max_seconds"], "fit budget exhausted")
    return {"target": int(target), "params": theta.tolist(), "steps": settings["steps"],
            "wall_seconds": time.monotonic()-start,
            "objective": "smooth float64 surrogate over frozen native non-target scores",
            "evaluation": "native-anchored float32; identity bypass is bit-exact",
            "model_weights_modified": False, "calibration_parameters_fitted": True}
