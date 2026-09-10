"""Fixed-context native scoring adapter; no gradient surrogate or remote actions."""
import hashlib
from importlib.metadata import version
import time
import numpy as np
from pilot_core import require


def array_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def validate_native(scores, rows, axis):
    axis = np.asarray(axis, dtype=np.int64)
    require(len(axis) >= 2 and len(set(axis.tolist())) == len(axis), "invalid native class axis")
    require(np.array_equal(scores["class_axis"], axis), "native axis changed")
    for key in ("head_scores", "router_z_scores", "joint_scores"):
        x = np.asarray(scores[key])
        require(x.dtype == np.float32 and x.shape == (rows, len(axis)) and np.isfinite(x).all(),
                "native float32 score contract failed")
    h, z, joint = (scores[k] for k in ("head_scores", "router_z_scores", "joint_scores"))
    require(np.all((h >= 0) & (h <= 1)), "invalid head probabilities")
    require(np.array_equal(joint, h + np.float32(0.5) * z), "native joint identity failed")
    require(np.array_equal(scores["predicted_class_id"], axis[joint.argmax(axis=1)]), "native argmax changed")
    return scores


class CallBudget:
    """Failed calls are spent. Limits never reset between checkpoints or repeats."""
    def __init__(self, max_calls, max_seconds, synchronize=lambda: None):
        require(type(max_calls) is int and max_calls > 0 and np.isfinite(max_seconds) and max_seconds > 0,
                "positive call/time budgets required")
        self.max_calls, self.max_seconds, self.synchronize = max_calls, max_seconds, synchronize
        self.calls = self.context_rows = 0
        self.start = time.monotonic()

    def run(self, score, raw):
        require(self.calls < self.max_calls and time.monotonic() - self.start < self.max_seconds,
                "native extraction budget exhausted")
        self.calls += 1
        self.context_rows += len(raw)
        result = score(raw)
        self.synchronize()
        require(time.monotonic() - self.start < self.max_seconds, "native call exceeded time budget")
        return result


def score_parent_shards(score, class_shards, axis, selected, budget):
    """Select AFTER scoring full class/shard/512 contexts; never pack selected rows.

    selected keys: (true_class_id, shard_ordinal, shard_local_row).
    Dict iteration order is canonicalized by class ID; supplied shard order is kept.
    """
    wanted = list(selected)
    require(wanted and len(set(wanted)) == len(wanted), "duplicate or empty row selection")
    require(set(class_shards) >= {k[0] for k in wanted}, "missing parent class")
    selected = set(wanted)
    out, boundaries = {}, []
    for cid in sorted(class_shards):
        for shard_id, shard in enumerate(class_shards[cid]):
            require(shard.ndim == 2 and shard.dtype == np.float32 and len(shard) > 0, "parent feature contract")
            for start in range(0, len(shard), 512):
                raw = np.array(shard[start:start+512], dtype=np.float32, order="C", copy=True)
                require(np.isfinite(raw).all(), "nonfinite parent features")
                raw_sha = array_sha(raw)
                scores = validate_native(budget.run(score, raw), len(raw), axis)
                require(array_sha(raw) == raw_sha, "scorer modified parent inputs")
                boundaries.append([int(cid), shard_id, start, len(raw)])
                for local in range(start, start+len(raw)):
                    identity = (cid, shard_id, local)
                    if identity in selected:
                        i = local-start
                        out[identity] = {key: np.array(scores[key][i], copy=True)
                                         for key in ("head_scores", "router_z_scores", "joint_scores", "predicted_class_id")}
    require(set(out) == set(wanted), "selected row outside parent shard")
    return out, boundaries


class FixedContextTarget:
    def __init__(self, score, parent, row, axis, target, budget):
        require(parent.ndim == 2 and parent.dtype == np.float32 and 0 < len(parent) <= 512
                and np.isfinite(parent).all(), "one finite native parent context required")
        require(type(row) is int and 0 <= row < len(parent) and target in axis, "invalid target row/class")
        self.parent = np.array(parent, copy=True, order="C")
        self.parent.flags.writeable = False
        self.row, self.axis, self.target = row, list(axis), int(target)
        self.score, self.budget = score, budget
        self.context_sha256 = array_sha(self.parent)

    def __call__(self, replacements):
        xs = np.asarray(replacements)
        require(xs.ndim == 2 and xs.shape[1] == self.parent.shape[1]
                and xs.dtype in (np.float32, np.float64) and np.isfinite(xs).all(), "masked feature contract")
        # SHAP's delta masker promotes some float32 arrays to float64. Permit
        # representation-only promotion, never rounding a new feature value.
        with np.errstate(over="ignore", invalid="ignore"):
            native_xs = xs.astype(np.float32)
        require(np.isfinite(native_xs).all() and np.array_equal(xs, native_xs.astype(xs.dtype)),
                "masked features not losslessly float32 representable")
        xs = native_xs
        values = []
        for replacement in xs:
            raw = self.parent.copy()
            raw[self.row] = replacement
            before = array_sha(raw)
            scores = validate_native(self.budget.run(self.score, raw), len(raw), self.axis)
            require(array_sha(raw) == before, "scorer modified masked context")
            j = self.axis.index(self.target)
            native = scores["joint_scores"][self.row]
            # Subtraction occurs in native float32, then scalar promoted for SHAP.
            values.append(float(native[j] - np.max(np.delete(native, j))))
        require(array_sha(self.parent) == self.context_sha256, "frozen context changed")
        return np.asarray(values, dtype=np.float64)


def permutation_repeats(target, references, seeds, expected_version="0.51.0"):
    """Published SHAP implementation; this adapter is not a new SHAP algorithm."""
    require(version("shap") == expected_version, "SHAP version differs from reviewed binding")
    require(len(seeds) >= 4 and len(seeds) % 2 == 0 and len(set(seeds)) == len(seeds)
            and all(type(s) is int and 0 <= s < 2**32 for s in seeds), "independent repeat seeds required")
    references = np.asarray(references)
    require(references.ndim == 2 and references.dtype == np.float32 and 0 < len(references) <= 32
            and references.shape[1] == target.parent.shape[1] and np.isfinite(references).all(), "reference pool contract")
    import shap
    x = target.parent[target.row:target.row+1].copy()
    reference_sha = array_sha(references)
    class ExactIndependent(shap.maskers.Independent):
        def invariants(self, value):
            require(np.asarray(value).shape == (self.data.shape[1],), "masker input width changed")
            # The default isclose may skip a genuinely different float32 value.
            return np.equal(value, self.data)
    masker = ExactIndependent(references.copy(), max_samples=len(references))
    require(np.array_equal(masker.data, references), "masker reference pool changed")
    before = target.budget.calls
    exact = target(x)[0]
    values, residuals, bases = [], [], []
    for seed in seeds:
        explainer = shap.PermutationExplainer(target, masker, seed=seed)
        result = explainer(x, max_evals=2*x.shape[1]+1, batch_size=1, silent=True)
        v = np.asarray(result.values, dtype=np.float64)
        base = np.asarray(result.base_values, dtype=np.float64).reshape(-1)
        require(v.shape == x.shape and base.shape == (1,) and np.isfinite(v).all()
                and np.isfinite(base).all(), "invalid SHAP result")
        residual = float(base[0] + v[0].sum() - exact)
        require(abs(residual) <= 1e-6 + 1e-6*abs(exact), "SHAP additive reconstruction failed")
        values.append(v[0].copy()); residuals.append(residual); bases.append(float(base[0]))
    require(array_sha(references) == reference_sha and np.array_equal(masker.data, references), "references mutated")
    return {"attributions": np.stack(values), "base_values": bases, "residuals": residuals,
            "native_target": float(exact), "context_sha256": target.context_sha256,
            "reference_sha256": reference_sha, "shap_version": expected_version,
            "masker_invariance": "exact-value-equality; no isclose skipping",
            "native_calls": target.budget.calls-before, "statistical_certificate": False}


def capture_native(model, raw):
    """Capture logits from the SAME native score call, not a second forward pass."""
    require(raw.ndim == 2 and raw.dtype == np.float32 and 0 < len(raw) <= 512
            and np.isfinite(raw).all(), "one native context required for capture")
    captured, handles = {}, []
    axis = list(model.metadata["seen_classes"])
    def capture(cid):
        def hook(module, inputs, output):
            require(cid not in captured, "head invoked twice within a native context")
            captured[cid] = output.detach().cpu().numpy().copy()
        return hook
    try:
        for cid in axis:
            handles.append(model.heads[cid].register_forward_hook(capture(cid)))
        scores = model.score(raw)
        validate_native(scores, len(raw), axis)
        require(set(captured) == set(axis), "native logit capture incomplete")
        logits = np.stack([captured[cid] for cid in axis], axis=1)
        require(logits.dtype == np.float32 and logits.shape == (len(raw), len(axis), 2)
                and np.isfinite(logits).all(), "binary logit contract")
        return dict(scores, binary_logits=logits)
    finally:
        for handle in handles:
            handle.remove()
