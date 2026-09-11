"""Single-row SHAP feasibility; not a change, noise, or efficacy estimate."""
import time
import numpy as np
from importlib.metadata import version
from native_target import FixedContextTarget, array_sha
from pilot_core import require


def extract_one(target, references, seed, progress=lambda x: None):
    require(version("shap") == "0.51.0", "bound SHAP version required")
    require(type(seed) is int and 0 <= seed < 2**32, "invalid seed")
    refs = np.asarray(references)
    require(refs.dtype == np.float32 and refs.ndim == 2 and 0 < len(refs) <= 32
            and refs.shape[1] == target.parent.shape[1] and np.isfinite(refs).all(),
            "reference contract")
    import shap
    class ExactIndependent(shap.maskers.Independent):
        def invariants(self, x):
            require(np.asarray(x).shape == (self.data.shape[1],), "masker width changed")
            return np.equal(x, self.data)
    start = time.monotonic()
    before = target.budget.calls
    original_refs = array_sha(refs)
    x = target.parent[target.row:target.row+1].copy()
    direct = float(target(x)[0])
    masker = ExactIndependent(refs.copy(), max_samples=len(refs))
    require(np.array_equal(masker.data, refs), "reference pool changed")
    progress({"stage": "SHAP_START", "native_calls": target.budget.calls})
    explainer = shap.PermutationExplainer(target, masker, seed=seed)
    result = explainer(x, max_evals=2*x.shape[1]+1, batch_size=1, silent=True)
    values = np.asarray(result.values, dtype=np.float64)
    bases = np.asarray(result.base_values, dtype=np.float64).reshape(-1)
    require(values.shape == x.shape and bases.shape == (1,) and
            np.isfinite(values).all() and np.isfinite(bases).all(), "invalid attribution")
    residual = float(bases[0] + values[0].sum() - direct)
    require(abs(residual) <= 1e-6 + 1e-6*abs(direct), "reconstruction failed")
    require(array_sha(refs) == original_refs and np.array_equal(masker.data, refs),
            "references mutated")
    require(float(target(x)[0]) == direct, "post-extraction native target changed")
    return {"attributions": values[0].tolist(), "base_value": float(bases[0]),
            "native_target": direct, "additive_residual": residual,
            "l1_norm": float(np.abs(values).sum()), "nonzero_features": int(np.count_nonzero(values)),
            "reference_sha256": original_refs, "context_sha256": target.context_sha256,
            "native_calls": target.budget.calls-before, "elapsed_seconds": time.monotonic()-start,
            "shap_version": "0.51.0", "seed": seed,
            "max_evals": 2*x.shape[1]+1, "batch_size": 1,
            "masker_invariance": "exact-value-equality", "repeat_count": 1,
            "zero_vector_means_feasibility_only": True,
            "statistical_certificate": False}
