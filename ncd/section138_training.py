"""G-only schedule dispatch; F keeps the existing fixed DER training schedule."""
import numpy as np

from ncd.crr_benchmarks import _class_balanced_cross_entropy
from ncd.run_aacil_retention import train_gate
from ncd.section123_es_training import assert_disjoint_roles, train_es


def train_gate_schedule(model, x, y, *, schedule, config, seed, device,
                        training_ids, stopping_ids, calibration_ids,
                        stopping_x=None, stopping_y=None):
    """Caller supplies bound TRAIN roles; no retention targets enter G.

    ID vectors refer to the same original TRAIN index space, not separate
    reindexed subsets. The caller remains responsible for exposure construction.
    """
    if len(training_ids) != len(y) or len(x) != len(y):
        raise ValueError("Gate optimization role length mismatch")
    for ids in (training_ids, stopping_ids, calibration_ids):
        ids = np.asarray(ids)
        if (ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0)
                or len(np.unique(ids)) != len(ids)):
            raise ValueError("Gate role IDs must be unique nonnegative integer vectors")
    if not len(y) or np.asarray(y).dtype.kind not in "iu" or not np.isin(y, [0, 1]).all():
        raise ValueError("Nonempty binary gate labels required")
    if float(config.get("lambda_g", 0)) != 0:
        raise ValueError("Section138 gate must have no retention")
    if schedule == "fixed":
        train_ids = np.asarray(training_ids)
        cal_ids = np.asarray(calibration_ids)
        if np.intersect1d(train_ids, cal_ids).size:
            raise ValueError("Gate optimization overlaps calibration")
        elapsed, stats = train_gate(model, x, y, np.zeros((len(y), 2), dtype=np.float32),
                                   np.zeros(len(y), dtype=bool), config, seed, device, 0.)
        stats.update(schedule="fixed", gate_retention=False)
        return elapsed, stats
    if schedule != "section123_es":
        raise ValueError("Unregistered gate schedule")
    assert_disjoint_roles(training_ids, stopping_ids, calibration_ids)
    if stopping_y is None or stopping_x is None or len(stopping_ids) != len(stopping_y):
        raise ValueError("Gate stopping role length mismatch")
    stats = train_es(model, x, y, stopping_x, stopping_y,
                    objective=lambda m, bx, by: _class_balanced_cross_entropy(m(bx), by),
                    logits=lambda m, bx: m(bx), seen_classes=(0, 1), seed=seed,
                    device=device, config=config, smoke=bool(config.get("smoke", False)))
    stats.update(schedule="section123_es", gate_retention=False,
                 family_schedule_unchanged="fixed")
    return stats["elapsed_seconds"], stats
