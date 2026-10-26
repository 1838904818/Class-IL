"""Section156 model primitives; trajectory integration is required before use."""
import math

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier


def fit_histgb(x, y, *, stopping_x, stopping_y, seed, training_ids,
               stopping_ids, calibration_ids):
    """Fit only against the explicit registered stopping split (sklearn >=1.7)."""
    training_ids, stopping_ids, calibration_ids = map(np.asarray,
        (training_ids, stopping_ids, calibration_ids))
    for a, b in ((training_ids, stopping_ids), (training_ids, calibration_ids),
                 (stopping_ids, calibration_ids)):
        if np.intersect1d(a, b).size:
            raise ValueError("HistGB data roles overlap")
    if len(x) != len(training_ids) or len(stopping_x) != len(stopping_ids):
        raise ValueError("HistGB row identities do not match arrays")
    if not len(stopping_ids) or len(np.unique(y)) < 2:
        raise ValueError("HistGB needs stopping rows and at least two classes")
    model = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.1,
        max_leaf_nodes=63, l2_regularization=1.0, class_weight="balanced",
        early_stopping=True, random_state=seed)
    model.fit(x, y, X_val=stopping_x, y_val=stopping_y)
    return model


def partial_auc_penalty(logits, labels):
    """Registered batch pair surrogate; CE balancing belongs to the caller."""
    scores = logits[:, 1] - logits[:, 0]
    benign, attack = scores[labels == 0], scores[labels == 1]
    if not benign.numel() or not attack.numel():
        return scores.sum() * 0.0
    hard = benign.topk(math.ceil(0.05 * benign.numel())).values
    return torch.nn.functional.softplus(hard[:, None] - attack[None, :]).mean()


class EnsembleGate(torch.nn.Module):
    """Five independent gates; average margins, not average probabilities."""
    def __init__(self, members):
        super().__init__()
        if len(members) != 5:
            raise ValueError("The registered ensemble has five members")
        self.members = torch.nn.ModuleList(members)

    def forward(self, x):
        margins = [m(x).diff(dim=1).squeeze(1) for m in self.members]
        score = torch.stack(margins).mean(0)
        return torch.stack((-score / 2, score / 2), dim=1)


def select_benign_hard(ids, scores, *, stopping_ids, calibration_ids, limit=2000):
    """Select from the actual fresh draw only, with deterministic tie handling."""
    ids, scores = np.asarray(ids), np.asarray(scores)
    if ids.ndim != 1 or scores.shape != ids.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid benign scores")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate fresh benign identities")
    if np.intersect1d(ids, stopping_ids).size or np.intersect1d(ids, calibration_ids).size:
        raise ValueError("Benign-hard pool overlaps held-out roles")
    return ids[np.lexsort((ids, -scores))[:limit]]


def two_key_log_score(gate_attack_probability, family_benign_probability):
    p, q = np.asarray(gate_attack_probability), np.asarray(family_benign_probability)
    if p.shape != q.shape or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Invalid two-key probabilities")
    if ((p < 0) | (p > 1) | (q < 0) | (q > 1)).any():
        raise ValueError("Probabilities outside [0,1]")
    with np.errstate(divide="ignore"):
        return np.log(p) + np.log1p(-q)
