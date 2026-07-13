"""iCaRL — Incremental Classifier and Representation Learning (Rebuffi et al. 2017).

Three-part contribution:
  1. Herding exemplar selection — greedily picks representatives whose running mean
     best approximates the per-class feature mean.
  2. Knowledge distillation — old model's softmax outputs act as soft targets,
     preventing forgetting of previously learned class boundaries.
  3. Nearest-Mean-of-Exemplars (NME) classifier — inference via nearest prototype
     in feature space rather than the softmax head.

This is the D6 implementation (replacing the stub that forwarded to random Replay).
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F

from src.config import (
    EPOCHS_PER_TASK, REPLAY_BUFFER_PER_CLASS, LR,
    ICARL_T, ICARL_USE_NME,
)
from src.methods.base import (
    build_model, train_one_task, to_loader, subset_by_classes,
)


# ---------------------------------------------------------------------------
# Herding exemplar selection
# ---------------------------------------------------------------------------

def _herding_select(features: np.ndarray, k: int) -> np.ndarray:
    """Greedy herding: pick k indices so the running mean ≈ class feature mean.

    Args:
        features: (N, D) — extracted feature vectors for one class.
        k:        how many exemplars to keep.
    Returns:
        Array of k indices into `features`.
    """
    k = min(k, len(features))
    mu = features.mean(axis=0)          # class mean
    selected = []
    current_sum = np.zeros_like(mu)
    remaining = list(range(len(features)))

    for _ in range(k):
        # Candidate running means after adding each remaining sample
        candidate_means = (current_sum + features[remaining]) / (len(selected) + 1)
        # Pick the candidate that brings the running mean closest to mu
        dists = np.linalg.norm(candidate_means - mu, axis=1)
        best_local = int(np.argmin(dists))
        best_global = remaining[best_local]
        selected.append(best_global)
        current_sum += features[best_global]
        remaining.pop(best_local)
    return np.array(selected)


# ---------------------------------------------------------------------------
# NME evaluation helper
# ---------------------------------------------------------------------------

def _nme_predict(model, X: np.ndarray, class_means: dict) -> np.ndarray:
    """Nearest-Mean-of-Exemplars classifier.

    Args:
        model:        trained MLP.
        X:            (N, F) test features.
        class_means:  {class_idx: mean_feature_vector (D,)}.
    Returns:
        (N,) predicted class indices.
    """
    model.eval()
    with torch.no_grad():
        feats = model.extract_features(
            torch.from_numpy(X.astype(np.float32))
        ).numpy()  # (N, D)

    classes = sorted(class_means.keys())
    means = np.stack([class_means[c] for c in classes])  # (K, D)
    # L2 distance to each class mean; normalise feature vectors (optional)
    feats_n = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    means_n = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-8)
    dists = np.linalg.norm(feats_n[:, None, :] - means_n[None, :, :], axis=2)
    local_pred = np.argmin(dists, axis=1)
    return np.array([classes[i] for i in local_pred])


def _evaluate_task_nme(model, X, y, task_classes, class_means):
    """Task accuracy using NME classifier (for iCaRL)."""
    mask = np.isin(y, task_classes)
    if mask.sum() == 0:
        return float("nan")
    preds = _nme_predict(model, X[mask], class_means)
    return float((preds == y[mask]).mean())


# ---------------------------------------------------------------------------
# KD loss (distillation from frozen old model)
# ---------------------------------------------------------------------------

def _kd_loss(model, old_model, bx, old_classes, temperature=ICARL_T):
    """Soft cross-entropy distillation loss on old class logits."""
    if old_model is None or len(old_classes) == 0:
        return torch.tensor(0.0)
    with torch.no_grad():
        old_logits = old_model(bx)[:, old_classes]   # (B, K_old)
        soft_targets = F.softmax(old_logits / temperature, dim=1)
    new_logits = model(bx)[:, old_classes]
    log_probs = F.log_softmax(new_logits / temperature, dim=1)
    return F.kl_div(log_probs, soft_targets, reduction="batchmean") * (temperature ** 2)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_icarl(X_tr, y_tr, X_te, y_te, tasks, in_dim, n_classes,
              buffer_per_class: int = REPLAY_BUFFER_PER_CLASS):
    """Full iCaRL: herding exemplars + KD training + NME inference."""
    model = build_model(in_dim, n_classes)
    acc_matrix = np.full((len(tasks), len(tasks)), np.nan)

    buf_X: list[np.ndarray] = []
    buf_y: list[np.ndarray] = []
    class_means: dict[int, np.ndarray] = {}   # updated after each task
    old_model = None
    seen_classes: list[int] = []

    for i, task_i in enumerate(tasks):
        Xi, yi = subset_by_classes(X_tr, y_tr, task_i)

        # ---- (a) Assemble combined train set (new task + old exemplars) ----
        if buf_X:
            X_comb = np.vstack([Xi] + buf_X)
            y_comb = np.concatenate([yi] + buf_y)
        else:
            X_comb, y_comb = Xi, yi

        # ---- (b) Train with CE + KD ----------------------------------------
        old_classes_seen = list(seen_classes)  # classes from *previous* tasks

        def extra_loss(model_ref, bx, _by):
            return _kd_loss(model_ref, old_model, bx, old_classes_seen)

        train_one_task(model, X_comb, y_comb, epochs=EPOCHS_PER_TASK,
                       extra_loss_fn=extra_loss if old_model is not None else None)

        # ---- (c) Freeze old model for next task's KD -----------------------
        old_model = copy.deepcopy(model)
        old_model.eval()
        for p in old_model.parameters():
            p.requires_grad_(False)

        # ---- (d) Herding exemplar selection for new task classes -----------
        model.eval()
        with torch.no_grad():
            feats_all = model.extract_features(
                torch.from_numpy(Xi.astype(np.float32))
            ).numpy()

        for c in task_i:
            mask = (yi == c)
            if mask.sum() == 0:
                continue
            feats_c = feats_all[mask]
            idx_local = _herding_select(feats_c, buffer_per_class)
            Xi_c_raw = Xi[np.where(mask)[0][idx_local]]
            yi_c_raw = yi[np.where(mask)[0][idx_local]]
            buf_X.append(Xi_c_raw)
            buf_y.append(yi_c_raw)

        # ---- (e) Update class means (all seen classes, using exemplars) ----
        for c in task_i:
            seen_classes.append(c)
        # Recompute means for ALL seen classes from their exemplar buffers
        all_buf_X = np.vstack(buf_X) if buf_X else np.empty((0, in_dim))
        all_buf_y = np.concatenate(buf_y) if buf_y else np.empty(0, dtype=np.int64)
        if len(all_buf_X) == 0:
            continue  # no exemplars yet (shouldn't happen after task 0)
        model.eval()
        with torch.no_grad():
            feats_buf = model.extract_features(
                torch.from_numpy(all_buf_X.astype(np.float32))
            ).numpy()
        for c in seen_classes:
            mask = (all_buf_y == c)
            if mask.sum() > 0:
                mu = feats_buf[mask].mean(axis=0)
                mu = mu / (np.linalg.norm(mu) + 1e-8)
                class_means[c] = mu

        # ---- (f) Evaluate all seen tasks -----------------------------------
        for j, task_j in enumerate(tasks[: i + 1]):
            if ICARL_USE_NME:
                acc_matrix[i, j] = _evaluate_task_nme(
                    model, X_te, y_te, task_j, class_means)
            else:
                from src.metrics import evaluate_task
                acc_matrix[i, j] = evaluate_task(model, X_te, y_te, task_j)

    return acc_matrix, model
