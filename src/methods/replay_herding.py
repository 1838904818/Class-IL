"""Replay with herding (Welling 2009) exemplar selection.

Herding greedily selects exemplars so the running mean of selected examples
best approximates the per-class feature mean (in the model's embedding space).

This is the D11 ablation baseline that isolates the effect of exemplar selection
strategy. Combined with iCaRL's KD, it becomes the full iCaRL method.
"""
import numpy as np
import torch

from src.config import EPOCHS_PER_TASK, REPLAY_BUFFER_PER_CLASS
from src.methods.base import build_model, train_one_task, subset_by_classes
from src.methods.icarl import _herding_select   # shared utility
from src.metrics import evaluate_task


def run_replay_herding(X_tr, y_tr, X_te, y_te, tasks, in_dim, n_classes,
                       buffer_per_class: int = REPLAY_BUFFER_PER_CLASS):
    """Replay with herding exemplar selection (no KD, no NME — ablation baseline)."""
    model = build_model(in_dim, n_classes)
    acc_matrix = np.full((len(tasks), len(tasks)), np.nan)
    buf_X: list[np.ndarray] = []
    buf_y: list[np.ndarray] = []

    for i, task_i in enumerate(tasks):
        Xi, yi = subset_by_classes(X_tr, y_tr, task_i)

        if buf_X:
            X_combined = np.vstack([Xi] + buf_X)
            y_combined = np.concatenate([yi] + buf_y)
        else:
            X_combined, y_combined = Xi, yi

        train_one_task(model, X_combined, y_combined, epochs=EPOCHS_PER_TASK)

        # Herding exemplar selection
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
            raw_idx_c = np.where(mask)[0]
            sel_local = _herding_select(feats_c, buffer_per_class)
            sel_global = raw_idx_c[sel_local]
            buf_X.append(Xi[sel_global])
            buf_y.append(yi[sel_global])

        for j, task_j in enumerate(tasks[: i + 1]):
            acc_matrix[i, j] = evaluate_task(model, X_te, y_te, task_j)

    return acc_matrix, model
