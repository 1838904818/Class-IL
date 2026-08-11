"""Replay with random reservoir sampling.

Keep up to N exemplars per class chosen uniformly at random. On every new task,
mix the buffer back into the training data.
"""
import numpy as np

from src.config import EPOCHS_PER_TASK, REPLAY_BUFFER_PER_CLASS
from src.methods.base import build_model, train_one_task, subset_by_classes
from src.metrics import evaluate_task


def run_replay(X_tr, y_tr, X_te, y_te, tasks, in_dim, n_classes,
               buffer_per_class: int = REPLAY_BUFFER_PER_CLASS):
    """Random reservoir per-class buffer, default 200 samples per class."""
    model = build_model(in_dim, n_classes)
    acc_matrix = np.full((len(tasks), len(tasks)), np.nan)
    buf_X, buf_y = [], []

    for i, task_i in enumerate(tasks):
        Xi, yi = subset_by_classes(X_tr, y_tr, task_i)

        # Mix in past exemplars
        if buf_X:
            X_combined = np.vstack([Xi] + buf_X)
            y_combined = np.concatenate([yi] + buf_y)
        else:
            X_combined, y_combined = Xi, yi

        train_one_task(model, X_combined, y_combined, epochs=EPOCHS_PER_TASK)

        # Add per-class exemplars to buffer (random subsampling)
        for c in task_i:
            mask = (yi == c)
            if mask.sum() == 0:
                continue
            idx = np.random.choice(
                np.where(mask)[0],
                size=min(buffer_per_class, mask.sum()),
                replace=False,
            )
            buf_X.append(Xi[idx])
            buf_y.append(yi[idx])

        for j, task_j in enumerate(tasks[: i + 1]):
            acc_matrix[i, j] = evaluate_task(model, X_te, y_te, task_j)
    return acc_matrix, model
