"""Naive sequential fine-tuning — lower bound. No anti-forgetting mechanism."""
import numpy as np

from src.config import EPOCHS_PER_TASK
from src.methods.base import build_model, train_one_task, subset_by_classes
from src.metrics import evaluate_task


def run_naive(X_tr, y_tr, X_te, y_te, tasks, in_dim, n_classes):
    """Train each task sequentially with only its own data. No defenses."""
    model = build_model(in_dim, n_classes)
    acc_matrix = np.full((len(tasks), len(tasks)), np.nan)
    for i, task_i in enumerate(tasks):
        Xi, yi = subset_by_classes(X_tr, y_tr, task_i)
        train_one_task(model, Xi, yi, epochs=EPOCHS_PER_TASK)
        for j, task_j in enumerate(tasks[: i + 1]):
            acc_matrix[i, j] = evaluate_task(model, X_te, y_te, task_j)
    return acc_matrix, model
