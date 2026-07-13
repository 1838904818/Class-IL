"""Joint training — upper bound. NOT a CL method.

All data is presented at once, equivalent to training time of len(tasks)*EPOCHS_PER_TASK.
"""
import numpy as np

from src.config import EPOCHS_PER_TASK
from src.methods.base import build_model, train_one_task
from src.metrics import evaluate_task


def run_joint(X_tr, y_tr, X_te, y_te, tasks, in_dim, n_classes):
    """Train once on all classes; report final accuracy per task on the last row."""
    model = build_model(in_dim, n_classes)
    train_one_task(model, X_tr, y_tr, epochs=EPOCHS_PER_TASK * len(tasks))
    acc_matrix = np.full((len(tasks), len(tasks)), np.nan)
    for j, task_j in enumerate(tasks):
        acc_matrix[-1, j] = evaluate_task(model, X_te, y_te, task_j)
    return acc_matrix, model
