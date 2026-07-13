"""Learning without Forgetting (Li & Hoiem 2016).

Keep a frozen copy of the model after each task; on the next task, distill
soft predictions of the old model into the new model on the old-class outputs.
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F

from src.config import EPOCHS_PER_TASK, LWF_T, LWF_ALPHA
from src.methods.base import build_model, train_one_task, subset_by_classes
from src.metrics import evaluate_task


def run_lwf(X_tr, y_tr, X_te, y_te, tasks, in_dim, n_classes):
    model = build_model(in_dim, n_classes)
    acc_matrix = np.full((len(tasks), len(tasks)), np.nan)
    old_model = None
    seen_classes = []

    for i, task_i in enumerate(tasks):
        Xi, yi = subset_by_classes(X_tr, y_tr, task_i)

        def lwf_loss(m, bx, by):
            if old_model is None or not seen_classes:
                return torch.tensor(0.0)
            with torch.no_grad():
                old_logits = old_model(bx)
            old_idx = torch.tensor(seen_classes, dtype=torch.long)
            new_logits = m(bx)
            # Distill on old-class logits only
            t = LWF_T
            old_soft = F.softmax(old_logits[:, old_idx] / t, dim=1)
            new_log_soft = F.log_softmax(new_logits[:, old_idx] / t, dim=1)
            return LWF_ALPHA * (t * t) * (-(old_soft * new_log_soft).sum(1).mean())

        train_one_task(model, Xi, yi, epochs=EPOCHS_PER_TASK, extra_loss_fn=lwf_loss)

        # Freeze a copy for distillation on the next task
        old_model = copy.deepcopy(model)
        old_model.eval()
        for p in old_model.parameters():
            p.requires_grad = False
        seen_classes.extend(task_i)

        for j, task_j in enumerate(tasks[: i + 1]):
            acc_matrix[i, j] = evaluate_task(model, X_te, y_te, task_j)
    return acc_matrix, model
