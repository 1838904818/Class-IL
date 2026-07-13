"""Frozen-encoder Nearest-Class-Mean (NCM) baseline.

This is the recent (2022-2024) "frozen features + class prototypes" line of
class-incremental learning — FeTrIL (Petit et al., WACV 2023), ADAM and RanPAC
(2023) — adapted to our MLP-IDS backbone for a fair, same-encoder comparison.

The MLP `feat` is trained on Task 0 then FROZEN. Each class is represented by
the mean of its frozen features (a prototype); inference is nearest-prototype
over all classes seen so far, with no task ID. Old-class prototypes never
change, so forgetting is structurally low. The baseline isolates whether low
forgetting is attributable to the frozen encoder alone or whether the
per-family LoRA adapters and DPMeans router add value beyond a prototype
classifier using the same frozen features.
"""
import numpy as np
import torch
import torch.nn as nn

from src.config import EPOCHS_PER_TASK, HIDDEN
from src.methods.base import build_model, train_one_task, subset_by_classes
from src.metrics.accuracy import evaluate_task


class NCMClassifier(nn.Module):
    """Frozen feat + per-class prototypes; forward returns -distance (argmax =
    nearest seen prototype), so it plugs into evaluate_task unchanged."""

    def __init__(self, feat: nn.Module, n_classes: int, hidden: int):
        super().__init__()
        self.feat = feat
        self.register_buffer("protos", torch.zeros(n_classes, hidden))
        self.register_buffer("seen", torch.zeros(n_classes, dtype=torch.bool))

    def set_proto(self, c: int, vec: np.ndarray):
        self.protos[c] = torch.as_tensor(vec, dtype=torch.float32)
        self.seen[c] = True

    def extract_features(self, x):
        return self.feat(x)

    def forward(self, x):
        f = self.feat(x)                                  # (N, hidden)
        d = torch.cdist(f, self.protos)                   # (N, n_classes)
        d = d.masked_fill(~self.seen.unsqueeze(0), 1e9)   # exclude unseen classes
        return -d                                         # argmax => nearest seen proto


def run_ncm_frozen(X_tr, y_tr, X_te, y_te, tasks, in_dim, n_classes):
    # Task 0: train feat+head end-to-end, then freeze the feature extractor.
    base = build_model(in_dim, n_classes)
    X0, y0 = subset_by_classes(X_tr, y_tr, tasks[0])
    train_one_task(base, X0, y0, epochs=EPOCHS_PER_TASK)
    for p in base.feat.parameters():
        p.requires_grad_(False)

    model = NCMClassifier(base.feat, n_classes, HIDDEN)
    model.eval()

    def feats(X):
        with torch.no_grad():
            return model.feat(torch.from_numpy(X.astype(np.float32))).numpy()

    acc_matrix = np.full((len(tasks), len(tasks)), np.nan)
    for i, task_i in enumerate(tasks):
        for c in task_i:                       # add this task's class prototypes
            Xc, _ = subset_by_classes(X_tr, y_tr, [c])
            if len(Xc) > 0:
                model.set_proto(c, feats(Xc).mean(axis=0))
        for j, task_j in enumerate(tasks[: i + 1]):
            acc_matrix[i, j] = evaluate_task(model, X_te, y_te, task_j)
    return acc_matrix, model
