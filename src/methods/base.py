"""Shared backbone (MLP), training loop, and data-loader helpers.

All CL methods build on these primitives. The contract is:
- MLP: shared backbone class.
- to_loader(X, y): wrap arrays into a DataLoader.
- train_one_task(model, X, y, epochs, extra_loss_fn): one task of training,
  optionally with an additional regularization/distillation term.
- subset_by_classes(X, y, classes): filter (X, y) to rows whose label is in `classes`.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.config import HIDDEN, BATCH_SIZE, LR


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """2-layer MLP with ReLU. Shared by all methods for fair comparison."""

    def __init__(self, in_dim: int, hidden: int, n_classes: int):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        return self.head(self.feat(x))

    def extract_features(self, x):
        """Pre-head features (needed by iCaRL nearest-mean classifier)."""
        return self.feat(x)


def build_model(in_dim: int, n_classes: int, hidden: int = HIDDEN) -> MLP:
    """Factory — keeps `MLP(in_dim, HIDDEN, n_classes)` from sprinkling everywhere."""
    return MLP(in_dim, hidden, n_classes)


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------
def to_loader(X, y, shuffle: bool = True, batch: int = BATCH_SIZE) -> DataLoader:
    """Wrap (X, y) numpy arrays into a torch DataLoader."""
    ds = TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y.astype(np.int64)),
    )
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


def subset_by_classes(X, y, classes):
    """Return (X, y) filtered to rows whose y is in `classes`."""
    mask = np.isin(y, classes)
    return X[mask], y[mask]


def make_task_split(n_classes: int, classes_per_task: int):
    """Partition [0, n_classes) into successive task class-sets.

    First task is forced to have >= 2 classes so cross-entropy can train.
    Returns list[list[int]] like [[0,1], [2], [3], [4]].
    """
    tasks = []
    cls = 0
    while cls < n_classes:
        size = classes_per_task
        if cls == 0 and classes_per_task < 2:
            size = 2
        end = min(cls + size, n_classes)
        tasks.append(list(range(cls, end)))
        cls = end
    return tasks


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one_task(model, X, y, epochs: int, extra_loss_fn=None, lr: float = LR):
    """Train `model` on (X, y) for `epochs` epochs.

    Args:
        extra_loss_fn: optional callable (model, batch_x, batch_y) -> tensor scalar.
                       Result is added to the per-batch cross-entropy.
                       Used by EWC / LwF / iCaRL to inject regularization/distillation.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = to_loader(X, y)
    device = next(model.parameters()).device  # CPU default; GPU if model moved
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits = model(bx)
            loss = F.cross_entropy(logits, by)
            if extra_loss_fn is not None:
                loss = loss + extra_loss_fn(model, bx, by)
            loss.backward()
            opt.step()
    return model
