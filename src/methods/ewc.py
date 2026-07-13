"""Elastic Weight Consolidation (Kirkpatrick et al. 2017).

After each task we snapshot:
  - the model's parameters at task end
  - the Fisher information matrix estimated on that task's data

On the next task we add λ/2 · Σ_n F_n (θ_n - θ_n^*)^2  to the loss.
"""
import numpy as np
import torch
import torch.nn.functional as F

from src.config import EPOCHS_PER_TASK, EWC_LAMBDA
from src.methods.base import build_model, train_one_task, subset_by_classes, to_loader
from src.metrics import evaluate_task


def compute_fisher(model, X, y, sample_size: int = 2000):
    """Empirical diagonal Fisher information on (X, y), per-parameter.

    Estimates E[(∂ log p / ∂θ)^2] by sampling at most `sample_size` rows.
    """
    model.eval()
    idx = np.random.choice(len(X), size=min(sample_size, len(X)), replace=False)
    loader = to_loader(X[idx], y[idx], shuffle=False, batch=64)
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    n_samples = 0
    for bx, by in loader:
        model.zero_grad()
        logits = model(bx)
        logp = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(logp, by)
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2 * bx.size(0)
        n_samples += bx.size(0)
    for n in fisher:
        fisher[n] /= max(n_samples, 1)
    return fisher


def run_ewc(X_tr, y_tr, X_te, y_te, tasks, in_dim, n_classes):
    model = build_model(in_dim, n_classes)
    acc_matrix = np.full((len(tasks), len(tasks)), np.nan)
    fisher_list = []   # list of (params_snapshot, fisher_snapshot)

    for i, task_i in enumerate(tasks):
        Xi, yi = subset_by_classes(X_tr, y_tr, task_i)

        def ewc_loss(m, bx, by):
            if not fisher_list:
                return torch.tensor(0.0)
            loss = torch.tensor(0.0)
            for prev_params, prev_fisher in fisher_list:
                for n, p in m.named_parameters():
                    if n in prev_fisher:
                        loss = loss + (prev_fisher[n] * (p - prev_params[n]) ** 2).sum()
            return EWC_LAMBDA * loss

        train_one_task(model, Xi, yi, epochs=EPOCHS_PER_TASK, extra_loss_fn=ewc_loss)

        # Snapshot params + Fisher for use on later tasks
        params_snap = {n: p.detach().clone() for n, p in model.named_parameters()}
        fisher_snap = compute_fisher(model, Xi, yi)
        fisher_list.append((params_snap, fisher_snap))

        for j, task_j in enumerate(tasks[: i + 1]):
            acc_matrix[i, j] = evaluate_task(model, X_te, y_te, task_j)
    return acc_matrix, model
