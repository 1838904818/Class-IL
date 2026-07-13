"""Accuracy + Forgetting metrics following Chaudhry et al. 2018 / Masana 2023."""
import numpy as np
import torch


def avg_accuracy(acc_matrix: np.ndarray) -> float:
    """A_T = mean accuracy across all seen tasks after the final task.

    Args:
        acc_matrix: shape (T, T), lower-triangular. acc_matrix[i, j] is accuracy
                    on task j after training on task i, for j <= i.
    Returns:
        Mean of the last row, ignoring NaN entries.
    """
    return float(np.nanmean(acc_matrix[-1]))


def avg_forgetting(acc_matrix: np.ndarray) -> float:
    """F_T = mean over j<T of (max_{t<T} a_{t,j} - a_{T,j}).

    Forgetting per past task = its peak accuracy minus its final accuracy.
    Returns 0 if there is only one task (nothing to forget).
    """
    T = acc_matrix.shape[0]
    if T < 2:
        return 0.0
    final = acc_matrix[-1]
    forgetting = []
    for j in range(T - 1):
        max_acc = np.nanmax(acc_matrix[: T - 1, j])
        if np.isnan(max_acc) or np.isnan(final[j]):
            continue
        forgetting.append(max_acc - final[j])
    return float(np.mean(forgetting)) if forgetting else 0.0


def evaluate_per_class(model, X: np.ndarray, y: np.ndarray, n_classes: int):
    """Per-class accuracy on (X, y). Returns (accs[n_classes], preds[n])."""
    model.eval()
    accs = np.full(n_classes, np.nan)
    with torch.no_grad():
        logits = model(torch.from_numpy(X.astype(np.float32))).numpy()
        pred = logits.argmax(1)
    for c in range(n_classes):
        m = y == c
        if m.sum() > 0:
            accs[c] = (pred[m] == c).mean()
    return accs, pred


def evaluate_task(model, X: np.ndarray, y: np.ndarray, task_classes) -> float:
    """Accuracy on samples whose true class is in task_classes."""
    mask = np.isin(y, task_classes)
    if mask.sum() == 0:
        return float("nan")
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X[mask].astype(np.float32))).numpy()
    return float((logits.argmax(1) == y[mask]).mean())
