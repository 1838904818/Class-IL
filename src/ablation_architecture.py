"""D12 — Architecture ablation.

Sweeps MLP depth (1, 2, 3 hidden layers) × hidden width (64, 128, 256, 512)
on NSL-KDD (fast, representative) using the best replay method (iCaRL)
and random Replay as a second reference, to answer:
  - Does a wider/deeper backbone reduce catastrophic forgetting?
  - Is the default 2×128 near-optimal for IDS feature sets?

Results saved to:   results/arch_ablation_results.json
Figure saved to:    results/figures/07_arch_ablation.png

Usage:
    python -m src.ablation_architecture
    python -m src.ablation_architecture --dataset CIC-IDS-2017
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    RESULTS_DIR, FIG_DIR, seed_all, BATCH_SIZE, LR, EPOCHS_PER_TASK,
    REPLAY_BUFFER_PER_CLASS, ICARL_T, ICARL_USE_NME,
)
from src.data import DATASET_LOADERS
from src.metrics import avg_accuracy, avg_forgetting
from src.methods.base import make_task_split, subset_by_classes, to_loader

mpl.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


HIDDEN_SIZES = [64, 128, 256, 512]
DEPTHS       = [1, 2, 3]           # number of hidden layers
TEST_METHODS = ["Replay", "iCaRL"]  # representative methods for the ablation

METHOD_COLORS = {"Replay": "#F96167", "iCaRL": "#E91E63"}


# ---------------------------------------------------------------------------
# Flexible-depth MLP (replaces the fixed 2-layer version for ablation only)
# ---------------------------------------------------------------------------
class FlexMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int, n_layers: int = 2):
        super().__init__()
        layers = []
        prev = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(prev, hidden), nn.ReLU()]
            prev = hidden
        self.feat = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_classes)

    def forward(self, x):
        return self.head(self.feat(x))

    def extract_features(self, x):
        return self.feat(x)


def _train(model, X, y, epochs=EPOCHS_PER_TASK, extra_loss_fn=None):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loader = to_loader(X, y)
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            opt.zero_grad()
            logits = model(bx)
            loss = F.cross_entropy(logits, by)
            if extra_loss_fn:
                loss = loss + extra_loss_fn(model, bx, by)
            loss.backward()
            opt.step()


def _eval_task(model, X_te, y_te, task_classes):
    model.eval()
    mask = np.isin(y_te, task_classes)
    if mask.sum() == 0:
        return float("nan")
    with torch.no_grad():
        logits = model(torch.from_numpy(X_te[mask].astype(np.float32)))
        preds = logits.argmax(1).numpy()
    return float((preds == y_te[mask]).mean())


def _run_replay(X_tr, y_tr, X_te, y_te, tasks, in_dim, n_classes, hidden, n_layers):
    """Random replay with given architecture."""
    model = FlexMLP(in_dim, hidden, n_classes, n_layers)
    acc_matrix = np.full((len(tasks), len(tasks)), np.nan)
    buf_X, buf_y = [], []
    for i, task_i in enumerate(tasks):
        Xi, yi = subset_by_classes(X_tr, y_tr, task_i)
        if buf_X:
            Xc = np.vstack([Xi] + buf_X)
            yc = np.concatenate([yi] + buf_y)
        else:
            Xc, yc = Xi, yi
        _train(model, Xc, yc)
        for c in task_i:
            mask = (yi == c)
            if mask.sum() == 0: continue
            idx = np.random.choice(mask.sum(), min(REPLAY_BUFFER_PER_CLASS, mask.sum()), replace=False)
            buf_X.append(Xi[np.where(mask)[0][idx]])
            buf_y.append(yi[np.where(mask)[0][idx]])
        for j, task_j in enumerate(tasks[:i+1]):
            acc_matrix[i, j] = _eval_task(model, X_te, y_te, task_j)
    return acc_matrix


def _run_icarl(X_tr, y_tr, X_te, y_te, tasks, in_dim, n_classes, hidden, n_layers):
    """iCaRL (herding + KD + NME) with given architecture."""
    import copy
    from src.methods.icarl import _herding_select, _evaluate_task_nme

    model = FlexMLP(in_dim, hidden, n_classes, n_layers)
    acc_matrix = np.full((len(tasks), len(tasks)), np.nan)
    buf_X, buf_y = [], []
    class_means = {}
    old_model = None
    seen_classes = []

    for i, task_i in enumerate(tasks):
        Xi, yi = subset_by_classes(X_tr, y_tr, task_i)
        if buf_X:
            Xc = np.vstack([Xi] + buf_X)
            yc = np.concatenate([yi] + buf_y)
        else:
            Xc, yc = Xi, yi

        old_classes_seen = list(seen_classes)

        def extra_loss(m, bx, _by):
            if old_model is None or len(old_classes_seen) == 0:
                return torch.tensor(0.0)
            with torch.no_grad():
                old_logits = old_model(bx)[:, old_classes_seen]
                soft_targets = F.softmax(old_logits / ICARL_T, dim=1)
            new_logits = m(bx)[:, old_classes_seen]
            log_probs = F.log_softmax(new_logits / ICARL_T, dim=1)
            return F.kl_div(log_probs, soft_targets, reduction="batchmean") * (ICARL_T ** 2)

        _train(model, Xc, yc, extra_loss_fn=extra_loss if old_model is not None else None)
        old_model = copy.deepcopy(model)
        old_model.eval()
        for p in old_model.parameters():
            p.requires_grad_(False)

        model.eval()
        with torch.no_grad():
            feats = model.extract_features(torch.from_numpy(Xi.astype(np.float32))).numpy()
        for c in task_i:
            mask = (yi == c)
            if mask.sum() == 0: continue
            idx_l = _herding_select(feats[mask], REPLAY_BUFFER_PER_CLASS)
            g = np.where(mask)[0][idx_l]
            buf_X.append(Xi[g]); buf_y.append(yi[g])

        for c in task_i:
            seen_classes.append(c)
        all_buf_X = np.vstack(buf_X) if buf_X else np.empty((0, in_dim))
        all_buf_y = np.concatenate(buf_y) if buf_y else np.empty(0, dtype=np.int64)
        model.eval()
        with torch.no_grad():
            feats_b = model.extract_features(torch.from_numpy(all_buf_X.astype(np.float32))).numpy()
        for c in seen_classes:
            msk = (all_buf_y == c)
            if msk.sum() > 0:
                mu = feats_b[msk].mean(0)
                class_means[c] = mu / (np.linalg.norm(mu) + 1e-8)

        for j, task_j in enumerate(tasks[:i+1]):
            if ICARL_USE_NME and class_means:
                acc_matrix[i, j] = _evaluate_task_nme(model, X_te, y_te, task_j, class_means)
            else:
                acc_matrix[i, j] = _eval_task(model, X_te, y_te, task_j)

    return acc_matrix


METHOD_RUNNERS = {"Replay": _run_replay, "iCaRL": _run_icarl}


def run_arch_ablation(dataset_name: str = "NSL-KDD", seed: int = 42):
    loader_fn, classes_per_task = DATASET_LOADERS[dataset_name]
    print(f"\nArchitecture ablation on {dataset_name}  (seed={seed})")
    print(f"Depths:  {DEPTHS}\nWidths:  {HIDDEN_SIZES}\nMethods: {TEST_METHODS}\n")

    seed_all(seed)
    X_tr, y_tr, X_te, y_te, class_order = loader_fn()
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)
    in_dim, n_classes = X_tr_s.shape[1], len(class_order)
    tasks = make_task_split(n_classes, classes_per_task)
    print(f"  Train: {X_tr_s.shape}  Test: {X_te_s.shape}  Classes: {n_classes}\n")

    sweep = {m: {} for m in TEST_METHODS}

    for depth in DEPTHS:
        for hidden in HIDDEN_SIZES:
            key = f"d{depth}_h{hidden}"
            params = f"depth={depth}, hidden={hidden}"
            for mname in TEST_METHODS:
                seed_all(seed)
                t0 = time.time()
                runner = METHOD_RUNNERS[mname]
                am = runner(X_tr_s, y_tr, X_te_s, y_te, tasks, in_dim, n_classes, hidden, depth)
                elapsed = time.time() - t0
                aa, af = avg_accuracy(am), avg_forgetting(am)
                print(f"  {mname:8s}  {params}  AvgAcc={aa:.4f}  AvgForget={af:.4f}  t={elapsed:.1f}s")
                sweep[mname][key] = {
                    "depth": depth, "hidden": hidden,
                    "avg_accuracy": aa, "avg_forgetting": af,
                    "acc_matrix": am.tolist(), "time_sec": elapsed,
                }

    result = {
        "dataset": dataset_name, "seed": seed,
        "depths": DEPTHS, "hidden_sizes": HIDDEN_SIZES,
        "methods": TEST_METHODS, "sweep": sweep,
    }
    out_path = RESULTS_DIR / "arch_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")

    _plot_arch_ablation(result)
    return result


def _plot_arch_ablation(result: dict):
    dataset = result["dataset"]
    depths = result["depths"]
    hidden_sizes = result["hidden_sizes"]
    methods = result["methods"]
    sweep = result["sweep"]

    fig, axes = plt.subplots(len(methods), 2,
                             figsize=(12, 4.5 * len(methods)))
    if len(methods) == 1:
        axes = np.array([axes])

    for row, mname in enumerate(methods):
        for col, metric in enumerate(["avg_accuracy", "avg_forgetting"]):
            ax = axes[row, col]
            for depth in depths:
                vals = [sweep[mname][f"d{depth}_h{h}"][metric] for h in hidden_sizes]
                ax.plot(hidden_sizes, vals, marker="o", linewidth=2,
                        label=f"{depth} layer{'s' if depth>1 else ''}")
            ax.set_xlabel("Hidden width")
            ax.set_xscale("log")
            ax.set_xticks(hidden_sizes)
            ax.set_xticklabels(hidden_sizes)
            label = "Avg Accuracy ↑" if metric == "avg_accuracy" else "Avg Forgetting ↓"
            ax.set_ylabel(label)
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{mname} — {label}")
            ax.legend(loc="lower right" if col == 0 else "upper right", framealpha=0.9)
            ax.grid(linestyle=":", alpha=0.5)
            # mark default config (2 layers, 128 units)
            default_val = sweep[mname]["d2_h128"][metric]
            ax.axvline(128, color="grey", linestyle="--", alpha=0.5, linewidth=1)
            if depth == 2:
                ax.plot(128, default_val, marker="*", markersize=14,
                        color="black", zorder=5, label="default (2×128)")

    fig.suptitle(f"D12 — Architecture ablation: depth × width on {dataset}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    out_fig = FIG_DIR / "07_arch_ablation.png"
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_fig}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="NSL-KDD",
                        choices=list(DATASET_LOADERS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_arch_ablation(args.dataset, args.seed)
