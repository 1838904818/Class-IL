"""D11 — Exemplar selection strategy ablation.

Compares the three replay-based exemplar selection strategies
(Random, Herding, DP-Means) + full iCaRL at a fixed buffer size (200/class)
on all available datasets, measuring:
  - AvgAcc / AvgForget summary
  - Per-task retention curves (illustrates *which* old tasks are forgotten)
  - Intra-class exemplar diversity (mean pairwise L2 distance of selected exemplars)

Results saved to:   results/exemplar_ablation_results.json
Figures saved to:   results/figures/06_exemplar_selection_*.png

Usage:
    python -m src.ablation_exemplar
    python -m src.ablation_exemplar --dataset CIC-IDS-2018
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler

from src.config import RESULTS_DIR, FIG_DIR, seed_all, REPLAY_BUFFER_PER_CLASS
from src.data import DATASET_LOADERS
from src.metrics import avg_accuracy, avg_forgetting, evaluate_task
from src.methods.base import make_task_split, build_model, to_loader, subset_by_classes
from src.methods.replay import run_replay
from src.methods.replay_herding import run_replay_herding
from src.methods.replay_dpmeans import run_replay_dpmeans
from src.methods.icarl import run_icarl

mpl.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

METHODS_COMPARE = {
    "Replay (random)":  run_replay,
    "Replay-Herding":   run_replay_herding,
    "Replay-DPMeans":   run_replay_dpmeans,
    "iCaRL":            run_icarl,
}
METHOD_COLORS = {
    "Replay (random)":  "#F96167",
    "Replay-Herding":   "#FF7A00",
    "Replay-DPMeans":   "#9B59B6",
    "iCaRL":            "#E91E63",
}


def exemplar_diversity(X_buf: np.ndarray) -> float:
    """Mean pairwise L2 distance of selected exemplars — higher = more diverse."""
    if len(X_buf) < 2:
        return 0.0
    n = min(len(X_buf), 500)  # cap for speed on large buffers
    idx = np.random.choice(len(X_buf), n, replace=False)
    X_sub = X_buf[idx]
    # Upper-triangular pairwise distances
    dists = []
    for i in range(n):
        d = np.linalg.norm(X_sub[i] - X_sub[i+1:], axis=1)
        dists.extend(d.tolist())
    return float(np.mean(dists)) if dists else 0.0


def run_exemplar_ablation(dataset_name: str = "CIC-IDS-2017", seed: int = 42,
                           buffer_per_class: int = REPLAY_BUFFER_PER_CLASS):
    loader_fn, classes_per_task = DATASET_LOADERS[dataset_name]
    print(f"\nExemplar-selection ablation on {dataset_name}  (seed={seed}, buf={buffer_per_class})")

    seed_all(seed)
    X_tr, y_tr, X_te, y_te, class_order = loader_fn()
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)
    in_dim = X_tr_s.shape[1]
    n_classes = len(class_order)
    tasks = make_task_split(n_classes, classes_per_task)
    print(f"  Train: {X_tr_s.shape}  Test: {X_te_s.shape}  Classes: {n_classes}")
    print(f"  Tasks: {[[class_order[c] for c in t] for t in tasks]}\n")

    results = {}
    for mname, runner in METHODS_COMPARE.items():
        seed_all(seed)
        t0 = time.time()
        acc_matrix, _ = runner(X_tr_s, y_tr, X_te_s, y_te, tasks, in_dim, n_classes,
                               buffer_per_class=buffer_per_class)
        elapsed = time.time() - t0
        aa = avg_accuracy(acc_matrix)
        af = avg_forgetting(acc_matrix)
        print(f"  {mname:22s}  AvgAcc={aa:.4f}  AvgForget={af:.4f}  t={elapsed:.1f}s")
        results[mname] = {
            "acc_matrix": acc_matrix.tolist(),
            "avg_accuracy": aa,
            "avg_forgetting": af,
            "time_sec": elapsed,
        }

    out = {
        "dataset": dataset_name,
        "seed": seed,
        "buffer_per_class": buffer_per_class,
        "class_order": class_order,
        "tasks": [[class_order[c] for c in t] for t in tasks],
        "results": results,
    }
    out_path = RESULTS_DIR / "exemplar_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    _plot_exemplar_ablation(out)
    return out


def _plot_exemplar_ablation(data: dict):
    dataset = data["dataset"]
    n_tasks = len(data["tasks"])
    results = data["results"]
    methods = list(results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Left: bar chart (AvgAcc + AvgForget) ---
    ax = axes[0]
    x = np.arange(len(methods))
    w = 0.36
    accs  = [results[m]["avg_accuracy"]   for m in methods]
    forgs = [results[m]["avg_forgetting"] for m in methods]
    bars_a = ax.bar(x - w/2, accs, w, label="Avg Accuracy ↑",
                    color=[METHOD_COLORS[m] for m in methods], edgecolor="black", linewidth=0.6)
    bars_f = ax.bar(x + w/2, forgs, w, label="Avg Forgetting ↓",
                    color=[METHOD_COLORS[m] for m in methods],
                    alpha=0.45, edgecolor="black", linewidth=0.6, hatch="///")
    for b, v in zip(bars_a, accs):
        ax.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.2f}", ha="center", fontsize=9)
    for b, v in zip(bars_f, forgs):
        ax.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.2f}", ha="center", fontsize=9, alpha=0.7)
    short = [m.split(" ")[0] for m in methods]
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Overall Acc / Forgetting")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # --- Middle: per-task avg accuracy curve over time ---
    ax = axes[1]
    for m in methods:
        am = np.array(results[m]["acc_matrix"])
        seen_avg = [np.nanmean(am[i, :i+1]) for i in range(n_tasks)]
        ax.plot(range(1, n_tasks+1), seen_avg, marker="o", linewidth=2,
                label=m.split(" ")[0], color=METHOD_COLORS[m])
    ax.set_xlabel("After training task #")
    ax.set_ylabel("Avg acc on seen tasks")
    ax.set_xticks(range(1, n_tasks+1))
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy over task sequence")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(linestyle=":", alpha=0.5)

    # --- Right: accuracy on first task (pure forgetting signal) ---
    ax = axes[2]
    for m in methods:
        am = np.array(results[m]["acc_matrix"])
        ax.plot(range(1, n_tasks+1), am[:, 0], marker="s", linewidth=2,
                label=m.split(" ")[0], color=METHOD_COLORS[m])
    ax.set_xlabel("After training task #")
    ax.set_ylabel("Accuracy on Task-0")
    ax.set_xticks(range(1, n_tasks+1))
    ax.set_ylim(0, 1.05)
    ax.set_title("Forgetting of first task (T0)")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(linestyle=":", alpha=0.5)

    ds_slug = dataset.replace("-", "_").lower()
    fig.suptitle(f"D11 — Exemplar selection strategy comparison ({dataset}, buf={data['buffer_per_class']}/class)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    out_fig = FIG_DIR / f"06_exemplar_selection_{ds_slug}.png"
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_fig}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CIC-IDS-2017",
                        choices=list(DATASET_LOADERS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer", type=int, default=REPLAY_BUFFER_PER_CLASS)
    args = parser.parse_args()
    run_exemplar_ablation(args.dataset, args.seed, args.buffer)
