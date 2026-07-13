"""D10 — Buffer size ablation.

Sweeps REPLAY_BUFFER_PER_CLASS ∈ {25, 50, 100, 200, 500} for all four replay-based
methods (Replay, Replay-Herding, Replay-DPMeans, iCaRL) on CIC-IDS-2017 (seed=42).

Results saved to:   results/buffer_ablation_results.json
Figure saved to:    results/figures/05_buffer_ablation.png

Usage:
    python -m src.ablation_buffer
    python -m src.ablation_buffer --dataset NSL-KDD
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler

from src.config import RESULTS_DIR, FIG_DIR, seed_all
from src.data import DATASET_LOADERS
from src.metrics import avg_accuracy, avg_forgetting
from src.methods.base import make_task_split

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


BUFFER_SIZES = [25, 50, 100, 200, 500]

# Only replay-based methods (Joint/Naive/EWC/LwF ignore buffer_per_class)
REPLAY_METHODS = {
    "Replay":         "src.methods.replay",
    "Replay-Herding": "src.methods.replay_herding",
    "Replay-DPMeans": "src.methods.replay_dpmeans",
    "iCaRL":          "src.methods.icarl",
}

METHOD_RUNNER = {
    "Replay":         "run_replay",
    "Replay-Herding": "run_replay_herding",
    "Replay-DPMeans": "run_replay_dpmeans",
    "iCaRL":          "run_icarl",
}

METHOD_COLORS = {
    "Replay":         "#F96167",
    "Replay-Herding": "#FF7A00",
    "Replay-DPMeans": "#9B59B6",
    "iCaRL":          "#E91E63",
}


def _import_runner(method_name: str):
    """Dynamically import the runner function for a method."""
    import importlib
    mod = importlib.import_module(REPLAY_METHODS[method_name])
    return getattr(mod, METHOD_RUNNER[method_name])


def run_ablation(dataset_name: str = "CIC-IDS-2017", seed: int = 42):
    loader_fn, classes_per_task = DATASET_LOADERS[dataset_name]
    print(f"\nBuffer-size ablation on {dataset_name}  (seed={seed})")
    print(f"Buffer sizes: {BUFFER_SIZES}")
    print(f"Methods: {list(REPLAY_METHODS)}\n")

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

    sweep = {}
    for method_name in REPLAY_METHODS:
        runner = _import_runner(method_name)
        sweep[method_name] = {"buffer_sizes": BUFFER_SIZES, "avg_accuracy": [], "avg_forgetting": [], "time_sec": []}
        for buf in BUFFER_SIZES:
            seed_all(seed)
            t0 = time.time()
            acc_matrix, _ = runner(X_tr_s, y_tr, X_te_s, y_te, tasks, in_dim, n_classes,
                                   buffer_per_class=buf)
            elapsed = time.time() - t0
            aa = avg_accuracy(acc_matrix)
            af = avg_forgetting(acc_matrix)
            sweep[method_name]["avg_accuracy"].append(aa)
            sweep[method_name]["avg_forgetting"].append(af)
            sweep[method_name]["time_sec"].append(elapsed)
            print(f"  {method_name:20s}  buf={buf:4d}  AvgAcc={aa:.4f}  AvgForget={af:.4f}  t={elapsed:.1f}s")

    result = {
        "dataset": dataset_name,
        "seed": seed,
        "buffer_sizes": BUFFER_SIZES,
        "sweep": sweep,
    }

    out_path = RESULTS_DIR / "buffer_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")

    _plot_ablation(result)
    return result


def _plot_ablation(result: dict):
    sweep = result["sweep"]
    buf_sizes = result["buffer_sizes"]
    dataset = result["dataset"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for m, data in sweep.items():
        ax.plot(buf_sizes, data["avg_accuracy"], marker="o", linewidth=2,
                label=m, color=METHOD_COLORS[m])
    ax.set_xlabel("Buffer size per class (k)")
    ax.set_ylabel("Avg Accuracy ↑")
    ax.set_xscale("log")
    ax.set_xticks(buf_sizes)
    ax.set_xticklabels(buf_sizes)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{dataset} — Accuracy vs. buffer size")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(linestyle=":", alpha=0.5)

    ax = axes[1]
    for m, data in sweep.items():
        ax.plot(buf_sizes, data["avg_forgetting"], marker="s", linewidth=2,
                label=m, color=METHOD_COLORS[m])
    ax.set_xlabel("Buffer size per class (k)")
    ax.set_ylabel("Avg Forgetting ↓")
    ax.set_xscale("log")
    ax.set_xticks(buf_sizes)
    ax.set_xticklabels(buf_sizes)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{dataset} — Forgetting vs. buffer size")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(linestyle=":", alpha=0.5)

    fig.suptitle("D10 — Ablation: Effect of replay buffer size on Class-IL performance",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    out_fig = FIG_DIR / "05_buffer_ablation.png"
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_fig}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CIC-IDS-2017",
                        choices=list(DATASET_LOADERS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_ablation(args.dataset, args.seed)
