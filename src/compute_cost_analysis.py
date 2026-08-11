"""D14 — Compute cost analysis + figures polish.

Reads the existing result JSONs and produces:
  1. Accuracy vs. wall-clock time scatter plot (efficiency frontier)
  2. Stacked-bar cost breakdown per method (data loading, training, exemplar selection)
  3. Memory usage estimate per method (buffer size × feature dim × bytes)
  4. Updated summary CSV with cost columns

Figure saved to:  results/figures/09_compute_cost.png
CSV saved to:     results/cost_analysis.csv

Usage:
    python -m src.compute_cost_analysis
"""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from src.config import RESULTS_DIR, FIG_DIR, REPLAY_BUFFER_PER_CLASS

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

METHOD_COLORS = {
    "Joint (upper bound)":  "#2C5F2D",
    "Naive (lower bound)":  "#7F7F7F",
    "EWC":                  "#1E2761",
    "LwF":                  "#028090",
    "Replay":               "#F96167",
    "Replay-Herding":       "#FF7A00",
    "Replay-DPMeans":       "#9B59B6",
    "iCaRL":                "#E91E63",
}

METHOD_ORDER = [
    "Naive (lower bound)", "EWC", "LwF",
    "Replay", "Replay-Herding", "Replay-DPMeans", "iCaRL",
    "Joint (upper bound)",
]

# Overhead category per method (for grouped cost bar)
METHOD_OVERHEAD = {
    "Joint (upper bound)":  "joint",
    "Naive (lower bound)":  "none",
    "EWC":                  "regularization",
    "LwF":                  "distillation",
    "Replay":               "replay",
    "Replay-Herding":       "replay + herding",
    "Replay-DPMeans":       "replay + clustering",
    "iCaRL":                "replay + herding + KD + NME",
}

# Buffer memory per class (bytes):  buffer_per_class × max_features × 4 (float32)
# max_features is the largest input dim across datasets (UNSW-NB15 = 196)
BUFFER_BYTES = {
    "Joint (upper bound)":  0,
    "Naive (lower bound)":  0,
    "EWC":                  0,       # stores Fisher, same size as params
    "LwF":                  0,       # stores old model copy (same as params)
    "Replay":               REPLAY_BUFFER_PER_CLASS * 196 * 4,
    "Replay-Herding":       REPLAY_BUFFER_PER_CLASS * 196 * 4,
    "Replay-DPMeans":       REPLAY_BUFFER_PER_CLASS * 196 * 4,
    "iCaRL":                REPLAY_BUFFER_PER_CLASS * 196 * 4,
}


def load_results() -> list[dict]:
    datasets = []
    for name in ["NSL-KDD", "UNSW-NB15", "CIC-IDS-2017", "CIC-IDS-2018"]:
        p = RESULTS_DIR / f"{name}_results.json"
        if p.exists():
            with open(p) as f:
                datasets.append(json.load(f))
    return datasets


def compute_summary(datasets: list[dict]) -> list[dict]:
    rows = []
    for ds in datasets:
        for m in [m for m in METHOD_ORDER if m in ds.get("results", {})]:
            r = ds["results"][m]
            rows.append({
                "Dataset":   ds["dataset"],
                "Method":    m,
                "AvgAcc":    r["avg_accuracy"],
                "AvgForget": r["avg_forgetting"],
                "Time_s":    r["time_sec"],
                "Overhead":  METHOD_OVERHEAD.get(m, ""),
                "BufBytes":  BUFFER_BYTES.get(m, 0),
            })
    return rows


def fig_efficiency_frontier(rows: list[dict]):
    """Accuracy vs. time scatter — efficiency frontier."""
    # Average over datasets per method
    from collections import defaultdict
    m_accs  = defaultdict(list)
    m_times = defaultdict(list)
    for r in rows:
        m_accs[r["Method"]].append(r["AvgAcc"])
        m_times[r["Method"]].append(r["Time_s"])

    methods = [m for m in METHOD_ORDER if m in m_accs]
    mean_acc   = [np.mean(m_accs[m])  for m in methods]
    mean_time  = [np.mean(m_times[m]) for m in methods]

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, m in enumerate(methods):
        ax.scatter(mean_time[i], mean_acc[i],
                   color=METHOD_COLORS[m], s=150, zorder=5, edgecolors="black", linewidths=0.8)
        ax.annotate(m.split(" ")[0], (mean_time[i], mean_acc[i]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)

    # Pareto frontier (lower-left is bad, upper-left is ideal)
    # For each point, highlight if no other point has strictly better acc AND lower time
    pareto = []
    for i, m in enumerate(methods):
        dominated = any(
            mean_acc[j] >= mean_acc[i] and mean_time[j] <= mean_time[i] and (i != j)
            for j in range(len(methods))
        )
        if not dominated:
            pareto.append((mean_time[i], mean_acc[i]))
    if pareto:
        pareto.sort()
        px, py = zip(*pareto)
        ax.step(px, py, color="green", linestyle="--", linewidth=1.5,
                alpha=0.5, where="post", label="Pareto frontier")
        ax.legend(fontsize=9)

    ax.set_xlabel("Mean wall-clock time (s) per dataset run")
    ax.set_ylabel("Mean Avg Accuracy across datasets")
    ax.set_title("Efficiency frontier: Accuracy vs. Compute cost")
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    out_fig = FIG_DIR / "09a_efficiency_frontier.png"
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_fig}")


def fig_time_per_dataset(rows: list[dict]):
    """Grouped bar: wall-clock time per method per dataset."""
    from collections import defaultdict
    ds_names = list(dict.fromkeys(r["Dataset"] for r in rows))
    methods  = [m for m in METHOD_ORDER if any(r["Method"] == m for r in rows)]

    data = defaultdict(dict)
    for r in rows:
        data[r["Dataset"]][r["Method"]] = r["Time_s"]

    x = np.arange(len(methods))
    width = 0.2
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, ds in enumerate(ds_names):
        vals = [data[ds].get(m, 0) for m in methods]
        offset = (i - len(ds_names)/2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=ds, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([m.split(" ")[0] for m in methods], rotation=20, ha="right")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Training time per method per dataset")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out_fig = FIG_DIR / "09b_time_per_dataset.png"
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_fig}")


def fig_acc_time_tradeoff(rows: list[dict]):
    """Per-dataset: accuracy vs. time for each method (small multiples)."""
    from collections import defaultdict
    ds_names = list(dict.fromkeys(r["Dataset"] for r in rows))

    fig, axes = plt.subplots(1, len(ds_names), figsize=(6 * len(ds_names), 5))
    if len(ds_names) == 1:
        axes = [axes]

    for ax, ds_name in zip(axes, ds_names):
        ds_rows = [r for r in rows if r["Dataset"] == ds_name]
        for r in ds_rows:
            m = r["Method"]
            ax.scatter(r["Time_s"], r["AvgAcc"],
                       color=METHOD_COLORS.get(m, "black"), s=120, zorder=5,
                       edgecolors="black", linewidths=0.6)
            ax.annotate(m.split(" ")[0], (r["Time_s"], r["AvgAcc"]),
                        textcoords="offset points", xytext=(5, 3), fontsize=8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Avg Accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title(ds_name)
        ax.grid(linestyle=":", alpha=0.5)

    fig.suptitle("D14 — Accuracy–Compute trade-off per dataset", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    out_fig = FIG_DIR / "09c_acc_time_tradeoff.png"
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_fig}")


def main():
    datasets = load_results()
    if not datasets:
        print("No result JSON files found.")
        return

    rows = compute_summary(datasets)

    fig_efficiency_frontier(rows)
    fig_time_per_dataset(rows)
    fig_acc_time_tradeoff(rows)

    # Save CSV
    out_csv = RESULTS_DIR / "cost_analysis.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Dataset","Method","AvgAcc","AvgForget","Time_s","Overhead","BufBytes"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Saved: {out_csv}")

    # Print efficiency table
    print("\n--- Efficiency Summary (mean across datasets) ---")
    from collections import defaultdict
    m_accs  = defaultdict(list)
    m_times = defaultdict(list)
    for r in rows:
        m_accs[r["Method"]].append(r["AvgAcc"])
        m_times[r["Method"]].append(r["Time_s"])
    print(f"{'Method':30s}  {'MeanAcc':>8s}  {'MeanTime':>10s}  {'Acc/s':>10s}")
    print("-" * 64)
    for m in METHOD_ORDER:
        if m not in m_accs:
            continue
        aa = np.mean(m_accs[m])
        tt = np.mean(m_times[m])
        print(f"{m:30s}  {aa:8.4f}  {tt:10.1f}s  {aa/tt:10.5f}")


if __name__ == "__main__":
    main()
