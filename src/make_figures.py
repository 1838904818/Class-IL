"""Generate result figures from JSON output of run_experiment.py."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

METHOD_COLORS = {
    "Joint (upper bound)":  "#2C5F2D",   # forest green — upper bound
    "Naive (lower bound)":  "#7F7F7F",   # grey — lower bound
    "EWC":                  "#1E2761",   # navy — regularization
    "LwF":                  "#028090",   # teal — distillation
    "Replay":               "#F96167",   # coral — rehearsal baseline
    "Replay-Herding":       "#FF7A00",   # orange — herding exemplar selection
    "Replay-DPMeans":       "#9B59B6",   # purple — DP-Means (★ main contribution)
    "iCaRL":                "#E91E63",   # magenta — full iCaRL
}

# Full order; figures skip methods not present in a given JSON
METHOD_ORDER = [
    "Naive (lower bound)", "EWC", "LwF",
    "Replay", "Replay-Herding", "Replay-DPMeans", "iCaRL",
    "Joint (upper bound)",
]


def _present_methods(ds):
    """Return METHOD_ORDER filtered to methods actually present in `ds`."""
    return [m for m in METHOD_ORDER if m in ds.get("results", {})]


def fig_avg_acc_bar(datasets):
    """Bar chart: avg accuracy & forgetting per method on each dataset."""
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 5))
    if n == 1:
        axes = [axes]
    for ax_idx, ds in enumerate(datasets):
        ax = axes[ax_idx]
        methods = _present_methods(ds)
        accs = [ds["results"][m]["avg_accuracy"] for m in methods]
        forgs = [ds["results"][m]["avg_forgetting"] for m in methods]

        x = np.arange(len(methods))
        w = 0.36
        bars_a = ax.bar(x - w/2, accs, w, label="Avg Accuracy ↑",
                        color=[METHOD_COLORS[m] for m in methods], edgecolor="black", linewidth=0.6)
        bars_f = ax.bar(x + w/2, forgs, w, label="Avg Forgetting ↓",
                        color=[METHOD_COLORS[m] for m in methods],
                        alpha=0.45, edgecolor="black", linewidth=0.6, hatch="///")
        for b, v in zip(bars_a, accs):
            ax.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.2f}", ha="center", fontsize=9)
        for b, v in zip(bars_f, forgs):
            ax.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.2f}", ha="center", fontsize=9, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([m.split(" ")[0] for m in methods], rotation=0)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{ds['dataset']} — Class-IL Performance")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.suptitle("Gap 3: Class-Incremental Learning Benchmark on Network Traffic Data",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(FIG / "01_avg_accuracy_forgetting.png", bbox_inches="tight")
    plt.close(fig)


def fig_per_task_curves(ds):
    """For each method, plot accuracy on task 0 over time as more tasks come in.
    Demonstrates catastrophic forgetting visually."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    n_tasks = len(ds["tasks"])
    methods_here = _present_methods(ds)

    # Left: avg accuracy across seen tasks after each step
    ax = axes[0]
    for m in methods_here:
        if m == "Joint (upper bound)":
            continue
        am = np.array(ds["results"][m]["acc_matrix"])
        seen_avg = []
        for i in range(n_tasks):
            row = am[i, : i + 1]
            seen_avg.append(np.nanmean(row))
        ax.plot(range(1, n_tasks + 1), seen_avg, marker="o", linewidth=2,
                label=m.split(" ")[0], color=METHOD_COLORS[m])
    if "Joint (upper bound)" in ds["results"]:
        joint_final = ds["results"]["Joint (upper bound)"]["avg_accuracy"]
        ax.axhline(joint_final, ls="--", color=METHOD_COLORS["Joint (upper bound)"],
                   label=f"Joint upper bound ({joint_final:.2f})")
    ax.set_xticks(range(1, n_tasks + 1))
    ax.set_xlabel("After training task #")
    ax.set_ylabel("Avg accuracy on all seen tasks")
    ax.set_title(f"{ds['dataset']} — Accuracy degradation as new classes arrive")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(linestyle=":", alpha=0.5)

    # Right: accuracy on FIRST task over time → shows pure forgetting
    ax = axes[1]
    for m in methods_here:
        if m == "Joint (upper bound)":
            continue
        am = np.array(ds["results"][m]["acc_matrix"])
        task0_acc = am[:, 0]
        ax.plot(range(1, n_tasks + 1), task0_acc, marker="s", linewidth=2,
                label=m.split(" ")[0], color=METHOD_COLORS[m])
    if "Joint (upper bound)" in ds["results"]:
        ax.axhline(joint_final, ls="--", color=METHOD_COLORS["Joint (upper bound)"], alpha=0.5)
    ax.set_xticks(range(1, n_tasks + 1))
    ax.set_xlabel("After training task #")
    ax.set_ylabel("Accuracy on Task-0 (first classes)")
    ax.set_title(f"{ds['dataset']} — Forgetting of first task")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(linestyle=":", alpha=0.5)

    fig.suptitle(f"Per-task accuracy dynamics — {ds['dataset']}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fname = f"02_per_task_curves_{ds['dataset'].replace('-','_').lower()}.png"
    fig.savefig(FIG / fname, bbox_inches="tight")
    plt.close(fig)


def fig_heatmap(ds):
    """T x T accuracy heatmap per method (shows the full forgetting matrix)."""
    methods = [m for m in _present_methods(ds) if m != "Joint (upper bound)"]
    fig, axes = plt.subplots(1, len(methods), figsize=(4.0 * len(methods), 4.4))
    n_tasks = len(ds["tasks"])
    task_labels = [f"T{i}\n{'/'.join(ds['tasks'][i])[:18]}" for i in range(n_tasks)]
    for ax, m in zip(axes, methods):
        am = np.array(ds["results"][m]["acc_matrix"])
        im = ax.imshow(am, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        for i in range(n_tasks):
            for j in range(n_tasks):
                v = am[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="black" if 0.3 < v < 0.85 else "white", fontsize=9)
        ax.set_xticks(range(n_tasks))
        ax.set_yticks(range(n_tasks))
        ax.set_xticklabels([f"Eval T{j}" for j in range(n_tasks)], rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels([f"After T{i}" for i in range(n_tasks)], fontsize=9)
        ax.set_title(m.split(" ")[0])
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="Accuracy")
    fig.suptitle(f"Accuracy matrix (rows=after training task i, cols=evaluated on task j) — {ds['dataset']}",
                 fontsize=13, fontweight="bold")
    fname = f"03_heatmap_{ds['dataset'].replace('-','_').lower()}.png"
    fig.savefig(FIG / fname, bbox_inches="tight")
    plt.close(fig)


def fig_per_class_final(ds):
    """Per-class accuracy at end of training — shows which classes get forgotten."""
    classes = ds["class_order"]
    methods = _present_methods(ds)
    n_cls = len(classes)
    x = np.arange(n_cls)
    width = 0.16
    fig, ax = plt.subplots(figsize=(max(8, n_cls * 1.0), 5))
    for i, m in enumerate(methods):
        accs = ds["results"][m]["per_class_accuracy"]
        offset = (i - len(methods)/2) * width + width/2
        ax.bar(x + offset, accs, width, label=m.split(" ")[0],
               color=METHOD_COLORS[m], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("Accuracy at end of all tasks")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{ds['dataset']} — Per-class accuracy after final task (= forgetting profile)")
    ax.legend(loc="upper right", ncol=len(methods)//2 + 1, framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fname = f"04_per_class_final_{ds['dataset'].replace('-','_').lower()}.png"
    fig.savefig(FIG / fname, bbox_inches="tight")
    plt.close(fig)


def main():
    datasets = []
    for name in ["NSL-KDD", "UNSW-NB15", "CIC-IDS-2017", "CIC-IDS-2018"]:
        path = RES / f"{name}_results.json"
        if path.exists():
            with open(path) as f:
                datasets.append(json.load(f))
        else:
            print(f"  skipping {name} (no results file)")

    if not datasets:
        print("No result JSON files found — run the benchmark first.")
        return

    fig_avg_acc_bar(datasets)
    for ds in datasets:
        fig_per_task_curves(ds)
        fig_heatmap(ds)
        fig_per_class_final(ds)

    # Summary table
    rows = []
    for ds in datasets:
        for m in _present_methods(ds):
            r = ds["results"][m]
            rows.append({
                "Dataset": ds["dataset"],
                "Method": m,
                "AvgAcc": f"{r['avg_accuracy']:.4f}",
                "AvgForget": f"{r['avg_forgetting']:.4f}",
                "Time(s)": f"{r['time_sec']:.1f}",
            })
    import csv
    with open(RES / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print("Saved figures to", FIG)
    print("Saved summary to", RES / "summary.csv")
    for r in rows:
        print(r)

    # plt.show() disabled — opening 7 GUI windows freezes the machine.
    # Open results/figures/ manually to view PNG output.
    # print("\nDisplaying figures — close any window to exit.")
    # plt.show()


if __name__ == "__main__":
    main()
