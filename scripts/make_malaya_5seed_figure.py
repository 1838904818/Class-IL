from __future__ import annotations

"""Plot the validated paired five-seed Malaya model-size comparison."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PERFORMANCE_METRICS = (
    ("final_overall_accuracy", "Accuracy"),
    ("final_macro_f1", "Macro-F1"),
    ("final_balanced_accuracy", "Balanced\naccuracy"),
    ("average_task_accuracy", "Average task\naccuracy"),
)
MODELS = (
    ("baseline", "MLP 128x2", "#315A7D"),
    ("larger", "MLP 256x4", "#C6922E"),
)
SEEDS = (1, 2, 3, 4, 42)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the paired five-seed Malaya validation figure."
    )
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("dataset") != "malaya-network-gt"
        or tuple(report.get("seeds", ())) != SEEDS
    ):
        raise ValueError("unexpected comparison report identity")
    return report


def _performance_panel(axis, report: dict, view: str, title: str) -> None:
    x = np.arange(len(PERFORMANCE_METRICS), dtype=float)
    width = 0.34
    for model_index, (model_key, label, color) in enumerate(MODELS):
        means = []
        standard_deviations = []
        for metric, _ in PERFORMANCE_METRICS:
            values = report["views"][view]["metrics"][metric][model_key]
            means.append(float(values["mean"]))
            standard_deviations.append(float(values["sample_std"]))
        offset = (model_index - 0.5) * width
        bars = axis.bar(
            x + offset,
            means,
            width=width,
            yerr=standard_deviations,
            capsize=3,
            color=color,
            edgecolor="#263442",
            linewidth=0.7,
            label=label,
            zorder=3,
        )
        for bar, value in zip(bars, means):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.024,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="semibold",
            )
    axis.set_ylim(0, 0.75)
    axis.set_xticks(x, [label for _, label in PERFORMANCE_METRICS])
    axis.set_title(title, loc="left", fontweight="semibold", pad=10)
    axis.set_ylabel("Mean over five seeds")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)


def _paired_accuracy_panel(axis, report: dict) -> None:
    metric = report["views"]["duplicate_excluded"]["metrics"][
        "final_overall_accuracy"
    ]
    baseline = np.asarray(metric["baseline"]["per_seed"], dtype=float)
    larger = np.asarray(metric["larger"]["per_seed"], dtype=float)
    for seed, left, right in zip(SEEDS, baseline, larger):
        axis.plot([0, 1], [left, right], color="#9CA3AF", linewidth=1.4, zorder=1)
        axis.scatter(0, left, color=MODELS[0][2], s=42, zorder=3)
        axis.scatter(1, right, color=MODELS[1][2], s=42, zorder=3)
        label_offset = {1: -0.006, 4: 0.006}.get(seed, 0.0)
        axis.text(
            1.045,
            right + label_offset,
            str(seed),
            va="center",
            fontsize=8,
            color="#4B5563",
        )
    axis.set_xlim(-0.25, 1.25)
    axis.set_ylim(0.40, 0.64)
    axis.set_xticks([0, 1], [MODELS[0][1], MODELS[1][1]])
    axis.set_title(
        "Paired duplicate-excluded accuracy",
        loc="left",
        fontweight="semibold",
        pad=10,
    )
    axis.set_ylabel("Accuracy")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    delta = metric["paired_delta_larger_minus_baseline"]
    low, high = delta["ci95_student_t"]
    axis.text(
        0.02,
        0.03,
        f"Mean delta (larger - baseline): {delta['mean']:.3f}\n"
        f"95% paired-t CI [{low:.3f}, {high:.3f}]",
        transform=axis.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D1D5DB"},
    )


def _forgetting_panel(axis, report: dict) -> None:
    x = np.arange(2, dtype=float)
    width = 0.34
    view_items = (("official", "Official"), ("duplicate_excluded", "Duplicate-excluded"))
    for model_index, (model_key, label, color) in enumerate(MODELS):
        means = []
        standard_deviations = []
        for view, _ in view_items:
            values = report["views"][view]["metrics"]["average_forgetting"][model_key]
            means.append(float(values["mean"]))
            standard_deviations.append(float(values["sample_std"]))
        bars = axis.bar(
            x + (model_index - 0.5) * width,
            means,
            width=width,
            yerr=standard_deviations,
            capsize=3,
            color=color,
            edgecolor="#263442",
            linewidth=0.7,
            label=label,
            zorder=3,
        )
        for bar, value in zip(bars, means):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.006,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="semibold",
            )
    axis.set_ylim(0, 0.085)
    axis.set_xticks(x, [label for _, label in view_items])
    axis.set_title("Average forgetting", loc="left", fontweight="semibold", pad=10)
    axis.set_ylabel("Mean over five seeds (lower is better)")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)


def main() -> None:
    args = _parser().parse_args()
    report = _load(args.comparison.resolve())
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "#FCFCFD",
            "axes.edgecolor": "#374151",
            "text.color": "#1F2937",
            "axes.labelcolor": "#374151",
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(16, 9))
    figure.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.10, wspace=0.24, hspace=0.38)
    _performance_panel(axes[0, 0], report, "official", "Official capture holdout (n=10,370)")
    _performance_panel(
        axes[0, 1],
        report,
        "duplicate_excluded",
        "Duplicate-excluded holdout (n=6,590)",
    )
    _paired_accuracy_panel(axes[1, 0], report)
    _forgetting_panel(axes[1, 1], report)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.065, 0.90), frameon=False, ncol=2)
    figure.suptitle(
        "MalayaNetwork_GT paired five-seed validation",
        x=0.065,
        y=0.975,
        ha="left",
        fontsize=20,
        fontweight="bold",
    )
    figure.text(
        0.065,
        0.925,
        "Application/service classification; five two-class increments; joint-uncapped prediction; error bars are sample SD.",
        ha="left",
        fontsize=11,
        color="#4B5563",
    )
    figure.text(
        0.065,
        0.025,
        "Same split and training budget for both models. Exact Wilcoxon p cannot be below 0.0625 with five non-zero pairs.",
        ha="left",
        fontsize=9,
        color="#6B7280",
    )
    figure.savefig(args.output.resolve(), dpi=180, facecolor="white")
    plt.close(figure)
    print(json.dumps({"figure": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
