"""Plot the frozen Section 19 eligible X/Y observations; no estimation/training."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload["execution_status"] != "complete" or payload["smoke"]:
        raise ValueError("A complete formal result is required")
    methods = ["replay", "icarl", "replay_lwf", "kd_perturbed", "rcl_tab",
               "global_ir", "oldpair_kd", "crr"]
    labels = ["Replay", "iCaRL", "Replay + LwF", "KD-perturbed", "RCL-tab",
              "Global-IR", "Old-pair KD", "CRR"]
    colors = ["#3666a6", "#cf791f", "#3666a6", "#cf791f", "#787437",
              "#b45c87", "#787437", "#b45c87"]
    markers = ["o", "s", "^", "v", "D", "P", "X", "*"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharex=True, sharey=True)
    for ax, setting in zip(axes, ["S4", "S5"]):
        rows = [r for r in payload["rows"] if r["setting"] == setting
                and r["shift_recall_before"] >= .2]
        assert len(rows) == (89 if setting == "S4" else 80)
        for method, label, color, marker in zip(methods, labels, colors, markers):
            selected = [r for r in rows if r["method"] == method]
            x = [r["old_pair_pi_drift"] for r in selected]
            y = [r["shift_recall_after"] - r["shift_recall_before"] for r in selected]
            assert all(math.isfinite(v) for v in x + y)
            assert all(r["feature_count"] == 39 for r in selected)
            ax.scatter(x, y, s=31, alpha=.70, color=color, marker=marker,
                       linewidths=.6, label=label)
        title = "S4: ToN → CSE (89 eligible units)" if setting == "S4" else "S5: CSE → ToN (80 eligible units)"
        ax.set_title(title, fontsize=11)
        ax.axhline(0, color="#555555", linewidth=.8)
        ax.set_xlim(-.025, .70)
        ax.set_ylim(-.95, .25)
        ax.set_xlabel("Old-pair PI drift (X)")
        ax.grid(alpha=.16)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Shifted recall change (Y = after − before)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .06),
               ncol=4, frameon=False, fontsize=9)
    fig.text(.5, .014, "Unadjusted X/Y scatter; the primary test correlates residual ranks of X and −Y controlling Z. No jitter.",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, .18, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Plotted 89 S4 and 80 S5 eligible units; shared axes; 39-feature schema verified.")


if __name__ == "__main__":
    main()
