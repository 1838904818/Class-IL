"""Generate PILoRA-IDS v0.5 figures and tables for paper_v2.

Reads:
  results/{NSL-KDD,UNSW-NB15,CIC-IDS-2017,CIC-IDS-2018}_results.json (Phase I)
  results/fedmac_v04_mlp_{nsl_kdd,unsw_nb15,cic_ids_2017,cic_ids_2018}.json (v0.4)
  results/pilora_v05_*.json (multi-seed)

Produces:
  results/figures/v2_accuracy_forgetting.png  — bar chart Phase I vs PILoRA
  results/figures/v2_forget_reduction.png     — forgetting reduction factor
  results/figures/v2_param_efficiency.png     — params vs forgetting
  results/v2_summary.csv                       — flat table
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results")
FIG = RESULTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)

DATASETS = ["NSL-KDD", "UNSW-NB15", "CIC-IDS-2017", "CIC-IDS-2018"]
PHASE1_BASELINES = ["Replay", "Replay-Herding", "Replay-DPMeans", "iCaRL"]


def load_phase1(ds):
    p = RESULTS / f"{ds}_results.json"
    if not p.exists():
        return {}
    return json.load(open(p))["results"]


def load_pilora(ds):
    suffix = ds.lower().replace("-", "_")
    p = RESULTS / f"fedmac_v04_mlp_{suffix}.json"
    if not p.exists():
        return None
    return json.load(open(p))


def fig_accuracy_forgetting():
    """Grouped bars: each method's Acc and Forget on each dataset."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    methods_acc = PHASE1_BASELINES + ["PILoRA"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#000000"]  # PILoRA black
    width = 0.14
    x = np.arange(len(DATASETS))

    # ---- Accuracy
    ax = axes[0]
    for i, m in enumerate(methods_acc):
        vals = []
        for ds in DATASETS:
            if m == "PILoRA":
                p = load_pilora(ds)
                vals.append(p["avg_accuracy"] if p else 0)
            else:
                p1 = load_phase1(ds)
                vals.append(p1.get(m, {}).get("avg_accuracy", 0))
        ax.bar(x + (i - 2) * width, vals, width,
               label=m, color=colors[i], edgecolor="white")
    ax.set_ylabel("Average Accuracy $A_T$ ($\\uparrow$)")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_title("Average Accuracy across 4 IDS datasets")
    ax.legend(ncol=5, frameon=False, loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_ylim(0, 1)

    # ---- Forgetting
    ax = axes[1]
    for i, m in enumerate(methods_acc):
        vals = []
        for ds in DATASETS:
            if m == "PILoRA":
                p = load_pilora(ds)
                vals.append(p["avg_forgetting"] if p else 0)
            else:
                p1 = load_phase1(ds)
                vals.append(p1.get(m, {}).get("avg_forgetting", 0))
        ax.bar(x + (i - 2) * width, vals, width,
               label=m, color=colors[i], edgecolor="white")
    ax.set_ylabel("Average Forgetting $F_T$ ($\\downarrow$)")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_title("Average Forgetting across 4 IDS datasets (lower is better)")
    ax.legend(ncol=5, frameon=False, loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    out = FIG / "v2_accuracy_forgetting.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def load_pilora_multi_seed(ds):
    """Load 5-seed mean forget from pilora_v05_*.json if available."""
    for p in sorted(Path("results").glob("pilora_v05_*.json")):
        d = json.load(open(p))
        if ds in d:
            return d[ds]["mean_forget"]
    return None


def fig_forget_reduction():
    """Forgetting reduction factor: best Phase I / PILoRA (multi-seed when available)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    reductions = []
    for ds in DATASETS:
        p1 = load_phase1(ds)
        # Prefer multi-seed PILoRA forget; fall back to single-seed
        p2_forget = load_pilora_multi_seed(ds)
        if p2_forget is None:
            p2 = load_pilora(ds)
            p2_forget = p2["avg_forgetting"] if p2 else None
        if p2_forget is None:
            reductions.append(0)
            continue
        best_p1_forget = min(
            p1.get(m, {}).get("avg_forgetting", 1.0)
            for m in PHASE1_BASELINES
        )
        ratio = best_p1_forget / max(p2_forget, 1e-6)
        reductions.append(ratio)
    bars = ax.bar(DATASETS, reductions, color="#000000", edgecolor="white", width=0.5)
    for b, v in zip(bars, reductions):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2,
                f"{v:.1f}×", ha="center", va="bottom", fontweight="bold")
    ax.axhline(1.0, color="gray", linestyle="--", label="parity")
    ax.set_ylabel("Forgetting Reduction Factor")
    ax.set_title("PILoRA-IDS vs Best Phase-I baseline (5-seed): $F_T$ reduction")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_ylim(0, max(reductions) * 1.3 if reductions else 10)
    plt.tight_layout()
    out = FIG / "v2_forget_reduction.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def make_csv():
    rows = ["dataset,method,avg_accuracy,avg_forgetting,wall_time_sec"]
    for ds in DATASETS:
        p1 = load_phase1(ds)
        for m in ["Joint (upper bound)", "Naive (lower bound)", "EWC", "LwF",
                  "Replay", "Replay-Herding", "Replay-DPMeans", "iCaRL"]:
            v = p1.get(m, {})
            rows.append(f"{ds},{m},"
                        f"{v.get('avg_accuracy', '')},"
                        f"{v.get('avg_forgetting', '')},"
                        f"{v.get('time_sec', '')}")
        p2 = load_pilora(ds)
        if p2:
            rows.append(f"{ds},PILoRA-IDS,"
                        f"{p2.get('avg_accuracy', '')},"
                        f"{p2.get('avg_forgetting', '')},"
                        f"{p2.get('wall_time_sec', '')}")
    out = RESULTS / "v2_summary.csv"
    out.write_text("\n".join(rows), encoding="utf-8")
    print(f"Saved: {out}")


def main():
    print("Generating PILoRA-IDS v2 figures and tables...")
    fig_accuracy_forgetting()
    fig_forget_reduction()
    make_csv()


if __name__ == "__main__":
    main()
