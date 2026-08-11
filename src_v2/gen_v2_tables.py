"""Generate LaTeX tables for paper_v2 from PILoRA-IDS results.

Tables produced (printed to stdout):
  1. Main benchmark (Phase I vs PILoRA single-seed) — Table 2 of paper_v2
  2. Multi-seed (5 seeds) comparison — Table 3 of paper_v2
  3. Forget reduction summary — Table 4

Usage:
    python -X utf8 -u -m src_v2.gen_v2_tables
"""
import json
from pathlib import Path

import numpy as np

RESULTS = Path("results")

DATASETS = ["NSL-KDD", "UNSW-NB15", "CIC-IDS-2017", "CIC-IDS-2018"]


def load_phase1(ds):
    p = RESULTS / f"{ds}_results.json"
    return json.load(open(p))["results"] if p.exists() else {}


def load_phase1_5seed(ds):
    p = RESULTS / f"{ds}_results_5seed.json"
    return json.load(open(p))["results"] if p.exists() else None


def load_pilora_single(ds):
    suffix = ds.lower().replace("-", "_")
    p = RESULTS / f"fedmac_v04_mlp_{suffix}.json"
    return json.load(open(p)) if p.exists() else None


def load_pilora_multi(ds):
    """Search across pilora_v05_* json files for this dataset entry."""
    for p in sorted(RESULTS.glob("pilora_v05_*.json")):
        d = json.load(open(p))
        if ds in d:
            return d[ds]
    return None


def fmt(x, dp=3):
    return "---" if x is None else f"{x:.{dp}f}"


def table_main_benchmark():
    print("% Table 2: Main benchmark (single seed)")
    print("\\begin{table*}[t]")
    print("\\caption{Main benchmark: average accuracy $A_T$ / average forgetting $F_T$ on four IDS datasets (seed 42). Best CL method per metric in \\textbf{bold}.}")
    print("\\label{tab:main}")
    print("\\centering\\small")
    print("\\begin{tabular}{lcccccccc}\\toprule")
    print("& \\multicolumn{2}{c}{NSL-KDD} & \\multicolumn{2}{c}{UNSW-NB15} &")
    print("  \\multicolumn{2}{c}{CIC-IDS-2017} & \\multicolumn{2}{c}{CIC-IDS-2018} \\\\")
    print("Method & $A_T\\uparrow$ & $F_T\\downarrow$ & $A_T\\uparrow$ & $F_T\\downarrow$ &")
    print("  $A_T\\uparrow$ & $F_T\\downarrow$ & $A_T\\uparrow$ & $F_T\\downarrow$ \\\\")
    print("\\midrule")

    methods = [
        "Joint (upper bound)", "Naive (lower bound)",
        "EWC", "LwF",
        "Replay", "Replay-Herding", "Replay-DPMeans", "iCaRL",
    ]
    rows = {m: {} for m in methods}
    pilora_row = {}
    for ds in DATASETS:
        p1 = load_phase1(ds)
        for m in methods:
            v = p1.get(m, {})
            rows[m][ds] = (v.get("avg_accuracy"), v.get("avg_forgetting"))
        p2 = load_pilora_single(ds)
        pilora_row[ds] = (p2.get("avg_accuracy") if p2 else None,
                          p2.get("avg_forgetting") if p2 else None)

    # Find best per-dataset per-metric (excluding Joint, Naive)
    cl_methods = methods[2:] + ["PILoRA"]
    best_acc, best_fgt = {}, {}
    for ds in DATASETS:
        vals = {m: rows[m][ds][0] for m in methods[2:]}
        vals["PILoRA"] = pilora_row[ds][0]
        best_acc[ds] = max(vals, key=lambda m: vals[m] if vals[m] is not None else -1)
        fgts = {m: rows[m][ds][1] for m in methods[2:]}
        fgts["PILoRA"] = pilora_row[ds][1]
        valid_fgts = {k: v for k, v in fgts.items() if v is not None}
        best_fgt[ds] = min(valid_fgts, key=lambda m: valid_fgts[m])

    def cell(method, ds, val_idx):
        v = rows[method][ds][val_idx] if method != "PILoRA" else pilora_row[ds][val_idx]
        if v is None:
            return "---"
        is_best = (val_idx == 0 and best_acc[ds] == method) or \
                  (val_idx == 1 and best_fgt[ds] == method)
        s = f"{v:.3f}"
        if is_best:
            s = f"\\textbf{{{s}}}"
        return s

    short = {
        "Joint (upper bound)": "Joint (UB)",
        "Naive (lower bound)": "Naive (LB)",
        "EWC": "EWC", "LwF": "LwF",
        "Replay": "Replay", "Replay-Herding": "Replay-Herding",
        "Replay-DPMeans": "Replay-DPMeans", "iCaRL": "iCaRL",
    }
    for m in methods:
        cells = []
        for ds in DATASETS:
            cells.append(cell(m, ds, 0))
            cells.append(cell(m, ds, 1) if m != "Joint (upper bound)" else "---")
        print(f"{short[m]:<20} & " + " & ".join(cells) + " \\\\")
    # PILoRA row
    cells = []
    for ds in DATASETS:
        cells.append(cell("PILoRA", ds, 0))
        cells.append(cell("PILoRA", ds, 1))
    print(f"\\textbf{{PILoRA (ours)}}   & " + " & ".join(cells) + " \\\\")
    print("\\bottomrule\\end{tabular}\\end{table*}\n")


def table_multi_seed():
    print("% Table 3: Multi-seed validation (n=5)")
    print("\\begin{table}[h]")
    print("\\caption{Multi-seed (n=5) PILoRA-IDS results. Mean $\\pm$ std.}")
    print("\\label{tab:multiseed}")
    print("\\centering\\small")
    print("\\begin{tabular}{lcc}\\toprule")
    print("Dataset & $A_T$ (mean$\\pm$std) & $F_T$ (mean$\\pm$std) \\\\")
    print("\\midrule")
    for ds in DATASETS:
        r = load_pilora_multi(ds)
        if r is None:
            print(f"{ds} & --- & --- \\\\")
            continue
        print(f"{ds} & {r['mean_acc']:.3f}$\\pm${r['std_acc']:.3f} & "
              f"{r['mean_forget']:.3f}$\\pm${r['std_forget']:.3f} \\\\")
    print("\\bottomrule\\end{tabular}\\end{table}\n")


def table_forget_reduction():
    print("% Table 4: Forgetting reduction factor")
    print("\\begin{table}[h]")
    print("\\caption{$F_T$ reduction of PILoRA-IDS vs.\\ strongest Phase-I baseline.}")
    print("\\label{tab:forget-reduction}")
    print("\\centering\\small")
    print("\\begin{tabular}{lccc}\\toprule")
    print("Dataset & Best baseline & Best $F_T$ & PILoRA $F_T$ & Reduction \\\\")
    print("\\midrule")
    PHASE1_BASELINES = ["Replay", "Replay-Herding", "Replay-DPMeans", "iCaRL"]
    for ds in DATASETS:
        p1 = load_phase1(ds)
        p2 = load_pilora_single(ds)
        if not p2:
            continue
        best_m, best_f = min(
            ((m, p1.get(m, {}).get("avg_forgetting", 1.0)) for m in PHASE1_BASELINES),
            key=lambda x: x[1],
        )
        our_f = p2["avg_forgetting"]
        ratio = best_f / max(our_f, 1e-6)
        print(f"{ds} & {best_m} & {best_f:.3f} & {our_f:.3f} & {ratio:.1f}$\\times$ \\\\")
    print("\\bottomrule\\end{tabular}\\end{table}\n")


def main():
    table_main_benchmark()
    table_multi_seed()
    table_forget_reduction()


if __name__ == "__main__":
    main()
