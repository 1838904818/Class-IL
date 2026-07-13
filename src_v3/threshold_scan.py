"""Threshold-sensitivity scan for the silent-drift headline.

The silent-drift rate depends on three cut-offs: ACC_DROP_THR (accuracy is
"preserved"), JACCARD_THR (top-k feature set "churned"), and TOP_K. The scan
tests whether the headline ordering (PILoRA lowest, shared-parameter baselines
high) is robust to the selected cut-offs. It does so without re-running any
model, by recomputing
the rate over a grid of (JACCARD_THR x ACC_DROP_THR) from the stored per-
transition metrics, and additionally with Kendall-tau as an alternative churn
metric (Jaccard is the most tie-sensitive choice; tau is continuous).

What this CAN vary from cached JSON: JACCARD_THR, ACC_DROP_THR, and the churn
metric (Jaccard vs tau). What it CANNOT vary here: TOP_K — only top-15 Jaccard
is serialized; varying K needs re-running SHAP and is left as a noted limitation.

Verdict reported: does the PILoRA-min / Naive-max ORDERING survive every cell?

Usage:
    python -X utf8 -u -m src_v3.threshold_scan
    python -X utf8 -u -m src_v3.threshold_scan --datasets NSL-KDD,UNSW-NB15,CIC-IDS-2018
"""
import argparse
import json
from pathlib import Path

import numpy as np

RESULTS = Path("results")
TOP_K = 15
METHOD_ORDER = ["Naive", "Replay", "Replay-DPMeans", "PILoRA"]

JACCARD_GRID = [0.5, 0.6, 0.7, 0.8]
ACC_DROP_GRID = [-0.02, -0.05, -0.10]
# tau alternative: churn when rank-correlation falls below these
TAU_GRID = [0.5, 0.6, 0.7, 0.8]


def load_points(datasets):
    """Per-(dataset,seed,method,class,transition) records from cached JSONs,
    restricted to the requested completed datasets for comparability."""
    jac_key = f"top{TOP_K}_jaccard"
    points = []
    seen = {}
    for p in sorted(RESULTS.glob("shap_stability_*_seed*.json")):
        d = json.load(open(p))
        ds = d["config"]["dataset"]
        if datasets and ds not in datasets:
            continue
        seed = d["config"]["seed"]
        seen.setdefault(ds, set()).add(seed)
        for method, m in d["methods"].items():
            for tr in m["transitions"]:
                for c, met in tr["per_class"].items():
                    if met.get("acc_change") is None:
                        continue
                    points.append({
                        "dataset": ds, "seed": seed, "method": method,
                        "jaccard": met[jac_key], "tau": met["kendall_tau"],
                        "acc_change": met["acc_change"],
                    })
    return points, seen


def rate(points, method, acc_thr, churn_key, churn_thr):
    pts = [p for p in points if p["method"] == method]
    preserved = [p for p in pts if p["acc_change"] > acc_thr]
    churned = [p for p in preserved if p[churn_key] < churn_thr]
    n_p = len(preserved)
    return (100 * len(churned) / n_p if n_p else float("nan"), len(churned), n_p)


def ordering_holds(rates):
    """PILoRA never exceeds any baseline and is strictly below the maximum
    (it may tie a baseline at 0 in some tau cells); Naive is the (weak) max."""
    vals = {m: r for m, r in rates.items() if not np.isnan(r)}
    if "PILoRA" not in vals:
        return False
    pil = vals["PILoRA"]
    others = [v for m, v in vals.items() if m != "PILoRA"]
    naive_ok = ("Naive" not in vals) or (vals["Naive"] >= max(vals.values()) - 1e-9)
    return all(pil <= o + 1e-9 for o in others) and pil < (max(others) if others else pil + 1) and naive_ok


def scan(points, churn_key, churn_grid, label):
    print("\n" + "=" * 92)
    print(f" {label}: silent-drift % over (acc_drop x {churn_key}_thr) grid")
    print(" rows = accuracy-preserved threshold; each cell = Naive / Replay / DPMeans / PILoRA")
    print("=" * 92)
    methods = [m for m in METHOD_ORDER
               if any(p["method"] == m for p in points)]
    header = "acc_drop \\ thr   " + "".join(f"{t:<22}" for t in churn_grid)
    print(header)
    all_hold = True
    cells = {}
    for acc_thr in ACC_DROP_GRID:
        row = f"{acc_thr:<16}"
        for thr in churn_grid:
            rates = {m: rate(points, m, acc_thr, churn_key, thr)[0] for m in methods}
            cells[(acc_thr, thr)] = rates
            hold = ordering_holds(rates)
            all_hold = all_hold and hold
            cellstr = "/".join(
                ("--" if np.isnan(rates[m]) else f"{rates[m]:.0f}") for m in methods)
            mark = "OK" if hold else "XX"
            row += f"{cellstr+' '+mark:<22}"
        print(row)
    return all_hold, cells, methods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="NSL-KDD,UNSW-NB15,CIC-IDS-2018",
                    help="comma list; restricts to completed datasets for comparability")
    args = ap.parse_args()
    datasets = [s.strip() for s in args.datasets.split(",")] if args.datasets else None

    points, seen = load_points(datasets)
    if not points:
        print("No matching shap_stability_*_seed*.json points found.")
        return
    print(f"Loaded {len(points)} points over datasets/seeds:")
    for ds, seeds in sorted(seen.items()):
        print(f"  {ds:<16} seeds={sorted(seeds)}")

    hold_j, cells_j, methods = scan(points, "jaccard", JACCARD_GRID,
                                    "A. Jaccard churn metric")
    hold_t, cells_t, _ = scan(points, "tau", TAU_GRID,
                              "B. Kendall-tau churn metric (Jaccard-independent robustness)")

    print("\n" + "=" * 92)
    print(" VERDICT")
    print("=" * 92)
    print(f" PILoRA-min / Naive-max ordering holds in ALL Jaccard cells: {hold_j}")
    print(f" PILoRA-min / Naive-max ordering holds in ALL tau cells:     {hold_t}")
    # PILoRA staying exactly 0 across cells is the structural-guarantee signature
    pil0 = all(
        (cells_j[k]["PILoRA"] == 0.0 if not np.isnan(cells_j[k]["PILoRA"]) else True)
        for k in cells_j)
    print(f" PILoRA == 0.0% in every Jaccard cell (structural signature):  {pil0}")
    print("\n NOTE: TOP_K held at 15 (only top-15 Jaccard is cached); varying K"
          "\n       requires re-running SHAP and is a stated limitation.")

    out = {
        "datasets": sorted(seen.keys()),
        "seeds": {ds: sorted(s) for ds, s in seen.items()},
        "top_k": TOP_K, "n_points": len(points),
        "jaccard_grid": JACCARD_GRID, "acc_drop_grid": ACC_DROP_GRID,
        "tau_grid": TAU_GRID,
        "jaccard_cells": {f"{a}|{t}": v for (a, t), v in cells_j.items()},
        "tau_cells": {f"{a}|{t}": v for (a, t), v in cells_t.items()},
        "ordering_holds_all_jaccard_cells": bool(hold_j),
        "ordering_holds_all_tau_cells": bool(hold_t),
        "pilora_zero_all_jaccard_cells": bool(pil0),
    }
    outp = RESULTS / "shap_threshold_scan.json"
    with open(outp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {outp}")


if __name__ == "__main__":
    main()
