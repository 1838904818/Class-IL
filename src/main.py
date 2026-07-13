"""Main orchestrator — run Class-IL benchmark over datasets × methods × seeds.

Usage:
    python -m src.main                    # default: all datasets, all methods, seed=42
    python -m src.main --seeds 42 43 44   # multi-seed
    python -m src.main --datasets NSL-KDD UNSW-NB15
    python -m src.main --methods Joint Replay
    python -m src.main --skip-figures     # don't auto-generate figures
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.config import SEED, RESULTS_DIR, seed_all
from src.data import DATASET_LOADERS
from src.methods import METHODS
from src.methods.base import make_task_split
from src.metrics import avg_accuracy, avg_forgetting, evaluate_per_class


# ---------------------------------------------------------------------------
# Single-seed dataset run
# ---------------------------------------------------------------------------
def run_one_seed(name, loader_fn, classes_per_task, seed,
                 method_names):
    """Run one (dataset, seed) over all selected methods. Returns a dict."""
    print(f"\n{'=' * 70}\nDataset: {name}  |  seed={seed}")

    seed_all(seed)
    X_tr, y_tr, X_te, y_te, class_order = loader_fn()
    print(f"  Train: {X_tr.shape}  Test: {X_te.shape}  Classes: {len(class_order)}")
    print(f"  Class order: {class_order}")

    # Standardize
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr).astype(np.float32)
    X_te_s = sc.transform(X_te).astype(np.float32)

    in_dim = X_tr_s.shape[1]
    n_classes = len(class_order)
    tasks = make_task_split(n_classes, classes_per_task)
    print(f"  Tasks ({len(tasks)}): {[[class_order[c] for c in t] for t in tasks]}")

    results = {}
    for mname in method_names:
        runner = METHODS[mname]
        t0 = time.time()
        print(f"\n  >> {mname}")

        # Re-seed per method so methods are reproducibly comparable
        seed_all(seed)
        acc_matrix, model = runner(X_tr_s, y_tr, X_te_s, y_te, tasks, in_dim, n_classes)

        aa = avg_accuracy(acc_matrix)
        af = avg_forgetting(acc_matrix)
        elapsed = time.time() - t0
        print(f"     AvgAcc={aa:.4f}  AvgForget={af:.4f}  time={elapsed:.1f}s")

        per_class, _ = evaluate_per_class(model, X_te_s, y_te, n_classes)
        results[mname] = {
            "acc_matrix": acc_matrix.tolist(),
            "avg_accuracy": aa,
            "avg_forgetting": af,
            "per_class_accuracy": per_class.tolist(),
            "time_sec": elapsed,
        }
    return {
        "dataset": name,
        "seed": seed,
        "class_order": class_order,
        "tasks": [[class_order[c] for c in t] for t in tasks],
        "results": results,
    }


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------
def aggregate_seeds(per_seed_outs):
    """Combine multiple single-seed runs into a single dict with mean/std stats."""
    base = per_seed_outs[0]
    out = {
        "dataset":     base["dataset"],
        "class_order": base["class_order"],
        "tasks":       base["tasks"],
        "seeds":       [r["seed"] for r in per_seed_outs],
        "results":     {},
    }
    for mname in base["results"]:
        accs   = [r["results"][mname]["avg_accuracy"] for r in per_seed_outs]
        forgs  = [r["results"][mname]["avg_forgetting"] for r in per_seed_outs]
        times  = [r["results"][mname]["time_sec"] for r in per_seed_outs]
        per_cls = np.array([r["results"][mname]["per_class_accuracy"] for r in per_seed_outs])
        acc_mats = np.array([r["results"][mname]["acc_matrix"] for r in per_seed_outs])
        out["results"][mname] = {
            "acc_matrix":          np.nanmean(acc_mats, axis=0).tolist(),
            "acc_matrix_std":      np.nanstd(acc_mats, axis=0).tolist(),
            "avg_accuracy":        float(np.mean(accs)),
            "avg_accuracy_std":    float(np.std(accs)),
            "avg_forgetting":      float(np.mean(forgs)),
            "avg_forgetting_std":  float(np.std(forgs)),
            "per_class_accuracy":  np.nanmean(per_cls, axis=0).tolist(),
            "per_class_accuracy_std": np.nanstd(per_cls, axis=0).tolist(),
            "time_sec":            float(np.mean(times)),
            "per_seed": {
                "avg_accuracy":   accs,
                "avg_forgetting": forgs,
            },
        }
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[SEED],
                        help="Random seeds; >1 enables multi-seed runs.")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Dataset names; default = all in DATASET_LOADERS.")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Method names; default = all in METHODS.")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Don't auto-run make_figures at the end.")
    parser.add_argument("--out-suffix", default="",
                        help="Append to JSON filename, e.g. '_5seed'.")
    args = parser.parse_args()

    datasets_to_run = args.datasets or list(DATASET_LOADERS.keys())
    methods_to_run = args.methods or list(METHODS.keys())
    seeds = args.seeds

    print(f"Datasets: {datasets_to_run}")
    print(f"Methods : {methods_to_run}")
    print(f"Seeds   : {seeds}")

    for ds_name in datasets_to_run:
        loader_fn, classes_per_task = DATASET_LOADERS[ds_name]
        per_seed = []
        for s in seeds:
            try:
                per_seed.append(run_one_seed(ds_name, loader_fn, classes_per_task,
                                             s, methods_to_run))
            except FileNotFoundError as e:
                print(f"  Skipping {ds_name}: {e}")
                break
        if not per_seed:
            continue
        out = aggregate_seeds(per_seed) if len(per_seed) > 1 else per_seed[0]
        out_path = RESULTS_DIR / f"{ds_name}_results{args.out_suffix}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved: {out_path}")

    print("\n" + "=" * 70)
    print("Experiments done.")

    if not args.skip_figures:
        print("\nGenerating figures ...")
        from src.make_figures import main as figures_main
        figures_main()

    print("\n" + "=" * 70)
    print("ALL DONE.")
    print(f"  Figures: {RESULTS_DIR / 'figures'}")
    print(f"  Summary: {RESULTS_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
