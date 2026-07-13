"""Run PILoRA-IDS v0.2 across all 4 IDS datasets and compare to Phase I.

For each dataset:
  1. Load and task-split (uses Phase I's default settings)
  2. Run PILoRA-IDS v0.2 with Tier 1+2+3 optimisations
  3. Compute Phase I baseline accuracy from existing JSON
  4. Save result

Then prints a final side-by-side table.

Usage:
    python -X utf8 -u -m src_v2.benchmark_all_datasets [--datasets NSL-KDD,UNSW-NB15,...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.config import seed_all, RESULTS_DIR
from src.data import DATASET_LOADERS
from src.methods.base import make_task_split
from src_v2.methods.pilora import run_pilora


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="NSL-KDD,UNSW-NB15,CIC-IDS-2017,CIC-IDS-2018",
        help="Comma-separated dataset names.",
    )
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--epochs-per-task", type=int, default=10)
    parser.add_argument("--exemplar-capacity", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=8)
    args = parser.parse_args()

    datasets = [s.strip() for s in args.datasets.split(",")]

    print("=" * 75)
    print(f" PILoRA-IDS v0.2 Benchmark — {len(datasets)} datasets")
    print(f"   pretrain_epochs={args.pretrain_epochs}, "
          f"epochs_per_task={args.epochs_per_task}, "
          f"exemplar={args.exemplar_capacity}")
    print("=" * 75)
    sys.stdout.flush()

    all_results = {}

    for ds_name in datasets:
        if ds_name not in DATASET_LOADERS:
            print(f"\n[skip] unknown dataset {ds_name}")
            continue

        print(f"\n{'─' * 75}")
        print(f" Dataset: {ds_name}")
        print(f"{'─' * 75}")
        sys.stdout.flush()
        seed_all(42)

        loader_fn, classes_per_task = DATASET_LOADERS[ds_name]
        try:
            X_tr, y_tr, X_te, y_te, class_names = loader_fn()
        except Exception as e:
            print(f"  [skip] load failed: {e}")
            continue

        n_classes = len(class_names)
        in_dim = X_tr.shape[1]
        tasks = make_task_split(n_classes, classes_per_task=classes_per_task)
        print(f"  Train={X_tr.shape}  Test={X_te.shape}  Classes={n_classes}  Tasks={len(tasks)}")
        sys.stdout.flush()

        t0 = time.time()
        try:
            acc_matrix, agent = run_pilora(
                X_tr, y_tr, X_te, y_te,
                tasks=tasks,
                in_dim=in_dim,
                n_classes=n_classes,
                d_model=args.d_model,
                n_layers=args.n_layers,
                chunk_size=args.chunk_size,
                lora_rank=args.lora_rank,
                pretrain_epochs=args.pretrain_epochs,
                pretrain_samples=None,
                epochs_per_task=args.epochs_per_task,
                exemplar_capacity=args.exemplar_capacity,
                verbose=True,
            )
        except Exception as e:
            print(f"  [error] {type(e).__name__}: {e}")
            continue
        elapsed = time.time() - t0

        T = len(tasks)
        final_acc = float(np.nanmean(acc_matrix[T - 1]))
        forget_per_task = []
        for j in range(T - 1):
            col = acc_matrix[:T - 1, j]
            if np.isnan(col).all():
                continue
            max_acc = float(np.nanmax(col))
            final_v = acc_matrix[T - 1, j]
            if not np.isnan(final_v):
                forget_per_task.append(max_acc - final_v)
        forget = float(np.mean(forget_per_task)) if forget_per_task else 0.0

        print(f"\n  >> {ds_name} v0.2 result")
        print(f"     AvgAcc       = {final_acc:.4f}")
        print(f"     AvgForget    = {forget:.4f}")
        print(f"     Wall time    = {elapsed:.1f}s")
        print(f"     Families     = {len(agent.pool.families)}")
        print(f"     Centroids    = {agent.router.n_centroids_total()}")
        print(f"     LoRA total   = {agent.pool.num_params_total()}")
        print(f"     Buffer size  = {agent.buffer.n_total()}")
        sys.stdout.flush()

        all_results[ds_name] = {
            "method": "PILoRA-IDS-v0.2",
            "avg_accuracy": final_acc,
            "avg_forgetting": forget,
            "wall_time_sec": elapsed,
            "acc_matrix": acc_matrix.tolist(),
            "n_families": len(agent.pool.families),
            "n_centroids": agent.router.n_centroids_total(),
            "lora_total": agent.pool.num_params_total(),
            "buffer_n": agent.buffer.n_total(),
            "config": {
                "pretrain_epochs": args.pretrain_epochs,
                "epochs_per_task": args.epochs_per_task,
                "exemplar_capacity": args.exemplar_capacity,
                "d_model": args.d_model,
                "n_layers": args.n_layers,
                "lora_rank": args.lora_rank,
                "chunk_size": args.chunk_size,
            },
        }
        # Save incrementally so we don't lose progress on long runs
        out = RESULTS_DIR / "fedmac_v02_all_datasets.json"
        with open(out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved (incremental): {out}")
        sys.stdout.flush()

    # ----- final comparison table
    print("\n" + "=" * 90)
    print(" Phase I vs PILoRA-IDS v0.2 — Final Comparison")
    print("=" * 90)
    print(f"\n{'Dataset':<15}{'Phase I Best':<25}{'PILoRA v0.2':<25}{'Gap':<10}")
    print("-" * 90)
    for ds_name in datasets:
        if ds_name not in all_results:
            continue
        # Read Phase I best
        p1_path = RESULTS_DIR / f"{ds_name}_results.json"
        if not p1_path.exists():
            continue
        p1 = json.load(open(p1_path)).get("results", {})
        # Find best (highest acc among CL methods, excluding Joint/Naive)
        best_method, best_acc = None, -1.0
        for m in ["Replay", "Replay-Herding", "Replay-DPMeans", "iCaRL"]:
            v = p1.get(m, {}).get("avg_accuracy", 0)
            if v > best_acc:
                best_acc = v
                best_method = m
        v2_acc = all_results[ds_name]["avg_accuracy"]
        gap = v2_acc - best_acc
        gap_str = f"{gap:+.4f}"
        print(f"{ds_name:<15}{best_method+' '+f'{best_acc:.4f}':<25}"
              f"{f'{v2_acc:.4f}':<25}{gap_str:<10}")

    print(f"\nResults: {RESULTS_DIR / 'fedmac_v02_all_datasets.json'}")


if __name__ == "__main__":
    main()
