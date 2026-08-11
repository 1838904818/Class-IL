"""LoRA rank ablation for PILoRA-IDS v0.5.

Tests rank ∈ {2, 4, 8, 16, 32} on NSL-KDD (default) to find the
optimal capacity-accuracy trade-off.

Usage:
    python -X utf8 -u -m src_v2.ablation_lora_rank
"""
import argparse
import json
import sys
import time

import numpy as np

from src.config import seed_all, RESULTS_DIR
from src.data import DATASET_LOADERS
from src.methods.base import make_task_split
from src_v2.methods.pilora import run_pilora


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="NSL-KDD")
    parser.add_argument("--ranks", default="2,4,8,16,32")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ranks = [int(s.strip()) for s in args.ranks.split(",")]
    print("=" * 78)
    print(f" PILoRA-IDS LoRA Rank Ablation on {args.dataset}")
    print(f" ranks = {ranks},  seed = {args.seed}")
    print("=" * 78)
    sys.stdout.flush()

    loader_fn, classes_per_task = DATASET_LOADERS[args.dataset]
    X_tr, y_tr, X_te, y_te, class_names = loader_fn()
    n_classes = len(class_names)
    in_dim = X_tr.shape[1]
    tasks = make_task_split(n_classes, classes_per_task=classes_per_task)
    print(f"\nTrain={X_tr.shape}  Test={X_te.shape}  Tasks={len(tasks)}")
    sys.stdout.flush()

    rows = []
    for r in ranks:
        seed_all(args.seed)
        print(f"\n>>> LoRA rank = {r}")
        sys.stdout.flush()
        t0 = time.time()
        acc_matrix, agent = run_pilora(
            X_tr, y_tr, X_te, y_te,
            tasks=tasks,
            in_dim=in_dim,
            n_classes=n_classes,
            d_model=128,
            n_layers=2,
            encoder_type="mlp",
            supervised_pretrain_task0=True,
            pretrain_epochs=8,
            epochs_per_task=10,
            exemplar_capacity=50,
            lora_rank=r,
            verbose=False,
        )
        elapsed = time.time() - t0
        T = len(tasks)
        acc = float(np.nanmean(acc_matrix[T - 1]))
        fgt_per = []
        for j in range(T - 1):
            col = acc_matrix[:T - 1, j]
            if np.isnan(col).all():
                continue
            max_a = float(np.nanmax(col))
            v = acc_matrix[T - 1, j]
            if not np.isnan(v):
                fgt_per.append(max_a - v)
        fgt = float(np.mean(fgt_per)) if fgt_per else 0.0
        params_per_family = agent.pool.num_params_per_family()
        rows.append({
            "rank": r,
            "acc": acc,
            "forget": fgt,
            "time_sec": elapsed,
            "params_per_family": params_per_family,
            "params_total": agent.pool.num_params_total(),
        })
        print(f"  rank={r}: Acc={acc:.4f}  Forget={fgt:.4f}  "
              f"params/family={params_per_family}  Time={elapsed:.1f}s")
        sys.stdout.flush()

    print("\n" + "=" * 78)
    print(f"{'Rank':<8}{'Acc':<10}{'Forget':<10}{'Params/Fam':<14}{'Total':<10}{'Time(s)':<10}")
    print("-" * 78)
    for r in rows:
        print(f"{r['rank']:<8}{r['acc']:.4f}    {r['forget']:.4f}    "
              f"{r['params_per_family']:<14}{r['params_total']:<10}{r['time_sec']:.1f}")

    out = RESULTS_DIR / f"pilora_rank_ablation_{args.dataset.lower().replace('-','_')}.json"
    with open(out, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "seed": args.seed,
            "rows": rows,
        }, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
