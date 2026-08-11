"""PILoRA-IDS v0.6 — optimal config (d=1, w=128 encoder + focal loss + rank=8).

Confirms the depth=1 encoder finding on NSL-KDD and (optionally) other datasets.
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
    args = parser.parse_args()

    seed_all(42)
    print(f"=" * 70)
    print(f" PILoRA-IDS v0.6 (depth=1 encoder) on {args.dataset}")
    print(f"=" * 70)
    sys.stdout.flush()

    loader_fn, classes_per_task = DATASET_LOADERS[args.dataset]
    X_tr, y_tr, X_te, y_te, class_names = loader_fn()
    n_classes = len(class_names)
    in_dim = X_tr.shape[1]
    tasks = make_task_split(n_classes, classes_per_task=classes_per_task)

    t0 = time.time()
    acc_matrix, agent = run_pilora(
        X_tr, y_tr, X_te, y_te,
        tasks=tasks,
        in_dim=in_dim,
        n_classes=n_classes,
        d_model=128,
        n_layers=1,                  # ← key change: depth=1
        encoder_type="mlp",
        supervised_pretrain_task0=True,
        pretrain_epochs=8,
        epochs_per_task=10,
        lora_rank=8,
        exemplar_capacity=50,
        verbose=False,
    )
    elapsed = time.time() - t0

    T = len(tasks)
    acc = float(np.nanmean(acc_matrix[T - 1]))
    fgt = []
    for j in range(T - 1):
        col = acc_matrix[:T - 1, j]
        if np.isnan(col).all(): continue
        max_a = float(np.nanmax(col))
        v = acc_matrix[T - 1, j]
        if not np.isnan(v): fgt.append(max_a - v)
    f = float(np.mean(fgt)) if fgt else 0.0

    print(f"\n  AvgAcc       = {acc:.4f}")
    print(f"  AvgForget    = {f:.4f}")
    print(f"  Wall time    = {elapsed:.1f}s")
    print(f"  Encoder params={sum(p.numel() for p in agent.encoder.parameters())}")
    sys.stdout.flush()

    out = RESULTS_DIR / f"pilora_v06_{args.dataset.lower().replace('-','_')}.json"
    with open(out, "w") as fh:
        json.dump({
            "dataset": args.dataset,
            "method": "PILoRA-IDS-v0.6",
            "config": {"d_model": 128, "n_layers": 1, "lora_rank": 8},
            "avg_accuracy": acc,
            "avg_forgetting": f,
            "wall_time_sec": elapsed,
            "acc_matrix": acc_matrix.tolist(),
        }, fh, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
