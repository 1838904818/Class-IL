"""PILoRA-IDS v0.4 — MLP encoder + LoRA + DPMeans router (Option B).

Replaces the FlowTransformer encoder with Phase I's proven MLP backbone.
Tests whether the LoRA + DPMeans architecture is fundamentally sound, or
whether v0.3's plateau at 0.38 was a transformer-specific limitation.

Run:
    python -X utf8 -u -m src_v2.smoke_test_mlp_encoder
"""
from __future__ import annotations

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
    seed_all(42)
    sys.stdout.flush()

    print("=" * 70)
    print(" PILoRA-IDS v0.4 — Option B (MLP encoder + LoRA + DPMeans)")
    print("=" * 70)
    sys.stdout.flush()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="NSL-KDD")
    args = parser.parse_args()
    loader_fn, classes_per_task = DATASET_LOADERS[args.dataset]
    DATASET = args.dataset
    X_tr, y_tr, X_te, y_te, class_names = loader_fn()
    n_classes = len(class_names)
    in_dim = X_tr.shape[1]
    tasks = make_task_split(n_classes, classes_per_task=classes_per_task)

    sys.stdout.flush()
    print(f"\nTrain: {X_tr.shape}  Test: {X_te.shape}  Classes: {n_classes}")
    print(f"Class order: {class_names}")
    print(f"Tasks ({len(tasks)}): {[[class_names[c] for c in t] for t in tasks]}")
    sys.stdout.flush()

    print("\n>> PILoRA-IDS v0.4 (MLP encoder)")
    sys.stdout.flush()
    t0 = time.time()
    acc_matrix, agent = run_pilora(
        X_tr, y_tr, X_te, y_te,
        tasks=tasks,
        in_dim=in_dim,
        n_classes=n_classes,
        d_model=128,
        n_layers=2,
        encoder_type="mlp",            # ← key change
        supervised_pretrain_task0=True,
        pretrain_epochs=8,
        lora_rank=8,
        epochs_per_task=10,
        exemplar_capacity=50,
        verbose=True,
    )
    elapsed = time.time() - t0

    T = len(tasks)
    final_acc = float(np.nanmean(acc_matrix[T - 1]))
    forget = []
    for j in range(T - 1):
        col = acc_matrix[:T - 1, j]
        if np.isnan(col).all():
            continue
        max_acc = float(np.nanmax(col))
        final_v = acc_matrix[T - 1, j]
        if not np.isnan(final_v):
            forget.append(max_acc - final_v)
    forgetting = float(np.mean(forget)) if forget else 0.0

    print(f"\n  AvgAcc       = {final_acc:.4f}")
    print(f"  AvgForget    = {forgetting:.4f}")
    print(f"  Wall time    = {elapsed:.1f}s")
    print(f"  Families     = {len(agent.pool.families)}")
    print(f"  Centroids    = {agent.router.n_centroids_total()}")
    print(f"  LoRA params  = {agent.pool.num_params_total()}")
    print(f"  Buffer       = {agent.buffer.n_total()}")
    sys.stdout.flush()

    # Phase I comparison
    phase1_path = RESULTS_DIR / f"{DATASET}_results.json"
    if phase1_path.exists():
        phase1 = json.load(open(phase1_path)).get("results", {})
        print("\n=== Comparison ===")
        print(f"  {'Method':<25}{'Acc':<10}{'Forget':<10}")
        print(f"  {'-'*45}")
        for method in ["Naive (lower bound)", "EWC", "LwF", "Replay",
                       "Replay-Herding", "Replay-DPMeans", "iCaRL",
                       "Joint (upper bound)"]:
            if method in phase1:
                a = phase1[method].get("avg_accuracy", 0)
                f = phase1[method].get("avg_forgetting", 0)
                print(f"  {method:<25}{a:<10.4f}{f:<10.4f}")
        print(f"  {'PILoRA v0.4 (Option B)':<25}{final_acc:<10.4f}{forgetting:<10.4f}")

    # Save
    suffix = DATASET.lower().replace("-", "_")
    out_path = RESULTS_DIR / f"fedmac_v04_mlp_{suffix}.json"
    with open(out_path, "w") as f:
        json.dump({
            "dataset": DATASET,
            "method": "FedMAC-v0.4-MLP",
            "encoder_type": "mlp",
            "avg_accuracy": final_acc,
            "avg_forgetting": forgetting,
            "wall_time_sec": elapsed,
            "acc_matrix": acc_matrix.tolist(),
            "n_families": len(agent.pool.families),
            "n_centroids": agent.router.n_centroids_total(),
            "lora_params_total": agent.pool.num_params_total(),
            "buffer_n": agent.buffer.n_total(),
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
