"""Single-site smoke test for PILoRA-IDS Phase II prototype.

Trains a single PILoRAAgent on NSL-KDD with the Phase I task split and
compares its final accuracy to the Phase I baselines (Replay-DPMeans, iCaRL).

Run:
    python -X utf8 -u -m src_v2.smoke_test_single_site
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

    print("=" * 70)
    print("PILoRA-IDS Single-Site Smoke Test (NSL-KDD)")
    print("=" * 70)

    # ---------------- load data
    loader_fn, classes_per_task = DATASET_LOADERS["NSL-KDD"]
    X_tr, y_tr, X_te, y_te, class_names = loader_fn()
    n_classes = len(class_names)
    in_dim = X_tr.shape[1]
    tasks = make_task_split(n_classes, classes_per_task=classes_per_task)
    sys.stdout.flush()
    print(f"\nTrain: {X_tr.shape}  Test: {X_te.shape}  Classes: {n_classes}")
    print(f"Class order: {class_names}")
    print(f"Tasks ({len(tasks)}): {[[class_names[c] for c in t] for t in tasks]}")

    # ---------------- run PILoRA-IDS v0.2 (Tier 1+2+3 optimised)
    #   Tier 1: pretrain=5 ep, full data, task epochs=10
    #   Tier 2: scalar-head + joint-softmax (in fedmac.py)
    #   Tier 3: raw exemplar buffer 50/family (in fedmac.py)
    print("\n>> PILoRA-IDS v0.2 (single-site)")
    sys.stdout.flush()
    t0 = time.time()
    acc_matrix, agent = run_pilora(
        X_tr, y_tr, X_te, y_te,
        tasks=tasks,
        in_dim=in_dim,
        n_classes=n_classes,
        d_model=64,
        n_layers=2,
        lora_rank=8,
        pretrain_epochs=5,
        pretrain_samples=None,        # full data (Tier 1)
        epochs_per_task=10,
        exemplar_capacity=50,
        verbose=True,
    )
    elapsed = time.time() - t0

    # ---------------- summarise
    T = len(tasks)
    final_acc = float(np.nanmean(acc_matrix[T - 1]))
    forgetting_per_task = []
    for j in range(T - 1):
        max_acc = np.nanmax(acc_matrix[:T - 1, j]) if not np.isnan(acc_matrix[:T - 1, j]).all() else 0
        final = acc_matrix[T - 1, j] if not np.isnan(acc_matrix[T - 1, j]) else 0
        forgetting_per_task.append(max_acc - final)
    forgetting = float(np.mean(forgetting_per_task)) if forgetting_per_task else 0.0

    print(f"\n  AvgAcc       = {final_acc:.4f}")
    print(f"  AvgForget    = {forgetting:.4f}")
    print(f"  Wall time    = {elapsed:.1f}s")
    print(f"\nRouter summary:")
    print(agent.router.summary())
    n_lora_total = agent.pool.num_params_total()
    n_lora_per_family = agent.pool.num_params_per_family()
    print(f"\nLoRA pool: {len(agent.pool.families)} families, "
          f"{n_lora_per_family:,} params each, {n_lora_total:,} total")

    # ---------------- compare to Phase I baselines (load if available)
    phase1_path = RESULTS_DIR / "NSL-KDD_results.json"
    if phase1_path.exists():
        with open(phase1_path) as f:
            phase1 = json.load(f)
        print("\nPhase I baselines on the same task split:")
        for method, m in phase1.get("results", {}).items():
            acc = m.get("avg_accuracy")
            fgt = m.get("avg_forgetting")
            if acc is not None and fgt is not None:
                print(f"  {method:20s}  Acc={acc:.4f}  Forget={fgt:.4f}")
        print(f"  {'PILoRA (Phase II)':20s}  Acc={final_acc:.4f}  "
              f"Forget={forgetting:.4f}")

    # ---------------- save
    out_path = RESULTS_DIR / "fedmac_single_site_nsl_kdd.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "dataset": "NSL-KDD",
                "method": "PILoRA-IDS-SingleSite",
                "avg_accuracy": final_acc,
                "avg_forgetting": forgetting,
                "wall_time_sec": elapsed,
                "acc_matrix": acc_matrix.tolist(),
                "n_families": len(agent.pool.families),
                "lora_params_per_family": n_lora_per_family,
                "lora_params_total": n_lora_total,
                "n_centroids_total": agent.router.n_centroids_total(),
                "centroid_bytes": agent.router.comm_payload_bytes(),
                "comment": "Phase II prototype v0.1 — Transformer + LoRA + DPMeans routing",
            },
            f,
            indent=2,
        )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
