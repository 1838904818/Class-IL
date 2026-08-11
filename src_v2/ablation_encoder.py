"""Encoder ablation for PILoRA-IDS v0.5.

Tests MLP encoder depth × width on NSL-KDD.
  depths ∈ {1, 2, 3}
  widths ∈ {64, 128, 256}

Usage:
    python -X utf8 -u -m src_v2.ablation_encoder
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
    parser.add_argument("--depths", default="1,2,3")
    parser.add_argument("--widths", default="64,128,256")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    depths = [int(s) for s in args.depths.split(",")]
    widths = [int(s) for s in args.widths.split(",")]
    print("=" * 78)
    print(f" PILoRA-IDS Encoder Ablation on {args.dataset}")
    print(f" depths × widths = {depths} × {widths}, seed = {args.seed}")
    print("=" * 78)
    sys.stdout.flush()

    loader_fn, classes_per_task = DATASET_LOADERS[args.dataset]
    X_tr, y_tr, X_te, y_te, class_names = loader_fn()
    n_classes = len(class_names)
    in_dim = X_tr.shape[1]
    tasks = make_task_split(n_classes, classes_per_task=classes_per_task)

    rows = []
    for d in depths:
        for w in widths:
            seed_all(args.seed)
            print(f"\n>>> depth={d}, width={w}")
            sys.stdout.flush()
            t0 = time.time()
            try:
                acc_matrix, agent = run_pilora(
                    X_tr, y_tr, X_te, y_te,
                    tasks=tasks,
                    in_dim=in_dim,
                    n_classes=n_classes,
                    d_model=w,
                    n_layers=d,
                    encoder_type="mlp",
                    supervised_pretrain_task0=True,
                    pretrain_epochs=8,
                    epochs_per_task=10,
                    exemplar_capacity=50,
                    lora_rank=8,
                    verbose=False,
                )
            except Exception as e:
                print(f"  [error] {type(e).__name__}: {e}")
                continue
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
            encoder_params = sum(p.numel() for p in agent.encoder.parameters())
            rows.append({
                "depth": d,
                "width": w,
                "acc": acc,
                "forget": fgt,
                "time_sec": elapsed,
                "encoder_params": encoder_params,
                "lora_total": agent.pool.num_params_total(),
            })
            print(f"  d={d}, w={w}: Acc={acc:.4f}  Forget={fgt:.4f}  "
                  f"encoder={encoder_params}  Time={elapsed:.1f}s")
            sys.stdout.flush()

    print("\n" + "=" * 78)
    print(f"{'Depth':<8}{'Width':<8}{'Acc':<10}{'Forget':<10}{'Encoder':<12}{'Time(s)':<10}")
    print("-" * 78)
    for r in rows:
        print(f"{r['depth']:<8}{r['width']:<8}{r['acc']:.4f}    "
              f"{r['forget']:.4f}    {r['encoder_params']:<12}{r['time_sec']:.1f}")

    out = RESULTS_DIR / f"pilora_encoder_ablation_{args.dataset.lower().replace('-','_')}.json"
    with open(out, "w") as f:
        json.dump({"dataset": args.dataset, "seed": args.seed, "rows": rows},
                  f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
