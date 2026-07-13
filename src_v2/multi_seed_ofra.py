"""Multi-seed OFRA benchmark across selected NIDS datasets.

Runs OFRA with seeds {42, 0, 1, 2, 3} on each dataset; saves
per-seed and aggregate (mean ± std) accuracy + forgetting.

Usage:
    python -X utf8 -u -m src_v2.multi_seed_ofra [--datasets NSL-KDD,...]
                                                  [--seeds 42,0,1,2,3]
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
from src_v2.methods.ofra import run_ofra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",
                        default="NSL-KDD,UNSW-NB15,CIC-IDS-2017,CIC-IDS-2018,NF-ToN-IoT-v2")
    parser.add_argument("--seeds", default="42,0,1,2,3")
    parser.add_argument("--pretrain-epochs", type=int, default=8)
    parser.add_argument("--epochs-per-task", type=int, default=10)
    parser.add_argument("--exemplar-capacity", type=int, default=50)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument(
        "--encoder-type",
        choices=("mlp", "transformer"),
        default="mlp",
    )
    parser.add_argument(
        "--loss-fn",
        choices=("focal", "ce"),
        default="focal",
    )
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--out-suffix", default="5seed")
    args = parser.parse_args()

    datasets = [s.strip() for s in args.datasets.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    print("=" * 78)
    print(f" OFRA Multi-Seed: {len(datasets)} datasets × {len(seeds)} seeds")
    print(f"   encoder={args.encoder_type}, loss={args.loss_fn}, rank={args.lora_rank}")
    print("=" * 78)
    sys.stdout.flush()

    all_results = {}

    for ds_name in datasets:
        if ds_name not in DATASET_LOADERS:
            continue
        completed_seeds = []
        per_seed_acc, per_seed_forget, per_seed_time = [], [], []

        print(f"\n{'─' * 78}\n Dataset: {ds_name}\n{'─' * 78}")
        sys.stdout.flush()

        loader_fn, classes_per_task = DATASET_LOADERS[ds_name]
        try:
            X_tr, y_tr, X_te, y_te, class_names = loader_fn()
        except Exception as e:
            print(f"  [skip] {e}")
            continue
        n_classes = len(class_names)
        in_dim = X_tr.shape[1]
        tasks = make_task_split(n_classes, classes_per_task=classes_per_task)
        print(f"  Tasks: {tasks}")
        sys.stdout.flush()

        for seed in seeds:
            print(f"\n  >>> Seed {seed}")
            sys.stdout.flush()
            seed_all(seed)
            t0 = time.time()
            try:
                acc_matrix, agent = run_ofra(
                    X_tr, y_tr, X_te, y_te,
                    tasks=tasks,
                    in_dim=in_dim,
                    n_classes=n_classes,
                    d_model=args.d_model,
                    n_layers=args.n_layers,
                    encoder_type=args.encoder_type,
                    supervised_pretrain_task0=(args.encoder_type == "mlp"),
                    pretrain_epochs=args.pretrain_epochs,
                    pretrain_samples=None,
                    epochs_per_task=args.epochs_per_task,
                    exemplar_capacity=args.exemplar_capacity,
                    lora_rank=args.lora_rank,
                    loss_fn=args.loss_fn,
                    verbose=False,
                )
            except Exception as e:
                print(f"     [error] {type(e).__name__}: {e}")
                continue
            elapsed = time.time() - t0

            T = len(tasks)
            acc = float(np.nanmean(acc_matrix[T - 1]))
            fgt_per = []
            for j in range(T - 1):
                col = acc_matrix[:T - 1, j]
                if np.isnan(col).all():
                    continue
                max_acc = float(np.nanmax(col))
                final_v = acc_matrix[T - 1, j]
                if not np.isnan(final_v):
                    fgt_per.append(max_acc - final_v)
            fgt = float(np.mean(fgt_per)) if fgt_per else 0.0

            per_seed_acc.append(acc)
            per_seed_forget.append(fgt)
            per_seed_time.append(elapsed)
            completed_seeds.append(seed)
            print(f"     Acc={acc:.4f}  Forget={fgt:.4f}  Time={elapsed:.1f}s")
            sys.stdout.flush()

        if per_seed_acc:
            mean_acc = float(np.mean(per_seed_acc))
            std_acc = float(np.std(per_seed_acc, ddof=1)) if len(per_seed_acc) > 1 else 0.0
            mean_fgt = float(np.mean(per_seed_forget))
            std_fgt = float(np.std(per_seed_forget, ddof=1)) if len(per_seed_forget) > 1 else 0.0
            print(f"\n  {ds_name}: Acc={mean_acc:.4f}±{std_acc:.4f}  "
                  f"Forget={mean_fgt:.4f}±{std_fgt:.4f}")
            sys.stdout.flush()

            all_results[ds_name] = {
                "method": "OFRA-IDS-v0.5",
                "seeds": completed_seeds,
                "per_seed_acc": per_seed_acc,
                "per_seed_forget": per_seed_forget,
                "per_seed_time": per_seed_time,
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "mean_forget": mean_fgt,
                "std_forget": std_fgt,
                "config": vars(args),
            }
            out = RESULTS_DIR / f"ofra_{args.out_suffix}.json"
            with open(out, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  Saved (incremental): {out}")

    # ------ Summary table
    print("\n" + "=" * 90)
    print(f" OFRA-IDS v0.5 Multi-Seed Summary ({len(seeds)} seeds)")
    print("=" * 90)
    print(f"{'Dataset':<15}{'Mean Acc':<18}{'Mean Forget':<18}{'Mean Time(s)':<15}")
    print("-" * 90)
    for ds, r in all_results.items():
        print(f"{ds:<15}{r['mean_acc']:.4f}±{r['std_acc']:.4f}    "
              f"{r['mean_forget']:.4f}±{r['std_forget']:.4f}    "
              f"{float(np.mean(r['per_seed_time'])):>8.1f}")


if __name__ == "__main__":
    main()
