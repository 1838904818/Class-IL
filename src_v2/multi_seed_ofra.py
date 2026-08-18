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
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.config import DATA_DIR, RESULTS_DIR, SEED, seed_all
from src.methods.base import make_task_split
from src_v2.methods.ofra import (
    DEFAULT_PREDICTION_ARMS,
    PREDICTION_CALIBRATION,
    normalize_prediction_arms,
    run_ofra,
)
from src_v2.reproducibility import (
    array_manifest,
    canonical_sha256,
    code_manifest,
    dataset_manifest,
    environment_manifest,
)


def parse_prediction_arms(value: str) -> dict[str, dict[str, float]]:
    """Resolve a comma-separated list of built-in prediction arm names."""
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise ValueError("--prediction-arms must select at least one arm")
    if len(set(names)) != len(names):
        raise ValueError("--prediction-arms contains duplicate names")
    unknown = [name for name in names if name not in DEFAULT_PREDICTION_ARMS]
    if unknown:
        raise ValueError(
            f"Unknown prediction arm(s): {unknown}; expected a subset of "
            f"{list(DEFAULT_PREDICTION_ARMS)}"
        )
    return normalize_prediction_arms({
        name: DEFAULT_PREDICTION_ARMS[name]
        for name in names
    })


def main():
    # Dataset loaders are needed only for an actual benchmark run. Keep this
    # optional data-layer import out of prediction-arm and diagnostic utilities.
    from src.data import DATASET_LOADERS

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
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument("--router-fit-max-samples", type=int, default=None)
    parser.add_argument(
        "--prediction-arms",
        default="p-only,z-only,joint",
        help=(
            "comma-separated diagnostic arms selected from "
            "p-only,z-only,joint"
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out-suffix", default="5seed")
    args = parser.parse_args()

    required_environment = {
        "PYTHONHASHSEED": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    mismatched_environment = {
        key: {"expected": expected, "actual": os.environ.get(key)}
        for key, expected in required_environment.items()
        if os.environ.get(key) != expected
    }
    if mismatched_environment:
        raise RuntimeError(
            "Strict reproducibility environment is not configured: "
            f"{mismatched_environment}"
        )

    datasets = [s.strip() for s in args.datasets.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    prediction_arms = parse_prediction_arms(args.prediction_arms)
    prediction_arm_protocol = {
        name: {
            **config,
            "calibration": PREDICTION_CALIBRATION,
        }
        for name, config in prediction_arms.items()
    }
    unknown_datasets = [name for name in datasets if name not in DATASET_LOADERS]
    if unknown_datasets:
        raise ValueError(f"Unknown dataset(s): {unknown_datasets}")
    if not seeds:
        raise ValueError("At least one seed is required")
    if args.router_fit_max_samples is not None and args.router_fit_max_samples <= 0:
        raise ValueError("--router-fit-max-samples must be positive")
    resolved_device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    run_config = {
        **vars(args),
        "resolved_device": resolved_device,
        "prediction_arms_resolved": prediction_arm_protocol,
    }
    base_protocol_config = {
        "schema_version": 3,
        "datasets": datasets,
        "seeds": seeds,
        "split_seed": SEED,
        "model": {
            "pretrain_epochs": args.pretrain_epochs,
            "epochs_per_task": args.epochs_per_task,
            "exemplar_capacity": args.exemplar_capacity,
            "lora_rank": args.lora_rank,
            "encoder_type": args.encoder_type,
            "loss_fn": args.loss_fn,
            "n_layers": args.n_layers,
            "d_model": args.d_model,
            "resolved_device": resolved_device,
            "router_fit_max_samples": args.router_fit_max_samples,
            "batch_size": 256,
            "learning_rate": 1e-3,
            "pretrain_batch_size": 256,
            "pretrain_learning_rate": 1e-3,
            "supervised_pretrain_task0": args.encoder_type == "mlp",
            "pretrain_samples": None,
            "chunk_size": 8,
            "lora_alpha": 16.0,
            "focal_gamma": 2.0,
            "focal_alpha": 0.75,
            "minority_threshold": 1000,
            "negative_ratio": 4,
            "router_lambda_quantile": 0.30,
            "router_novelty_factor": 1.5,
            "router_max_centroids_per_family": 32,
            "router_distance_batch_size": 65536,
            "prediction_router_weight": 0.5,
            "prediction_head_weight": 1.0,
            "prediction_calibration": "softmax_prob",
            "legacy_acc_matrix_arm": "joint",
            "prediction_arms": prediction_arm_protocol,
            "checkpoint_diagnostics": {
                "scope": "all test samples from classes seen by checkpoint",
                "metrics": [
                    "accuracy",
                    "confusion_matrix",
                    "per_class_recall",
                    "per_class_f1",
                    "macro_f1",
                    "balanced_accuracy",
                ],
            },
        },
        "data_controls": {
            "FULL_DATA": os.environ.get("FULL_DATA"),
            "CIC17_NORMAL_CAP": os.environ.get("CIC17_NORMAL_CAP"),
            "MAX_PER_CLASS": os.environ.get("MAX_PER_CLASS"),
        },
    }
    code_provenance = code_manifest(Path(__file__).resolve().parents[1])
    environment_provenance = environment_manifest()

    print("=" * 78)
    print(f" OFRA Multi-Seed: {len(datasets)} datasets × {len(seeds)} seeds")
    print(f"   encoder={args.encoder_type}, loss={args.loss_fn}, rank={args.lora_rank}")
    print(f"   device={resolved_device}, router_fit_max={args.router_fit_max_samples or 'uncapped'}")
    print("=" * 78)
    sys.stdout.flush()

    all_results = {}

    for ds_name in datasets:
        completed_seeds = []
        per_seed_acc, per_seed_forget, per_seed_time = [], [], []
        per_seed_matrices = []
        per_seed_router_fit = []
        per_seed_diagnostics = []

        print(f"\n{'─' * 78}\n Dataset: {ds_name}\n{'─' * 78}")
        sys.stdout.flush()

        loader_fn, classes_per_task = DATASET_LOADERS[ds_name]
        print("  Hashing exact dataset inputs for provenance...")
        dataset_provenance = dataset_manifest(DATA_DIR, ds_name)
        X_tr, y_tr, X_te, y_te, class_names = loader_fn()
        n_classes = len(class_names)
        in_dim = X_tr.shape[1]
        tasks = make_task_split(n_classes, classes_per_task=classes_per_task)
        input_provenance = {
            "X_tr": array_manifest(X_tr),
            "y_tr": array_manifest(y_tr),
            "X_te": array_manifest(X_te),
            "y_te": array_manifest(y_te),
        }
        dataset_protocol = {
            "name": ds_name,
            "class_names": list(class_names),
            "classes_per_task": classes_per_task,
            "tasks": tasks,
            "train_rows": int(len(X_tr)),
            "test_rows": int(len(X_te)),
            "features": int(in_dim),
        }
        protocol_config = {**base_protocol_config, "dataset": dataset_protocol}
        protocol_sha256 = canonical_sha256(protocol_config)
        run_fingerprint_sha256 = canonical_sha256({
            "protocol_sha256": protocol_sha256,
            "code_manifest_sha256": code_provenance["manifest_sha256"],
            "dataset_manifest_sha256": dataset_provenance["manifest_sha256"],
            "processed_inputs": input_provenance,
            "environment": environment_provenance,
        })
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
                    verbose=args.verbose,
                    device=resolved_device,
                    router_fit_max_samples=args.router_fit_max_samples,
                    prediction_arms=prediction_arms,
                    class_names=class_names,
                )
            except Exception as e:
                print(f"     [error] {type(e).__name__}: {e}")
                raise
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
            per_seed_matrices.append([
                [None if np.isnan(value) else float(value) for value in row]
                for row in acc_matrix
            ])
            per_seed_router_fit.append(dict(agent.router.fit_sample_counts))
            if agent.checkpoint_diagnostics is None:
                raise RuntimeError(
                    "run_ofra did not return requested checkpoint diagnostics"
                )
            if len(agent.checkpoint_diagnostics["checkpoints"]) != len(tasks):
                raise RuntimeError(
                    "checkpoint diagnostic count does not match task count"
                )
            per_seed_diagnostics.append({
                "seed": int(seed),
                "checkpoints": agent.checkpoint_diagnostics["checkpoints"],
            })
            completed_seeds.append(seed)
            print(f"     Acc={acc:.4f}  Forget={fgt:.4f}  Time={elapsed:.1f}s")
            sys.stdout.flush()

        if completed_seeds != seeds:
            raise RuntimeError(
                f"Completed seeds {completed_seeds}, expected {seeds}"
            )

        if per_seed_acc:
            mean_acc = float(np.mean(per_seed_acc))
            std_acc = float(np.std(per_seed_acc, ddof=1)) if len(per_seed_acc) > 1 else 0.0
            mean_fgt = float(np.mean(per_seed_forget))
            std_fgt = float(np.std(per_seed_forget, ddof=1)) if len(per_seed_forget) > 1 else 0.0
            print(f"\n  {ds_name}: Acc={mean_acc:.4f}±{std_acc:.4f}  "
                  f"Forget={mean_fgt:.4f}±{std_fgt:.4f}")
            sys.stdout.flush()

            result_record = {
                "method": "OFRA-IDS-v0.5",
                "seeds": completed_seeds,
                "per_seed_acc": per_seed_acc,
                "per_seed_forget": per_seed_forget,
                "per_seed_time": per_seed_time,
                "per_seed_acc_matrix": per_seed_matrices,
                "per_seed_router_fit": per_seed_router_fit,
                "diagnostics": {
                    "schema_version": 1,
                    "scope": (
                        "test samples from all classes seen by each checkpoint"
                    ),
                    "legacy_acc_matrix_arm": {
                        "name": "joint",
                        "head_weight": 1.0,
                        "router_weight": 0.5,
                        "calibration": PREDICTION_CALIBRATION,
                    },
                    "prediction_arms": prediction_arm_protocol,
                    "per_seed": per_seed_diagnostics,
                },
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "mean_forget": mean_fgt,
                "std_forget": std_fgt,
                "dataset_summary": {
                    "train_rows": int(len(X_tr)),
                    "test_rows": int(len(X_te)),
                    "features": int(in_dim),
                    "class_names": list(class_names),
                    "train_class_counts": {
                        class_names[index]: int(np.count_nonzero(y_tr == index))
                        for index in range(n_classes)
                    },
                    "test_class_counts": {
                        class_names[index]: int(np.count_nonzero(y_te == index))
                        for index in range(n_classes)
                    },
                },
                "config": run_config,
                "protocol": protocol_config,
                "protocol_sha256": protocol_sha256,
                "provenance": {
                    "run_fingerprint_sha256": run_fingerprint_sha256,
                    "code": code_provenance,
                    "dataset": dataset_provenance,
                    "processed_inputs": input_provenance,
                    "environment": environment_provenance,
                },
            }
            deterministic_payload = {
                key: value
                for key, value in result_record.items()
                if key not in {"per_seed_time", "config"}
            }
            result_record["deterministic_result_sha256"] = canonical_sha256(
                deterministic_payload
            )
            all_results[ds_name] = result_record
            out = RESULTS_DIR / f"ofra_{args.out_suffix}.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, allow_nan=False)
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
