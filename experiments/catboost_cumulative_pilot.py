"""Hash-bound cumulative CatBoost diagnostic on one streaming dataset.

This is a classifier ceiling/diagnostic, not OFRA.  Each checkpoint trains a
fresh multiclass CatBoost model on all training rows from classes seen so far.
CPU training is intentional because CatBoost documents GPU training as
non-deterministic due to floating-point summation order.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from catboost import CatBoostClassifier, Pool, __version__ as catboost_version

from streaming_full.data import (
    FrozenTask0Stats,
    array_sha256,
    canonical_sha256,
    load_evaluation_view,
    load_manifest,
    sha256_file,
)


SCHEMA_VERSION = 1
ARM = "cumulative_multiclass"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluation-view", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-leaf-reg", type=float, default=3.0)
    parser.add_argument("--random-strength", type=float, default=1.0)
    parser.add_argument("--thread-count", type=int, default=8)
    parser.add_argument("--shap-sample-per-class", type=int, default=64)
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-name")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_split(record, split: str) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for shard in getattr(record, split):
        value = np.load(shard.path, mmap_mode="r", allow_pickle=False)
        try:
            arrays.append(np.asarray(value, dtype=np.float32).copy())
        finally:
            mapping = getattr(value, "_mmap", None)
            if mapping is not None:
                mapping.close()
    return np.vstack(arrays)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, width: int) -> np.ndarray:
    flat = y_true.astype(np.int64) * width + y_pred.astype(np.int64)
    return np.bincount(flat, minlength=width * width).reshape(width, width)


def metrics_from_confusion(
    matrix: np.ndarray, class_ids: list[int], class_names: dict[int, str]
) -> dict:
    matrix = np.asarray(matrix, dtype=np.int64)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=support > 0,
    )
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=predicted > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(recall),
        where=(precision + recall) > 0,
    )
    total = int(matrix.sum())
    return {
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "macro_f1": float(f1.mean()) if len(f1) else 0.0,
        "balanced_accuracy": float(recall.mean()) if len(recall) else 0.0,
        "total_rows": total,
        "confusion_matrix": matrix.tolist(),
        "per_class": [
            {
                "class_id": int(class_id),
                "class_name": class_names[class_id],
                "support": int(support[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index, class_id in enumerate(class_ids)
        ],
    }


def task_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tasks: Iterable[Iterable[int]],
    checkpoint: int,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for task_index, task in enumerate(tasks):
        if task_index > checkpoint:
            break
        mask = np.isin(y_true, np.asarray(list(task), dtype=np.int64))
        if mask.any():
            result[str(task_index)] = float(np.mean(y_pred[mask] == y_true[mask]))
    return result


def summary_from_checkpoints(checkpoints: list[dict], view_name: str, task_count: int) -> dict:
    matrix: list[list[float | None]] = [
        [None for _ in range(task_count)] for _ in range(task_count)
    ]
    for checkpoint in checkpoints:
        row = int(checkpoint["checkpoint"])
        for task_id, value in checkpoint["views"][view_name]["task_accuracy"].items():
            matrix[row][int(task_id)] = float(value)
    forgetting: list[float] = []
    for task_id in range(task_count - 1):
        prior = [
            matrix[row][task_id]
            for row in range(task_count - 1)
            if matrix[row][task_id] is not None
        ]
        final = matrix[-1][task_id]
        if prior and final is not None:
            forgetting.append(float(max(prior) - final))
    final = checkpoints[-1]["views"][view_name]
    final_tasks = [value for value in matrix[-1] if value is not None]
    return {
        "task_accuracy_matrix": matrix,
        "average_task_accuracy": float(np.mean(final_tasks)),
        "average_forgetting": float(np.mean(forgetting)) if forgetting else 0.0,
        "final_overall_accuracy": final["accuracy"],
        "final_macro_f1": final["macro_f1"],
        "final_balanced_accuracy": final["balanced_accuracy"],
    }


def deterministic_result_view(value):
    if isinstance(value, dict):
        return {
            key: deterministic_result_view(item)
            for key, item in value.items()
            if key not in {"training_seconds"}
        }
    if isinstance(value, list):
        return [deterministic_result_view(item) for item in value]
    return value


def maybe_start_wandb(args: argparse.Namespace, config: dict):
    if not args.wandb_project:
        return None
    if not args.wandb_entity:
        raise ValueError("--wandb-entity is required with --wandb-project")
    import wandb

    return wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_name,
        job_type="classifier-diagnostic",
        config=config,
        tags=["catboost", "cumulative-multiclass", "pilot", "not-ofra"],
        settings=wandb.Settings(
            disable_code=True,
            disable_git=True,
            save_code=False,
            console="off",
            x_disable_stats=True,
            x_disable_machine_info=True,
        ),
    )


def main() -> int:
    args = parse_args()
    for name in ("iterations", "depth", "thread_count", "shap_sample_per_class"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    seed_everything(args.seed)
    manifest = load_manifest(args.manifest, verify_hashes=True)
    view = (
        load_evaluation_view(args.evaluation_view, manifest, verify_hashes=True)
        if args.evaluation_view
        else None
    )
    train_raw = {
        record.class_id: load_split(record, "train") for record in manifest.classes
    }
    test_raw = {
        record.class_id: load_split(record, "test") for record in manifest.classes
    }
    stats = FrozenTask0Stats(manifest.feature_dim)
    for class_id in manifest.tasks[0]:
        stats.update(train_raw[class_id])
    stats.freeze(manifest.tasks[0])
    train = {class_id: stats.transform(values) for class_id, values in train_raw.items()}
    test = {class_id: stats.transform(values) for class_id, values in test_raw.items()}

    duplicate_masks: dict[int, np.ndarray] = {}
    if view is not None:
        for class_id, shards in view.masks.items():
            duplicate_masks[class_id] = np.concatenate(
                [np.load(shard.path, allow_pickle=False) for shard in shards]
            ).astype(np.bool_, copy=False)
            if len(duplicate_masks[class_id]) != len(test[class_id]):
                raise RuntimeError("evaluation-view mask length mismatch")

    config = {
        "schema_version": SCHEMA_VERSION,
        "role": "diagnostic_cumulative_oracle_not_ofra",
        "dataset": manifest.dataset,
        "manifest_sha256": manifest.manifest_sha256,
        "evaluation_view_sha256": view.manifest_sha256 if view else None,
        "seed": args.seed,
        "model": {
            "package": "catboost",
            "version": catboost_version,
            "task_type": "CPU",
            "loss_function": "MultiClass",
            "iterations": args.iterations,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
            "l2_leaf_reg": args.l2_leaf_reg,
            "random_strength": args.random_strength,
            "auto_class_weights": "Balanced",
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 1.0,
            "thread_count": args.thread_count,
            "use_best_model": False,
        },
        "normalization": stats.record(),
        "validation_split": None,
        "test_based_early_stopping": False,
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    config["canonical_sha256"] = canonical_sha256(config)
    run = maybe_start_wandb(args, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )

    checkpoints: list[dict] = []
    seen: list[int] = []
    try:
        for checkpoint, task in enumerate(manifest.tasks):
            seen.extend(int(value) for value in task)
            class_to_column = {class_id: index for index, class_id in enumerate(seen)}
            x_train = np.vstack([train[class_id] for class_id in seen])
            y_train = np.concatenate(
                [
                    np.full(len(train[class_id]), class_to_column[class_id], dtype=np.int64)
                    for class_id in seen
                ]
            )
            checkpoint_seed = int(args.seed * 1000 + checkpoint)
            model = CatBoostClassifier(
                loss_function="MultiClass",
                iterations=args.iterations,
                depth=args.depth,
                learning_rate=args.learning_rate,
                l2_leaf_reg=args.l2_leaf_reg,
                random_strength=args.random_strength,
                auto_class_weights="Balanced",
                bootstrap_type="Bayesian",
                bagging_temperature=1.0,
                random_seed=checkpoint_seed,
                task_type="CPU",
                thread_count=args.thread_count,
                use_best_model=False,
                allow_writing_files=False,
                verbose=False,
            )
            started = time.perf_counter()
            model.fit(x_train, y_train)
            training_seconds = time.perf_counter() - started

            x_test = np.vstack([test[class_id] for class_id in seen])
            y_true_class = np.concatenate(
                [np.full(len(test[class_id]), class_id, dtype=np.int64) for class_id in seen]
            )
            probabilities = np.asarray(model.predict_proba(x_test), dtype=np.float64)
            predicted_columns = probabilities.argmax(axis=1).astype(np.int64)
            class_axis = np.asarray(seen, dtype=np.int64)
            y_pred_class = class_axis[predicted_columns]
            y_true_columns = np.asarray(
                [class_to_column[int(value)] for value in y_true_class], dtype=np.int64
            )
            official = metrics_from_confusion(
                confusion_matrix(y_true_columns, predicted_columns, len(seen)),
                seen,
                manifest.class_names,
            )
            official["task_accuracy"] = task_accuracy(
                y_true_class, y_pred_class, manifest.tasks, checkpoint
            )
            views = {"official": official}
            if view is not None:
                keep = np.concatenate([duplicate_masks[class_id] for class_id in seen])
                filtered = metrics_from_confusion(
                    confusion_matrix(
                        y_true_columns[keep], predicted_columns[keep], len(seen)
                    ),
                    seen,
                    manifest.class_names,
                )
                filtered["task_accuracy"] = task_accuracy(
                    y_true_class[keep], y_pred_class[keep], manifest.tasks, checkpoint
                )
                views[view.name] = filtered

            model_path = args.output_dir / f"checkpoint_{checkpoint:03d}.cbm"
            model.save_model(model_path)
            checkpoint_record = {
                "checkpoint": checkpoint,
                "seen_classes": list(seen),
                "train_rows": int(len(x_train)),
                "training_seconds": training_seconds,
                "tree_count": int(model.tree_count_),
                "model_file_bytes": int(model_path.stat().st_size),
                "model_file_sha256": sha256_file(model_path),
                "feature_importance_prediction_values_change": np.asarray(
                    model.get_feature_importance(type="PredictionValuesChange"),
                    dtype=np.float64,
                ).tolist(),
                "views": views,
            }
            if checkpoint == len(manifest.tasks) - 1:
                shap_parts = [
                    test[class_id][: args.shap_sample_per_class] for class_id in seen
                ]
                shap_x = np.vstack(shap_parts)
                shap_values = np.asarray(
                    model.get_feature_importance(Pool(shap_x), type="ShapValues"),
                    dtype=np.float64,
                )
                shap_path = args.output_dir / "final_native_shap_values.npy"
                np.save(shap_path, shap_values, allow_pickle=False)
                if shap_values.ndim == 3:
                    feature_values = shap_values[..., :-1]
                    mean_abs = np.mean(np.abs(feature_values), axis=(0, 1))
                elif shap_values.ndim == 2:
                    feature_values = shap_values[:, :-1]
                    mean_abs = np.mean(np.abs(feature_values), axis=0)
                else:
                    raise RuntimeError("unexpected CatBoost SHAP value shape")
                order = np.argsort(-mean_abs, kind="stable")
                checkpoint_record["native_shap"] = {
                    "scope": "final_checkpoint_deterministic_first_rows_per_class",
                    "sample_per_class": args.shap_sample_per_class,
                    "sample_rows": int(len(shap_x)),
                    "sample_sha256": array_sha256(shap_x),
                    "values_shape": list(shap_values.shape),
                    "values_file_sha256": sha256_file(shap_path),
                    "top_15": [
                        {
                            "feature_index": int(index),
                            "mean_absolute_shap": float(mean_abs[index]),
                        }
                        for index in order[:15]
                    ],
                }
            checkpoints.append(checkpoint_record)

            if run is not None:
                payload = {
                    "checkpoint/index": checkpoint,
                    "resources/training_seconds": training_seconds,
                    "resources/tree_count": int(model.tree_count_),
                    "resources/model_file_bytes": int(model_path.stat().st_size),
                }
                for view_name, metrics in views.items():
                    for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
                        payload[f"diagnostic/{view_name}/{metric}/{ARM}"] = metrics[metric]
                run.log(payload, step=checkpoint)
            print(
                json.dumps(
                    {
                        "event": "checkpoint_complete",
                        "seed": args.seed,
                        "checkpoint": checkpoint,
                        "training_seconds": training_seconds,
                        "official_accuracy": official["accuracy"],
                        "official_macro_f1": official["macro_f1"],
                        "official_balanced_accuracy": official["balanced_accuracy"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        summaries = {
            view_name: summary_from_checkpoints(
                checkpoints, view_name, len(manifest.tasks)
            )
            for view_name in checkpoints[-1]["views"]
        }
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "pilot_diagnostic_not_ofra",
            "dataset": manifest.dataset,
            "seed": args.seed,
            "config_sha256": config["canonical_sha256"],
            "checkpoints": checkpoints,
            "summary": {"views": summaries},
            "interpretation_guardrails": [
                "This is a cumulative multiclass CatBoost diagnostic, not OFRA.",
                "Each checkpoint model is retrained from scratch on all seen-class training rows.",
                "CPU training is used for reproducibility; no validation or test early stopping is used.",
                "Native CatBoost SHAP is not the OFRA routed-margin SHAP protocol.",
            ],
        }
        result["deterministic_result_sha256"] = canonical_sha256(
            deterministic_result_view(result)
        )
        result_path = args.output_dir / "result.json"
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        if run is not None:
            for view_name, summary in summaries.items():
                for metric, value in summary.items():
                    if isinstance(value, (int, float)):
                        run.summary[f"final/{view_name}/{metric}/{ARM}"] = value
            run.summary["reproducibility/deterministic_result_sha256"] = result[
                "deterministic_result_sha256"
            ]
            run.summary["reproducibility/result_file_sha256"] = sha256_file(result_path)
            run.finish(exit_code=0)
        print(
            json.dumps(
                {
                    "event": "complete",
                    "result": str(result_path),
                    **summaries["official"],
                },
                sort_keys=True,
            )
        )
    except Exception:
        if run is not None:
            run.finish(exit_code=1)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
