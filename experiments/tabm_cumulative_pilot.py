"""Hash-bound cumulative TabM diagnostic on one streaming dataset.

This is deliberately not labelled as OFRA.  At each class-incremental
checkpoint it trains a fresh multiclass TabM on all training rows from the
classes seen so far.  The result estimates classifier headroom under the same
dataset cache, Task-0 normalization and test views as the OFRA pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import tabm

from streaming_full.data import (
    FrozenTask0Stats,
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--d-block", type=int, default=256)
    parser.add_argument("--n-blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0003)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-name")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def cuda_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError("CUDA index requested for a non-CUDA device")
    return device.index if device.index is not None else torch.cuda.current_device()


def load_split(record, split: str) -> np.ndarray:
    shards = getattr(record, split)
    arrays: list[np.ndarray] = []
    for shard in shards:
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


def metrics_from_confusion(matrix: np.ndarray, class_ids: list[int], class_names: dict[int, str]) -> dict:
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
    per_class = [
        {
            "class_id": int(class_id),
            "class_name": class_names[class_id],
            "support": int(support[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index, class_id in enumerate(class_ids)
    ]
    return {
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "macro_f1": float(f1.mean()) if len(f1) else 0.0,
        "balanced_accuracy": float(recall.mean()) if len(recall) else 0.0,
        "total_rows": total,
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
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


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def deterministic_result_view(value):
    """Remove runtime-only fields before hashing scientific results."""
    if isinstance(value, dict):
        omitted = {
            "seconds",
            "training_seconds",
            "cuda_device",
            "cuda_total_memory_bytes",
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
        }
        return {
            key: deterministic_result_view(item)
            for key, item in value.items()
            if key not in omitted
        }
    if isinstance(value, list):
        return [deterministic_result_view(item) for item in value]
    return value


@torch.inference_mode()
def predict_probabilities(
    model: torch.nn.Module,
    values: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        tensor = torch.from_numpy(
            np.ascontiguousarray(values[start : start + batch_size], dtype=np.float32)
        ).to(device)
        logits = model(tensor)
        if logits.ndim != 3:
            raise RuntimeError(f"TabM logits must have shape (B,K,C), got {logits.shape}")
        probability = torch.softmax(logits, dim=-1).mean(dim=1)
        output.append(probability.cpu().numpy().astype(np.float32))
    return np.vstack(output)


def train_checkpoint(
    *,
    feature_dim: int,
    class_count: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint: int,
) -> tuple[torch.nn.Module, list[dict], dict]:
    checkpoint_seed = int(args.seed * 1000 + checkpoint)
    seed_everything(checkpoint_seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(cuda_index(device))
    model = tabm.TabM.make(
        n_num_features=feature_dim,
        d_out=class_count,
        arch_type="tabm",
        k=args.k,
        d_block=args.d_block,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
    ).to(device)
    counts = np.bincount(train_y, minlength=class_count).astype(np.float64)
    weights = len(train_y) / (class_count * counts)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(train_x, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(train_y, dtype=np.int64)),
    )
    generator = torch.Generator().manual_seed(checkpoint_seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    history: list[dict] = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        correct = 0
        rows = 0
        epoch_started = time.perf_counter()
        for values, labels in loader:
            values = values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(values)
            if logits.ndim != 3 or logits.shape[1] != args.k:
                raise RuntimeError("unexpected TabM ensemble output")
            expanded = labels[:, None].expand(-1, logits.shape[1])
            loss = F.cross_entropy(
                logits.flatten(0, 1),
                expanded.flatten(),
                weight=class_weights,
            )
            loss.backward()
            optimizer.step()
            ensemble = torch.softmax(logits.detach(), dim=-1).mean(dim=1)
            correct += int((ensemble.argmax(dim=1) == labels).sum().item())
            loss_sum += float(loss.item()) * len(labels)
            rows += len(labels)
        history.append(
            {
                "epoch": epoch + 1,
                "rows": rows,
                "loss": loss_sum / rows,
                "accuracy": correct / rows,
                "seconds": time.perf_counter() - epoch_started,
            }
        )
        print(
            json.dumps(
                {
                    "event": "epoch_complete",
                    "checkpoint": checkpoint,
                    **history[-1],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(cuda_index(device))
    resource = {
        "training_seconds": time.perf_counter() - started,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "state_sha256": state_sha256(model),
    }
    if device.type == "cuda":
        device_index = cuda_index(device)
        properties = torch.cuda.get_device_properties(device_index)
        resource.update(
            {
                "cuda_device": torch.cuda.get_device_name(device_index),
                "cuda_total_memory_bytes": int(properties.total_memory),
                "cuda_peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device_index)
                ),
                "cuda_peak_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device_index)
                ),
            }
        )
    return model, history, resource


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
        tags=["tabm", "cumulative-multiclass", "pilot", "not-ofra"],
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
    if args.epochs <= 0 or args.batch_size <= 0 or args.k <= 0:
        raise ValueError("epochs, batch size and k must be positive")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(cuda_index(device))
        torch.cuda.reset_peak_memory_stats(cuda_index(device))

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
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "model": {
            "package": "tabm",
            "version": getattr(tabm, "__version__", "unknown"),
            "arch_type": "tabm",
            "k": args.k,
            "d_block": args.d_block,
            "n_blocks": args.n_blocks,
            "dropout": args.dropout,
            "num_embeddings": None,
            "prediction_aggregation": "mean_probability",
        },
        "optimization": {
            "optimizer": "AdamW",
            "parameter_groups": "all_trainable_parameters",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "loss": "member_wise_balanced_cross_entropy",
            "early_stopping": False,
            "validation_split": None,
        },
        "normalization": stats.record(),
        "device": str(device),
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
            model, history, resources = train_checkpoint(
                feature_dim=manifest.feature_dim,
                class_count=len(seen),
                train_x=x_train,
                train_y=y_train,
                args=args,
                device=device,
                checkpoint=checkpoint,
            )

            x_test = np.vstack([test[class_id] for class_id in seen])
            y_true_class = np.concatenate(
                [np.full(len(test[class_id]), class_id, dtype=np.int64) for class_id in seen]
            )
            probabilities = predict_probabilities(
                model,
                x_test,
                batch_size=args.eval_batch_size,
                device=device,
            )
            predicted_columns = probabilities.argmax(axis=1)
            class_axis = np.asarray(seen, dtype=np.int64)
            y_pred_class = class_axis[predicted_columns]
            y_true_columns = np.asarray(
                [class_to_column[int(value)] for value in y_true_class], dtype=np.int64
            )
            official_matrix = confusion_matrix(y_true_columns, predicted_columns, len(seen))
            official = metrics_from_confusion(
                official_matrix, seen, manifest.class_names
            )
            official["task_accuracy"] = task_accuracy(
                y_true_class, y_pred_class, manifest.tasks, checkpoint
            )

            views = {"official": official}
            if view is not None:
                keep = np.concatenate([duplicate_masks[class_id] for class_id in seen])
                filtered_matrix = confusion_matrix(
                    y_true_columns[keep], predicted_columns[keep], len(seen)
                )
                filtered = metrics_from_confusion(
                    filtered_matrix, seen, manifest.class_names
                )
                filtered["task_accuracy"] = task_accuracy(
                    y_true_class[keep], y_pred_class[keep], manifest.tasks, checkpoint
                )
                views[view.name] = filtered

            checkpoint_record = {
                "checkpoint": checkpoint,
                "seen_classes": list(seen),
                "train_rows": len(x_train),
                "training_history": history,
                "resources": resources,
                "views": views,
            }
            checkpoint_path = args.output_dir / f"checkpoint_{checkpoint:03d}.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "seen_classes": list(seen),
                    "config_sha256": config["canonical_sha256"],
                    "state_sha256": resources["state_sha256"],
                },
                checkpoint_path,
            )
            checkpoint_record["checkpoint_file_sha256"] = sha256_file(checkpoint_path)
            checkpoints.append(checkpoint_record)
            if run is not None:
                payload = {"checkpoint/index": checkpoint}
                for view_name, metrics in views.items():
                    for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
                        payload[f"diagnostic/{view_name}/{metric}/{ARM}"] = metrics[metric]
                payload["resources/training_seconds"] = resources["training_seconds"]
                payload["resources/parameters"] = resources["parameters"]
                if "cuda_peak_allocated_bytes" in resources:
                    payload["resources/cuda_peak_allocated_bytes"] = resources[
                        "cuda_peak_allocated_bytes"
                    ]
                run.log(payload, step=checkpoint)
            print(
                json.dumps(
                    {
                        "event": "checkpoint_complete",
                        "checkpoint": checkpoint,
                        "seen_classes": seen,
                        "official_accuracy": official["accuracy"],
                        "official_macro_f1": official["macro_f1"],
                        "official_balanced_accuracy": official["balanced_accuracy"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

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
                "This is a cumulative multiclass TabM diagnostic, not OFRA.",
                "Each checkpoint model is retrained from scratch on all seen-class training rows.",
                "No validation split or test-based early stopping is used.",
                "A seed-1 pilot is not a five-seed result.",
            ],
        }
        result["deterministic_result_sha256"] = canonical_sha256(
            deterministic_result_view(result)
        )
        result_path = args.output_dir / "result.json"
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        result["result_file_sha256"] = sha256_file(result_path)
        if run is not None:
            for view_name, summary in summaries.items():
                for metric, value in summary.items():
                    if isinstance(value, (int, float)):
                        run.summary[f"final/{view_name}/{metric}/{ARM}"] = value
            run.summary["reproducibility/deterministic_result_sha256"] = result[
                "deterministic_result_sha256"
            ]
            run.summary["reproducibility/result_file_sha256"] = result[
                "result_file_sha256"
            ]
            rows = []
            for item in checkpoints:
                for view_name, metrics in item["views"].items():
                    rows.append(
                        [
                            item["checkpoint"],
                            view_name,
                            metrics["accuracy"],
                            metrics["macro_f1"],
                            metrics["balanced_accuracy"],
                            item["train_rows"],
                        ]
                    )
            import wandb

            run.log(
                {
                    "results/checkpoint_metrics": wandb.Table(
                        columns=[
                            "checkpoint",
                            "view",
                            "accuracy",
                            "macro_f1",
                            "balanced_accuracy",
                            "train_rows",
                        ],
                        data=rows,
                    )
                }
            )
            run.finish(exit_code=0)
        print(json.dumps({"event": "complete", "result": str(result_path), **summaries["official"]}, sort_keys=True))
    except Exception:
        if run is not None:
            run.finish(exit_code=1)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
