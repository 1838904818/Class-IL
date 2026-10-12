"""Run the final, preregistered Section 15.2 local hijack attempt."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from ncd.crr_benchmarks import (
    _logit_predictions,
    add_noise_markers,
    boundary_for,
    build_hijack,
    replay_buffer,
    run_method,
    train_base_checkpoint,
)
from ncd.data import PreparedData, prepare_dataset
from ncd.source_commit import source_commit
from ncd.train_cil import Tee


def _write_csv(path: Path, rows: list[dict]) -> None:
    scalar = [
        {
            key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
            for key, value in row.items()
        }
        for row in rows
    ]
    fields = list(dict.fromkeys(key for row in scalar for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scalar)


def validation_split(data: PreparedData, old_classes: tuple[int, ...], seed: int) -> tuple[PreparedData, np.ndarray, np.ndarray]:
    validation_indices: list[np.ndarray] = []
    fit_mask = np.ones(len(data.y_train), dtype=bool)
    for class_id in old_classes:
        indices = np.flatnonzero(data.y_train == class_id)
        rng = np.random.default_rng(seed * 1009 + class_id)
        count = max(1, int(round(0.2 * len(indices))))
        chosen = np.sort(rng.choice(indices, count, replace=False))
        validation_indices.append(chosen)
        fit_mask[chosen] = False
    rows = np.concatenate(validation_indices)
    fit = PreparedData(
        x_train=data.x_train[fit_mask],
        y_train=data.y_train[fit_mask],
        x_test=data.x_test,
        y_test=data.y_test,
        class_order=data.class_order,
        tasks=data.tasks,
        feature_names=data.feature_names,
        source_files=data.source_files,
    )
    return fit, data.x_train[rows], data.y_train[rows]


def choose_target(
    model,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    data: PreparedData,
    old_classes: tuple[int, ...],
    marker_columns: int,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> dict:
    eligible = [
        class_id
        for class_id in old_classes
        if int(np.sum(data.y_train == class_id)) >= 200
    ]
    if not eligible:
        raise ValueError("no old class has at least 200 training examples")
    augmented = add_noise_markers(validation_x, marker_columns, seed + 401)
    pred = _logit_predictions(model, augmented, old_classes, device, batch_size)
    candidates = []
    for class_id in eligible:
        mask = validation_y == class_id
        candidates.append(
            {
                "class_id": class_id,
                "class_name": data.class_order[class_id],
                "validation_recall": float(np.mean(pred[mask] == class_id)),
                "training_rows": int(np.sum(data.y_train == class_id)),
                "validation_rows": int(mask.sum()),
            }
        )
    selected = min(
        candidates,
        key=lambda row: (
            row["validation_recall"],
            -row["training_rows"],
            row["class_id"],
        ),
    )
    return {"selected": selected, "candidates": candidates}


def run(config: dict) -> dict:
    device = torch.device(str(config.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    rows: list[dict] = []
    selections: list[dict] = []
    data_root = (REPO_ROOT / config["data_root"]).resolve()
    marker_columns = int(config.get("marker_columns", 4))
    batch_size = int(config.get("batch_size", 256))
    started = time.perf_counter()
    for dataset_name in ("NSL-KDD", "UNSW-NB15"):
        data = prepare_dataset(dataset_name, data_root)
        original = boundary_for(data)
        for seed in (1, 2):
            fit, validation_x, validation_y = validation_split(data, original.old_classes, seed)
            base_model, base_x, base_y = train_base_checkpoint(
                fit,
                original,
                seed=seed,
                marker_columns=marker_columns,
                hidden=int(config.get("hidden", 128)),
                epochs=int(config.get("epochs_base", 8)),
                batch_size=batch_size,
                learning_rate=float(config.get("learning_rate", 1e-3)),
                device=device,
                rows_per_class=config.get("rows_per_class"),
            )
            choice = choose_target(
                base_model,
                validation_x,
                validation_y,
                data,
                original.old_classes,
                marker_columns,
                seed,
                device,
                batch_size,
            )
            target = int(choice["selected"]["class_id"])
            contrast = next(class_id for class_id in original.old_classes if class_id != target)
            boundary = replace(original, target_class=target, contrast_class=contrast)
            selections.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "old_classes": list(original.old_classes),
                    **choice,
                }
            )
            buffer_x, buffer_y = replay_buffer(
                base_model,
                base_x,
                base_y,
                original.old_classes,
                per_class=int(config.get("buffer_per_class", 200)),
                seed=seed,
                method="replay",
                device=device,
                batch_size=batch_size,
            )
            for rho_index, rho in enumerate((0.9, 1.0)):
                benchmark = build_hijack(
                    fit,
                    boundary,
                    buffer_x,
                    buffer_y,
                    rho=rho,
                    marker_columns=marker_columns,
                    seed=seed,
                    rows_per_class=config.get("rows_per_class"),
                    construction_step=3,
                )
                _, metrics = run_method(
                    base_model,
                    benchmark,
                    boundary,
                    method="replay",
                    regularization=0.0,
                    epochs=int(config.get("epochs_transition", 8)),
                    batch_size=batch_size,
                    learning_rate=float(config.get("learning_rate", 1e-3)),
                    temperature=float(config.get("temperature", 2.0)),
                    seed=seed * 1000000 + rho_index,
                    device=device,
                    rcl_features=int(config.get("rcl_features", 8)),
                    rcl_temperature=float(config.get("rcl_temperature", 0.5)),
                    rcl_beta=float(config.get("rcl_beta", 1.0)),
                    rcl_prediction_weight=float(config.get("rcl_prediction_weight", 0.5)),
                )
                row = {
                    "dataset": dataset_name,
                    "seed": seed,
                    "method": "replay",
                    "rho": rho,
                    "target_class": target,
                    "target_name": data.class_order[target],
                    "target_validation_recall": choice["selected"]["validation_recall"],
                    "target_training_rows": choice["selected"]["training_rows"],
                    **metrics,
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
    passed = all(
        float(row["marker_recall_drop"]) >= 0.15
        and float(row["marker_decorrelation_recall_drop"]) >= 0.15
        for row in rows
    )
    return {
        "schema_version": 1,
        "git_commit": source_commit(REPO_ROOT),
        "effective_config": config,
        "rows": rows,
        "target_selections": selections,
        "gate": {
            "passed": passed,
            "threshold": 0.15,
            "requires_all_dataset_seed_strength_cells": True,
            "fallback_section_15_3_active": not passed,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "scope": "Final preregistered Section 15.2 Replay hijack attempt only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log-path", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).resolve().read_text(encoding="utf-8"))
    if args.device:
        config["device"] = args.device
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        try:
            result = run(config)
            result_path = output / "hijack_result.json"
            result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
            _write_csv(output / "hijack_rows.csv", result["rows"])
            (output / "HIJACK_VERDICT.json").write_text(
                json.dumps(result["gate"], indent=2, sort_keys=True), encoding="utf-8"
            )
            (output / "COMPLETED.json").write_text(
                json.dumps(
                    {"git_commit": result["git_commit"], "row_count": len(result["rows"])},
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            print(json.dumps({"complete": str(result_path), **result["gate"]}))
        except Exception:
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
