from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .data import (
    DatasetManifest,
    canonical_sha256,
    resolve_manifest_semantics,
    sha256_bytes,
)


EXPOSURE_PREFLIGHT_SCHEMA_VERSION = 1
EXPOSURE_PREFLIGHT_ALGORITHM = "ofra_binary_family_exposure_preflight_v1"


def exposure_counter_dtype(epochs: int) -> type[np.unsignedinteger]:
    if not isinstance(epochs, int) or epochs < 0:
        raise ValueError("epochs must be a non-negative integer")
    for dtype in (np.uint8, np.uint16, np.uint32):
        if epochs <= np.iinfo(dtype).max:
            return dtype
    raise ValueError("epochs_per_task exceeds exact exposure-counter capacity")


def _optimizer_schedule(
    positive_rows: int, negative_rows: int, positive_batch_rows: int
) -> dict[str, int | float]:
    previous_negative = 0
    zero_negative_steps = 0
    steps = 0
    minimum_negative: int | None = None
    maximum_negative: int | None = None
    positive_boundaries = range(
        positive_batch_rows, positive_rows, positive_batch_rows
    )
    for positive_after in itertools.chain(positive_boundaries, (positive_rows,)):
        negative_after = (negative_rows * positive_after) // positive_rows
        batch_negative = negative_after - previous_negative
        previous_negative = negative_after
        steps += 1
        zero_negative_steps += int(batch_negative == 0)
        minimum_negative = (
            batch_negative
            if minimum_negative is None
            else min(minimum_negative, batch_negative)
        )
        maximum_negative = (
            batch_negative
            if maximum_negative is None
            else max(maximum_negative, batch_negative)
        )
    return {
        "optimizer_steps_per_epoch": steps,
        "zero_negative_steps_per_epoch": zero_negative_steps,
        "zero_negative_step_fraction": zero_negative_steps / steps,
        "negative_batch_rows_min": int(minimum_negative),
        "negative_batch_rows_max": int(maximum_negative),
    }


def _warning_records(
    *,
    candidate_limited: bool,
    shortfall: int,
    prior: float,
    zero_fraction: float,
) -> list[dict[str, object]]:
    warnings: list[dict[str, object]] = []
    if candidate_limited:
        warnings.append(
            {
                "code": "negative_candidate_shortfall",
                "severity": "warning",
                "shortfall_rows_per_epoch": shortfall,
            }
        )
    if prior >= 0.95 or prior <= 0.05:
        warnings.append(
            {
                "code": "extreme_count_exposure_prior",
                "severity": "high",
                "positive_exposure_prior": prior,
            }
        )
    if zero_fraction > 0.0:
        warnings.append(
            {
                "code": "optimizer_steps_without_negative_examples",
                "severity": "high" if zero_fraction >= 0.5 else "warning",
                "zero_negative_step_fraction": zero_fraction,
            }
        )
    return warnings


def _config_record(config: object) -> dict[str, object]:
    if not is_dataclass(config):
        raise TypeError("config must be a RunConfig dataclass instance")
    value = asdict(config)
    required = (
        "epochs_per_task",
        "batch_size",
        "negative_ratio",
        "exemplar_capacity",
        "exemplar_candidate_capacity",
    )
    if any(key not in value for key in required):
        raise ValueError("config lacks exposure-relevant RunConfig fields")
    return value


def _build_report(
    *,
    dataset: str,
    manifest_sha256: str,
    problem_type: str,
    metric_profile: str,
    task_semantics: str,
    normal_class_id: int | None,
    tasks: Sequence[Sequence[int]],
    class_names: Mapping[int, str],
    train_rows: Mapping[int, int],
    config: object,
) -> dict[str, object]:
    config_record = _config_record(config)
    epochs = int(config_record["epochs_per_task"])
    batch_size = int(config_record["batch_size"])
    negative_ratio = int(config_record["negative_ratio"])
    exemplar_capacity = int(config_record["exemplar_capacity"])
    candidate_capacity = int(config_record["exemplar_candidate_capacity"])
    if (
        epochs < 0
        or batch_size <= 0
        or negative_ratio <= 0
        or exemplar_capacity <= 0
        or candidate_capacity <= 0
    ):
        raise ValueError("invalid exposure-relevant RunConfig values")
    counter_dtype = exposure_counter_dtype(epochs)
    positive_batch_rows = max(1, batch_size // (negative_ratio + 1))
    exemplar_rows = {
        class_id: min(
            int(train_rows[class_id]), exemplar_capacity, candidate_capacity
        )
        for class_id in train_rows
    }
    task_records: list[dict[str, object]] = []
    seen: list[int] = []
    warning_counts = {"high": 0, "warning": 0}
    for task_index, task_values in enumerate(tasks):
        task = [int(value) for value in task_values]
        families: list[dict[str, object]] = []
        for class_id in task:
            sources = [
                {
                    "source_kind": "current_task_train",
                    "class_id": other,
                    "class_name": class_names[other],
                    "rows": int(train_rows[other]),
                }
                for other in task
                if other != class_id
            ] + [
                {
                    "source_kind": "prior_exemplar",
                    "class_id": prior_class,
                    "class_name": class_names[prior_class],
                    "rows": int(exemplar_rows[prior_class]),
                }
                for prior_class in sorted(seen)
            ]
            positive = int(train_rows[class_id])
            candidates = sum(int(source["rows"]) for source in sources)
            desired = positive * negative_ratio
            selected = min(candidates, desired)
            if positive <= 0 or candidates <= 0 or selected <= 0:
                raise ValueError(
                    f"class {class_id} has no positive or negative training population"
                )
            prior = positive / (positive + selected)
            schedule = _optimizer_schedule(
                positive, selected, positive_batch_rows
            )
            warnings = _warning_records(
                candidate_limited=selected < desired,
                shortfall=desired - selected,
                prior=prior,
                zero_fraction=float(schedule["zero_negative_step_fraction"]),
            )
            for warning in warnings:
                warning_counts[str(warning["severity"])] += 1
            families.append(
                {
                    "task_index": task_index,
                    "class_id": class_id,
                    "class_name": class_names[class_id],
                    "positive_rows_per_epoch": positive,
                    "positive_exposures_total": positive * epochs,
                    "negative_candidate_pool": {
                        "sources": sources,
                        "rows": candidates,
                    },
                    "desired_negative_rows_per_epoch": desired,
                    "selected_negative_rows_per_epoch": selected,
                    "negative_exposures_total": selected * epochs,
                    "candidate_limited": selected < desired,
                    "negative_shortfall_rows_per_epoch": desired - selected,
                    "negative_ratio_realized": selected / positive,
                    "positive_exposure_prior": prior,
                    "log_positive_to_negative_exposure_ratio": math.log(
                        positive / selected
                    ),
                    "positive_rows_per_optimizer_step": positive_batch_rows,
                    **schedule,
                    "optimizer_steps_total": int(
                        schedule["optimizer_steps_per_epoch"]
                    )
                    * epochs,
                    "zero_negative_steps_total": int(
                        schedule["zero_negative_steps_per_epoch"]
                    )
                    * epochs,
                    "exposure_counter_dtype": np.dtype(counter_dtype).name,
                    "exposure_counter_capacity": int(np.iinfo(counter_dtype).max),
                    "warnings": warnings,
                }
            )
        task_records.append(
            {
                "task_index": task_index,
                "class_ids": task,
                "families": families,
            }
        )
        seen.extend(task)
    semantic: dict[str, object] = {
        "schema_version": EXPOSURE_PREFLIGHT_SCHEMA_VERSION,
        "report": EXPOSURE_PREFLIGHT_ALGORITHM,
        "dataset": dataset,
        "streaming_manifest_sha256": manifest_sha256,
        "problem_type": problem_type,
        "metric_profile": metric_profile,
        "task_semantics": task_semantics,
        "normal_class_id": (
            int(normal_class_id) if normal_class_id is not None else None
        ),
        "tasks": [list(map(int, task)) for task in tasks],
        "run_config": config_record,
        "run_config_sha256": canonical_sha256(config_record),
        "derivation": {
            "positive_population": "all class train-shard rows once per epoch",
            "negative_candidate_pool": (
                "other current-task train rows plus frozen exemplars from prior tasks"
            ),
            "selected_negative_formula": "N=min(candidate_pool_rows, negative_ratio*P)",
            "positive_batch_rows_formula": (
                "max(1, batch_size//(negative_ratio+1))"
            ),
            "zero_negative_steps": (
                "exact cumulative-floor schedule used by train_family"
            ),
            "test_data_read": False,
            "training_executed": False,
        },
        "warning_policy": {
            "extreme_prior": "positive_exposure_prior >= 0.95 or <= 0.05",
            "high_zero_negative_fraction": "fraction >= 0.5",
            "warnings_are_diagnostic_not_automatic_training_failures": True,
        },
        "tasks_detail": task_records,
        "warning_counts": warning_counts,
    }
    semantic["deterministic_result_sha256"] = canonical_sha256(semantic)
    return semantic


def exposure_preflight_for_manifest(
    manifest: DatasetManifest, config: object
) -> dict[str, object]:
    return _build_report(
        dataset=manifest.dataset,
        manifest_sha256=manifest.manifest_sha256,
        problem_type=manifest.problem_type,
        metric_profile=manifest.metric_profile,
        task_semantics=manifest.task_semantics,
        normal_class_id=manifest.normal_class_id,
        tasks=manifest.tasks,
        class_names=manifest.class_names,
        train_rows={
            record.class_id: sum(shard.rows for shard in record.train)
            for record in manifest.classes
        },
        config=config,
    )


def exposure_preflight_from_path(
    manifest_path: str | Path, config: object
) -> dict[str, object]:
    """Derive exposure counts from manifest JSON metadata without opening shards."""

    path = Path(manifest_path).resolve()
    raw_bytes = path.read_bytes()
    raw = json.loads(raw_bytes)
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("streaming manifest schema_version must be 1")
    dataset = raw.get("dataset")
    problem_type, metric_profile, task_semantics, normal_class_id = (
        resolve_manifest_semantics(raw, context="streaming manifest")
    )
    classes_raw = raw.get("classes")
    tasks_raw = raw.get("tasks")
    if (
        not isinstance(dataset, str)
        or not dataset
        or not isinstance(classes_raw, list)
        or not classes_raw
        or not isinstance(tasks_raw, list)
        or not tasks_raw
    ):
        raise ValueError("streaming manifest exposure metadata is invalid")
    class_names: dict[int, str] = {}
    train_rows: dict[int, int] = {}
    for item in classes_raw:
        if not isinstance(item, dict):
            raise ValueError("streaming manifest class record is invalid")
        class_id = item.get("id")
        name = item.get("name")
        shards = item.get("train")
        if (
            not isinstance(class_id, int)
            or class_id < 0
            or class_id in class_names
            or not isinstance(name, str)
            or not name
            or not isinstance(shards, list)
            or not shards
        ):
            raise ValueError("streaming manifest class/train metadata is invalid")
        rows = 0
        for shard in shards:
            if not isinstance(shard, dict) or not isinstance(shard.get("rows"), int):
                raise ValueError("streaming manifest train-shard row metadata is invalid")
            if shard["rows"] <= 0:
                raise ValueError("streaming manifest train-shard rows must be positive")
            rows += shard["rows"]
        class_names[class_id] = name
        train_rows[class_id] = rows
    if sorted(class_names) != list(range(len(class_names))):
        raise ValueError("streaming manifest class identifiers must be contiguous")
    flattened: list[int] = []
    tasks: list[list[int]] = []
    for task in tasks_raw:
        if (
            not isinstance(task, list)
            or not task
            or not all(isinstance(value, int) for value in task)
            or len(set(task)) != len(task)
        ):
            raise ValueError("streaming manifest task metadata is invalid")
        tasks.append(list(task))
        flattened.extend(task)
    if sorted(flattened) != sorted(class_names) or len(flattened) != len(class_names):
        raise ValueError("streaming manifest tasks must cover each class exactly once")
    return _build_report(
        dataset=dataset,
        manifest_sha256=sha256_bytes(raw_bytes),
        problem_type=problem_type,
        metric_profile=metric_profile,
        task_semantics=task_semantics,
        normal_class_id=normal_class_id,
        tasks=tasks,
        class_names=class_names,
        train_rows=train_rows,
        config=config,
    )


def validate_exposure_preflight(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("exposure preflight must be a JSON object")
    resolve_manifest_semantics(value, context="exposure preflight")
    stored = value.get("deterministic_result_sha256")
    basis = dict(value)
    basis.pop("deterministic_result_sha256", None)
    if stored != canonical_sha256(basis):
        raise ValueError("exposure preflight deterministic self-hash mismatch")
    if (
        value.get("schema_version") != EXPOSURE_PREFLIGHT_SCHEMA_VERSION
        or value.get("report") != EXPOSURE_PREFLIGHT_ALGORITHM
    ):
        raise ValueError("unsupported exposure preflight schema/report")
    return value


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Derive and hash OFRA family exposure counts from streaming-manifest "
            "metadata without reading test data or training."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Read and compare the existing output; never write it.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from .validation import RunConfig

    raw_config = json.loads(args.config_json.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise ValueError("configuration JSON must contain one object")
    allowed = {field.name for field in fields(RunConfig)}
    unknown = sorted(set(raw_config) - allowed)
    if unknown:
        raise ValueError(f"unknown RunConfig keys: {', '.join(unknown)}")
    config = RunConfig(**raw_config)
    config.validate()
    expected = exposure_preflight_from_path(args.manifest, config)
    output = args.output.resolve()
    if args.verify_only:
        actual = validate_exposure_preflight(
            json.loads(output.read_text(encoding="utf-8"))
        )
        if actual != expected:
            raise RuntimeError("existing exposure preflight differs from manifest/config")
        action = "verified"
    elif output.exists():
        actual = validate_exposure_preflight(
            json.loads(output.read_text(encoding="utf-8"))
        )
        if actual != expected:
            raise RuntimeError("refusing to overwrite a different exposure preflight")
        action = "already_identical"
    else:
        _atomic_json(output, expected)
        action = "written"
    print(
        json.dumps(
            {
                "action": action,
                "output": str(output),
                "dataset": expected["dataset"],
                "deterministic_result_sha256": expected[
                    "deterministic_result_sha256"
                ],
                "warning_counts": expected["warning_counts"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
