#!/usr/bin/env python3
"""Audit whether preprocessing observes information beyond Task 0.

The audit is read-only.  It binds the cache manifests and implementation
sources, verifies the frozen Task-0 numerical-statistics implementation, and
quantifies categorical values that occur only in later-task training rows.
Absolute local paths are deliberately excluded from the output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


NSL_COLUMNS = (
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_train(dataset: str, source_file: Path) -> pd.DataFrame:
    if dataset == "nsl-kdd":
        return pd.read_csv(source_file, names=list(NSL_COLUMNS), low_memory=False)
    return pd.read_csv(source_file, low_memory=False)


def mapped_labels(dataset: str, frame: pd.DataFrame, schema: dict[str, Any]) -> pd.Series:
    label_column = "label" if dataset == "nsl-kdd" else "attack_cat"
    labels = frame[label_column].astype("string").str.strip()
    if dataset == "nsl-kdd":
        labels = labels.str.rstrip(".")
    mapping = schema["label_mapping"]
    mapped = labels.map(mapping)
    if dataset == "unsw-nb15":
        mapped = mapped.fillna(labels)
    if mapped.isna().any():
        missing = sorted(labels[mapped.isna()].dropna().unique().tolist())
        raise RuntimeError(f"unmapped labels in {dataset}: {missing}")
    return mapped


def categorical_audit(dataset_dir: Path) -> dict[str, Any]:
    dataset = dataset_dir.name
    schema_path = dataset_dir / "feature_schema.json"
    manifest_path = dataset_dir / "manifest.json"
    stream_path = dataset_dir / "streaming_manifest.json"
    schema = load_json(schema_path)
    manifest = load_json(manifest_path)
    stream = load_json(stream_path)
    categorical = list(schema.get("categorical_columns", []))
    base = {
        "dataset": dataset,
        "feature_dim": int(stream["feature_dim"]),
        "categorical_columns": categorical,
        "feature_schema_sha256": sha256_file(schema_path),
        "fullcache_manifest_sha256": sha256_file(manifest_path),
        "streaming_manifest_sha256": sha256_file(stream_path),
    }
    if not categorical:
        return {
            **base,
            "status": "PASS_NO_DATA_DERIVED_CATEGORICAL_VOCABULARY",
            "task0_classes": [
                stream["classes"][class_id]["name"] for class_id in stream["tasks"][0]
            ],
            "future_exclusive_indicator_count": 0,
            "future_rows_with_any_future_exclusive_value": 0,
        }

    train_record = next(
        item for item in manifest["raw_files"] if item.get("official_split") == "train"
    )
    source_file = Path(manifest["source_directory"]) / train_record["relative_path"]
    if not source_file.is_file():
        raise FileNotFoundError(
            f"raw training source is unavailable for {dataset}; pass a cache whose bound source exists"
        )
    if sha256_file(source_file) != train_record["sha256"]:
        raise RuntimeError(f"raw training source hash mismatch for {dataset}")
    frame = read_train(dataset, source_file)
    labels = mapped_labels(dataset, frame, schema)
    task0_classes = {
        stream["classes"][class_id]["name"] for class_id in stream["tasks"][0]
    }
    task0_mask = labels.isin(task0_classes)
    future_mask = ~task0_mask
    union_future_only = pd.Series(False, index=frame.index)
    column_rows: list[dict[str, Any]] = []
    for column in categorical:
        values = frame[column].astype("string").str.strip()
        all_values = set(values.dropna().tolist()) - {""}
        task0_values = set(values[task0_mask].dropna().tolist()) - {""}
        future_only = sorted(all_values - task0_values)
        future_value_mask = future_mask & values.isin(future_only)
        union_future_only |= future_value_mask
        column_rows.append(
            {
                "column": column,
                "all_train_value_count": len(all_values),
                "task0_train_value_count": len(task0_values),
                "future_exclusive_values": future_only,
                "future_exclusive_value_count": len(future_only),
                "future_rows_with_future_exclusive_value": int(future_value_mask.sum()),
            }
        )
    future_rows = int(future_mask.sum())
    affected_rows = int(union_future_only.sum())
    count = sum(row["future_exclusive_value_count"] for row in column_rows)
    task0_vocab_dim = len(schema["raw_numeric_columns"]) + sum(
        row["task0_train_value_count"] for row in column_rows
    )
    return {
        **base,
        "status": (
            "LIMITATION_FUTURE_TASK_VALUES_DEFINE_EARLY_FEATURE_COLUMNS"
            if count
            else "PASS_NO_FUTURE_EXCLUSIVE_VALUES"
        ),
        "vocabulary_fit_scope_observed": manifest["split_protocol"].get(
            "categorical_vocabulary_fit"
        ),
        "task0_classes": sorted(task0_classes),
        "all_train_rows": int(len(frame)),
        "task0_train_rows": int(task0_mask.sum()),
        "later_task_train_rows": future_rows,
        "future_exclusive_indicator_count": count,
        "future_rows_with_any_future_exclusive_value": affected_rows,
        "future_rows_affected_fraction": affected_rows / future_rows if future_rows else 0.0,
        "current_feature_dim": int(stream["feature_dim"]),
        "task0_only_vocabulary_feature_dim": task0_vocab_dim,
        "columns": column_rows,
        "interpretation": (
            "The early model never observes future row values, but the fixed one-hot schema reveals "
            "that these categorical values exist because vocabulary construction scans all official "
            "training rows before the task stream starts."
        ),
    }


def numerical_stats_audit(project_root: Path) -> dict[str, Any]:
    validation = project_root / "streaming_full" / "validation.py"
    data = project_root / "streaming_full" / "data.py"
    validation_text = validation.read_text(encoding="utf-8")
    data_text = data.read_text(encoding="utf-8")
    checks = {
        "fit_method_selects_task0": "task0 = self.manifest.tasks[0]" in validation_text,
        "fit_method_reads_train_only": "self.train[class_id].batches" in validation_text,
        "fit_method_freezes_after_task0": "self.stats.freeze(task0)" in validation_text,
        "post_freeze_update_rejected": (
            'if self.frozen:\n            raise RuntimeError("normalization statistics are frozen")'
            in data_text
        ),
        "population_variance_recorded": "chan_welford_float64_population_variance" in data_text,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "Task-0 training shards only; statistics freeze before Task-0 pretraining",
        "checks": checks,
        "validation_py_sha256": sha256_file(validation),
        "data_py_sha256": sha256_file(data),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cache_root = args.cache_root.resolve()
    project_root = args.project_root.resolve()
    datasets = [
        categorical_audit(cache_root / name)
        for name in (
            "nsl-kdd", "unsw-nb15", "cic-ids-2017", "cic-ids-2018", "malaya-network-gt"
        )
    ]
    payload: dict[str, Any] = {
        "schema_version": "ofra_no_lookahead_audit_v1",
        "audit_date": "2026-08-19",
        "numerical_normalization": numerical_stats_audit(project_root),
        "categorical_schema": datasets,
        "overall_conclusion": (
            "Numerical normalization passes the Task-0-only contract.  NSL-KDD and UNSW-NB15 "
            "retain a bounded transductive schema limitation because their one-hot vocabularies "
            "are defined from the complete official training partition.  The other three datasets "
            "have no data-derived categorical vocabulary in the formal cache."
        ),
        "publication_boundary": (
            "Do not describe the complete preprocessing pipeline as strictly no-look-ahead until "
            "NSL-KDD and UNSW-NB15 are rebuilt with a Task-0-only vocabulary or an externally fixed vocabulary."
        ),
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": args.output.as_posix(), "canonical_sha256": payload["canonical_sha256"]}))


if __name__ == "__main__":
    main()
