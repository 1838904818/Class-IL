from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from streaming_full.data import (
    DatasetManifest,
    EVALUATION_VIEW_ALGORITHM,
    EVALUATION_VIEW_EQUALITY_KEY,
    EVALUATION_VIEW_NAME,
    Shard,
    canonical_sha256,
    dataset_logical_fingerprints,
    load_evaluation_view,
    load_manifest,
    sha256_file,
)


ALGORITHM = EVALUATION_VIEW_ALGORITHM
EQUALITY_KEY = EVALUATION_VIEW_EQUALITY_KEY


def _atomic_json(path: Path, value: object) -> None:
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


def _source_manifest() -> dict[str, object]:
    root = Path(__file__).parent
    files = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(root.glob("*.py"))
    ]
    return {"files": files, "canonical_source_sha256": canonical_sha256(files)}


def _relative(manifest: DatasetManifest, shard: Shard) -> str:
    try:
        return shard.path.relative_to(manifest.path.parent).as_posix()
    except ValueError:
        return shard.path.name


def _row_void(values: np.ndarray, feature_dim: int) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype="<f4")
    if array.ndim != 2 or array.shape[1] != feature_dim:
        raise ValueError(f"invalid feature batch shape: {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("non-finite feature found while building overlap view")
    return array.view(np.dtype((np.void, feature_dim * 4))).reshape(-1)


def _query_masks(
    connection: sqlite3.Connection, keys: list[bytes], chunk: int = 400
) -> dict[bytes, int]:
    result: dict[bytes, int] = {}
    for start in range(0, len(keys), chunk):
        block = keys[start : start + chunk]
        placeholders = ",".join("?" for _ in block)
        rows = connection.execute(
            f"SELECT feature, label_mask FROM train_features "
            f"WHERE feature IN ({placeholders})",
            tuple(sqlite3.Binary(value) for value in block),
        ).fetchall()
        result.update((bytes(feature), int(mask)) for feature, mask in rows)
    return result


def _index_train(
    connection: sqlite3.Connection,
    manifest: DatasetManifest,
    batch_rows: int,
) -> None:
    upsert = """
        INSERT INTO train_features(feature, label_mask) VALUES (?, ?)
        ON CONFLICT(feature) DO UPDATE SET
            label_mask = train_features.label_mask | excluded.label_mask
    """
    for record in manifest.classes:
        bit = 1 << record.class_id
        for shard in record.train:
            array = np.load(shard.path, mmap_mode="r", allow_pickle=False)
            try:
                for start in range(0, shard.rows, batch_rows):
                    rows = _row_void(
                        np.asarray(array[start : start + batch_rows]),
                        manifest.feature_dim,
                    )
                    unique = np.unique(rows)
                    connection.executemany(
                        upsert,
                        (
                            (sqlite3.Binary(bytes(value)), bit)
                            for value in unique.tolist()
                        ),
                    )
            finally:
                mapping = getattr(array, "_mmap", None)
                if mapping is not None:
                    mapping.close()
        connection.commit()


def _mask_one_shard(
    connection: sqlite3.Connection,
    manifest: DatasetManifest,
    class_id: int,
    shard: Shard,
    output: Path,
    batch_rows: int,
) -> tuple[dict[str, object], dict[str, object]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    mask_array = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.bool_, shape=(shard.rows,)
    )
    test_bit = 1 << class_id
    same_only = different_only = mixed_same = excluded = 0
    label_presence = [0 for _ in manifest.classes]
    array = np.load(shard.path, mmap_mode="r", allow_pickle=False)
    try:
        for start in range(0, shard.rows, batch_rows):
            rows = _row_void(
                np.asarray(array[start : start + batch_rows]), manifest.feature_dim
            )
            unique, inverse, counts = np.unique(
                rows, return_inverse=True, return_counts=True
            )
            keys = [bytes(value) for value in unique.tolist()]
            train_masks = _query_masks(connection, keys)
            unique_overlap = np.asarray(
                [key in train_masks for key in keys], dtype=np.bool_
            )
            keep = ~unique_overlap[inverse]
            mask_array[start : start + len(rows)] = keep
            overlap_indices = np.flatnonzero(unique_overlap)
            overlap_keys = [keys[int(index)] for index in overlap_indices]
            connection.executemany(
                "INSERT OR IGNORE INTO global_overlap_features(feature) VALUES (?)",
                ((sqlite3.Binary(key),) for key in overlap_keys),
            )
            connection.executemany(
                "INSERT OR IGNORE INTO class_overlap_features(test_class, feature) "
                "VALUES (?, ?)",
                (
                    (class_id, sqlite3.Binary(key))
                    for key in overlap_keys
                ),
            )
            for index in overlap_indices:
                key = keys[int(index)]
                count = int(counts[int(index)])
                train_mask = train_masks[key]
                excluded += count
                if train_mask == test_bit:
                    same_only += count
                elif not train_mask & test_bit:
                    different_only += count
                else:
                    mixed_same += count
                for train_class in range(len(manifest.classes)):
                    if train_mask & (1 << train_class):
                        label_presence[train_class] += count
        mask_array.flush()
        connection.commit()
    finally:
        del mask_array
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            mapping.close()
    retained = shard.rows - excluded
    mask_hash = sha256_file(output)
    mask_record = {
        "path": output.as_posix(),
        "dtype": "bool",
        "shape": [shard.rows],
        "rows": shard.rows,
        "bytes": output.stat().st_size,
        "sha256": mask_hash,
        "true_count": retained,
        "false_count": excluded,
    }
    audit_record = {
        "official_rows": shard.rows,
        "retained_rows": retained,
        "excluded_rows": excluded,
        "same_label_only_rows": same_only,
        "different_label_only_rows": different_only,
        "mixed_including_same_label_rows": mixed_same,
        "label_presence_rows": label_presence,
        "mask_sha256": mask_hash,
    }
    return mask_record, audit_record


def _primary_record(manifest: DatasetManifest) -> dict[str, object]:
    fingerprints = dataset_logical_fingerprints(manifest)
    fullcache = manifest.source_provenance.get("fullcache_manifest")
    record: dict[str, object] = {
        "streaming_manifest_sha256": manifest.manifest_sha256,
        **fingerprints,
    }
    if isinstance(fullcache, dict):
        record["fullcache_manifest_sha256"] = fullcache.get("sha256")
        record["fullcache_manifest_canonical_sha256"] = fullcache.get(
            "canonical_sha256"
        )
    split_audit = manifest.source_provenance.get("split_overlap_audit")
    if isinstance(split_audit, dict):
        record["split_overlap_audit_sha256"] = split_audit.get("sha256")
        record["split_overlap_audit_canonical_sha256"] = split_audit.get(
            "canonical_sha256"
        )
    return record


def _reconcile_frozen_split_overlap(
    manifest: DatasetManifest,
    totals: dict[str, int],
    overlap_unique_feature_rows: int,
) -> dict[str, object]:
    provenance = manifest.source_provenance.get("split_overlap_audit")
    if not isinstance(provenance, dict):
        raise ValueError("primary manifest lacks frozen split-overlap provenance")
    path = Path(str(provenance.get("path"))).resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("dataset") != manifest.dataset:
        raise ValueError("frozen split-overlap audit dataset mismatch")
    if "test_rows_in_overlap" in report:
        expected = {
            "test_rows": report.get("test_rows"),
            "excluded_rows": report.get("test_rows_in_overlap"),
            "overlap_unique_feature_rows": report.get(
                "overlap_unique_feature_rows"
            ),
        }
        actual = {
            "test_rows": totals["official_rows"],
            "excluded_rows": totals["excluded_rows"],
            "overlap_unique_feature_rows": overlap_unique_feature_rows,
        }
        mode = "official_split_exact_row_audit"
    elif "exact_cross_split_overlap_unique_feature_rows" in report:
        expected = {
            "excluded_rows": 0,
            "overlap_unique_feature_rows": report.get(
                "exact_cross_split_overlap_unique_feature_rows"
            ),
        }
        actual = {
            "excluded_rows": totals["excluded_rows"],
            "overlap_unique_feature_rows": overlap_unique_feature_rows,
        }
        mode = "feature_group_split_zero_overlap_contract"
    else:
        raise ValueError("unsupported frozen split-overlap audit schema")
    if expected != actual:
        raise RuntimeError(
            f"derived overlap view disagrees with frozen split audit: {actual} != {expected}"
        )
    return {
        "verified": True,
        "mode": mode,
        "path": str(path),
        "sha256": provenance.get("sha256"),
        "canonical_sha256": provenance.get("canonical_sha256"),
        "expected": expected,
        "actual": actual,
    }


def build_overlap_view(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    view_name: str = "duplicate_excluded",
    work_directory: str | Path | None = None,
    batch_rows: int = 50_000,
) -> Path:
    if not EVALUATION_VIEW_NAME.fullmatch(view_name) or view_name == "official":
        raise ValueError("view_name must be a safe non-reserved lower-case name")
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    manifest = load_manifest(manifest_path, verify_hashes=True)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation-view artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    work_root = Path(work_directory).resolve() if work_directory else output.parent
    work_root.mkdir(parents=True, exist_ok=True)
    source = _source_manifest()
    primary_record = _primary_record(manifest)
    try:
        with tempfile.TemporaryDirectory(
            prefix=f"ofra-{view_name}-", dir=work_root
        ) as temporary:
            database = Path(temporary) / "train-features.sqlite3"
            connection = sqlite3.connect(database)
            try:
                connection.execute("PRAGMA journal_mode=OFF")
                connection.execute("PRAGMA synchronous=OFF")
                connection.execute("PRAGMA temp_store=FILE")
                connection.execute("PRAGMA cache_size=-131072")
                connection.execute("PRAGMA page_size=65536")
                connection.execute(
                    "CREATE TABLE train_features ("
                    "feature BLOB PRIMARY KEY, label_mask INTEGER NOT NULL"
                    ") WITHOUT ROWID"
                )
                connection.execute(
                    "CREATE TABLE global_overlap_features ("
                    "feature BLOB PRIMARY KEY"
                    ") WITHOUT ROWID"
                )
                connection.execute(
                    "CREATE TABLE class_overlap_features ("
                    "test_class INTEGER NOT NULL, feature BLOB NOT NULL, "
                    "PRIMARY KEY(test_class, feature)"
                    ") WITHOUT ROWID"
                )
                _index_train(connection, manifest, batch_rows)

                classes_manifest: list[dict[str, object]] = []
                classes_audit: list[dict[str, object]] = []
                presence_matrix = [
                    [0 for _ in manifest.classes] for _ in manifest.classes
                ]
                for record in manifest.classes:
                    shard_manifest: list[dict[str, object]] = []
                    class_audit = {
                        "class_id": record.class_id,
                        "class_name": record.name,
                        "official_rows": 0,
                        "retained_rows": 0,
                        "excluded_rows": 0,
                        "same_label_only_rows": 0,
                        "different_label_only_rows": 0,
                        "mixed_including_same_label_rows": 0,
                        "overlap_unique_feature_rows": 0,
                    }
                    for ordinal, shard in enumerate(record.test):
                        relative_mask = Path("masks") / (
                            f"class_{record.class_id:03d}_test_{ordinal:04d}.keep.npy"
                        )
                        mask_record, shard_audit = _mask_one_shard(
                            connection,
                            manifest,
                            record.class_id,
                            shard,
                            staging / relative_mask,
                            batch_rows,
                        )
                        mask_record["path"] = relative_mask.as_posix()
                        shard_manifest.append(
                            {
                                "ordinal": ordinal,
                                "parent": {
                                    "relative_path": _relative(manifest, shard),
                                    "rows": shard.rows,
                                    "sha256": shard.sha256,
                                },
                                "mask": mask_record,
                            }
                        )
                        for key in (
                            "official_rows",
                            "retained_rows",
                            "excluded_rows",
                            "same_label_only_rows",
                            "different_label_only_rows",
                            "mixed_including_same_label_rows",
                        ):
                            class_audit[key] += int(shard_audit[key])
                        for train_class, count in enumerate(
                            shard_audit["label_presence_rows"]
                        ):
                            presence_matrix[record.class_id][train_class] += int(count)
                    class_audit["overlap_unique_feature_rows"] = int(
                        connection.execute(
                            "SELECT COUNT(*) FROM class_overlap_features "
                            "WHERE test_class = ?",
                            (record.class_id,),
                        ).fetchone()[0]
                    )
                    classes_audit.append(class_audit)
                    classes_manifest.append(
                        {
                            "id": record.class_id,
                            "name": record.name,
                            "official_rows": class_audit["official_rows"],
                            "retained_rows": class_audit["retained_rows"],
                            "excluded_rows": class_audit["excluded_rows"],
                            "shards": shard_manifest,
                        }
                    )
                global_overlap_unique = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM global_overlap_features"
                    ).fetchone()[0]
                )
            finally:
                connection.close()

        totals = {
            "official_rows": sum(int(item["official_rows"]) for item in classes_audit),
            "retained_rows": sum(int(item["retained_rows"]) for item in classes_audit),
            "excluded_rows": sum(int(item["excluded_rows"]) for item in classes_audit),
        }
        if any(int(item["retained_rows"]) <= 0 for item in classes_audit):
            raise RuntimeError("duplicate exclusion leaves at least one class empty")
        frozen_reconciliation = _reconcile_frozen_split_overlap(
            manifest, totals, global_overlap_unique
        )
        semantic_audit: dict[str, object] = {
            "schema_version": 1,
            "report": "ofra_exact_train_test_overlap_mask_derivation_v1",
            "dataset": manifest.dataset,
            "feature_dim": manifest.feature_dim,
            "problem_type": manifest.problem_type,
            "metric_profile": manifest.metric_profile,
            "task_semantics": manifest.task_semantics,
            "normal_class_id": manifest.normal_class_id,
            "algorithm": ALGORITHM,
            "equality_key": EQUALITY_KEY,
            "resource_model": {
                "train_and_unique_overlap_indexes": "disk_backed_sqlite_exact_blob_keys",
                "maximum_feature_rows_per_numpy_batch": batch_rows,
                "global_feature_key_set_materialized_in_python": False,
            },
            "primary": primary_record,
            "builder_source": source,
            "frozen_split_overlap_reconciliation": frozen_reconciliation,
            "totals": totals,
            "overlap_unique_feature_rows": global_overlap_unique,
            "classes": classes_audit,
            "label_presence_matrix": {
                "axis": [
                    {"id": record.class_id, "name": record.name}
                    for record in manifest.classes
                ],
                "semantics": (
                    "cell[test_class][train_class] counts excluded test-row "
                    "occurrences whose exact feature occurs in that train class; "
                    "rows may contribute to multiple train-class columns"
                ),
                "rows": presence_matrix,
            },
        }
        deterministic = canonical_sha256(semantic_audit)
        audit = {
            **semantic_audit,
            "created_utc": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
            "deterministic_result_sha256": deterministic,
        }
        audit["canonical_report_sha256"] = canonical_sha256(audit)
        audit_path = staging / "duplicate_exclusion_audit.json"
        _atomic_json(audit_path, audit)
        audit_sha = sha256_file(audit_path)

        view: dict[str, object] = {
            "schema_version": 1,
            "kind": "streaming_full_test_selection_view",
            "name": view_name,
            "dataset": manifest.dataset,
            "feature_dim": manifest.feature_dim,
            "problem_type": manifest.problem_type,
            "metric_profile": manifest.metric_profile,
            "task_semantics": manifest.task_semantics,
            "normal_class_id": manifest.normal_class_id,
            "tasks": [list(task) for task in manifest.tasks],
            "primary": primary_record,
            "selection": {
                "algorithm": ALGORITHM,
                "equality_key": EQUALITY_KEY,
                "mask_semantics": "true=retain official test row; false=exclude",
                "audit": {
                    "path": audit_path.name,
                    "sha256": audit_sha,
                    "canonical_sha256": audit["canonical_report_sha256"],
                    "deterministic_result_sha256": deterministic,
                },
            },
            "classes": classes_manifest,
            "totals": totals,
        }
        view["canonical_manifest"] = {
            "algorithm": "sha256-canonical-json-excluding-this-field-v1",
            "sha256": canonical_sha256(view),
        }
        view_path = staging / "evaluation_view_manifest.json"
        _atomic_json(view_path, view)
        load_evaluation_view(view_path, manifest, verify_hashes=True)
        os.replace(staging, output)
        return output / view_path.name
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a test-only exact train/test-overlap exclusion mask."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view-name", default="duplicate_excluded")
    parser.add_argument("--work-directory", type=Path)
    parser.add_argument("--batch-rows", type=int, default=50_000)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = build_overlap_view(
        args.manifest,
        args.output_dir,
        view_name=args.view_name,
        work_directory=args.work_directory,
        batch_rows=args.batch_rows,
    )
    print(path)
    return 0
