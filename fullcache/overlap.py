"""Exact, bounded-memory official train/test feature-overlap audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .specs import DATASET_SPECS


OFFICIAL_DATASETS = (
    "nsl-kdd",
    "unsw-nb15",
    "nf-ton-iot-v2",
    "malaya-network-gt",
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _row_bytes_counter(array: np.ndarray) -> Counter[bytes]:
    values = np.ascontiguousarray(array, dtype="<f4")
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D shard batch, got {values.shape}")
    width = values.shape[1] * values.dtype.itemsize
    flat = memoryview(values).cast("B")
    return Counter(
        bytes(flat[start : start + width])
        for start in range(0, len(flat), width)
    )


def exact_official_split_overlap(
    dataset_root: str | Path,
    dataset: str,
    shard_entries: Iterable[dict[str, Any]],
    feature_dim: int,
    *,
    batch_rows: int = 50_000,
    work_directory: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Count exact cleaned-feature overlap without retaining a full hash set.

    SQLite is keyed by the complete canonical little-endian float32 row bytes.
    SHA256 is computed only for reporting and collision auditing; it is not the
    equality key. Official splits are observed, never changed.
    """

    if dataset not in OFFICIAL_DATASETS:
        raise ValueError(f"Official-split overlap does not apply to {dataset}")
    if feature_dim < 1 or batch_rows < 1:
        raise ValueError("feature_dim and batch_rows must be positive")
    dataset_root = Path(dataset_root).resolve()
    work_root = (
        Path(work_directory).expanduser().resolve()
        if work_directory is not None
        else dataset_root
    )
    work_root.mkdir(parents=True, exist_ok=True)
    database_path = work_root / f".overlap-{dataset}-{uuid.uuid4().hex}.sqlite3"
    entries = list(shard_entries)
    if not entries:
        raise ValueError("No shard entries supplied for overlap audit")

    connection = sqlite3.connect(database_path)
    total_rows = {"train": 0, "test": 0}
    digest_collisions = 0
    peak_database_bytes = 0
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-65536")
        connection.execute("PRAGMA page_size=65536")
        connection.execute(
            """
            CREATE TABLE feature_rows (
                feature BLOB PRIMARY KEY,
                train_count INTEGER NOT NULL,
                test_count INTEGER NOT NULL
            ) WITHOUT ROWID
            """
        )

        for index, entry in enumerate(entries):
            split = entry.get("split")
            if split not in {"train", "test"}:
                raise ValueError(f"Invalid shard split: {split!r}")
            path = dataset_root / entry["relative_path"]
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.ndim != 2 or array.shape[1] != feature_dim:
                raise ValueError(f"Invalid shard shape during overlap audit: {path}")
            if progress:
                progress(
                    f"[{dataset}] exact overlap {index + 1}/{len(entries)}: "
                    f"{entry['relative_path']}"
                )
            for start in range(0, len(array), batch_rows):
                batch = np.asarray(array[start : start + batch_rows], dtype="<f4")
                if not np.isfinite(batch).all():
                    raise ValueError(f"Non-finite row in overlap input: {path}")
                counts = _row_bytes_counter(batch)
                if split == "train":
                    connection.executemany(
                        """
                        INSERT INTO feature_rows(feature, train_count, test_count)
                        VALUES (?, ?, 0)
                        ON CONFLICT(feature) DO UPDATE SET
                            train_count = train_count + excluded.train_count
                        """,
                        ((sqlite3.Binary(row), count) for row, count in counts.items()),
                    )
                else:
                    connection.executemany(
                        """
                        INSERT INTO feature_rows(feature, train_count, test_count)
                        VALUES (?, 0, ?)
                        ON CONFLICT(feature) DO UPDATE SET
                            test_count = test_count + excluded.test_count
                        """,
                        ((sqlite3.Binary(row), count) for row, count in counts.items()),
                    )
                total_rows[split] += len(batch)
            connection.commit()
            peak_database_bytes = max(peak_database_bytes, database_path.stat().st_size)

        def scalar(query: str) -> int:
            value = connection.execute(query).fetchone()[0]
            return int(value or 0)

        train_unique = scalar(
            "SELECT COUNT(*) FROM feature_rows WHERE train_count > 0"
        )
        test_unique = scalar(
            "SELECT COUNT(*) FROM feature_rows WHERE test_count > 0"
        )
        union_unique = scalar("SELECT COUNT(*) FROM feature_rows")
        overlap_unique = scalar(
            "SELECT COUNT(*) FROM feature_rows "
            "WHERE train_count > 0 AND test_count > 0"
        )
        train_rows_in_overlap = scalar(
            "SELECT SUM(train_count) FROM feature_rows "
            "WHERE train_count > 0 AND test_count > 0"
        )
        test_rows_in_overlap = scalar(
            "SELECT SUM(test_count) FROM feature_rows "
            "WHERE train_count > 0 AND test_count > 0"
        )

        connection.execute(
            "CREATE TABLE overlap_sha256 (digest BLOB PRIMARY KEY) WITHOUT ROWID"
        )
        cursor = connection.execute(
            "SELECT feature FROM feature_rows "
            "WHERE train_count > 0 AND test_count > 0"
        )
        while True:
            rows = cursor.fetchmany(10_000)
            if not rows:
                break
            connection.executemany(
                "INSERT OR IGNORE INTO overlap_sha256(digest) VALUES (?)",
                (
                    (sqlite3.Binary(hashlib.sha256(bytes(row[0])).digest()),)
                    for row in rows
                ),
            )
        connection.commit()
        overlap_sha256_digests = scalar("SELECT COUNT(*) FROM overlap_sha256")
        digest_collisions = overlap_unique - overlap_sha256_digests
        peak_database_bytes = max(peak_database_bytes, database_path.stat().st_size)

        report: dict[str, Any] = {
            "schema_version": 1,
            "dataset": dataset,
            "created_utc": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
            "algorithm": "sqlite_exact_float32_row_bytes_v1",
            "equality_key": (
                "complete canonical little-endian float32 feature-row bytes; "
                "no probabilistic hash is used for equality"
            ),
            "official_split_action": "audit_only_no_delete_no_resplit",
            "feature_dim": feature_dim,
            "bytes_per_feature_row": feature_dim * 4,
            "bounded_memory_batch_rows": batch_rows,
            "temporary_database_peak_bytes": peak_database_bytes,
            "train_rows": total_rows["train"],
            "test_rows": total_rows["test"],
            "train_unique_feature_rows": train_unique,
            "test_unique_feature_rows": test_unique,
            "union_unique_feature_rows": union_unique,
            "overlap_unique_feature_rows": overlap_unique,
            "overlap_unique_sha256_digests": overlap_sha256_digests,
            "sha256_digest_collisions_detected": digest_collisions,
            "train_rows_in_overlap": train_rows_in_overlap,
            "test_rows_in_overlap": test_rows_in_overlap,
            "train_rows_in_overlap_rate": (
                train_rows_in_overlap / total_rows["train"]
                if total_rows["train"]
                else None
            ),
            "test_rows_in_overlap_rate": (
                test_rows_in_overlap / total_rows["test"]
                if total_rows["test"]
                else None
            ),
        }
        deterministic_fields = {
            key: value
            for key, value in report.items()
            if key
            not in {
                "created_utc",
                "bounded_memory_batch_rows",
                "temporary_database_peak_bytes",
            }
        }
        report["deterministic_result_sha256"] = _canonical_sha256(
            deterministic_fields
        )
        report["canonical_report_sha256"] = _canonical_sha256(report)
        return report
    finally:
        connection.close()
        if database_path.exists():
            database_path.unlink()


def audit_completed_cache(
    cache_root: str | Path,
    dataset: str,
    *,
    batch_rows: int = 50_000,
    work_directory: str | Path | None = None,
    progress: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    dataset_root = Path(cache_root).expanduser().resolve() / dataset
    manifest = json.loads(
        (dataset_root / "manifest.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (dataset_root / "feature_schema.json").read_text(encoding="utf-8")
    )
    return exact_official_split_overlap(
        dataset_root,
        dataset,
        manifest["shards"],
        int(schema["feature_count"]),
        batch_rows=batch_rows,
        work_directory=work_directory,
        progress=progress,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fullcache.overlap",
        description=(
            "Recompute an exact official train/test cleaned-feature overlap "
            "audit using an external SQLite set."
        ),
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--dataset", action="append", choices=[*OFFICIAL_DATASETS, "all"]
    )
    parser.add_argument("--batch-rows", type=int, default=50_000)
    parser.add_argument("--work-directory", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    requested = args.dataset or ["all"]
    datasets = list(OFFICIAL_DATASETS) if "all" in requested else requested
    reports = [
        audit_completed_cache(
            args.cache_root,
            dataset,
            batch_rows=args.batch_rows,
            work_directory=args.work_directory,
        )
        for dataset in datasets
    ]
    payload = {"reports": reports}
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
