"""Bounded-memory verification for completed fullcache artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .core import builder_source_record, feature_hash_splits, sha256_file
from .overlap import OFFICIAL_DATASETS, exact_official_split_overlap
from .specs import DATASET_SPECS


def verify_dataset_cache(
    cache_root: str | Path,
    dataset: str,
    *,
    verify_raw_hashes: bool = True,
    scan_rows: int = 100_000,
    recompute_overlap: bool = False,
    overlap_batch_rows: int = 50_000,
    overlap_work_directory: str | Path | None = None,
) -> dict[str, Any]:
    if dataset not in DATASET_SPECS:
        raise KeyError(f"Unsupported dataset: {dataset}")
    if scan_rows < 1:
        raise ValueError("scan_rows must be positive")
    spec = DATASET_SPECS[dataset]
    dataset_root = Path(cache_root).expanduser().resolve() / dataset
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != dataset or manifest.get("uncapped") is not True:
        raise ValueError(f"Invalid uncapped manifest identity: {manifest_path}")

    canonical_manifest = manifest.get("canonical_manifest", {})
    manifest_basis = dict(manifest)
    manifest_basis.pop("canonical_manifest", None)
    manifest_basis_bytes = json.dumps(
        manifest_basis,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    actual_manifest_canonical = hashlib.sha256(manifest_basis_bytes).hexdigest()
    if canonical_manifest.get("sha256") != actual_manifest_canonical:
        raise ValueError(f"Canonical manifest hash mismatch: {manifest_path}")

    source_record = manifest.get("builder_source", {})
    source_basis = dict(source_record)
    recorded_source_hash = source_basis.pop("canonical_source_sha256", None)
    source_basis_bytes = json.dumps(
        source_basis,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if hashlib.sha256(source_basis_bytes).hexdigest() != recorded_source_hash:
        raise ValueError("Builder source record has an invalid canonical hash")
    if builder_source_record()["canonical_source_sha256"] != recorded_source_hash:
        raise ValueError(
            "Installed fullcache Python source differs from the source recorded "
            "by this cache"
        )

    for relative, expected in manifest["sidecars"].items():
        actual = sha256_file(dataset_root / relative)
        if actual.lower() != expected.lower():
            raise ValueError(f"Sidecar SHA256 mismatch: {relative}")

    raw_files_verified = 0
    if verify_raw_hashes:
        source = Path(manifest["source_directory"])
        for entry in manifest["raw_files"]:
            path = source / entry["relative_path"]
            if path.stat().st_size != entry["size_bytes"]:
                raise ValueError(f"Raw size mismatch: {path}")
            if sha256_file(path).lower() != entry["sha256"].lower():
                raise ValueError(f"Raw SHA256 mismatch: {path}")
            raw_files_verified += 1

    feature_dim = len(
        json.loads((dataset_root / "feature_schema.json").read_text("utf-8"))[
            "feature_columns"
        ]
    )
    actual_counts = {
        split: {class_name: 0 for class_name in spec.class_order}
        for split in ("train", "test")
    }
    rows_scanned = 0
    strategy = manifest["split_protocol"]["strategy"]
    capture_splits: dict[str, str] = {}
    if strategy == "frozen_capture_manifest":
        for raw_entry in manifest["raw_files"]:
            relative = raw_entry.get("relative_path")
            capture_split = raw_entry.get("capture_split")
            if (
                not isinstance(relative, str)
                or capture_split not in {"train", "test"}
                or relative in capture_splits
            ):
                raise ValueError("Invalid or duplicate frozen capture split record")
            capture_splits[relative] = capture_split
        declared_test = set(manifest["split_protocol"]["test_captures"])
        actual_test = {
            relative
            for relative, capture_split in capture_splits.items()
            if capture_split == "test"
        }
        if actual_test != declared_test:
            raise ValueError("Frozen test captures disagree with raw-file provenance")
    for entry in manifest["shards"]:
        path = dataset_root / entry["relative_path"]
        if sha256_file(path).lower() != entry["sha256"].lower():
            raise ValueError(f"Shard SHA256 mismatch: {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 2 or array.shape != (entry["rows"], feature_dim):
            raise ValueError(
                f"Shard shape mismatch: {path}; manifest={entry['rows']}x{feature_dim}, "
                f"file={array.shape}"
            )
        if array.dtype.kind != "f" or array.dtype.itemsize != 4:
            raise ValueError(f"Shard is not float32: {path} ({array.dtype})")
        split = entry["split"]
        class_name = entry["class"]
        if strategy == "frozen_capture_manifest":
            source_file = entry.get("source_file")
            if capture_splits.get(source_file) != split:
                raise ValueError(
                    f"Frozen capture crosses or lacks its declared split: {source_file}"
                )
        actual_counts[split][class_name] += int(len(array))
        for start in range(0, len(array), scan_rows):
            batch = np.asarray(array[start : start + scan_rows], dtype=np.float32)
            if not np.isfinite(batch).all():
                raise ValueError(f"Non-finite feature in shard: {path}")
            if strategy == "feature_hash_group":
                assigned = feature_hash_splits(
                    batch,
                    seed=int(manifest["split_protocol"]["split_seed"]),
                    test_fraction=float(manifest["split_protocol"]["test_fraction"]),
                )
                if not np.all(assigned == split):
                    raise ValueError(
                        f"Feature-hash split violation in {path}: rows assigned to both splits"
                    )
            rows_scanned += len(batch)

    if actual_counts != manifest["row_counts"]:
        raise ValueError(
            f"Split/class row-count mismatch for {dataset}: "
            f"manifest={manifest['row_counts']}, actual={actual_counts}"
        )
    if rows_scanned != manifest["quality_counts"]["rows_written"]:
        raise ValueError(
            f"Total row-count mismatch for {dataset}: scanned={rows_scanned}, "
            f"manifest={manifest['quality_counts']['rows_written']}"
        )

    overlap_path = dataset_root / "split_overlap_audit.json"
    overlap = json.loads(overlap_path.read_text(encoding="utf-8"))
    overlap_basis = dict(overlap)
    overlap_canonical = overlap_basis.pop("canonical_report_sha256", None)
    overlap_basis_bytes = json.dumps(
        overlap_basis,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if hashlib.sha256(overlap_basis_bytes).hexdigest() != overlap_canonical:
        raise ValueError("Split-overlap audit canonical hash mismatch")
    overlap_recomputed = False
    if dataset in OFFICIAL_DATASETS:
        if overlap.get("train_rows") != sum(actual_counts["train"].values()):
            raise ValueError("Overlap audit train-row count disagrees with shards")
        if overlap.get("test_rows") != sum(actual_counts["test"].values()):
            raise ValueError("Overlap audit test-row count disagrees with shards")
        if overlap.get("equality_key", "").startswith("complete canonical") is False:
            raise ValueError("Official overlap audit did not use exact full-row equality")
        if recompute_overlap:
            recomputed = exact_official_split_overlap(
                dataset_root,
                dataset,
                manifest["shards"],
                feature_dim,
                batch_rows=overlap_batch_rows,
                work_directory=overlap_work_directory,
                progress=None,
            )
            if recomputed["deterministic_result_sha256"] != overlap.get(
                "deterministic_result_sha256"
            ):
                raise ValueError("Recomputed exact overlap audit does not match manifest")
            overlap_recomputed = True
    elif overlap.get("exact_cross_split_overlap_unique_feature_rows") != 0:
        raise ValueError("Feature-hash split reports non-zero exact overlap")

    streaming_path = dataset_root / manifest["streaming_runner"]["relative_manifest"]
    from streaming_full.data import load_manifest

    runner_manifest = load_manifest(streaming_path, verify_hashes=False)
    if runner_manifest.tasks != spec.tasks:
        raise ValueError(
            f"streaming_full task contract mismatch: {runner_manifest.tasks} != {spec.tasks}"
        )
    if (
        runner_manifest.normal_class_id != spec.normal_class_id
        or runner_manifest.feature_dim != feature_dim
    ):
        raise ValueError("streaming_full class-anchor or feature dimension mismatch")
    if (
        runner_manifest.problem_type != spec.problem_type
        or runner_manifest.task_semantics != spec.task_semantics
        or runner_manifest.metric_profile != spec.metric_profile
    ):
        raise ValueError("streaming_full semantic profile mismatch")

    return {
        "dataset": dataset,
        "status": "ok",
        "rows_scanned": rows_scanned,
        "shards_verified": len(manifest["shards"]),
        "raw_files_verified": raw_files_verified,
        "raw_hashes_skipped": not verify_raw_hashes,
        "streaming_manifest": str(streaming_path),
        "streaming_manifest_sha256": sha256_file(streaming_path),
        "builder_source_sha256": recorded_source_hash,
        "canonical_manifest_sha256": actual_manifest_canonical,
        "overlap_unique_feature_rows": overlap.get(
            "overlap_unique_feature_rows",
            overlap.get("exact_cross_split_overlap_unique_feature_rows"),
        ),
        "train_rows_in_overlap": overlap.get("train_rows_in_overlap", 0),
        "test_rows_in_overlap": overlap.get("test_rows_in_overlap", 0),
        "overlap_recomputed": overlap_recomputed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fullcache.verify",
        description="Verify raw/shard hashes, shapes, finiteness, counts, and split rules.",
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=[*DATASET_SPECS, "all"],
        help="Dataset to verify; repeat as needed. Defaults to all cached datasets.",
    )
    parser.add_argument(
        "--recompute-overlap",
        action="store_true",
        help="Rebuild the exact SQLite overlap set and compare its deterministic hash.",
    )
    parser.add_argument("--overlap-batch-rows", type=int, default=50_000)
    parser.add_argument("--overlap-work-directory", type=Path)
    parser.add_argument("--scan-rows", type=int, default=100_000)
    parser.add_argument(
        "--skip-raw-hashes",
        action="store_true",
        help="Skip the expensive second read of raw files; shard checks still run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    requested = args.dataset or ["all"]
    if "all" in requested:
        if len(requested) != 1:
            raise SystemExit("--dataset all cannot be combined with another dataset")
        datasets = [
            dataset
            for dataset in DATASET_SPECS
            if (args.cache_root / dataset / "manifest.json").is_file()
        ]
    else:
        datasets = list(dict.fromkeys(requested))
    if not datasets:
        raise SystemExit("No completed dataset manifests found")
    results = [
        verify_dataset_cache(
            args.cache_root,
            dataset,
            verify_raw_hashes=not args.skip_raw_hashes,
            scan_rows=args.scan_rows,
            recompute_overlap=args.recompute_overlap,
            overlap_batch_rows=args.overlap_batch_rows,
            overlap_work_directory=args.overlap_work_directory,
        )
        for dataset in datasets
    ]
    print(json.dumps({"verified": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
