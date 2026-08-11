"""Command-line interface for the uncapped streaming cache builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import BuildOptions, build_dataset_cache, write_root_manifest
from .specs import DATASET_SPECS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fullcache",
        description=(
            "Build uncapped, chunked float32 feature shards with raw/shard "
            "SHA256 manifests. No model training is performed."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing dataset subdirectories (for example datasets/).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New cache root. Each dataset is written atomically below it.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=[*DATASET_SPECS, "all"],
        help="Dataset to build; repeat as needed. Defaults to all seven datasets.",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=100_000,
        help="Maximum CSV rows parsed at once (default: 100000).",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed mixed into CIC feature-group hashes (default: 42).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace a completed dataset cache after a successful rebuild.",
    )
    parser.add_argument(
        "--allow-partial-files",
        action="store_true",
        help="Permit incomplete raw mirrors; recorded as incomplete in the manifest.",
    )
    parser.add_argument(
        "--allow-unmapped-labels",
        action="store_true",
        help="Drop and count unknown labels instead of failing the build.",
    )
    parser.add_argument(
        "--overlap-batch-rows",
        type=int,
        default=50_000,
        help="Rows per bounded-memory exact official-split overlap batch.",
    )
    parser.add_argument(
        "--overlap-work-directory",
        type=Path,
        help="Optional directory for the temporary exact-overlap SQLite database.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    requested = args.dataset or ["all"]
    if "all" in requested:
        if len(requested) != 1:
            raise SystemExit("--dataset all cannot be combined with another dataset")
        datasets = list(DATASET_SPECS)
    else:
        datasets = list(dict.fromkeys(requested))

    options = BuildOptions(
        chunk_rows=args.chunk_rows,
        split_seed=args.split_seed,
        test_fraction=0.20,
        strict_files=not args.allow_partial_files,
        strict_unmapped_labels=not args.allow_unmapped_labels,
        overwrite=args.overwrite,
        overlap_batch_rows=args.overlap_batch_rows,
        overlap_work_directory=args.overlap_work_directory,
    )
    summaries: list[dict[str, object]] = []
    for dataset in datasets:
        manifest = build_dataset_cache(
            dataset,
            args.data_root,
            args.output_root,
            options,
        )
        summaries.append(
            {
                "dataset": dataset,
                "rows": manifest["quality_counts"]["rows_written"],
                "shards": len(manifest["shards"]),
                "raw_files": len(manifest["raw_files"]),
                "file_contract_complete": manifest["file_contract"]["complete"],
                "streaming_manifest": str(
                    (
                        args.output_root
                        / dataset
                        / manifest["streaming_runner"]["relative_manifest"]
                    ).resolve()
                ),
                "official_split_overlap_unique_rows": manifest[
                    "split_overlap_audit"
                ].get("overlap_unique_feature_rows"),
            }
        )
    root_manifest = write_root_manifest(args.output_root)
    print(
        json.dumps(
            {
                "completed": summaries,
                "root_manifest": str((args.output_root / "manifest.json").resolve()),
                "indexed_datasets": len(root_manifest["datasets"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
