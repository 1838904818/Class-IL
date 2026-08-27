"""Build a deterministic, hash-audited OFRA derived training protocol.

The builder caps only fit-training rows and reserves a training-only calibration
split. Official test shards are hard-linked or copied byte-for-byte and are
never sampled. An optional held-out class is moved to a singleton final task so
the preceding checkpoint can be evaluated as an open-world state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections.abc import Iterable
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
ALGORITHM = "deterministic_per_class_fit_cap_with_train_calibration_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derived_seed(master_seed: int, source_sha256: str, class_id: int) -> int:
    payload = f"{master_seed}|{source_sha256}|{class_id}|fit-calibration-v1"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _resolve_shards(base: Path, records: object, *, label: str) -> list[Path]:
    if not isinstance(records, list) or not records:
        raise ValueError(f"{label} must be a non-empty list")
    result: list[Path] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise TypeError(f"{label}[{index}] lacks a path")
        path = (base / record["path"]).resolve()
        if not path.is_file() or path.suffix.lower() != ".npy":
            raise FileNotFoundError(path)
        expected = record.get("sha256")
        if not isinstance(expected, str) or sha256_file(path) != expected.lower():
            raise ValueError(f"source shard hash mismatch: {path}")
        result.append(path)
    return result


def _inspect_shards(paths: Iterable[Path], feature_dim: int) -> tuple[int, np.dtype]:
    total = 0
    dtype: np.dtype | None = None
    for path in paths:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            if values.ndim != 2 or values.shape[1] != feature_dim:
                raise ValueError(f"unexpected shard shape for {path}: {values.shape}")
            if not np.issubdtype(values.dtype, np.number):
                raise TypeError(f"non-numeric shard: {path}")
            if dtype is None:
                dtype = values.dtype
            elif values.dtype != dtype:
                raise TypeError("all shards for one class must have the same dtype")
            total += len(values)
        finally:
            mapping = getattr(values, "_mmap", None)
            if mapping is not None:
                mapping.close()
    if dtype is None or total <= 0:
        raise ValueError("class has no source rows")
    return total, dtype


def _extract_rows(
    shards: list[Path],
    indices: np.ndarray,
    output: Path,
    *,
    feature_dim: int,
    dtype: np.dtype,
) -> None:
    ordered = np.asarray(indices, dtype=np.int64)
    if ordered.ndim != 1 or len(ordered) != len(np.unique(ordered)):
        raise ValueError("selected indices must be one-dimensional and unique")
    ordered.sort()
    target = np.lib.format.open_memmap(
        output, mode="w+", dtype=dtype, shape=(len(ordered), feature_dim)
    )
    written = 0
    offset = 0
    try:
        for path in shards:
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            try:
                end = offset + len(values)
                left = int(np.searchsorted(ordered, offset, side="left"))
                right = int(np.searchsorted(ordered, end, side="left"))
                if right > left:
                    local = ordered[left:right] - offset
                    count = right - left
                    target[written : written + count] = values[local]
                    written += count
                offset = end
            finally:
                mapping = getattr(values, "_mmap", None)
                if mapping is not None:
                    mapping.close()
        if written != len(ordered):
            raise RuntimeError(
                f"selected-row extraction mismatch: expected={len(ordered)} wrote={written}"
            )
        target.flush()
    finally:
        mapping = getattr(target, "_mmap", None)
        if mapping is not None:
            mapping.close()


def _materialize_test(
    source: Path, destination: Path, mode: str, *, expected_sha256: str
) -> None:
    expected_sha256 = expected_sha256.lower()
    source_before = sha256_file(source)
    if source_before != expected_sha256:
        raise RuntimeError(
            f"source test shard changed before materialization: {source}"
        )
    if mode == "hardlink":
        os.link(source, destination)
        source_stat = source.stat()
        destination_stat = destination.stat()
        if (
            source_stat.st_dev != destination_stat.st_dev
            or source_stat.st_ino != destination_stat.st_ino
        ):
            raise RuntimeError("hardlink materialization did not preserve device/inode")
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"unsupported test materialization mode: {mode}")
    source_after = sha256_file(source)
    destination_after = sha256_file(destination)
    if source_after != expected_sha256 or destination_after != expected_sha256:
        raise RuntimeError("official test shard changed during materialization")


def _reorder_tasks(tasks: list[list[int]], held_out_class_id: int | None) -> list[list[int]]:
    copied = [list(map(int, task)) for task in tasks]
    if held_out_class_id is None:
        return copied
    if held_out_class_id in copied[0]:
        raise ValueError(
            "held-out class cannot come from Task 0 because OFRA normalisation and "
            "initial pretraining require at least two Task-0 classes"
        )
    if not any(held_out_class_id in task for task in copied):
        raise ValueError("held-out class is absent from the task stream")
    reordered: list[list[int]] = []
    for task in copied:
        remaining = [class_id for class_id in task if class_id != held_out_class_id]
        if remaining:
            reordered.append(remaining)
    reordered.append([held_out_class_id])
    return reordered


def build_protocol(
    source_manifest: Path,
    output_dir: Path,
    *,
    train_cap: int,
    calibration_fraction: float,
    seed: int,
    held_out_class_id: int | None = None,
    test_mode: str = "hardlink",
) -> dict[str, Path | str]:
    source_manifest = source_manifest.resolve()
    output_dir = output_dir.resolve()
    if train_cap <= 0:
        raise ValueError("train_cap must be positive")
    if not 0.0 <= calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in [0, 1)")
    if test_mode not in {"hardlink", "copy"}:
        raise ValueError("test_mode must be 'hardlink' or 'copy'")

    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    if source.get("schema_version") != 1:
        raise ValueError("source manifest must use schema_version=1")
    feature_dim = int(source["feature_dim"])
    classes = source.get("classes")
    tasks = source.get("tasks")
    if not isinstance(classes, list) or not isinstance(tasks, list):
        raise TypeError("source manifest lacks classes/tasks")
    source_sha = sha256_file(source_manifest)
    builder_path = Path(__file__).resolve()
    builder_sha = sha256_file(builder_path)
    source_base = source_manifest.parent
    output_dir.mkdir(parents=True, exist_ok=False)

    derived_classes: list[dict] = []
    audit_classes: list[dict] = []
    for class_record in classes:
        class_id = int(class_record["id"])
        class_name = str(class_record["name"])
        class_dir = output_dir / f"class_{class_id:02d}"
        class_dir.mkdir()
        train_shards = _resolve_shards(
            source_base, class_record.get("train"), label=f"class {class_id} train"
        )
        test_records = class_record.get("test")
        test_shards = _resolve_shards(
            source_base, test_records, label=f"class {class_id} test"
        )
        source_rows, dtype = _inspect_shards(train_shards, feature_dim)
        class_seed = derived_seed(seed, source_sha, class_id)
        rng = np.random.default_rng(class_seed)

        if calibration_fraction > 0.0 and source_rows > 1:
            calibration_rows = min(
                source_rows - 1,
                max(1, math.floor(source_rows * calibration_fraction)),
            )
        else:
            calibration_rows = 0
        permutation = rng.permutation(source_rows)
        calibration_indices = np.sort(permutation[:calibration_rows])
        fit_pool = permutation[calibration_rows:]
        fit_rows = min(train_cap, len(fit_pool))
        fit_indices = np.sort(fit_pool[:fit_rows])
        if np.intersect1d(calibration_indices, fit_indices).size:
            raise RuntimeError("fit/calibration row overlap")

        fit_path = class_dir / "train_fit.npy"
        calibration_path = class_dir / "train_calibration.npy"
        _extract_rows(
            train_shards,
            fit_indices,
            fit_path,
            feature_dim=feature_dim,
            dtype=dtype,
        )
        _extract_rows(
            train_shards,
            calibration_indices,
            calibration_path,
            feature_dim=feature_dim,
            dtype=dtype,
        )

        derived_tests: list[dict] = []
        test_audit: list[dict] = []
        if not isinstance(test_records, list):
            raise TypeError(f"class {class_id} test records must be a list")
        for shard_index, (source_test, test_record) in enumerate(
            zip(test_shards, test_records, strict=True)
        ):
            if not isinstance(test_record, dict) or not isinstance(
                test_record.get("sha256"), str
            ):
                raise TypeError(f"class {class_id} test record lacks sha256")
            destination = class_dir / f"test_{shard_index:03d}.npy"
            _materialize_test(
                source_test,
                destination,
                test_mode,
                expected_sha256=test_record["sha256"],
            )
            values = np.load(destination, mmap_mode="r", allow_pickle=False)
            try:
                rows = len(values)
            finally:
                mapping = getattr(values, "_mmap", None)
                if mapping is not None:
                    mapping.close()
            digest = sha256_file(destination)
            relative = destination.relative_to(output_dir).as_posix()
            derived_tests.append({"path": relative, "rows": rows, "sha256": digest})
            test_audit.append(
                {
                    "source_path": str(source_test),
                    "derived_path": relative,
                    "rows": rows,
                    "sha256": digest,
                    "byte_identical": True,
                    "materialization": test_mode,
                }
            )

        fit_hash = sha256_file(fit_path)
        calibration_hash = sha256_file(calibration_path)
        derived_classes.append(
            {
                "id": class_id,
                "name": class_name,
                "train": [
                    {
                        "path": fit_path.relative_to(output_dir).as_posix(),
                        "rows": fit_rows,
                        "sha256": fit_hash,
                    }
                ],
                "test": derived_tests,
            }
        )
        audit_classes.append(
            {
                "id": class_id,
                "name": class_name,
                "seed": class_seed,
                "source_train_rows": source_rows,
                "calibration_rows": calibration_rows,
                "fit_pool_rows": len(fit_pool),
                "fit_rows": fit_rows,
                "fit_capped": fit_rows < len(fit_pool),
                "fit_indices_sha256": canonical_sha256(fit_indices.tolist()),
                "calibration_indices_sha256": canonical_sha256(
                    calibration_indices.tolist()
                ),
                "fit_calibration_disjoint": True,
                "fit": {
                    "path": fit_path.relative_to(output_dir).as_posix(),
                    "sha256": fit_hash,
                },
                "calibration": {
                    "path": calibration_path.relative_to(output_dir).as_posix(),
                    "sha256": calibration_hash,
                },
                "official_test": test_audit,
            }
        )

    audit = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": ALGORITHM,
        "builder": {"path": str(builder_path), "sha256": builder_sha},
        "source_manifest": {
            "path": str(source_manifest),
            "sha256": source_sha,
        },
        "configuration": {
            "train_cap": train_cap,
            "calibration_fraction": calibration_fraction,
            "seed": seed,
            "held_out_class_id": held_out_class_id,
            "test_mode": test_mode,
        },
        "invariants": {
            "official_test_sampled": False,
            "official_test_byte_identical": True,
            "calibration_source": "source training rows only",
            "fit_calibration_disjoint": True,
        },
        "classes": audit_classes,
    }
    audit_path = output_dir / "sampling_audit.json"
    _write_json(audit_path, audit)
    audit_sha = sha256_file(audit_path)

    derived = {
        **{key: value for key, value in source.items() if key not in {"classes", "tasks", "source"}},
        "dataset": f"{source['dataset']}-traincap{train_cap}",
        "classes": derived_classes,
        "tasks": _reorder_tasks(tasks, held_out_class_id),
        "source": {
            "builder": "ofra-derived-train-protocol-v1",
            "source_manifest_sha256": source_sha,
            "sampling_audit_sha256": audit_sha,
            "official_test_policy": "byte-identical full source test shards",
        },
    }
    manifest_path = output_dir / "streaming_manifest.json"
    _write_json(manifest_path, derived)
    manifest_sha = sha256_file(manifest_path)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha},
        "audit": {"path": str(audit_path), "sha256": audit_sha},
        "class_count": len(derived_classes),
        "fit_rows": sum(item["fit_rows"] for item in audit_classes),
        "calibration_rows": sum(item["calibration_rows"] for item in audit_classes),
        "official_test_rows": sum(
            shard["rows"]
            for item in audit_classes
            for shard in item["official_test"]
        ),
    }
    summary_path = output_dir / "BUILD_SUMMARY.json"
    _write_json(summary_path, summary)
    return {
        "manifest": manifest_path,
        "manifest_sha256": manifest_sha,
        "audit": audit_path,
        "audit_sha256": audit_sha,
        "summary": summary_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-cap", type=int, default=50_000)
    parser.add_argument("--calibration-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--held-out-class-id", type=int)
    parser.add_argument("--test-mode", choices=("hardlink", "copy"), default="hardlink")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_protocol(
        args.source_manifest,
        args.output_dir,
        train_cap=args.train_cap,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
        held_out_class_id=args.held_out_class_id,
        test_mode=args.test_mode,
    )
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))


if __name__ == "__main__":
    main()
