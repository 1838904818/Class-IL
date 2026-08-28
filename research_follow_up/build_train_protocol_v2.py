"""Build the OFRA adaptive normal-class-cap training protocol.

This version reserves a deterministic training-only calibration split, keeps
every available attack fit row, and caps only the normal class.  The normal
class cap is derived from the largest attack-class fit pool, so the policy has
no fixed row-count threshold.  Official test shards are copied or hard-linked
byte-for-byte and are never sampled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from build_train_protocol import (
    SCHEMA_VERSION,
    _extract_rows,
    _inspect_shards,
    _materialize_test,
    _reorder_tasks,
    _resolve_shards,
    _write_json,
    canonical_sha256,
    sha256_file,
)

ALGORITHM = (
    "deterministic_normal_to_largest_attack_fit_cap_with_train_calibration_v2"
)
POLICY = "normal_to_largest_attack"


def derived_seed(master_seed: int, source_sha256: str, class_id: int) -> int:
    payload = f"{master_seed}|{source_sha256}|{class_id}|{POLICY}|v2"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little")


def _calibration_rows(source_rows: int, fraction: float) -> int:
    if fraction > 0.0 and source_rows > 1:
        return min(source_rows - 1, max(1, math.floor(source_rows * fraction)))
    return 0


def build_protocol_v2(
    source_manifest: Path,
    output_dir: Path,
    *,
    calibration_fraction: float,
    seed: int,
    held_out_class_id: int | None = None,
    test_mode: str = "hardlink",
) -> dict[str, Path | str | int]:
    source_manifest = source_manifest.resolve()
    output_dir = output_dir.resolve()
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
    normal_class_id = source.get("normal_class_id")
    if isinstance(normal_class_id, bool) or not isinstance(normal_class_id, int):
        raise TypeError("source manifest must declare an integer normal_class_id")
    class_ids = {int(record["id"]) for record in classes}
    if normal_class_id not in class_ids:
        raise ValueError("normal_class_id is absent from classes")
    attack_ids = sorted(class_ids - {normal_class_id})
    if not attack_ids:
        raise ValueError("adaptive normal-class cap requires at least one attack class")

    source_sha = sha256_file(source_manifest)
    builder_path = Path(__file__).resolve()
    helper_path = Path(__file__).with_name("build_train_protocol.py").resolve()
    source_base = source_manifest.parent

    prepared: list[dict[str, object]] = []
    for class_record in classes:
        class_id = int(class_record["id"])
        train_shards = _resolve_shards(
            source_base, class_record.get("train"), label=f"class {class_id} train"
        )
        test_records = class_record.get("test")
        test_shards = _resolve_shards(
            source_base, test_records, label=f"class {class_id} test"
        )
        source_rows, dtype = _inspect_shards(train_shards, feature_dim)
        class_seed = derived_seed(seed, source_sha, class_id)
        permutation = np.random.default_rng(class_seed).permutation(source_rows)
        calibration_rows = _calibration_rows(source_rows, calibration_fraction)
        calibration_indices = np.sort(permutation[:calibration_rows])
        fit_pool = permutation[calibration_rows:]
        prepared.append(
            {
                "record": class_record,
                "id": class_id,
                "name": str(class_record["name"]),
                "seed": class_seed,
                "train_shards": train_shards,
                "test_records": test_records,
                "test_shards": test_shards,
                "source_rows": source_rows,
                "dtype": dtype,
                "calibration_indices": calibration_indices,
                "fit_pool": fit_pool,
            }
        )

    attack_fit_reference = max(
        len(item["fit_pool"])
        for item in prepared
        if int(item["id"]) != normal_class_id
    )
    if attack_fit_reference <= 0:
        raise ValueError("attack fit pools are empty")

    output_dir.mkdir(parents=True, exist_ok=False)
    derived_classes: list[dict] = []
    audit_classes: list[dict] = []
    for item in prepared:
        class_id = int(item["id"])
        class_dir = output_dir / f"class_{class_id:02d}"
        class_dir.mkdir()
        fit_pool = np.asarray(item["fit_pool"], dtype=np.int64)
        fit_target = (
            min(len(fit_pool), attack_fit_reference)
            if class_id == normal_class_id
            else len(fit_pool)
        )
        fit_indices = np.sort(fit_pool[:fit_target])
        calibration_indices = np.asarray(item["calibration_indices"], dtype=np.int64)
        if np.intersect1d(calibration_indices, fit_indices).size:
            raise RuntimeError("fit/calibration row overlap")

        fit_path = class_dir / "train_fit.npy"
        calibration_path = class_dir / "train_calibration.npy"
        _extract_rows(
            item["train_shards"],
            fit_indices,
            fit_path,
            feature_dim=feature_dim,
            dtype=item["dtype"],
        )
        _extract_rows(
            item["train_shards"],
            calibration_indices,
            calibration_path,
            feature_dim=feature_dim,
            dtype=item["dtype"],
        )

        test_records = item["test_records"]
        if not isinstance(test_records, list):
            raise TypeError(f"class {class_id} test records must be a list")
        derived_tests: list[dict] = []
        test_audit: list[dict] = []
        for shard_index, (source_test, test_record) in enumerate(
            zip(item["test_shards"], test_records, strict=True)
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
                "name": item["name"],
                "train": [
                    {
                        "path": fit_path.relative_to(output_dir).as_posix(),
                        "rows": fit_target,
                        "sha256": fit_hash,
                    }
                ],
                "test": derived_tests,
            }
        )
        audit_classes.append(
            {
                "id": class_id,
                "name": item["name"],
                "role": "normal" if class_id == normal_class_id else "attack",
                "seed": int(item["seed"]),
                "source_train_rows": int(item["source_rows"]),
                "calibration_rows": len(calibration_indices),
                "fit_pool_rows": len(fit_pool),
                "fit_rows": fit_target,
                "fit_capped": fit_target < len(fit_pool),
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

    if any(
        item["fit_capped"]
        for item in audit_classes
        if int(item["id"]) != normal_class_id
    ):
        raise RuntimeError("adaptive policy must not cap attack classes")

    audit = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": ALGORITHM,
        "builder": {"path": str(builder_path), "sha256": sha256_file(builder_path)},
        "helper_dependency": {
            "path": str(helper_path),
            "sha256": sha256_file(helper_path),
        },
        "source_manifest": {"path": str(source_manifest), "sha256": source_sha},
        "configuration": {
            "sampling_policy": POLICY,
            "normal_class_id": normal_class_id,
            "normal_fit_cap_rows": attack_fit_reference,
            "normal_fit_cap_source": "largest attack-class fit pool after calibration split",
            "calibration_fraction": calibration_fraction,
            "seed": seed,
            "held_out_class_id": held_out_class_id,
            "test_mode": test_mode,
        },
        "invariants": {
            "attack_fit_rows_sampled": False,
            "minority_oversampling": False,
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
        **{
            key: value
            for key, value in source.items()
            if key not in {"classes", "tasks", "source"}
        },
        "dataset": f"{source['dataset']}-normalcap-largest-attack",
        "classes": derived_classes,
        "tasks": _reorder_tasks(tasks, held_out_class_id),
        "source": {
            "builder": "ofra-derived-train-protocol-v2",
            "algorithm": ALGORITHM,
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
        "policy": POLICY,
        "normal_class_id": normal_class_id,
        "normal_fit_cap_rows": attack_fit_reference,
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
        "normal_fit_cap_rows": attack_fit_reference,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--held-out-class-id", type=int)
    parser.add_argument("--test-mode", choices=("hardlink", "copy"), default="hardlink")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_protocol_v2(
        args.source_manifest,
        args.output_dir,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
        held_out_class_id=args.held_out_class_id,
        test_mode=args.test_mode,
    )
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))


if __name__ == "__main__":
    main()
