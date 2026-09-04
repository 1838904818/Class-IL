#!/usr/bin/env python3
"""Fail closed on source integrity and CPU same-batch fidelity before attribution."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from streaming_full.data import canonical_sha256, load_manifest, sha256_file
from streaming_full.monitoring import (
    load_checkpoint,
    validate_checkpoint_manifest,
    validate_monitoring_result,
)

from .analyze import (
    BATCH_PARTITION_REFERENCE_JOINT_SCORE_TOLERANCE,
    BATCH_PARTITION_REFERENCE_MARGIN_TOLERANCE,
    CROSS_DEVICE_REFERENCE_TOLERANCES,
    EXACT_FORWARD_ABSOLUTE_TOLERANCE,
    SURROGATE_DISTANCE_SQUARED_FLOOR,
    _probe_rows,
    audit_attribution_gradients,
    audit_checkpoint_reconstruction,
    audit_exact_forward_and_batch_partition,
)
from .attribution_scope import ATTRIBUTION_TARGET_SCOPE
from .verify_multiseed_bindings import SEEDS, validate_binding_shape


SCHEMA_VERSION = "ofra_score_fidelity_preflight_v4"
CONFIRMATION = "RUN_SCORE_FIDELITY_PREFLIGHT_CPU"
EXPECTED_CLASS_COUNTS = (2, 4, 6, 8, 10)
EXPECTED_CHECKPOINT_COUNT = len(EXPECTED_CLASS_COUNTS) * len(SEEDS)
EXPECTED_CLASS_CHECKPOINT_COUNT = sum(EXPECTED_CLASS_COUNTS) * len(SEEDS)
EXPECTED_RECONSTRUCTION_ARRAY_TOLERANCES = {
    "class_axis": 0.0,
    **CROSS_DEVICE_REFERENCE_TOLERANCES,
    "predicted_class_id": 0.0,
}
EXPECTED_RECONSTRUCTION_ROW_COUNT = (
    EXPECTED_CHECKPOINT_COUNT * len(EXPECTED_RECONSTRUCTION_ARRAY_TOLERANCES)
)
CROSS_DEVICE_POLICY = (
    "descriptive finite CPU-reload versus archived-GPU audit with exact class "
    "axis and own-argmax checks; no numerical equivalence claim"
)
BATCH_PARTITION_POLICY = (
    "descriptive finite class-batch versus full-probe CPU audit with own-argmax "
    "checks; no numerical equivalence claim"
)
PASS_SCOPE = (
    "source/hash/shape/finite/argmax integrity, CPU same-batch exact-forward "
    "fidelity, and all-target gradient finiteness; no archived-GPU or "
    "batch-partition numerical equivalence claim"
)
def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict) -> None:
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


def _without_self_hash(value: dict) -> dict:
    return {key: item for key, item in value.items() if key != "canonical_sha256"}


def _finite_float(value: object, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"fidelity preflight non-numeric field: {field}") from error
    if not math.isfinite(result):
        raise RuntimeError(f"fidelity preflight non-finite field: {field}")
    return result


def validate_report(path: Path, expected_bindings: dict) -> dict:
    report = _json(path)
    if report.get("schema_version") != SCHEMA_VERSION or report.get("status") != "passed":
        raise RuntimeError("fidelity preflight report is incomplete")
    if report.get("bindings") != expected_bindings:
        raise RuntimeError("fidelity preflight report source bindings changed")
    if report.get("seeds") != list(SEEDS):
        raise RuntimeError("fidelity preflight seed registry changed")
    if report.get("canonical_sha256") != canonical_sha256(_without_self_hash(report)):
        raise RuntimeError("fidelity preflight report self-hash mismatch")
    if report.get("execution_order") != "completed before any attribution call":
        raise RuntimeError("fidelity preflight execution-order contract changed")
    if int(report.get("checkpoint_count", -1)) != EXPECTED_CHECKPOINT_COUNT:
        raise RuntimeError("fidelity preflight checkpoint coverage is incomplete")
    fidelity_rows = report.get("fidelity_rows", [])
    partition_rows = report.get("batch_partition_rows", [])
    reconstruction_rows = report.get("artifact_reconstruction_rows", [])
    gradient_rows = report.get("attribution_gradient_rows", [])
    if not all(
        isinstance(rows, list)
        for rows in (fidelity_rows, partition_rows, reconstruction_rows, gradient_rows)
    ):
        raise RuntimeError("fidelity preflight row registry is invalid")
    if int(report.get("class_checkpoint_count", -1)) != EXPECTED_CLASS_CHECKPOINT_COUNT:
        raise RuntimeError("fidelity preflight class coverage is incomplete")
    if len(fidelity_rows) != EXPECTED_CLASS_CHECKPOINT_COUNT:
        raise RuntimeError("fidelity preflight fidelity-row coverage is incomplete")
    if len(partition_rows) != len(fidelity_rows):
        raise RuntimeError("fidelity preflight partition coverage is incomplete")
    if len(gradient_rows) != len(fidelity_rows):
        raise RuntimeError("fidelity preflight gradient coverage is incomplete")
    if int(report.get("attribution_gradient_target_count", -1)) != len(gradient_rows):
        raise RuntimeError("fidelity preflight gradient target summary is incomplete")

    expected_pairs = {
        (seed, checkpoint)
        for seed in SEEDS
        for checkpoint in range(len(EXPECTED_CLASS_COUNTS))
    }
    if int(report.get("artifact_reconstruction_row_count", -1)) != EXPECTED_RECONSTRUCTION_ROW_COUNT:
        raise RuntimeError("fidelity preflight reconstruction-row count is incomplete")
    if len(reconstruction_rows) != EXPECTED_RECONSTRUCTION_ROW_COUNT:
        raise RuntimeError("fidelity preflight reconstruction rows are incomplete")
    reconstruction_keys = {
        (int(row["seed"]), int(row["checkpoint"]), str(row["array"]))
        for row in reconstruction_rows
    }
    if len(reconstruction_keys) != len(reconstruction_rows):
        raise RuntimeError("fidelity preflight reconstruction rows are duplicated")
    expected_reconstruction_keys = {
        (seed, checkpoint, array)
        for seed, checkpoint in expected_pairs
        for array in EXPECTED_RECONSTRUCTION_ARRAY_TOLERANCES
    }
    if reconstruction_keys != expected_reconstruction_keys:
        raise RuntimeError("fidelity preflight reconstruction array domain is incomplete")
    observed_reconstruction_maxima = {
        array: max(
            _finite_float(row["max_abs_error"], f"reconstruction.{array}.max_abs_error")
            for row in reconstruction_rows
            if row["array"] == array
        )
        for array in EXPECTED_RECONSTRUCTION_ARRAY_TOLERANCES
    }
    observed_reconstruction_p99_maxima = {
        array: max(
            _finite_float(row["p99_abs_error"], f"reconstruction.{array}.p99_abs_error")
            for row in reconstruction_rows
            if row["array"] == array
        )
        for array in EXPECTED_RECONSTRUCTION_ARRAY_TOLERANCES
    }
    observed_reference_exceedance_counts = {
        array: sum(
            int(row["reference_exceedance_count"])
            for row in reconstruction_rows
            if row["array"] == array
        )
        for array in EXPECTED_RECONSTRUCTION_ARRAY_TOLERANCES
    }
    for row in reconstruction_rows:
        array = str(row["array"])
        tolerance = _finite_float(
            row["reference_absolute_tolerance"],
            f"reconstruction.{array}.reference_tolerance",
        )
        expected_tolerance = EXPECTED_RECONSTRUCTION_ARRAY_TOLERANCES[array]
        if tolerance != expected_tolerance:
            raise RuntimeError("fidelity preflight reconstruction reference tolerance changed")
        maximum_error = _finite_float(
            row["max_abs_error"], f"reconstruction.{array}.max_abs_error"
        )
        p99_error = _finite_float(
            row["p99_abs_error"], f"reconstruction.{array}.p99_abs_error"
        )
        if p99_error > maximum_error:
            raise RuntimeError("fidelity preflight reconstruction p99 exceeds maximum")
        exceedance_count = int(row["reference_exceedance_count"])
        if exceedance_count < 0:
            raise RuntimeError("fidelity preflight reconstruction exceedance count is invalid")
        within_reference = row.get("within_reference_tolerance")
        if not isinstance(within_reference, bool):
            raise RuntimeError("fidelity preflight reconstruction reference flag is invalid")
        if within_reference != (maximum_error <= tolerance):
            raise RuntimeError("fidelity preflight reconstruction reference flag mismatch")
        if within_reference != (exceedance_count == 0):
            raise RuntimeError("fidelity preflight reconstruction exceedance mismatch")
        if not isinstance(row.get("exact_match"), bool):
            raise RuntimeError("fidelity preflight reconstruction exact-match flag is invalid")
        if row["exact_match"] != (maximum_error == 0.0):
            raise RuntimeError("fidelity preflight reconstruction exact-match flag mismatch")
        if array == "class_axis" and not row["exact_match"]:
            raise RuntimeError("fidelity preflight class axis is not exact")
        if array == "predicted_class_id":
            decision_count = int(row["decision_count"])
            mismatch_count = int(row["mismatch_count"])
            if decision_count <= 0 or not (0 <= mismatch_count <= decision_count):
                raise RuntimeError("fidelity preflight prediction counts are invalid")
            if row.get("equivalence_claimed") is not False:
                raise RuntimeError("fidelity preflight makes a cross-device equivalence claim")
            if exceedance_count != mismatch_count:
                raise RuntimeError("fidelity preflight prediction exceedance mismatch")
    if report.get("artifact_reconstruction_max_abs_error_by_array") != observed_reconstruction_maxima:
        raise RuntimeError("fidelity preflight reconstruction summary mismatch")
    if report.get("artifact_reconstruction_p99_abs_error_max_by_array") != observed_reconstruction_p99_maxima:
        raise RuntimeError("fidelity preflight reconstruction p99 summary mismatch")
    if report.get("artifact_reconstruction_reference_exceedance_count_by_array") != observed_reference_exceedance_counts:
        raise RuntimeError("fidelity preflight reconstruction exceedance summary mismatch")
    prediction_rows = [
        row for row in reconstruction_rows if row["array"] == "predicted_class_id"
    ]
    observed_decision_count = sum(int(row["decision_count"]) for row in prediction_rows)
    observed_mismatch_count = sum(int(row["mismatch_count"]) for row in prediction_rows)
    if int(report.get("cross_device_decision_count", -1)) != observed_decision_count:
        raise RuntimeError("fidelity preflight decision-count summary mismatch")
    if int(report.get("cross_device_prediction_mismatch_count", -1)) != observed_mismatch_count:
        raise RuntimeError("fidelity preflight mismatch-count summary mismatch")
    observed_mismatch_rate = observed_mismatch_count / observed_decision_count
    if _finite_float(
        report.get("cross_device_prediction_mismatch_rate"),
        "cross_device_prediction_mismatch_rate",
    ) != observed_mismatch_rate:
        raise RuntimeError("fidelity preflight mismatch-rate summary mismatch")
    if report.get("cross_device_equivalence_claimed") is not False:
        raise RuntimeError("fidelity preflight cross-device equivalence claim changed")
    if report.get("cross_device_reference_absolute_tolerances") != CROSS_DEVICE_REFERENCE_TOLERANCES:
        raise RuntimeError("fidelity preflight cross-device reference registry changed")
    if report.get("cross_device_policy") != CROSS_DEVICE_POLICY:
        raise RuntimeError("fidelity preflight cross-device policy changed")
    if report.get("pass_scope") != PASS_SCOPE:
        raise RuntimeError("fidelity preflight pass scope changed")
    if report.get("attribution_target_scope") != ATTRIBUTION_TARGET_SCOPE:
        raise RuntimeError("fidelity preflight attribution target scope changed")
    observed_reconstruction_pairs = {
        (seed, checkpoint) for seed, checkpoint, _ in reconstruction_keys
    }
    if observed_reconstruction_pairs != expected_pairs:
        raise RuntimeError("fidelity preflight reconstruction coverage is incomplete")
    grouped_classes: dict[tuple[int, int], set[int]] = {}
    for row in fidelity_rows:
        key = (int(row["seed"]), int(row["checkpoint"]))
        grouped_classes.setdefault(key, set()).add(int(row["class_id"]))
    if set(grouped_classes) != expected_pairs:
        raise RuntimeError("fidelity preflight checkpoint/class domain is incomplete")
    for key, classes in grouped_classes.items():
        if len(classes) != EXPECTED_CLASS_COUNTS[key[1]]:
            raise RuntimeError("fidelity preflight per-checkpoint class coverage is incomplete")
    fidelity_keys = {
        (int(row["seed"]), int(row["checkpoint"]), int(row["class_id"]))
        for row in fidelity_rows
    }
    partition_keys = {
        (int(row["seed"]), int(row["checkpoint"]), int(row["class_id"]))
        for row in partition_rows
    }
    gradient_keys = {
        (int(row["seed"]), int(row["checkpoint"]), int(row["class_id"]))
        for row in gradient_rows
    }
    if (
        len(fidelity_keys) != len(fidelity_rows)
        or partition_keys != fidelity_keys
        or gradient_keys != fidelity_keys
        or len(gradient_keys) != len(gradient_rows)
    ):
        raise RuntimeError("fidelity preflight class rows are duplicated or misaligned")

    if _finite_float(
        report.get("surrogate_distance_squared_floor"),
        "gradient_surrogate.distance_squared_floor",
    ) != SURROGATE_DISTANCE_SQUARED_FLOOR:
        raise RuntimeError("fidelity preflight gradient-surrogate floor changed")
    if int(report.get("attribution_gradient_nonfinite_count", -1)) != 0:
        raise RuntimeError("fidelity preflight reports non-finite attribution gradients")
    for row in gradient_rows:
        if int(row.get("probe_rows", 0)) <= 0 or int(row.get("feature_dim", 0)) <= 0:
            raise RuntimeError("fidelity preflight gradient row has invalid dimensions")
        if int(row.get("gradient_nonfinite_count", -1)) != 0:
            raise RuntimeError("fidelity preflight gradient row is non-finite")
        if row.get("exact_forward_value_changed") is not False:
            raise RuntimeError("fidelity preflight gradient row changes exact forward values")
        if _finite_float(
            row.get("surrogate_distance_squared_floor"),
            "gradient_row.distance_squared_floor",
        ) != SURROGATE_DISTANCE_SQUARED_FLOOR:
            raise RuntimeError("fidelity preflight gradient-row floor changed")
        _finite_float(row.get("output_max_abs"), "gradient_row.output_max_abs")
        _finite_float(row.get("gradient_max_abs"), "gradient_row.gradient_max_abs")

    observed_same_batch_max = max(
        max(
            _finite_float(row["joint_score_max_abs_error"], "fidelity.joint_score"),
            _finite_float(row["max_abs_error"], "fidelity.margin"),
        )
        for row in fidelity_rows
    )
    observed_partition_joint_max = max(
        _finite_float(
            row["class_vs_full_batch_joint_score_max_abs_difference"],
            "batch_partition.joint_score",
        )
        for row in partition_rows
    )
    observed_partition_max = max(
        _finite_float(
            row["class_vs_full_batch_margin_max_abs_difference"],
            "batch_partition.margin",
        )
        for row in partition_rows
    )
    if _finite_float(report.get("same_batch_absolute_tolerance"), "same_batch_tolerance") != EXACT_FORWARD_ABSOLUTE_TOLERANCE:
        raise RuntimeError("fidelity preflight exact-forward tolerance changed")
    if _finite_float(report.get("batch_partition_joint_score_reference_tolerance"), "batch_partition_joint_reference") != BATCH_PARTITION_REFERENCE_JOINT_SCORE_TOLERANCE:
        raise RuntimeError("fidelity preflight partition joint-score reference changed")
    if _finite_float(report.get("batch_partition_margin_reference_tolerance"), "batch_partition_margin_reference") != BATCH_PARTITION_REFERENCE_MARGIN_TOLERANCE:
        raise RuntimeError("fidelity preflight partition margin reference changed")
    if _finite_float(report.get("same_batch_max_abs_error"), "same_batch_max") != observed_same_batch_max:
        raise RuntimeError("fidelity preflight exact-forward summary mismatch")
    if _finite_float(report.get("batch_partition_joint_score_max_abs_difference"), "batch_partition_joint_max") != observed_partition_joint_max:
        raise RuntimeError("fidelity preflight batch-partition joint-score summary mismatch")
    if _finite_float(report.get("batch_partition_margin_max_abs_difference"), "batch_partition_margin_max") != observed_partition_max:
        raise RuntimeError("fidelity preflight batch-partition summary mismatch")
    if observed_same_batch_max > EXACT_FORWARD_ABSOLUTE_TOLERANCE:
        raise RuntimeError("fidelity preflight exact-forward bound failed")
    observed_partition_joint_exceedance_count = 0
    observed_partition_margin_exceedance_count = 0
    observed_partition_decision_count = 0
    observed_partition_mismatch_count = 0
    for row in partition_rows:
        joint_error = _finite_float(
            row["class_vs_full_batch_joint_score_max_abs_difference"],
            "batch_partition.joint_score",
        )
        margin_error = _finite_float(
            row["class_vs_full_batch_margin_max_abs_difference"],
            "batch_partition.margin",
        )
        joint_count = int(row["joint_score_reference_exceedance_count"])
        margin_count = int(row["margin_reference_exceedance_count"])
        if joint_count < 0 or margin_count < 0:
            raise RuntimeError("fidelity preflight partition exceedance count is invalid")
        if _finite_float(
            row["joint_score_reference_absolute_tolerance"],
            "batch_partition.row_joint_reference",
        ) != BATCH_PARTITION_REFERENCE_JOINT_SCORE_TOLERANCE:
            raise RuntimeError("fidelity preflight partition row joint reference changed")
        if _finite_float(
            row["margin_reference_absolute_tolerance"],
            "batch_partition.row_margin_reference",
        ) != BATCH_PARTITION_REFERENCE_MARGIN_TOLERANCE:
            raise RuntimeError("fidelity preflight partition row margin reference changed")
        if row.get("joint_score_within_reference_tolerance") != (
            joint_error <= BATCH_PARTITION_REFERENCE_JOINT_SCORE_TOLERANCE
        ):
            raise RuntimeError("fidelity preflight partition joint reference flag mismatch")
        if row.get("margin_within_reference_tolerance") != (
            margin_error <= BATCH_PARTITION_REFERENCE_MARGIN_TOLERANCE
        ):
            raise RuntimeError("fidelity preflight partition margin reference flag mismatch")
        if row["joint_score_within_reference_tolerance"] != (joint_count == 0):
            raise RuntimeError("fidelity preflight partition joint exceedance mismatch")
        if row["margin_within_reference_tolerance"] != (margin_count == 0):
            raise RuntimeError("fidelity preflight partition margin exceedance mismatch")
        decision_count = int(row["decision_count"])
        mismatch_count = int(row["mismatch_count"])
        if decision_count <= 0 or not (0 <= mismatch_count <= decision_count):
            raise RuntimeError("fidelity preflight partition decision counts are invalid")
        if row.get("equivalence_claimed") is not False:
            raise RuntimeError("fidelity preflight makes a batch-partition equivalence claim")
        if row.get("exact_match") != (mismatch_count == 0):
            raise RuntimeError("fidelity preflight partition exact-match flag mismatch")
        observed_partition_joint_exceedance_count += joint_count
        observed_partition_margin_exceedance_count += margin_count
        observed_partition_decision_count += decision_count
        observed_partition_mismatch_count += mismatch_count
    partition_summaries = {
        "batch_partition_joint_score_reference_exceedance_count": observed_partition_joint_exceedance_count,
        "batch_partition_margin_reference_exceedance_count": observed_partition_margin_exceedance_count,
        "batch_partition_decision_count": observed_partition_decision_count,
        "batch_partition_prediction_mismatch_count": observed_partition_mismatch_count,
    }
    for key, expected in partition_summaries.items():
        if int(report.get(key, -1)) != expected:
            raise RuntimeError(f"fidelity preflight partition summary mismatch: {key}")
    observed_partition_mismatch_rate = (
        observed_partition_mismatch_count / observed_partition_decision_count
    )
    if _finite_float(
        report.get("batch_partition_prediction_mismatch_rate"),
        "batch_partition_prediction_mismatch_rate",
    ) != observed_partition_mismatch_rate:
        raise RuntimeError("fidelity preflight partition mismatch-rate summary mismatch")
    if report.get("batch_partition_policy") != BATCH_PARTITION_POLICY:
        raise RuntimeError("fidelity preflight partition policy changed")
    if report.get("batch_partition_equivalence_claimed") is not False:
        raise RuntimeError("fidelity preflight batch-partition equivalence claim changed")
    return report


def run(args: argparse.Namespace) -> Path:
    if args.confirm != CONFIRMATION:
        raise RuntimeError(f"refusing fidelity preflight without --confirm {CONFIRMATION}")
    if args.cpu_threads <= 0:
        raise ValueError("cpu threads must be positive")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized in the CPU-only fidelity preflight")

    binding_path = args.bindings.resolve(strict=True)
    method_protocol = args.method_protocol.resolve(strict=True)
    output = args.output.resolve()
    bindings = _json(binding_path)
    validate_binding_shape(bindings)
    source_bindings = {
        "bindings_file_sha256": sha256_file(binding_path),
        "method_protocol_file_sha256": sha256_file(method_protocol),
        "analyzer_file_sha256": sha256_file(Path(__file__).with_name("analyze.py")),
        "preflight_file_sha256": sha256_file(Path(__file__).resolve()),
    }
    if output.exists():
        if not args.reuse_complete_output:
            raise FileExistsError(f"fidelity preflight output exists: {output}")
        validate_report(output, source_bindings)
        print(json.dumps({"event": "score_fidelity_preflight_reused", "path": str(output)}, sort_keys=True))
        return output

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    manifest_record = bindings["shared_files"]["streaming_manifest"]
    manifest_path = Path(str(manifest_record["path"])).resolve(strict=True)
    if sha256_file(manifest_path) != manifest_record["sha256"]:
        raise RuntimeError("fidelity preflight streaming manifest binding mismatch")
    # The immediately preceding multiseed verifier checks every cache shard.
    # Avoid repeating that expensive full-cache hash pass here.
    manifest = load_manifest(manifest_path, verify_hashes=False)
    if manifest.dataset != bindings.get("dataset"):
        raise RuntimeError("fidelity preflight dataset mismatch")

    reconstruction_rows: list[dict] = []
    fidelity_rows: list[dict] = []
    partition_rows: list[dict] = []
    gradient_rows: list[dict] = []
    checkpoint_count = 0
    for seed_record in bindings["seeds"]:
        seed = int(seed_record["seed"])
        result_root = Path(str(seed_record["result_root"])).resolve(strict=True)
        result = _json(result_root / f"result_seed_{seed}.json")
        protocol = _json(result_root / "protocol.json")
        validate_monitoring_result(
            result,
            output_base=result_root,
            expected_protocol=protocol["monitoring"],
        )
        probe_record = seed_record["files"]["probe_manifest"]
        probe_path = Path(str(probe_record["path"])).resolve(strict=True)
        if sha256_file(probe_path) != probe_record["sha256"]:
            raise RuntimeError(f"seed {seed}: probe manifest binding mismatch")
        probe = _json(probe_path)
        official = _probe_rows(
            manifest,
            list(probe["official_test"]["samples"]),
            split="official_test",
        )

        for checkpoint_record in seed_record["checkpoint_artifacts"]:
            checkpoint_index = int(checkpoint_record["checkpoint"])
            checkpoint_path = Path(str(checkpoint_record["manifest_path"])).resolve(strict=True)
            metadata = validate_checkpoint_manifest(checkpoint_path)
            checkpoint = load_checkpoint(checkpoint_path, device="cpu")
            seen = {int(value) for value in metadata["seen_classes"]}
            mask = np.asarray(
                [int(record["class_id"]) in seen for record in official.records],
                dtype=bool,
            )
            raw_seen = official.values[mask]
            record_seen = [
                record for keep, record in zip(mask.tolist(), official.records) if keep
            ]
            reconstructed = checkpoint.score(raw_seen)
            score_path = checkpoint_path.parent / str(metadata["probe_scores_file"])
            with np.load(score_path, allow_pickle=False) as archive:
                saved = {name: np.array(archive[name], copy=True) for name in archive.files}
            reconstructed_audit = audit_checkpoint_reconstruction(
                checkpoint_index, saved, reconstructed
            )
            current_fidelity, current_partition = audit_exact_forward_and_batch_partition(
                checkpoint_index,
                checkpoint,
                raw_seen,
                record_seen,
                reconstructed,
            )
            current_gradients = audit_attribution_gradients(
                checkpoint_index,
                checkpoint,
                raw_seen,
                record_seen,
            )
            reconstruction_rows.extend(
                [{"seed": seed, **row} for row in reconstructed_audit]
            )
            fidelity_rows.extend([{"seed": seed, **row} for row in current_fidelity])
            partition_rows.extend([{"seed": seed, **row} for row in current_partition])
            gradient_rows.extend([{"seed": seed, **row} for row in current_gradients])
            checkpoint_count += 1
            print(
                json.dumps(
                    {
                        "event": "score_fidelity_preflight_checkpoint_complete",
                        "seed": seed,
                        "checkpoint": checkpoint_index,
                        "class_count": len(current_fidelity),
                        "gradient_targets_checked": len(current_gradients),
                        "same_batch_max_abs_error": max(
                            max(row["joint_score_max_abs_error"], row["max_abs_error"])
                            for row in current_fidelity
                        ),
                        "batch_partition_joint_score_max_abs_difference": max(
                            row["class_vs_full_batch_joint_score_max_abs_difference"]
                            for row in current_partition
                        ),
                        "batch_partition_margin_max_abs_difference": max(
                            row["class_vs_full_batch_margin_max_abs_difference"]
                            for row in current_partition
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del checkpoint, reconstructed, raw_seen, record_seen, saved
            gc.collect()

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "dataset": bindings["dataset"],
        "seeds": list(SEEDS),
        "checkpoint_count": checkpoint_count,
        "class_checkpoint_count": len(fidelity_rows),
        "attribution_gradient_target_count": len(gradient_rows),
        "attribution_gradient_nonfinite_count": sum(
            int(row["gradient_nonfinite_count"]) for row in gradient_rows
        ),
        "surrogate_distance_squared_floor": SURROGATE_DISTANCE_SQUARED_FLOOR,
        "artifact_reconstruction_row_count": len(reconstruction_rows),
        "artifact_reconstruction_max_abs_error_by_array": {
            array: max(
                row["max_abs_error"]
                for row in reconstruction_rows
                if row["array"] == array
            )
            for array in EXPECTED_RECONSTRUCTION_ARRAY_TOLERANCES
        },
        "artifact_reconstruction_p99_abs_error_max_by_array": {
            array: max(
                row["p99_abs_error"]
                for row in reconstruction_rows
                if row["array"] == array
            )
            for array in EXPECTED_RECONSTRUCTION_ARRAY_TOLERANCES
        },
        "artifact_reconstruction_reference_exceedance_count_by_array": {
            array: sum(
                int(row["reference_exceedance_count"])
                for row in reconstruction_rows
                if row["array"] == array
            )
            for array in EXPECTED_RECONSTRUCTION_ARRAY_TOLERANCES
        },
        "cross_device_reference_absolute_tolerances": CROSS_DEVICE_REFERENCE_TOLERANCES,
        "cross_device_policy": CROSS_DEVICE_POLICY,
        "cross_device_equivalence_claimed": False,
        "pass_scope": PASS_SCOPE,
        "attribution_target_scope": ATTRIBUTION_TARGET_SCOPE,
        "cross_device_decision_count": sum(
            int(row["decision_count"])
            for row in reconstruction_rows
            if row["array"] == "predicted_class_id"
        ),
        "cross_device_prediction_mismatch_count": sum(
            int(row["mismatch_count"])
            for row in reconstruction_rows
            if row["array"] == "predicted_class_id"
        ),
        "cross_device_prediction_mismatch_rate": (
            sum(
                int(row["mismatch_count"])
                for row in reconstruction_rows
                if row["array"] == "predicted_class_id"
            )
            / sum(
                int(row["decision_count"])
                for row in reconstruction_rows
                if row["array"] == "predicted_class_id"
            )
        ),
        "same_batch_absolute_tolerance": EXACT_FORWARD_ABSOLUTE_TOLERANCE,
        "same_batch_max_abs_error": max(
            max(row["joint_score_max_abs_error"], row["max_abs_error"])
            for row in fidelity_rows
        ),
        "batch_partition_joint_score_reference_tolerance": BATCH_PARTITION_REFERENCE_JOINT_SCORE_TOLERANCE,
        "batch_partition_joint_score_max_abs_difference": max(
            row["class_vs_full_batch_joint_score_max_abs_difference"]
            for row in partition_rows
        ),
        "batch_partition_margin_reference_tolerance": BATCH_PARTITION_REFERENCE_MARGIN_TOLERANCE,
        "batch_partition_margin_max_abs_difference": max(
            row["class_vs_full_batch_margin_max_abs_difference"] for row in partition_rows
        ),
        "batch_partition_joint_score_reference_exceedance_count": sum(
            row["joint_score_reference_exceedance_count"] for row in partition_rows
        ),
        "batch_partition_margin_reference_exceedance_count": sum(
            row["margin_reference_exceedance_count"] for row in partition_rows
        ),
        "batch_partition_decision_count": sum(
            row["decision_count"] for row in partition_rows
        ),
        "batch_partition_prediction_mismatch_count": sum(
            row["mismatch_count"] for row in partition_rows
        ),
        "batch_partition_prediction_mismatch_rate": (
            sum(row["mismatch_count"] for row in partition_rows)
            / sum(row["decision_count"] for row in partition_rows)
        ),
        "batch_partition_policy": BATCH_PARTITION_POLICY,
        "batch_partition_equivalence_claimed": False,
        "execution_order": "completed before any attribution call",
        "bindings": source_bindings,
        "artifact_reconstruction_rows": reconstruction_rows,
        "fidelity_rows": fidelity_rows,
        "batch_partition_rows": partition_rows,
        "attribution_gradient_rows": gradient_rows,
    }
    report["canonical_sha256"] = canonical_sha256(report)
    _atomic_json(output, report)
    validate_report(output, source_bindings)
    print(
        json.dumps(
            {
                "event": "all_seed_score_fidelity_preflight_complete",
                "path": str(output),
                "canonical_sha256": report["canonical_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return output


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--bindings", type=Path, required=True)
    value.add_argument("--method-protocol", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--cpu-threads", type=int, required=True)
    value.add_argument("--reuse-complete-output", action="store_true")
    value.add_argument("--confirm", required=True)
    return value


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
