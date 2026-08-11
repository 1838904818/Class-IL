"""Hash-bound formal-v2 explanation-drift and ETG analysis.

This module is deliberately offline.  It consumes the inference-only monitor
artifacts emitted by ``streaming_full.monitoring`` and never mutates a training
run or writes to W&B.  Every input is validated before attribution begins and
every emitted file is bound into a self-hashed manifest.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import math
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from streaming_full.data import (
    DatasetManifest,
    array_sha256,
    canonical_sha256,
    load_manifest,
    sha256_file,
)
from streaming_full.monitoring import (
    load_checkpoint,
    validate_checkpoint_manifest,
    validate_monitoring_result,
)


SCHEMA_VERSION = "formal_explanation_etg_v2.0"
CONFIRMATION = "RUN_FORMAL_V2_CPU"
K_VALUES = (5, 10, 15, 20)
JACCARD_THRESHOLDS = (0.50, 0.60, 0.70, 0.80)
ALLOWED_RECALL_DROPS = (0.00, 0.02, 0.05, 0.10)
PRIMARY_K = 15
PRIMARY_JACCARD_THRESHOLD = 0.70
PRIMARY_ALLOWED_RECALL_DROP = 0.05
RANDOM_CONTROL_COUNT = 50
RANDOM_CONTROL_QUANTILE = 0.95
CROSS_DEVICE_TOLERANCES = {
    "head_scores": 2e-6,
    "router_z_scores": 2e-3,
    "joint_scores": 1e-3,
}
PREDICTION_TIE_TOLERANCE_FACTOR = 2.0


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def _without_timing(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_timing(item)
            for key, item in value.items()
            if key not in {"timing", "deterministic_result_sha256"}
        }
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return value


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


def _npy_bytes(value: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, np.ascontiguousarray(value), allow_pickle=False)
    return output.getvalue()


def _deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as archive:
            for name in sorted(arrays):
                if not name or name.endswith(".npy") or "/" in name or "\\" in name:
                    raise ValueError(f"invalid NPZ key: {name!r}")
                info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, _npy_bytes(np.asarray(arrays[name])))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise RuntimeError(f"invalid resumable NPZ {path}: {error}") from error


def _persist_resume_state(
    staging: Path,
    *,
    attribution_arrays: dict[str, np.ndarray],
    control_arrays: dict[str, np.ndarray],
    checkpoint_rows: list[dict],
    progress: dict,
) -> None:
    """Commit one completed class atomically, with progress written last."""

    _deterministic_npz(staging / "resume_attributions.npz", attribution_arrays)
    _deterministic_npz(staging / "resume_etg_controls_and_masses.npz", control_arrays)
    _atomic_json(staging / "resume_checkpoint_rows.json", checkpoint_rows)
    _atomic_json(staging / "progress.json", progress)


def _resume_array_pair(name: str) -> tuple[int, int] | None:
    parts = name.split("_")
    if (
        len(parts) >= 5
        and parts[0] == "checkpoint"
        and parts[2] == "class"
        and parts[1].isdigit()
        and parts[3].isdigit()
    ):
        return int(parts[1]), int(parts[3])
    return None


def _reconcile_resume_state(
    *,
    progress: dict,
    attribution_arrays: dict[str, np.ndarray],
    control_arrays: dict[str, np.ndarray],
    checkpoint_rows: list[dict],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict]]:
    """Recover the last progress-marked commit after a cross-file interruption.

    ``_persist_resume_state`` writes progress last.  The other files may
    therefore contain a strict superset of the last committed state if the
    process stops between replacements.  Only pairs recorded by progress are
    committed; orphaned arrays and rows are deterministically discarded.
    """

    raw_completed = progress.get("completed_class_checkpoint_rows", [])
    if not isinstance(raw_completed, list) or not all(
        isinstance(row, dict) for row in raw_completed
    ):
        raise RuntimeError("resumable completed-pair registry is invalid")
    completed = {
        (int(row["checkpoint"]), int(row["class_id"])) for row in raw_completed
    }
    if len(completed) != len(raw_completed):
        raise RuntimeError("resumable completed-pair registry has duplicates")

    committed_rows = [
        row
        for row in checkpoint_rows
        if (int(row["checkpoint"]), int(row["class_id"])) in completed
    ]
    committed_row_pairs = {
        (int(row["checkpoint"]), int(row["class_id"])) for row in committed_rows
    }
    if len(committed_rows) != len(committed_row_pairs):
        raise RuntimeError("resumable checkpoint rows contain duplicates")
    if committed_row_pairs != completed:
        raise RuntimeError("resumable committed rows are incomplete")

    committed_attributions = {
        name: value
        for name, value in attribution_arrays.items()
        if _resume_array_pair(name) in completed
    }
    completed_classes = {class_id for _, class_id in completed}
    committed_controls: dict[str, np.ndarray] = {}
    for name, value in control_arrays.items():
        pair = _resume_array_pair(name)
        if pair is not None:
            if pair in completed:
                committed_controls[name] = value
            continue
        parts = name.split("_")
        if (
            len(parts) == 4
            and parts[0] == "class"
            and parts[1].isdigit()
            and parts[2:] == ["feature", "sets"]
            and int(parts[1]) in completed_classes
        ):
            committed_controls[name] = value

    for checkpoint, class_id in completed:
        prefix = f"checkpoint_{checkpoint:03d}_class_{class_id:03d}"
        required_attributions = {
            f"{prefix}_mean_abs",
            f"{prefix}_mean_signed",
            f"{prefix}_sample_id_sha256",
        }
        missing = required_attributions - committed_attributions.keys()
        if missing:
            raise RuntimeError(
                f"resumable committed attribution state is incomplete: {sorted(missing)}"
            )
        if f"{prefix}_random_masses" not in committed_controls:
            raise RuntimeError("resumable committed ETG mass state is incomplete")
        if f"class_{class_id:03d}_feature_sets" not in committed_controls:
            raise RuntimeError("resumable committed ETG control state is incomplete")

    return committed_attributions, committed_controls, committed_rows


def _csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return format(value, ".17g")
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _derived_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _topk(vector: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or k <= 0 or k > len(values):
        raise ValueError("invalid top-k input")
    return np.argsort(-values, kind="stable")[:k].astype(np.int64)


def jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    a, b = set(int(value) for value in left), set(int(value) for value in right)
    union = a | b
    if not union:
        raise ValueError("Jaccard is undefined for two empty sets")
    return len(a & b) / len(union)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def _kendall(left: np.ndarray, right: np.ndarray) -> float:
    from scipy.stats import kendalltau

    value = float(kendalltau(left, right, variant="b").statistic)
    return value if math.isfinite(value) else 0.0


def sensitivity_rows(transitions: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for k in K_VALUES:
        key = f"jaccard_top{k}"
        for threshold in JACCARD_THRESHOLDS:
            for allowed in ALLOWED_RECALL_DROPS:
                eligible = [row for row in transitions if row["delta_recall"] > -allowed]
                events = [row for row in eligible if row[key] < threshold]
                rows.append(
                    {
                        "k": k,
                        "jaccard_threshold": threshold,
                        "allowed_recall_drop": allowed,
                        "events": len(events),
                        "eligible_transitions": len(eligible),
                        "rate": len(events) / len(eligible) if eligible else None,
                        "is_primary": (
                            k == PRIMARY_K
                            and threshold == PRIMARY_JACCARD_THRESHOLD
                            and allowed == PRIMARY_ALLOWED_RECALL_DROP
                        ),
                    }
                )
    return rows


@dataclass
class FrozenRows:
    values: np.ndarray
    records: list[dict]


def _probe_rows(
    manifest: DatasetManifest, records: list[dict], *, split: str
) -> FrozenRows:
    if split not in {"official_test", "task0_train_background"}:
        raise ValueError(f"unsupported probe split: {split}")
    opened: dict[Path, np.ndarray] = {}
    rows: list[np.ndarray] = []
    try:
        for record in records:
            class_id = int(record["class_id"])
            class_record = manifest.class_map[class_id]
            shards = class_record.test if split == "official_test" else class_record.train
            ordinal = int(record["shard_ordinal"])
            if ordinal < 0 or ordinal >= len(shards):
                raise RuntimeError("probe shard ordinal is out of range")
            shard = shards[ordinal]
            if record.get("parent_shard_sha256") != shard.sha256:
                raise RuntimeError("probe parent shard SHA-256 mismatch")
            identity = {
                "dataset": manifest.dataset,
                "streaming_manifest_sha256": manifest.manifest_sha256,
                "split": split,
                "class_id": class_id,
                "shard_ordinal": ordinal,
                "local_row": int(record["local_row"]),
                "parent_shard_sha256": shard.sha256,
            }
            if canonical_sha256(identity) != record.get("sample_id_sha256"):
                raise RuntimeError("probe sample identity SHA-256 mismatch")
            if shard.path not in opened:
                opened[shard.path] = np.load(shard.path, mmap_mode="r", allow_pickle=False)
            array = opened[shard.path]
            local_row = int(record["local_row"])
            if local_row < 0 or local_row >= len(array):
                raise RuntimeError("probe local row is out of range")
            row = np.asarray(array[local_row], dtype=np.float32)[None, :].copy()
            if array_sha256(row) != record.get("feature_row_sha256"):
                raise RuntimeError("probe feature-row SHA-256 mismatch")
            rows.append(row)
    finally:
        for array in opened.values():
            mapping = getattr(array, "_mmap", None)
            if mapping is not None:
                mapping.close()
    if not rows:
        raise RuntimeError(f"no rows in frozen split {split}")
    return FrozenRows(np.vstack(rows).astype(np.float32, copy=False), records)


class JointMarginModel(torch.nn.Module):
    """Differentiable representation of official joint_cap3000 margin g_c."""

    def __init__(self, checkpoint, target_class: int):
        super().__init__()
        self.encoder = checkpoint.encoder
        self.heads = torch.nn.ModuleDict(
            {str(class_id): head for class_id, head in checkpoint.heads.items()}
        )
        self.classes = [int(value) for value in checkpoint.metadata["seen_classes"]]
        if target_class not in self.classes or len(self.classes) < 2:
            raise ValueError("target class must be on a multi-class checkpoint axis")
        self.target_class = int(target_class)
        self.register_buffer("normalization_mean", torch.tensor(checkpoint.mean, dtype=torch.float32))
        self.register_buffer("normalization_scale", torch.tensor(checkpoint.scale, dtype=torch.float32))
        for class_id in self.classes:
            self.register_buffer(
                f"centroids_{class_id}",
                torch.tensor(checkpoint.router.cap[class_id].centroids, dtype=torch.float32),
            )
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        normalized = (raw - self.normalization_mean) / self.normalization_scale
        embedding = self.encoder(normalized)
        head = torch.stack(
            [self.heads[str(class_id)].positive_probability(embedding) for class_id in self.classes],
            dim=1,
        )
        router_raw = torch.stack(
            [
                -torch.cdist(embedding, getattr(self, f"centroids_{class_id}"), p=2).min(dim=1).values
                for class_id in self.classes
            ],
            dim=1,
        )
        router_z = (router_raw - router_raw.mean(dim=1, keepdim=True)) / (
            router_raw.std(dim=1, keepdim=True, unbiased=False) + 1e-8
        )
        joint = head + 0.5 * router_z
        target = self.classes.index(self.target_class)
        competitors = torch.cat((joint[:, :target], joint[:, target + 1 :]), dim=1)
        return (joint[:, target] - competitors.max(dim=1).values).unsqueeze(1)


def prediction_tie_audit(
    saved: dict[str, np.ndarray],
    reconstructed: dict[str, np.ndarray],
    *,
    joint_tolerance: float,
) -> dict:
    """Audit cross-device argmax changes without hiding substantive mismatches.

    A class change is accepted only when the saved winner and reconstructed
    winner are mutually within twice the already-enforced per-score absolute
    tolerance.  The factor of two is the exact worst-case separation when two
    score columns each move by ``joint_tolerance`` in opposite directions.
    """

    axis = np.asarray(saved["class_axis"], dtype=np.int64)
    if not np.array_equal(axis, reconstructed["class_axis"]):
        raise RuntimeError("class axis differs during prediction tie audit")
    saved_prediction = np.asarray(saved["predicted_class_id"], dtype=np.int64)
    reconstructed_prediction = np.asarray(
        reconstructed["predicted_class_id"], dtype=np.int64
    )
    if saved_prediction.shape != reconstructed_prediction.shape:
        raise RuntimeError("prediction shape differs during tie audit")
    mismatch_rows = np.flatnonzero(saved_prediction != reconstructed_prediction)
    if not len(mismatch_rows):
        return {
            "exact_match": True,
            "mismatch_count": 0,
            "tie_compatible_mismatch_count": 0,
            "tie_gap_limit": PREDICTION_TIE_TOLERANCE_FACTOR * joint_tolerance,
            "maximum_saved_winner_gap": 0.0,
            "maximum_reconstructed_winner_gap": 0.0,
        }

    column_for_class = {int(class_id): column for column, class_id in enumerate(axis)}
    try:
        saved_columns = np.asarray(
            [column_for_class[int(saved_prediction[row])] for row in mismatch_rows]
        )
        reconstructed_columns = np.asarray(
            [column_for_class[int(reconstructed_prediction[row])] for row in mismatch_rows]
        )
    except KeyError as error:
        raise RuntimeError("prediction outside the frozen class axis") from error

    saved_joint = np.asarray(saved["joint_scores"], dtype=np.float64)
    reconstructed_joint = np.asarray(reconstructed["joint_scores"], dtype=np.float64)
    saved_gap = (
        saved_joint[mismatch_rows, saved_columns]
        - saved_joint[mismatch_rows, reconstructed_columns]
    )
    reconstructed_gap = (
        reconstructed_joint[mismatch_rows, reconstructed_columns]
        - reconstructed_joint[mismatch_rows, saved_columns]
    )
    rounding_slack = 8.0 * np.finfo(np.float32).eps
    gap_limit = PREDICTION_TIE_TOLERANCE_FACTOR * joint_tolerance + rounding_slack
    compatible = (
        (saved_gap >= -rounding_slack)
        & (reconstructed_gap >= -rounding_slack)
        & (saved_gap <= gap_limit)
        & (reconstructed_gap <= gap_limit)
    )
    if not bool(np.all(compatible)):
        bad = mismatch_rows[~compatible]
        raise RuntimeError(
            "prediction mismatch is not explained by a cross-device near tie; "
            f"rows={bad[:10].tolist()} count={len(bad)}"
        )
    return {
        "exact_match": False,
        "mismatch_count": int(len(mismatch_rows)),
        "tie_compatible_mismatch_count": int(np.count_nonzero(compatible)),
        "tie_gap_limit": float(gap_limit),
        "maximum_saved_winner_gap": float(np.max(saved_gap)),
        "maximum_reconstructed_winner_gap": float(np.max(reconstructed_gap)),
    }


def audit_checkpoint_reconstruction(
    checkpoint_index: int,
    saved: dict[str, np.ndarray],
    reconstructed: dict[str, np.ndarray],
) -> list[dict]:
    rows: list[dict] = []
    if not np.array_equal(saved["class_axis"], reconstructed["class_axis"]):
        raise RuntimeError(
            f"checkpoint {checkpoint_index} score reconstruction mismatch: class_axis"
        )
    rows.append(
        {
            "checkpoint": checkpoint_index,
            "array": "class_axis",
            "max_abs_error": 0.0,
            "absolute_tolerance": 0.0,
            "exact_match": True,
        }
    )
    for name, tolerance in CROSS_DEVICE_TOLERANCES.items():
        actual = np.asarray(reconstructed[name])
        expected = np.asarray(saved[name])
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise RuntimeError(
                f"checkpoint {checkpoint_index} score reconstruction mismatch: {name}"
            )
        maximum_error = float(np.max(np.abs(actual - expected)))
        if maximum_error > tolerance:
            raise RuntimeError(
                f"checkpoint {checkpoint_index} score reconstruction mismatch: {name}"
            )
        rows.append(
            {
                "checkpoint": checkpoint_index,
                "array": name,
                "max_abs_error": maximum_error,
                "absolute_tolerance": tolerance,
                "exact_match": bool(maximum_error == 0.0),
            }
        )
    try:
        prediction = prediction_tie_audit(
            saved,
            reconstructed,
            joint_tolerance=CROSS_DEVICE_TOLERANCES["joint_scores"],
        )
    except RuntimeError as error:
        raise RuntimeError(
            f"checkpoint {checkpoint_index} score reconstruction mismatch: "
            f"predicted_class_id ({error})"
        ) from error
    rows.append(
        {
            "checkpoint": checkpoint_index,
            "array": "predicted_class_id",
            "max_abs_error": 0.0,
            "absolute_tolerance": 0.0,
            **prediction,
        }
    )
    return rows


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted, dtype=np.float64)
    return exponent / exponent.sum(axis=1, keepdims=True)


def _etg_mass(
    checkpoint,
    raw: np.ndarray,
    target_class: int,
    top_features: np.ndarray,
    controls: np.ndarray,
) -> dict:
    seen = [int(value) for value in checkpoint.metadata["seen_classes"]]
    column = seen.index(int(target_class))
    original = checkpoint.score(raw)["joint_scores"]
    q_original = _softmax(original)[:, column]
    sets = np.vstack((np.asarray(top_features, dtype=np.int64)[None, :], controls))
    masses: list[float] = []
    for feature_set in sets:
        masked = raw.copy()
        masked[:, feature_set] = checkpoint.mean[feature_set]
        q_masked = _softmax(checkpoint.score(masked)["joint_scores"])[:, column]
        masses.append(float(np.mean(q_original - q_masked)))
    top_mass = masses[0]
    random_masses = np.asarray(masses[1:], dtype=np.float64)
    null = float(np.quantile(random_masses, RANDOM_CONTROL_QUANTILE, method="linear"))
    return {
        "rationale_mass": top_mass,
        "random_null_95": null,
        "mass_margin": top_mass - null,
        "admitted": bool(top_mass > null),
        "random_masses": random_masses,
    }


def build_etg_ledger(checkpoint_rows: list[dict], transition_rows: list[dict]) -> list[dict]:
    """Strict ETG-v2 with a one-checkpoint review/re-certification lag.

    A drift event moves a certified class to DRIFTED.  At its next available
    checkpoint, re-certification requires both mass > null and Jaccard >= .70
    relative to the last certified reference.  A failed re-certification moves
    the class to UNEXPLAINABLE.  UNEXPLAINABLE is terminal in this analysis.
    """
    by_class: dict[int, list[dict]] = {}
    for row in checkpoint_rows:
        by_class.setdefault(int(row["class_id"]), []).append(row)
    primary = {
        (int(row["to_checkpoint"]), int(row["class_id"])): row
        for row in transition_rows
    }
    ledger: list[dict] = []
    for class_id, rows in sorted(by_class.items()):
        rows.sort(key=lambda item: int(item["checkpoint"]))
        state = "UNCERTIFIED"
        certified_reference: set[int] | None = None
        drift_checkpoint: int | None = None
        for row in rows:
            checkpoint = int(row["checkpoint"])
            top = set(int(value) for value in row["top15_indices"])
            old_state = state
            action = "monitor_no_change"
            reference_jaccard = None
            if state == "UNCERTIFIED":
                if row["admitted"]:
                    state = "CERTIFIED_STABLE"
                    certified_reference = top
                    action = "admission_certified"
                else:
                    state = "UNEXPLAINABLE"
                    action = "admission_refused_explanation_alert_withheld"
            elif state == "DRIFTED":
                assert certified_reference is not None and drift_checkpoint is not None
                reference_jaccard = jaccard(top, certified_reference)
                if row["admitted"] and reference_jaccard >= PRIMARY_JACCARD_THRESHOLD:
                    state = "CERTIFIED_STABLE"
                    certified_reference = top
                    action = "strict_recertified"
                else:
                    state = "UNEXPLAINABLE"
                    action = "strict_recertification_failed_explanation_alert_withheld"
                drift_checkpoint = None
            elif state == "UNEXPLAINABLE":
                action = "explanation_alert_withheld"
            elif state == "CERTIFIED_STABLE":
                transition = primary.get((checkpoint, class_id))
                if transition is not None and bool(transition["primary_event"]):
                    state = "DRIFTED"
                    drift_checkpoint = checkpoint
                    action = "human_review_escalation"
            ledger.append(
                {
                    "checkpoint": checkpoint,
                    "class_id": class_id,
                    "class_name": row["class_name"],
                    "state_before": old_state,
                    "state_after": state,
                    "action": action,
                    "rationale_mass": row["rationale_mass"],
                    "random_null_95": row["random_null_95"],
                    "mass_margin": row["mass_margin"],
                    "mass_admitted": row["admitted"],
                    "certified_reference_jaccard": reference_jaccard,
                    "drift_checkpoint": drift_checkpoint,
                }
            )
    return sorted(ledger, key=lambda row: (row["checkpoint"], row["class_id"]))


def _validate_inputs(result_dir: Path, seed: int, method_protocol: Path) -> tuple[dict, dict, DatasetManifest, dict]:
    result_path = result_dir / f"result_seed_{seed}.json"
    protocol_path = result_dir / "protocol.json"
    result = _json(result_path)
    protocol = _json(protocol_path)
    stored_protocol = protocol.get("protocol_sha256")
    if stored_protocol != canonical_sha256({k: v for k, v in protocol.items() if k != "protocol_sha256"}):
        raise RuntimeError("training protocol has an invalid self-hash")
    if result.get("protocol_sha256") != stored_protocol:
        raise RuntimeError("result/training protocol binding mismatch")
    if result.get("dataset") != protocol.get("dataset") or int(result.get("seed", -1)) != seed:
        raise RuntimeError("result dataset/seed mismatch")
    if result.get("deterministic_result_sha256") != canonical_sha256(_without_timing(result)):
        raise RuntimeError("result deterministic SHA-256 mismatch")
    monitor_protocol = protocol.get("monitoring")
    if not isinstance(monitor_protocol, dict) or not monitor_protocol.get("enabled"):
        raise RuntimeError("formal-v2 requires enabled monitored checkpoints")
    validate_monitoring_result(result, output_base=result_dir, expected_protocol=monitor_protocol)
    manifest_path = Path(str(protocol.get("manifest"))).resolve()
    manifest = load_manifest(manifest_path, verify_hashes=True)
    if manifest.manifest_sha256 != protocol.get("manifest_sha256"):
        raise RuntimeError("streaming manifest SHA-256 differs from training protocol")
    probe_record = result["monitoring"]["probe_manifest"]
    probe_path = result_dir / str(probe_record["relative_path"])
    probe = _json(probe_path)
    if sha256_file(probe_path) != probe_record.get("file_sha256"):
        raise RuntimeError("probe manifest file SHA-256 mismatch")
    if probe.get("canonical_sha256") != monitor_protocol.get("probe_contract_sha256"):
        raise RuntimeError("probe contract SHA-256 mismatch")
    if not method_protocol.is_file():
        raise FileNotFoundError(method_protocol)
    return result, protocol, manifest, probe


def _controls(dataset: str, seed: int, class_id: int, feature_dim: int) -> np.ndarray:
    rng = np.random.default_rng(_derived_seed(SCHEMA_VERSION, dataset, seed, class_id, "etg-controls"))
    return np.vstack(
        [rng.choice(feature_dim, PRIMARY_K, replace=False) for _ in range(RANDOM_CONTROL_COUNT)]
    ).astype(np.int64)


def run(args: argparse.Namespace) -> Path:
    if args.confirm != CONFIRMATION:
        raise RuntimeError(f"refusing analysis without --confirm {CONFIRMATION}")
    if args.device != "cpu":
        raise RuntimeError("formal-v2 analyzer is CPU-only")
    result_dir = args.result_dir.resolve()
    method_protocol = args.method_protocol.resolve()
    result, training_protocol, manifest, probe = _validate_inputs(
        result_dir, args.seed, method_protocol
    )
    if tuple(args.k_values) != K_VALUES or tuple(args.jaccard_thresholds) != JACCARD_THRESHOLDS or tuple(args.allowed_recall_drops) != ALLOWED_RECALL_DROPS:
        raise RuntimeError("formal-v2 sensitivity grid is frozen and cannot be changed")
    if args.shap_nsamples <= 0:
        raise ValueError("SHAP nsamples must be positive")

    output = args.output_dir.resolve()
    staging = output.with_name(f".{output.name}.staging")
    if output.exists() and not args.replace_output:
        raise FileExistsError(f"output exists; use --replace-output after validation: {output}")
    resuming = staging.exists()
    if resuming and not args.resume:
        raise FileExistsError(
            f"resumable staging exists; inspect it and use --resume: {staging}"
        )
    if not resuming:
        staging.mkdir(parents=True)

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized in the CPU-only analyzer")

    official = _probe_rows(manifest, list(probe["official_test"]["samples"]), split="official_test")
    background = _probe_rows(
        manifest,
        list(probe["task0_train_background"]["samples"]),
        split="task0_train_background",
    )
    feature_schema_path = manifest.path.parent / "feature_schema.json"
    feature_schema = _json(feature_schema_path)
    feature_names = list(feature_schema.get("feature_columns", []))
    if len(feature_names) != manifest.feature_dim:
        raise RuntimeError("feature schema width mismatch")
    fullcache_manifest_path = Path(
        str(training_protocol["input_source_provenance"]["fullcache_manifest"]["path"])
    ).resolve()
    fullcache_manifest = _json(fullcache_manifest_path)
    if sha256_file(fullcache_manifest_path) != training_protocol["input_source_provenance"]["fullcache_manifest"]["sha256"]:
        raise RuntimeError("full-cache manifest file SHA-256 mismatch")
    if feature_schema.get("feature_schema_sha256") != fullcache_manifest.get("feature_schema_sha256"):
        raise RuntimeError("feature schema identity differs from training protocol")

    analyzer_sha = sha256_file(Path(__file__).resolve())
    protocol_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "post-training offline analysis; method choices fixed before attribution execution",
        "dataset": manifest.dataset,
        "seed": args.seed,
        "score_target": "g_c(x)=joint_cap3000_c(x)-max_{f!=c}joint_cap3000_f(x)",
        "joint_formula": "positive_family_probability + 0.5 * cap3000_router_population_zscore",
        "attribution": {
            "library": "shap",
            "explainer": "GradientExplainer",
            "estimand": "expected gradients approximation to SHAP values",
            "nsamples": args.shap_nsamples,
            "local_smoothing": 0,
            "background": "all fixed Task-0 background rows in monitor contract",
            "probe": "all fixed true-class official-test probe rows available at checkpoint",
            "aggregation": "per-class mean absolute SHAP; signed mean retained as companion",
            "input_space": "raw cached features; frozen Task-0 normalization occurs inside model",
            "device": "cpu",
            "cpu_threads": args.cpu_threads,
        },
        "stability": {
            "k_values": list(K_VALUES),
            "jaccard_thresholds": list(JACCARD_THRESHOLDS),
            "allowed_recall_drops": list(ALLOWED_RECALL_DROPS),
            "primary": {"k": 15, "jaccard_threshold": 0.70, "allowed_recall_drop": 0.05},
            "denominator": "common-class adjacent-checkpoint transitions with delta_recall > -allowed_drop",
            "recall_source": "full official view, joint_cap3000 arm",
        },
        "etg": {
            "version": "strict_etg_v2_one_checkpoint_lag",
            "rationale_mass": "mean class softmax-probability deletion mass on fixed true-class probes",
            "reference_value": "frozen Task-0 normalization mean (standardized zero)",
            "random_controls": RANDOM_CONTROL_COUNT,
            "control_size": PRIMARY_K,
            "null_quantile": RANDOM_CONTROL_QUANTILE,
            "admission": "rationale_mass > random_null_95",
            "strict_recertification": "mass admitted and Jaccard(current,last certified reference)>=0.70 at next checkpoint",
            "unexplainable_policy": "terminal for this analysis; classifier retained, explanation notification withheld",
        },
        "bindings": {
            "training_protocol_sha256": training_protocol["protocol_sha256"],
            "training_protocol_file_sha256": sha256_file(result_dir / "protocol.json"),
            "result_deterministic_sha256": result["deterministic_result_sha256"],
            "result_file_sha256": sha256_file(result_dir / f"result_seed_{args.seed}.json"),
            "streaming_manifest_sha256": manifest.manifest_sha256,
            "probe_contract_sha256": probe["canonical_sha256"],
            "probe_manifest_file_sha256": sha256_file(result_dir / result["monitoring"]["probe_manifest"]["relative_path"]),
            "method_protocol_file": str(method_protocol),
            "method_protocol_file_sha256": sha256_file(method_protocol),
            "feature_schema_file_sha256": sha256_file(feature_schema_path),
            "analyzer_file_sha256": analyzer_sha,
        },
        "dependencies": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    import shap
    import scipy

    protocol_payload["dependencies"]["shap"] = shap.__version__
    protocol_payload["dependencies"]["scipy"] = scipy.__version__
    protocol_payload["analysis_protocol_sha256"] = canonical_sha256(protocol_payload)
    protocol_path = staging / "analysis_protocol.json"
    if resuming:
        if not protocol_path.is_file():
            raise RuntimeError("resumable staging has no analysis protocol")
        previous_protocol = _json(protocol_path)
        if previous_protocol.get("analysis_protocol_sha256") != protocol_payload[
            "analysis_protocol_sha256"
        ]:
            raise RuntimeError("resumable staging protocol does not match this run")
    _atomic_json(protocol_path, protocol_payload)

    raw_class_names = training_protocol["class_names"]
    if not isinstance(raw_class_names, dict):
        raise RuntimeError("training protocol class-name registry is invalid")
    class_records = {int(class_id): str(name) for class_id, name in raw_class_names.items()}
    attribution_arrays: dict[str, np.ndarray] = {}
    control_arrays: dict[str, np.ndarray] = {}
    checkpoint_rows: list[dict] = []
    score_fidelity: list[dict] = []
    artifact_reconstruction_fidelity: list[dict] = []
    saved_scores: dict[int, dict[str, np.ndarray]] = {}
    expected_total = sum(
        len(checkpoint["seen_classes"]) for checkpoint in result["checkpoints"]
    )
    progress_path = staging / "progress.json"
    if resuming:
        progress = _json(progress_path)
        if (
            progress.get("schema_version") != SCHEMA_VERSION
            or progress.get("dataset") != manifest.dataset
            or int(progress.get("seed", -1)) != args.seed
            or int(progress.get("total_class_checkpoint_rows", -1)) != expected_total
        ):
            raise RuntimeError("resumable progress contract mismatch")
        required_resume_files = (
            staging / "resume_attributions.npz",
            staging / "resume_etg_controls_and_masses.npz",
            staging / "resume_checkpoint_rows.json",
        )
        if progress.get("completed_class_checkpoint_rows"):
            if not all(path.is_file() for path in required_resume_files):
                raise RuntimeError("resumable progress is missing committed state files")
            attribution_arrays = _load_npz(required_resume_files[0])
            control_arrays = _load_npz(required_resume_files[1])
            raw_rows = json.loads(required_resume_files[2].read_text(encoding="utf-8"))
            if not isinstance(raw_rows, list) or not all(
                isinstance(row, dict) for row in raw_rows
            ):
                raise RuntimeError("resumable checkpoint rows are invalid")
            checkpoint_rows = raw_rows
            (
                attribution_arrays,
                control_arrays,
                checkpoint_rows,
            ) = _reconcile_resume_state(
                progress=progress,
                attribution_arrays=attribution_arrays,
                control_arrays=control_arrays,
                checkpoint_rows=checkpoint_rows,
            )
        progress["status"] = "running"
        progress["active"] = None
    else:
        progress = {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "dataset": manifest.dataset,
            "seed": args.seed,
            "total_class_checkpoint_rows": expected_total,
            "completed_class_checkpoint_rows": [],
            "active": None,
        }
        _atomic_json(progress_path, progress)
    completed_pairs = {
        (int(row["checkpoint"]), int(row["class_id"]))
        for row in progress["completed_class_checkpoint_rows"]
    }
    row_pairs = {
        (int(row["checkpoint"]), int(row["class_id"])) for row in checkpoint_rows
    }
    if completed_pairs != row_pairs:
        raise RuntimeError("resumable progress and checkpoint rows disagree")

    checkpoint_contracts: list[tuple[dict, dict, Path, dict]] = []
    for checkpoint_entry, result_checkpoint in zip(
        result["monitoring"]["checkpoints"], result["checkpoints"]
    ):
        checkpoint_index = int(checkpoint_entry["checkpoint"])
        manifest_path = result_dir / str(checkpoint_entry["checkpoint_manifest_relative_path"])
        metadata = validate_checkpoint_manifest(manifest_path)
        checkpoint = load_checkpoint(manifest_path, device="cpu")
        seen = [int(value) for value in metadata["seen_classes"]]
        mask = np.asarray([int(record["class_id"]) in set(seen) for record in official.records], dtype=bool)
        raw_seen = official.values[mask]
        record_seen = [record for keep, record in zip(mask.tolist(), official.records) if keep]
        reconstructed = checkpoint.score(raw_seen)
        score_path = manifest_path.parent / str(metadata["probe_scores_file"])
        with np.load(score_path, allow_pickle=False) as archive:
            saved = {name: np.array(archive[name], copy=True) for name in archive.files}
        artifact_reconstruction_fidelity.extend(
            audit_checkpoint_reconstruction(checkpoint_index, saved, reconstructed)
        )
        saved_scores[checkpoint_index] = saved
        checkpoint_contracts.append(
            (checkpoint_entry, result_checkpoint, manifest_path, metadata)
        )
        print(
            json.dumps(
                {
                    "event": "checkpoint_reconstruction_audit_complete",
                    "checkpoint": checkpoint_index,
                    "prediction_mismatches": next(
                        row["mismatch_count"]
                        for row in artifact_reconstruction_fidelity
                        if row["checkpoint"] == checkpoint_index
                        and row["array"] == "predicted_class_id"
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del checkpoint, reconstructed, raw_seen, record_seen
        gc.collect()

    print(
        json.dumps(
            {
                "event": "all_checkpoint_reconstruction_audits_complete",
                "checkpoints": len(checkpoint_contracts),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for checkpoint_entry, result_checkpoint, manifest_path, metadata in checkpoint_contracts:
        checkpoint_index = int(checkpoint_entry["checkpoint"])
        checkpoint = load_checkpoint(manifest_path, device="cpu")
        seen = [int(value) for value in metadata["seen_classes"]]
        seen_set = set(seen)
        mask = np.asarray(
            [int(record["class_id"]) in seen_set for record in official.records],
            dtype=bool,
        )
        raw_seen = official.values[mask]
        record_seen = [
            record for keep, record in zip(mask.tolist(), official.records) if keep
        ]
        reconstructed = checkpoint.score(raw_seen)

        recall = {
            int(row["class_id"]): float(row["recall"])
            for row in result_checkpoint["views"]["official"]["arms"]["joint_cap3000"]["per_class"]
        }
        for class_id in seen:
            class_mask = np.asarray([int(record["class_id"]) == class_id for record in record_seen], dtype=bool)
            class_raw = raw_seen[class_mask]
            if not len(class_raw):
                raise RuntimeError("seen class has no fixed probe rows")
            margin_model = JointMarginModel(checkpoint, class_id)
            with torch.no_grad():
                margin = margin_model(torch.from_numpy(np.ascontiguousarray(class_raw))).numpy().ravel()
            class_axis = reconstructed["class_axis"].tolist()
            target_column = class_axis.index(class_id)
            other_columns = [index for index, value in enumerate(class_axis) if value != class_id]
            exact_margin = reconstructed["joint_scores"][class_mask, target_column] - reconstructed["joint_scores"][class_mask][:, other_columns].max(axis=1)
            max_error = float(np.max(np.abs(margin - exact_margin)))
            if max_error > 2e-3:
                raise RuntimeError(f"differentiable score target fidelity failure: {max_error}")
            score_fidelity.append({"checkpoint": checkpoint_index, "class_id": class_id, "max_abs_error": max_error})

            if (checkpoint_index, class_id) in completed_pairs:
                print(
                    json.dumps(
                        {
                            "event": "class_attribution_resume_skip",
                            "checkpoint": checkpoint_index,
                            "class_id": class_id,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue

            progress["active"] = {
                "phase": "shap",
                "checkpoint": checkpoint_index,
                "class_id": class_id,
            }
            _atomic_json(staging / "progress.json", progress)
            explainer = shap.GradientExplainer(
                margin_model,
                torch.from_numpy(np.ascontiguousarray(background.values)),
                batch_size=args.shap_batch_size,
                local_smoothing=0,
            )
            shap_seed = _derived_seed(
                SCHEMA_VERSION, manifest.dataset, args.seed, checkpoint_index, class_id, "shap"
            ) % (2**32)
            values = np.asarray(
                explainer.shap_values(
                    torch.from_numpy(np.ascontiguousarray(class_raw)),
                    nsamples=args.shap_nsamples,
                    rseed=shap_seed,
                ),
                dtype=np.float64,
            )
            if values.ndim == 3 and values.shape[-1] == 1:
                values = values[..., 0]
            if values.shape != class_raw.shape or not np.isfinite(values).all():
                raise RuntimeError("SHAP output shape/finiteness failure")
            progress["active"] = {
                "phase": "etg",
                "checkpoint": checkpoint_index,
                "class_id": class_id,
            }
            _atomic_json(staging / "progress.json", progress)
            print(
                json.dumps(
                    {
                        "event": "shap_complete",
                        "checkpoint": checkpoint_index,
                        "class_id": class_id,
                        "probe_rows": len(class_raw),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            mean_abs = np.mean(np.abs(values), axis=0)
            mean_signed = np.mean(values, axis=0)
            prefix = f"checkpoint_{checkpoint_index:03d}_class_{class_id:03d}"
            attribution_arrays[f"{prefix}_mean_abs"] = mean_abs
            attribution_arrays[f"{prefix}_mean_signed"] = mean_signed
            attribution_arrays[f"{prefix}_sample_id_sha256"] = np.asarray(
                [str(record["sample_id_sha256"]).encode("ascii") for keep, record in zip(class_mask.tolist(), record_seen) if keep], dtype="S64"
            )
            top_sets = {k: _topk(mean_abs, k) for k in K_VALUES}
            controls = _controls(manifest.dataset, args.seed, class_id, manifest.feature_dim)
            control_arrays[f"class_{class_id:03d}_feature_sets"] = controls
            mass = _etg_mass(checkpoint, class_raw, class_id, top_sets[PRIMARY_K], controls)
            control_arrays[f"{prefix}_random_masses"] = mass.pop("random_masses")
            checkpoint_rows.append(
                {
                    "dataset": manifest.dataset,
                    "seed": args.seed,
                    "checkpoint": checkpoint_index,
                    "class_id": class_id,
                    "class_name": class_records[class_id],
                    "probe_rows": len(class_raw),
                    "background_rows": len(background.values),
                    "recall": recall[class_id],
                    "rationale_mass": mass["rationale_mass"],
                    "random_null_95": mass["random_null_95"],
                    "mass_margin": mass["mass_margin"],
                    "admitted": mass["admitted"],
                    "top15_indices": top_sets[15].tolist(),
                    "top15_features": [feature_names[index] for index in top_sets[15]],
                }
            )
            progress["completed_class_checkpoint_rows"].append(
                {"checkpoint": checkpoint_index, "class_id": class_id}
            )
            completed_pairs.add((checkpoint_index, class_id))
            progress["active"] = None
            _persist_resume_state(
                staging,
                attribution_arrays=attribution_arrays,
                control_arrays=control_arrays,
                checkpoint_rows=checkpoint_rows,
                progress=progress,
            )
            print(
                json.dumps(
                    {
                        "event": "class_attribution_complete",
                        "checkpoint": checkpoint_index,
                        "class_id": class_id,
                        "probe_rows": len(class_raw),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        del checkpoint, reconstructed, raw_seen, record_seen
        gc.collect()

    transition_rows: list[dict] = []
    by_key = {(row["checkpoint"], row["class_id"]): row for row in checkpoint_rows}
    for checkpoint in range(1, len(result["checkpoints"])):
        previous = checkpoint - 1
        common = sorted(set(result["checkpoints"][previous]["seen_classes"]) & set(result["checkpoints"][checkpoint]["seen_classes"]))
        prev_scores, curr_scores = saved_scores[previous], saved_scores[checkpoint]
        previous_ids = {value.decode("ascii"): i for i, value in enumerate(prev_scores["sample_id_sha256"])}
        current_ids = {value.decode("ascii"): i for i, value in enumerate(curr_scores["sample_id_sha256"])}
        common_ids = sorted(set(previous_ids) & set(current_ids))
        flips = sum(
            int(prev_scores["predicted_class_id"][previous_ids[sample_id]] != curr_scores["predicted_class_id"][current_ids[sample_id]])
            for sample_id in common_ids
        )
        checkpoint_flip_rate = flips / len(common_ids) if common_ids else None
        for class_id in common:
            left = attribution_arrays[f"checkpoint_{previous:03d}_class_{class_id:03d}_mean_abs"]
            right = attribution_arrays[f"checkpoint_{checkpoint:03d}_class_{class_id:03d}_mean_abs"]
            record = {
                "dataset": manifest.dataset,
                "seed": args.seed,
                "from_checkpoint": previous,
                "to_checkpoint": checkpoint,
                "class_id": class_id,
                "class_name": class_records[class_id],
                "recall_before": by_key[(previous, class_id)]["recall"],
                "recall_after": by_key[(checkpoint, class_id)]["recall"],
                "delta_recall": by_key[(checkpoint, class_id)]["recall"] - by_key[(previous, class_id)]["recall"],
                "cosine_similarity": _cosine(left, right),
                "kendall_tau_b": _kendall(left, right),
                "prediction_flip_rate_all_common_probe_rows": checkpoint_flip_rate,
                "prediction_flip_events": flips,
                "prediction_flip_denominator": len(common_ids),
            }
            for k in K_VALUES:
                record[f"jaccard_top{k}"] = jaccard(_topk(left, k), _topk(right, k))
            record["primary_eligible"] = record["delta_recall"] > -PRIMARY_ALLOWED_RECALL_DROP
            record["primary_event"] = bool(record["primary_eligible"] and record["jaccard_top15"] < PRIMARY_JACCARD_THRESHOLD)
            transition_rows.append(record)

    grid = sensitivity_rows(transition_rows)
    ledger = build_etg_ledger(checkpoint_rows, transition_rows)
    ledger_map = {(row["checkpoint"], row["class_id"]): row for row in ledger}
    for row in checkpoint_rows:
        item = ledger_map[(row["checkpoint"], row["class_id"])]
        row["etg_state"] = item["state_after"]
        row["etg_action"] = item["action"]
        transition = next((value for value in transition_rows if value["to_checkpoint"] == row["checkpoint"] and value["class_id"] == row["class_id"]), None)
        row["jaccard_top15_from_previous"] = transition["jaccard_top15"] if transition else None
        row["silent_drift_primary_event"] = transition["primary_event"] if transition else None

    primary = next(row for row in grid if row["is_primary"])
    etg_summary = {
        "certified_admissions": sum(row["action"] == "admission_certified" for row in ledger),
        "refused_admissions": sum(row["action"].startswith("admission_refused") for row in ledger),
        "escalations": sum(row["action"] == "human_review_escalation" for row in ledger),
        "strict_recertifications": sum(row["action"] == "strict_recertified" for row in ledger),
        "strict_recertification_failures": sum(row["action"].startswith("strict_recertification_failed") for row in ledger),
        "explanation_alerts_withheld_records": sum("withheld" in row["action"] for row in ledger),
        "final_states": {},
    }
    for class_id in sorted({row["class_id"] for row in ledger}):
        final = [row for row in ledger if row["class_id"] == class_id][-1]["state_after"]
        etg_summary["final_states"][final] = etg_summary["final_states"].get(final, 0) + 1

    _deterministic_npz(staging / "attributions.npz", attribution_arrays)
    _deterministic_npz(staging / "etg_controls_and_masses.npz", control_arrays)
    checkpoint_fields = [
        "dataset", "seed", "checkpoint", "class_id", "class_name", "probe_rows", "background_rows", "recall",
        "jaccard_top15_from_previous", "silent_drift_primary_event", "rationale_mass", "random_null_95", "mass_margin",
        "admitted", "etg_state", "etg_action", "top15_indices", "top15_features",
    ]
    transition_fields = [
        "dataset", "seed", "from_checkpoint", "to_checkpoint", "class_id", "class_name", "recall_before", "recall_after",
        "delta_recall", "jaccard_top5", "jaccard_top10", "jaccard_top15", "jaccard_top20", "cosine_similarity",
        "kendall_tau_b", "primary_eligible", "primary_event", "prediction_flip_rate_all_common_probe_rows",
        "prediction_flip_events", "prediction_flip_denominator",
    ]
    grid_fields = ["k", "jaccard_threshold", "allowed_recall_drop", "events", "eligible_transitions", "rate", "is_primary"]
    ledger_fields = [
        "checkpoint", "class_id", "class_name", "state_before", "state_after", "action", "rationale_mass", "random_null_95",
        "mass_margin", "mass_admitted", "certified_reference_jaccard", "drift_checkpoint",
    ]
    _csv(staging / "checkpoint_metrics.csv", checkpoint_rows, checkpoint_fields)
    _csv(staging / "transition_metrics.csv", transition_rows, transition_fields)
    _csv(staging / "threshold_sensitivity.csv", grid, grid_fields)
    _csv(staging / "etg_ledger.csv", ledger, ledger_fields)

    analysis = {
        "schema_version": SCHEMA_VERSION,
        "dataset": manifest.dataset,
        "seed": args.seed,
        "analysis_protocol_sha256": protocol_payload["analysis_protocol_sha256"],
        "primary_silent_explanation_drift": primary,
        "etg_summary": etg_summary,
        "score_target_fidelity": {
            "maximum_absolute_error": max(row["max_abs_error"] for row in score_fidelity),
            "absolute_tolerance": 2e-3,
            "note": "CPU differentiable torch.cdist representation versus CPU NumPy router; saved GPU score hashes remain validated exactly",
            "rows": score_fidelity,
        },
        "cpu_reload_vs_saved_gpu_score_fidelity": {
            "maximum_absolute_error": max(row["max_abs_error"] for row in artifact_reconstruction_fidelity),
            "absolute_tolerances": CROSS_DEVICE_TOLERANCES,
            "integer_class_axis_exact": True,
            "prediction_policy": "saved predictions remain authoritative; CPU reconstruction differences are accepted only for audited near ties within twice the joint-score absolute tolerance",
            "prediction_near_tie_mismatch_count": sum(
                int(row.get("tie_compatible_mismatch_count", 0))
                for row in artifact_reconstruction_fidelity
                if row["array"] == "predicted_class_id"
            ),
            "rows": artifact_reconstruction_fidelity,
        },
        "counts": {
            "checkpoints": len(result["checkpoints"]),
            "class_checkpoint_rows": len(checkpoint_rows),
            "class_adjacent_transitions": len(transition_rows),
            "sensitivity_cells": len(grid),
            "background_rows": len(background.values),
            "probe_rows_total_contract": len(official.values),
        },
        "checkpoint_rows": checkpoint_rows,
        "transition_rows": transition_rows,
        "threshold_sensitivity": grid,
        "etg_ledger": ledger,
        "interpretation_guardrails": [
            "The drift rate unit is class by adjacent-checkpoint transition, not packets, flows, samples, or real-world drift events.",
            "ETG actions are simulated explanation-governance outcomes, not measured NIDS alerts or completed human reviews.",
            "GradientExplainer reports an expected-gradients approximation; it is not exact combinatorial Shapley enumeration.",
            "This seed-1 artifact is not a five-seed aggregate and must be labelled partial until all registered seeds finish.",
        ],
    }
    analysis["canonical_sha256"] = canonical_sha256(analysis)
    _atomic_json(staging / "analysis.json", analysis)

    progress["status"] = "complete"
    progress["active"] = None
    _atomic_json(staging / "progress.json", progress)
    for resume_name in (
        "resume_attributions.npz",
        "resume_etg_controls_and_masses.npz",
        "resume_checkpoint_rows.json",
    ):
        resume_path = staging / resume_name
        if resume_path.exists():
            resume_path.unlink()

    files = {}
    for path in sorted(staging.iterdir()):
        if path.name != "analysis_manifest.json" and path.is_file():
            files[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    manifest_out = {
        "schema_version": SCHEMA_VERSION,
        "dataset": manifest.dataset,
        "seed": args.seed,
        "analysis_protocol_sha256": protocol_payload["analysis_protocol_sha256"],
        "analysis_canonical_sha256": analysis["canonical_sha256"],
        "input_bindings": protocol_payload["bindings"],
        "files": files,
        "publication_status": "partial_seed_result; eligible for clearly labelled new-run tracking only",
    }
    manifest_out["canonical_sha256"] = canonical_sha256(manifest_out)
    _atomic_json(staging / "analysis_manifest.json", manifest_out)
    validate_output(staging)
    if output.exists():
        shutil.rmtree(output)
    os.replace(staging, output)
    return output


def validate_output(path: Path) -> dict:
    manifest = _json(path / "analysis_manifest.json")
    stored = manifest.get("canonical_sha256")
    if stored != canonical_sha256({k: v for k, v in manifest.items() if k != "canonical_sha256"}):
        raise RuntimeError("analysis manifest self-hash mismatch")
    for name, record in manifest.get("files", {}).items():
        target = path / name
        if not target.is_file() or sha256_file(target) != record.get("sha256") or target.stat().st_size != record.get("bytes"):
            raise RuntimeError(f"analysis output file integrity failure: {target}")
    protocol = _json(path / "analysis_protocol.json")
    if protocol.get("analysis_protocol_sha256") != canonical_sha256({k: v for k, v in protocol.items() if k != "analysis_protocol_sha256"}):
        raise RuntimeError("analysis protocol self-hash mismatch")
    analysis = _json(path / "analysis.json")
    if analysis.get("canonical_sha256") != canonical_sha256({k: v for k, v in analysis.items() if k != "canonical_sha256"}):
        raise RuntimeError("analysis self-hash mismatch")
    if analysis.get("analysis_protocol_sha256") != protocol.get("analysis_protocol_sha256"):
        raise RuntimeError("analysis/protocol binding mismatch")
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--result-dir", type=Path, required=True)
    value.add_argument("--seed", type=int, required=True)
    value.add_argument("--method-protocol", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--device", default="cpu", choices=("cpu",))
    value.add_argument("--cpu-threads", type=int, default=2)
    value.add_argument("--shap-nsamples", type=int, default=64)
    value.add_argument("--shap-batch-size", type=int, default=16)
    value.add_argument("--k-values", type=int, nargs="+", default=list(K_VALUES))
    value.add_argument("--jaccard-thresholds", type=float, nargs="+", default=list(JACCARD_THRESHOLDS))
    value.add_argument("--allowed-recall-drops", type=float, nargs="+", default=list(ALLOWED_RECALL_DROPS))
    value.add_argument("--confirm", required=True)
    value.add_argument("--replace-output", action="store_true")
    value.add_argument("--resume", action="store_true")
    return value


if __name__ == "__main__":
    try:
        destination = run(parser().parse_args())
        print(json.dumps({"status": "complete", "output": str(destination)}, sort_keys=True))
    except Exception as error:
        print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}, sort_keys=True), file=sys.stderr)
        raise
