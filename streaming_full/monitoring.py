"""Auditable, raw-row-free checkpoint monitoring for explanation analysis.

The monitor deliberately does not implement an explanation method.  It freezes
sample identities and persists the exact inputs to the official
``joint_cap3000`` decision rule so that a separately versioned explanation
protocol can be run later without retraining.
"""
from __future__ import annotations

import io
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

from .data import (
    ClassShards,
    DatasetManifest,
    array_sha256,
    canonical_sha256,
    sha256_file,
)
from .models import FamilyHead, build_encoder
from .routers import DualRouter, FamilyRouterState

if TYPE_CHECKING:
    from .validation import StreamingOFRA


MONITOR_SCHEMA_VERSION = 1
ROUTER_STANDARDIZATION = {
    "algorithm": "per_sample_population_zscore_v1",
    "axis": "seen_class_axis",
    "ddof": 0,
    "epsilon": 1e-8,
}


def _atomic_write_json(path: Path, value: object) -> None:
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


def _npy_bytes(value: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, np.ascontiguousarray(value), allow_pickle=False)
    return output.getvalue()


def _atomic_write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write byte-reproducible, pickle-free NPZ with fixed ZIP metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with zipfile.ZipFile(
            temporary, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as archive:
            for name in sorted(arrays):
                if not name or name.endswith(".npy") or "/" in name or "\\" in name:
                    raise ValueError(f"invalid deterministic NPZ key: {name!r}")
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, _npy_bytes(np.asarray(arrays[name])))
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _selected_global_indices(row_count: int, requested: int) -> np.ndarray:
    count = min(int(row_count), int(requested))
    if row_count <= 0 or count <= 0:
        raise ValueError("monitor probes require positive source rows and sample count")
    # Deterministic midpoint stratification covers the complete class stream and
    # is independent of the training seed and process RNG.
    indices = np.asarray(
        [((2 * position + 1) * row_count) // (2 * count) for position in range(count)],
        dtype=np.int64,
    )
    if len(np.unique(indices)) != count or indices[0] < 0 or indices[-1] >= row_count:
        raise RuntimeError("monitor midpoint selection is not unique/in range")
    return indices


def _sample_records(
    manifest: DatasetManifest,
    sources: dict[int, ClassShards],
    *,
    split: str,
    class_ids: Sequence[int],
    per_class: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for class_id in class_ids:
        source = sources[int(class_id)]
        indices = _selected_global_indices(len(source), per_class)
        rows = source.take(indices)
        for global_index, row in zip(indices.tolist(), rows):
            shard_ordinal = int(
                np.searchsorted(source.offsets[1:], global_index, side="right")
            )
            local_row = int(global_index - source.offsets[shard_ordinal])
            shard = source.shards[shard_ordinal]
            identity_payload = {
                "dataset": manifest.dataset,
                "streaming_manifest_sha256": manifest.manifest_sha256,
                "split": split,
                "class_id": int(class_id),
                "shard_ordinal": shard_ordinal,
                "local_row": local_row,
                "parent_shard_sha256": shard.sha256,
            }
            records.append(
                {
                    **identity_payload,
                    "sample_id_sha256": canonical_sha256(identity_payload),
                    "feature_row_sha256": array_sha256(
                        np.ascontiguousarray(row[None, :], dtype=np.float32)
                    ),
                }
            )
    return records


def build_probe_contract(
    manifest: DatasetManifest,
    *,
    official_test_per_class: int,
    task0_background_per_class: int,
    train_sources: dict[int, ClassShards] | None = None,
    test_sources: dict[int, ClassShards] | None = None,
) -> dict[str, object]:
    """Create a coordinate/hash-only probe contract; no feature row is retained."""

    owns_sources = train_sources is None or test_sources is None
    if owns_sources:
        train_sources = {
            record.class_id: ClassShards(record.train, manifest.feature_dim)
            for record in manifest.classes
        }
        test_sources = {
            record.class_id: ClassShards(record.test, manifest.feature_dim)
            for record in manifest.classes
        }
    assert train_sources is not None and test_sources is not None
    try:
        payload: dict[str, object] = {
            "schema_version": MONITOR_SCHEMA_VERSION,
            "algorithm": "seed_independent_midpoint_shard_coordinate_probe_v1",
            "dataset": manifest.dataset,
            "streaming_manifest_sha256": manifest.manifest_sha256,
            "raw_feature_rows_persisted": False,
            "official_test": {
                "per_class_requested": int(official_test_per_class),
                "selection": "midpoint_stratification_over_manifest_class_stream",
                "samples": _sample_records(
                    manifest,
                    test_sources,
                    split="official_test",
                    class_ids=[record.class_id for record in manifest.classes],
                    per_class=official_test_per_class,
                ),
            },
            "task0_train_background": {
                "per_class_requested": int(task0_background_per_class),
                "selection": "midpoint_stratification_over_manifest_class_stream",
                "source_classes": [int(value) for value in manifest.tasks[0]],
                "samples": _sample_records(
                    manifest,
                    train_sources,
                    split="task0_train_background",
                    class_ids=manifest.tasks[0],
                    per_class=task0_background_per_class,
                ),
            },
        }
        payload["canonical_sha256"] = canonical_sha256(payload)
        return payload
    finally:
        if owns_sources:
            for source in (*train_sources.values(), *test_sources.values()):
                source.close()


def monitoring_protocol_record(
    manifest: DatasetManifest,
    *,
    enabled: bool,
    official_test_per_class: int,
    task0_background_per_class: int,
) -> dict[str, object]:
    base: dict[str, object] = {
        "enabled": bool(enabled),
        "schema_version": MONITOR_SCHEMA_VERSION,
        "purpose": (
            "persist exact decision inputs for separately versioned offline "
            "prediction-flip, explanation-drift, and ETG analyses"
        ),
        "explanation_method_in_this_protocol": None,
        "checkpoint_state_scope": [
            "encoder_state",
            "seen_family_head_states",
            "frozen_normalization_mean_and_scale",
            "seen_cap3000_router_centroids_counts_lambda",
            "seen_class_axis",
        ],
        "excluded_state": [
            "optimizer_state",
            "uncapped_router_state",
            "exemplars",
            "training_rows",
            "raw_probe_rows",
        ],
        "score_trace": [
            "true_class_id",
            "predicted_class_id",
            "head_scores",
            "router_z_scores",
            "joint_scores",
        ],
        "joint_formula": "head_score + 0.5 * router_z_score",
        "router_standardization": ROUTER_STANDARDIZATION,
        "integrity": "SHA-256 for every JSON/NPZ artifact; fail closed on mismatch",
    }
    if enabled:
        contract = build_probe_contract(
            manifest,
            official_test_per_class=official_test_per_class,
            task0_background_per_class=task0_background_per_class,
        )
        base["probe_contract"] = contract
        base["probe_contract_sha256"] = contract["canonical_sha256"]
    return base


def _state_arrays(agent: "StreamingOFRA", seen_classes: list[int]) -> tuple[dict, dict]:
    arrays: dict[str, np.ndarray] = {
        "normalization_mean": np.asarray(agent.stats.mean, dtype=np.float64),
        "normalization_scale": np.asarray(agent.stats.scale, dtype=np.float64),
    }
    encoder_keys: dict[str, str] = {}
    for index, (name, tensor) in enumerate(sorted(agent.encoder.state_dict().items())):
        key = f"encoder_{index:06d}"
        arrays[key] = tensor.detach().cpu().contiguous().numpy()
        encoder_keys[name] = key
    head_keys: dict[str, dict[str, str]] = {}
    router_keys: dict[str, dict[str, str]] = {}
    router_lambda: dict[str, float] = {}
    for class_id in seen_classes:
        class_key = str(class_id)
        head_keys[class_key] = {}
        for index, (name, tensor) in enumerate(
            sorted(agent.heads[class_key].state_dict().items())
        ):
            key = f"head_{class_id}_{index:06d}"
            arrays[key] = tensor.detach().cpu().contiguous().numpy()
            head_keys[class_key][name] = key
        router = agent.routers.cap[class_id]
        centroid_key = f"router_{class_id}_centroids"
        count_key = f"router_{class_id}_counts"
        arrays[centroid_key] = np.asarray(router.centroids, dtype=np.float32)
        arrays[count_key] = np.asarray(router.counts)
        router_keys[class_key] = {
            "centroids": centroid_key,
            "counts": count_key,
        }
        router_lambda[class_key] = float(router.lam)
    schema = {
        "encoder": encoder_keys,
        "heads": head_keys,
        "normalization": {
            "mean": "normalization_mean",
            "scale": "normalization_scale",
        },
        "cap3000_router": router_keys,
        "cap3000_router_lambda": router_lambda,
    }
    return arrays, schema


def _raw_probe_rows(
    agent: "StreamingOFRA", probe_contract: dict, seen_classes: list[int]
) -> tuple[np.ndarray, list[dict]]:
    wanted = set(seen_classes)
    records = [
        record
        for record in probe_contract["official_test"]["samples"]
        if int(record["class_id"]) in wanted
    ]
    rows: list[np.ndarray] = []
    for record in records:
        class_id = int(record["class_id"])
        source = agent.test[class_id]
        shard_id = int(record["shard_ordinal"])
        local_row = int(record["local_row"])
        global_index = int(source.offsets[shard_id] + local_row)
        value = source.take(np.asarray([global_index], dtype=np.int64))
        if array_sha256(value) != record["feature_row_sha256"]:
            raise RuntimeError("monitor probe row no longer matches its frozen hash")
        rows.append(value)
    if not rows:
        raise RuntimeError("checkpoint monitor has no seen-class probe rows")
    return np.vstack(rows).astype(np.float32, copy=False), records


@torch.no_grad()
def persist_checkpoint(
    agent: "StreamingOFRA",
    *,
    checkpoint: int,
    seen_classes: list[int],
    output_base: Path,
    probe_contract: dict,
) -> dict[str, object]:
    """Persist one inference-only checkpoint and its fixed-probe score trace."""

    if probe_contract.get("canonical_sha256") != canonical_sha256(
        {key: value for key, value in probe_contract.items() if key != "canonical_sha256"}
    ):
        raise RuntimeError("monitor probe contract has an invalid canonical hash")
    seed_relative = Path("monitoring") / f"seed_{agent.seed}"
    seed_dir = output_base / seed_relative
    probe_path = seed_dir / "probe_manifest.json"
    _atomic_write_json(probe_path, probe_contract)

    checkpoint_relative = seed_relative / f"checkpoint_{checkpoint:03d}"
    checkpoint_dir = output_base / checkpoint_relative
    state_path = checkpoint_dir / "inference_state.npz"
    scores_path = checkpoint_dir / "probe_scores.npz"
    metadata_path = checkpoint_dir / "checkpoint_manifest.json"

    state_arrays, state_schema = _state_arrays(agent, seen_classes)
    _atomic_write_deterministic_npz(state_path, state_arrays)

    raw, records = _raw_probe_rows(agent, probe_contract, seen_classes)
    normalized = agent.stats.transform(raw)
    embeddings = agent._embed(normalized)
    head, _ = agent._head_scores(embeddings, seen_classes)
    router = agent.routers.scores(embeddings, seen_classes, "cap3000")
    joint = head + np.float32(0.5) * router
    predicted_columns = joint.argmax(axis=1)
    class_axis = np.asarray(seen_classes, dtype=np.int64)
    predicted = class_axis[predicted_columns]
    score_arrays = {
        "sample_id_sha256": np.asarray(
            [str(record["sample_id_sha256"]).encode("ascii") for record in records],
            dtype="S64",
        ),
        "true_class_id": np.asarray(
            [int(record["class_id"]) for record in records], dtype=np.int64
        ),
        "shard_ordinal": np.asarray(
            [int(record["shard_ordinal"]) for record in records], dtype=np.int64
        ),
        "local_row": np.asarray(
            [int(record["local_row"]) for record in records], dtype=np.int64
        ),
        "class_axis": class_axis,
        "head_scores": np.asarray(head, dtype=np.float32),
        "router_z_scores": np.asarray(router, dtype=np.float32),
        "joint_scores": np.asarray(joint, dtype=np.float32),
        "predicted_class_id": np.asarray(predicted, dtype=np.int64),
    }
    _atomic_write_deterministic_npz(scores_path, score_arrays)

    config = agent.config
    metadata: dict[str, object] = {
        "schema_version": MONITOR_SCHEMA_VERSION,
        "dataset": agent.manifest.dataset,
        "seed": int(agent.seed),
        "checkpoint": int(checkpoint),
        "seen_classes": [int(value) for value in seen_classes],
        "feature_dim": int(agent.manifest.feature_dim),
        "architecture": {
            "encoder_type": config.encoder_type,
            "d_model": int(config.d_model),
            "n_layers": int(config.n_layers),
            "ft_heads": int(config.ft_heads),
            "ft_dim_head": int(config.ft_dim_head),
            "ft_attn_dropout": float(config.ft_attn_dropout),
            "ft_ff_dropout": float(config.ft_ff_dropout),
            "ft_num_residual_streams": int(config.ft_num_residual_streams),
            "lora_rank": int(config.lora_rank),
            "lora_alpha": float(config.lora_alpha),
        },
        "training_device": str(agent.device),
        "reconstruction_scope": "official/joint_cap3000 inference",
        "raw_feature_rows_persisted": False,
        "state_schema": state_schema,
        "inference_state_file": state_path.name,
        "inference_state_sha256": sha256_file(state_path),
        "probe_scores_file": scores_path.name,
        "probe_scores_sha256": sha256_file(scores_path),
        "probe_manifest_relative_path": probe_path.relative_to(output_base).as_posix(),
        "probe_manifest_file_sha256": sha256_file(probe_path),
        "probe_contract_sha256": probe_contract["canonical_sha256"],
        "score_rows": int(len(raw)),
        "joint_formula": "head_score + 0.5 * router_z_score",
        "router_standardization": ROUTER_STANDARDIZATION,
        "trained_state_sha256": agent.state_sha256(),
    }
    metadata["canonical_sha256"] = canonical_sha256(metadata)
    _atomic_write_json(metadata_path, metadata)
    validate_checkpoint_manifest(metadata_path)
    return {
        "checkpoint": int(checkpoint),
        "seen_classes": [int(value) for value in seen_classes],
        "checkpoint_manifest_relative_path": metadata_path.relative_to(
            output_base
        ).as_posix(),
        "checkpoint_manifest_file_sha256": sha256_file(metadata_path),
        "checkpoint_manifest_canonical_sha256": metadata["canonical_sha256"],
        "inference_state_sha256": metadata["inference_state_sha256"],
        "probe_scores_sha256": metadata["probe_scores_sha256"],
    }


def _load_npz_copy(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise RuntimeError(f"cannot validate monitor NPZ {path}: {error}") from error


def validate_checkpoint_manifest(path: Path) -> dict[str, object]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot validate checkpoint manifest {path}: {error}") from error
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise RuntimeError(f"invalid checkpoint manifest schema: {path}")
    stored = metadata.get("canonical_sha256")
    actual = canonical_sha256(
        {key: value for key, value in metadata.items() if key != "canonical_sha256"}
    )
    if stored != actual:
        raise RuntimeError(f"checkpoint manifest canonical hash mismatch: {path}")
    state_path = path.parent / str(metadata.get("inference_state_file"))
    scores_path = path.parent / str(metadata.get("probe_scores_file"))
    if sha256_file(state_path) != metadata.get("inference_state_sha256"):
        raise RuntimeError(f"checkpoint inference state SHA-256 mismatch: {state_path}")
    if sha256_file(scores_path) != metadata.get("probe_scores_sha256"):
        raise RuntimeError(f"checkpoint probe score SHA-256 mismatch: {scores_path}")
    state = _load_npz_copy(state_path)
    scores = _load_npz_copy(scores_path)
    schema = metadata.get("state_schema")
    if not isinstance(schema, dict):
        raise RuntimeError(f"checkpoint lacks state schema: {path}")
    referenced = {
        *schema["encoder"].values(),
        *(
            value
            for mapping in schema["heads"].values()
            for value in mapping.values()
        ),
        *schema["normalization"].values(),
        *(
            value
            for mapping in schema["cap3000_router"].values()
            for value in mapping.values()
        ),
    }
    if set(state) != referenced:
        raise RuntimeError(f"checkpoint state arrays disagree with schema: {path}")
    required_scores = {
        "sample_id_sha256",
        "true_class_id",
        "shard_ordinal",
        "local_row",
        "class_axis",
        "head_scores",
        "router_z_scores",
        "joint_scores",
        "predicted_class_id",
    }
    if set(scores) != required_scores:
        raise RuntimeError(f"checkpoint probe score registry mismatch: {path}")
    seen = np.asarray(metadata.get("seen_classes"), dtype=np.int64)
    rows = int(metadata.get("score_rows", -1))
    width = len(seen)
    if (
        not np.array_equal(scores["class_axis"], seen)
        or scores["head_scores"].shape != (rows, width)
        or scores["router_z_scores"].shape != (rows, width)
        or scores["joint_scores"].shape != (rows, width)
        or scores["sample_id_sha256"].shape != (rows,)
        or scores["true_class_id"].shape != (rows,)
        or scores["predicted_class_id"].shape != (rows,)
    ):
        raise RuntimeError(f"checkpoint probe score shape/class axis mismatch: {path}")
    expected_joint = scores["head_scores"] + np.float32(0.5) * scores[
        "router_z_scores"
    ]
    expected_predicted = seen[expected_joint.argmax(axis=1)]
    if not np.array_equal(scores["joint_scores"], expected_joint) or not np.array_equal(
        scores["predicted_class_id"], expected_predicted
    ):
        raise RuntimeError(f"checkpoint joint-score/prediction formula mismatch: {path}")
    if metadata.get("router_standardization") != ROUTER_STANDARDIZATION:
        raise RuntimeError(f"checkpoint router standardization contract mismatch: {path}")
    return metadata


def validate_monitoring_result(
    result: dict,
    *,
    output_base: Path,
    expected_protocol: dict[str, object],
) -> None:
    record = result.get("monitoring")
    enabled = bool(expected_protocol.get("enabled"))
    if not isinstance(record, dict) or record.get("enabled") is not enabled:
        raise RuntimeError("result monitoring state differs from protocol")
    if not enabled:
        if record != {"enabled": False}:
            raise RuntimeError("disabled monitoring result contains artifacts")
        return
    probe = record.get("probe_manifest")
    checkpoints = record.get("checkpoints")
    if not isinstance(probe, dict) or not isinstance(checkpoints, list):
        raise RuntimeError("enabled result lacks monitoring artifacts")
    probe_path = output_base / str(probe.get("relative_path"))
    if sha256_file(probe_path) != probe.get("file_sha256"):
        raise RuntimeError(f"monitor probe manifest SHA-256 mismatch: {probe_path}")
    probe_value = json.loads(probe_path.read_text(encoding="utf-8"))
    expected_hash = expected_protocol.get("probe_contract_sha256")
    if (
        probe_value.get("canonical_sha256") != expected_hash
        or probe.get("canonical_sha256") != expected_hash
    ):
        raise RuntimeError("monitor probe contract differs from protocol")
    if len(checkpoints) != len(result.get("checkpoints", [])):
        raise RuntimeError("monitor checkpoint coverage mismatch")
    for index, checkpoint in enumerate(checkpoints):
        if checkpoint.get("checkpoint") != index:
            raise RuntimeError("monitor checkpoint order mismatch")
        path = output_base / str(checkpoint.get("checkpoint_manifest_relative_path"))
        if sha256_file(path) != checkpoint.get("checkpoint_manifest_file_sha256"):
            raise RuntimeError(f"monitor checkpoint manifest SHA-256 mismatch: {path}")
        metadata = validate_checkpoint_manifest(path)
        if (
            metadata.get("canonical_sha256")
            != checkpoint.get("checkpoint_manifest_canonical_sha256")
            or metadata.get("seen_classes") != result["checkpoints"][index]["seen_classes"]
            or metadata.get("probe_contract_sha256") != expected_hash
        ):
            raise RuntimeError("monitor checkpoint metadata/result binding mismatch")


@dataclass
class ReloadedCheckpoint:
    metadata: dict[str, object]
    mean: np.ndarray
    scale: np.ndarray
    encoder: torch.nn.Module
    heads: dict[int, FamilyHead]
    router: DualRouter
    device: torch.device

    @torch.no_grad()
    def score(self, raw: np.ndarray) -> dict[str, np.ndarray]:
        values = np.asarray(raw, dtype=np.float64)
        normalized = ((values - self.mean) / self.scale).astype(np.float32)
        architecture = self.metadata["architecture"]
        batch_size = 4096
        embeddings: list[np.ndarray] = []
        self.encoder.eval()
        for start in range(0, len(normalized), batch_size):
            tensor = torch.from_numpy(
                np.ascontiguousarray(normalized[start : start + batch_size])
            ).to(self.device)
            embeddings.append(self.encoder(tensor).cpu().numpy().astype(np.float32))
        embedding = np.vstack(embeddings)
        seen = [int(value) for value in self.metadata["seen_classes"]]
        tensor = torch.from_numpy(np.ascontiguousarray(embedding)).to(self.device)
        head = np.empty((len(raw), len(seen)), dtype=np.float32)
        for column, class_id in enumerate(seen):
            self.heads[class_id].eval()
            logits = self.heads[class_id](tensor)
            head[:, column] = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        router = self.router.scores(embedding, seen, "cap3000")
        joint = head + np.float32(0.5) * router
        axis = np.asarray(seen, dtype=np.int64)
        return {
            "class_axis": axis,
            "head_scores": head,
            "router_z_scores": router,
            "joint_scores": joint,
            "predicted_class_id": axis[joint.argmax(axis=1)],
        }


def load_checkpoint(path: Path, *, device: str = "cpu") -> ReloadedCheckpoint:
    metadata = validate_checkpoint_manifest(path)
    state = _load_npz_copy(path.parent / str(metadata["inference_state_file"]))
    architecture = metadata["architecture"]
    torch_device = torch.device(device)
    encoder = build_encoder(
        encoder_type=str(architecture["encoder_type"]),
        n_features=int(metadata["feature_dim"]),
        d_model=int(architecture["d_model"]),
        n_layers=int(architecture["n_layers"]),
        ft_heads=int(architecture["ft_heads"]),
        ft_dim_head=int(architecture["ft_dim_head"]),
        ft_attn_dropout=float(architecture["ft_attn_dropout"]),
        ft_ff_dropout=float(architecture["ft_ff_dropout"]),
        ft_num_residual_streams=int(architecture["ft_num_residual_streams"]),
    ).to(torch_device)
    schema = metadata["state_schema"]

    def tensors(mapping: dict[str, str]) -> dict[str, torch.Tensor]:
        return {
            name: torch.from_numpy(np.array(state[key], copy=True)).to(torch_device)
            for name, key in mapping.items()
        }

    encoder.load_state_dict(tensors(schema["encoder"]), strict=True)
    encoder.eval()
    heads: dict[int, FamilyHead] = {}
    router = DualRouter()
    for value in metadata["seen_classes"]:
        class_id = int(value)
        class_key = str(class_id)
        head = FamilyHead(
            d_model=int(architecture["d_model"]),
            rank=int(architecture["lora_rank"]),
            alpha=float(architecture["lora_alpha"]),
        ).to(torch_device)
        head.load_state_dict(tensors(schema["heads"][class_key]), strict=True)
        head.eval()
        heads[class_id] = head
        router_keys = schema["cap3000_router"][class_key]
        router.cap[class_id] = FamilyRouterState(
            centroids=np.asarray(state[router_keys["centroids"]], dtype=np.float32),
            counts=np.asarray(state[router_keys["counts"]]),
            lam=float(schema["cap3000_router_lambda"][class_key]),
            stats={"reloaded_for_inference": True},
        )
    return ReloadedCheckpoint(
        metadata=metadata,
        mean=np.asarray(state[schema["normalization"]["mean"]], dtype=np.float64),
        scale=np.asarray(state[schema["normalization"]["scale"]], dtype=np.float64),
        encoder=encoder,
        heads=heads,
        router=router,
        device=torch_device,
    )

