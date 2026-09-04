from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .data import canonical_sha256, sha256_file
from .routers import DualRouter, FamilyRouterState

if TYPE_CHECKING:
    from .validation import StreamingOFRA


RECOVERY_SCHEMA_VERSION = 1


class RecoveryPause(RuntimeError):
    """Clean, hash-bound pause at an epoch or task boundary."""

    def __init__(self, record: dict[str, object]):
        super().__init__("formal run paused at a validated recovery boundary")
        self.record = record


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def capture_rng_state() -> dict[str, object]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": str(numpy_state[0]),
            "state": torch.from_numpy(np.array(numpy_state[1], copy=True)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.random.get_rng_state().cpu(),
        "torch_cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def restore_rng_state(record: dict[str, object]) -> None:
    random.setstate(record["python"])
    numpy_record = record["numpy"]
    assert isinstance(numpy_record, dict)
    numpy_tensor = numpy_record["state"]
    assert isinstance(numpy_tensor, torch.Tensor)
    np.random.set_state(
        (
            str(numpy_record["algorithm"]),
            numpy_tensor.cpu().numpy().astype(np.uint32, copy=True),
            int(numpy_record["position"]),
            int(numpy_record["has_gauss"]),
            float(numpy_record["cached_gaussian"]),
        )
    )
    torch_cpu = record["torch_cpu"]
    assert isinstance(torch_cpu, torch.Tensor)
    torch.random.set_rng_state(torch_cpu.cpu())
    cuda_states = record.get("torch_cuda", [])
    if torch.cuda.is_available() and cuda_states:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])


def _router_payload(router: DualRouter) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, states in (("cap", router.cap), ("uncapped", router.uncapped)):
        result[name] = {
            str(class_id): {
                "centroids": torch.from_numpy(np.array(state.centroids, copy=True)),
                "counts": torch.from_numpy(np.array(state.counts, copy=True)),
                "lambda": float(state.lam),
                "stats": state.stats,
            }
            for class_id, state in sorted(states.items())
        }
    return result


def _restore_router(payload: dict[str, object]) -> DualRouter:
    router = DualRouter()
    for name, destination in (("cap", router.cap), ("uncapped", router.uncapped)):
        raw_states = payload.get(name, {})
        if not isinstance(raw_states, dict):
            raise RuntimeError("recovery router registry is invalid")
        for raw_class_id, raw_state in raw_states.items():
            if not isinstance(raw_state, dict):
                raise RuntimeError("recovery router state is invalid")
            centroids = raw_state["centroids"]
            counts = raw_state["counts"]
            if not isinstance(centroids, torch.Tensor) or not isinstance(
                counts, torch.Tensor
            ):
                raise RuntimeError("recovery router arrays are invalid")
            destination[int(raw_class_id)] = FamilyRouterState(
                centroids=centroids.cpu().numpy().astype(np.float32, copy=True),
                counts=counts.cpu().numpy().astype(np.int64, copy=True),
                lam=float(raw_state["lambda"]),
                stats=dict(raw_state["stats"]),
            )
    return router


def export_agent_state(agent: "StreamingOFRA") -> dict[str, object]:
    return {
        "normalization": {
            "count": int(agent.stats.count),
            "mean": torch.from_numpy(np.array(agent.stats.mean, copy=True)),
            "m2": torch.from_numpy(np.array(agent.stats.m2, copy=True)),
            "frozen": bool(agent.stats.frozen),
            "source_classes": list(agent.stats.source_classes),
        },
        "encoder_state": _cpu_tree(agent.encoder.state_dict()),
        "encoder_training": bool(agent.encoder.training),
        "encoder_requires_grad": [
            bool(parameter.requires_grad) for parameter in agent.encoder.parameters()
        ],
        "heads": {
            key: {
                "state": _cpu_tree(module.state_dict()),
                "training": bool(module.training),
                "requires_grad": [
                    bool(parameter.requires_grad) for parameter in module.parameters()
                ],
            }
            for key, module in sorted(agent.heads.items(), key=lambda item: int(item[0]))
        },
        "exemplars": {
            str(class_id): torch.from_numpy(np.array(values, copy=True))
            for class_id, values in sorted(agent.exemplars.items())
        },
        "exemplar_records": agent.exemplar_records,
        "training_exposure_records": agent.training_exposure_records,
        "training_prior_records": agent.training_prior_records,
        "routers": _router_payload(agent.routers),
        "router_records": agent.router_records,
        "timing": agent.timing,
    }


def restore_agent_state(agent: "StreamingOFRA", payload: dict[str, object]) -> None:
    normalization = payload.get("normalization")
    if not isinstance(normalization, dict):
        raise RuntimeError("recovery checkpoint lacks normalization state")
    mean = normalization.get("mean")
    m2 = normalization.get("m2")
    if not isinstance(mean, torch.Tensor) or not isinstance(m2, torch.Tensor):
        raise RuntimeError("recovery normalization arrays are invalid")
    agent.stats.count = int(normalization["count"])
    agent.stats.mean = mean.cpu().numpy().astype(np.float64, copy=True)
    agent.stats.m2 = m2.cpu().numpy().astype(np.float64, copy=True)
    agent.stats.frozen = bool(normalization["frozen"])
    agent.stats.source_classes = [int(value) for value in normalization["source_classes"]]

    encoder_state = payload.get("encoder_state")
    if not isinstance(encoder_state, dict):
        raise RuntimeError("recovery checkpoint lacks encoder state")
    agent.encoder.load_state_dict(encoder_state, strict=True)
    encoder_requires_grad = payload.get("encoder_requires_grad")
    if not isinstance(encoder_requires_grad, list) or len(encoder_requires_grad) != len(
        list(agent.encoder.parameters())
    ):
        raise RuntimeError("recovery encoder gradient registry is invalid")
    for parameter, required in zip(agent.encoder.parameters(), encoder_requires_grad):
        parameter.requires_grad = bool(required)
    agent.encoder.train(bool(payload.get("encoder_training")))

    raw_heads = payload.get("heads", {})
    if not isinstance(raw_heads, dict):
        raise RuntimeError("recovery head registry is invalid")
    for key, record in sorted(raw_heads.items(), key=lambda item: int(item[0])):
        if not isinstance(record, dict):
            raise RuntimeError("recovery head record is invalid")
        head = agent._new_head(int(key))
        head.load_state_dict(record["state"], strict=True)
        required_values = record.get("requires_grad")
        if not isinstance(required_values, list) or len(required_values) != len(
            list(head.parameters())
        ):
            raise RuntimeError("recovery head gradient registry is invalid")
        for parameter, required in zip(head.parameters(), required_values):
            parameter.requires_grad = bool(required)
        head.train(bool(record.get("training")))

    raw_exemplars = payload.get("exemplars", {})
    if not isinstance(raw_exemplars, dict):
        raise RuntimeError("recovery exemplar registry is invalid")
    agent.exemplars = {
        int(class_id): values.cpu().numpy().astype(np.float32, copy=True)
        for class_id, values in raw_exemplars.items()
        if isinstance(values, torch.Tensor)
    }
    if len(agent.exemplars) != len(raw_exemplars):
        raise RuntimeError("recovery exemplar arrays are invalid")
    agent.exemplar_records = dict(payload.get("exemplar_records", {}))
    agent.training_exposure_records = dict(
        payload.get("training_exposure_records", {})
    )
    agent.training_prior_records = dict(payload.get("training_prior_records", {}))
    raw_router = payload.get("routers")
    if not isinstance(raw_router, dict):
        raise RuntimeError("recovery checkpoint lacks router state")
    agent.routers = _restore_router(raw_router)
    agent.router_records = dict(payload.get("router_records", {}))
    agent.timing = dict(payload.get("timing", {}))


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(value, temporary)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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


@dataclass
class RecoveryController:
    directory: Path
    dataset: str
    seed: int
    protocol_sha256: str
    deadline_unix: float | None = None
    stop_margin_seconds: float = 1800.0
    minimum_next_unit_seconds: float = 0.0
    pause_after_checkpoints: int | None = None

    def __post_init__(self) -> None:
        self.directory = Path(self.directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.payload_path = self.directory / "recovery_state.pt"
        self.manifest_path = self.directory / "recovery_manifest.json"
        self.checkpoint_count = 0

    def load(self) -> dict[str, object] | None:
        if not self.payload_path.exists() and not self.manifest_path.exists():
            return None
        if not self.payload_path.is_file() or not self.manifest_path.is_file():
            raise RuntimeError("recovery checkpoint pair is incomplete")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise RuntimeError("invalid recovery manifest schema")
        stored_canonical = manifest.get("canonical_sha256")
        actual_canonical = canonical_sha256(
            {key: value for key, value in manifest.items() if key != "canonical_sha256"}
        )
        if stored_canonical != actual_canonical:
            raise RuntimeError("recovery manifest canonical SHA-256 mismatch")
        if sha256_file(self.payload_path) != manifest.get("payload_sha256"):
            raise RuntimeError("recovery payload SHA-256 mismatch")
        identity = {
            "dataset": self.dataset,
            "seed": int(self.seed),
            "protocol_sha256": self.protocol_sha256,
        }
        if any(manifest.get(key) != value for key, value in identity.items()):
            raise RuntimeError("recovery checkpoint identity differs from requested run")
        payload = torch.load(self.payload_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise RuntimeError("invalid recovery payload schema")
        if any(payload.get(key) != value for key, value in identity.items()):
            raise RuntimeError("recovery payload identity differs from requested run")
        self.checkpoint_count = int(manifest.get("sequence", 0))
        return payload

    def save(
        self,
        *,
        agent: "StreamingOFRA",
        run_state: dict[str, object],
        phase_state: dict[str, object],
        unit_seconds: float,
    ) -> dict[str, object]:
        self.checkpoint_count += 1
        payload: dict[str, object] = {
            "schema_version": RECOVERY_SCHEMA_VERSION,
            "dataset": self.dataset,
            "seed": int(self.seed),
            "protocol_sha256": self.protocol_sha256,
            "sequence": int(self.checkpoint_count),
            "agent": export_agent_state(agent),
            "run_state": run_state,
            "phase_state": _cpu_tree(phase_state),
            "rng_state": capture_rng_state(),
        }
        _atomic_torch_save(self.payload_path, payload)
        manifest: dict[str, object] = {
            "schema_version": RECOVERY_SCHEMA_VERSION,
            "dataset": self.dataset,
            "seed": int(self.seed),
            "protocol_sha256": self.protocol_sha256,
            "sequence": int(self.checkpoint_count),
            "stage": str(phase_state.get("stage")),
            "unit_seconds": float(unit_seconds),
            "payload_file": self.payload_path.name,
            "payload_sha256": sha256_file(self.payload_path),
            "saved_unix": float(time.time()),
        }
        manifest["canonical_sha256"] = canonical_sha256(manifest)
        _atomic_json(self.manifest_path, manifest)
        return manifest

    def should_pause(self, *, unit_seconds: float) -> bool:
        if (
            self.pause_after_checkpoints is not None
            and self.checkpoint_count >= self.pause_after_checkpoints
        ):
            return True
        if self.deadline_unix is None:
            return False
        next_unit = max(
            float(self.minimum_next_unit_seconds),
            float(unit_seconds) * 1.25,
        )
        return time.time() + self.stop_margin_seconds + next_unit >= self.deadline_unix

    def checkpoint_and_maybe_pause(
        self,
        *,
        agent: "StreamingOFRA",
        run_state: dict[str, object],
        phase_state: dict[str, object],
        unit_seconds: float,
    ) -> None:
        manifest = self.save(
            agent=agent,
            run_state=run_state,
            phase_state=phase_state,
            unit_seconds=unit_seconds,
        )
        if self.should_pause(unit_seconds=unit_seconds):
            agent.close()
            raise RecoveryPause(manifest)
