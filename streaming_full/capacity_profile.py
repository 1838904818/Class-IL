"""Bounded, non-reportable capacity profiling for the current training path.

The profile uses the same Task-0 batch construction, frozen normalisation,
FT-Transformer encoder and supervised-pretraining optimizer as the formal
runner, but stops after a fixed number of initial pretraining batches.  It is
only for sizing a Slurm request and estimating recovery boundaries: it does
not emit accuracy, forgetting, SHAP, ETG, or any reportable experimental
result.

The module deliberately does not import W&B and is intended to run with
``WANDB_MODE=disabled``.  Its output contains only aggregate timing, memory,
input fingerprints, and a non-secret runtime-environment fingerprint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import site
import subprocess
import sys
import time
from dataclasses import replace
from importlib import metadata
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

from .data import (
    BlockShuffleSampler,
    canonical_sha256,
    derived_seed,
    load_manifest,
    sha256_file,
)
from .runner import _load_config
from .validation import (
    IndexChunkCursor,
    RunConfig,
    StreamingOFRA,
    _resolve_device,
    _seed_process,
)


PROFILE_SCHEMA_VERSION = "ofra_capacity_profile_v1"
PROFILE_KIND = "bounded_task0_pretrain_throughput_and_memory_only"


def _distribution_record(name: str) -> dict[str, object]:
    """Return a non-secret installed-distribution fingerprint.

    The wheel ``RECORD`` hash is not a full integrity verifier by itself, but
    it binds the package's declared file inventory and hashes alongside the
    runtime version.  The formal job can compare this record before training.
    """

    try:
        distribution = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return {"installed": False}
    record = distribution.read_text("RECORD")
    return {
        "installed": True,
        "version": distribution.version,
        "record_sha256": (
            hashlib.sha256(record.encode("utf-8")).hexdigest()
            if record is not None
            else None
        ),
    }


def runtime_environment_record(device: torch.device) -> dict[str, object]:
    """Build a deterministic, non-secret environment record in an allocation."""

    python_path = Path(sys.executable).resolve()
    torch_path = Path(torch.__file__).resolve()
    nvidia_smi: str | None = None
    if device.type == "cuda":
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            )
            nvidia_smi = completed.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            nvidia_smi = "unavailable"

    record: dict[str, object] = {
        "schema_version": "ofra_runtime_environment_v1",
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable_sha256": sha256_file(python_path),
            "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
            "user_site_enabled": bool(site.ENABLE_USER_SITE),
        },
        "packages": {
            name: _distribution_record(name)
            for name in ("torch", "tab-transformer-pytorch", "wandb")
        },
        "torch": {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "module_sha256": sha256_file(torch_path),
            "deterministic_algorithms_enabled": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "environment": {
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "device": {"type": device.type},
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        record["device"] = {
            "type": "cuda",
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": int(properties.total_memory),
            "capability": list(torch.cuda.get_device_capability(device)),
            "nvidia_smi": nvidia_smi,
        }
    record["canonical_sha256"] = canonical_sha256(record)
    return record


def _task0_batches(
    agent: StreamingOFRA,
    *,
    max_batches: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, int]]:
    """Yield the initial formal pretraining batches without changing the path.

    This is intentionally a bounded prefix of epoch 0.  The quota allocation,
    block sampler, class mixing and tensor conversion are copied verbatim from
    :meth:`StreamingOFRA.pretrain_encoder`; only the explicit stop condition is
    new.  The profile's timing therefore describes the same model/data path,
    not a synthetic tensor benchmark.
    """

    if max_batches <= 0:
        raise ValueError("max_batches must be positive")
    if agent.config.gradient_accumulation_steps != 1:
        raise ValueError(
            "capacity profile v1 supports gradient_accumulation_steps=1 only; "
            "the locked formal configuration uses 1"
        )

    task0 = agent.manifest.tasks[0]
    class_ids = [int(class_id) for class_id in task0]
    class_rows = np.asarray(
        [len(agent.train[class_id]) for class_id in class_ids], dtype=np.int64
    )
    total_rows = int(class_rows.sum())
    epoch = 0
    samplers: dict[int, BlockShuffleSampler] = {}
    cursors: dict[int, IndexChunkCursor] = {}
    for class_id, row_count in zip(class_ids, class_rows.tolist()):
        sampler_seed = derived_seed(
            agent.seed,
            agent.manifest.dataset,
            "pretrain_class_blocks",
            epoch,
            class_id,
        )
        blocks, _ = agent.train[class_id].index_blocks(
            agent.config.shuffle_block_rows,
            base_offset=0,
            group_base=0,
        )
        sampler = BlockShuffleSampler(
            blocks,
            population_rows=row_count,
            sample_rows=row_count,
            seed=sampler_seed,
            block_rows=agent.config.shuffle_block_rows,
        )
        samplers[class_id] = sampler
        cursors[class_id] = IndexChunkCursor(
            sampler.iter_chunks(agent.config.batch_size)
        )
    mix_seed = derived_seed(
        agent.seed, agent.manifest.dataset, "pretrain_batch_mix", epoch
    )
    mix_rng = np.random.default_rng(mix_seed)
    consumed = np.zeros(len(class_ids), dtype=np.int64)
    total = 0
    emitted = 0
    while total < total_rows and emitted < max_batches:
        batch_rows = min(agent.config.batch_size, total_rows - total)
        total_after = total + batch_rows
        ideal = class_rows.astype(np.float64) * (total_after / total_rows)
        deficit = ideal - consumed
        quota = np.floor(np.maximum(deficit, 0.0)).astype(np.int64)
        remaining_by_class = class_rows - consumed
        np.minimum(quota, remaining_by_class, out=quota)
        unassigned = int(batch_rows - quota.sum())
        while unassigned:
            available = remaining_by_class - quota > 0
            if not available.any():
                raise RuntimeError("pretrain quota allocation exhausted early")
            priority = deficit - quota
            priority[~available] = -np.inf
            selected = int(np.argmax(priority))
            quota[selected] += 1
            unassigned -= 1

        raw_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        for class_index, (class_id, count) in enumerate(
            zip(class_ids, quota.tolist())
        ):
            if not count:
                continue
            chunks = cursors[class_id].take(count)
            raw_parts.extend(agent.train[class_id].take(indices) for indices in chunks)
            label_parts.append(np.full(count, class_id, dtype=np.int64))
            consumed[class_index] += count
        raw = np.vstack(raw_parts)
        labels = np.concatenate(label_parts)
        batch_order = mix_rng.permutation(batch_rows)
        values = torch.from_numpy(agent.stats.transform(raw[batch_order])).to(
            agent.device
        )
        target = torch.from_numpy(labels[batch_order]).to(agent.device)
        yield values, target, int(len(target))
        total = total_after
        emitted += 1


def _one_training_step(
    agent: StreamingOFRA,
    temporary_head: torch.nn.Linear,
    optimizer: torch.optim.Optimizer,
    values: torch.Tensor,
    target: torch.Tensor,
) -> float:
    logits = temporary_head(agent.encoder(values))
    loss = F.cross_entropy(logits, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def run_capacity_profile(
    manifest_path: str | Path,
    *,
    config: RunConfig,
    seed: int,
    warmup_batches: int,
    timed_batches: int,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Run one bounded profile and optionally persist its aggregate record."""

    config.validate()
    if warmup_batches < 0:
        raise ValueError("warmup_batches must be non-negative")
    if timed_batches <= 0:
        raise ValueError("timed_batches must be positive")
    profile_started = time.perf_counter()
    _seed_process(int(seed), config.deterministic)
    manifest_started = time.perf_counter()
    manifest = load_manifest(
        manifest_path, verify_hashes=config.verify_shard_hashes
    )
    manifest_validation_seconds = time.perf_counter() - manifest_started
    device = _resolve_device(config.device)
    total_profile_batches = warmup_batches + timed_batches
    initialization_started = time.perf_counter()
    agent = StreamingOFRA(manifest, config, int(seed), device)
    initialization_seconds = time.perf_counter() - initialization_started
    try:
        task0_rows = sum(len(agent.train[class_id]) for class_id in manifest.tasks[0])
        maximum_batches = int(
            (task0_rows + config.batch_size - 1) // config.batch_size
        )
        if total_profile_batches > maximum_batches:
            raise ValueError(
                "requested warmup plus timed batches exceeds one formal Task-0 epoch: "
                f"{total_profile_batches} > {maximum_batches}"
            )
        normalization = agent.fit_task0_stats()
        normalization_seconds = float(agent.timing["normalization_seconds"])
        temporary_head = torch.nn.Linear(
            config.d_model, len(manifest.classes)
        ).to(device)
        optimizer = torch.optim.Adam(
            [*agent.encoder.parameters(), *temporary_head.parameters()],
            lr=config.learning_rate,
        )
        agent.encoder.train()
        temporary_head.train()
        batches = _task0_batches(agent, max_batches=total_profile_batches)
        for _ in range(warmup_batches):
            values, target, _ = next(batches)
            _one_training_step(agent, temporary_head, optimizer, values, target)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        timed_started = time.perf_counter()
        timed_rows = 0
        for _ in range(timed_batches):
            values, target, rows = next(batches)
            _one_training_step(agent, temporary_head, optimizer, values, target)
            timed_rows += rows
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timed_seconds = time.perf_counter() - timed_started
        resources: dict[str, object]
        if device.type == "cuda":
            properties = torch.cuda.get_device_properties(device)
            resources = {
                "cuda_peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
                "cuda_peak_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                ),
                "cuda_total_memory_bytes": int(properties.total_memory),
                "cuda_peak_allocated_fraction": float(
                    torch.cuda.max_memory_allocated(device) / properties.total_memory
                ),
            }
        else:
            resources = {"device": "cpu"}
        record: dict[str, object] = {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "profile_kind": PROFILE_KIND,
            "non_reportable": True,
            "purpose": (
                "Slurm capacity measurement and recovery-boundary planning only; "
                "not a model-performance or methodology experiment"
            ),
            "input": {
                "dataset": manifest.dataset,
                "manifest_sha256": sha256_file(Path(manifest_path)),
                "task_index": 0,
                "task_classes": [int(value) for value in manifest.tasks[0]],
                "task0_train_rows": int(task0_rows),
                "feature_dim": int(manifest.feature_dim),
                "seed": int(seed),
                "config": {
                    "encoder_type": config.encoder_type,
                    "d_model": config.d_model,
                    "n_layers": config.n_layers,
                    "ft_heads": config.ft_heads,
                    "ft_dim_head": config.ft_dim_head,
                    "batch_size": config.batch_size,
                    "gradient_accumulation_steps": config.gradient_accumulation_steps,
                    "learning_rate": config.learning_rate,
                    "deterministic": config.deterministic,
                },
                "sampling": {
                    "source": "bounded_prefix_of_formal_task0_pretrain_epoch_0",
                    "warmup_batches": int(warmup_batches),
                    "timed_batches": int(timed_batches),
                },
            },
            "normalization": normalization,
            "phases": {
                "manifest_validation_seconds": float(manifest_validation_seconds),
                "model_initialization_seconds": float(initialization_seconds),
                "task0_normalization_seconds": normalization_seconds,
            },
            "warmup": {
                "batches": int(warmup_batches),
            },
            "measurement": {
                "timed_batches": int(timed_batches),
                "timed_rows": int(timed_rows),
                "timed_seconds": float(timed_seconds),
                "training_rows_per_second": (
                    float(timed_rows / timed_seconds) if timed_seconds > 0.0 else None
                ),
            },
            "resources": resources,
            "environment": runtime_environment_record(device),
        }
        record["phases"]["profile_total_seconds"] = float(
            time.perf_counter() - profile_started
        )
        record["canonical_sha256"] = canonical_sha256(record)
        if output_path is not None:
            _atomic_json(Path(output_path), record)
        return record
    finally:
        agent.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--warmup-batches", type=int, default=25)
    parser.add_argument("--timed-batches", type=int, default=2500)
    parser.add_argument("--device", help="Override config device, for example cuda:0.")
    parser.add_argument(
        "--skip-shard-hash-verification",
        action="store_true",
        help="Not valid for formal evidence; retained only for controlled diagnostics.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    values = _load_config(args.config_json)
    if args.device:
        values["device"] = args.device
    if args.skip_shard_hash_verification:
        values["verify_shard_hashes"] = False
    config = RunConfig(**values)
    result = run_capacity_profile(
        args.manifest,
        config=config,
        seed=args.seed,
        warmup_batches=args.warmup_batches,
        timed_batches=args.timed_batches,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "canonical_sha256": result["canonical_sha256"],
                "training_rows_per_second": result["measurement"][
                    "training_rows_per_second"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
