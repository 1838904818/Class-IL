from __future__ import annotations

import json
import hashlib
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import (
    BlockShuffleSampler,
    ClassShards,
    DatasetManifest,
    EvaluationView,
    FrozenTask0Stats,
    GENERIC_METRIC_PROFILE,
    MatrixReservoir,
    NIDS_METRIC_PROFILE,
    IndexBlock,
    array_sha256,
    canonical_sha256,
    derived_seed,
    dataset_logical_fingerprints,
    load_evaluation_view,
    load_manifest,
    sha256_file,
)
from .models import FamilyHead, MLPEncoder, focal_loss
from .routers import (
    DualRouter,
    OnePassBudgetRouter,
    cap_state_from_reservoir,
    matched_cap_state,
    uncapped_state,
)


RESULT_SCHEMA_VERSION = 2
EXPOSURE_PRIOR_FORMULA_REGISTRY = {
    "algorithm": "binary_count_exposure_prior_offset_v1",
    "margin_formula": "(positive_logit-negative_logit)-log(P_total/N_total)",
    "probability_formula": "sigmoid(corrected_margin)",
    "reference_prior": 0.5,
    "interpretation": "diagnostic exposure-prior score; not calibrated posterior",
}
OUTPUT_DIRECTORY_LOCK_CONTRACT = {
    "algorithm": "nonblocking_os_file_lock_v1",
    "file": ".streaming_full.lock",
    "scope": "protocol_results_summary_transaction",
    "windows": "msvcrt.LK_NBLCK one byte",
    "posix": "fcntl.LOCK_EX|LOCK_NB",
}


PRIMARY_ARMS: dict[str, dict[str, object]] = {
    "head_only": {"router": None, "head_weight": 1.0, "router_weight": 0.0},
    "router_only_cap3000": {"router": "cap3000", "head_weight": 0.0, "router_weight": 1.0},
    "joint_cap3000": {"router": "cap3000", "head_weight": 1.0, "router_weight": 0.5},
    "router_only_uncapped": {"router": "uncapped", "head_weight": 0.0, "router_weight": 1.0},
    "joint_uncapped": {"router": "uncapped", "head_weight": 1.0, "router_weight": 0.5},
}

DIAGNOSTIC_ARMS: dict[str, dict[str, object]] = {
    "head_only_exposure_prior_corrected": {
        "router": None,
        "head_weight": 1.0,
        "router_weight": 0.0,
        "head_score": "exposure_prior_corrected",
    },
    "joint_cap3000_exposure_prior_corrected": {
        "router": "cap3000",
        "head_weight": 1.0,
        "router_weight": 0.5,
        "head_score": "exposure_prior_corrected",
    },
    "joint_uncapped_exposure_prior_corrected": {
        "router": "uncapped",
        "head_weight": 1.0,
        "router_weight": 0.5,
        "head_score": "exposure_prior_corrected",
    },
}

ARMS: dict[str, dict[str, object]] = {**PRIMARY_ARMS, **DIAGNOSTIC_ARMS}


@dataclass(frozen=True)
class RunConfig:
    pretrain_epochs: int = 8
    epochs_per_task: int = 10
    batch_size: int = 256
    eval_batch_size: int = 4096
    shuffle_block_rows: int = 4096
    learning_rate: float = 1e-3
    d_model: int = 128
    n_layers: int = 2
    lora_rank: int = 8
    lora_alpha: float = 16.0
    loss_fn: str = "focal"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    minority_threshold: int = 1000
    negative_ratio: int = 4
    exemplar_capacity: int = 50
    exemplar_candidate_capacity: int = 5000
    router_cap_samples: int = 3000
    router_lambda_quantile: float = 0.30
    router_max_centroids: int = 32
    device: str = "auto"
    deterministic: bool = True
    verify_shard_hashes: bool = True
    verbose: bool = True

    def __post_init__(self) -> None:
        # JSON writers may emit integral-valued floats such as 16.0 as 16.
        # Normalize the declared floating-point fields so protocol and preflight
        # hashes do not depend on the producing JSON serializer.
        for name in (
            "learning_rate",
            "lora_alpha",
            "focal_gamma",
            "focal_alpha",
            "router_lambda_quantile",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a JSON number")
            object.__setattr__(self, name, float(value))

    def validate(self) -> None:
        integer_positive = {
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "shuffle_block_rows": self.shuffle_block_rows,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "lora_rank": self.lora_rank,
            "negative_ratio": self.negative_ratio,
            "router_cap_samples": self.router_cap_samples,
            "router_max_centroids": self.router_max_centroids,
        }
        for name, value in integer_positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.pretrain_epochs < 0 or self.epochs_per_task < 0:
            raise ValueError("epoch counts must be non-negative")
        if self.batch_size < self.negative_ratio + 1:
            raise ValueError("batch_size must be at least negative_ratio + 1")
        if self.exemplar_capacity < 0:
            raise ValueError("exemplar_capacity must be non-negative")
        if self.exemplar_candidate_capacity < self.exemplar_capacity:
            raise ValueError("exemplar_candidate_capacity must cover exemplar_capacity")
        if self.loss_fn not in {"focal", "ce"}:
            raise ValueError("loss_fn must be focal or ce")
        if not 0.0 < self.router_lambda_quantile < 1.0:
            raise ValueError("router_lambda_quantile must be in (0, 1)")


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _seed_process(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")


class OutputDirectoryLock:
    """Non-blocking process lock covering one output directory transaction."""

    def __init__(self, output: Path):
        self.path = output / ".streaming_full.lock"
        self.handle = None

    def __enter__(self) -> "OutputDirectoryLock":
        self.handle = self.path.open("a+b")
        self.handle.seek(0, os.SEEK_END)
        if self.handle.tell() == 0:
            self.handle.write(b"\0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError) as error:
            self.handle.close()
            self.handle = None
            raise RuntimeError(
                f"output directory is locked by another process: {self.path.parent}"
            ) from error
        return self

    def __exit__(self, *_: object) -> None:
        if self.handle is None:
            return
        try:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


def _rng_state_record() -> dict[str, object]:
    numpy_state = np.random.get_state()
    cuda_states = []
    if torch.cuda.is_available():
        for device, state in enumerate(torch.cuda.get_rng_state_all()):
            cuda_states.append(
                {
                    "device": device,
                    "sha256": array_sha256(state.cpu().numpy()),
                }
            )
    record: dict[str, object] = {
        "python_random_sha256": canonical_sha256(random.getstate()),
        "numpy_global": {
            "algorithm": numpy_state[0],
            "state_sha256": array_sha256(numpy_state[1]),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu_sha256": array_sha256(
            torch.random.get_rng_state().cpu().numpy()
        ),
        "torch_cuda": cuda_states,
    }
    record["canonical_sha256"] = canonical_sha256(record)
    return record


def _module_state_sha256(module: nn.Module) -> str:
    records = []
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        records.append(
            {
                "name": name,
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "sha256": array_sha256(value),
            }
        )
    return canonical_sha256(records)


def _update_index_digest(digest: "hashlib._Hash", indices: np.ndarray) -> None:
    values = np.ascontiguousarray(indices, dtype="<i8")
    digest.update(len(values).to_bytes(8, "little"))
    digest.update(memoryview(values).cast("B"))


def _exposure_counter_dtype(epochs: int) -> type[np.unsignedinteger]:
    """Choose the smallest unsigned counter that cannot overflow by epoch count."""
    from .exposure_preflight import exposure_counter_dtype

    return exposure_counter_dtype(epochs)


class LabeledShardPool:
    def __init__(self, sources: dict[int, ClassShards], class_ids: Sequence[int], feature_dim: int):
        self.sources = sources
        self.class_ids = tuple(int(value) for value in class_ids)
        self.feature_dim = int(feature_dim)
        self.offsets = np.cumsum(
            [0, *(len(sources[class_id]) for class_id in self.class_ids)],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def take(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        indices = np.asarray(indices, dtype=np.int64)
        values = np.empty((len(indices), self.feature_dim), dtype=np.float32)
        labels = np.empty(len(indices), dtype=np.int64)
        source_ids = np.searchsorted(self.offsets[1:], indices, side="right")
        for source_id in np.unique(source_ids):
            mask = source_ids == source_id
            class_id = self.class_ids[int(source_id)]
            local = indices[mask] - self.offsets[source_id]
            values[mask] = self.sources[class_id].take(local)
            labels[mask] = class_id
        return values, labels

    def index_blocks(self, block_rows: int) -> list[IndexBlock]:
        blocks: list[IndexBlock] = []
        group_base = 0
        for source_id, class_id in enumerate(self.class_ids):
            update, group_base = self.sources[class_id].index_blocks(
                block_rows,
                base_offset=int(self.offsets[source_id]),
                group_base=group_base,
            )
            blocks.extend(update)
        return blocks


class NormalizedNegativePool:
    """Current-task raw negatives plus already-normalized old exemplars."""

    def __init__(
        self,
        raw_sources: Sequence[ClassShards],
        normalized_arrays: Sequence[np.ndarray],
        stats: FrozenTask0Stats,
        feature_dim: int,
    ):
        self.raw_sources = tuple(raw_sources)
        self.normalized_arrays = tuple(np.asarray(value, dtype=np.float32) for value in normalized_arrays)
        self.stats = stats
        self.feature_dim = int(feature_dim)
        sizes = [*(len(value) for value in self.raw_sources), *(len(value) for value in self.normalized_arrays)]
        self.offsets = np.cumsum([0, *sizes], dtype=np.int64)

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def take(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        result = np.empty((len(indices), self.feature_dim), dtype=np.float32)
        source_ids = np.searchsorted(self.offsets[1:], indices, side="right")
        raw_count = len(self.raw_sources)
        for source_id in np.unique(source_ids):
            mask = source_ids == source_id
            local = indices[mask] - self.offsets[source_id]
            if source_id < raw_count:
                result[mask] = self.stats.transform(self.raw_sources[int(source_id)].take(local))
            else:
                result[mask] = self.normalized_arrays[int(source_id) - raw_count][local]
        return result

    def index_blocks(self, block_rows: int) -> list[IndexBlock]:
        blocks: list[IndexBlock] = []
        group_base = 0
        for source_id, source in enumerate(self.raw_sources):
            update, group_base = source.index_blocks(
                block_rows,
                base_offset=int(self.offsets[source_id]),
                group_base=group_base,
            )
            blocks.extend(update)
        array_base = len(self.raw_sources)
        for array_id, values in enumerate(self.normalized_arrays):
            source_id = array_base + array_id
            base = int(self.offsets[source_id])
            for start in range(0, len(values), block_rows):
                blocks.append(
                    IndexBlock(
                        group=group_base,
                        start=base + start,
                        rows=min(block_rows, len(values) - start),
                    )
                )
            group_base += 1
        return blocks


class IndexChunkCursor:
    def __init__(self, chunks: Iterable[np.ndarray]):
        self.chunks = iter(chunks)
        self.pending = np.empty(0, dtype=np.int64)
        self.consumed = 0

    def take(self, count: int) -> list[np.ndarray]:
        if count < 0:
            raise ValueError("cursor count must be non-negative")
        result: list[np.ndarray] = []
        remaining = count
        while remaining:
            if not len(self.pending):
                try:
                    self.pending = np.asarray(next(self.chunks), dtype=np.int64)
                except StopIteration as error:
                    raise RuntimeError("index stream ended before its declared row count") from error
            take = min(remaining, len(self.pending))
            result.append(self.pending[:take])
            self.pending = self.pending[take:]
            self.consumed += take
            remaining -= take
        return result


class StreamingOFRA:
    def __init__(
        self,
        manifest: DatasetManifest,
        config: RunConfig,
        seed: int,
        device: torch.device,
        evaluation_views: Sequence[EvaluationView] = (),
    ):
        self.manifest = manifest
        self.config = config
        self.seed = int(seed)
        self.device = device
        self.train = {
            record.class_id: ClassShards(record.train, manifest.feature_dim)
            for record in manifest.classes
        }
        self.test = {
            record.class_id: ClassShards(record.test, manifest.feature_dim)
            for record in manifest.classes
        }
        self.evaluation_views = {
            view.name: view for view in sorted(evaluation_views, key=lambda item: item.name)
        }
        for class_id in self.train:
            if len(self.train[class_id]) <= 0 or len(self.test[class_id]) <= 0:
                raise ValueError(f"class {class_id} must have non-empty train and test splits")
        self.stats = FrozenTask0Stats(manifest.feature_dim)
        self.encoder = MLPEncoder(
            manifest.feature_dim,
            d_model=config.d_model,
            n_layers=config.n_layers,
        ).to(device)
        self.heads = nn.ModuleDict()
        self.exemplars: dict[int, np.ndarray] = {}
        self.exemplar_records: dict[int, dict] = {}
        self.training_exposure_records: dict[int, dict] = {}
        self.training_prior_records: dict[int, dict] = {}
        self.routers = DualRouter()
        self.router_records: dict[int, dict] = {}
        self.timing: dict[str, object] = {
            "normalization_seconds": 0.0,
            "pretrain_seconds": 0.0,
            "head_training_seconds": {},
            "router_exemplar_seconds": {},
            "evaluation_seconds": {},
        }

    def close(self) -> None:
        for source in (*self.train.values(), *self.test.values()):
            source.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(message, flush=True)

    def fit_task0_stats(self) -> dict:
        started = time.perf_counter()
        task0 = self.manifest.tasks[0]
        for class_id in task0:
            for batch in self.train[class_id].batches(self.config.eval_batch_size):
                self.stats.update(batch)
        self.stats.freeze(task0)
        self.timing["normalization_seconds"] = time.perf_counter() - started
        return self.stats.record()

    def pretrain_encoder(self) -> list[dict]:
        started = time.perf_counter()
        task0 = self.manifest.tasks[0]
        class_ids = [int(class_id) for class_id in task0]
        class_rows = np.asarray([len(self.train[class_id]) for class_id in class_ids], dtype=np.int64)
        total_rows = int(class_rows.sum())
        temporary_head = nn.Linear(self.config.d_model, len(self.manifest.classes)).to(self.device)
        optimizer = torch.optim.Adam(
            [*self.encoder.parameters(), *temporary_head.parameters()],
            lr=self.config.learning_rate,
        )
        history: list[dict] = []
        self.encoder.train()
        for epoch in range(self.config.pretrain_epochs):
            samplers: dict[int, BlockShuffleSampler] = {}
            cursors: dict[int, IndexChunkCursor] = {}
            for class_id, row_count in zip(class_ids, class_rows.tolist()):
                sampler_seed = derived_seed(
                    self.seed,
                    self.manifest.dataset,
                    "pretrain_class_blocks",
                    epoch,
                    class_id,
                )
                blocks, _ = self.train[class_id].index_blocks(
                    self.config.shuffle_block_rows,
                    base_offset=0,
                    group_base=0,
                )
                sampler = BlockShuffleSampler(
                    blocks,
                    population_rows=row_count,
                    sample_rows=row_count,
                    seed=sampler_seed,
                    block_rows=self.config.shuffle_block_rows,
                )
                samplers[class_id] = sampler
                cursors[class_id] = IndexChunkCursor(
                    sampler.iter_chunks(self.config.batch_size)
                )
            mix_seed = derived_seed(
                self.seed, self.manifest.dataset, "pretrain_batch_mix", epoch
            )
            mix_rng = np.random.default_rng(mix_seed)
            loss_sum = 0.0
            correct = 0
            total = 0
            consumed = np.zeros(len(class_ids), dtype=np.int64)
            mixed_batches = 0
            single_class_batches = 0
            max_quota_error = 0.0
            while total < total_rows:
                batch_rows = min(self.config.batch_size, total_rows - total)
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
                    raw_parts.extend(self.train[class_id].take(indices) for indices in chunks)
                    label_parts.append(np.full(count, class_id, dtype=np.int64))
                    consumed[class_index] += count
                raw = np.vstack(raw_parts)
                labels = np.concatenate(label_parts)
                batch_order = mix_rng.permutation(batch_rows)
                raw = raw[batch_order]
                labels = labels[batch_order]
                if np.count_nonzero(quota) > 1:
                    mixed_batches += 1
                else:
                    single_class_batches += 1
                quota_error = np.max(np.abs(consumed - ideal))
                max_quota_error = max(max_quota_error, float(quota_error))
                values = torch.from_numpy(self.stats.transform(raw)).to(self.device)
                target = torch.from_numpy(labels).to(self.device)
                logits = temporary_head(self.encoder(values))
                loss = F.cross_entropy(logits, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.item()) * len(target)
                correct += int((logits.argmax(dim=1) == target).sum().item())
                total += len(target)
            if total != total_rows or not np.array_equal(consumed, class_rows):
                raise RuntimeError("pretrain sampler row accounting failed")
            history.append(
                {
                    "epoch": epoch + 1,
                    "rows": total,
                    "loss": loss_sum / max(total, 1),
                    "accuracy": correct / max(total, 1),
                    "sampler": {
                        "algorithm": (
                            "proportional_cumulative_quota_over_per_class_block_samplers_v1"
                        ),
                        "class_samplers": {
                            str(class_id): {
                                **samplers[class_id].record(),
                                "quota_rows": int(class_rows[index]),
                            }
                            for index, class_id in enumerate(class_ids)
                        },
                        "batch_mix_seed": int(mix_seed),
                        "mixed_batches": mixed_batches,
                        "single_class_batches": single_class_batches,
                        "max_abs_cumulative_quota_error_rows": max_quota_error,
                        "class_rows_seen": {
                            str(class_id): int(consumed[index])
                            for index, class_id in enumerate(class_ids)
                        },
                    },
                }
            )
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.encoder.eval()
        self.timing["pretrain_seconds"] = time.perf_counter() - started
        return history

    def _new_head(self, class_id: int) -> FamilyHead:
        key = str(class_id)
        if key in self.heads:
            raise ValueError(f"head already exists for class {class_id}")
        head = FamilyHead(
            self.config.d_model,
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
        ).to(self.device)
        self.heads[key] = head
        return head

    def train_family(self, class_id: int, task_classes: Sequence[int], task_index: int) -> list[dict]:
        started = time.perf_counter()
        head = self._new_head(class_id)
        initial_head_sha256 = _module_state_sha256(head)
        optimizer = torch.optim.Adam(head.parameters(), lr=self.config.learning_rate)
        positive = self.train[class_id]
        raw_negative_ids = [int(value) for value in task_classes if value != class_id]
        old_exemplar_ids = sorted(self.exemplars)
        raw_negative_sources = [self.train[value] for value in raw_negative_ids]
        old_exemplars = [self.exemplars[value] for value in old_exemplar_ids]
        negative = NormalizedNegativePool(
            raw_negative_sources,
            old_exemplars,
            self.stats,
            self.manifest.feature_dim,
        )
        if len(negative) == 0:
            raise ValueError(f"class {class_id} has no binary negative candidates")
        n_positive = len(positive)
        target_negative = min(len(negative), n_positive * self.config.negative_ratio)
        use_focal = self.config.loss_fn == "focal" and n_positive < self.config.minority_threshold
        history: list[dict] = []
        counter_dtype = _exposure_counter_dtype(self.config.epochs_per_task)
        negative_exposures = np.zeros(len(negative), dtype=counter_dtype)
        negative_sources = [
            {
                "source_kind": "current_task_train",
                "class_id": class_value,
                "rows": len(self.train[class_value]),
            }
            for class_value in raw_negative_ids
        ] + [
            {
                "source_kind": "prior_exemplar",
                "class_id": class_value,
                "rows": len(self.exemplars[class_value]),
                "normalized_sha256": array_sha256(self.exemplars[class_value]),
            }
            for class_value in old_exemplar_ids
        ]
        head.train()
        for epoch in range(self.config.epochs_per_task):
            positive_seed = derived_seed(
                self.seed,
                self.manifest.dataset,
                "binary_positive_blocks",
                task_index,
                class_id,
                epoch,
            )
            negative_seed = derived_seed(
                self.seed,
                self.manifest.dataset,
                "binary_negative_blocks",
                task_index,
                class_id,
                epoch,
            )
            mix_seed = derived_seed(
                self.seed,
                self.manifest.dataset,
                "binary_batch_mix",
                task_index,
                class_id,
                epoch,
            )
            positive_blocks, _ = positive.index_blocks(
                self.config.shuffle_block_rows,
                base_offset=0,
                group_base=0,
            )
            positive_sampler = BlockShuffleSampler(
                positive_blocks,
                population_rows=n_positive,
                sample_rows=n_positive,
                seed=positive_seed,
                block_rows=self.config.shuffle_block_rows,
            )
            negative_sampler = BlockShuffleSampler(
                negative.index_blocks(self.config.shuffle_block_rows),
                population_rows=len(negative),
                sample_rows=target_negative,
                seed=negative_seed,
                block_rows=self.config.shuffle_block_rows,
            )
            positive_cursor = IndexChunkCursor(
                positive_sampler.iter_chunks(self.config.batch_size)
            )
            negative_cursor = IndexChunkCursor(
                negative_sampler.iter_chunks(self.config.batch_size)
            )
            mix_rng = np.random.default_rng(mix_seed)
            positive_batch_rows = max(
                1, self.config.batch_size // (self.config.negative_ratio + 1)
            )
            loss_sum = 0.0
            correct = 0
            seen = 0
            positive_seen = 0
            negative_seen = 0
            positive_digest = hashlib.sha256()
            negative_digest = hashlib.sha256()
            source_selected = np.zeros(len(negative_sources), dtype=np.int64)
            optimizer_steps = 0
            zero_negative_steps = 0
            batch_rows_observed: list[int] = []
            negative_batch_rows: list[int] = []
            label_loss_sum = np.zeros(2, dtype=np.float64)
            label_count = np.zeros(2, dtype=np.int64)
            label_margin_sum = np.zeros(2, dtype=np.float64)
            label_margin_square_sum = np.zeros(2, dtype=np.float64)
            binary_confusion = np.zeros((2, 2), dtype=np.int64)
            while positive_seen < n_positive:
                positive_count = min(positive_batch_rows, n_positive - positive_seen)
                positive_parts = positive_cursor.take(positive_count)
                for indices in positive_parts:
                    _update_index_digest(positive_digest, indices)
                positive_values = np.vstack(
                    [
                        self.stats.transform(positive.take(indices))
                        for indices in positive_parts
                    ]
                )
                positive_after = positive_seen + positive_count
                negative_after = (target_negative * positive_after) // n_positive
                negative_count = negative_after - negative_seen
                if negative_count:
                    negative_parts = negative_cursor.take(negative_count)
                    for indices in negative_parts:
                        _update_index_digest(negative_digest, indices)
                        negative_exposures[indices] += 1
                        source_ids = np.searchsorted(
                            negative.offsets[1:], indices, side="right"
                        )
                        source_selected += np.bincount(
                            source_ids, minlength=len(negative_sources)
                        )
                    negative_values = np.vstack(
                        [negative.take(indices) for indices in negative_parts]
                    )
                else:
                    zero_negative_steps += 1
                    negative_values = np.empty(
                        (0, self.manifest.feature_dim), dtype=np.float32
                    )
                values = np.vstack([positive_values, negative_values])
                target = np.concatenate(
                    [
                        np.ones(positive_count, dtype=np.int64),
                        np.zeros(negative_count, dtype=np.int64),
                    ]
                )
                batch_order = mix_rng.permutation(len(target))
                values = values[batch_order]
                target = target[batch_order]
                tensor = torch.from_numpy(values).to(self.device)
                labels = torch.from_numpy(target).to(self.device)
                with torch.no_grad():
                    embedding = self.encoder(tensor)
                logits = head(embedding)
                if use_focal:
                    loss = focal_loss(
                        logits,
                        labels,
                        gamma=self.config.focal_gamma,
                        alpha=self.config.focal_alpha,
                    )
                else:
                    loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.item()) * len(labels)
                predicted = logits.argmax(dim=1)
                correct += int((predicted == labels).sum().item())
                with torch.no_grad():
                    ce = F.cross_entropy(logits, labels, reduction="none")
                    if use_focal:
                        probability = torch.softmax(logits, dim=1)
                        pt = probability.gather(1, labels[:, None]).squeeze(1)
                        alpha_t = torch.where(
                            labels == 1,
                            torch.full_like(pt, self.config.focal_alpha),
                            torch.full_like(pt, 1.0 - self.config.focal_alpha),
                        )
                        element_loss = (
                            alpha_t
                            * (1.0 - pt).pow(self.config.focal_gamma)
                            * ce
                        )
                    else:
                        element_loss = ce
                    margins = logits[:, 1] - logits[:, 0]
                    for label_value in (0, 1):
                        selected = labels == label_value
                        count = int(selected.sum().item())
                        if count:
                            selected_loss = element_loss[selected]
                            selected_margin = margins[selected]
                            label_count[label_value] += count
                            label_loss_sum[label_value] += float(
                                selected_loss.sum().item()
                            )
                            label_margin_sum[label_value] += float(
                                selected_margin.sum().item()
                            )
                            label_margin_square_sum[label_value] += float(
                                (selected_margin * selected_margin).sum().item()
                            )
                    for truth in (0, 1):
                        selected = labels == truth
                        if selected.any():
                            binary_confusion[truth] += np.bincount(
                                predicted[selected].detach().cpu().numpy(),
                                minlength=2,
                            )
                seen += len(labels)
                positive_seen = positive_after
                negative_seen = negative_after
                optimizer_steps += 1
                batch_rows_observed.append(len(labels))
                negative_batch_rows.append(negative_count)
            if positive_seen != n_positive or negative_seen != target_negative:
                raise RuntimeError("binary sampler row accounting failed")

            def label_diagnostic(label_value: int) -> dict[str, object]:
                count = int(label_count[label_value])
                mean_margin = label_margin_sum[label_value] / max(count, 1)
                variance = max(
                    0.0,
                    label_margin_square_sum[label_value] / max(count, 1)
                    - mean_margin * mean_margin,
                )
                return {
                    "examples": count,
                    "loss_numerator": float(label_loss_sum[label_value]),
                    "mean_loss": float(label_loss_sum[label_value] / max(count, 1)),
                    "mean_margin": float(mean_margin),
                    "population_std_margin": float(math.sqrt(variance)),
                }

            history.append(
                {
                    "epoch": epoch + 1,
                    "positive_rows": n_positive,
                    "negative_rows": target_negative,
                    "negative_ratio_realized": target_negative / n_positive,
                    "loss": loss_sum / max(seen, 1),
                    "accuracy": correct / max(seen, 1),
                    "loss_kind": "focal" if use_focal else "cross_entropy",
                    "loss_parameters": {
                        "gamma": self.config.focal_gamma if use_focal else None,
                        "positive_alpha": self.config.focal_alpha if use_focal else None,
                        "negative_alpha": (
                            1.0 - self.config.focal_alpha if use_focal else None
                        ),
                    },
                    "loss_diagnostics": {
                        "negative": label_diagnostic(0),
                        "positive": label_diagnostic(1),
                        "binary_confusion_matrix": binary_confusion.tolist(),
                        "true_positive_rate": (
                            float(binary_confusion[1, 1] / binary_confusion[1].sum())
                            if binary_confusion[1].sum()
                            else None
                        ),
                        "true_negative_rate": (
                            float(binary_confusion[0, 0] / binary_confusion[0].sum())
                            if binary_confusion[0].sum()
                            else None
                        ),
                    },
                    "positive_sampler": positive_sampler.record(),
                    "negative_sampler": negative_sampler.record(),
                    "exposure": {
                        "positive_unique_sampler_units": n_positive,
                        "negative_unique_sampler_units": target_negative,
                        "positive_exposures_used_by_loss": n_positive,
                        "negative_exposures_used_by_loss": target_negative,
                        "positive_index_stream_sha256": positive_digest.hexdigest(),
                        "negative_index_stream_sha256": negative_digest.hexdigest(),
                        "negative_selected_by_source": [
                            {
                                **source,
                                "selected_rows": int(source_selected[index]),
                            }
                            for index, source in enumerate(negative_sources)
                        ],
                        "optimizer_steps": optimizer_steps,
                        "zero_negative_steps": zero_negative_steps,
                        "batch_rows_min": min(batch_rows_observed),
                        "batch_rows_max": max(batch_rows_observed),
                        "batch_rows_mean": float(np.mean(batch_rows_observed)),
                        "negative_batch_rows_min": min(negative_batch_rows),
                        "negative_batch_rows_max": max(negative_batch_rows),
                    },
                    "batch_mix_seed": int(mix_seed),
                }
            )
        head.eval()
        for parameter in head.parameters():
            parameter.requires_grad = False
        exposure_histogram = np.bincount(
            negative_exposures.astype(np.int64),
            minlength=self.config.epochs_per_task + 1,
        )
        positive_total = n_positive * self.config.epochs_per_task
        negative_total = target_negative * self.config.epochs_per_task
        prior_positive = n_positive / (n_positive + target_negative)
        log_offset = float(math.log(n_positive / target_negative))
        self.training_exposure_records[class_id] = {
            "task_index": int(task_index),
            "positive_class_id": int(class_id),
            "unique_definition": (
                "stable sampler units: shard rows and exemplar-array rows are "
                "distinct even when their feature bytes are equal"
            ),
            "positive_population_rows": n_positive,
            "negative_candidate_rows": len(negative),
            "negative_sources": negative_sources,
            "desired_negative_rows_per_epoch": n_positive
            * self.config.negative_ratio,
            "selected_negative_rows_per_epoch": target_negative,
            "negative_ratio_realized": target_negative / n_positive,
            "positive_rows_per_optimizer_step": positive_batch_rows,
            "candidate_coverage_per_epoch": target_negative / len(negative),
            "candidate_limited": target_negative < n_positive * self.config.negative_ratio,
            "epochs": self.config.epochs_per_task,
            "positive_unique_sampler_units_across_epochs": (
                n_positive if self.config.epochs_per_task else 0
            ),
            "negative_unique_sampler_units_across_epochs": int(
                np.count_nonzero(negative_exposures)
            ),
            "positive_exposures_used_by_loss": positive_total,
            "negative_exposures_used_by_loss": negative_total,
            "negative_exposure_multiplicity_histogram": {
                str(index): int(count)
                for index, count in enumerate(exposure_histogram.tolist())
            },
            "negative_exposure_counter_dtype": np.dtype(counter_dtype).name,
            "negative_exposure_counter_capacity": int(np.iinfo(counter_dtype).max),
            "initial_head_state_sha256": initial_head_sha256,
            "final_head_state_sha256": _module_state_sha256(head),
        }
        self.training_prior_records[class_id] = {
            "algorithm": "binary_count_exposure_prior_offset_v1",
            "positive_exposures": positive_total,
            "negative_exposures": negative_total,
            "per_epoch_positive_rows": n_positive,
            "per_epoch_negative_rows": target_negative,
            "positive_exposure_prior": float(prior_positive),
            "log_positive_to_negative_exposure_ratio": log_offset,
            "offset_applied_to_positive_margin": -log_offset,
            "basis": (
                "actual_examples_used_by_loss"
                if self.config.epochs_per_task
                else "per_epoch_sampler_contract_no_optimization_epochs"
            ),
            "loss_kind": "focal" if use_focal else "cross_entropy",
            "focal_gamma": self.config.focal_gamma if use_focal else None,
            "focal_positive_alpha": self.config.focal_alpha if use_focal else None,
            "interpretation": (
                "diagnostic count-exposure offset only; not a calibrated posterior"
            ),
        }
        self.timing["head_training_seconds"][str(class_id)] = time.perf_counter() - started
        return history

    @torch.no_grad()
    def _embed(self, normalized: np.ndarray) -> np.ndarray:
        output: list[np.ndarray] = []
        self.encoder.eval()
        for start in range(0, len(normalized), self.config.eval_batch_size):
            tensor = torch.from_numpy(
                np.ascontiguousarray(normalized[start : start + self.config.eval_batch_size], dtype=np.float32)
            ).to(self.device)
            output.append(self.encoder(tensor).cpu().numpy().astype(np.float32))
        return np.vstack(output) if output else np.empty((0, self.config.d_model), dtype=np.float32)

    def build_routers_and_exemplars(self, class_id: int) -> dict:
        started = time.perf_counter()
        source = self.train[class_id]
        cap_reservoir_seed = derived_seed(
            self.seed, self.manifest.dataset, "router_cap_reservoir", class_id
        )
        cap_fit_seed = derived_seed(
            self.seed, self.manifest.dataset, "router_cap_fit", class_id
        )
        exemplar_reservoir_seed = derived_seed(
            self.seed, self.manifest.dataset, "exemplar_reservoir", class_id
        )
        exemplar_selection_seed = derived_seed(
            self.seed, self.manifest.dataset, "exemplar_farthest", class_id
        )
        cap_reservoir = MatrixReservoir(
            self.config.router_cap_samples,
            self.config.d_model,
            np.random.default_rng(cap_reservoir_seed),
        )
        exemplar_reservoir = MatrixReservoir(
            self.config.exemplar_candidate_capacity,
            self.manifest.feature_dim,
            np.random.default_rng(exemplar_reservoir_seed),
        )
        # Pass 1 builds the bounded cap arm and an independent exemplar pool.
        for raw in source.batches(self.config.eval_batch_size):
            exemplar_reservoir.update(raw)
            embedding = self._embed(self.stats.transform(raw))
            cap_reservoir.update(embedding)
        cap_initial = cap_state_from_reservoir(
            cap_reservoir,
            quantile=self.config.router_lambda_quantile,
            max_centroids=self.config.router_max_centroids,
            rng=np.random.default_rng(cap_fit_seed),
        )
        cap_state = matched_cap_state(
            cap_initial,
            cap_reservoir,
            max_centroids=self.config.router_max_centroids,
            batch_size=self.config.eval_batch_size,
        )
        self.routers.cap[class_id] = cap_state
        one_pass = OnePassBudgetRouter(
            self.config.d_model,
            initial_centroids=cap_state.centroids,
            initial_counts=cap_state.counts,
            initial_lambda=cap_state.lam,
            max_centroids=self.config.router_max_centroids,
        )
        selected_indices = np.sort(cap_reservoir.indices())
        if (
            len(selected_indices) != len(np.unique(selected_indices))
            or selected_indices[0] < 0
            or selected_indices[-1] >= len(source)
        ):
            raise RuntimeError("cap reservoir stream-index audit failed")
        # Pass 2 refines only the complement of the cap sample. Therefore a
        # class no larger than the cap has byte-identical cap/uncapped states.
        stream_offset = 0
        selected_rows_skipped = 0
        for raw in source.batches(self.config.eval_batch_size):
            batch_end = stream_offset + len(raw)
            left = int(np.searchsorted(selected_indices, stream_offset, side="left"))
            right = int(np.searchsorted(selected_indices, batch_end, side="left"))
            if right > left:
                selected_local = selected_indices[left:right] - stream_offset
                keep = np.ones(len(raw), dtype=bool)
                keep[selected_local] = False
                extra_raw = raw[keep]
                selected_rows_skipped += right - left
            else:
                extra_raw = raw
            if len(extra_raw):
                embedding = self._embed(self.stats.transform(extra_raw))
                one_pass.update(embedding)
            stream_offset = batch_end
        if stream_offset != len(source) or selected_rows_skipped != len(selected_indices):
            raise RuntimeError("cap-complement stream accounting failed")
        self.routers.uncapped[class_id] = uncapped_state(
            one_pass,
            selected_index_sha256=cap_state.stats["selected_index_sha256"],
        )
        uncapped_state_record = self.routers.uncapped[class_id].stats
        uncapped_state_record["selected_rows_skipped"] = selected_rows_skipped
        uncapped_state_record["complement_rows_expected"] = len(source) - len(
            selected_indices
        )
        uncapped_state_record["state_identical_to_cap"] = bool(
            np.array_equal(self.routers.uncapped[class_id].centroids, cap_state.centroids)
            and np.array_equal(self.routers.uncapped[class_id].counts, cap_state.counts)
        )
        if (
            self.routers.uncapped[class_id].stats["shared_initial_centroid_sha256"]
            != cap_state.stats["centroid_sha256"]
            or self.routers.uncapped[class_id].stats["shared_initial_count_sha256"]
            != cap_state.stats["count_sha256"]
            or self.routers.uncapped[class_id].stats["shared_initial_lambda"]
            != cap_state.stats["lambda"]
        ):
            raise RuntimeError("cap and uncapped router initialisation diverged")
        if len(source) <= self.config.router_cap_samples and not uncapped_state_record[
            "state_identical_to_cap"
        ]:
            raise RuntimeError("cap and uncapped states must match when the cap covers the class")

        candidates_raw = exemplar_reservoir.array()
        candidates = self.stats.transform(candidates_raw)
        candidate_embeddings = self._embed(candidates)
        selected_indices = self._farthest_first(
            candidate_embeddings,
            min(self.config.exemplar_capacity, len(candidate_embeddings)),
            np.random.default_rng(exemplar_selection_seed),
        )
        selected = candidates[selected_indices]
        self.exemplars[class_id] = selected.astype(np.float32, copy=True)
        self.exemplar_records[class_id] = {
            "candidate_reservoir": exemplar_reservoir.record(),
            "candidate_reservoir_seed": int(exemplar_reservoir_seed),
            "selected_rows": int(len(selected)),
            "selected_normalized_sha256": array_sha256(selected),
            "selection_seed": int(exemplar_selection_seed),
            "selection": "greedy_farthest_first_on_frozen_encoder_embeddings",
        }
        self.router_records[class_id] = {
            "cap3000": self.routers.cap[class_id].stats,
            "uncapped": self.routers.uncapped[class_id].stats,
        }
        self.router_records[class_id]["cap3000"]["reservoir_seed"] = int(
            cap_reservoir_seed
        )
        self.router_records[class_id]["cap3000"]["fit_seed"] = int(cap_fit_seed)
        self.timing["router_exemplar_seconds"][str(class_id)] = time.perf_counter() - started
        return {
            "router": self.router_records[class_id],
            "exemplar": self.exemplar_records[class_id],
        }

    @staticmethod
    def _farthest_first(
        embeddings: np.ndarray,
        count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if count <= 0:
            return np.empty(0, dtype=np.int64)
        selected = [int(rng.integers(0, len(embeddings)))]
        used = np.zeros(len(embeddings), dtype=bool)
        minimum = np.full(len(embeddings), np.inf, dtype=np.float32)
        while len(selected) < count:
            newest = selected[-1]
            delta = embeddings - embeddings[newest]
            distance = np.einsum("ij,ij->i", delta, delta)
            np.minimum(minimum, distance, out=minimum)
            used[newest] = True
            minimum[used] = -np.inf
            selected.append(int(np.argmax(minimum)))
        return np.asarray(selected, dtype=np.int64)

    @torch.no_grad()
    def _head_scores(
        self, embeddings: np.ndarray, seen_classes: list[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        tensor = torch.from_numpy(np.ascontiguousarray(embeddings, dtype=np.float32)).to(self.device)
        result = np.empty((len(embeddings), len(seen_classes)), dtype=np.float32)
        corrected = np.empty_like(result)
        for column, class_id in enumerate(seen_classes):
            logits = self.heads[str(class_id)](tensor)
            result[:, column] = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            margin = logits[:, 1] - logits[:, 0]
            offset = self.training_prior_records[class_id][
                "log_positive_to_negative_exposure_ratio"
            ]
            corrected[:, column] = torch.sigmoid(margin - float(offset)).cpu().numpy()
        return result, corrected

    def state_sha256(self) -> str:
        router_records = []
        for variant, states in (
            ("cap3000", self.routers.cap),
            ("uncapped", self.routers.uncapped),
        ):
            for class_id, state in sorted(states.items()):
                router_records.append(
                    {
                        "variant": variant,
                        "class_id": int(class_id),
                        "centroids_sha256": array_sha256(state.centroids),
                        "counts_sha256": array_sha256(state.counts),
                        "lambda": float(state.lam),
                    }
                )
        record = {
            "encoder": _module_state_sha256(self.encoder),
            "encoder_training": bool(self.encoder.training),
            "heads": [
                {
                    "class_id": int(class_id),
                    "state_sha256": _module_state_sha256(self.heads[str(class_id)]),
                    "training": bool(self.heads[str(class_id)].training),
                }
                for class_id in sorted(int(key) for key in self.heads.keys())
            ],
            "normalization": {
                "count": int(self.stats.count),
                "mean_sha256": array_sha256(self.stats.mean),
                "m2_sha256": array_sha256(self.stats.m2),
                "frozen": bool(self.stats.frozen),
                "source_classes": list(self.stats.source_classes),
            },
            "routers": router_records,
            "exemplars": [
                {"class_id": int(class_id), "sha256": array_sha256(values)}
                for class_id, values in sorted(self.exemplars.items())
            ],
        }
        return canonical_sha256(record)

    def evaluate_checkpoint(self, checkpoint: int, seen_classes: list[int]) -> dict:
        started = time.perf_counter()
        state_before = self.state_sha256()
        rng_before = _rng_state_record()
        width = len(seen_classes)
        class_to_index = {class_id: index for index, class_id in enumerate(seen_classes)}
        view_names = ["official", *self.evaluation_views]
        confusion = {
            view: {
                arm: np.zeros((width, width), dtype=np.int64) for arm in ARMS
            }
            for view in view_names
        }
        excluded_confusion = {
            view: {
                arm: np.zeros((width, width), dtype=np.int64) for arm in ARMS
            }
            for view in self.evaluation_views
        }
        for true_class in seen_classes:
            true_index = class_to_index[true_class]
            current_shard = -1
            open_masks: dict[str, np.ndarray] = {}

            def close_masks() -> None:
                while open_masks:
                    _, mask = open_masks.popitem()
                    mapping = getattr(mask, "_mmap", None)
                    if mapping is not None:
                        mapping.close()

            for shard_id, local_start, raw in self.test[
                true_class
            ].batches_with_coordinates(self.config.eval_batch_size):
                if shard_id != current_shard:
                    close_masks()
                    current_shard = shard_id
                    for view_name, view in self.evaluation_views.items():
                        open_masks[view_name] = np.load(
                            view.masks[true_class][shard_id].path,
                            mmap_mode="r",
                            allow_pickle=False,
                        )
                normalized = self.stats.transform(raw)
                embeddings = self._embed(normalized)
                head, corrected_head = self._head_scores(embeddings, seen_classes)
                cap = self.routers.scores(embeddings, seen_classes, "cap3000")
                uncapped = self.routers.scores(embeddings, seen_classes, "uncapped")
                score_map = {
                    "head_only": head,
                    "router_only_cap3000": cap,
                    "joint_cap3000": head + 0.5 * cap,
                    "router_only_uncapped": uncapped,
                    "joint_uncapped": head + 0.5 * uncapped,
                    "head_only_exposure_prior_corrected": corrected_head,
                    "joint_cap3000_exposure_prior_corrected": corrected_head
                    + 0.5 * cap,
                    "joint_uncapped_exposure_prior_corrected": corrected_head
                    + 0.5 * uncapped,
                }
                for arm, scores in score_map.items():
                    predicted = scores.argmax(axis=1)
                    counts = np.bincount(predicted, minlength=width)
                    confusion["official"][arm][true_index] += counts
                    for view_name, mask in open_masks.items():
                        keep = np.asarray(
                            mask[local_start : local_start + len(predicted)],
                            dtype=np.bool_,
                        )
                        retained = np.bincount(
                            predicted[keep], minlength=width
                        )
                        excluded = np.bincount(
                            predicted[~keep], minlength=width
                        )
                        confusion[view_name][arm][true_index] += retained
                        excluded_confusion[view_name][arm][true_index] += excluded
            close_masks()
        result: dict[str, object] = {
            "checkpoint": checkpoint,
            "seen_classes": seen_classes,
            "views": {},
            "view_decomposition": {},
        }
        for view_name, arm_matrices in confusion.items():
            view_result: dict[str, object] = {"arms": {}}
            expected_support: list[int] | None = None
            for arm, matrix in arm_matrices.items():
                support = matrix.sum(axis=1).tolist()
                if expected_support is None:
                    expected_support = support
                elif support != expected_support:
                    raise RuntimeError("evaluation arms disagree on view support")
                metrics = _metrics_from_confusion(
                    matrix,
                    seen_classes,
                    self.manifest.class_names,
                    self.manifest.normal_class_id,
                    metric_profile=self.manifest.metric_profile,
                )
                task_accuracy: dict[str, float] = {}
                for task_id, task in enumerate(
                    self.manifest.tasks[: checkpoint + 1]
                ):
                    rows = [class_to_index[class_id] for class_id in task]
                    denominator = int(matrix[rows, :].sum())
                    numerator = int(sum(matrix[row, row] for row in rows))
                    task_accuracy[str(task_id)] = (
                        numerator / denominator if denominator else 0.0
                    )
                metrics["task_accuracy"] = task_accuracy
                view_result["arms"][arm] = metrics
            view_result["support_by_class"] = {
                str(class_id): int(expected_support[index])
                for index, class_id in enumerate(seen_classes)
            }
            view_result["test_rows"] = int(sum(expected_support or []))
            result["views"][view_name] = view_result

        official_matrices = confusion["official"]
        for view_name, excluded_matrices in excluded_confusion.items():
            decomposition: dict[str, object] = {
                "relationship": "official_equals_retained_plus_excluded",
                "arms": {},
            }
            for arm in ARMS:
                retained = confusion[view_name][arm]
                excluded = excluded_matrices[arm]
                if not np.array_equal(official_matrices[arm], retained + excluded):
                    raise RuntimeError(
                        f"confusion conservation failed for {view_name}/{arm}"
                    )
                decomposition["arms"][arm] = {
                    "excluded_confusion_matrix": excluded.tolist(),
                    "matrix_conservation_verified": True,
                }
            result["view_decomposition"][view_name] = decomposition

        state_after = self.state_sha256()
        rng_after = _rng_state_record()
        if state_after != state_before:
            raise RuntimeError("evaluation mutated trained model/router/exemplar state")
        if rng_after["canonical_sha256"] != rng_before["canonical_sha256"]:
            raise RuntimeError("evaluation consumed or mutated process RNG state")
        result["evaluation_invariants"] = {
            "algorithm": "single_forward_mask_projection_v1",
            "state_before_sha256": state_before,
            "state_after_sha256": state_after,
            "rng_before_sha256": rng_before["canonical_sha256"],
            "rng_after_sha256": rng_after["canonical_sha256"],
            "state_unchanged": True,
            "rng_unchanged": True,
        }
        self.timing["evaluation_seconds"][str(checkpoint)] = time.perf_counter() - started
        return result


def _metrics_from_confusion(
    matrix: np.ndarray,
    labels: Sequence[int],
    class_names: dict[int, str],
    normal_class_id: int | None,
    *,
    metric_profile: str = NIDS_METRIC_PROFILE,
) -> dict:
    matrix = np.asarray(matrix, dtype=np.int64)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros(len(labels), dtype=np.float64),
        where=predicted > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(len(labels), dtype=np.float64),
        where=support > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(len(labels), dtype=np.float64),
        where=(precision + recall) > 0,
    )
    total = int(matrix.sum())
    per_class = []
    for index, class_id in enumerate(labels):
        per_class.append(
            {
                "class_id": int(class_id),
                "class_name": class_names[class_id],
                "support": int(support[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
        )
    result = {
        "confusion_matrix": matrix.tolist(),
        "total_rows": total,
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "per_class": per_class,
    }
    if metric_profile == NIDS_METRIC_PROFILE:
        if normal_class_id not in labels:
            raise ValueError("normal_class_id is absent from this checkpoint")
        normal_index = labels.index(normal_class_id)
        normal_support = int(support[normal_index])
        benign_false_positives = int(
            normal_support - matrix[normal_index, normal_index]
        )
        attack_indices = [
            index for index in range(len(labels)) if index != normal_index
        ]
        attack_support = int(support[attack_indices].sum())
        attacks_predicted_as_attack = int(
            matrix[np.ix_(attack_indices, attack_indices)].sum()
        )
        result["binary_detection"] = {
            "normal_class_id": int(normal_class_id),
            "normal_class_name": class_names[normal_class_id],
            "benign_false_positive_rate": (
                float(benign_false_positives / normal_support) if normal_support else 0.0
            ),
            "benign_false_positives": benign_false_positives,
            "benign_support": normal_support,
            "attack_detection_recall": (
                float(attacks_predicted_as_attack / attack_support) if attack_support else 0.0
            ),
            "attacks_predicted_as_attack": attacks_predicted_as_attack,
            "attack_support": attack_support,
        }
    elif metric_profile == GENERIC_METRIC_PROFILE:
        if normal_class_id is not None:
            raise ValueError(
                "generic_multiclass metrics require normal_class_id to be null"
            )
    else:
        raise ValueError(f"unsupported metric_profile: {metric_profile!r}")
    return result


COMMON_SUMMARY_METRICS = (
    "average_task_accuracy",
    "average_forgetting",
    "final_overall_accuracy",
    "final_macro_f1",
    "final_balanced_accuracy",
)
BINARY_SUMMARY_METRICS = (
    "final_benign_false_positive_rate",
    "final_attack_detection_recall",
)
SUMMARY_METRICS = COMMON_SUMMARY_METRICS + BINARY_SUMMARY_METRICS


def _summarise_checkpoints(checkpoints: list[dict], task_count: int) -> dict:
    view_names = list(checkpoints[0]["views"])
    views: dict[str, object] = {}
    for view_name in view_names:
        arm_summary: dict[str, object] = {}
        for arm in ARMS:
            matrix: list[list[float | None]] = [
                [None for _ in range(task_count)] for _ in range(task_count)
            ]
            for checkpoint in checkpoints:
                row = int(checkpoint["checkpoint"])
                task_accuracy = checkpoint["views"][view_name]["arms"][arm][
                    "task_accuracy"
                ]
                for task_id, value in task_accuracy.items():
                    matrix[row][int(task_id)] = float(value)
            final_values = [value for value in matrix[-1] if value is not None]
            forgetting: list[float] = []
            for task_id in range(task_count - 1):
                prior = [
                    matrix[row][task_id]
                    for row in range(task_count - 1)
                    if matrix[row][task_id] is not None
                ]
                final = matrix[-1][task_id]
                if prior and final is not None:
                    forgetting.append(float(max(prior) - final))
            final_metrics = checkpoints[-1]["views"][view_name]["arms"][arm]
            summary_record = {
                "task_accuracy_matrix": matrix,
                "average_task_accuracy": float(np.mean(final_values)),
                "average_forgetting": (
                    float(np.mean(forgetting)) if forgetting else 0.0
                ),
                "final_overall_accuracy": final_metrics["accuracy"],
                "final_macro_f1": final_metrics["macro_f1"],
                "final_balanced_accuracy": final_metrics["balanced_accuracy"],
            }
            binary_detection = final_metrics.get("binary_detection")
            if binary_detection is not None:
                summary_record.update(
                    {
                        "final_benign_false_positive_rate": binary_detection[
                            "benign_false_positive_rate"
                        ],
                        "final_attack_detection_recall": binary_detection[
                            "attack_detection_recall"
                        ],
                    }
                )
            arm_summary[arm] = summary_record
        views[view_name] = arm_summary

    sensitivity: dict[str, object] = {}
    official = views["official"]
    for view_name in view_names:
        if view_name == "official":
            continue
        contrast = f"{view_name}_minus_official"
        sensitivity[contrast] = {}
        for arm in ARMS:
            metric_names = [
                metric
                for metric in SUMMARY_METRICS
                if metric in official[arm] and metric in views[view_name][arm]
            ]
            sensitivity[contrast][arm] = {
                metric: float(
                    views[view_name][arm][metric] - official[arm][metric]
                )
                for metric in metric_names
            }
    return {"views": views, "sensitivity": sensitivity}


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


def run_seed(
    manifest: DatasetManifest,
    config: RunConfig,
    seed: int,
    protocol_sha256: str,
    evaluation_views: Sequence[EvaluationView] = (),
) -> dict:
    _seed_process(seed, config.deterministic)
    device = _resolve_device(config.device)
    run_started = time.perf_counter()
    agent = StreamingOFRA(
        manifest, config, seed, device, evaluation_views=evaluation_views
    )
    normalization = agent.fit_task0_stats()
    pretrain = agent.pretrain_encoder()
    checkpoints: list[dict] = []
    training_history: dict[str, list[dict]] = {}
    seen: list[int] = []
    for task_index, task in enumerate(manifest.tasks):
        agent._log(f"seed={seed} task={task_index} classes={list(task)}")
        for class_id in task:
            training_history[str(class_id)] = agent.train_family(class_id, task, task_index)
        for class_id in task:
            agent.build_routers_and_exemplars(class_id)
            seen.append(class_id)
        checkpoints.append(agent.evaluate_checkpoint(task_index, list(seen)))
    summary = _summarise_checkpoints(checkpoints, len(manifest.tasks))
    agent.timing["total_seconds"] = time.perf_counter() - run_started
    exposure_records = {
        str(key): value for key, value in agent.training_exposure_records.items()
    }
    prior_records = {
        str(key): value for key, value in agent.training_prior_records.items()
    }
    exemplar_records = {
        str(key): value for key, value in agent.exemplar_records.items()
    }
    router_records = {
        str(key): value for key, value in agent.router_records.items()
    }
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "dataset": manifest.dataset,
        "problem_type": manifest.problem_type,
        "metric_profile": manifest.metric_profile,
        "task_semantics": manifest.task_semantics,
        "normal_class_id": manifest.normal_class_id,
        "seed": int(seed),
        "protocol_sha256": protocol_sha256,
        "normalization": normalization,
        "pretrain_history": pretrain,
        "training_history": training_history,
        "training_exposure_records": exposure_records,
        "training_prior_records": prior_records,
        "exemplar_records": exemplar_records,
        "router_records": router_records,
        "checkpoints": checkpoints,
        "summary": summary,
        "training_invariants": {
            "training_input_logical_sha256": dataset_logical_fingerprints(manifest)[
                "train_logical_sha256"
            ],
            "training_record_sha256": canonical_sha256(
                {
                    "normalization": normalization,
                    "pretrain_history": pretrain,
                    "training_history": training_history,
                    "training_exposure_records": exposure_records,
                    "training_prior_records": prior_records,
                    "exemplar_records": exemplar_records,
                    "router_records": router_records,
                }
            ),
        },
        "timing": agent.timing,
    }
    result["deterministic_result_sha256"] = canonical_sha256(_without_timing(result))
    agent.close()
    return result


def _source_manifest(package_dir: Path) -> dict:
    files = []
    for path in sorted(package_dir.glob("*.py")):
        files.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"files": files, "manifest_sha256": canonical_sha256(files)}


def _dataset_files(manifest: DatasetManifest) -> list[dict]:
    files = []
    for record in manifest.classes:
        for split, shards in (("train", record.train), ("test", record.test)):
            for shard in shards:
                files.append(
                    {
                        "class_id": record.class_id,
                        "split": split,
                        "path": str(shard.path),
                        "rows": shard.rows,
                        "sha256": shard.sha256,
                    }
                )
    return files


def _environment(device: torch.device) -> dict:
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    if device.type == "cuda":
        result["gpu_name"] = torch.cuda.get_device_name(device)
        result["cudnn_version"] = torch.backends.cudnn.version()
    return result


def _evaluation_protocol(
    manifest: DatasetManifest, views: Sequence[EvaluationView]
) -> dict[str, object]:
    fingerprints = dataset_logical_fingerprints(manifest)
    descriptors: dict[str, object] = {
        "official": {
            "kind": "primary_manifest_test",
            "streaming_manifest_sha256": manifest.manifest_sha256,
            "logical_sha256": fingerprints["official_test_logical_sha256"],
            "test_files": [
                {
                    "class_id": record.class_id,
                    "ordinal": ordinal,
                    "rows": shard.rows,
                    "sha256": shard.sha256,
                }
                for record in manifest.classes
                for ordinal, shard in enumerate(record.test)
            ],
        }
    }
    for view in sorted(views, key=lambda item: item.name):
        descriptors[view.name] = {
            "kind": "test_only_mask_projection",
            "manifest": str(view.path),
            "manifest_sha256": view.manifest_sha256,
            "canonical_sha256": view.canonical_sha256,
            "audit": view.audit_provenance,
            "retained_rows": {
                str(key): value for key, value in view.retained_rows.items()
            },
            "excluded_rows": {
                str(key): value for key, value in view.excluded_rows.items()
            },
            "masks": [
                {
                    "class_id": class_id,
                    "ordinal": ordinal,
                    "path": str(mask.path),
                    "rows": mask.rows,
                    "sha256": mask.sha256,
                    "true_count": mask.true_count,
                    "false_count": mask.false_count,
                    "parent_sha256": mask.parent_sha256,
                }
                for class_id, masks in sorted(view.masks.items())
                for ordinal, mask in enumerate(masks)
            ],
        }
    return {
        "algorithm": "single_forward_mask_projection_v1",
        "view_order": ["official", *sorted(view.name for view in views)],
        "views": descriptors,
        "training_input_logical_sha256": fingerprints["train_logical_sha256"],
        "empty_retained_class_policy": "error_before_training",
        "conservation_invariant": (
            "confusion_official_equals_retained_plus_excluded_per_checkpoint_arm"
        ),
    }


def _protocol(
    manifest: DatasetManifest,
    config: RunConfig,
    seeds: Sequence[int],
    evaluation_views: Sequence[EvaluationView] = (),
) -> dict:
    from .exposure_preflight import exposure_preflight_for_manifest

    device = _resolve_device(config.device)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "runner": "streaming_full_validation_v2_masked_views",
        "manifest": str(manifest.path),
        "manifest_sha256": manifest.manifest_sha256,
        "input_source_provenance": manifest.source_provenance,
        "dataset": manifest.dataset,
        "feature_dim": manifest.feature_dim,
        "problem_type": manifest.problem_type,
        "metric_profile": manifest.metric_profile,
        "task_semantics": manifest.task_semantics,
        "normal_class": {
            "class_id": manifest.normal_class_id,
            "class_name": (
                manifest.class_names[manifest.normal_class_id]
                if manifest.normal_class_id is not None
                else None
            ),
            "source": (
                "explicit manifest normal_class_id"
                if manifest.normal_class_id is not None
                else "not applicable for application_classification"
            ),
        },
        "tasks": [list(task) for task in manifest.tasks],
        "class_names": manifest.class_names,
        "seeds": [int(seed) for seed in seeds],
        "config": asdict(config),
        "normalization_scope": "Task-0 train shards only; frozen before Task-0 pretraining",
        "prediction_arms": ARMS,
        "primary_prediction_arms": list(PRIMARY_ARMS),
        "diagnostic_prediction_arms": list(DIAGNOSTIC_ARMS),
        "exposure_prior_diagnostic": {
            **EXPOSURE_PRIOR_FORMULA_REGISTRY,
            "P_N_source": "actual examples used by family loss across epochs",
            "focal_caveat": (
                "does not correct focal alpha/gamma or negative-pool covariate shift; "
                "the score is diagnostic, not a calibrated posterior"
            ),
        },
        "exposure_preflight": exposure_preflight_for_manifest(manifest, config),
        "evaluation": _evaluation_protocol(manifest, evaluation_views),
        "output_directory_lock": OUTPUT_DIRECTORY_LOCK_CONTRACT,
        "router_algorithms": {
            "cap3000": "full_dpmeans_topk_plus_original_order_matched_refinement_v3",
            "uncapped": "matched_cap_state_plus_complement_refinement_v3",
        },
        "router_comparison_design": (
            "pass 1 fits the cap reservoir and matched-refines it in original stream order; "
            "uncapped copies final cap centroids/counts/lambda; pass 2 refines only rows not "
            "selected by the cap reservoir; classes no larger than the cap are identical"
        ),
        "negative_sampling": (
            "all positives plus fresh without-replacement negatives, at most "
            f"{config.negative_ratio} per positive"
        ),
        "training_sampler": {
            "algorithm": BlockShuffleSampler.algorithm,
            "block_rows": config.shuffle_block_rows,
            "task0_pretraining_batch_composition": (
                "natural-frequency cumulative quotas over independent per-class block cursors; "
                "within-batch deterministic shuffle"
            ),
            "positive_semantics": "every positive row exactly once per epoch",
            "negative_semantics": (
                "exact uniform without-replacement subset via sequential hypergeometric block allocation"
            ),
            "locality": "shuffle shard order, block order within shard, and rows within block",
        },
        "exemplar_selection": {
            "candidate_sampler": "deterministic_reservoir_algorithm_r",
            "candidate_capacity": config.exemplar_candidate_capacity,
            "selector": "greedy_farthest_first_on_candidate_embeddings",
            "selected_capacity": config.exemplar_capacity,
            "scope": "auditable approximation over reservoir candidates; not exact full-class farthest-first",
        },
        "dataset_files": _dataset_files(manifest),
        "source": _source_manifest(Path(__file__).parent),
        "environment": _environment(device),
        "environment_controls": {
            key: os.environ.get(key)
            for key in (
                "PYTHONHASHSEED",
                "CUBLAS_WORKSPACE_CONFIG",
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
    }


def _aggregate(results: list[dict]) -> dict:
    output: dict[str, object] = {
        "seeds": [result["seed"] for result in results],
        "views": {},
        "sensitivity": {},
    }
    view_names = list(results[0]["summary"]["views"])
    for view_name in view_names:
        output["views"][view_name] = {}
        for arm in ARMS:
            records = [
                result["summary"]["views"][view_name][arm] for result in results
            ]
            metrics = {}
            metric_names = [
                key for key in SUMMARY_METRICS if key in records[0]
            ]
            if any(
                [key for key in SUMMARY_METRICS if key in record] != metric_names
                for record in records[1:]
            ):
                raise RuntimeError("per-seed summary metric registry mismatch")
            for key in metric_names:
                values = np.asarray(
                    [record[key] for record in records], dtype=np.float64
                )
                metrics[key] = {
                    "per_seed": values.tolist(),
                    "mean": float(values.mean()),
                    "sample_std": (
                        float(values.std(ddof=1)) if len(values) > 1 else 0.0
                    ),
                }
            output["views"][view_name][arm] = metrics
    for contrast in results[0]["summary"]["sensitivity"]:
        output["sensitivity"][contrast] = {}
        for arm in ARMS:
            output["sensitivity"][contrast][arm] = {}
            metric_names = [
                metric
                for metric in SUMMARY_METRICS
                if metric in results[0]["summary"]["sensitivity"][contrast][arm]
            ]
            if any(
                [
                    metric
                    for metric in SUMMARY_METRICS
                    if metric
                    in result["summary"]["sensitivity"][contrast][arm]
                ]
                != metric_names
                for result in results[1:]
            ):
                raise RuntimeError("per-seed sensitivity metric registry mismatch")
            for metric in metric_names:
                values = np.asarray(
                    [
                        result["summary"]["sensitivity"][contrast][arm][metric]
                        for result in results
                    ],
                    dtype=np.float64,
                )
                output["sensitivity"][contrast][arm][metric] = {
                    "per_seed": values.tolist(),
                    "mean": float(values.mean()),
                    "sample_std": (
                        float(values.std(ddof=1)) if len(values) > 1 else 0.0
                    ),
                }
    return output


def _atomic_write_json(path: Path, value: object) -> None:
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


def _load_json_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot validate existing JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"existing JSON file is not an object: {path}")
    return value


def _validate_protocol_file(path: Path, expected_sha256: str) -> dict:
    protocol = _load_json_object(path)
    stored = protocol.get("protocol_sha256")
    payload = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    actual = canonical_sha256(payload)
    if stored != actual:
        raise RuntimeError(f"existing protocol has an invalid self-hash: {path}")
    if stored != expected_sha256:
        raise RuntimeError(
            f"existing output protocol differs from this run: {stored} != {expected_sha256}"
        )
    return protocol


def _validate_training_instrumentation(
    result: dict,
    path: Path,
    expected_preflight: dict[str, object] | None = None,
) -> None:
    field_names = (
        "training_history",
        "training_exposure_records",
        "training_prior_records",
        "exemplar_records",
        "router_records",
    )
    values = {name: result.get(name) for name in field_names}
    if any(not isinstance(value, dict) for value in values.values()):
        raise RuntimeError(f"result lacks training instrumentation objects: {path}")
    checkpoints = result.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise RuntimeError(f"result lacks task checkpoints: {path}")
    seen = checkpoints[-1].get("seen_classes")
    if not isinstance(seen, list) or not all(isinstance(value, int) for value in seen):
        raise RuntimeError(f"result final class axis is invalid: {path}")
    expected_keys = {str(value) for value in seen}
    expected_families: dict[str, dict] | None = None
    if expected_preflight is not None:
        from .exposure_preflight import validate_exposure_preflight

        preflight = validate_exposure_preflight(expected_preflight)
        expected_families = {
            str(family["class_id"]): family
            for task in preflight["tasks_detail"]
            for family in task["families"]
        }
        if set(expected_families) != expected_keys:
            raise RuntimeError(f"protocol exposure preflight class coverage mismatch: {path}")
    for name, value in values.items():
        if set(value) != expected_keys:
            raise RuntimeError(f"result {name} class coverage mismatch: {path}")

    for class_id in expected_keys:
        exposure = values["training_exposure_records"][class_id]
        prior = values["training_prior_records"][class_id]
        history = values["training_history"][class_id]
        if not isinstance(exposure, dict) or not isinstance(prior, dict) or not isinstance(
            history, list
        ):
            raise RuntimeError(f"result class {class_id} instrumentation is invalid: {path}")
        epochs = exposure.get("epochs")
        candidates = exposure.get("negative_candidate_rows")
        positives = exposure.get("positive_population_rows")
        selected = exposure.get("selected_negative_rows_per_epoch")
        if (
            not isinstance(epochs, int)
            or epochs < 0
            or not isinstance(candidates, int)
            or candidates <= 0
            or not isinstance(positives, int)
            or positives <= 0
            or not isinstance(selected, int)
            or not 0 < selected <= candidates
            or len(history) != epochs
        ):
            raise RuntimeError(f"result class {class_id} exposure counts are invalid: {path}")
        dtype = _exposure_counter_dtype(epochs)
        if (
            exposure.get("negative_exposure_counter_dtype") != np.dtype(dtype).name
            or exposure.get("negative_exposure_counter_capacity")
            != int(np.iinfo(dtype).max)
        ):
            raise RuntimeError(f"result class {class_id} exposure-counter contract failed: {path}")
        histogram_raw = exposure.get("negative_exposure_multiplicity_histogram")
        if not isinstance(histogram_raw, dict):
            raise RuntimeError(f"result class {class_id} exposure histogram is invalid: {path}")
        try:
            histogram = {int(key): int(value) for key, value in histogram_raw.items()}
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"result class {class_id} exposure histogram is invalid: {path}"
            ) from error
        if (
            set(histogram) != set(range(epochs + 1))
            or any(value < 0 for value in histogram.values())
            or sum(histogram.values()) != candidates
        ):
            raise RuntimeError(f"result class {class_id} exposure histogram accounting failed: {path}")
        positive_total = positives * epochs
        negative_total = sum(
            multiplicity * count for multiplicity, count in histogram.items()
        )
        if (
            negative_total != selected * epochs
            or exposure.get("positive_exposures_used_by_loss") != positive_total
            or exposure.get("negative_exposures_used_by_loss") != negative_total
            or exposure.get("negative_unique_sampler_units_across_epochs")
            != candidates - histogram[0]
        ):
            raise RuntimeError(f"result class {class_id} total exposure accounting failed: {path}")
        sources = exposure.get("negative_sources")
        if (
            not isinstance(sources, list)
            or any(not isinstance(source, dict) for source in sources)
            or sum(int(source.get("rows", -1)) for source in sources) != candidates
        ):
            raise RuntimeError(f"result class {class_id} negative-source accounting failed: {path}")
        if (
            prior.get("algorithm") != "binary_count_exposure_prior_offset_v1"
            or prior.get("positive_exposures") != positive_total
            or prior.get("negative_exposures") != negative_total
            or prior.get("per_epoch_positive_rows") != positives
            or prior.get("per_epoch_negative_rows") != selected
        ):
            raise RuntimeError(f"result class {class_id} prior/exposure binding failed: {path}")
        expected_log = math.log(positives / selected)
        if not math.isclose(
            float(prior.get("log_positive_to_negative_exposure_ratio")),
            expected_log,
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            float(prior.get("offset_applied_to_positive_margin")),
            -expected_log,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"result class {class_id} prior offset formula failed: {path}")
        for epoch in history:
            epoch_exposure = epoch.get("exposure") if isinstance(epoch, dict) else None
            if not isinstance(epoch_exposure, dict):
                raise RuntimeError(f"result class {class_id} epoch exposure is invalid: {path}")
            source_records = epoch_exposure.get("negative_selected_by_source")
            if (
                epoch.get("positive_rows") != positives
                or epoch.get("negative_rows") != selected
                or not isinstance(source_records, list)
                or sum(int(item.get("selected_rows", -1)) for item in source_records)
                != selected
            ):
                raise RuntimeError(f"result class {class_id} epoch exposure accounting failed: {path}")
        if expected_families is not None:
            planned = expected_families[class_id]
            planned_sources = [
                {
                    "source_kind": source["source_kind"],
                    "class_id": source["class_id"],
                    "rows": source["rows"],
                }
                for source in planned["negative_candidate_pool"]["sources"]
            ]
            actual_sources = [
                {
                    "source_kind": source.get("source_kind"),
                    "class_id": source.get("class_id"),
                    "rows": source.get("rows"),
                }
                for source in sources
            ]
            planned_counts = {
                "task_index": planned["task_index"],
                "positive_population_rows": planned["positive_rows_per_epoch"],
                "negative_candidate_rows": planned["negative_candidate_pool"][
                    "rows"
                ],
                "desired_negative_rows_per_epoch": planned[
                    "desired_negative_rows_per_epoch"
                ],
                "selected_negative_rows_per_epoch": planned[
                    "selected_negative_rows_per_epoch"
                ],
                "candidate_limited": planned["candidate_limited"],
                "positive_rows_per_optimizer_step": planned[
                    "positive_rows_per_optimizer_step"
                ],
                "positive_exposures_used_by_loss": planned[
                    "positive_exposures_total"
                ],
                "negative_exposures_used_by_loss": planned[
                    "negative_exposures_total"
                ],
                "negative_exposure_counter_dtype": planned[
                    "exposure_counter_dtype"
                ],
                "negative_exposure_counter_capacity": planned[
                    "exposure_counter_capacity"
                ],
            }
            if (
                any(exposure.get(key) != value for key, value in planned_counts.items())
                or actual_sources != planned_sources
                or not math.isclose(
                    float(prior["positive_exposure_prior"]),
                    float(planned["positive_exposure_prior"]),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                or not math.isclose(
                    float(prior["log_positive_to_negative_exposure_ratio"]),
                    float(planned["log_positive_to_negative_exposure_ratio"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise RuntimeError(f"result class {class_id} differs from exposure preflight: {path}")
            for epoch in history:
                epoch_exposure = epoch["exposure"]
                if (
                    epoch_exposure.get("optimizer_steps")
                    != planned["optimizer_steps_per_epoch"]
                    or epoch_exposure.get("zero_negative_steps")
                    != planned["zero_negative_steps_per_epoch"]
                    or epoch_exposure.get("negative_batch_rows_min")
                    != planned["negative_batch_rows_min"]
                    or epoch_exposure.get("negative_batch_rows_max")
                    != planned["negative_batch_rows_max"]
                ):
                    raise RuntimeError(
                        f"result class {class_id} optimizer schedule differs from preflight: {path}"
                    )

    invariants = result.get("training_invariants")
    if not isinstance(invariants, dict):
        raise RuntimeError(f"result lacks training invariants: {path}")
    training_record = {
        "normalization": result.get("normalization"),
        "pretrain_history": result.get("pretrain_history"),
        **values,
    }
    if invariants.get("training_record_sha256") != canonical_sha256(training_record):
        raise RuntimeError(f"result training-record invariant hash mismatch: {path}")


def _validate_evaluation_instrumentation(
    result: dict,
    path: Path,
    *,
    expected_metric_profile: str | None = None,
    expected_normal_class_id: int | None = None,
) -> list[str]:
    metric_profile = expected_metric_profile or result.get(
        "metric_profile", NIDS_METRIC_PROFILE
    )
    if metric_profile not in {NIDS_METRIC_PROFILE, GENERIC_METRIC_PROFILE}:
        raise RuntimeError(f"result metric profile is invalid: {path}")
    if expected_metric_profile is None:
        expected_normal_class_id = result.get("normal_class_id")
    if metric_profile == NIDS_METRIC_PROFILE and not isinstance(
        expected_normal_class_id, int
    ):
        raise RuntimeError(f"NIDS result lacks a normal class identifier: {path}")
    if (
        metric_profile == GENERIC_METRIC_PROFILE
        and expected_normal_class_id is not None
    ):
        raise RuntimeError(f"generic result declares a normal class identifier: {path}")
    summary = result.get("summary")
    if not isinstance(summary, dict) or not isinstance(summary.get("views"), dict):
        raise RuntimeError(f"result lacks schema-v2 view summaries: {path}")
    view_names = list(summary["views"])
    if not view_names or view_names[0] != "official" or len(set(view_names)) != len(
        view_names
    ):
        raise RuntimeError(f"result evaluation-view order is invalid: {path}")
    for view_name, arms in summary["views"].items():
        if not isinstance(view_name, str) or not isinstance(arms, dict) or set(
            arms
        ) != set(ARMS):
            raise RuntimeError(f"result view arm registry mismatch: {path}")
    checkpoints = result.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise RuntimeError(f"result lacks evaluation checkpoints: {path}")
    previous_seen: list[int] = []
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        seen = checkpoint.get("seen_classes") if isinstance(checkpoint, dict) else None
        views = checkpoint.get("views") if isinstance(checkpoint, dict) else None
        if (
            not isinstance(seen, list)
            or checkpoint.get("checkpoint") != checkpoint_index
            or not all(isinstance(value, int) for value in seen)
            or len(set(seen)) != len(seen)
            or seen[: len(previous_seen)] != previous_seen
            or len(seen) <= len(previous_seen)
            or not isinstance(views, dict)
            or list(views) != view_names
        ):
            raise RuntimeError(f"result checkpoint view/class axis mismatch: {path}")
        previous_seen = list(seen)
        width = len(seen)
        matrices: dict[str, dict[str, np.ndarray]] = {}
        for view_name in view_names:
            view = views[view_name]
            arms = view.get("arms") if isinstance(view, dict) else None
            if not isinstance(arms, dict) or set(arms) != set(ARMS):
                raise RuntimeError(f"result checkpoint arm registry mismatch: {path}")
            matrices[view_name] = {}
            expected_support: list[int] | None = None
            for arm, metrics in arms.items():
                try:
                    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
                except (KeyError, TypeError, ValueError) as error:
                    raise RuntimeError(f"result confusion matrix is invalid: {path}") from error
                if matrix.shape != (width, width) or np.any(matrix < 0):
                    raise RuntimeError(f"result confusion matrix shape/count is invalid: {path}")
                support = matrix.sum(axis=1).tolist()
                if expected_support is None:
                    expected_support = support
                elif support != expected_support:
                    raise RuntimeError(f"result evaluation arms disagree on support: {path}")
                total = int(matrix.sum())
                expected_accuracy = float(np.trace(matrix) / total) if total else 0.0
                support_array = matrix.sum(axis=1)
                predicted_array = matrix.sum(axis=0)
                true_positive = np.diag(matrix)
                precision = np.divide(
                    true_positive,
                    predicted_array,
                    out=np.zeros(width, dtype=np.float64),
                    where=predicted_array > 0,
                )
                recall = np.divide(
                    true_positive,
                    support_array,
                    out=np.zeros(width, dtype=np.float64),
                    where=support_array > 0,
                )
                f1 = np.divide(
                    2.0 * precision * recall,
                    precision + recall,
                    out=np.zeros(width, dtype=np.float64),
                    where=(precision + recall) > 0,
                )
                expected_common = {
                    "accuracy": expected_accuracy,
                    "macro_f1": float(f1.mean()),
                    "balanced_accuracy": float(recall.mean()),
                }
                try:
                    common_matches = all(
                        math.isclose(
                            float(metrics.get(name)),
                            value,
                            rel_tol=0.0,
                            abs_tol=1e-15,
                        )
                        for name, value in expected_common.items()
                    )
                except (TypeError, ValueError):
                    common_matches = False
                if metrics.get("total_rows") != total or not common_matches:
                    raise RuntimeError(f"result stored metrics disagree with confusion matrix: {path}")
                binary = metrics.get("binary_detection")
                if metric_profile == GENERIC_METRIC_PROFILE:
                    if "binary_detection" in metrics:
                        raise RuntimeError(
                            f"generic result contains NIDS-only binary metrics: {path}"
                        )
                else:
                    if not isinstance(binary, dict) or expected_normal_class_id not in seen:
                        raise RuntimeError(f"NIDS binary metrics are invalid: {path}")
                    normal_index = seen.index(expected_normal_class_id)
                    normal_support = int(support_array[normal_index])
                    false_positives = int(
                        normal_support - matrix[normal_index, normal_index]
                    )
                    attack_indices = [
                        index for index in range(width) if index != normal_index
                    ]
                    attack_support = int(support_array[attack_indices].sum())
                    detected = int(
                        matrix[np.ix_(attack_indices, attack_indices)].sum()
                    )
                    expected_binary = {
                        "normal_class_id": expected_normal_class_id,
                        "benign_false_positive_rate": (
                            false_positives / normal_support if normal_support else 0.0
                        ),
                        "benign_false_positives": false_positives,
                        "benign_support": normal_support,
                        "attack_detection_recall": (
                            detected / attack_support if attack_support else 0.0
                        ),
                        "attacks_predicted_as_attack": detected,
                        "attack_support": attack_support,
                    }
                    try:
                        binary_matches = all(
                            binary.get(name) == value
                            if isinstance(value, int)
                            else math.isclose(
                                float(binary.get(name)),
                                value,
                                rel_tol=0.0,
                                abs_tol=1e-15,
                            )
                            for name, value in expected_binary.items()
                        )
                    except (TypeError, ValueError):
                        binary_matches = False
                    if not binary_matches:
                        raise RuntimeError(
                            f"NIDS binary metrics disagree with confusion matrix: {path}"
                        )
                matrices[view_name][arm] = matrix
            expected_support = expected_support or []
            if (
                view.get("support_by_class")
                != {
                    str(class_id): int(expected_support[index])
                    for index, class_id in enumerate(seen)
                }
                or view.get("test_rows") != sum(expected_support)
            ):
                raise RuntimeError(f"result view support accounting failed: {path}")
        decomposition = checkpoint.get("view_decomposition")
        if not isinstance(decomposition, dict) or set(decomposition) != set(
            view_names[1:]
        ):
            raise RuntimeError(f"result view decomposition registry mismatch: {path}")
        for view_name in view_names[1:]:
            record = decomposition[view_name]
            if (
                not isinstance(record, dict)
                or record.get("relationship")
                != "official_equals_retained_plus_excluded"
                or set(record.get("arms", {})) != set(ARMS)
            ):
                raise RuntimeError(f"result view decomposition contract failed: {path}")
            for arm in ARMS:
                arm_record = record["arms"][arm]
                excluded = np.asarray(
                    arm_record.get("excluded_confusion_matrix"), dtype=np.int64
                )
                if (
                    excluded.shape != (width, width)
                    or np.any(excluded < 0)
                    or arm_record.get("matrix_conservation_verified") is not True
                    or not np.array_equal(
                        matrices["official"][arm],
                        matrices[view_name][arm] + excluded,
                    )
                ):
                    raise RuntimeError(f"result confusion conservation failed: {path}")
        invariants = checkpoint.get("evaluation_invariants")
        if (
            not isinstance(invariants, dict)
            or invariants.get("state_unchanged") is not True
            or invariants.get("rng_unchanged") is not True
            or invariants.get("state_before_sha256")
            != invariants.get("state_after_sha256")
            or invariants.get("rng_before_sha256") != invariants.get("rng_after_sha256")
        ):
            raise RuntimeError(f"result evaluation state/RNG invariant failed: {path}")
    return view_names


def _validate_result_file(
    path: Path,
    *,
    dataset: str,
    seed: int,
    protocol_sha256: str,
    expected_problem_type: str | None = None,
    expected_metric_profile: str | None = None,
    expected_task_semantics: str | None = None,
    expected_normal_class_id: int | None = None,
    expected_view_order: Sequence[str] | None = None,
    expected_exposure_preflight: dict[str, object] | None = None,
) -> dict:
    result = _load_json_object(path)
    identity = {
        "dataset": result.get("dataset"),
        "seed": result.get("seed"),
        "protocol_sha256": result.get("protocol_sha256"),
    }
    expected = {
        "dataset": dataset,
        "seed": int(seed),
        "protocol_sha256": protocol_sha256,
    }
    semantic_expected = {
        "problem_type": expected_problem_type,
        "metric_profile": expected_metric_profile,
        "task_semantics": expected_task_semantics,
        "normal_class_id": expected_normal_class_id,
    }
    for name, value in semantic_expected.items():
        if value is not None or (
            name == "normal_class_id" and expected_problem_type is not None
        ):
            identity[name] = result.get(name)
            expected[name] = value
    if identity != expected:
        raise RuntimeError(
            f"existing result identity mismatch in {path}: {identity} != {expected}"
        )
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise RuntimeError(f"formal result schema is not version 2: {path}")
    view_order = _validate_evaluation_instrumentation(
        result,
        path,
        expected_metric_profile=expected_metric_profile,
        expected_normal_class_id=expected_normal_class_id,
    )
    if expected_view_order is not None and view_order != list(expected_view_order):
        raise RuntimeError(f"result evaluation-view order differs from protocol: {path}")
    _validate_training_instrumentation(
        result, path, expected_preflight=expected_exposure_preflight
    )
    recomputed_summary = _summarise_checkpoints(
        result["checkpoints"], len(result["checkpoints"])
    )
    if canonical_sha256(recomputed_summary) != canonical_sha256(result["summary"]):
        raise RuntimeError(f"result per-seed summary does not match checkpoints: {path}")
    stored = result.get("deterministic_result_sha256")
    actual = canonical_sha256(_without_timing(result))
    if stored != actual:
        raise RuntimeError(f"existing result has an invalid deterministic hash: {path}")
    return result


def _emit_seed_event(
    event: str,
    *,
    dataset: str,
    seed: int,
    protocol_sha256: str,
    elapsed_seconds: float,
    **details: object,
) -> None:
    payload = {
        "event": event,
        "dataset": dataset,
        "seed": int(seed),
        "protocol_sha256": protocol_sha256,
        "elapsed_seconds": float(elapsed_seconds),
        **details,
    }
    print("OFRA_RUN_EVENT " + json.dumps(payload, ensure_ascii=False), flush=True)


def _run_manifest_transaction(
    manifest_path: str | Path,
    *,
    seeds: Sequence[int],
    output_dir: str | Path,
    config: RunConfig | None = None,
    evaluation_view_paths: Sequence[str | Path] = (),
) -> dict:
    config = config or RunConfig()
    config.validate()
    seeds = [int(seed) for seed in seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty list of unique integers")
    manifest = load_manifest(manifest_path, verify_hashes=config.verify_shard_hashes)
    evaluation_views = [
        load_evaluation_view(path, manifest, verify_hashes=config.verify_shard_hashes)
        for path in evaluation_view_paths
    ]
    names = [view.name for view in evaluation_views]
    if len(names) != len(set(names)):
        raise ValueError("evaluation-view names must be unique")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Establish the requested numerical mode before recording the environment.
    # Each seed is reset again immediately before its training trajectory.
    _seed_process(seeds[0], config.deterministic)
    protocol = _protocol(manifest, config, seeds, evaluation_views)
    protocol_sha256 = canonical_sha256(protocol)
    protocol["protocol_sha256"] = protocol_sha256
    protocol_path = output / "protocol.json"
    existing_results = list(output.glob("result_seed_*.json"))
    expected_result_names = {f"result_seed_{seed}.json" for seed in seeds}
    unexpected_results = sorted(
        path.name for path in existing_results if path.name not in expected_result_names
    )
    if unexpected_results:
        raise RuntimeError(
            f"output contains result files outside the requested protocol: {unexpected_results}"
        )
    if protocol_path.exists():
        _validate_protocol_file(protocol_path, protocol_sha256)
    elif existing_results:
        raise RuntimeError("result files exist without protocol.json; refusing unsafe resume")
    else:
        _atomic_write_json(protocol_path, protocol)
    results = []
    for seed in seeds:
        seed_started = time.perf_counter()
        result_path = output / f"result_seed_{seed}.json"
        _emit_seed_event(
            "start",
            dataset=manifest.dataset,
            seed=seed,
            protocol_sha256=protocol_sha256,
            elapsed_seconds=0.0,
            resume_candidate=result_path.exists(),
        )
        try:
            if result_path.exists():
                result = _validate_result_file(
                    result_path,
                    dataset=manifest.dataset,
                    seed=seed,
                    protocol_sha256=protocol_sha256,
                    expected_problem_type=manifest.problem_type,
                    expected_metric_profile=manifest.metric_profile,
                    expected_task_semantics=manifest.task_semantics,
                    expected_normal_class_id=manifest.normal_class_id,
                    expected_view_order=protocol["evaluation"]["view_order"],
                    expected_exposure_preflight=protocol["exposure_preflight"],
                )
                _emit_seed_event(
                    "skip",
                    dataset=manifest.dataset,
                    seed=seed,
                    protocol_sha256=protocol_sha256,
                    elapsed_seconds=time.perf_counter() - seed_started,
                    reason="validated_existing_result",
                    deterministic_result_sha256=result["deterministic_result_sha256"],
                )
            else:
                result = run_seed(
                    manifest,
                    config,
                    seed,
                    protocol_sha256,
                    evaluation_views=evaluation_views,
                )
                generated_view_order = _validate_evaluation_instrumentation(
                    result,
                    result_path,
                    expected_metric_profile=manifest.metric_profile,
                    expected_normal_class_id=manifest.normal_class_id,
                )
                if generated_view_order != protocol["evaluation"]["view_order"]:
                    raise RuntimeError(
                        "generated result evaluation-view order differs from protocol"
                    )
                _validate_training_instrumentation(
                    result,
                    result_path,
                    expected_preflight=protocol["exposure_preflight"],
                )
                _atomic_write_json(result_path, result)
                _emit_seed_event(
                    "end",
                    dataset=manifest.dataset,
                    seed=seed,
                    protocol_sha256=protocol_sha256,
                    elapsed_seconds=time.perf_counter() - seed_started,
                    status="completed",
                    deterministic_result_sha256=result["deterministic_result_sha256"],
                )
        except BaseException as error:
            _emit_seed_event(
                "fail",
                dataset=manifest.dataset,
                seed=seed,
                protocol_sha256=protocol_sha256,
                elapsed_seconds=time.perf_counter() - seed_started,
                error_type=type(error).__name__,
                error=str(error),
            )
            raise
        results.append(result)
    summary = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "dataset": manifest.dataset,
        "problem_type": manifest.problem_type,
        "metric_profile": manifest.metric_profile,
        "task_semantics": manifest.task_semantics,
        "normal_class_id": manifest.normal_class_id,
        "protocol_sha256": protocol_sha256,
        "aggregate": _aggregate(results),
        "result_files": [f"result_seed_{seed}.json" for seed in seeds],
        "deterministic_result_sha256": {
            str(result["seed"]): result["deterministic_result_sha256"]
            for result in results
        },
    }
    _atomic_write_json(output / "summary.json", summary)
    return {"protocol": protocol, "results": results, "summary": summary}


def run_manifest(
    manifest_path: str | Path,
    *,
    seeds: Sequence[int],
    output_dir: str | Path,
    config: RunConfig | None = None,
    evaluation_view_paths: Sequence[str | Path] = (),
) -> dict:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with OutputDirectoryLock(output):
        return _run_manifest_transaction(
            manifest_path,
            seeds=seeds,
            output_dir=output,
            config=config,
            evaluation_view_paths=evaluation_view_paths,
        )
