"""Synthetic-only, immutable reference for increment-available train sampling.

No file, test-set, network, feature-matrix, or future-class-count interface is
provided. This is not a production data builder or a model-training program.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


SAMPLING_DOMAIN = "ofra-prospective-sampling-reference-v1"
MAX_REFERENCE_ROWS = 10_000


def _nonnegative_integer(value: object, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class Policy:
    """Predeclared policy; no dataset-wide counts or manifest hashes are inputs."""

    seed: int = 42
    cap_mode: str = "task0_max_attack_fit"
    fixed_cap: int | None = None
    calibration_numerator: int = 1
    calibration_denominator: int = 10

    def __post_init__(self) -> None:
        _nonnegative_integer(self.seed, "seed")
        _nonnegative_integer(self.calibration_numerator, "calibration_numerator")
        _nonnegative_integer(self.calibration_denominator, "calibration_denominator")
        if not 0 <= self.calibration_numerator < self.calibration_denominator:
            raise ValueError("calibration fraction must lie in [0, 1)")
        if self.cap_mode not in {"task0_max_attack_fit", "fixed"}:
            raise ValueError("unsupported cap_mode")
        if self.cap_mode == "fixed":
            if type(self.fixed_cap) is not int or self.fixed_cap <= 0:
                raise ValueError("fixed mode requires a predeclared positive fixed_cap")
        elif self.fixed_cap is not None:
            raise ValueError("Task-0 mode cannot accept a fallback fixed_cap")


@dataclass(frozen=True, slots=True)
class RowRef:
    """Stable, globally unique source row ID and its first availability."""

    row_id: str
    available_increment: int
    partition: str = "train"

    def __post_init__(self) -> None:
        if type(self.row_id) is not str or not self.row_id or len(self.row_id) > 128:
            raise ValueError("row_id must be a nonempty string of at most 128 characters")
        _nonnegative_integer(self.available_increment, "row available_increment")
        if self.partition != "train":
            raise ValueError("only source-train rows are admissible")


@dataclass(frozen=True, slots=True)
class TrainBatch:
    """One newly released class cohort, never a full-data manifest.

    The train-only cohort digest is recorded as provenance, not used for RNG.
    Its factual origin must be verified by a future governed ingestion layer.
    """

    class_id: int
    role: str
    available_increment: int
    rows: tuple[RowRef, ...]
    source_train_sha256: str
    partition: str = "train"

    def __post_init__(self) -> None:
        _nonnegative_integer(self.class_id, "class_id")
        _nonnegative_integer(self.available_increment, "batch available_increment")
        if self.role not in {"normal", "attack"}:
            raise ValueError("role must be normal or attack")
        if self.partition != "train":
            raise ValueError("only source-train batches are admissible")
        if type(self.rows) is not tuple or any(type(row) is not RowRef for row in self.rows):
            raise TypeError("rows must be a tuple of RowRef values")
        digest = self.source_train_sha256
        if type(digest) is not str or len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest
        ):
            raise ValueError("source_train_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class ClassSelection:
    class_id: int
    role: str
    increment: int
    source_train_sha256: str
    input_ids: tuple[str, ...]
    fit_ids: tuple[str, ...]
    calibration_ids: tuple[str, ...]
    omitted_ids: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class State:
    policy: Policy
    increment: int = -1
    frozen_normal_cap: int | None = None
    selections: tuple[ClassSelection, ...] = ()


def initial_state(policy: Policy) -> State:
    if type(policy) is not Policy:
        raise TypeError("policy must be a Policy value")
    return State(policy=policy)


def _rank(seed: int, stage: str, class_id: int, row_id: str) -> bytes:
    # No class count, current increment, full manifest, test, or source digest.
    payload = json.dumps(
        [SAMPLING_DOMAIN, seed, stage, class_id, row_id],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _calibration_count(n_rows: int, policy: Policy) -> int:
    if n_rows <= 1 or policy.calibration_numerator == 0:
        return 0
    return min(
        n_rows - 1,
        max(1, n_rows * policy.calibration_numerator // policy.calibration_denominator),
    )


def _partition(batch: TrainBatch, policy: Policy) -> tuple[tuple[str, ...], tuple[str, ...]]:
    ranked = sorted(
        (row.row_id for row in batch.rows),
        key=lambda row_id: (_rank(policy.seed, "calibration", batch.class_id, row_id), row_id),
    )
    n_calibration = _calibration_count(len(ranked), policy)
    return tuple(ranked[:n_calibration]), tuple(ranked[n_calibration:])


def append_increment(state: State, *, increment: int, batches: tuple[TrainBatch, ...]) -> State:
    """Freeze Task-0 cap once and append new-class selections without revisions.

    Only new classes released at exactly this increment may be supplied. This
    reference intentionally rejects within-class append/update semantics. It
    rejects future inputs rather than receiving a full manifest and filtering
    it. A caller can still mislabel provenance: types are not a security wall.
    """
    if type(state) is not State:
        raise TypeError("state must come from initial_state/append_increment")
    _nonnegative_integer(increment, "increment")
    if increment != state.increment + 1:
        raise ValueError("increments must be consecutive and append-only")
    if type(batches) is not tuple or not batches or any(type(batch) is not TrainBatch for batch in batches):
        raise TypeError("batches must be a nonempty tuple of TrainBatch values")

    class_ids = {selection.class_id for selection in state.selections}
    row_ids = {row_id for selection in state.selections for row_id in selection.input_ids}
    normal_ids = {selection.class_id for selection in state.selections if selection.role == "normal"}
    for batch in batches:
        if batch.available_increment != increment:
            raise ValueError("batch is not a newly available class at the current increment")
        if batch.class_id in class_ids:
            raise ValueError("class_id collision or attempted historical class rewrite")
        class_ids.add(batch.class_id)
        if batch.role == "normal":
            if increment != 0 or normal_ids:
                raise ValueError("exactly one normal class must be introduced at Task 0")
            normal_ids.add(batch.class_id)
        for row in batch.rows:
            if row.available_increment > increment:
                raise ValueError("future train row is not available at the current increment")
            if row.row_id in row_ids:
                raise ValueError("row_id collision within or across current/historical cohorts")
            row_ids.add(row.row_id)
    if not normal_ids:
        raise ValueError("Task 0 must declare exactly one normal class")
    if len(row_ids) > MAX_REFERENCE_ROWS:
        raise ValueError("synthetic reference row limit exceeded; not a real-data builder")

    ordered = sorted(batches, key=lambda batch: batch.class_id)
    partitions = {batch.class_id: _partition(batch, state.policy) for batch in ordered}
    cap = state.frozen_normal_cap
    if increment == 0:
        if state.policy.cap_mode == "fixed":
            cap = state.policy.fixed_cap
        else:
            counts = [len(partitions[batch.class_id][1]) for batch in ordered if batch.role == "attack"]
            if not counts or max(counts) == 0:
                raise ValueError(
                    "no Task-0 attack fitting rows; stop or separately predeclare a fixed-cap arm"
                )
            cap = max(counts)
    if type(cap) is not int or cap <= 0:
        raise ValueError("state lacks a valid frozen normal cap")

    additions = []
    for batch in ordered:
        calibration, fit_pool = partitions[batch.class_id]
        ranked_fit = sorted(
            fit_pool,
            key=lambda row_id: (_rank(state.policy.seed, "normal-fit", batch.class_id, row_id), row_id),
        )
        target = min(cap, len(ranked_fit)) if batch.role == "normal" else len(ranked_fit)
        warnings = []
        if not batch.rows:
            warnings.append("empty_class_no_fit_or_calibration_support")
        elif len(batch.rows) == 1:
            warnings.append("singleton_class_no_calibration_support")
        additions.append(
            ClassSelection(
                class_id=batch.class_id,
                role=batch.role,
                increment=increment,
                source_train_sha256=batch.source_train_sha256,
                input_ids=tuple(sorted(row.row_id for row in batch.rows)),
                fit_ids=tuple(sorted(ranked_fit[:target])),
                calibration_ids=tuple(sorted(calibration)),
                omitted_ids=tuple(sorted(ranked_fit[target:])),
                warnings=tuple(warnings),
            )
        )
    return State(
        policy=state.policy,
        increment=increment,
        frozen_normal_cap=cap,
        selections=state.selections + tuple(additions),
    )
