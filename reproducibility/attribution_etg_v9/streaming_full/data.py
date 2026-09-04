from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np


EVALUATION_VIEW_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
EVALUATION_VIEW_ALGORITHM = "exclude_test_if_exact_train_float32_feature_exists_v1"
EVALUATION_VIEW_EQUALITY_KEY = (
    "complete canonical little-endian float32 feature-row bytes; "
    "no probabilistic hash is used for equality"
)

INTRUSION_DETECTION = "intrusion_detection"
APPLICATION_CLASSIFICATION = "application_classification"
NIDS_METRIC_PROFILE = "nids_multiclass_with_binary_detection"
GENERIC_METRIC_PROFILE = "generic_multiclass"
CLASS_INCREMENTAL = "class_incremental"

SEMANTIC_PROFILE_REGISTRY = {
    INTRUSION_DETECTION: NIDS_METRIC_PROFILE,
    APPLICATION_CLASSIFICATION: GENERIC_METRIC_PROFILE,
}


def resolve_manifest_semantics(
    raw: dict,
    *,
    context: str = "manifest",
) -> tuple[str, str, str, int | None]:
    """Validate and normalize a manifest's task and metric semantics.

    Manifests written before semantic profiles were introduced omitted both
    fields.  Those manifests retain their original intrusion-detection
    interpretation.  Partially declared or mismatched profiles are rejected so
    that a generic dataset can never silently acquire NIDS-only metrics.
    """

    problem_declared = "problem_type" in raw
    profile_declared = "metric_profile" in raw
    if problem_declared != profile_declared:
        raise ValueError(
            f"{context} problem_type and metric_profile must be declared together"
        )
    if problem_declared:
        problem_type = raw.get("problem_type")
        metric_profile = raw.get("metric_profile")
    else:
        problem_type = INTRUSION_DETECTION
        metric_profile = NIDS_METRIC_PROFILE
    if problem_type not in SEMANTIC_PROFILE_REGISTRY:
        raise ValueError(
            f"{context} problem_type must be one of "
            f"{sorted(SEMANTIC_PROFILE_REGISTRY)}"
        )
    expected_profile = SEMANTIC_PROFILE_REGISTRY[problem_type]
    if metric_profile != expected_profile:
        raise ValueError(
            f"{context} metric_profile must be {expected_profile!r} for "
            f"problem_type {problem_type!r}"
        )

    task_semantics = raw.get("task_semantics", CLASS_INCREMENTAL)
    if task_semantics != CLASS_INCREMENTAL:
        raise ValueError(
            f"{context} task_semantics must be {CLASS_INCREMENTAL!r}"
        )

    normal_declared = "normal_class_id" in raw
    normal_class_id = raw.get("normal_class_id")
    if problem_type == INTRUSION_DETECTION:
        if (
            isinstance(normal_class_id, bool)
            or not isinstance(normal_class_id, int)
            or normal_class_id < 0
        ):
            raise ValueError(
                f"{context} normal_class_id must be a non-negative integer "
                "for intrusion_detection"
            )
    elif not normal_declared or normal_class_id is not None:
        raise ValueError(
            f"{context} normal_class_id must be explicitly null for "
            "application_classification"
        )
    return problem_type, metric_profile, task_semantics, normal_class_id


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(payload)


def _close_memmap(array: np.ndarray) -> None:
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


def derived_seed(master_seed: int, dataset: str, *parts: object) -> int:
    payload = "|".join([str(master_seed), dataset, *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little")


@dataclass(frozen=True)
class Shard:
    path: Path
    rows: int
    sha256: str | None = None


@dataclass(frozen=True)
class ClassRecord:
    class_id: int
    name: str
    train: tuple[Shard, ...]
    test: tuple[Shard, ...]


@dataclass(frozen=True)
class DatasetManifest:
    path: Path
    dataset: str
    feature_dim: int
    problem_type: str
    metric_profile: str
    task_semantics: str
    normal_class_id: int | None
    tasks: tuple[tuple[int, ...], ...]
    classes: tuple[ClassRecord, ...]
    manifest_sha256: str
    source_provenance: dict[str, object]

    @property
    def class_map(self) -> dict[int, ClassRecord]:
        return {record.class_id: record for record in self.classes}

    @property
    def class_names(self) -> dict[int, str]:
        return {record.class_id: record.name for record in self.classes}


@dataclass(frozen=True)
class EvaluationMaskShard:
    path: Path
    rows: int
    sha256: str
    true_count: int
    false_count: int
    parent_relative_path: str
    parent_sha256: str


@dataclass(frozen=True)
class EvaluationView:
    path: Path
    name: str
    dataset: str
    feature_dim: int
    problem_type: str
    metric_profile: str
    task_semantics: str
    normal_class_id: int | None
    tasks: tuple[tuple[int, ...], ...]
    class_names: dict[int, str]
    masks: dict[int, tuple[EvaluationMaskShard, ...]]
    manifest_sha256: str
    canonical_sha256: str
    audit_provenance: dict[str, object]
    retained_rows: dict[int, int]
    excluded_rows: dict[int, int]


def _logical_shard_path(manifest: DatasetManifest, shard: Shard) -> str:
    try:
        return shard.path.relative_to(manifest.path.parent).as_posix()
    except ValueError:
        return shard.path.name


def dataset_logical_fingerprints(manifest: DatasetManifest) -> dict[str, str]:
    """Return path-independent logical fingerprints for train and official test."""

    identity = {
        "dataset": manifest.dataset,
        "feature_dim": manifest.feature_dim,
        "problem_type": manifest.problem_type,
        "metric_profile": manifest.metric_profile,
        "task_semantics": manifest.task_semantics,
        "normal_class_id": manifest.normal_class_id,
        "tasks": [list(task) for task in manifest.tasks],
        "classes": [
            {"id": record.class_id, "name": record.name}
            for record in manifest.classes
        ],
    }

    def split_record(split: str) -> dict[str, object]:
        return {
            **identity,
            "split": split,
            "shards": [
                {
                    "class_id": record.class_id,
                    "ordinal": ordinal,
                    "relative_path": _logical_shard_path(manifest, shard),
                    "rows": shard.rows,
                    "sha256": shard.sha256,
                }
                for record in manifest.classes
                for ordinal, shard in enumerate(getattr(record, split))
            ],
        }

    return {
        "train_logical_sha256": canonical_sha256(split_record("train")),
        "official_test_logical_sha256": canonical_sha256(split_record("test")),
    }


def _parse_shards(
    base: Path,
    raw: object,
    *,
    feature_dim: int,
    context: str,
    verify_hashes: bool,
) -> tuple[Shard, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{context} must be a non-empty list")
    shards: list[Shard] = []
    for index, item in enumerate(raw):
        if isinstance(item, str):
            item = {"path": item}
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError(f"{context}[{index}] must contain a string path")
        path = (base / item["path"]).resolve()
        if not path.is_file() or path.suffix.lower() != ".npy":
            raise FileNotFoundError(f"invalid shard path: {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            shape = array.shape
            dtype = array.dtype
            file_rows = len(array)
        finally:
            _close_memmap(array)
        if len(shape) != 2 or shape[1] != feature_dim:
            raise ValueError(f"{path} has shape {shape}; expected (*, {feature_dim})")
        if not np.issubdtype(dtype, np.number):
            raise ValueError(f"{path} must contain numeric features")
        rows = int(item.get("rows", file_rows))
        if rows != file_rows:
            raise ValueError(f"row count mismatch for {path}: manifest={rows}, file={file_rows}")
        expected_hash = item.get("sha256")
        if expected_hash is not None and not isinstance(expected_hash, str):
            raise ValueError(f"sha256 for {path} must be a string")
        if verify_hashes:
            if expected_hash is None:
                raise ValueError(f"verified manifests require SHA-256 for {path}")
            actual = sha256_file(path)
            if actual.lower() != expected_hash.lower():
                raise ValueError(f"SHA-256 mismatch for {path}")
        shards.append(Shard(path=path, rows=rows, sha256=expected_hash))
    return tuple(shards)


def _json_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON at {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _verify_fullcache_source(
    streaming_path: Path,
    streaming_sha256: str,
    streaming_source: dict,
    dataset: str,
    semantic_identity: tuple[str, str, str, int | None],
) -> dict[str, object]:
    fullcache_path = (streaming_path.parent / "manifest.json").resolve()
    if fullcache_path == streaming_path or not fullcache_path.is_file():
        raise FileNotFoundError(
            "ofra-fullcache source requires sibling manifest.json sidecar"
        )
    manifest = _json_object(fullcache_path, "fullcache manifest")
    canonical_record = manifest.get("canonical_manifest")
    if not isinstance(canonical_record, dict):
        raise ValueError("fullcache manifest lacks canonical_manifest")
    manifest_basis = dict(manifest)
    manifest_basis.pop("canonical_manifest", None)
    canonical_actual = canonical_sha256(manifest_basis)
    if canonical_record.get("sha256") != canonical_actual:
        raise ValueError("fullcache canonical_manifest SHA-256 mismatch")
    if manifest.get("tool") != "ofra-fullcache" or manifest.get("dataset") != dataset:
        raise ValueError("fullcache tool/dataset identity mismatch")
    fullcache_semantic_record = dict(manifest)
    if (
        "problem_type" not in fullcache_semantic_record
        and "metric_profile" not in fullcache_semantic_record
        and "normal_class_id" not in fullcache_semantic_record
    ):
        # Historical fullcache manifests stored the NIDS normal class only in
        # their streaming sidecar.  Supply that already-validated streaming
        # value solely for legacy semantic normalization.
        fullcache_semantic_record["normal_class_id"] = semantic_identity[3]
    fullcache_semantics = resolve_manifest_semantics(
        fullcache_semantic_record, context="fullcache manifest"
    )
    if fullcache_semantics != semantic_identity:
        raise ValueError("streaming/fullcache semantic identity mismatch")
    if manifest.get("uncapped") is not True:
        raise ValueError("fullcache manifest must declare uncapped=true")

    sidecars = manifest.get("sidecars")
    if not isinstance(sidecars, dict):
        raise ValueError("fullcache manifest sidecars must be an object")
    if sidecars.get("streaming_manifest.json") != streaming_sha256:
        raise ValueError("fullcache sidecar hash does not bind this streaming manifest")

    builder_source = manifest.get("builder_source")
    if not isinstance(builder_source, dict):
        raise ValueError("fullcache manifest lacks builder_source")
    builder_basis = dict(builder_source)
    builder_expected = builder_basis.pop("canonical_source_sha256", None)
    builder_actual = canonical_sha256(builder_basis)
    if builder_expected != builder_actual:
        raise ValueError("fullcache builder_source canonical SHA-256 mismatch")
    if streaming_source.get("builder_source_sha256") != builder_actual:
        raise ValueError("streaming source builder hash disagrees with fullcache manifest")

    if streaming_source.get("builder_version") != manifest.get("tool_version"):
        raise ValueError("streaming/fullcache builder version mismatch")
    if streaming_source.get("feature_schema_sha256") != manifest.get(
        "feature_schema_sha256"
    ):
        raise ValueError("streaming/fullcache feature schema hash mismatch")

    overlap_expected = sidecars.get("split_overlap_audit.json")
    if not isinstance(overlap_expected, str):
        raise ValueError("fullcache manifest lacks split-overlap sidecar hash")
    if streaming_source.get("split_overlap_audit_sha256") != overlap_expected:
        raise ValueError("streaming source split-overlap hash mismatch")
    overlap_path = (streaming_path.parent / "split_overlap_audit.json").resolve()
    if not overlap_path.is_file() or sha256_file(overlap_path) != overlap_expected:
        raise ValueError("split-overlap sidecar file SHA-256 mismatch")
    overlap = _json_object(overlap_path, "split-overlap audit")
    overlap_basis = dict(overlap)
    overlap_canonical = overlap_basis.pop("canonical_report_sha256", None)
    if overlap_canonical != canonical_sha256(overlap_basis):
        raise ValueError("split-overlap audit canonical SHA-256 mismatch")

    runner = manifest.get("streaming_runner")
    if not isinstance(runner, dict) or runner.get("relative_manifest") != streaming_path.name:
        raise ValueError("fullcache streaming_runner does not point to this manifest")
    if runner.get("normal_class_id") != semantic_identity[3]:
        raise ValueError("streaming/fullcache runner normal-class identity mismatch")
    runner_declares_problem = "problem_type" in runner
    runner_declares_profile = "metric_profile" in runner
    if runner_declares_problem != runner_declares_profile:
        raise ValueError(
            "fullcache streaming_runner problem_type and metric_profile must be "
            "declared together"
        )
    if runner_declares_problem and resolve_manifest_semantics(
        runner, context="fullcache streaming_runner"
    ) != semantic_identity:
        raise ValueError("streaming/fullcache runner semantic identity mismatch")
    return {
        "builder": "ofra-fullcache",
        "fullcache_manifest": {
            "path": str(fullcache_path),
            "sha256": sha256_file(fullcache_path),
            "canonical_sha256": canonical_actual,
        },
        "streaming_manifest_sha256": streaming_sha256,
        "builder_source_canonical_sha256": builder_actual,
        "split_overlap_audit": {
            "path": str(overlap_path),
            "sha256": overlap_expected,
            "canonical_sha256": overlap_canonical,
        },
    }


def load_manifest(path: str | Path, *, verify_hashes: bool = True) -> DatasetManifest:
    manifest_path = Path(path).resolve()
    raw_bytes = manifest_path.read_bytes()
    raw = json.loads(raw_bytes)
    if raw.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    dataset = raw.get("dataset")
    feature_dim = raw.get("feature_dim")
    problem_type, metric_profile, task_semantics, normal_class_id = (
        resolve_manifest_semantics(raw)
    )
    tasks_raw = raw.get("tasks")
    classes_raw = raw.get("classes")
    source_raw = raw.get("source", {})
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("manifest dataset must be a non-empty string")
    if not isinstance(feature_dim, int) or feature_dim <= 0:
        raise ValueError("manifest feature_dim must be positive")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise ValueError("manifest tasks must be a non-empty list")
    if not isinstance(classes_raw, list) or not classes_raw:
        raise ValueError("manifest classes must be a non-empty list")
    if not isinstance(source_raw, dict):
        raise ValueError("manifest source must be an object when provided")

    builder = source_raw.get("builder")
    if builder is not None and not isinstance(builder, str):
        raise ValueError("manifest source.builder must be a string")
    if builder == "ofra-fullcache":
        source_provenance = _verify_fullcache_source(
            manifest_path,
            sha256_bytes(raw_bytes),
            source_raw,
            dataset.strip(),
            (problem_type, metric_profile, task_semantics, normal_class_id),
        )
    else:
        source_provenance = {
            "builder": builder or "unspecified",
            "fullcache_manifest": None,
        }

    classes: list[ClassRecord] = []
    ids: list[int] = []
    base = manifest_path.parent
    for index, item in enumerate(classes_raw):
        if not isinstance(item, dict):
            raise ValueError(f"classes[{index}] must be an object")
        class_id = item.get("id")
        name = item.get("name")
        if not isinstance(class_id, int) or class_id < 0:
            raise ValueError(f"classes[{index}].id must be a non-negative integer")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"classes[{index}].name must be non-empty")
        train = _parse_shards(
            base,
            item.get("train"),
            feature_dim=feature_dim,
            context=f"classes[{index}].train",
            verify_hashes=verify_hashes,
        )
        test = _parse_shards(
            base,
            item.get("test"),
            feature_dim=feature_dim,
            context=f"classes[{index}].test",
            verify_hashes=verify_hashes,
        )
        ids.append(class_id)
        classes.append(ClassRecord(class_id, name.strip(), train, test))
    if len(set(ids)) != len(ids):
        raise ValueError("class identifiers must be unique")
    if sorted(ids) != list(range(len(ids))):
        raise ValueError("class identifiers must be contiguous from zero")

    tasks: list[tuple[int, ...]] = []
    flattened: list[int] = []
    for index, task in enumerate(tasks_raw):
        if not isinstance(task, list) or not task or not all(isinstance(x, int) for x in task):
            raise ValueError(f"tasks[{index}] must be a non-empty integer list")
        if len(set(task)) != len(task):
            raise ValueError(f"tasks[{index}] contains duplicate classes")
        tasks.append(tuple(task))
        flattened.extend(task)
    if sorted(flattened) != sorted(ids) or len(flattened) != len(ids):
        raise ValueError("tasks must contain every class exactly once")
    if problem_type == INTRUSION_DETECTION:
        if normal_class_id not in ids:
            raise ValueError("manifest normal_class_id must identify a declared class")
        if normal_class_id not in tasks[0]:
            raise ValueError(
                "normal_class_id must be present from Task 0 for checkpoint metrics"
            )
        if not any(class_id != normal_class_id for class_id in tasks[0]):
            raise ValueError(
                "Task 0 must include at least one attack class for binary metrics"
            )
    elif len(tasks[0]) < 2:
        raise ValueError(
            "Task 0 must include at least two classes for one-vs-rest training"
        )

    return DatasetManifest(
        path=manifest_path,
        dataset=dataset.strip(),
        feature_dim=feature_dim,
        problem_type=problem_type,
        metric_profile=metric_profile,
        task_semantics=task_semantics,
        normal_class_id=normal_class_id,
        tasks=tuple(tasks),
        classes=tuple(sorted(classes, key=lambda record: record.class_id)),
        manifest_sha256=sha256_bytes(raw_bytes),
        source_provenance=source_provenance,
    )


def _resolve_contained_path(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} must be a non-empty relative path")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the evaluation-view root") from error
    return candidate


def load_evaluation_view(
    path: str | Path,
    primary: DatasetManifest,
    *,
    verify_hashes: bool = True,
) -> EvaluationView:
    """Load a test-only row-selection view bound to one primary manifest."""

    view_path = Path(path).resolve()
    raw_bytes = view_path.read_bytes()
    raw = json.loads(raw_bytes)
    if not isinstance(raw, dict):
        raise ValueError("evaluation-view manifest must contain an object")
    if raw.get("schema_version") != 1 or raw.get("kind") != (
        "streaming_full_test_selection_view"
    ):
        raise ValueError("unsupported evaluation-view manifest schema/kind")
    canonical = raw.get("canonical_manifest")
    if not isinstance(canonical, dict):
        raise ValueError("evaluation-view manifest lacks canonical_manifest")
    basis = dict(raw)
    basis.pop("canonical_manifest", None)
    canonical_actual = canonical_sha256(basis)
    if canonical.get("sha256") != canonical_actual:
        raise ValueError("evaluation-view canonical SHA-256 mismatch")

    name = raw.get("name")
    if (
        not isinstance(name, str)
        or not EVALUATION_VIEW_NAME.fullmatch(name)
        or name == "official"
    ):
        raise ValueError("evaluation-view name is invalid or reserved")
    problem_type, metric_profile, task_semantics, normal_class_id = (
        resolve_manifest_semantics(raw, context="evaluation-view manifest")
    )
    expected_identity = {
        "dataset": primary.dataset,
        "feature_dim": primary.feature_dim,
        "tasks": [list(task) for task in primary.tasks],
    }
    identity = {key: raw.get(key) for key in expected_identity}
    if identity != expected_identity:
        raise ValueError(
            f"evaluation-view dataset/task identity mismatch: {identity}"
        )
    semantic_identity = {
        "problem_type": problem_type,
        "metric_profile": metric_profile,
        "task_semantics": task_semantics,
        "normal_class_id": normal_class_id,
    }
    expected_semantic_identity = {
        "problem_type": primary.problem_type,
        "metric_profile": primary.metric_profile,
        "task_semantics": primary.task_semantics,
        "normal_class_id": primary.normal_class_id,
    }
    if semantic_identity != expected_semantic_identity:
        raise ValueError(
            "evaluation-view problem/metric semantic identity mismatch"
        )

    primary_record = raw.get("primary")
    if not isinstance(primary_record, dict):
        raise ValueError("evaluation-view manifest lacks primary binding")
    fingerprints = dataset_logical_fingerprints(primary)
    expected_primary = {
        "streaming_manifest_sha256": primary.manifest_sha256,
        **fingerprints,
    }
    for key, expected in expected_primary.items():
        if primary_record.get(key) != expected:
            raise ValueError(f"evaluation-view primary {key} mismatch")
    fullcache = primary.source_provenance.get("fullcache_manifest")
    if isinstance(fullcache, dict):
        for key in ("sha256", "canonical_sha256"):
            if primary_record.get(f"fullcache_manifest_{key}") != fullcache.get(key):
                raise ValueError(f"evaluation-view fullcache {key} mismatch")
    split_audit = primary.source_provenance.get("split_overlap_audit")
    if isinstance(split_audit, dict):
        for key in ("sha256", "canonical_sha256"):
            if primary_record.get(f"split_overlap_audit_{key}") != split_audit.get(
                key
            ):
                raise ValueError(f"evaluation-view split-overlap audit {key} mismatch")

    selection = raw.get("selection")
    if (
        not isinstance(selection, dict)
        or selection.get("algorithm") != EVALUATION_VIEW_ALGORITHM
        or selection.get("equality_key") != EVALUATION_VIEW_EQUALITY_KEY
    ):
        raise ValueError("evaluation-view selection algorithm mismatch")
    audit_record = selection.get("audit")
    if not isinstance(audit_record, dict):
        raise ValueError("evaluation-view selection lacks audit binding")
    root = view_path.parent
    audit_path = _resolve_contained_path(root, audit_record.get("path"), "audit path")
    if not audit_path.is_file():
        raise FileNotFoundError(f"missing evaluation-view audit: {audit_path}")
    audit_sha = sha256_file(audit_path)
    if audit_sha != audit_record.get("sha256"):
        raise ValueError("evaluation-view audit file SHA-256 mismatch")
    audit = _json_object(audit_path, "evaluation-view audit")
    audit_basis = dict(audit)
    audit_canonical = audit_basis.pop("canonical_report_sha256", None)
    if audit_canonical != canonical_sha256(audit_basis):
        raise ValueError("evaluation-view audit canonical SHA-256 mismatch")
    if audit_canonical != audit_record.get("canonical_sha256"):
        raise ValueError("evaluation-view audit canonical binding mismatch")
    if audit.get("deterministic_result_sha256") != audit_record.get(
        "deterministic_result_sha256"
    ):
        raise ValueError("evaluation-view audit deterministic hash mismatch")
    expected_audit_identity = {
        "schema_version": 1,
        "report": "ofra_exact_train_test_overlap_mask_derivation_v1",
        "dataset": primary.dataset,
        "feature_dim": primary.feature_dim,
        "algorithm": selection["algorithm"],
        "equality_key": selection.get("equality_key"),
        "primary": primary_record,
    }
    if {key: audit.get(key) for key in expected_audit_identity} != (
        expected_audit_identity
    ):
        raise ValueError("evaluation-view audit identity/primary binding mismatch")
    builder_source = audit.get("builder_source")
    if (
        not isinstance(builder_source, dict)
        or not isinstance(builder_source.get("files"), list)
        or builder_source.get("canonical_source_sha256")
        != canonical_sha256(builder_source["files"])
    ):
        raise ValueError("evaluation-view audit builder-source hash mismatch")
    audit_semantic = dict(audit)
    audit_semantic.pop("canonical_report_sha256", None)
    audit_semantic.pop("created_utc", None)
    audit_semantic.pop("deterministic_result_sha256", None)
    if audit.get("deterministic_result_sha256") != canonical_sha256(
        audit_semantic
    ):
        raise ValueError("evaluation-view audit deterministic content hash mismatch")

    classes_raw = raw.get("classes")
    if not isinstance(classes_raw, list) or len(classes_raw) != len(primary.classes):
        raise ValueError("evaluation-view class axis mismatch")
    masks: dict[int, tuple[EvaluationMaskShard, ...]] = {}
    retained_rows: dict[int, int] = {}
    excluded_rows: dict[int, int] = {}
    for expected, item in zip(primary.classes, classes_raw):
        if not isinstance(item, dict) or "train" in item:
            raise ValueError("evaluation-view classes must be test-only objects")
        if item.get("id") != expected.class_id or item.get("name") != expected.name:
            raise ValueError("evaluation-view class identity/order mismatch")
        shards_raw = item.get("shards")
        if not isinstance(shards_raw, list) or len(shards_raw) != len(expected.test):
            raise ValueError("evaluation-view mask shard count mismatch")
        class_masks: list[EvaluationMaskShard] = []
        for ordinal, (parent_shard, mask_item) in enumerate(
            zip(expected.test, shards_raw)
        ):
            if not isinstance(mask_item, dict) or mask_item.get("ordinal") != ordinal:
                raise ValueError("evaluation-view mask shard order mismatch")
            parent_relative = _logical_shard_path(primary, parent_shard)
            parent_expected = {
                "relative_path": parent_relative,
                "rows": parent_shard.rows,
                "sha256": parent_shard.sha256,
            }
            if mask_item.get("parent") != parent_expected:
                raise ValueError("evaluation-view parent test shard mismatch")
            mask_record = mask_item.get("mask")
            if not isinstance(mask_record, dict):
                raise ValueError("evaluation-view mask record is invalid")
            mask_path = _resolve_contained_path(
                root, mask_record.get("path"), "mask path"
            )
            if not mask_path.is_file() or mask_path.suffix.lower() != ".npy":
                raise FileNotFoundError(f"invalid evaluation-view mask: {mask_path}")
            if verify_hashes and sha256_file(mask_path) != mask_record.get("sha256"):
                raise ValueError("evaluation-view mask file SHA-256 mismatch")
            array = np.load(mask_path, mmap_mode="r", allow_pickle=False)
            try:
                if array.dtype != np.dtype(np.bool_) or array.shape != (
                    parent_shard.rows,
                ):
                    raise ValueError("evaluation-view mask dtype/shape mismatch")
                true_count = int(np.count_nonzero(array))
            finally:
                _close_memmap(array)
            false_count = parent_shard.rows - true_count
            expected_mask = {
                "dtype": "bool",
                "shape": [parent_shard.rows],
                "rows": parent_shard.rows,
                "bytes": mask_path.stat().st_size,
                "sha256": mask_record.get("sha256"),
                "true_count": true_count,
                "false_count": false_count,
            }
            if {key: mask_record.get(key) for key in expected_mask} != expected_mask:
                raise ValueError("evaluation-view mask metadata/count mismatch")
            class_masks.append(
                EvaluationMaskShard(
                    path=mask_path,
                    rows=parent_shard.rows,
                    sha256=str(mask_record["sha256"]),
                    true_count=true_count,
                    false_count=false_count,
                    parent_relative_path=parent_relative,
                    parent_sha256=str(parent_shard.sha256),
                )
            )
        retained = sum(value.true_count for value in class_masks)
        excluded = sum(value.false_count for value in class_masks)
        if retained <= 0:
            raise ValueError(
                f"evaluation-view class {expected.class_id} has zero retained rows"
            )
        if (
            item.get("official_rows") != retained + excluded
            or item.get("retained_rows") != retained
            or item.get("excluded_rows") != excluded
        ):
            raise ValueError("evaluation-view per-class row accounting mismatch")
        masks[expected.class_id] = tuple(class_masks)
        retained_rows[expected.class_id] = retained
        excluded_rows[expected.class_id] = excluded

    totals = raw.get("totals")
    expected_totals = {
        "official_rows": sum(retained_rows.values()) + sum(excluded_rows.values()),
        "retained_rows": sum(retained_rows.values()),
        "excluded_rows": sum(excluded_rows.values()),
    }
    if not isinstance(totals, dict) or totals != expected_totals:
        raise ValueError("evaluation-view total row accounting mismatch")
    if audit.get("totals") != expected_totals:
        raise ValueError("evaluation-view audit/view total accounting mismatch")
    reconciliation = audit.get("frozen_split_overlap_reconciliation")
    if (
        not isinstance(reconciliation, dict)
        or reconciliation.get("verified") is not True
        or not isinstance(reconciliation.get("actual"), dict)
        or reconciliation.get("actual")
        != {
            **{
                key: expected_totals[value]
                for key, value in (
                    ("test_rows", "official_rows"),
                    ("excluded_rows", "excluded_rows"),
                )
                if key in reconciliation.get("actual", {})
            },
            "overlap_unique_feature_rows": audit.get(
                "overlap_unique_feature_rows"
            ),
        }
        or reconciliation.get("actual") != reconciliation.get("expected")
        or reconciliation.get("sha256")
        != primary_record.get("split_overlap_audit_sha256")
        or reconciliation.get("canonical_sha256")
        != primary_record.get("split_overlap_audit_canonical_sha256")
    ):
        raise ValueError("evaluation-view frozen split-overlap reconciliation failed")
    audit_classes = audit.get("classes")
    if not isinstance(audit_classes, list) or len(audit_classes) != len(
        primary.classes
    ):
        raise ValueError("evaluation-view audit class axis mismatch")
    for expected, view_item, audit_item in zip(
        primary.classes, classes_raw, audit_classes
    ):
        if not isinstance(audit_item, dict):
            raise ValueError("evaluation-view audit class record is invalid")
        excluded = excluded_rows[expected.class_id]
        category_keys = (
            "same_label_only_rows",
            "different_label_only_rows",
            "mixed_including_same_label_rows",
        )
        categories = [audit_item.get(key) for key in category_keys]
        if not all(isinstance(value, int) and value >= 0 for value in categories):
            raise ValueError("evaluation-view audit overlap categories are invalid")
        expected_class = {
            "class_id": expected.class_id,
            "class_name": expected.name,
            "official_rows": view_item["official_rows"],
            "retained_rows": view_item["retained_rows"],
            "excluded_rows": view_item["excluded_rows"],
        }
        if {key: audit_item.get(key) for key in expected_class} != expected_class:
            raise ValueError("evaluation-view audit/view class accounting mismatch")
        if sum(categories) != excluded:
            raise ValueError("evaluation-view audit overlap categories do not partition excluded rows")
        unique_rows = audit_item.get("overlap_unique_feature_rows")
        if not isinstance(unique_rows, int) or not 0 <= unique_rows <= excluded:
            raise ValueError("evaluation-view audit unique-overlap count is invalid")
    matrix = audit.get("label_presence_matrix")
    expected_axis = [
        {"id": record.class_id, "name": record.name}
        for record in primary.classes
    ]
    if not isinstance(matrix, dict) or matrix.get("axis") != expected_axis:
        raise ValueError("evaluation-view audit label-presence axis mismatch")
    matrix_rows = matrix.get("rows")
    width = len(primary.classes)
    if (
        not isinstance(matrix_rows, list)
        or len(matrix_rows) != width
        or any(not isinstance(row, list) or len(row) != width for row in matrix_rows)
        or any(
            not isinstance(value, int) or value < 0
            for row in matrix_rows
            for value in row
        )
    ):
        raise ValueError("evaluation-view audit label-presence matrix is invalid")
    for class_id, row in enumerate(matrix_rows):
        excluded = excluded_rows[class_id]
        same_or_mixed = (
            audit_classes[class_id]["same_label_only_rows"]
            + audit_classes[class_id]["mixed_including_same_label_rows"]
        )
        if row[class_id] != same_or_mixed or not (
            excluded <= sum(row) <= excluded * width
        ):
            raise ValueError("evaluation-view audit label-presence accounting mismatch")
    class_unique = [
        int(item["overlap_unique_feature_rows"]) for item in audit_classes
    ]
    global_unique = audit.get("overlap_unique_feature_rows")
    if (
        not isinstance(global_unique, int)
        or global_unique < 0
        or global_unique > expected_totals["excluded_rows"]
        or global_unique < max(class_unique, default=0)
        or global_unique > sum(class_unique)
    ):
        raise ValueError("evaluation-view audit global unique-overlap count is invalid")
    return EvaluationView(
        path=view_path,
        name=name,
        dataset=primary.dataset,
        feature_dim=primary.feature_dim,
        problem_type=primary.problem_type,
        metric_profile=primary.metric_profile,
        task_semantics=primary.task_semantics,
        normal_class_id=primary.normal_class_id,
        tasks=primary.tasks,
        class_names=primary.class_names,
        masks=masks,
        manifest_sha256=sha256_bytes(raw_bytes),
        canonical_sha256=canonical_actual,
        audit_provenance={
            "path": str(audit_path),
            "sha256": audit_sha,
            "canonical_sha256": audit_canonical,
            "deterministic_result_sha256": audit.get(
                "deterministic_result_sha256"
            ),
        },
        retained_rows=retained_rows,
        excluded_rows=excluded_rows,
    )


class ClassShards:
    """Memory-mapped class view with a bounded, explicitly closed LRU."""

    def __init__(
        self,
        shards: Sequence[Shard],
        feature_dim: int,
        *,
        max_open_memmaps: int = 4,
    ):
        if max_open_memmaps <= 0:
            raise ValueError("max_open_memmaps must be positive")
        self.shards = tuple(shards)
        self.feature_dim = int(feature_dim)
        self.max_open_memmaps = int(max_open_memmaps)
        self.offsets = np.cumsum([0, *(shard.rows for shard in self.shards)], dtype=np.int64)
        self._memmaps: OrderedDict[int, np.ndarray] = OrderedDict()

    @property
    def cached_shard_count(self) -> int:
        return len(self._memmaps)

    def _memmap(self, shard_id: int) -> np.ndarray:
        if shard_id in self._memmaps:
            array = self._memmaps.pop(shard_id)
            self._memmaps[shard_id] = array
            return array
        array = np.load(self.shards[shard_id].path, mmap_mode="r", allow_pickle=False)
        while len(self._memmaps) >= self.max_open_memmaps:
            _, evicted = self._memmaps.popitem(last=False)
            _close_memmap(evicted)
        self._memmaps[shard_id] = array
        return array

    def close(self) -> None:
        while self._memmaps:
            _, array = self._memmaps.popitem(last=False)
            _close_memmap(array)

    def __enter__(self) -> "ClassShards":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def index_blocks(
        self,
        block_rows: int,
        *,
        base_offset: int = 0,
        group_base: int = 0,
    ) -> tuple[list["IndexBlock"], int]:
        if block_rows <= 0:
            raise ValueError("block_rows must be positive")
        blocks: list[IndexBlock] = []
        for shard_id, shard in enumerate(self.shards):
            shard_start = int(base_offset + self.offsets[shard_id])
            for local_start in range(0, shard.rows, block_rows):
                blocks.append(
                    IndexBlock(
                        group=group_base + shard_id,
                        start=shard_start + local_start,
                        rows=min(block_rows, shard.rows - local_start),
                    )
                )
        return blocks, group_base + len(self.shards)

    def batches(self, batch_size: int) -> Iterator[np.ndarray]:
        for _, _, batch in self.batches_with_coordinates(batch_size):
            yield batch

    def batches_with_coordinates(
        self, batch_size: int
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        """Yield shard ordinal and local row start with each copied batch."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for shard_id, shard in enumerate(self.shards):
            for start in range(0, shard.rows, batch_size):
                array = self._memmap(shard_id)
                batch = np.array(
                    array[start : start + batch_size], dtype=np.float32, copy=True
                )
                if not np.isfinite(batch).all():
                    raise ValueError(f"non-finite feature found in {shard.path}")
                yield shard_id, start, batch

    def take(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        if len(indices) == 0:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        if indices.min() < 0 or indices.max() >= len(self):
            raise IndexError("class-shard index out of range")
        result = np.empty((len(indices), self.feature_dim), dtype=np.float32)
        shard_ids = np.searchsorted(self.offsets[1:], indices, side="right")
        for shard_id in np.unique(shard_ids):
            mask = shard_ids == shard_id
            local = indices[mask] - self.offsets[shard_id]
            array = self._memmap(int(shard_id))
            values = np.array(array[local], dtype=np.float32, copy=True)
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite feature found in {self.shards[int(shard_id)].path}")
            result[mask] = values
        return result


@dataclass(frozen=True)
class IndexBlock:
    group: int
    start: int
    rows: int


class BlockShuffleSampler:
    """Bounded-memory shard/block shuffle with an exact uniform row subset.

    A sequential multivariate-hypergeometric allocation chooses the requested
    number of rows without replacement. Shard groups, blocks within each
    shard, and selected rows within each block are then independently shuffled.
    """

    algorithm = "hierarchical_shard_block_shuffle_hypergeometric_v1"

    def __init__(
        self,
        blocks: Sequence[IndexBlock],
        *,
        population_rows: int,
        sample_rows: int,
        seed: int,
        block_rows: int,
    ):
        if population_rows <= 0 or not 0 <= sample_rows <= population_rows:
            raise ValueError("invalid block-sampler population/sample size")
        if block_rows <= 0:
            raise ValueError("block_rows must be positive")
        if sum(block.rows for block in blocks) != population_rows:
            raise ValueError("block rows do not cover the sampler population")
        self.population_rows = int(population_rows)
        self.sample_rows = int(sample_rows)
        self.seed = int(seed)
        self.block_rows = int(block_rows)
        self.rng = np.random.default_rng(self.seed)
        remaining_population = self.population_rows
        remaining_sample = self.sample_rows
        selected: list[tuple[IndexBlock, int]] = []
        for block in blocks:
            if remaining_sample == 0:
                take = 0
            elif remaining_sample == remaining_population:
                take = block.rows
            else:
                take = int(
                    self.rng.hypergeometric(
                        block.rows,
                        remaining_population - block.rows,
                        remaining_sample,
                    )
                )
            if take:
                selected.append((block, take))
            remaining_population -= block.rows
            remaining_sample -= take
        if remaining_population != 0 or remaining_sample != 0:
            raise RuntimeError("block-sampler allocation accounting failed")

        grouped: dict[int, list[tuple[IndexBlock, int]]] = {}
        for item in selected:
            grouped.setdefault(item[0].group, []).append(item)
        group_ids = list(grouped)
        self.rng.shuffle(group_ids)
        self.ordered: list[tuple[IndexBlock, int]] = []
        for group_id in group_ids:
            group = grouped[group_id]
            self.rng.shuffle(group)
            self.ordered.extend(group)
        self._consumed = False

    def iter_chunks(self, max_rows: int) -> Iterator[np.ndarray]:
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")
        if self._consumed:
            raise RuntimeError("a block sampler can only be consumed once")
        self._consumed = True
        emitted = 0
        for block, take in self.ordered:
            if take == block.rows:
                local = self.rng.permutation(block.rows)
            else:
                local = self.rng.choice(block.rows, take, replace=False, shuffle=True)
            indices = local.astype(np.int64, copy=False) + block.start
            for start in range(0, take, max_rows):
                chunk = indices[start : start + max_rows]
                emitted += len(chunk)
                yield chunk
        if emitted != self.sample_rows:
            raise RuntimeError("block-sampler emitted-row accounting failed")

    def record(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "seed": self.seed,
            "block_rows": self.block_rows,
            "population_rows": self.population_rows,
            "sample_rows": self.sample_rows,
            "selected_blocks": len(self.ordered),
            "global_index_array_materialized": False,
        }


class CompositePool:
    """Random-access concatenation of shard sets and in-memory arrays."""

    def __init__(self, sources: Sequence[ClassShards | np.ndarray], feature_dim: int):
        self.sources = tuple(sources)
        self.feature_dim = int(feature_dim)
        sizes = []
        for source in self.sources:
            if isinstance(source, ClassShards):
                sizes.append(len(source))
            else:
                value = np.asarray(source)
                if value.ndim != 2 or value.shape[1] != feature_dim:
                    raise ValueError("in-memory source has an incompatible feature shape")
                sizes.append(len(value))
        self.offsets = np.cumsum([0, *sizes], dtype=np.int64)

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def take(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        if len(indices) == 0:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        if indices.min() < 0 or indices.max() >= len(self):
            raise IndexError("composite-pool index out of range")
        result = np.empty((len(indices), self.feature_dim), dtype=np.float32)
        source_ids = np.searchsorted(self.offsets[1:], indices, side="right")
        for source_id in np.unique(source_ids):
            mask = source_ids == source_id
            local = indices[mask] - self.offsets[source_id]
            source = self.sources[int(source_id)]
            if isinstance(source, ClassShards):
                result[mask] = source.take(local)
            else:
                result[mask] = np.asarray(source[local], dtype=np.float32)
        return result


class FrozenTask0Stats:
    """Float64 Chan/Welford statistics fitted only on Task-0 train shards."""

    def __init__(self, feature_dim: int):
        self.feature_dim = int(feature_dim)
        self.count = 0
        self.mean = np.zeros(feature_dim, dtype=np.float64)
        self.m2 = np.zeros(feature_dim, dtype=np.float64)
        self.frozen = False
        self.source_classes: list[int] = []

    def update(self, batch: np.ndarray) -> None:
        if self.frozen:
            raise RuntimeError("normalization statistics are frozen")
        values = np.asarray(batch, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.feature_dim or not len(values):
            raise ValueError("invalid statistics batch")
        if not np.isfinite(values).all():
            raise ValueError("statistics batch contains non-finite values")
        batch_count = len(values)
        batch_mean = values.mean(axis=0, dtype=np.float64)
        centered = values - batch_mean
        batch_m2 = np.einsum("ij,ij->j", centered, centered, dtype=np.float64)
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * (batch_count / total)
        self.m2 += batch_m2 + delta * delta * (self.count * batch_count / total)
        self.count = total

    def freeze(self, source_classes: Iterable[int]) -> None:
        if self.count <= 0:
            raise RuntimeError("cannot freeze empty statistics")
        self.source_classes = [int(value) for value in source_classes]
        self.frozen = True

    @property
    def scale(self) -> np.ndarray:
        if not self.frozen:
            raise RuntimeError("statistics are not frozen")
        value = np.sqrt(self.m2 / self.count, dtype=np.float64)
        value[value == 0] = 1.0
        return value

    def transform(self, batch: np.ndarray) -> np.ndarray:
        if not self.frozen:
            raise RuntimeError("statistics are not frozen")
        values = np.asarray(batch, dtype=np.float64)
        result = (values - self.mean) / self.scale
        result = result.astype(np.float32)
        if not np.isfinite(result).all():
            raise FloatingPointError("normalization produced non-finite values")
        return result

    def record(self) -> dict:
        return {
            "algorithm": "chan_welford_float64_population_variance",
            "count": int(self.count),
            "source_classes": list(self.source_classes),
            "mean_sha256": array_sha256(self.mean),
            "scale_sha256": array_sha256(self.scale),
        }


class MatrixReservoir:
    """Deterministic, vectorized Algorithm-R matrix reservoir."""

    def __init__(self, capacity: int, width: int, rng: np.random.Generator):
        if capacity < 0 or width <= 0:
            raise ValueError("invalid reservoir shape")
        self.capacity = int(capacity)
        self.width = int(width)
        self.rng = rng
        self.values = np.empty((capacity, width), dtype=np.float32)
        self.stream_indices = np.empty(capacity, dtype=np.int64)
        self.seen = 0
        self.retained = 0

    def update(self, batch: np.ndarray) -> None:
        values = np.asarray(batch, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.width:
            raise ValueError("reservoir batch has an incompatible shape")
        start = 0
        if self.retained < self.capacity:
            fill = min(len(values), self.capacity - self.retained)
            self.values[self.retained : self.retained + fill] = values[:fill]
            self.stream_indices[self.retained : self.retained + fill] = np.arange(
                self.seen,
                self.seen + fill,
                dtype=np.int64,
            )
            self.retained += fill
            self.seen += fill
            start = fill
        remaining = len(values) - start
        if remaining <= 0:
            return
        if not self.capacity:
            self.seen += remaining
            return
        # At one-indexed stream position t, Algorithm R draws uniformly from
        # [0, t). NumPy's broadcast high generates every variable-bound draw
        # without modulo bias. Replacement decisions do not depend on current
        # reservoir contents, so duplicate slots can be committed in bulk by
        # keeping their last accepted row, exactly matching sequential state.
        high = np.arange(
            self.seen + 1,
            self.seen + remaining + 1,
            dtype=np.int64,
        )
        slots = self.rng.integers(0, high)
        accepted_mask = slots < self.capacity
        if accepted_mask.any():
            accepted_rows = np.flatnonzero(accepted_mask) + start
            accepted_slots = slots[accepted_mask].astype(np.int64, copy=False)
            last_row_for_slot = np.full(self.capacity, -1, dtype=np.int64)
            np.maximum.at(last_row_for_slot, accepted_slots, accepted_rows)
            touched = np.flatnonzero(last_row_for_slot >= 0)
            self.values[touched] = values[last_row_for_slot[touched]]
            self.stream_indices[touched] = self.seen + (
                last_row_for_slot[touched] - start
            )
        self.seen += remaining

    def array(self) -> np.ndarray:
        return self.values[: self.retained].copy()

    def indices(self) -> np.ndarray:
        return self.stream_indices[: self.retained].copy()

    def record(self) -> dict:
        value = self.array()
        indices = self.indices()
        return {
            "algorithm": "reservoir_algorithm_r_vectorized_variable_high_v2",
            "seen": int(self.seen),
            "retained": int(self.retained),
            "capacity": int(self.capacity),
            "sha256": array_sha256(value),
            "selected_index_sha256": array_sha256(indices),
            "selected_index_count": int(len(indices)),
            "selected_indices_unique": bool(len(np.unique(indices)) == len(indices)),
            "selected_index_min": int(indices.min()) if len(indices) else None,
            "selected_index_max": int(indices.max()) if len(indices) else None,
        }
