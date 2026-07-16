"""Out-of-core preprocessing and immutable NumPy shard generation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .specs import DATASET_SPECS, DatasetSpec


TOOL_VERSION = "1.0.0"
FLOAT32_ABS_LIMIT = 3.0e38
HASH_BUCKETS = 10_000
ROW_HASH_ALGORITHM = "fnv1a64-float32-words-v1"
SPLIT_MIX_ALGORITHM = "splitmix64-seeded-v1"


@dataclass(frozen=True)
class BuildOptions:
    """Runtime controls that do not alter the dataset's feature contract."""

    chunk_rows: int = 100_000
    split_seed: int = 42
    test_fraction: float = 0.20
    strict_files: bool = True
    strict_unmapped_labels: bool = True
    overwrite: bool = False
    overlap_batch_rows: int = 50_000
    overlap_work_directory: str | Path | None = None

    def validate(self) -> None:
        if self.chunk_rows < 1:
            raise ValueError("chunk_rows must be a positive integer")
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError("test_fraction must be strictly between 0 and 1")
        if not 0 <= self.split_seed < 2**64:
            raise ValueError("split_seed must fit in an unsigned 64-bit integer")
        if self.overlap_batch_rows < 1:
            raise ValueError("overlap_batch_rows must be a positive integer")


@dataclass(frozen=True)
class _Schema:
    columns: tuple[str, ...]
    tokens: tuple[str, ...]
    raw_numeric_columns: tuple[str, ...]
    raw_numeric_tokens: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    categorical_tokens: tuple[str, ...]
    categorical_vocabulary: dict[str, tuple[str, ...]]
    sha256: str
    raw_file_headers: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _CaptureContract:
    path: Path
    sha256: str
    source_revision: str
    test_captures: frozenset[str]
    verified_file_sha256: dict[str, str]
    manifest: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _column_token(value: object) -> str:
    return " ".join(str(value).strip().split()).casefold()


def _normalise_columns(columns: Iterable[object]) -> list[str]:
    return [" ".join(str(column).strip().split()) for column in columns]


def _token_map(columns: Iterable[str], *, source: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for column in columns:
        token = _column_token(column)
        if token in result:
            raise ValueError(
                f"{source} has columns that collide after whitespace/case "
                f"normalisation: {result[token]!r} and {column!r}"
            )
        result[token] = column
    return result


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    _write_json(temporary, value)
    os.replace(temporary, path)


def sha256_file(path: Path, buffer_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(buffer_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def builder_source_record() -> dict[str, Any]:
    """Return byte-level provenance for every Python file in ``fullcache``."""

    package_directory = Path(__file__).resolve().parent
    files = []
    for path in sorted(package_directory.glob("*.py"), key=lambda item: item.name):
        files.append(
            {
                "relative_path": f"fullcache/{path.name}",
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    record = {"algorithm": "sha256-file-bytes-and-canonical-json-v1", "files": files}
    record["canonical_source_sha256"] = hashlib.sha256(
        _json_bytes(record)
    ).hexdigest()
    return record


def stable_row_hash64(features: np.ndarray) -> np.ndarray:
    """Hash canonical little-endian float32 feature rows without Python loops.

    The loop is over columns, not rows.  Positive and negative zero are
    canonicalised because they are identical model inputs.  NaN and infinity
    must be removed before this function is called.
    """

    array = np.array(features, dtype="<f4", order="C", copy=True)
    if array.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("stable_row_hash64 received non-finite features")
    array[array == 0.0] = 0.0
    words = array.view("<u4").reshape(array.shape)
    hashes = np.full(array.shape[0], np.uint64(14695981039346656037))
    prime = np.uint64(1099511628211)
    with np.errstate(over="ignore"):
        for column in range(words.shape[1]):
            hashes ^= words[:, column].astype(np.uint64, copy=False)
            hashes *= prime
        hashes ^= np.uint64(words.shape[1])
        hashes *= prime
    return hashes


def _splitmix64(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        values += np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        values = (values ^ (values >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
    return values ^ (values >> np.uint64(31))


def feature_hash_splits(
    features: np.ndarray, *, seed: int = 42, test_fraction: float = 0.20
) -> np.ndarray:
    """Return ``train``/``test`` strings using feature-only group hashing."""

    test_buckets = int(round(test_fraction * HASH_BUCKETS))
    if not 0 < test_buckets < HASH_BUCKETS:
        raise ValueError("test_fraction produces an empty train or test partition")
    row_hashes = stable_row_hash64(features)
    mixed = _splitmix64(row_hashes ^ np.uint64(seed))
    is_test = (mixed % np.uint64(HASH_BUCKETS)) < np.uint64(test_buckets)
    return np.where(is_test, "test", "train")


def _source_directory(data_root: Path, spec: DatasetSpec) -> Path:
    return (data_root / spec.subdirectory).resolve()


def _relative_source_path(path: Path, source: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(source.resolve())
    except ValueError as error:
        raise ValueError(f"Source file escapes dataset root: {path}") from error
    value = relative.as_posix()
    if not value or PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts:
        raise ValueError(f"Unsafe source-relative path: {value!r}")
    return value


def _mapped_parent_label(path: Path, source: Path, spec: DatasetSpec) -> str:
    relative = PurePosixPath(_relative_source_path(path, source))
    if len(relative.parts) < 2:
        raise ValueError(
            f"{spec.name} capture must be nested below a class directory: {relative}"
        )
    raw = relative.parts[-2]
    normalised = spec.label_normalizer(pd.Series([raw], dtype="string")).iloc[0]
    mapped = spec.label_mapping.get(str(normalised))
    if mapped is None or mapped not in spec.class_order:
        raise ValueError(
            f"{spec.name} has unmapped parent-directory label {raw!r} in {relative}"
        )
    return mapped


def _load_capture_contract(
    source: Path, files: list[Path], spec: DatasetSpec
) -> _CaptureContract | None:
    if spec.split_strategy != "frozen_capture_manifest":
        return None
    contract_locations = (
        spec.source_contract_relative,
        spec.bundled_contract_relative,
    )
    if (
        sum(value is not None for value in contract_locations) != 1
        or spec.source_contract_sha256 is None
        or spec.source_revision is None
    ):
        raise ValueError(f"{spec.name} has an incomplete frozen capture contract")

    if spec.bundled_contract_relative is not None:
        package_root = Path(__file__).resolve().parent
        contract_path = (package_root / spec.bundled_contract_relative).resolve()
        try:
            contract_path.relative_to(package_root)
        except ValueError as error:
            raise ValueError(
                f"{spec.name} bundled capture contract escapes fullcache"
            ) from error
    else:
        assert spec.source_contract_relative is not None
        contract_path = (source / spec.source_contract_relative).resolve()
    if not contract_path.is_file():
        raise FileNotFoundError(f"Missing frozen capture contract: {contract_path}")
    contract_sha256 = sha256_file(contract_path)
    if contract_sha256.casefold() != spec.source_contract_sha256.casefold():
        raise ValueError(
            f"{spec.name} capture-contract SHA256 mismatch: "
            f"expected {spec.source_contract_sha256}, found {contract_sha256}"
        )
    try:
        manifest = json.loads(contract_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid frozen capture contract: {contract_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("Frozen capture contract must be a JSON object")

    required_identity = {
        "dataset": "MalayaNetwork_GT",
        "source_revision": spec.source_revision,
        "split_strategy": (
            "one_capture_per_class_closest_to_20_percent_then_lexicographic"
        ),
        "split_seed": None,
        "class_order": list(spec.class_order),
        "excluded_identifier_features": list(spec.drop_columns),
    }
    for key, expected in required_identity.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"{spec.name} frozen capture contract has invalid {key}: "
                f"{manifest.get(key)!r} != {expected!r}"
            )

    raw_hashes = manifest.get("source_csv_sha256")
    test_values = manifest.get("test_captures")
    if not isinstance(raw_hashes, dict) or not isinstance(test_values, list):
        raise ValueError(
            "Frozen capture contract requires source_csv_sha256 and test_captures"
        )
    relative_to_path = {
        _relative_source_path(path, source): path
        for path in files
    }
    if len(relative_to_path) != len(files):
        raise ValueError(f"{spec.name} has duplicate source-relative capture paths")
    if set(raw_hashes) != set(relative_to_path):
        raise ValueError(
            f"{spec.name} capture list differs from the frozen contract: "
            f"missing={sorted(set(raw_hashes) - set(relative_to_path))}, "
            f"unexpected={sorted(set(relative_to_path) - set(raw_hashes))}"
        )
    folded = [relative.casefold() for relative in relative_to_path]
    if len(folded) != len(set(folded)):
        raise ValueError(f"{spec.name} capture paths collide case-insensitively")

    test_captures: list[str] = []
    for value in test_values:
        if not isinstance(value, str) or value not in relative_to_path:
            raise ValueError(f"Invalid frozen test capture: {value!r}")
        test_captures.append(value)
    if len(test_captures) != len(set(test_captures)):
        raise ValueError("Frozen test-capture list contains duplicates")
    if len(test_captures) != len(spec.class_order):
        raise ValueError(
            f"{spec.name} requires exactly one test capture per class; "
            f"found {len(test_captures)} for {len(spec.class_order)} classes"
        )

    split_classes = {"train": Counter(), "test": Counter()}
    test_set = frozenset(test_captures)
    for relative, path in relative_to_path.items():
        split = "test" if relative in test_set else "train"
        split_classes[split][_mapped_parent_label(path, source, spec)] += 1
    for class_name in spec.class_order:
        if split_classes["test"][class_name] != 1:
            raise ValueError(
                f"{spec.name} requires one frozen test capture for {class_name}; "
                f"found {split_classes['test'][class_name]}"
            )
        if split_classes["train"][class_name] < 1:
            raise ValueError(f"{spec.name} has no training capture for {class_name}")

    verified: dict[str, str] = {}
    for relative, path in relative_to_path.items():
        expected = raw_hashes[relative]
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            raise ValueError(f"Invalid frozen SHA256 for capture {relative}")
        actual = sha256_file(path)
        if actual.casefold() != expected.casefold():
            raise ValueError(
                f"{spec.name} raw capture SHA256 mismatch for {relative}: "
                f"expected {expected}, found {actual}"
            )
        verified[relative] = actual

    return _CaptureContract(
        path=contract_path,
        sha256=contract_sha256,
        source_revision=spec.source_revision,
        test_captures=test_set,
        verified_file_sha256=verified,
        manifest=manifest,
    )


def _discover_files(data_root: Path, spec: DatasetSpec) -> list[Path]:
    source = _source_directory(data_root, spec)
    files = sorted(
        (path for path in source.glob(spec.file_glob) if path.is_file()),
        key=lambda path: _relative_source_path(path, source).casefold(),
    )
    if not files:
        raise FileNotFoundError(f"No {spec.file_glob} files found in {source}")
    return files


def _validate_source_files(
    files: list[Path], spec: DatasetSpec, strict: bool, source: Path
) -> dict[str, Any]:
    actual_names = [_relative_source_path(path, source) for path in files]
    expected_names = list(spec.expected_filenames)
    actual_folded = {name.casefold(): name for name in actual_names}
    expected_folded = {name.casefold(): name for name in expected_names}
    missing = [
        name for folded, name in expected_folded.items() if folded not in actual_folded
    ]
    unexpected = [
        name for folded, name in actual_folded.items()
        if expected_names and folded not in expected_folded
    ]
    complete = len(files) == spec.expected_file_count and not missing and not unexpected
    part_contract: dict[str, Any] | None = None
    if spec.name == "cic-iot-2023":
        observed_parts: list[int] = []
        invalid_part_names: list[str] = []
        for name in actual_names:
            match = re.match(r"^part-(\d{5})(?:-|\.csv$)", name, flags=re.IGNORECASE)
            if match:
                observed_parts.append(int(match.group(1)))
            else:
                invalid_part_names.append(name)
        expected_parts = set(range(169))
        observed_set = set(observed_parts)
        duplicate_parts = sorted(
            part for part, count in Counter(observed_parts).items() if count > 1
        )
        missing_parts = sorted(expected_parts - observed_set)
        unexpected_parts = sorted(observed_set - expected_parts)
        part_contract = {
            "expected_part_range": [0, 168],
            "missing_part_indices": missing_parts,
            "unexpected_part_indices": unexpected_parts,
            "duplicate_part_indices": duplicate_parts,
            "invalid_part_filenames": invalid_part_names,
            "complete": not (
                missing_parts
                or unexpected_parts
                or duplicate_parts
                or invalid_part_names
            ),
        }
        complete = complete and part_contract["complete"]
    if strict and not complete:
        detail = {
            "expected_file_count": spec.expected_file_count,
            "actual_file_count": len(files),
            "missing_expected_files": missing,
            "unexpected_files": unexpected,
            "part_contract": part_contract,
        }
        raise ValueError(
            f"{spec.name} raw-file contract is incomplete: "
            f"{json.dumps(detail, ensure_ascii=False)}"
        )
    result = {
        "expected_file_count": spec.expected_file_count,
        "actual_file_count": len(files),
        "complete": complete,
        "missing_expected_files": missing,
        "unexpected_files": unexpected,
    }
    if part_contract is not None:
        result["part_contract"] = part_contract
    return result


def _read_header(path: Path, spec: DatasetSpec) -> list[str]:
    if spec.read_names:
        return _normalise_columns(spec.read_names)
    kwargs: dict[str, Any] = {"nrows": 0}
    if spec.encoding:
        kwargs["encoding"] = spec.encoding
    header = pd.read_csv(path, **kwargs)
    return _normalise_columns(header.columns)


def _resolve_label_column(
    token_to_column: dict[str, str], spec: DatasetSpec, source: str
) -> str:
    for candidate in spec.label_candidates:
        match = token_to_column.get(_column_token(candidate))
        if match is not None:
            return match
    raise ValueError(
        f"{source} has none of the required label columns: "
        f"{list(spec.label_candidates)}"
    )


def _preflight_schema(
    files: list[Path], spec: DatasetSpec, chunk_rows: int, source: Path
) -> _Schema:
    drop_tokens = {_column_token(column) for column in spec.drop_columns}
    identifier_tokens = {
        _column_token(column) for column in spec.identifier_columns
    }
    reference_columns: tuple[str, ...] | None = None
    reference_tokens: tuple[str, ...] | None = None
    raw_headers: list[dict[str, Any]] = []

    for path in files:
        relative_path = _relative_source_path(path, source)
        columns = _read_header(path, spec)
        token_to_column = _token_map(columns, source=relative_path)
        if spec.label_source == "column":
            label_column = _resolve_label_column(token_to_column, spec, relative_path)
            label_token = _column_token(label_column)
        elif spec.label_source == "parent_directory":
            label_column = None
            label_token = None
            _mapped_parent_label(path, source, spec)
        else:
            raise ValueError(f"Unsupported label_source for {spec.name}: {spec.label_source}")
        features = [
            column
            for column in columns
            if (label_token is None or _column_token(column) != label_token)
            and _column_token(column) not in drop_tokens
        ]
        feature_tokens = tuple(_column_token(column) for column in features)
        removed_identifiers = [
            column for column in columns if _column_token(column) in identifier_tokens
        ]
        removed_metadata = [
            column
            for column in columns
            if _column_token(column) in drop_tokens
            and _column_token(column) not in identifier_tokens
            and _column_token(column) != label_token
        ]

        if reference_columns is None:
            reference_columns = tuple(features)
            reference_tokens = feature_tokens
        else:
            assert reference_tokens is not None
            missing = [
                reference_columns[index]
                for index, token in enumerate(reference_tokens)
                if token not in feature_tokens
            ]
            extra = [
                features[index]
                for index, token in enumerate(feature_tokens)
                if token not in reference_tokens
            ]
            if missing or extra or len(feature_tokens) != len(reference_tokens):
                raise ValueError(
                    f"Feature schema mismatch in {relative_path}: "
                    f"missing={missing}, extra={extra}"
                )

        raw_headers.append(
            {
                "relative_path": relative_path,
                "raw_column_count": len(columns),
                "label_column": label_column,
                "label_source": spec.label_source,
                "source_class": (
                    _mapped_parent_label(path, source, spec)
                    if spec.label_source == "parent_directory"
                    else None
                ),
                "feature_count_after_drop": len(features),
                "categorical_columns": [
                    column
                    for column in features
                    if _column_token(column)
                    in {_column_token(value) for value in spec.categorical_columns}
                ],
                "identifier_columns_removed": removed_identifiers,
                "metadata_columns_removed": removed_metadata,
            }
        )

    assert reference_columns is not None and reference_tokens is not None
    categorical_tokens = tuple(
        _column_token(column) for column in spec.categorical_columns
    )
    for token, column in zip(categorical_tokens, spec.categorical_columns):
        if token not in reference_tokens:
            raise ValueError(
                f"{spec.name} is missing configured categorical column {column!r}"
            )
    raw_numeric_columns = tuple(
        reference_columns[index]
        for index, token in enumerate(reference_tokens)
        if token not in categorical_tokens
    )
    raw_numeric_tokens = tuple(
        token for token in reference_tokens if token not in categorical_tokens
    )

    categorical_vocabulary: dict[str, tuple[str, ...]] = {}
    if categorical_tokens:
        vocab_sets: dict[str, set[str]] = {
            column: set() for column in spec.categorical_columns
        }
        train_files = [path for path in files if _official_split(path) == "train"]
        if not train_files:
            raise ValueError(f"{spec.name} has no official training file")
        for path in train_files:
            for chunk in _read_chunks(path, spec, chunk_rows):
                chunk.columns = _normalise_columns(chunk.columns)
                token_to_column = _token_map(
                    chunk.columns, source=f"{path.name} vocabulary pass"
                )
                for configured, token in zip(
                    spec.categorical_columns, categorical_tokens
                ):
                    values = (
                        chunk[token_to_column[token]]
                        .astype("string")
                        .str.strip()
                        .dropna()
                    )
                    vocab_sets[configured].update(
                        str(value) for value in values if str(value) != ""
                    )
        categorical_vocabulary = {
            column: tuple(sorted(values))
            for column, values in vocab_sets.items()
        }
        empty = [
            column for column, values in categorical_vocabulary.items() if not values
        ]
        if empty:
            raise ValueError(
                f"{spec.name} training data has empty categorical vocabularies: {empty}"
            )
        output_columns = list(raw_numeric_columns)
        for column in spec.categorical_columns:
            output_columns.extend(
                f"{column}_{value}" for value in categorical_vocabulary[column]
            )
        output_tokens = tuple(_column_token(column) for column in output_columns)
        if len(output_tokens) != len(set(output_tokens)):
            raise ValueError(
                f"{spec.name} one-hot feature names collide after normalisation"
            )
        reference_columns = tuple(output_columns)
        reference_tokens = output_tokens

    if len(reference_columns) != spec.expected_feature_count:
        raise ValueError(
            f"{spec.name} must have {spec.expected_feature_count} model features "
            f"after identifier/metadata removal; found {len(reference_columns)}"
        )
    schema_hash = hashlib.sha256(
        _json_bytes(
            {
                "dtype": "float32-little-endian",
                "columns": list(reference_columns),
            }
        )
    ).hexdigest()
    return _Schema(
        columns=reference_columns,
        tokens=reference_tokens,
        raw_numeric_columns=raw_numeric_columns,
        raw_numeric_tokens=raw_numeric_tokens,
        categorical_columns=tuple(spec.categorical_columns),
        categorical_tokens=categorical_tokens,
        categorical_vocabulary=categorical_vocabulary,
        sha256=schema_hash,
        raw_file_headers=tuple(raw_headers),
    )


def _official_split(path: Path) -> str:
    name = path.name.casefold()
    has_train = "train" in name
    has_test = "test" in name
    if has_train == has_test:
        raise ValueError(
            f"Cannot infer an unambiguous official split from NF-ToN filename: "
            f"{path.name}"
        )
    return "train" if has_train else "test"


def _read_chunks(path: Path, spec: DatasetSpec, chunk_rows: int):
    kwargs: dict[str, Any] = {
        "chunksize": chunk_rows,
        "low_memory": False,
    }
    if spec.encoding:
        kwargs["encoding"] = spec.encoding
    if spec.read_names:
        kwargs["header"] = None
        kwargs["names"] = list(spec.read_names)
    yield from pd.read_csv(path, **kwargs)


def _prepare_chunk(
    chunk: pd.DataFrame,
    spec: DatasetSpec,
    schema: _Schema,
    *,
    strict_unmapped: bool,
    source_label: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    chunk = chunk.copy()
    chunk.columns = _normalise_columns(chunk.columns)
    token_to_column = _token_map(chunk.columns, source=f"{spec.name} chunk")
    if spec.label_source == "column":
        label_column = _resolve_label_column(token_to_column, spec, f"{spec.name} chunk")
        labels = chunk[label_column]
    elif spec.label_source == "parent_directory":
        if source_label is None:
            raise ValueError(f"{spec.name} chunk is missing its source-directory label")
        labels = pd.Series(source_label, index=chunk.index, dtype="string")
    else:
        raise ValueError(f"Unsupported label_source for {spec.name}: {spec.label_source}")
    stripped_labels = labels.astype("string").str.strip()
    raw_missing = (
        chunk.isna().any(axis=1)
        if spec.drop_any_raw_missing
        else pd.Series(False, index=chunk.index)
    )
    candidate_tokens = {_column_token(candidate) for candidate in spec.label_candidates}
    repeated_header = (
        stripped_labels.str.casefold().isin(candidate_tokens).fillna(False)
        if spec.label_source == "column"
        else pd.Series(False, index=chunk.index)
    )
    missing_label = (labels.isna() | stripped_labels.eq("")).fillna(True)

    normalised = spec.label_normalizer(labels)
    mapped = normalised.map(spec.label_mapping)
    unknown = (
        ~repeated_header & ~missing_label & ~raw_missing & mapped.isna()
    ).fillna(False)
    defaulted = pd.Series(False, index=chunk.index)
    if spec.unknown_label_family is not None:
        defaulted = unknown.copy()
        mapped.loc[defaulted] = spec.unknown_label_family
        unknown = pd.Series(False, index=chunk.index)
    defaulted_values = Counter(
        str(value) for value in normalised.loc[defaulted].dropna().tolist()
    )
    unknown_values = Counter(
        str(value) for value in normalised.loc[unknown].dropna().tolist()
    )
    if unknown_values and strict_unmapped:
        examples = dict(unknown_values.most_common(10))
        raise ValueError(
            f"{spec.name} contains unmapped non-empty labels: "
            f"{json.dumps(examples, ensure_ascii=False)}"
        )

    label_valid = mapped.notna() & ~repeated_header & ~missing_label & ~raw_missing
    selected_rows = np.flatnonzero(label_valid.to_numpy(dtype=bool))
    stats: dict[str, Any] = {
        "raw_rows": int(len(chunk)),
        "repeated_header_rows": int(repeated_header.sum()),
        "missing_label_rows": int(missing_label.sum()),
        "raw_missing_field_rows": int(raw_missing.sum()),
        "unmapped_label_rows": int(unknown.sum()),
        "defaulted_label_rows": int(defaulted.sum()),
        "mapped_label_rows": int(len(selected_rows)),
        "invalid_numeric_or_nonfinite_rows": 0,
        "valid_rows": 0,
        "unmapped_label_counts": dict(unknown_values),
        "defaulted_label_counts": dict(defaulted_values),
        "unknown_categorical_rows": 0,
        "missing_categorical_values": 0,
        "unknown_categorical_counts": {},
    }
    if not len(selected_rows):
        return (
            np.empty((0, len(schema.columns)), dtype="<f4"),
            np.empty(0, dtype=object),
            stats,
        )

    actual_numeric_columns = [
        token_to_column[token] if token in token_to_column else None
        for token in schema.raw_numeric_tokens
    ]
    actual_categorical_columns = [
        token_to_column[token] if token in token_to_column else None
        for token in schema.categorical_tokens
    ]
    actual_input_columns = actual_numeric_columns + actual_categorical_columns
    if any(column is None for column in actual_input_columns):
        missing = [
            column
            for column, actual in zip(
                schema.raw_numeric_columns + schema.categorical_columns,
                actual_input_columns,
            )
            if actual is None
        ]
        raise ValueError(f"Chunk is missing locked feature columns: {missing}")

    frame = chunk.iloc[selected_rows][actual_numeric_columns]
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64, copy=False)
    good = np.isfinite(values).all(axis=1) & (
        np.abs(values) < FLOAT32_ABS_LIMIT
    ).all(axis=1)
    stats["invalid_numeric_or_nonfinite_rows"] = int((~good).sum())
    stats["valid_rows"] = int(good.sum())

    features = np.zeros(
        (int(good.sum()), len(schema.columns)), dtype="<f4", order="C"
    )
    features[:, : len(schema.raw_numeric_columns)] = np.asarray(
        values[good], dtype="<f4", order="C"
    )

    if schema.categorical_columns:
        output_rows = np.arange(int(good.sum()))
        offset = len(schema.raw_numeric_columns)
        any_unknown = np.zeros(int(good.sum()), dtype=bool)
        unknown_counts: dict[str, dict[str, int]] = {}
        missing_values = 0
        for configured, actual in zip(
            schema.categorical_columns, actual_categorical_columns
        ):
            assert actual is not None
            values_as_text = (
                chunk.iloc[selected_rows][actual].astype("string").str.strip()
            )
            vocabulary = schema.categorical_vocabulary[configured]
            vocabulary_index = {
                value: index for index, value in enumerate(vocabulary)
            }
            codes = (
                values_as_text.map(vocabulary_index)
                .fillna(-1)
                .to_numpy(dtype=np.int64)[good]
            )
            valid_category = codes >= 0
            if valid_category.any():
                features[
                    output_rows[valid_category], offset + codes[valid_category]
                ] = 1.0
            output_values = values_as_text.iloc[np.flatnonzero(good)]
            missing = output_values.isna() | output_values.eq("")
            unknown_category = (~missing).to_numpy(dtype=bool) & ~valid_category
            any_unknown |= unknown_category
            missing_values += int(missing.sum())
            if unknown_category.any():
                unknown_counts[configured] = dict(
                    Counter(
                        str(value)
                        for value in output_values.iloc[
                            np.flatnonzero(unknown_category)
                        ].tolist()
                    )
                )
            offset += len(vocabulary)
        stats["unknown_categorical_rows"] = int(any_unknown.sum())
        stats["missing_categorical_values"] = missing_values
        stats["unknown_categorical_counts"] = unknown_counts

    features[features == 0.0] = 0.0
    families = mapped.iloc[selected_rows].to_numpy(dtype=object)[good]
    return features, families, stats


def _add_stats(total: dict[str, Any], update: dict[str, Any]) -> None:
    for field in (
        "raw_rows",
        "repeated_header_rows",
        "missing_label_rows",
        "raw_missing_field_rows",
        "unmapped_label_rows",
        "defaulted_label_rows",
        "mapped_label_rows",
        "invalid_numeric_or_nonfinite_rows",
        "valid_rows",
        "unknown_categorical_rows",
        "missing_categorical_values",
    ):
        total[field] += int(update[field])
    total["unmapped_label_counts"].update(update["unmapped_label_counts"])
    total["defaulted_label_counts"].update(update["defaulted_label_counts"])
    for column, counts in update["unknown_categorical_counts"].items():
        total["unknown_categorical_counts"][column].update(counts)


def _safe_class_directory(class_name: str) -> str:
    if class_name in {"", ".", ".."} or any(
        separator in class_name for separator in ("/", "\\")
    ):
        raise ValueError(f"Unsafe class name for output directory: {class_name!r}")
    return class_name


def _write_streaming_manifest(
    staging: Path,
    spec: DatasetSpec,
    schema: _Schema,
    shard_entries: list[dict[str, Any]],
    *,
    builder_source_sha256: str,
    split_overlap_audit_sha256: str,
) -> dict[str, Any]:
    """Write the fail-closed schema consumed by ``streaming_full``."""

    expected_ids = list(range(len(spec.class_order)))
    flattened = [class_id for task in spec.tasks for class_id in task]
    if sorted(flattened) != expected_ids or len(flattened) != len(expected_ids):
        raise ValueError(
            f"{spec.name} tasks must contain every class id exactly once"
        )
    if spec.normal_class_id is not None:
        if (
            not spec.tasks
            or spec.normal_class_id not in spec.tasks[0]
            or len(spec.tasks[0]) < 2
        ):
            raise ValueError(
                f"{spec.name} Task 0 must explicitly include its normal class and an attack"
            )
        if not 0 <= spec.normal_class_id < len(spec.class_order):
            raise ValueError(f"{spec.name} normal_class_id is outside the class order")
        if spec.class_order[spec.normal_class_id] != "Normal":
            raise ValueError(
                f"{spec.name} normal_class_id must identify the explicit Normal class"
            )

    classes: list[dict[str, Any]] = []
    missing: list[str] = []
    for class_id, class_name in enumerate(spec.class_order):
        record: dict[str, Any] = {"id": class_id, "name": class_name}
        for split in ("train", "test"):
            shards = [
                {
                    "path": entry["relative_path"],
                    "rows": entry["rows"],
                    "sha256": entry["sha256"],
                }
                for entry in shard_entries
                if entry["class"] == class_name and entry["split"] == split
            ]
            if not shards:
                missing.append(f"{class_name}/{split}")
            record[split] = shards
        classes.append(record)
    if missing:
        raise RuntimeError(
            f"{spec.name} is not safe for streaming validation; missing non-empty "
            f"class/split shards: {missing}"
        )

    payload = {
        "schema_version": 1,
        "dataset": spec.name,
        "problem_type": spec.problem_type,
        "task_semantics": spec.task_semantics,
        "metric_profile": spec.metric_profile,
        "feature_dim": len(schema.columns),
        "normal_class_id": spec.normal_class_id,
        "tasks": [list(task) for task in spec.tasks],
        "classes": classes,
        "source": {
            "builder": "ofra-fullcache",
            "builder_version": TOOL_VERSION,
            "feature_schema_sha256": schema.sha256,
            "class_cap": None,
            "builder_source_sha256": builder_source_sha256,
            "split_overlap_audit_sha256": split_overlap_audit_sha256,
        },
    }
    if spec.source_revision is not None:
        payload["source_revision"] = spec.source_revision
    if spec.source_contract_sha256 is not None:
        payload["source_contract_sha256"] = spec.source_contract_sha256
    _write_json(staging / "streaming_manifest.json", payload)
    return payload


def _commit_staging(staging: Path, final: Path, overwrite: bool) -> None:
    if not final.exists():
        staging.rename(final)
        return
    if not overwrite:
        raise FileExistsError(f"Output already exists: {final}")
    backup = final.with_name(f".{final.name}.previous-{uuid.uuid4().hex}")
    final.rename(backup)
    try:
        staging.rename(final)
    except Exception:
        backup.rename(final)
        raise
    else:
        shutil.rmtree(backup)


def build_dataset_cache(
    dataset: str,
    data_root: str | Path,
    output_root: str | Path,
    options: BuildOptions | None = None,
    *,
    progress: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """Build one uncapped dataset cache and return its manifest.

    Only one CSV chunk and its derived class/split arrays are held at a time.
    The completed dataset directory replaces an old one only after every
    shard and manifest has been written successfully.
    """

    if dataset not in DATASET_SPECS:
        raise KeyError(f"Unsupported dataset {dataset!r}; choose {list(DATASET_SPECS)}")
    spec = DATASET_SPECS[dataset]
    options = options or BuildOptions()
    options.validate()
    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    final = output_root / spec.name
    if final.exists() and not options.overwrite:
        raise FileExistsError(
            f"Output already exists: {final}. Use overwrite=True only after review."
        )

    source = _source_directory(data_root, spec)
    files = _discover_files(data_root, spec)
    file_contract = _validate_source_files(
        files, spec, options.strict_files, source
    )
    capture_contract = _load_capture_contract(source, files, spec)
    if capture_contract is not None:
        contract_reference = (
            f"bundled:{spec.bundled_contract_relative}"
            if spec.bundled_contract_relative is not None
            else spec.source_contract_relative
        )
        file_contract["frozen_capture_contract"] = {
            "contract_reference": contract_reference,
            "sha256": capture_contract.sha256,
            "source_revision": capture_contract.source_revision,
            "test_capture_count": len(capture_contract.test_captures),
            "train_capture_count": len(files) - len(capture_contract.test_captures),
            "capture_disjoint": True,
            "all_source_hashes_verified_before_build": True,
        }
    schema = _preflight_schema(files, spec, options.chunk_rows, source)
    source_record = builder_source_record()
    if progress:
        progress(
            f"[{spec.name}] {len(files)} files; locked {len(schema.columns)} "
            f"float32 features; no class cap"
        )

    staging = output_root / f".{spec.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    if capture_contract is not None:
        shutil.copyfile(
            capture_contract.path, staging / "source_capture_split.json"
        )
    shard_entries: list[dict[str, Any]] = []
    shard_sequence: defaultdict[tuple[str, str], int] = defaultdict(int)
    row_counts: dict[str, dict[str, int]] = {
        split: {class_name: 0 for class_name in spec.class_order}
        for split in ("train", "test")
    }
    quality: dict[str, Any] = {
        "raw_rows": 0,
        "repeated_header_rows": 0,
        "missing_label_rows": 0,
        "raw_missing_field_rows": 0,
        "unmapped_label_rows": 0,
        "defaulted_label_rows": 0,
        "mapped_label_rows": 0,
        "invalid_numeric_or_nonfinite_rows": 0,
        "valid_rows": 0,
        "unmapped_label_counts": Counter(),
        "defaulted_label_counts": Counter(),
        "unknown_categorical_rows": 0,
        "missing_categorical_values": 0,
        "unknown_categorical_counts": defaultdict(Counter),
        "chunks_processed": 0,
    }
    raw_entries: list[dict[str, Any]] = []
    header_by_name = {
        entry["relative_path"].casefold(): dict(entry)
        for entry in schema.raw_file_headers
    }
    test_buckets = int(round(options.test_fraction * HASH_BUCKETS))
    per_file_quality_fields = (
        "raw_rows",
        "repeated_header_rows",
        "missing_label_rows",
        "raw_missing_field_rows",
        "unmapped_label_rows",
        "defaulted_label_rows",
        "mapped_label_rows",
        "invalid_numeric_or_nonfinite_rows",
        "valid_rows",
        "unknown_categorical_rows",
        "missing_categorical_values",
        "chunks_processed",
    )

    try:
        for file_index, path in enumerate(files):
            source_relative = _relative_source_path(path, source)
            quality_before = {
                field: int(quality[field]) for field in per_file_quality_fields
            }
            row_counts_before = {
                split: dict(counts) for split, counts in row_counts.items()
            }
            if spec.split_strategy == "official_train_test_files":
                source_split = _official_split(path)
            elif spec.split_strategy == "frozen_capture_manifest":
                assert capture_contract is not None
                source_split = (
                    "test"
                    if source_relative in capture_contract.test_captures
                    else "train"
                )
            else:
                source_split = None
            raw_entry = header_by_name[source_relative.casefold()]
            raw_entry.update(
                {
                    "size_bytes": int(path.stat().st_size),
                    "sha256": (
                        capture_contract.verified_file_sha256[source_relative]
                        if capture_contract is not None
                        else sha256_file(path)
                    ),
                }
            )
            if source_split is not None:
                raw_entry[
                    "capture_split"
                    if spec.split_strategy == "frozen_capture_manifest"
                    else "official_split"
                ] = source_split
            raw_entries.append(raw_entry)

            if progress:
                progress(f"[{spec.name}] reading {source_relative}")
            for chunk_index, chunk in enumerate(
                _read_chunks(path, spec, options.chunk_rows)
            ):
                quality["chunks_processed"] += 1
                features, families, chunk_stats = _prepare_chunk(
                    chunk,
                    spec,
                    schema,
                    strict_unmapped=options.strict_unmapped_labels,
                    source_label=(
                        _mapped_parent_label(path, source, spec)
                        if spec.label_source == "parent_directory"
                        else None
                    ),
                )
                _add_stats(quality, chunk_stats)
                if not len(features):
                    continue

                if source_split is None:
                    splits = feature_hash_splits(
                        features,
                        seed=options.split_seed,
                        test_fraction=options.test_fraction,
                    )
                else:
                    splits = np.full(len(features), source_split, dtype=object)

                for split in ("train", "test"):
                    split_mask = splits == split
                    if not split_mask.any():
                        continue
                    for class_index, class_name in enumerate(spec.class_order):
                        mask = split_mask & (families == class_name)
                        rows = int(mask.sum())
                        if not rows:
                            continue
                        class_directory = _safe_class_directory(class_name)
                        directory = staging / split / class_directory
                        directory.mkdir(parents=True, exist_ok=True)
                        sequence_key = (split, class_name)
                        sequence = shard_sequence[sequence_key]
                        shard_sequence[sequence_key] += 1
                        name = (
                            f"part-{sequence:06d}-f{file_index:03d}-"
                            f"c{chunk_index:06d}.npy"
                        )
                        shard_path = directory / name
                        shard = np.ascontiguousarray(features[mask], dtype="<f4")
                        np.save(shard_path, shard, allow_pickle=False)
                        relative = shard_path.relative_to(staging).as_posix()
                        row_counts[split][class_name] += rows
                        shard_entries.append(
                            {
                                "relative_path": relative,
                                "split": split,
                                "class": class_name,
                                "class_index": class_index,
                                "rows": rows,
                                "columns": int(shard.shape[1]),
                                "dtype": "float32-little-endian",
                                "size_bytes": int(shard_path.stat().st_size),
                                "sha256": sha256_file(shard_path),
                                "source_file": source_relative,
                                "source_chunk_index": chunk_index,
                            }
                        )

            raw_entry["quality_counts"] = {
                field: int(quality[field]) - quality_before[field]
                for field in per_file_quality_fields
            }
            raw_entry["rows_written_by_split_and_class"] = {
                split: {
                    class_name: row_counts[split][class_name]
                    - row_counts_before[split][class_name]
                    for class_name in spec.class_order
                }
                for split in ("train", "test")
            }

        written_rows = sum(
            count for split_counts in row_counts.values() for count in split_counts.values()
        )
        if written_rows != quality["valid_rows"]:
            raise RuntimeError(
                f"Internal row-accounting failure: wrote {written_rows}, "
                f"validated {quality['valid_rows']}"
            )
        if written_rows == 0:
            raise RuntimeError(f"{spec.name} produced no valid rows")

        quality["unmapped_label_counts"] = dict(
            sorted(quality["unmapped_label_counts"].items())
        )
        quality["defaulted_label_counts"] = dict(
            sorted(quality["defaulted_label_counts"].items())
        )
        quality["unknown_categorical_counts"] = {
            column: dict(sorted(counts.items()))
            for column, counts in sorted(
                quality["unknown_categorical_counts"].items()
            )
        }
        quality["rows_written"] = written_rows
        quality["class_cap"] = None
        quality["sampling_applied"] = False

        feature_schema = {
            "format_version": 1,
            "dataset": spec.name,
            "dtype": "float32-little-endian",
            "feature_count": len(schema.columns),
            "feature_columns": list(schema.columns),
            "raw_numeric_columns": list(schema.raw_numeric_columns),
            "categorical_columns": list(schema.categorical_columns),
            "categorical_vocabulary_train_only": {
                column: list(values)
                for column, values in schema.categorical_vocabulary.items()
            },
            "categorical_unknown_policy": (
                "all-zero block; test categories never extend the train schema"
                if schema.categorical_columns
                else None
            ),
            "feature_schema_sha256": schema.sha256,
            "class_order": list(spec.class_order),
            "class_to_index": {
                class_name: index
                for index, class_name in enumerate(spec.class_order)
            },
            "label_mapping": dict(spec.label_mapping),
            "unknown_label_family": spec.unknown_label_family,
            "label_source": spec.label_source,
            "problem_type": spec.problem_type,
            "task_semantics": spec.task_semantics,
            "metric_profile": spec.metric_profile,
            "normal_class_id": spec.normal_class_id,
            "configured_drop_columns": list(spec.drop_columns),
            "configured_identifier_columns": list(spec.identifier_columns),
            "numeric_cleaning": {
                "coercion": "pandas.to_numeric(errors='coerce')",
                "finite_required": True,
                "absolute_value_strictly_below": FLOAT32_ABS_LIMIT,
                "zero_canonicalisation": "-0.0 -> +0.0",
                "drop_row_if_any_raw_field_missing": spec.drop_any_raw_missing,
            },
        }
        counts_payload = {
            "format_version": 1,
            "dataset": spec.name,
            "by_split_and_class": row_counts,
            "split_totals": {
                split: sum(class_counts.values())
                for split, class_counts in row_counts.items()
            },
            "total": written_rows,
            "quality_counts": quality,
        }
        _write_json(staging / "feature_schema.json", feature_schema)
        _write_json(staging / "row_counts.json", counts_payload)

        if spec.split_strategy in {
            "official_train_test_files",
            "frozen_capture_manifest",
        }:
            from .overlap import exact_official_split_overlap

            if progress:
                progress(
                    f"[{spec.name}] exact official train/test overlap audit "
                    f"(SQLite external set)"
                )
            overlap_report = exact_official_split_overlap(
                staging,
                spec.name,
                shard_entries,
                len(schema.columns),
                batch_rows=options.overlap_batch_rows,
                work_directory=options.overlap_work_directory,
                progress=None,
            )
        else:
            overlap_report = {
                "schema_version": 1,
                "dataset": spec.name,
                "applicable_to_official_split": False,
                "strategy": "feature_hash_group",
                "exact_cross_split_overlap_unique_feature_rows": 0,
                "reason": (
                    "The deterministic split is a pure function of the complete "
                    "cleaned float32 feature row, so identical features cannot cross."
                ),
                "official_split_action": "not_applicable",
            }
            overlap_report["canonical_report_sha256"] = hashlib.sha256(
                _json_bytes(overlap_report)
            ).hexdigest()
        _write_json(staging / "split_overlap_audit.json", overlap_report)
        overlap_sidecar_sha256 = sha256_file(
            staging / "split_overlap_audit.json"
        )
        streaming_manifest = _write_streaming_manifest(
            staging,
            spec,
            schema,
            shard_entries,
            builder_source_sha256=source_record["canonical_source_sha256"],
            split_overlap_audit_sha256=overlap_sidecar_sha256,
        )
        streaming_manifest_sha256 = sha256_file(
            staging / "streaming_manifest.json"
        )
        if spec.split_strategy == "feature_hash_group_80_20":
            split_protocol: dict[str, Any] = {
                "strategy": "feature_hash_group",
                "train_fraction": 1.0 - options.test_fraction,
                "test_fraction": options.test_fraction,
                "split_seed": options.split_seed,
                "row_hash_algorithm": ROW_HASH_ALGORITHM,
                "seed_mixer": SPLIT_MIX_ALGORITHM,
                "hash_input": "ordered cleaned model-feature float32 bytes only",
                "bucket_count": HASH_BUCKETS,
                "test_buckets": list(range(test_buckets)),
                "guarantee": (
                    "Rows with identical cleaned model features receive the same split, "
                    "including across files and chunks."
                ),
            }
        elif spec.split_strategy == "official_train_test_files":
            split_protocol = {
                "strategy": "official_train_test_files",
                "filename_rule": "case-insensitive train/test token",
                "resplit": False,
                "guarantee": "Every valid row retains its source file's official split.",
                "categorical_vocabulary_fit": (
                    "official train files only"
                    if schema.categorical_columns
                    else None
                ),
                "test_only_category_policy": (
                    "all-zero one-hot block"
                    if schema.categorical_columns
                    else None
                ),
            }
        elif spec.split_strategy == "frozen_capture_manifest":
            assert capture_contract is not None
            split_protocol = {
                "strategy": "frozen_capture_manifest",
                "resplit": False,
                "source_revision": capture_contract.source_revision,
                "source_contract_relative": "source_capture_split.json",
                "source_contract_sha256": capture_contract.sha256,
                "test_captures": sorted(capture_contract.test_captures),
                "train_capture_count": len(files) - len(capture_contract.test_captures),
                "test_capture_count": len(capture_contract.test_captures),
                "capture_disjoint": True,
                "row_level_resplit": False,
                "guarantee": (
                    "Every valid row from one capture stays in exactly one frozen "
                    "partition; no capture crosses train and test."
                ),
            }
        else:  # pragma: no cover - guarded by the declared dataset specs
            raise ValueError(f"Unsupported split strategy: {spec.split_strategy}")

        manifest = {
            "format_version": 1,
            "tool": "ofra-fullcache",
            "tool_version": TOOL_VERSION,
            "created_utc": _utc_now(),
            "dataset": spec.name,
            "problem_type": spec.problem_type,
            "task_semantics": spec.task_semantics,
            "metric_profile": spec.metric_profile,
            "normal_class_id": spec.normal_class_id,
            "source_directory": str(source),
            "output_layout": "<dataset>/<split>/<class>/part-*.npy",
            "uncapped": True,
            "builder_source": source_record,
            "chunk_rows": options.chunk_rows,
            "file_contract": file_contract,
            "split_protocol": split_protocol,
            "feature_schema_sha256": schema.sha256,
            "raw_files": raw_entries,
            "shards": shard_entries,
            "row_counts": row_counts,
            "quality_counts": quality,
            "split_overlap_audit": overlap_report,
            "sidecars": {
                "feature_schema.json": sha256_file(staging / "feature_schema.json"),
                "row_counts.json": sha256_file(staging / "row_counts.json"),
                "split_overlap_audit.json": overlap_sidecar_sha256,
                "streaming_manifest.json": streaming_manifest_sha256,
                **(
                    {"source_capture_split.json": capture_contract.sha256}
                    if capture_contract is not None
                    else {}
                ),
            },
            "streaming_runner": {
                "compatible": True,
                "relative_manifest": "streaming_manifest.json",
                "schema_version": streaming_manifest["schema_version"],
                "normal_class_id": streaming_manifest["normal_class_id"],
                "tasks": streaming_manifest["tasks"],
                "binding": (
                    "sibling manifest.json sidecars.streaming_manifest.json "
                    "binds the finalized streaming manifest bytes"
                ),
            },
        }
        manifest["canonical_manifest"] = {
            "algorithm": "sha256-canonical-json-excluding-this-field-v1",
            "sha256": hashlib.sha256(_json_bytes(manifest)).hexdigest(),
        }
        _write_json(staging / "manifest.json", manifest)
        _commit_staging(staging, final, options.overwrite)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    if progress:
        progress(
            f"[{spec.name}] wrote {manifest['quality_counts']['rows_written']:,} "
            f"rows in {len(manifest['shards']):,} shards"
        )
    return manifest


def write_root_manifest(output_root: str | Path) -> dict[str, Any]:
    """Index all completed dataset manifests below an output root."""

    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    datasets: list[dict[str, Any]] = []
    for dataset in DATASET_SPECS:
        manifest_path = output_root / dataset / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        datasets.append(
            {
                "dataset": dataset,
                "relative_manifest": manifest_path.relative_to(output_root).as_posix(),
                "manifest_sha256": sha256_file(manifest_path),
                "rows": int(manifest["quality_counts"]["rows_written"]),
                "shards": len(manifest["shards"]),
                "raw_files": len(manifest["raw_files"]),
                "feature_schema_sha256": manifest["feature_schema_sha256"],
                "canonical_manifest_sha256": manifest["canonical_manifest"][
                    "sha256"
                ],
                "streaming_manifest": f"{dataset}/streaming_manifest.json",
                "streaming_manifest_sha256": sha256_file(
                    output_root / dataset / "streaming_manifest.json"
                ),
            }
        )
    root_manifest = {
        "format_version": 1,
        "tool": "ofra-fullcache",
        "tool_version": TOOL_VERSION,
        "created_utc": _utc_now(),
        "datasets": datasets,
    }
    _write_json_atomic(output_root / "manifest.json", root_manifest)
    return root_manifest
