"""Malaya capture-shift data for Exp H, without exposure to held captures.

The historical cache contains raw numeric features (not scaled values). Its
frozen capture test is the shifted evaluation set. A deterministic grouped
holdout from the original training captures supplies in-distribution tests.
Malaya labels identify applications; this setting is not intrusion detection.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from ncd.data import PreparedData, make_tasks


REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_ORDER = [
    "BitTorrent", "ChromeRDP", "Discord", "EA Origin", "Microsoft Teams",
    "Slack", "Steam", "TeamViewer", "Webex", "Zoom",
]
SPLIT_SEED = 42


def _row_keys(values: np.ndarray) -> np.ndarray:
    """Exact canonical feature bytes, including equal treatment of signed zero."""
    canonical = np.ascontiguousarray(values, dtype="<f4").copy()
    canonical[canonical == 0] = 0.0
    return canonical.view(np.dtype((np.void, canonical.dtype.itemsize * canonical.shape[1]))).ravel()


def _load_split(root: Path, manifest: dict, split: str) -> tuple[np.ndarray, np.ndarray]:
    pieces: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for class_id, name in enumerate(CLASS_ORDER):
        shards = sorted(
            (item for item in manifest["shards"] if item["split"] == split and item["class"] == name),
            key=lambda item: item["relative_path"],
        )
        if not shards:
            raise ValueError(f"Malaya cache has no {split} shards for {name}")
        for shard in shards:
            values = np.load(root / shard["relative_path"], allow_pickle=False)
            if values.shape != (int(shard["rows"]), 77) or not np.isfinite(values).all():
                raise ValueError(f"Invalid Malaya feature shard: {shard['relative_path']}")
            pieces.append(np.asarray(values, dtype=np.float32))
            labels.append(np.full(len(values), class_id, dtype=np.int64))
    return np.concatenate(pieces), np.concatenate(labels)


def _grouped_holdout(values: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Approximate 20% per-class rows while never splitting identical features.

    Some feature groups contain thousands of rows. Stratifying group counts
    rather than row counts gives very uneven holdouts. We instead visit whole
    groups in a fixed random order and add a group if doing so reduces summed
    squared relative distance without exceeding a per-class row target. This uses only
    labels and counts, never scores, held-capture features, or outcomes.
    """
    keys = _row_keys(values)
    _, groups = np.unique(keys, return_inverse=True)
    group_count = int(groups.max()) + 1
    counts = np.zeros((group_count, len(CLASS_ORDER)), dtype=np.int64)
    np.add.at(counts, (groups, labels), 1)
    targets = np.maximum(1, np.rint(np.bincount(labels, minlength=len(CLASS_ORDER)) * 0.2))
    assigned_counts = np.zeros(len(CLASS_ORDER), dtype=np.int64)
    assigned_groups = np.zeros(group_count, dtype=bool)
    distance = float(np.sum(((assigned_counts - targets) / targets) ** 2))
    for group in np.random.default_rng(SPLIT_SEED).permutation(group_count):
        candidate = assigned_counts + counts[group]
        if np.any(candidate > targets):
            continue
        candidate_distance = float(np.sum(((candidate - targets) / targets) ** 2))
        if candidate_distance < distance:
            assigned_groups[group] = True
            assigned_counts = candidate
            distance = candidate_distance
    is_test = assigned_groups[groups]
    train_idx, test_idx = np.flatnonzero(~is_test), np.flatnonzero(is_test)
    if set(np.unique(labels[train_idx])) != set(range(len(CLASS_ORDER))):
        raise ValueError("Grouped Malaya training partition is missing a class")
    if set(np.unique(labels[test_idx])) != set(range(len(CLASS_ORDER))):
        raise ValueError("Grouped Malaya ID partition is missing a class")
    if np.intersect1d(keys[train_idx], keys[test_idx]).size:
        raise RuntimeError("Exact feature rows crossed Malaya fit and ID partitions")
    return train_idx, test_idx, {
        "algorithm": "seeded whole-feature-group greedy fill toward per-class row targets, no target overshoot",
        "seed": SPLIT_SEED,
        "target_id_fraction": 0.2,
        "actual_id_fraction": float(len(test_idx) / len(labels)),
        "groups": group_count,
        "per_class_target_id_rows": targets.astype(int).tolist(),
        "per_class_actual_id_rows": assigned_counts.tolist(),
        "group_key": "all 77 raw little-endian float32 feature bytes, signed zeros canonicalized",
        "fit_id_exact_feature_overlap": 0,
    }


def prepare_malaya(config: dict) -> tuple[PreparedData, np.ndarray, np.ndarray, dict]:
    """Return training/ID data, held-capture features and labels, and split facts.

    ``malaya_cache_root`` is an absolute path or a path relative to the repo.
    All held-capture rows matching the original training pool are removed;
    therefore shifted tests cannot overlap either training or ID holdout.
    The scaler is fitted only on the first two classes of the fit partition.
    """
    root = Path(config["malaya_cache_root"])
    if not root.is_absolute():
        root = REPO_ROOT / root
    root = root.resolve()
    manifest_path = root / "manifest.json"
    schema_path = root / "feature_schema.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    split = manifest.get("split_protocol", {})
    if manifest.get("dataset") != "malaya-network-gt" or not split.get("capture_disjoint"):
        raise ValueError("Exp H requires the frozen capture-disjoint Malaya cache")
    if schema.get("class_order") != CLASS_ORDER or int(schema.get("feature_count", 0)) != 77:
        raise ValueError("Malaya cache does not match the established 10-class, 77-feature schema")
    feature_names = list(schema["feature_columns"])
    identifiers = {"src_ip", "dst_ip", "src_port", "dst_port", "timestamp", "capture_id"}
    if identifiers.intersection(feature_names) or len(set(feature_names)) != 77:
        raise ValueError("Malaya features include identifiers or invalid feature names")
    train_captures = {item["source_file"] for item in manifest["shards"] if item["split"] == "train"}
    shifted_captures = {item["source_file"] for item in manifest["shards"] if item["split"] == "test"}
    if train_captures & shifted_captures:
        raise ValueError("Malaya source captures cross training and shifted partitions")

    source_x, source_y = _load_split(root, manifest, "train")
    shifted_x, shifted_y = _load_split(root, manifest, "test")
    train_idx, id_idx, holdout = _grouped_holdout(source_x, source_y)
    overlap = np.isin(_row_keys(shifted_x), np.unique(_row_keys(source_x)))
    retained_x, retained_y = shifted_x[~overlap], shifted_y[~overlap]
    if set(np.unique(retained_y)) != set(range(len(CLASS_ORDER))):
        raise ValueError("Duplicate exclusion removed an entire shifted Malaya class")

    tasks = make_tasks(len(CLASS_ORDER), 2)
    fit_x, fit_y = source_x[train_idx], source_y[train_idx]
    task0 = np.isin(fit_y, tasks[0])
    scaler = StandardScaler().fit(fit_x[task0].astype(np.float64))
    transform = lambda values: np.asarray(scaler.transform(values.astype(np.float64)), dtype=np.float32)
    x_train, x_id, x_shift = transform(fit_x), transform(source_x[id_idx]), transform(retained_x)
    if not all(np.isfinite(values).all() for values in (x_train, x_id, x_shift)):
        raise ValueError("Non-finite features after Task-0-only Malaya scaling")
    prepared = PreparedData(
        x_train=x_train,
        y_train=fit_y,
        x_test=x_id,
        y_test=source_y[id_idx],
        class_order=list(CLASS_ORDER),
        tasks=tasks,
        feature_names=[f"numeric__{name}" for name in feature_names],
        source_files=[str(manifest_path), str(schema_path), str(root / "source_capture_split.json")],
    )
    counts = []
    for class_id, name in enumerate(CLASS_ORDER):
        original_class = shifted_y == class_id
        counts.append({
            "class_id": class_id,
            "class_name": name,
            "fit_rows": int(np.sum(fit_y == class_id)),
            "id_rows": int(np.sum(source_y[id_idx] == class_id)),
            "source_pool_rows": int(np.sum(source_y == class_id)),
            "shift_original_rows": int(np.sum(original_class)),
            "shift_overlap_excluded_rows": int(np.sum(original_class & overlap)),
            "shift_rows": int(np.sum(retained_y == class_id)),
            "train_captures": sorted({item["source_file"] for item in manifest["shards"] if item["split"] == "train" and item["class"] == name}),
            "shift_captures": sorted({item["source_file"] for item in manifest["shards"] if item["split"] == "test" and item["class"] == name}),
        })
    metadata = {
        "setting": "S2",
        "dataset": "MalayaNetwork_GT",
        "problem_type": "application_classification",
        "scope_limit": "Capture-shift application classification; not direct NIDS attack-class evidence.",
        "feature_count": 77,
        "feature_names": feature_names,
        "feature_alignment": [
            {
                "setting": "S2",
                "training_feature": name,
                "shift_feature": name,
                "action": "retain",
                "notes": "Identical raw numeric feature in training and held captures; Task-0 fit scaling shared.",
            }
            for name in feature_names
        ] + [
            {
                "setting": "S2",
                "training_feature": name,
                "shift_feature": name,
                "action": "exclude",
                "notes": "Identifier or timestamp excluded from the historical feature cache.",
            }
            for name in ("src_ip", "dst_ip", "src_port", "dst_port", "timestamp")
        ],
        "source_features": "Raw numeric float32 cache; Task-0 scaler fitted only after ID exclusion.",
        "source_revision": split.get("source_revision"),
        "capture_disjoint": True,
        "train_capture_count": len(train_captures),
        "shift_capture_count": len(shifted_captures),
        "id_holdout": holdout,
        "preprocessing": "StandardScaler fit on fit-partition Task 0 classes only; no shifted/ID fitting",
        "preprocessing_fit_rows": int(task0.sum()),
        "shift_duplicate_exclusion": "Exact feature equality against full original training pool (fit plus ID)",
        "shift_original_rows": int(len(shifted_y)),
        "shift_overlap_excluded_rows": int(overlap.sum()),
        "shift_rows": int(len(retained_y)),
        "per_class_counts": counts,
    }
    return prepared, x_shift, retained_y, metadata
