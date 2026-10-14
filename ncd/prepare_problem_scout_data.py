"""Section 21 data: official NSL, globally cleaned UNSW, frozen grouped CIC.

This is data preparation only. Historical inputs are never modified. All tests
are uncapped. Training is uncapped by default; legacy caps require an explicit
CLI choice. Numeric/categorical preprocessing is fitted on Task-0 training only.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from collections import Counter

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ncd.audit_quality_tabular import exact_feature_groups, conflict_flags
from ncd.data import PreparedData, CACHE_DATASETS, _load_raw, _cap_for, make_tasks
from ncd.exp_h_data import CIC_ALIASES, _transform
from ncd.source_commit import source_commit

SLUGS = {"NSL-KDD": "nsl-kdd", "UNSW-NB15": "unsw-nb15",
         "CIC-IDS-2017": "cic-ids-2017", "CSE-CIC-IDS2018": "cic-ids-2018"}
TABULAR_CLASSES = {
    "NSL-KDD": ["Normal", "DoS", "Probe", "R2L", "U2R"],
    "UNSW-NB15": ["Normal", "Generic", "Exploits", "Fuzzers", "DoS",
                  "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"],
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative(path):
    return os.path.relpath(Path(path).resolve(), ROOT).replace("\\", "/")


def counts(y, classes):
    return {name: int(np.count_nonzero(y == i)) for i, name in enumerate(classes)}


def clean_unsw(train, y_train, test, y_test):
    """Exact mixed-type raw tuples; drop ALL conflict members, keep one/group."""
    groups, n_groups = exact_feature_groups(train, test)
    labels = np.concatenate([y_train, y_test])
    bad_rows, bad_groups = conflict_flags(groups, labels, n_groups)
    _, first = np.unique(groups, return_index=True)
    keep = first[~bad_rows[first]]
    frame = pd.concat([train, test], ignore_index=True)
    selected_labels = labels[keep]
    train_ids, test_ids = train_test_split(keep, test_size=0.2, random_state=42,
                                         stratify=selected_labels, shuffle=True)
    # Sort by pooled source position, not by target label or difficulty.
    train_ids, test_ids = np.sort(train_ids), np.sort(test_ids)
    if len(np.intersect1d(groups[train_ids], groups[test_ids])):
        raise AssertionError("UNSW cleaned raw-feature groups overlap")
    quality = {
        "raw_rows": len(frame), "raw_feature_groups": n_groups,
        "conflict_groups_removed": int(bad_groups.sum()),
        "conflict_member_rows_removed": int(bad_rows.sum()),
        "same_label_excess_rows_removed": int(len(frame) - bad_rows.sum() - len(keep)),
        "retained_rows": len(keep), "retained_train_rows": len(train_ids),
        "retained_test_rows": len(test_ids), "train_test_exact_overlap_rows": 0,
        "remaining_conflict_groups": 0,
        "equality": "All 42 raw model-input columns as exact parsed mixed-type tuples; excludes id, binary label, attack_cat; signed-zero normalized; no float32 conversion before grouping",
        "split": "After full pooled official train+test cleaning: stratified 80/20, sklearn random_state=42; no oversampling or test caps",
        "raw_feature_dtypes": {k: str(v) for k, v in frame.dtypes.items()},
    }
    assert quality["raw_rows"] == (quality["conflict_member_rows_removed"]
           + quality["same_label_excess_rows_removed"] + quality["retained_rows"])
    return (frame.iloc[train_ids].reset_index(drop=True), labels[train_ids],
            frame.iloc[test_ids].reset_index(drop=True), labels[test_ids], quality)


def row_hashes(x, chunk_rows=25000):
    """Routing hashes only; candidates are always checked with complete bytes."""
    result = np.empty(len(x), dtype=np.uint64)
    for start in range(0, len(x), chunk_rows):
        values = np.array(x[start:start + chunk_rows], dtype="<f4", order="C")
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite numeric cache values")
        values[values == 0] = 0.0
        words = values.view("<u4")
        h = np.full(len(values), np.uint64(1469598103934665603))
        for j in range(words.shape[1]):
            h ^= words[:, j].astype(np.uint64)
            h *= np.uint64(1099511628211)
        result[start:start + len(values)] = h
    return result


def _row_bytes(row):
    row = np.array(row, dtype="<f4", copy=True)
    row[row == 0] = 0.0
    return row.tobytes()


def numeric_overlap(train, test, chunk_rows=25000):
    """Exact all-feature byte comparison, using hashes solely for candidates."""
    if train.shape[1] != test.shape[1]:
        raise ValueError("Incompatible feature widths")
    a, b = row_hashes(train, chunk_rows), row_hashes(test, chunk_rows)
    shared = np.intersect1d(a, b)
    ia = np.flatnonzero(np.isin(a, shared))
    ib = np.flatnonzero(np.isin(b, shared))
    keys = {_row_bytes(train[i]) for i in ia}
    matched = sum(_row_bytes(test[i]) in keys for i in ib)
    return {"test_matching_any_train_rows": int(matched), "test_rows": len(test),
            "test_matching_any_train_rate": float(matched / len(test)) if len(test) else None,
            "hash_candidate_train_rows": len(ia), "hash_candidate_test_rows": len(ib),
            "equality": "Complete little-endian float32 feature bytes; signed-zero canonical; hash routing followed by full bytes, not hash-only equality"}


def tabular_collision_diagnosis(dataset, raw_root, config):
    """Explain prepared model-view collisions without changing either split."""
    train, yt, test, ye, classes, _, sources = _load_raw(dataset, raw_root)
    if dataset == "UNSW-NB15":
        train, yt, test, ye, _ = clean_unsw(train, yt, test, ye)
    data, _, _, info = prepare_setting(dataset, config)
    a, b = row_hashes(data.x_train), row_hashes(data.x_test)
    shared = np.intersect1d(a, b)
    ia, ib = np.flatnonzero(np.isin(a, shared)), np.flatnonzero(np.isin(b, shared))
    keys = {_row_bytes(data.x_train[i]): int(i) for i in ia}
    fields, label_pairs, examples = Counter(), Counter(), []
    for i in ib:
        j = keys.get(_row_bytes(data.x_test[i]))
        if j is None:
            continue
        different = tuple(c for c in train.columns if train.iloc[j][c] != test.iloc[i][c])
        fields[different] += 1
        label_pairs[(classes[yt[j]], classes[ye[i]])] += 1
        if different and len(examples) < 5:
            examples.append({"train_index": j, "test_index": int(i),
                             "different_raw_fields": {c: [str(train.iloc[j][c]), str(test.iloc[i][c])] for c in different}})
    return {"dataset": dataset, "preparation_script_sha256": info["preparation_script_sha256"],
            "raw": info["quality"]["raw"], "transformed": info["quality"]["transformed"],
            "raw_difference_patterns": [{"columns": list(k), "matched_test_rows": v} for k, v in fields.items()],
            "representative_match_label_pairs": [{"train_label": k[0], "test_label": k[1], "matched_test_rows": v} for k, v in label_pairs.items()],
            "examples": examples,
            "interpretation": "NSL official raw matches remain unchanged. UNSW raw groups were globally cleaned; any new model-view matches reflect Task0 transform/float32 representation, not missed raw duplicates. Test observations are not deleted post hoc."}


def save_dataset(out, dataset, x_train, y_train, x_test, y_test, classes,
                 tasks, names, metadata, x_shift=None, y_shift=None):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "data_quality.json").exists():
        raise FileExistsError(f"Completed prepared cache already exists: {out}")
    arrays = {"train_x.npy": x_train, "train_y.npy": y_train,
              "test_x.npy": x_test, "test_y.npy": y_test}
    if x_shift is not None:
        arrays.update({"shift_x.npy": x_shift, "shift_y.npy": y_shift})
    receipts = []
    for filename, array in arrays.items():
        path = out / filename
        # Streaming CIC arrays may already be the final mmap file.
        if not isinstance(array, np.memmap) or Path(array.filename).resolve() != path.resolve():
            if path.exists():
                raise FileExistsError(path)
            np.save(path, array, allow_pickle=False)
        elif array.mode != "r":
            array.flush()
        receipts.append({"path": filename, "bytes": path.stat().st_size,
                         "sha256": sha256(path), "shape": list(array.shape),
                         "dtype": str(array.dtype)})
    metadata.update({"status": "complete", "dataset": dataset,
                     "class_order": classes, "tasks": tasks,
                     "feature_names": list(names), "feature_count": len(names),
                     "train_counts": counts(y_train, classes), "id_counts": counts(y_test, classes),
                     "shift_counts": counts(y_shift, classes) if y_shift is not None else {},
                     "cache_files": receipts})
    (out / "data_quality.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def prepare_tabular(dataset, raw_root, out, base):
    train, yt, test, ye, classes, categorical, sources = _load_raw(dataset, raw_root)
    original = {"train": counts(yt, classes), "test": counts(ye, classes)}
    if dataset == "UNSW-NB15":
        train, yt, test, ye, raw_quality = clean_unsw(train, yt, test, ye)
    else:
        groups, n_groups = exact_feature_groups(train, test)
        matches = np.isin(groups[len(train):], groups[:len(train)])
        flags, bad = conflict_flags(groups, np.concatenate([yt, ye]), n_groups)
        raw_quality = {"raw_rows": len(train) + len(test), "raw_feature_groups": n_groups,
                       "train_test_exact_overlap_rows": int(matches.sum()),
                       "conflict_member_rows_retained": int(flags.sum()),
                       "conflict_groups_retained": int(bad.sum()),
                       "split": "Official KDDTrain+/KDDTest+ unchanged by Section21 instruction",
                       "equality": "Exact parsed mixed-type feature tuples; signed-zero normalized; no float32 conversion before checking"}
    tasks = make_tasks(len(classes), 1 if dataset == "NSL-KDD" else 2)
    # _transform also applies the exact Section15 feature order and one-hot policy.
    xt, xe, _, names = _transform(train, yt, test, test.iloc[:1], categorical, tasks)
    overlap = numeric_overlap(xt, xe)
    metadata = dict(base, original_counts=original, raw_feature_names=list(train.columns),
                    source_files=[relative(p) for p in sources],
                    input_files=[{"path": relative(p), "bytes": Path(p).stat().st_size, "sha256": sha256(p)} for p in sources],
                    quality={"raw": raw_quality, "transformed": overlap},
                    preprocessing="Section15 ColumnTransformer: median-impute/StandardScaler numeric + one-hot(handle_unknown=ignore), fitted only on Task0 train; final float32",
                    train_cap_policy="none", test_cap_policy="none",
                    caveats=["Raw equality is defined before model float32 conversion; task0-only one-hot and float32 rounding may collapse distinct raw vectors. Transformed overlap is reported separately."])
    return save_dataset(out, dataset, xt, yt, xe, ye, classes, tasks, names, metadata)


def verify_frozen_cache(dataset, cache_root):
    """Bind the exact historical full-byte audit to currently verified shards."""
    folder = cache_root / SLUGS[dataset]
    manifest_path, schema_path = folder / "manifest.json", folder / "feature_schema.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    expected_classes = CACHE_DATASETS[dataset]["class_order"]
    if schema["class_order"] != expected_classes or schema["feature_count"] != 78:
        raise ValueError("Frozen CIC class/feature schema differs")
    # Top-level feature_schema_sha256 binds the builder's semantic schema;
    # sidecars binds the serialized file bytes, including its metadata/format.
    if sha256(schema_path) != manifest["sidecars"]["feature_schema.json"]:
        raise ValueError("Frozen CIC schema hash mismatch")
    entries = manifest["shards"]
    expected = [e["relative_path"] for e in entries]
    actual = [p.relative_to(folder).as_posix() for p in folder.glob("*/*/part-*.npy")]
    if len(set(expected)) != len(expected) or set(actual) != set(expected):
        raise ValueError("Frozen CIC shard coverage differs")
    inputs = []
    for entry in entries:
        p = folder / entry["relative_path"]
        digest = sha256(p)
        if p.stat().st_size != entry["size_bytes"] or digest != entry["sha256"]:
            raise ValueError(f"Frozen CIC shard mismatch: {p}")
        arr = np.load(p, mmap_mode="r", allow_pickle=False)
        if arr.shape != (entry["rows"], 78) or arr.dtype != np.float32:
            raise ValueError(f"Frozen CIC shard shape/type mismatch: {p}")
        inputs.append({"path": relative(p), "bytes": p.stat().st_size, "sha256": digest})
    audit_path = ROOT / "ncd/results/data_quality/section18/cic/cic_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    evidence = next(d for d in audit["datasets"] if d["dataset"] == dataset)
    receipt = evidence["provenance"]["input_receipt_summary"]
    if receipt["manifest_sha256"] != sha256(manifest_path) or receipt["feature_schema_sha256"] != sha256(schema_path):
        raise ValueError("Historical exact audit is not bound to current verified cache")
    if any(r["test_matches_any_train_rows"] for r in evidence["views"]["uncapped"]["rows"]):
        raise ValueError("Historical full-byte CIC audit contains cross-split overlap")
    return folder, manifest, schema, inputs, {
        "verification": "Current SHA256 and size of every exact expected shard and schema checked against frozen manifest; manifest/schema hashes match Section18 full-row exact audit",
        "audit_source": relative(audit_path), "audit_sha256": sha256(audit_path),
        "manifest_sha256": sha256(manifest_path), "schema_sha256": sha256(schema_path),
        "full_uncapped_raw_audit": evidence["views"]["uncapped"],
        "raw_train_test_exact_overlap_rows": 0,
        "caveat": "Within-split duplicates and label conflicts are retained by explicit Section21 instruction to preserve these feature-hash grouped caches; zero cross-split overlap is not absence of label noise."}


def shard_selection(folder, classes, split, caps=None):
    selected = []
    for c, name in enumerate(classes):
        paths = sorted((folder / split / name).glob("part-*.npy"))
        sizes = [len(np.load(p, mmap_mode="r", allow_pickle=False)) for p in paths]
        total = sum(sizes)
        cap = _cap_for(caps, name) if caps is not None else None
        ids = np.sort(np.random.default_rng(42000 + (1000 if split == "test" else 0) + c).choice(total, cap, replace=False)) if cap is not None and total > cap else None
        offset = 0
        for path, n in zip(paths, sizes):
            local = None if ids is None else ids[(ids >= offset) & (ids < offset + n)] - offset
            if local is None or len(local):
                selected.append((path, c, local, n if local is None else len(local)))
            offset += n
    return selected


def numeric_chunks(selection, indices, chunk_rows):
    for path, c, selected, n in selection:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        for start in range(0, n, chunk_rows):
            rows = arr[start:start + chunk_rows] if selected is None else arr[selected[start:start + chunk_rows]]
            values = np.asarray(rows[:, indices], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite frozen cache {path}")
            yield values, c


class RawCacheView:
    """Bounded-memory indexed projection of source shards for exact checking."""
    def __init__(self, selection, indices):
        self.parts = [(np.load(p, mmap_mode="r", allow_pickle=False), ids, n)
                      for p, _, ids, n in selection]
        self.ends = np.cumsum([n for _, _, n in self.parts])
        self.indices = indices
        self.shape = (int(self.ends[-1]), len(indices))

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            part = int(np.searchsorted(self.ends, key, side="right"))
            offset = int(self.ends[part - 1]) if part else 0
            array, selected, _ = self.parts[part]
            index = key - offset
            return array[index if selected is None else selected[index], self.indices]
        start, stop, step = key.indices(len(self))
        if step != 1:
            raise ValueError("Only contiguous source slices are supported")
        pieces = []
        while start < stop:
            part = int(np.searchsorted(self.ends, start, side="right"))
            offset = int(self.ends[part - 1]) if part else 0
            end = min(stop, int(self.ends[part]))
            array, selected, _ = self.parts[part]
            ids = slice(start - offset, end - offset) if selected is None else selected[start - offset:end - offset]
            pieces.append(array[ids][:, self.indices])
            start = end
        return np.concatenate(pieces, axis=0)


def write_numeric_split(out, split, selection, indices, scaler, chunk_rows):
    n = sum(item[3] for item in selection)
    x = np.lib.format.open_memmap(out / f"{split}_x.npy", mode="w+", dtype=np.float32, shape=(n, len(indices)))
    y = np.lib.format.open_memmap(out / f"{split}_y.npy", mode="w+", dtype=np.int64, shape=(n,))
    cursor = 0
    for values, c in numeric_chunks(selection, indices, chunk_rows):
        z = scaler.transform(values).astype(np.float32)
        if not np.isfinite(z).all():
            raise ValueError("Nonfinite transformed CIC rows")
        x[cursor:cursor + len(z)] = z
        y[cursor:cursor + len(z)] = c
        cursor += len(z)
    x.flush(); y.flush()
    return x, y


def prepare_cic(dataset, cache_root, out, base, legacy_train_caps=False, chunk_rows=25000):
    folder, manifest, schema, inputs, raw_quality = verify_frozen_cache(dataset, cache_root)
    classes = list(schema["class_order"])
    names = list(schema["feature_columns"])
    caps = CACHE_DATASETS[dataset]["train_cap"] if legacy_train_caps else None
    train = shard_selection(folder, classes, "train", caps)
    test = shard_selection(folder, classes, "test", None)
    original_names = names.copy()
    if dataset == "CIC-IDS-2017":
        names = [n for n in names if n != "Fwd Header Length.1"]
    indices = [original_names.index(n) for n in names]
    tasks = make_tasks(len(classes), 2)
    raw_train_view = RawCacheView(train, indices)
    raw_quality["actual_projected_train_test"] = numeric_overlap(raw_train_view, RawCacheView(test, indices), chunk_rows)
    if raw_quality["actual_projected_train_test"]["test_matching_any_train_rows"]:
        raise ValueError("Actual projected raw CIC train/test overlap; requires explicit resolution")
    scaler = StandardScaler()
    for values, c in numeric_chunks([s for s in train if s[1] in tasks[0]], indices, chunk_rows):
        scaler.partial_fit(values)
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob("*.npy")) or (out / "data_quality.json").exists():
        raise FileExistsError(f"Prepared output must be new: {out}")
    xt, yt = write_numeric_split(out, "train", train, indices, scaler, chunk_rows)
    xe, ye = write_numeric_split(out, "test", test, indices, scaler, chunk_rows)
    xs = ys = None
    alignment = []
    shift_quality = None
    if dataset == "CIC-IDS-2017":
        target, _, target_schema, target_inputs, target_quality = verify_frozen_cache("CSE-CIC-IDS2018", cache_root)
        mapped = [CIC_ALIASES.get(n, n) for n in names]
        if len(mapped) != 77 or set(mapped) != set(target_schema["feature_columns"]) - {"Protocol"}:
            raise ValueError("S1 feature alignment changed")
        target_indices = [target_schema["feature_columns"].index(n) for n in mapped]
        target_classes = [c for c in classes if c in target_schema["class_order"]]
        shift = shard_selection(target, target_classes, "test", None)
        shift = [(p, classes.index(target_classes[c]), ids, n) for p, c, ids, n in shift]
        raw_quality["actual_projected_train_shift"] = numeric_overlap(raw_train_view, RawCacheView(shift, target_indices), chunk_rows)
        xs, ys = write_numeric_split(out, "shift", shift, target_indices, scaler, chunk_rows)
        shift_quality = numeric_overlap(xt, xs, chunk_rows)
        inputs.extend(target_inputs)
        alignment = [{"training_feature": a, "shift_feature": b, "action": "aligned"} for a, b in zip(names, mapped)]
        alignment.extend([{"training_feature": "Fwd Header Length.1", "shift_feature": "", "action": "drop"},
                          {"training_feature": "", "shift_feature": "Protocol", "action": "drop"}])
        raw_quality["shift_target_cache_verification"] = target_quality
    transformed = numeric_overlap(xt, xe, chunk_rows)
    metadata = dict(base, source_files=[relative(folder / "manifest.json"), relative(folder / "feature_schema.json")],
                    input_files=inputs, raw_feature_names=names,
                    quality={"raw": raw_quality, "transformed": transformed, "transformed_shift": shift_quality},
                    preprocessing="Finite raw cache values; numeric StandardScaler incrementally fitted in float64 on Task0 training only; test/shift never fit; outputs float32",
                    scaler={"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "var": scaler.var_.tolist(), "n_samples_seen": int(scaler.n_samples_seen_)},
                    train_cap_policy="explicit legacy NCD train caps" if legacy_train_caps else "none; full frozen cache train partition",
                    test_cap_policy="none; complete natural frozen test partition", feature_alignment=alignment,
                    caveats=["Original CICFlowMeter data/labels retained, not Engelen corrected regenerated captures.",
                             "Existing within-split duplicates and conflicting labels retained; grouped split prevents raw exact train/test overlap.",
                             "S1 PortScan has no CSE-CIC-IDS2018 target class and is excluded only from cross-environment evaluation."])
    return save_dataset(out, dataset, xt, yt, xe, ye, classes, tasks,
                        ["numeric__" + n for n in names], metadata, xs, ys)


def prepare_setting(dataset, config):
    root = Path(config.get("problem_scout_data_root", "../exp_h_section21_data/clean_v1"))
    if not root.is_absolute():
        root = ROOT / root
    folder = root.resolve() / SLUGS[dataset]
    receipt_path = folder / "data_quality.json"
    info = json.loads(receipt_path.read_text(encoding="utf-8"))
    if info.get("status") != "complete" or info.get("dataset") != dataset:
        raise ValueError("Prepared data is incomplete or wrong dataset")
    required = {f"{split}_{field}.npy" for split in ("train", "test") for field in ("x", "y")}
    if dataset == "CIC-IDS-2017":
        required |= {"shift_x.npy", "shift_y.npy"}
    listed = [r["path"] for r in info["cache_files"]]
    if len(listed) != len(set(listed)) or set(listed) != required:
        raise ValueError("Prepared cache receipt must cover exact expected arrays once")
    arrays = {}
    for item in info["cache_files"]:
        p = folder / item["path"]
        if p.stat().st_size != item["bytes"] or sha256(p) != item["sha256"]:
            raise ValueError(f"Prepared array hash/size differs: {p}")
        a = np.load(p, mmap_mode="r", allow_pickle=False)
        if list(a.shape) != item["shape"] or str(a.dtype) != item["dtype"]:
            raise ValueError("Prepared array schema differs")
        arrays[item["path"]] = a
    if dataset in CACHE_DATASETS and info["class_order"] != CACHE_DATASETS[dataset]["class_order"]:
        raise ValueError("Prepared class mapping changed")
    if dataset in TABULAR_CLASSES and info["class_order"] != TABULAR_CLASSES[dataset]:
        raise ValueError("Prepared class mapping changed")
    if info["tasks"] != make_tasks(len(info["class_order"]), 1 if dataset == "NSL-KDD" else 2):
        raise ValueError("Prepared task order changed")
    for split in ("train", "test", "shift"):
        if f"{split}_x.npy" not in arrays:
            continue
        x, y = arrays[f"{split}_x.npy"], arrays[f"{split}_y.npy"]
        if x.ndim != 2 or x.shape[1] != info["feature_count"] or x.dtype != np.float32:
            raise ValueError("Prepared model feature shape/type differs")
        if y.ndim != 1 or len(y) != len(x) or y.dtype != np.int64:
            raise ValueError("Prepared label shape/type differs")
        if len(x) == 0 or np.any(y < 0) or np.any(y >= len(info["class_order"])):
            raise ValueError("Prepared labels are empty or out of range")
        expected_counts = info[{"train": "train_counts", "test": "id_counts", "shift": "shift_counts"}[split]]
        if counts(y, info["class_order"]) != expected_counts:
            raise ValueError("Prepared class counts differ")
        for start in range(0, len(x), 25000):
            if not np.isfinite(x[start:start + 25000]).all():
                raise ValueError("Prepared model features are nonfinite")
    if len(info["feature_names"]) != info["feature_count"]:
        raise ValueError("Prepared feature name count differs")
    data = PreparedData(arrays["train_x.npy"], arrays["train_y.npy"], arrays["test_x.npy"], arrays["test_y.npy"],
                        info["class_order"], info["tasks"], info["feature_names"], [relative(receipt_path)])
    return data, arrays.get("shift_x.npy"), arrays.get("shift_y.npy"), info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=list(SLUGS), default=list(SLUGS))
    p.add_argument("--raw-root", type=Path, default=ROOT.parent / "kaggle_leon_release/datasets")
    p.add_argument("--cic-cache-root", type=Path, default=ROOT.parent / "ofra_formal_v3_cache_20260716")
    p.add_argument("--output-dir", type=Path, default=ROOT.parent / "exp_h_section21_data/clean_v1")
    p.add_argument("--results-dir", type=Path, default=ROOT / "ncd/results/problem_scout/section21")
    p.add_argument("--legacy-train-caps", action="store_true", help="Explicit choice only; never caps test/shift")
    p.add_argument("--chunk-rows", type=int, default=25000)
    p.add_argument("--log-name", default="preparation.log")
    p.add_argument("--diagnose-tabular-only", action="store_true")
    p.add_argument("--verify-cic-only", action="store_true")
    args = p.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    base = {"git_commit": source_commit(ROOT), "preparation_script_sha256": sha256(__file__),
            "seed": 42, "command": "python -I ncd/prepare_problem_scout_data.py " + " ".join(sys.argv[1:]),
            "log_path": relative(args.results_dir / args.log_name),
            "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__}}
    if args.diagnose_tabular_only:
        diagnoses = [tabular_collision_diagnosis(d, args.raw_root, {"problem_scout_data_root": str(args.output_dir)})
                     for d in args.datasets if d in TABULAR_CLASSES]
        result = dict(base, datasets=diagnoses)
        (args.results_dir / "tabular_representation_quality.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
        return
    if args.verify_cic_only:
        result = dict(base, datasets=[])
        for d in args.datasets:
            if d not in CACHE_DATASETS:
                continue
            _, manifest, schema, inputs, quality = verify_frozen_cache(d, args.cic_cache_root)
            result["datasets"].append({"dataset": d, "class_order": schema["class_order"],
                                       "row_counts": manifest["row_counts"], "input_files": inputs, "quality": quality})
            print(json.dumps({"event": "verified", "dataset": d, "shards": len(inputs), "row_counts": manifest["row_counts"]}), flush=True)
        (args.results_dir / "cic_frozen_cache_quality.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return
    for dataset in args.datasets:
        begin = time.time()
        print(json.dumps({"event": "start", "dataset": dataset}), flush=True)
        out = args.output_dir / SLUGS[dataset]
        if dataset in CACHE_DATASETS:
            result = prepare_cic(dataset, args.cic_cache_root, out, base, args.legacy_train_caps, args.chunk_rows)
        else:
            result = prepare_tabular(dataset, args.raw_root, out, base)
        (args.results_dir / f"{SLUGS[dataset]}_data_quality.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"event": "complete", "dataset": dataset, "train_counts": result["train_counts"],
                          "id_counts": result["id_counts"], "shift_counts": result["shift_counts"],
                          "quality": result["quality"].get("transformed"), "seconds": time.time() - begin}), flush=True)
        del result
        gc.collect()
    print(json.dumps({"event": "all_complete", "seconds": time.time() - started}), flush=True)


if __name__ == "__main__":
    main()
