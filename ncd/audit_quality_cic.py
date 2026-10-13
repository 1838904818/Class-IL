"""Read-only Section 18 audits of the actual NCD CIC feature caches.

Hashing only routes rows into bounded on-disk partitions. All duplicate,
cross-split and conflict decisions compare complete canonical feature bytes.
No training, preprocessing fit, resampling, or input changes are performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ncd.data import CACHE_DATASETS, _cap_for, _sample_shards, make_tasks
from ncd.exp_h_data import CIC_ALIASES


def canonical_rows(values):
    values = np.array(values, dtype="<f4", order="C", copy=True)
    values[values == 0] = 0.0
    values[np.isnan(values)] = np.float32(np.nan)
    return values


def feature_hash(values):
    words = values.view("<u4").reshape(values.shape)
    h = np.full(len(values), np.uint64(14695981039346656037))
    with np.errstate(over="ignore"):
        for column in range(words.shape[1]):
            h ^= words[:, column]
            h *= np.uint64(1099511628211)
        h ^= np.uint64(words.shape[1])
        h *= np.uint64(1099511628211)
    return h


def mix_hash(h):
    z = h.copy()
    with np.errstate(over="ignore"):
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z ^= z >> np.uint64(31)
    return z


def original_is_test(h):
    return mix_hash(h ^ np.uint64(42)) % np.uint64(10000) < 2000


FIELDS = ("train_rows", "test_rows", "train_unique_features", "train_duplicate_excess",
          "train_duplicate_affected_rows", "test_matches_any_train_rows", "conflict_train_rows",
          "conflict_test_rows", "conflict_rows", "conflict_unique_features")


def empty_stats(classes):
    return {name: dict.fromkeys(FIELDS, 0) for name in [*classes, "__all__"]}


def count_partition(records, classes):
    out = empty_stats(classes)
    if not len(records):
        return out
    _, groups = np.unique(records["features"], return_inverse=True)
    count = int(groups.max()) + 1
    train, test = records["split"] == 0, records["split"] == 1
    train_counts = np.bincount(groups[train], minlength=count)
    low, high = np.full(count, 255, np.uint8), np.zeros(count, np.uint8)
    np.minimum.at(low, groups, records["label"])
    np.maximum.at(high, groups, records["label"])
    conflict = low != high
    for class_id, name in enumerate([*classes, "__all__"]):
        member = np.ones(len(records), bool) if name == "__all__" else records["label"] == class_id
        tr, te = member & train, member & test
        unique = int(np.unique(groups[tr]).size)
        out[name].update(
            train_rows=int(tr.sum()), test_rows=int(te.sum()), train_unique_features=unique,
            train_duplicate_excess=int(tr.sum()) - unique,
            train_duplicate_affected_rows=int(np.sum(tr & (train_counts[groups] > 1))),
            test_matches_any_train_rows=int(np.sum(te & (train_counts[groups] > 0))),
            conflict_train_rows=int(np.sum(tr & conflict[groups])),
            conflict_test_rows=int(np.sum(te & conflict[groups])),
            conflict_rows=int(np.sum(member & conflict[groups])),
            conflict_unique_features=int(np.unique(groups[member & conflict[groups]]).size),
        )
    return out


def add_stats(total, partial):
    for name in total:
        for field in FIELDS:
            total[name][field] += partial[name][field]


def finish_stats(stats):
    rows = []
    for name, counts in stats.items():
        rows.append(dict(class_name=name, **counts,
                         train_duplicate_excess_rate=counts["train_duplicate_excess"] / counts["train_rows"] if counts["train_rows"] else None,
                         test_matches_any_train_rate=counts["test_matches_any_train_rows"] / counts["test_rows"] if counts["test_rows"] else None))
    overall = rows[-1]["test_matches_any_train_rate"]
    triggering = [r["class_name"] for r in rows[:-1] if r["test_matches_any_train_rate"] is not None and r["test_matches_any_train_rate"] >= .2]
    return dict(rows=rows, decision="去重重跑" if (overall is not None and overall >= .05) or triggering else "保持",
                trigger_classes=triggering, overall_test_match_rate=overall)


def audit_cache(dataset, root, output, chunk_rows, buckets, builder_core):
    spec = CACHE_DATASETS[dataset]
    root = root / spec["slug"]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    schema = json.loads((root / "feature_schema.json").read_text(encoding="utf-8"))
    classes = spec["class_order"]
    features = schema["feature_columns"]
    if schema["class_order"] != classes or manifest["split_protocol"]["strategy"] != "feature_hash_group":
        raise ValueError("Unexpected CIC cache split/schema")
    dtype = np.dtype([("features", f"V{4*len(features)}"), ("label", "u1"), ("split", "u1"), ("capped", "u1")])
    stats = {v: empty_stats(classes) for v in ("uncapped", "ncd_capped")}
    split_mismatches = 0
    header_mismatches = 0
    processed = 0
    maximum_partition_bytes = 0
    started = time.perf_counter()
    # Owned scratch only. Remove explicit partition files at the end, not inputs.
    scratch = Path(tempfile.mkdtemp(prefix="cic_exact_rows_", dir=output)).resolve()
    if scratch.parent != output.resolve():
        raise RuntimeError("Scratch directory escaped the requested output directory")
    files = [scratch / f"bucket_{i:03d}.bin" for i in range(buckets)]
    handles = [path.open("wb") for path in files]
    try:
        for split_id, split in enumerate(("train", "test")):
            for class_id, name in enumerate(classes):
                paths = sorted((root / split / name).glob("part-*.npy"))
                arrays = [np.load(path, mmap_mode="r", allow_pickle=False) for path in paths]
                total = sum(len(a) for a in arrays)
                if total != manifest["row_counts"][split][name]:
                    raise ValueError(f"Incomplete cache {dataset}/{split}/{name}")
                cap = _cap_for(spec[f"{split}_cap"], name)
                selected = None if cap is None or total <= cap else np.sort(np.random.default_rng(42000 + split_id*1000 + class_id).choice(total, cap, replace=False))
                offset = 0
                for array in arrays:
                    for start in range(0, len(array), chunk_rows):
                        values = canonical_rows(array[start:start+chunk_rows])
                        if not np.isfinite(values).all():
                            raise ValueError("Cache contains nonfinite features")
                        h = feature_hash(values)
                        split_mismatches += int(np.sum(original_is_test(h) != bool(split_id)))
                        if dataset == "CIC-IDS-2017":
                            header_mismatches += int(np.sum(values[:, features.index("Fwd Header Length")] != values[:, features.index("Fwd Header Length.1")]))
                        record = np.empty(len(values), dtype=dtype)
                        record["features"] = values.view(f"V{4*len(features)}").ravel()
                        record["label"], record["split"] = class_id, split_id
                        record["capped"] = int(selected is None)
                        if selected is not None:
                            lo, hi = np.searchsorted(selected, [offset+start, offset+start+len(values)])
                            record["capped"][selected[lo:hi] - offset - start] = 1
                        bucket = (mix_hash(h) % np.uint64(buckets)).astype(np.int64)
                        order = np.argsort(bucket, kind="stable")
                        sorted_bucket, ordered = bucket[order], record[order]
                        unique, boundaries = np.unique(sorted_bucket, return_index=True)
                        ends = np.r_[boundaries[1:], len(ordered)]
                        for number, left, right in zip(unique, boundaries, ends):
                            handles[int(number)].write(ordered[left:right].tobytes())
                        processed += len(values)
                    offset += len(array)
                print(json.dumps(dict(stage="partition", dataset=dataset, split=split, class_name=name, rows=total, processed=processed)), flush=True)
        for handle in handles:
            handle.close()
        if split_mismatches:
            raise RuntimeError(f"Recomputed cache split differs for {split_mismatches} rows")
        maximum_partition_bytes = max(path.stat().st_size for path in files)
        if maximum_partition_bytes > 256*1024*1024:
            raise RuntimeError("Exact partition exceeds 256 MiB; increase bucket count before retry")
        for index, path in enumerate(files):
            records = np.fromfile(path, dtype=dtype)
            add_stats(stats["uncapped"], count_partition(records, classes))
            add_stats(stats["ncd_capped"], count_partition(records[records["capped"] == 1], classes))
            del records
            path.unlink()
            if index % 16 == 0:
                print(json.dumps(dict(stage="exact_count", dataset=dataset, bucket=index, buckets=buckets)), flush=True)
        scratch.rmdir()
    finally:
        for handle in handles:
            if not handle.closed:
                handle.close()
    expected = next(item["sha256"] for item in manifest["builder_source"]["files"] if item["relative_path"] == "fullcache/core.py")
    actual = hashlib.sha256(builder_core.read_bytes()).hexdigest() if builder_core and builder_core.exists() else None
    return dict(dataset=dataset, feature_count=len(features), representation="Complete raw-valued cache float32 features before Task-0 scaling; label excluded; -0=+0; canonical NaNs (none observed). Float32 conversion may merge distinct raw-text/float64 inputs.",
                definitions={"train_duplicate_excess": "Within each class: N_train minus distinct feature vectors; global row counts distinct vectors across all classes, so global excess need not sum class excess when labels conflict.",
                             "train_duplicate_affected_rows": "Rows whose feature vector occurs at least twice in training, including across labels.",
                             "conflict_rows": "All rows, including first occurrences, in feature groups with more than one family label across train plus test.",
                             "test_matches_any_train_rows": "Test rows equal to any training feature vector, irrespective of labels; repeated test rows all counted."},
                provenance={"cache_split": "feature_hash_group; 80/20 deterministic feature-only hashing; not src random stratified splitter", "split_seed": 42,
                            "loader": "ncd/data.py::_load_frozen_cache -> _sample_shards", "builder_source_file": "fullcache/core.py", "builder_expected_sha256": expected,
                            "builder_inspected_sha256": actual, "builder_source_matches": expected == actual, "recomputed_split_mismatches": split_mismatches,
                            "source_raw_file_count": len(manifest["raw_files"]), "cache_rows": processed},
                caps={"train": spec["train_cap"], "test": spec["test_cap"], "sampler": "numpy.default_rng(42000 + split_index*1000 + class_id); sorted exact global row sample"},
                tasks=[[classes[c] for c in task] for task in make_tasks(len(classes), 2)],
                duplicate_header_different_rows=header_mismatches if dataset == "CIC-IDS-2017" else None,
                views={name: finish_stats(value) for name, value in stats.items()}, maximum_partition_bytes=maximum_partition_bytes,
                partition_routing="SplitMix64(FNV1a64(feature words)) modulo bucket count; equality always full feature bytes",
                elapsed_seconds=time.perf_counter()-started)


def small_view_audit(train_x, train_y, test_x, test_y, classes):
    width = train_x.shape[1]
    dtype = np.dtype([("features", f"V{4*width}"), ("label", "u1"), ("split", "u1")])
    records = np.empty(len(train_y)+len(test_y), dtype=dtype)
    for split, x, y, offset in ((0, train_x, train_y, 0), (1, test_x, test_y, len(train_y))):
        x = canonical_rows(x)
        records["features"][offset:offset+len(y)] = x.view(f"V{4*width}").ravel()
        records["label"][offset:offset+len(y)] = y
        records["split"][offset:offset+len(y)] = split
    return finish_stats(count_partition(records, classes))


def audit_s1(root):
    old_root, new_root = root / "cic-ids-2017", root / "cic-ids-2018"
    old, new = [json.loads((p / "feature_schema.json").read_text(encoding="utf-8")) for p in (old_root, new_root)]
    names = [n for n in old["feature_columns"] if n != "Fwd Header Length.1"]
    old_idx = [old["feature_columns"].index(n) for n in names]
    new_idx = [new["feature_columns"].index(CIC_ALIASES.get(n, n)) for n in names]
    classes = old["class_order"]
    loaded = {}
    for split in ("train", "test", "shift"):
        xs, ys = [], []
        spec = CACHE_DATASETS["CSE-CIC-IDS2018" if split == "shift" else "CIC-IDS-2017"]
        for c, name in enumerate(classes):
            if split == "shift" and name not in new["class_order"]:
                continue
            paths = sorted(((new_root / "test" if split == "shift" else old_root / split) / name).glob("part-*.npy"))
            cap = _cap_for(spec["train_cap" if split == "train" else "test_cap"], name)
            values = _sample_shards(paths, cap, 42000 + (0 if split == "train" else 1000) + c)
            xs.append(values[:, new_idx if split == "shift" else old_idx])
            ys.append(np.full(len(values), c, np.int64))
        loaded[split] = np.concatenate(xs), np.concatenate(ys)
    return dict(setting="S1", feature_count=77,
                source_id=small_view_audit(*loaded["train"], *loaded["test"], classes),
                target_shift=small_view_audit(*loaded["train"], *loaded["shift"], classes),
                note="Exact raw aligned 77-feature Exp H S1 views and actual existing caps. Target sampling class id follows CIC2017 order, as prepare_cic does.")


def normalized_raw_label(label):
    return re.sub(r"^Web Attack.*?(Brute Force|XSS|Sql Injection)$", r"Web Attack – \1", label.strip())


def rare_label_splits(dataset, cache_root, raw_root, manifest, schema, labels, label_files, chunk_rows):
    """Replay frozen builder cleaning/split for scoped low-count raw labels.

    Full CSV chunks preserve drop_any_raw_missing, including dropped identifiers.
    Only selected rows are converted to numeric features. This is not a data rewrite.
    """
    spec = CACHE_DATASETS[dataset]
    features = schema["feature_columns"]
    width = f"V{4*len(features)}"
    arrays = {label: [] for label in labels}
    observed = Counter()
    for relative in sorted({path for label in labels for path in label_files[label]}):
        path = raw_root/spec["slug"]/relative
        for frame in pd.read_csv(path, chunksize=chunk_rows, low_memory=False,
                                 encoding="latin-1" if dataset == "CIC-IDS-2017" else "utf-8"):
            frame.columns = [str(name).strip() for name in frame.columns]
            normal = frame["Label"].astype("string").map(lambda s: normalized_raw_label(str(s)))
            for label in labels:
                selected = frame.loc[normal.eq(label)]
                observed[label] += len(selected)
                selected = selected.loc[~selected.isna().any(axis=1)]
                if selected.empty:
                    continue
                values = selected[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
                good = np.isfinite(values).all(axis=1) & (np.abs(values) < 3e38).all(axis=1)
                if good.any():
                    arrays[label].append(canonical_rows(values[good]))
        print(json.dumps(dict(stage="rare_label_split", dataset=dataset, file=relative)), flush=True)
    result = {}
    for label in labels:
        values = np.concatenate(arrays[label]) if arrays[label] else np.empty((0,len(features)),np.float32)
        family = schema["label_mapping"].get(label)
        if family not in spec["class_order"]:
            raise ValueError(f"Rare label {label!r} does not map to frozen class sequence")
        test = original_is_test(feature_hash(values))
        counts = {"train": int((~test).sum()), "test": int(test.sum())}
        verified = {}
        for split, mask in (("train", ~test), ("test", test)):
            # Presence check uses full canonical byte equality, never hash alone.
            sought = values[mask].view(width).ravel()
            seen = np.zeros(len(sought),bool)
            for path in sorted((cache_root/spec["slug"]/split/family).glob("part-*.npy")):
                source = np.load(path,mmap_mode="r",allow_pickle=False)
                for start in range(0,len(source),chunk_rows):
                    have = canonical_rows(source[start:start+chunk_rows]).view(width).ravel()
                    seen |= np.isin(sought,have)
                if seen.all():
                    break
            verified[split] = int(seen.sum())
            if not seen.all():
                raise ValueError(f"Rare {label!r}/{split} feature rows absent from frozen cache")
        cap_unchanged = all(_cap_for(spec[f"{split}_cap"],family) is None or
                            manifest["row_counts"][split][family] <= _cap_for(spec[f"{split}_cap"],family)
                            for split in ("train","test"))
        if not cap_unchanged:
            raise ValueError(f"Rare {label!r} belongs to a capped family; exact source-index cap recovery required")
        result[label] = dict(raw_rows_seen=int(observed[label]), valid_cleaned_rows=len(values),
                             discarded_by_builder_cleaning=int(observed[label])-len(values),
                             uncapped=counts, ncd_capped=dict(counts),
                             verified_raw_row_presence_in_frozen_cache=verified,
                             cap_reason="Both family split counts are below caps, or this family has no cap; all rare rows retained.",
                             method="Frozen builder drop-any-raw-missing, numeric finite and abs<3e38, float32 canonical rows, full 78-feature FNV1a+SplitMix64 seed42 split; exact cache feature membership checked.")
    return result


def raw_labels(dataset, cache_root, raw_root, chunk_rows):
    spec = CACHE_DATASETS[dataset]
    manifest = json.loads((cache_root/spec["slug"]/"manifest.json").read_text(encoding="utf-8"))
    schema = json.loads((cache_root/spec["slug"]/"feature_schema.json").read_text(encoding="utf-8"))
    counts, missing, available, label_files, spellings = Counter(), [], [], {}, {}
    for item in manifest["raw_files"]:
        path = raw_root/spec["slug"]/item["relative_path"]
        if not path.exists():
            missing.append(item["relative_path"])
            continue
        available.append(item["relative_path"])
        for frame in pd.read_csv(path, usecols=lambda name: name.strip()=="Label", chunksize=chunk_rows,
                                 encoding="latin-1" if dataset == "CIC-IDS-2017" else "utf-8", low_memory=False):
            values = frame.iloc[:, 0].dropna().astype(str).str.strip()
            for label, number in values[values != "Label"].value_counts().items():
                normalized=normalized_raw_label(label)
                counts[normalized]+=int(number)
                label_files.setdefault(normalized,set()).add(item["relative_path"])
                spellings.setdefault(normalized,set()).add(label)
        print(json.dumps(dict(stage="raw_labels", dataset=dataset, file=item["relative_path"])), flush=True)
    rare = rare_label_splits(dataset,cache_root,raw_root,manifest,schema,
                             [label for label,count in counts.items() if count<=500],label_files,chunk_rows)
    rows = []
    for label, count in sorted(counts.items()):
        family = schema["label_mapping"].get(label)
        task = spec["class_order"].index(family)//2 if family in spec["class_order"] else None
        rows.append(dict(raw_label=label, observed_spellings=sorted(spellings[label]), source_files=sorted(label_files[label]), rows=count, below_100=count<100, mapped_family=family,
                         task_index=task, handling="Merged into family; not oversampled. Family retained in original class sequence." if family else "Unmapped; investigate",
                         rare_split_evidence=rare.get(label),
                         small_partition=any(n<100 for n in rare[label]["ncd_capped"].values()) if label in rare else None))
    return dict(dataset=dataset, available_files=available, missing_files=missing,
                complete_raw_file_coverage=not missing, count_stage="Raw label counts before finite/missing-feature cleaning; repeated header labels excluded.",
                small_partition_scope="Cleaned/capped train or test count <100, evaluated for observed raw totals <=500; null for other labels, not inferred false.", rows=rows)


def write_raw_label_csv(output, value):
    rows=[]
    for row in value["raw_label_counts"]["rows"]:
        evidence=row.get("rare_split_evidence")
        rows.append(dict(dataset=value["dataset"],raw_label=row["raw_label"],raw_rows=row["rows"],
                         below_100=row["below_100"],small_partition=row.get("small_partition"),
                         mapped_family=row["mapped_family"],task_index=row["task_index"],
                         train_before_cap=evidence["uncapped"]["train"] if evidence else None,
                         test_before_cap=evidence["uncapped"]["test"] if evidence else None,
                         train_after_cap=evidence["ncd_capped"]["train"] if evidence else None,
                         test_after_cap=evidence["ncd_capped"]["test"] if evidence else None,
                         discarded_by_cleaning=evidence["discarded_by_builder_cleaning"] if evidence else None,
                         source_files="; ".join(row["source_files"])))
    write_csv(output/(CACHE_DATASETS[value["dataset"]]["slug"]+"_raw_labels.csv"),rows)


def refresh_small_partitions(args):
    """Reuse completed exact-duplicate audit; inspect only missing small-label splits."""
    path=args.output_dir/"cic_audit.json"
    result=json.loads(path.read_text(encoding="utf-8"))
    previous_hash=result["audit_script_sha256"]
    for value in result["datasets"]:
        dataset=value["dataset"]
        spec=CACHE_DATASETS[dataset]
        cache=args.cache_root/spec["slug"]
        manifest=json.loads((cache/"manifest.json").read_text(encoding="utf-8"))
        schema=json.loads((cache/"feature_schema.json").read_text(encoding="utf-8"))
        rows=value["raw_label_counts"]["rows"]
        selected=[r for r in rows if r["rows"]<=500 and not r.get("rare_split_evidence")]
        evidence=rare_label_splits(dataset,args.cache_root,args.raw_root,manifest,schema,
                                    [r["raw_label"] for r in selected],
                                    {r["raw_label"]:r["source_files"] for r in selected},args.chunk_rows)
        for row in rows:
            if row["raw_label"] in evidence:
                added=evidence[row["raw_label"]]
                if added["raw_rows_seen"]!=row["rows"]:
                    raise ValueError("Raw label totals changed since the completed audit")
                row["rare_split_evidence"]=added
            detail=row.get("rare_split_evidence")
            row["small_partition"]=any(n<100 for n in detail["ncd_capped"].values()) if detail else None
        value["raw_label_counts"]["small_partition_scope"]="Cleaned/capped train or test count <100, evaluated for observed raw totals <=500; null for other labels, not inferred false."
        (args.output_dir/(spec["slug"]+"_audit.json")).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding="utf-8")
        write_raw_label_csv(args.output_dir,value)
    result.setdefault("full_audit_script_sha256",previous_hash)
    result["audit_script_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result.setdefault("incremental_updates",[]).append(dict(scope="Raw labels with totals <=500: missing exact small-partition split evidence only; full duplicate audit and Exp H S1 measurements reused unchanged.",
                                                             previous_script_sha256=previous_hash,
                                                             audit_script_sha256=result["audit_script_sha256"],
                                                             command=subprocess.list2cmdline(sys.argv)))
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print("SMALL_PARTITION_REFRESH_COMPLETE",flush=True)


def file_sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def refresh_receipts(args):
    """Bind completed statistics to every actual audited cache shard without recounting."""
    path=args.output_dir/"cic_audit.json"
    result=json.loads(path.read_text(encoding="utf-8"))
    previous_hash=result["audit_script_sha256"]
    receipts=dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  algorithm="SHA256 of complete file bytes; 8 MiB streaming reads, single process, no GPU",datasets=[])
    for value in result["datasets"]:
        dataset=value["dataset"]
        spec=CACHE_DATASETS[dataset]
        root=(args.cache_root/spec["slug"]).resolve()
        manifest_path=root/"manifest.json"
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        declared={item["relative_path"]:item for item in manifest["shards"]}
        if len(declared)!=len(manifest["shards"]):
            raise ValueError("Duplicate shard paths in manifest")
        actual=sorted(p.relative_to(root).as_posix() for split in ("train","test")
                      for name in spec["class_order"] for p in (root/split/name).glob("part-*.npy"))
        if set(actual)!=set(declared):
            raise ValueError("Actual audited shard path set differs from frozen manifest")
        shards=[]
        for index,relative in enumerate(actual):
            source=(root/relative).resolve()
            if not source.is_relative_to(root):
                raise ValueError("Cache shard escapes source root")
            expected=declared[relative]
            actual_hash=file_sha256(source)
            shards.append(dict(relative_path=relative,sha256=actual_hash,expected_sha256=expected.get("sha256"),
                               matches_expected=actual_hash==expected.get("sha256"),size_bytes=source.stat().st_size,
                               expected_size_bytes=expected.get("size_bytes"),rows=expected["rows"],
                               split=expected["split"],class_name=expected["class"]))
            if index%64==0:
                print(json.dumps(dict(stage="cache_sha256",dataset=dataset,shards_done=index+1,total=len(actual))),flush=True)
        sidecars=[]
        for relative,expected in sorted(manifest["sidecars"].items()):
            source=(root/relative).resolve()
            if not source.is_relative_to(root):
                raise ValueError("Cache sidecar escapes source root")
            found=file_sha256(source)
            sidecars.append(dict(relative_path=relative,sha256=found,expected_sha256=expected,
                                 matches_expected=found==expected,size_bytes=source.stat().st_size))
        schema=next(item for item in sidecars if item["relative_path"]=="feature_schema.json")
        receipt=dict(dataset=dataset,cache_root=Path(os.path.relpath(root,ROOT)).as_posix(),
                     manifest=dict(relative_path="manifest.json",sha256=file_sha256(manifest_path),size_bytes=manifest_path.stat().st_size,
                                   expected_sha256=None,note="Manifest itself has no external file-byte expected SHA in the inspected cache; this receipt pins its current bytes."),
                     feature_schema_sha256=schema["sha256"],
                     index_file="streaming_manifest.json",sidecars=sidecars,shards=shards,
                     shard_count=len(shards),total_shard_bytes=sum(s["size_bytes"] for s in shards),
                     all_shard_sha256_match=all(s["matches_expected"] for s in shards),
                     all_shard_sizes_match=all(s["size_bytes"]==s["expected_size_bytes"] for s in shards),
                     all_sidecar_sha256_match=all(s["matches_expected"] for s in sidecars),
                     raw_files=[dict(relative_path=item["relative_path"],manifest_declared_sha256=item["sha256"],
                                     present_locally=(args.raw_root/spec["slug"]/item["relative_path"]).is_file(),
                                     raw_sha256_verified_in_this_audit=False) for item in manifest["raw_files"]],
                     raw_hash_limit="Raw CSV labels and selected small-label feature rows were read, but full raw CSV byte hashes were not recomputed. Manifest-declared raw hashes are provenance assertions, not fresh verification.")
        if not all(receipt[k] for k in ("all_shard_sha256_match","all_shard_sizes_match","all_sidecar_sha256_match")):
            raise ValueError(f"Input receipt mismatch for {dataset}")
        receipts["datasets"].append(receipt)
        value["provenance"]["input_receipt_summary"]={k:receipt[k] for k in ("shard_count","total_shard_bytes","all_shard_sha256_match","all_shard_sizes_match","all_sidecar_sha256_match","feature_schema_sha256")}
        value["provenance"]["input_receipt_summary"]["manifest_sha256"]=receipt["manifest"]["sha256"]
        (args.output_dir/(spec["slug"]+"_audit.json")).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding="utf-8")
    receipt_path=args.output_dir/"cic_input_receipts.json"
    receipt_path.write_text(json.dumps(receipts,indent=2,ensure_ascii=False),encoding="utf-8")
    result["input_receipts"]=dict(path=Path(os.path.relpath(receipt_path,ROOT)).as_posix(),sha256=file_sha256(receipt_path))
    result.setdefault("full_audit_script_sha256",previous_hash)
    result["audit_script_sha256"]=file_sha256(Path(__file__))
    result.setdefault("incremental_updates",[]).append(dict(scope="Byte-hash every actually audited cache shard, manifest and sidecar; statistics reused unchanged, no full raw CSV hash verification.",
                                                             previous_script_sha256=previous_hash,audit_script_sha256=result["audit_script_sha256"],
                                                             command=subprocess.list2cmdline(sys.argv)))
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print("CACHE_INPUT_RECEIPTS_COMPLETE",flush=True)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def self_test():
    # Same vector across different labels must count as leakage and conflict.
    tr=np.array([[0.,1.],[0.,1.],[2.,3.],[4.,5.]], np.float32); ty=np.array([0,0,1,1])
    te=np.array([[-0.,1.],[2.,3.],[9.,9.]], np.float32); ey=np.array([1,1,0])
    result=small_view_audit(tr,ty,te,ey,["A","B"])
    a,b,total=result["rows"]
    assert a["train_duplicate_excess"]==1 and b["test_matches_any_train_rows"]==2
    assert total["test_matches_any_train_rows"]==2 and total["conflict_rows"]==3
    assert result["decision"]=="去重重跑"
    na=np.array([[np.nan,-0.],[np.nan,0.]],np.float32)
    assert canonical_rows(na)[0].tobytes()==canonical_rows(na)[1].tobytes()
    assert np.array_equal(feature_hash(canonical_rows(na)), np.repeat(feature_hash(canonical_rows(na))[0],2))
    print("EXACT_EQUALITY_CROSSLABEL_LEAKAGE_CONFLICT_SIGNED_ZERO_NAN_TESTS_OK", flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cache-root",type=Path,required=True)
    p.add_argument("--raw-root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--builder-core",type=Path)
    p.add_argument("--chunk-rows",type=int,default=50000)
    p.add_argument("--buckets",type=int,default=128)
    p.add_argument("--datasets",nargs="+",default=list(CACHE_DATASETS))
    p.add_argument("--skip-secondary",action="store_true")
    p.add_argument("--refresh-small-partitions",action="store_true",help="Reuse saved duplicate results; inspect only missing raw-label split evidence for totals <=500")
    p.add_argument("--refresh-receipts",action="store_true",help="Reuse saved statistics; hash actual cache inputs and compare all manifest shard and sidecar hashes")
    args=p.parse_args()
    if args.refresh_small_partitions and args.refresh_receipts:
        p.error("Choose one incremental refresh at a time")
    args.output_dir=args.output_dir.resolve(); args.output_dir.mkdir(parents=True,exist_ok=True)
    class Tee:
        def __init__(self, terminal, log):
            self.terminal, self.log = terminal, log
        def write(self, text):
            self.terminal.write(text); self.log.write(text); self.log.flush()
            return len(text)
        def flush(self):
            self.terminal.flush(); self.log.flush()
    log_path=args.output_dir/"audit.log"
    log=log_path.open("a",encoding="utf-8")
    sys.stdout,sys.stderr=Tee(sys.stdout,log),Tee(sys.stderr,log)
    if os.name=="nt":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x00004000)
    self_test()
    if args.refresh_small_partitions:
        refresh_small_partitions(args)
        return
    if args.refresh_receipts:
        refresh_receipts(args)
        return
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    portable=lambda path: Path(os.path.relpath(path.resolve(), ROOT)).as_posix()
    result=dict(git_commit=commit, audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), seed=42, log_path=portable(log_path), command=subprocess.list2cmdline(sys.argv), configuration={k:portable(v) if isinstance(v,Path) else v for k,v in vars(args).items()}, datasets=[], limitations=["Audit equality is exact in cleaned cache float32 space, not original textual CSV or pre-conversion float64 space.","Global conflicts use family labels used by NCD; raw attack subtype conflicts inside one family are outside this label definition."])
    for dataset in args.datasets:
        value=audit_cache(dataset,args.cache_root,args.output_dir,args.chunk_rows,args.buckets,args.builder_core)
        value["raw_label_counts"]=raw_labels(dataset,args.cache_root,args.raw_root,args.chunk_rows)
        result["datasets"].append(value)
        (args.output_dir/(CACHE_DATASETS[dataset]["slug"]+"_audit.json")).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding="utf-8")
        (args.output_dir/"cic_audit.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
        rows=[dict(dataset=dataset,view=view,**row) for view,stats in value["views"].items() for row in stats["rows"]]
        write_csv(args.output_dir/(CACHE_DATASETS[dataset]["slug"]+"_audit.csv"),rows)
        write_raw_label_csv(args.output_dir,value)
        print(json.dumps(dict(stage="dataset_complete",dataset=dataset,decision=value["views"]["ncd_capped"]["decision"],seconds=value["elapsed_seconds"]),ensure_ascii=False),flush=True)
    if not args.skip_secondary:
        result["exp_h_s1"]=audit_s1(args.cache_root)
        print("EXP_H_S1_EXACT_AUDIT_COMPLETE",flush=True)
    (args.output_dir/"cic_audit.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    refresh_receipts(args)
    print("CIC_AUDIT_COMPLETE",flush=True)


if __name__=="__main__":
    main()
