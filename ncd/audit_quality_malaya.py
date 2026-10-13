"""Read-only Section 18 Malaya exact-feature duplicate/overlap audit.

This audits the frozen original capture split and the actual Exp H fit/ID/shift
views separately. It neither edits source data nor changes/reruns experiments.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import csv
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ncd.exp_h_malaya import CLASS_ORDER, _grouped_holdout, _load_split, prepare_malaya


def relative(path: Path) -> str:
    return os.path.relpath(path.resolve(), ROOT).replace("\\", "/")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def keys(values: np.ndarray) -> np.ndarray:
    canonical = np.ascontiguousarray(values, dtype="<f4").copy()
    canonical[canonical == 0] = 0.0
    # NaN payloads are not scientific values; use one canonical quiet NaN.
    # Actual cached/prepared data must be finite, so this does not affect them.
    canonical[np.isnan(canonical)] = np.float32(np.nan)
    return canonical.view(np.dtype((np.void, canonical.shape[1] * 4))).ravel()


def stats_for_view(name: str, train: np.ndarray, y_train: np.ndarray,
                   test: np.ndarray, y_test: np.ndarray) -> dict:
    tk, ek = keys(train), keys(test)
    unique_train, train_inverse, train_counts = np.unique(tk, return_inverse=True, return_counts=True)
    test_matches = np.isin(ek, unique_train)
    combined_labels = np.concatenate([y_train, y_test])
    _, union_inverse, union_counts = np.unique(np.concatenate([tk, ek]), return_inverse=True, return_counts=True)
    label_presence = np.zeros((len(union_counts), len(CLASS_ORDER)), dtype=bool)
    label_presence[union_inverse, combined_labels] = True
    conflict_groups = label_presence.sum(axis=1) > 1
    conflict_rows = conflict_groups[union_inverse]
    train_label_presence = np.zeros((len(unique_train), len(CLASS_ORDER)), dtype=bool)
    train_label_presence[train_inverse, y_train] = True
    train_conflict_groups = train_label_presence.sum(axis=1) > 1
    rows = []
    for class_id, class_name in enumerate(CLASS_ORDER):
        tm, em = y_train == class_id, y_test == class_id
        n_train, n_test = int(tm.sum()), int(em.sum())
        excess = n_train - int(len(np.unique(tk[tm])))
        matched = int(test_matches[em].sum())
        rows.append({
            "view": name, "class_id": class_id, "class_name": class_name,
            "train_rows": n_train, "test_rows": n_test,
            "train_duplicate_excess_rows": excess,
            "train_duplicate_excess_rate": excess / n_train if n_train else None,
            "train_duplicate_affected_rows_any_label": int(np.sum(train_counts[train_inverse[tm]] > 1)),
            "test_rows_matching_any_train": matched,
            "test_overlap_rate": matched / n_test if n_test else None,
            "train_label_conflict_affected_rows": int(train_conflict_groups[train_inverse[tm]].sum()),
            "train_label_conflict_groups": int(np.sum(train_conflict_groups & train_label_presence[:, class_id])),
            "union_label_conflict_train_rows": int(conflict_rows[:len(train)][tm].sum()),
            "union_label_conflict_test_rows": int(conflict_rows[len(train):][em].sum()),
            "union_label_conflict_groups": int(np.sum(conflict_groups & label_presence[:, class_id])),
        })
    total_matches = int(test_matches.sum())
    total_rate = total_matches / len(test) if len(test) else None
    triggering_classes = [row["class_name"] for row in rows if row["test_overlap_rate"] is not None and row["test_overlap_rate"] >= .2]
    decision = "去重重跑" if (total_rate is not None and total_rate >= .05) or triggering_classes else "保持"
    total = {
        "train_rows": len(train), "test_rows": len(test),
        "train_unique_feature_rows": len(unique_train),
        "train_duplicate_excess_rows": len(train) - len(unique_train),
        "train_duplicate_excess_rate": 1-len(unique_train)/len(train),
        "train_duplicate_affected_rows_any_label": int(np.sum(train_counts[train_inverse] > 1)),
        "test_rows_matching_any_train": total_matches, "test_overlap_rate": total_rate,
        "train_label_conflict_groups": int(train_conflict_groups.sum()),
        "train_label_conflict_affected_rows": int(train_conflict_groups[train_inverse].sum()),
        "union_label_conflict_groups": int(conflict_groups.sum()),
        "union_label_conflict_affected_rows": int(conflict_rows.sum()),
        "union_label_conflict_train_rows": int(conflict_rows[:len(train)].sum()),
        "union_label_conflict_test_rows": int(conflict_rows[len(train):].sum()),
    }
    return {"view": name, "rows": rows, "total": total,
            "decision_18_2": decision, "triggering_classes_ge_20_percent": triggering_classes,
            "finite_train": bool(np.isfinite(train).all()), "finite_test": bool(np.isfinite(test).all()),
            "feature_count": train.shape[1]}


def raw_source_counts(manifest: dict, features: list[str]) -> tuple[dict, list[dict]]:
    source = Path(manifest["source_directory"])
    per_class = {name: {"train": 0, "test": 0, "total": 0} for name in CLASS_ORDER}
    verified = []
    for entry in manifest["raw_files"]:
        path = source / entry["relative_path"]
        sha = digest(path)
        if sha != entry["sha256"]:
            raise ValueError(f"Raw CSV content differs from frozen cache: {entry['relative_path']}")
        frame = pd.read_csv(path, usecols=features)
        name, split = entry["source_class"], entry["capture_split"]
        count = len(frame)
        if count != int(entry["quality_counts"]["raw_rows"]):
            raise ValueError("Raw CSV count differs from cache provenance")
        per_class[name][split] += count
        per_class[name]["total"] += count
        verified.append({"path": relative(path), "sha256": sha, "rows": count,
                         "class_name": name, "frozen_split": split})
    return per_class, verified


def audit(config: dict) -> dict:
    begin = time.perf_counter()
    cache = Path(config["malaya_cache_root"])
    if not cache.is_absolute():
        cache = ROOT / cache
    cache = cache.resolve()
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    schema = json.loads((cache / "feature_schema.json").read_text(encoding="utf-8"))
    shard_sources = []
    for shard in manifest["shards"]:
        path = cache / shard["relative_path"]
        sha = digest(path)
        if sha != shard["sha256"]:
            raise ValueError(f"Frozen Malaya shard changed: {shard['relative_path']}")
        shard_sources.append({"path": relative(path), "sha256": sha, "rows": shard["rows"]})
    print(f"SHARDS_VERIFIED {len(shard_sources)}", flush=True)
    raw_counts, csv_sources = raw_source_counts(manifest, schema["feature_columns"])
    print(f"RAW_FILES_VERIFIED {len(csv_sources)}", flush=True)
    original_train, original_y = _load_split(cache, manifest, "train")
    original_test, original_test_y = _load_split(cache, manifest, "test")
    train_index, id_index, holdout = _grouped_holdout(original_train, original_y)
    source_overlap = np.isin(keys(original_test), np.unique(keys(original_train)))
    actual_train, actual_y = original_train[train_index], original_y[train_index]
    actual_id, actual_id_y = original_train[id_index], original_y[id_index]
    actual_shift, actual_shift_y = original_test[~source_overlap], original_test_y[~source_overlap]
    # Calling the production data-preparation entry performs only CPU array
    # transforms. Compare its row labels and raw-view counts before auditing
    # the actual float32 matrices that the ongoing GPU run receives.
    prepared, transformed_shift, transformed_shift_y, metadata = prepare_malaya(config)
    if not np.array_equal(prepared.y_train, actual_y) or not np.array_equal(prepared.y_test, actual_id_y) or not np.array_equal(transformed_shift_y, actual_shift_y):
        raise ValueError("Reconstructed raw indices do not match actual Exp H loader")
    specifications = [
        ("original_cache_raw_train_vs_test", original_train, original_y, original_test, original_test_y),
        ("exp_h_raw_fit_vs_id", actual_train, actual_y, actual_id, actual_id_y),
        ("exp_h_raw_fit_vs_shift", actual_train, actual_y, actual_shift, actual_shift_y),
        ("exp_h_transformed_fit_vs_id", prepared.x_train, prepared.y_train, prepared.x_test, prepared.y_test),
        ("exp_h_transformed_fit_vs_shift", prepared.x_train, prepared.y_train, transformed_shift, transformed_shift_y),
    ]
    views = {}
    for arguments in specifications:
        print(f"AUDIT_VIEW {arguments[0]}", flush=True)
        views[arguments[0]] = stats_for_view(*arguments)
        print(json.dumps({"view": arguments[0], "total": views[arguments[0]]["total"], "decision": views[arguments[0]]["decision_18_2"]}, ensure_ascii=False), flush=True)
    counts = []
    for class_id, name in enumerate(CLASS_ORDER):
        row = next(item for item in metadata["per_class_counts"] if item["class_id"] == class_id)
        task = next(i for i, members in enumerate(prepared.tasks) if class_id in members)
        counts.append({
            "class_id": class_id, "class_name": name, "task_index": task,
            "raw_original_total_rows": raw_counts[name]["total"],
            "raw_original_train_rows": raw_counts[name]["train"],
            "raw_original_test_rows": raw_counts[name]["test"],
            "valid_cache_train_rows": int(np.sum(original_y == class_id)),
            "valid_cache_test_rows": int(np.sum(original_test_y == class_id)),
            "train_before_cap": row["fit_rows"], "train_after_cap": row["fit_rows"],
            "id_before_cap": row["id_rows"], "id_after_cap": row["id_rows"],
            "shift_before_overlap_exclusion": row["shift_original_rows"],
            "shift_overlap_excluded_rows": row["shift_overlap_excluded_rows"],
            "shift_after_overlap_exclusion": row["shift_rows"],
            "shift_before_cap": row["shift_rows"], "shift_after_cap": row["shift_rows"],
            "class_total_lt_100": raw_counts[name]["total"] < 100,
            "small_partitions_lt_100": [key for key, value in {"original_test": row["shift_original_rows"], "exp_h_id": row["id_rows"], "exp_h_shift": row["shift_rows"]}.items() if value < 100],
            "handling": "Retained under its original application label; no merge, class deletion, oversampling or synthetic samples. All caps are absent for Malaya.",
            "train_captures": row["train_captures"], "shift_captures": row["shift_captures"],
        })
    current_views = [detail for name, detail in views.items() if name.startswith("exp_h_")]
    return {
        "dataset": "MalayaNetwork_GT", "work_order": "Section 18", "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "audit_seconds": time.perf_counter()-begin, "config": config,
        "audit_execution": {"device": "cpu", "numeric_thread_limit": 1, "training_performed": False,
                            "note": "The saved Exp H config may name CUDA; this audit uses only CPU data preparation and exact array comparisons."},
        "command": "python ncd/audit_quality_malaya.py --config ncd/configs/exp_h_section17.json --output-dir ncd/results/data_quality/section18/malaya",
        "log_path": "ncd/results/data_quality/section18/malaya/run.log",
        "equality_definition": "Exact equality on all 77 little-endian float32 feature values, excluding labels and identifiers. Signed zero is canonicalized; NaN payloads are canonicalized, although all actual audited matrices are finite. No probabilistic row hashes determine equality.",
        "duplicate_definition": "Per-class train excess is class N minus unique all-feature rows within that class, divided by class N. Overall train excess uses all classes together, so per-class excesses need not sum to the overall total. Affected rows separately count every member of repeated feature groups, regardless of label.",
        "overlap_definition": "Numerator: every test row matching any training row on all 77 features regardless of label. Denominator: all rows of the named test view/class. Repeated matching test rows each count.",
        "conflict_definition": "Feature groups containing >=2 distinct labels; affected rows include every member of those groups. Union conflict counts include train and the named test together; training-only conflict counts are also given.",
        "decision_rule": "Test overlap >= 5% overall OR >= 20% in any class means 去重重跑; otherwise 保持. Applied separately to original and actual Exp H views; train duplication alone is not this rule's trigger.",
        "decisions": {
            "original_cache_train_test": views["original_cache_raw_train_vs_test"]["decision_18_2"],
            "current_exp_h_fit_id_shift": "去重重跑" if any(view["decision_18_2"] == "去重重跑" for view in current_views) else "保持",
            "scope": "Original-cache leakage does not imply leakage in Exp H's grouped ID and overlap-filtered shifted tests. No data or experiment was changed by this audit.",
        },
        "views": views, "class_counts_and_handling": counts,
        "tasks": [[CLASS_ORDER[i] for i in group] for group in prepared.tasks],
        "caps": {"applied": False, "reason": "Malaya cache uncapped; prepare_malaya has no train, ID or shift row cap. The NCD Normal 160k/40k and other 40k/10k settings are not applied to this loader."},
        "original_capture_split": {key: value for key, value in manifest["split_protocol"].items()},
        "exp_h_grouped_id_split": holdout,
        "exp_h_shift_exclusion": {"against": "All original training captures (fit plus grouped ID)", "before": len(original_test_y), "excluded": int(source_overlap.sum()), "after": len(actual_shift_y)},
        "problem_scope": "Application classification, not NIDS attack classes. None of the ten original application classes has fewer than 100 total source rows; small evaluation partitions are explicitly listed.",
        "source_files": [{"path": relative(cache / name), "sha256": digest(cache / name)} for name in ("manifest.json", "feature_schema.json", "source_capture_split.json")],
        "raw_csv_sources_verified": csv_sources, "cache_shards_verified": shard_sources,
        "audited_source_sha256": {"ncd/exp_h_malaya.py": digest(ROOT / "ncd/exp_h_malaya.py"), "ncd/audit_quality_malaya.py": digest(Path(__file__))},
    }


def markdown(result: dict) -> str:
    lines = ["# Malaya Section 18 data quality audit", "",
        f"Original frozen capture split: **{result['decisions']['original_cache_train_test']}**. Current Exp H grouped-ID and cleaned-shift views: **{result['decisions']['current_exp_h_fit_id_shift']}**.", "",
        result["decisions"]["scope"], "", result["problem_scope"], "",
        "## Definitions", "", result["equality_definition"], "", result["duplicate_definition"], "", result["overlap_definition"], "", result["conflict_definition"], "",
        "## Original cache counts and actual Exp H exclusions", "",
        "| Class | Original raw train | Original raw test | Exp H fit | Exp H ID | Shift before | Overlap excluded | Shift after | Task | Partitions below 100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["class_counts_and_handling"]:
        lines.append(f"| {row['class_name']} | {row['raw_original_train_rows']} | {row['raw_original_test_rows']} | {row['train_after_cap']} | {row['id_after_cap']} | {row['shift_before_overlap_exclusion']} | {row['shift_overlap_excluded_rows']} | {row['shift_after_cap']} | {row['task_index']} | {', '.join(row['small_partitions_lt_100']) or 'none'} |")
    lines.extend(["", result["caps"]["reason"], "No classes are merged or deleted; small partitions remain natural, without oversampling or synthetic data.", ""])
    for name, view in result["views"].items():
        total = view["total"]
        lines.extend([f"## {name}", "", f"Decision: **{view['decision_18_2']}**; overlap {total['test_rows_matching_any_train']}/{total['test_rows']} = {total['test_overlap_rate']:.4%}. Training excess duplicates {total['train_duplicate_excess_rows']}/{total['train_rows']} = {total['train_duplicate_excess_rate']:.4%}. Union label conflicts: {total['union_label_conflict_affected_rows']} affected rows in {total['union_label_conflict_groups']} groups.", "",
            "| Class | Train N | Train excess dup N (%) | Test N | Matching test N (%) | Train conflict rows/groups | Union conflict train/test rows | Union conflict groups |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in view["rows"]:
            lines.append(f"| {row['class_name']} | {row['train_rows']} | {row['train_duplicate_excess_rows']} ({row['train_duplicate_excess_rate']:.2%}) | {row['test_rows']} | {row['test_rows_matching_any_train']} ({row['test_overlap_rate']:.2%}) | {row['train_label_conflict_affected_rows']}/{row['train_label_conflict_groups']} | {row['union_label_conflict_train_rows']}/{row['union_label_conflict_test_rows']} | {row['union_label_conflict_groups']} |")
        lines.append("")
    lines.extend(["## Reproduction", "", f"Source commit: `{result['git_commit']}`. CPU-only audit; no training or dataset edits.", "", f"Command: `{result['command']}`", "", f"Log: `{result['log_path']}`", "", "Raw CSV and cache shard hashes were checked against the frozen manifest; the JSON records all source paths and exact counts. Numeric comparison uses full feature bytes, not hashes.", ""])
    return "\n".join(lines)


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, value):
        for stream in self.streams: stream.write(value); stream.flush()
    def flush(self):
        for stream in self.streams: stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "ncd/configs/exp_h_section17.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "ncd/results/data_quality/section18/malaya")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run.log").open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)), threadpool_limits(limits=1):
            result = audit(json.loads(args.config.read_text(encoding="utf-8")))
            result["command"] = subprocess.list2cmdline(["python", "-I", "ncd/audit_quality_malaya.py", "--config", relative(args.config), "--output-dir", relative(args.output_dir)])
            result["log_path"] = relative(args.output_dir / "run.log")
            (args.output_dir / "audit_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
            rows = [row for view in result["views"].values() for row in view["rows"]]
            with (args.output_dir / "audit_rows.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            (args.output_dir / "audit_report.md").write_text(markdown(result), encoding="utf-8")
            print(json.dumps({"decisions": result["decisions"], "elapsed_seconds": result["audit_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
