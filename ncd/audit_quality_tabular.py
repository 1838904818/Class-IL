"""Section 18 exact, read-only raw-feature audit for NSL-KDD and UNSW-NB15.

Inputs use the actual ncd.data._load_raw path.  No scaler, approximate distance,
rounding, resampling, or data changes are involved.  Pandas MultiIndex factorize
uses exact tuple equality to resolve hash collisions; a hash by itself is never
accepted as proof of equality.  Only numeric signed zero is canonicalized.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from ncd.data import NSL_ATTACK_FAMILY, NSL_COLUMNS, _load_raw, make_tasks
from ncd.source_commit import source_commit


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def exact_feature_groups(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Return exact equality groups without relying on unverified row hashes."""
    if list(train.columns) != list(test.columns):
        raise ValueError("train/test feature columns differ")
    if train.dtypes.astype(str).to_dict() != test.dtypes.astype(str).to_dict():
        raise ValueError("train/test dtype mismatch requires explicit exact coercion review")
    combined = pd.concat([train, test], ignore_index=True)
    if combined.isna().any().any():
        raise ValueError("raw missing values need an explicit equality definition")
    for column in combined.select_dtypes(include=["floating"]).columns:
        values = combined[column].to_numpy(copy=True)
        values[values == 0] = 0.0
        combined[column] = values
    codes, unique = pd.MultiIndex.from_frame(combined).factorize(sort=False)
    if np.any(codes < 0):
        raise ValueError("unexpected null group")
    return codes.astype(np.int64), len(unique)


def conflict_flags(groups: np.ndarray, labels: np.ndarray, group_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Identify groups with >=2 distinct exact labels and all their member rows."""
    unique_pairs = pd.DataFrame({"group": groups, "label": labels}).drop_duplicates()
    counts = np.bincount(unique_pairs["group"].to_numpy(dtype=np.int64), minlength=group_count)
    conflicting = counts > 1
    return conflicting[groups], conflicting


def _fraction(count: int, size: int) -> float | None:
    return count / size if size else None


def _relative(path: Path | str) -> str:
    return os.path.relpath(Path(path).resolve(), ROOT).replace("\\", "/")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def _raw_labels(dataset: str, root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict]:
    if dataset == "NSL-KDD":
        train = pd.read_csv(root / "nsl-kdd/KDDTrain+.txt", header=None, names=NSL_COLUMNS)
        test = pd.read_csv(root / "nsl-kdd/KDDTest+.txt", header=None, names=NSL_COLUMNS)
        source_train = train["label"].astype(str).str.strip().to_numpy()
        source_test = test["label"].astype(str).str.strip().to_numpy()
        return source_train, source_test, test["difficulty"].to_numpy() < 21, {
            label: NSL_ATTACK_FAMILY.get(label, "R2L")
            for label in sorted(set(source_train) | set(source_test))
        }
    train = pd.read_csv(root / "unsw-nb15/UNSW_NB15_training-set.csv")
    test = pd.read_csv(root / "unsw-nb15/UNSW_NB15_testing-set.csv")
    source_train = train["attack_cat"].astype(str).str.strip().to_numpy()
    source_test = test["attack_cat"].astype(str).str.strip().to_numpy()
    return source_train, source_test, None, {label: label for label in sorted(set(source_train) | set(source_test))}


def audit_dataset(dataset: str, data_root: Path) -> dict:
    train, y_train, test, y_test, class_order, categorical, sources = _load_raw(dataset, data_root)
    source_train, source_test, difficult, mapping = _raw_labels(dataset, data_root)
    if len(source_train) != len(train) or len(source_test) != len(test):
        raise ValueError("actual raw loader applied an unaccounted row selection")
    families_train = np.asarray(class_order, dtype=str)[y_train]
    families_test = np.asarray(class_order, dtype=str)[y_test]
    if not np.array_equal(families_train, np.asarray([mapping[label] for label in source_train])):
        raise ValueError("source-label mapping differs from actual loader labels")
    if not np.array_equal(families_test, np.asarray([mapping[label] for label in source_test])):
        raise ValueError("source-label mapping differs from actual loader labels")
    groups, group_count = exact_feature_groups(train, test)
    n_train, n_test = len(train), len(test)
    train_groups, test_groups = groups[:n_train], groups[n_train:]
    train_counts = np.bincount(train_groups, minlength=group_count)
    test_counts = np.bincount(test_groups, minlength=group_count)
    train_excess = pd.Series(train_groups).duplicated(keep="first").to_numpy()
    test_excess = pd.Series(test_groups).duplicated(keep="first").to_numpy()
    train_duplicate_members = train_counts[train_groups] > 1
    test_overlap = train_counts[test_groups] > 0
    conflicts: dict[str, dict] = {}
    for label_level, train_labels, test_labels in (
        ("model_family", families_train, families_test),
        ("source_label", source_train, source_test),
    ):
        scopes = {}
        for scope, scope_groups, scope_labels in (
            ("train", train_groups, train_labels),
            ("test", test_groups, test_labels),
            ("combined", groups, np.concatenate([train_labels, test_labels])),
        ):
            flags, group_flags = conflict_flags(scope_groups, scope_labels, group_count)
            scopes[scope] = {"affected_rows": flags, "conflicting_groups": group_flags}
        conflicts[label_level] = scopes

    tasks = make_tasks(len(class_order), 1 if dataset == "NSL-KDD" else 2)
    class_task = {int(c): i for i, task in enumerate(tasks) for c in task}
    family_rows = []
    for class_id, name in enumerate(class_order):
        tr, te = y_train == class_id, y_test == class_id
        combined_mask = np.concatenate([tr, te])
        row = {
            "dataset": dataset, "class_id": class_id, "class_name": name, "task_index": class_task[class_id],
            "train_before_cap": int(tr.sum()), "test_before_cap": int(te.sum()),
            "train_after_cap": int(tr.sum()), "test_after_cap": int(te.sum()), "cap": "none",
            "train_excess_duplicate_rows": int(np.sum(train_excess & tr)),
            "train_excess_duplicate_rate": _fraction(int(np.sum(train_excess & tr)), int(tr.sum())),
            "train_within_class_excess_duplicate_rows": int(pd.Series(train_groups[tr]).duplicated().sum()),
            "train_duplicate_affected_rows": int(np.sum(train_duplicate_members & tr)),
            "test_excess_duplicate_rows": int(np.sum(test_excess & te)),
            "test_matching_any_train_rows": int(np.sum(test_overlap & te)),
            "test_matching_any_train_rate": _fraction(int(np.sum(test_overlap & te)), int(te.sum())),
            "under_100_train": bool(tr.sum() < 100), "under_100_test": bool(te.sum() < 100),
            "handling": "retained as this family in its existing task; no oversampling",
        }
        row["train_within_class_excess_duplicate_rate"] = _fraction(
            row["train_within_class_excess_duplicate_rows"], int(tr.sum())
        )
        for label_level, scopes in conflicts.items():
            for scope, mask, scope_groups in (
                ("train", tr, train_groups), ("test", te, test_groups), ("combined", combined_mask, groups),
            ):
                flagged = scopes[scope]["affected_rows"] & mask
                row[f"{label_level}_{scope}_conflict_rows"] = int(flagged.sum())
                row[f"{label_level}_{scope}_conflict_groups"] = int(len(np.unique(scope_groups[flagged])))
        family_rows.append(row)

    rare_rows, all_label_rows = [], []
    for label in sorted(mapping):
        train_n, test_n = int(np.sum(source_train == label)), int(np.sum(source_test == label))
        family = mapping[label]
        class_id = class_order.index(family)
        row = {
            "dataset": dataset, "source_label": label, "model_family": family, "class_id": class_id,
            "task_index": class_task[class_id], "train_rows": train_n, "test_rows": test_n,
            "train_rows_after_cap": train_n, "test_rows_after_cap": test_n,
            "under_100_train": train_n < 100, "under_100_test": test_n < 100,
            "handling": ("merged into attack family; all source rows retained" if dataset == "NSL-KDD" and label != "normal" else "retained as family; all source rows retained"),
            "reason": "existing ncd.data label mapping and fixed task order; audit makes no changes",
        }
        if difficult is not None:
            row["KDDTest21_rows"] = int(np.sum((source_test == label) & difficult))
        all_label_rows.append(row)
        if train_n < 100 or test_n < 100:
            rare_rows.append(row)

    total_overlap = int(test_overlap.sum())
    overlap_rate = total_overlap / n_test
    reasons = []
    if overlap_rate >= 0.05:
        reasons.append(f"overall test/train exact overlap {overlap_rate:.8%} >= 5%")
    for row in family_rows:
        if row["test_matching_any_train_rate"] is not None and row["test_matching_any_train_rate"] >= 0.2:
            reasons.append(f"{row['class_name']} exact overlap {row['test_matching_any_train_rate']:.8%} >= 20%")
    totals = {
        "train_rows": n_train, "test_rows": n_test, "train_before_cap": n_train, "train_after_cap": n_train,
        "test_before_cap": n_test, "test_after_cap": n_test, "caps_applied": False,
        "feature_count": len(train.columns), "combined_exact_feature_groups": group_count,
        "train_excess_duplicate_rows": int(train_excess.sum()),
        "train_excess_duplicate_rate": float(train_excess.mean()),
        "train_duplicate_affected_rows": int(train_duplicate_members.sum()),
        "test_excess_duplicate_rows": int(test_excess.sum()),
        "test_matching_any_train_rows": total_overlap, "test_matching_any_train_rate": overlap_rate,
        "max_class_test_matching_any_train_rate": max(row["test_matching_any_train_rate"] or 0 for row in family_rows),
        "conflicts": {
            level: {
                scope: {"affected_rows": int(items["affected_rows"].sum()), "groups": int(items["conflicting_groups"].sum())}
                for scope, items in scopes.items()
            } for level, scopes in conflicts.items()
        },
    }
    result = {
        "dataset": dataset, "totals": totals, "family_rows": family_rows, "rare_source_labels": rare_rows,
        "all_source_label_counts": all_label_rows, "class_order": class_order, "tasks": tasks,
        "feature_columns": list(train.columns), "categorical_columns": categorical,
        "excluded_nonfeatures": ["label", "difficulty"] if dataset == "NSL-KDD" else ["id", "label", "attack_cat"],
        "source_files": [_relative(path) for path in sources], "source_label_mapping": mapping,
        "decision": "去重重跑" if reasons else "保持", "decision_reasons": reasons or ["overall exact overlap < 5% and every class < 20%"],
        "deduplication_performed": False,
    }
    if difficult is not None:
        subset_rows = []
        for class_id, name in enumerate(class_order):
            mask = difficult & (y_test == class_id)
            matched = int(np.sum(mask & test_overlap))
            subset_rows.append({"dataset": "NSL-KDD KDDTest-21", "class_id": class_id, "class_name": name,
                                "test_rows": int(mask.sum()), "test_matching_any_train_rows": matched,
                                "test_matching_any_train_rate": _fraction(matched, int(mask.sum()))})
        result["KDDTest21"] = {
            "selection": "KDDTest+.txt difficulty < 21", "expected_rows": 11850,
            "actual_rows": int(difficult.sum()), "official_definition": "https://www.unb.ca/cic/datasets/nsl.html",
            "test_matching_any_train_rows": int(np.sum(test_overlap & difficult)),
            "test_matching_any_train_rate": float(test_overlap[difficult].mean()), "family_rows": subset_rows,
            "scope": "hard subset of the same NSL test file, not an independent external distribution",
        }
        if difficult.sum() != 11850:
            raise ValueError("KDDTest21 difficulty subset count does not match the official subset size")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "kaggle_leon_release/datasets")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "ncd/results/data_quality/section18/tabular")
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "audit.log"
    commit = source_commit(ROOT)
    command = subprocess.list2cmdline([
        "python", "-I", "ncd/audit_quality_tabular.py", "--data-root", _relative(args.data_root),
        "--output-dir", _relative(args.output_dir),
    ])
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                started = time.perf_counter()
                results = []
                for name in ("NSL-KDD", "UNSW-NB15"):
                    result = audit_dataset(name, args.data_root)
                    results.append(result)
                    print(json.dumps({"dataset": name, "totals": result["totals"], "decision": result["decision"]}), flush=True)
                payload = {
                    "git_commit": commit, "command": command, "log_path": _relative(log_path),
                    "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
                    "effective_config": {"data_root": _relative(args.data_root), "datasets": ["NSL-KDD", "UNSW-NB15"], "sampling": "none", "equality": "exact raw model feature tuples; signed zeros normalized only"},
                    "definition": {
                        "train_excess_duplicate_rows": "all train rows except the first member of each global raw-feature equality group; labels excluded",
                        "train_within_class_excess_duplicate_rows": "excess rows within the stated class, also excluding labels from features",
                        "test_matching_any_train_rows": "test row whose entire raw feature vector equals any train row, regardless of labels",
                        "conflict_rows": "all rows belonging to a feature-equality group with at least two distinct labels in the explicitly named scope",
                        "conflict_groups": "distinct feature-equality groups with conflicting labels; per-class group counts need not add because a group can contain several classes",
                        "caps": "actual NSL/UNSW raw loader applies no per-class cap; before and after counts are equal",
                        "rare_labels": "train < 100 or test < 100, including labels absent from one split",
                        "decision": "去重重跑 if overall test overlap >= 5% or any model-family overlap >= 20%; otherwise 保持; this audit does not rerun or change data",
                    },
                    "datasets": results, "elapsed_seconds": time.perf_counter() - started,
                }
                (args.output_dir / "audit_tabular.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
                _write_csv(args.output_dir / "audit_family.csv", [row for result in results for row in result["family_rows"]])
                _write_csv(args.output_dir / "rare_labels.csv", [row for result in results for row in result["rare_source_labels"]])
                _write_csv(args.output_dir / "all_source_labels.csv", [row for result in results for row in result["all_source_label_counts"]])
                _write_csv(args.output_dir / "kddtest21_family.csv", results[0]["KDDTest21"]["family_rows"])
                print(f"AUDIT_COMPLETE elapsed_seconds={payload['elapsed_seconds']:.3f} output={_relative(args.output_dir)}", flush=True)
            except Exception:
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
