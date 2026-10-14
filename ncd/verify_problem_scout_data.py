"""Independent Section21 UNSW verification using pandas full-key grouping.

Does not import the preparation grouping/hashing/cleaning helpers. Never writes
datasets, changes the split, or removes model-representation collisions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(raw_root, cache, receipt):
    source = [raw_root / f"UNSW_NB15_{part}-set.csv" for part in ("training", "testing")]
    frame = pd.concat([pd.read_csv(p) for p in source], ignore_index=True)
    frame["attack_cat"] = frame["attack_cat"].astype(str).str.strip()
    features = [c for c in frame.columns if c not in ("id", "label", "attack_cat")]
    assert len(features) == 42
    group = frame.groupby(features, sort=False, dropna=False, observed=True)["attack_cat"]
    conflict = group.transform("nunique") > 1
    surviving = frame.loc[~conflict].drop_duplicates(features, keep="first")
    actual = {"raw_rows": len(frame), "raw_feature_groups": group.ngroups,
              "conflict_groups_removed": int((group.nunique() > 1).sum()),
              "conflict_member_rows_removed": int(conflict.sum()),
              "same_label_excess_rows_removed": int((~conflict).sum() - len(surviving)),
              "retained_rows": len(surviving)}
    expected = receipt["quality"]["raw"]
    for name, value in actual.items():
        if value != expected[name]:
            raise AssertionError((name, value, expected[name]))
    checked = []
    for r in receipt["cache_files"]:
        path = cache / r["path"]
        assert path.stat().st_size == r["bytes"] and sha(path) == r["sha256"]
        a = np.load(path, mmap_mode="r", allow_pickle=False)
        assert list(a.shape) == r["shape"] and str(a.dtype) == r["dtype"]
        assert np.isfinite(a).all()
        checked.append(r["path"])
    train = np.load(cache / "train_x.npy", allow_pickle=False)
    test = np.load(cache / "test_x.npy", allow_pickle=False)
    train[train == 0] = 0
    test[test == 0] = 0
    # Full row byte keys; no hash-only match inference.
    keys = set(map(bytes, train))
    matches = np.array([bytes(row) in keys for row in test])
    assert int(matches.sum()) == receipt["quality"]["transformed"]["test_matching_any_train_rows"]
    yt = np.load(cache / "train_y.npy", allow_pickle=False)
    ye = np.load(cache / "test_y.npy", allow_pickle=False)
    pooled = np.bincount(np.concatenate([yt, ye]), minlength=len(receipt["class_order"]))
    by_class = surviving["attack_cat"].value_counts()
    assert all(int(pooled[i]) == int(by_class.get(name, 0)) for i, name in enumerate(receipt["class_order"]))
    return {"status": "verified_with_disclosed_model_collisions", "independent_raw_counts": actual,
            "raw_grouping_method": "pandas groupby on all 42 full feature columns, dropna=False; nunique labels; drop_duplicates",
            "prepared_arrays_verified": checked, "train_rows": len(yt), "test_rows": len(ye),
            "full_byte_model_test_matches_train": int(matches.sum()),
            "model_collision_test_labels": {receipt["class_order"][int(c)]: int(n) for c, n in zip(*np.unique(ye[matches], return_counts=True))},
            "caveat": "Exact raw groups removed before split; numeric preprocessing/float32 can merge distinct raw observations. No test rows removed after observing this."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-path", required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    result = verify(args.raw_root, args.cache, receipt)
    result.update(verification_script_sha256=sha(Path(__file__)), input_receipt_sha256=sha(args.receipt),
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  seed=None, seed_note="Deterministic independent verification; no split or sampling is performed. Preparation split seed is 42.",
                  config={"raw_feature_count": 42, "full_key_grouping": True, "label_conflict_policy": "delete entire group", "change_prepared_arrays": False},
                  log_path=args.log_path,
                  command=subprocess.list2cmdline(["python", "-I", "ncd/verify_problem_scout_data.py", *sys.argv[1:]]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
