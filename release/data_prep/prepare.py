"""Rebuild and verify the Section 21 UNSW-NB15 arrays from the official CSVs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "ncd/results/problem_scout/section21/unsw-nb15_data_quality.json"
DATASET = "UNSW-NB15"
ARRAYS = {"train_x.npy", "train_y.npy", "test_x.npy", "test_y.npy"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True,
                        help="Directory containing unsw-nb15/ and its official CSVs")
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory for arrays and their data-quality result")
    args = parser.parse_args()
    raw_root = args.raw_root.resolve(strict=True)
    out = args.output.resolve()
    if out.exists() or raw_root == out or raw_root in out.parents:
        parser.error("--output must be new and outside --raw-root")

    sys.path.insert(0, str(ROOT))
    from ncd.prepare_problem_scout_data import prepare_tabular, sha256
    from ncd.source_commit import source_commit
    import numpy
    import pandas
    import sklearn

    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))
    for entry in expected["input_files"]:
        path = raw_root / "unsw-nb15" / Path(entry["path"]).name
        actual = sha256(path)
        if actual != entry["sha256"]:
            raise ValueError(f"Raw input {path.name}: expected {entry['sha256']}, got {actual}")
        print(f"Input {path.name}: {actual}", flush=True)
    out.mkdir(parents=True, exist_ok=False)
    result = prepare_tabular(DATASET, raw_root, out, {
        "git_commit": source_commit(ROOT),
        "seed": 42,
        "entrypoint_sha256": sha256(__file__),
        "preparation_script_sha256": sha256(ROOT / "ncd/prepare_problem_scout_data.py"),
        "reference_sha256": sha256(REFERENCE),
        "argv": sys.argv,
        "runtime": {"python": sys.version.split()[0], "numpy": numpy.__version__,
                    "pandas": pandas.__version__, "scikit_learn": sklearn.__version__},
    })
    reference_arrays = {entry["path"]: entry for entry in expected["cache_files"]}
    current_arrays = {entry["path"]: entry for entry in result["cache_files"]}
    if set(reference_arrays) != ARRAYS or set(current_arrays) != ARRAYS:
        raise ValueError("Unexpected array inventory")
    differences = []
    for name in sorted(ARRAYS):
        actual = current_arrays[name]
        print(f"Array {name}: {actual['sha256']}", flush=True)
        for field in ("bytes", "shape", "dtype", "sha256"):
            if actual[field] != reference_arrays[name][field]:
                differences.append(
                    f"{name} {field}: expected {reference_arrays[name][field]}, got {actual[field]}"
                )
    for field in ("class_order", "tasks", "feature_names", "train_counts", "id_counts"):
        if result[field] != expected[field]:
            differences.append(f"{field} differs from the Section 21 reference")
    if differences:
        raise ValueError("Section 21 mismatch:\n" + "\n".join(differences))
    print("UNSW-NB15: all four arrays match Section 21 byte for byte.", flush=True)


if __name__ == "__main__":
    main()
