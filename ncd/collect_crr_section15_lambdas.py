"""Collect fail-closed per-method Section 15 lambda selections."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ncd.crr_benchmarks import METHODS


def collect(paths: list[Path], dataset: str, commit: str) -> dict:
    values = {"replay": 0.0}
    inputs = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        method = str(payload["method"])
        if method in values or method not in METHODS:
            raise ValueError(f"invalid or duplicate lambda method: {method}")
        values[method] = float(payload["selected_lambda"])
        inputs.append(str(path))
    if set(values) != set(METHODS):
        raise ValueError(f"lambda set mismatch: {sorted(set(METHODS) - set(values))}")
    return {
        "schema_version": 1,
        "dataset": dataset,
        "git_commit": commit,
        "selection_seeds": [5, 6],
        "lambdas": values,
        "inputs": inputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--dataset", required=True, choices=("NSL-KDD", "UNSW-NB15"))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = collect(args.inputs, args.dataset, args.source_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["lambdas"], sort_keys=True))


if __name__ == "__main__":
    main()
