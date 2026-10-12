"""Merge fail-closed Section 15 per-method or per-seed result shards."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHODS = (
    "replay",
    "icarl",
    "replay_lwf",
    "kd_perturbed",
    "rcl_tab",
    "global_ir",
    "oldpair_kd",
    "crr",
)
SEEDS = (1, 2, 3, 4, 42)


def _write_csv(path: Path, rows: list[dict]) -> None:
    scalar = [
        {
            key: json.dumps(value, sort_keys=True)
            if isinstance(value, (dict, list))
            else value
            for key, value in row.items()
        }
        for row in rows
    ]
    fields = list(dict.fromkeys(key for row in scalar for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scalar)


def merge(paths: list[Path], mode: str, dataset: str, source_commit: str) -> dict:
    rows: list[dict] = []
    pi_rows: list[dict] = []
    inputs: list[dict] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["git_commit"] != source_commit:
            raise ValueError(f"source commit mismatch: {path}")
        rows.extend(payload["rows"])
        pi_rows.extend(payload.get("pi_drift_rows", []))
        inputs.append({"path": str(path), "row_count": len(payload["rows"])})

    if mode == "necessity":
        expected = {
            (dataset, seed, method, k)
            for seed in SEEDS
            for method in METHODS
            for k in (2.0, 3.0)
        }
        observed = {
            (str(row["dataset"]), int(row["seed"]), str(row["method"]), float(row["k"]))
            for row in rows
        }
    else:
        expected = {
            (dataset, seed, method) for seed in SEEDS for method in METHODS
        }
        observed = {
            (str(row["dataset"]), int(row["seed"]), str(row["method"]))
            for row in rows
        }
    if len(observed) != len(rows):
        raise ValueError("duplicate Section 15 result cells")
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"matrix mismatch missing={missing[:1]} extra={extra[:1]}")
    if mode == "necessity":
        replay_cost = {
            (int(row["seed"]), float(row["k"])): float(row["training_seconds"])
            for row in rows
            if row["method"] == "replay"
        }
        for row in rows:
            baseline = replay_cost[(int(row["seed"]), float(row["k"]))]
            row["cost_vs_replay"] = float(row["training_seconds"]) / max(baseline, 1e-12)
    else:
        replay_cost = {
            int(row["seed"]): float(row["training_seconds"])
            for row in rows
            if row["method"] == "replay"
        }
        for row in rows:
            baseline = replay_cost[int(row["seed"])]
            row["cost_vs_replay"] = float(row["training_seconds"]) / max(baseline, 1e-12)
    return {
        "schema_version": 1,
        "mode": mode,
        "dataset": dataset,
        "git_commit": source_commit,
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "rows": sorted(
            rows,
            key=lambda row: (
                int(row["seed"]),
                METHODS.index(str(row["method"])),
                float(row.get("k", 0.0)),
            ),
        ),
        "pi_drift_rows": pi_rows,
        "inputs": inputs,
        "scope": "Section 15 evidence; final innovation verdict requires all registered criteria",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--mode", required=True, choices=("necessity", "real"))
    parser.add_argument("--dataset", required=True, choices=("NSL-KDD", "UNSW-NB15"))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = merge(args.inputs, args.mode, args.dataset, args.source_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(args.output.with_suffix(".csv"), result["rows"])
    if result["pi_drift_rows"]:
        _write_csv(
            args.output.with_name(args.output.stem + "_pi_drift.csv"),
            result["pi_drift_rows"],
        )
    print(json.dumps({"mode": args.mode, "rows": len(result["rows"])}))


if __name__ == "__main__":
    main()
