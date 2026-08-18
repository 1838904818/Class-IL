#!/usr/bin/env python3
"""Aggregate the source-bound CIC-IDS-2018 FT256x4 five-seed closure run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


VIEWS = (
    "head_only",
    "router_only_cap3000",
    "joint_cap3000",
    "router_only_uncapped",
    "joint_uncapped",
)
METRICS = (
    "final_overall_accuracy",
    "final_macro_f1",
    "final_balanced_accuracy",
    "average_forgetting",
    "final_benign_false_positive_rate",
    "final_attack_detection_recall",
)
JOB_IDS = {1: 395350, 2: 399060, 3: 399246, 4: 399313, 42: 399593}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def paired(left: list[float], right: list[float]) -> dict[str, float]:
    differences = [a - b for a, b in zip(left, right)]
    mean = statistics.mean(differences)
    sd = statistics.stdev(differences)
    return {
        "mean_difference": mean,
        "sample_std_difference": sd,
        "standard_error": sd / math.sqrt(len(differences)),
        "values": differences,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.input_dir.resolve().glob("result_seed_*.json"))
    if len(files) != 5:
        raise RuntimeError(f"expected five seed results, found {len(files)}")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    seeds = [int(item["seed"]) for item in results]
    if seeds != [1, 2, 3, 4, 42]:
        raise RuntimeError(f"unexpected seed order: {seeds}")

    stable_checks = {
        "dataset": len({item["dataset"] for item in results}) == 1,
        "problem_type": len({item["problem_type"] for item in results}) == 1,
        "metric_profile": len({item["metric_profile"] for item in results}) == 1,
        "task_semantics": len(
            {json.dumps(item["task_semantics"], sort_keys=True) for item in results}
        ) == 1,
        "normalization_algorithm": len(
            {item["normalization"]["algorithm"] for item in results}
        ) == 1,
        "normalization_count": len({item["normalization"]["count"] for item in results}) == 1,
        "normalization_source_classes": len(
            {
                json.dumps(item["normalization"]["source_classes"], sort_keys=True)
                for item in results
            }
        ) == 1,
        "model_parameters": len(
            {json.dumps(item["model_parameters"], sort_keys=True) for item in results}
        ) == 1,
        "checkpoint_count": len({len(item["checkpoints"]) for item in results}) == 1,
    }
    if not all(stable_checks.values()):
        raise RuntimeError(f"cross-seed structural check failed: {stable_checks}")

    per_seed: dict[str, Any] = {}
    aggregate: dict[str, Any] = {}
    by_view_metric: dict[str, dict[str, list[float]]] = {}
    for view in VIEWS:
        by_view_metric[view] = {}
        aggregate[view] = {}
        for metric in METRICS:
            values = [
                float(item["summary"]["views"]["official"][view][metric])
                for item in results
            ]
            by_view_metric[view][metric] = values
            aggregate[view][metric] = summarize(values)
    for path, item in zip(files, results):
        seed = int(item["seed"])
        per_seed[str(seed)] = {
            "job_id": JOB_IDS[seed],
            "result_file_sha256": sha256_file(path),
            "protocol_sha256": item["protocol_sha256"],
            "deterministic_result_sha256": item["deterministic_result_sha256"],
            "views": {
                view: {
                    metric: float(item["summary"]["views"]["official"][view][metric])
                    for metric in METRICS
                }
                for view in VIEWS
            },
        }

    comparisons = {
        "joint_cap3000_minus_head_only": {
            metric: paired(
                by_view_metric["joint_cap3000"][metric],
                by_view_metric["head_only"][metric],
            )
            for metric in METRICS
        },
        "joint_cap3000_minus_joint_uncapped": {
            metric: paired(
                by_view_metric["joint_cap3000"][metric],
                by_view_metric["joint_uncapped"][metric],
            )
            for metric in METRICS
        },
    }
    output: dict[str, Any] = {
        "schema_version": "cic18_ft256x4_five_seed_closure_v1",
        "status": "completed",
        "dataset": results[0]["dataset"],
        "seeds": seeds,
        "model": "FT-Transformer 256x4",
        "training_schedule": "one Task-0 pretraining epoch and one epoch per later task",
        "job_ids": [JOB_IDS[seed] for seed in seeds],
        "cross_seed_structural_checks": stable_checks,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "paired_comparisons": comparisons,
        "interpretation_guardrails": [
            "This campaign uses FT256x4 with a 1+1 epoch schedule and is protocol-separated from the FT512x12 8/10-epoch manuscript campaign.",
            "Protocol hashes are seed-specific; structural equality is checked explicitly above.",
            "Five seeds support descriptive mean and sample standard deviation, not a universal deployment claim.",
        ],
    }
    output["canonical_sha256"] = canonical_sha256(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "canonical_sha256": output["canonical_sha256"]}))


if __name__ == "__main__":
    main()
