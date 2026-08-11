from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streaming_full.data import canonical_sha256, sha256_file
from streaming_full.summarize import _holm_adjust, exact_wilcoxon
from streaming_full.validation import _atomic_write_json


SEEDS = (1, 2, 3, 4, 42)
VIEWS = ("official", "duplicate_excluded")
ARM = "joint_uncapped"
METRICS = (
    "final_overall_accuracy",
    "final_macro_f1",
    "final_balanced_accuracy",
    "average_task_accuracy",
    "average_forgetting",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the paired five-seed Malaya MLP-size validation."
    )
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--larger-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_run(directory: Path) -> tuple[dict, dict[int, dict]]:
    directory = directory.resolve()
    protocol_path = directory / "protocol.json"
    summary_path = directory / "summary.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        protocol.get("dataset") != "malaya-network-gt"
        or protocol.get("problem_type") != "application_classification"
        or protocol.get("metric_profile") != "generic_multiclass"
        or protocol.get("normal_class", {}).get("class_id") is not None
        or protocol.get("seeds") != list(SEEDS)
    ):
        raise ValueError(f"invalid Malaya five-seed protocol: {protocol_path}")
    if summary.get("protocol_sha256") != protocol.get("protocol_sha256"):
        raise ValueError(f"summary/protocol hash mismatch: {directory}")
    results = {}
    for seed in SEEDS:
        path = directory / f"result_seed_{seed}.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        if (
            result.get("seed") != seed
            or result.get("protocol_sha256") != protocol["protocol_sha256"]
            or result.get("problem_type") != "application_classification"
            or result.get("metric_profile") != "generic_multiclass"
            or result.get("normal_class_id") is not None
        ):
            raise ValueError(f"invalid result identity: {path}")
        if "binary_detection" in json.dumps(result):
            raise ValueError(f"generic result contains binary detection output: {path}")
        results[seed] = result
    return protocol, results


def _assert_matched_design(baseline: dict, larger: dict) -> dict[str, object]:
    identical_fields = (
        "manifest_sha256",
        "dataset",
        "feature_dim",
        "problem_type",
        "metric_profile",
        "task_semantics",
        "tasks",
        "class_names",
        "seeds",
        "evaluation",
        "exposure_prior_diagnostic",
        "router_algorithms",
    )
    mismatches = [key for key in identical_fields if baseline.get(key) != larger.get(key)]
    if mismatches:
        raise ValueError("model-size protocols differ on matched fields: " + ", ".join(mismatches))
    baseline_config = dict(baseline["config"])
    larger_config = dict(larger["config"])
    changed = {
        key: {"baseline": baseline_config.get(key), "larger": larger_config.get(key)}
        for key in sorted(set(baseline_config) | set(larger_config))
        if baseline_config.get(key) != larger_config.get(key)
    }
    if set(changed) != {"d_model", "n_layers"}:
        raise ValueError(f"unexpected RunConfig differences: {changed}")
    return {"identical_fields": list(identical_fields), "changed_config_fields": changed}


def _metric_record(baseline: np.ndarray, larger: np.ndarray) -> dict[str, object]:
    delta = larger - baseline
    n = len(delta)
    sample_sd = float(delta.std(ddof=1))
    standard_error = sample_sd / math.sqrt(n)
    t_critical = float(stats.t.ppf(0.975, n - 1))
    paired_t = stats.ttest_rel(larger, baseline)
    wilcoxon = exact_wilcoxon(delta)
    return {
        "baseline": {
            "per_seed": baseline.tolist(),
            "mean": float(baseline.mean()),
            "sample_std": float(baseline.std(ddof=1)),
        },
        "larger": {
            "per_seed": larger.tolist(),
            "mean": float(larger.mean()),
            "sample_std": float(larger.std(ddof=1)),
        },
        "paired_delta_larger_minus_baseline": {
            "per_seed": delta.tolist(),
            "mean": float(delta.mean()),
            "sample_std": sample_sd,
            "ci95_student_t": [
                float(delta.mean() - t_critical * standard_error),
                float(delta.mean() + t_critical * standard_error),
            ],
            "paired_t": {
                "statistic": float(paired_t.statistic),
                "p_value_raw": float(paired_t.pvalue),
                "df": n - 1,
            },
            "wilcoxon": wilcoxon,
        },
    }


def main() -> None:
    args = _parser().parse_args()
    baseline_protocol, baseline_results = _load_run(args.baseline_dir)
    larger_protocol, larger_results = _load_run(args.larger_dir)
    design = _assert_matched_design(baseline_protocol, larger_protocol)

    views: dict[str, object] = {}
    for view in VIEWS:
        metrics = {}
        raw_p_values = []
        for metric in METRICS:
            baseline = np.asarray(
                [
                    baseline_results[seed]["summary"]["views"][view][ARM][metric]
                    for seed in SEEDS
                ],
                dtype=np.float64,
            )
            larger = np.asarray(
                [
                    larger_results[seed]["summary"]["views"][view][ARM][metric]
                    for seed in SEEDS
                ],
                dtype=np.float64,
            )
            record = _metric_record(baseline, larger)
            metrics[metric] = record
            raw_p_values.append(
                record["paired_delta_larger_minus_baseline"]["paired_t"][
                    "p_value_raw"
                ]
            )
        adjusted = _holm_adjust(raw_p_values)
        for metric, value in zip(METRICS, adjusted):
            metrics[metric]["paired_delta_larger_minus_baseline"]["paired_t"][
                "p_value_holm_within_view_m5"
            ] = float(value)
        views[view] = {"arm": ARM, "metrics": metrics}

    report = {
        "schema_version": 1,
        "report": "malaya_network_gt_paired_model_size_validation_v1",
        "created_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "dataset": "malaya-network-gt",
        "source_revision": "384a59278f98490ee6e93aae017e748078d29b6a",
        "seeds": list(SEEDS),
        "model_pair": {
            "baseline": {
                "name": "MLP 128x2",
                "encoder_parameters": 26496,
                "protocol_sha256": baseline_protocol["protocol_sha256"],
                "summary_file_sha256": sha256_file(args.baseline_dir / "summary.json"),
            },
            "larger": {
                "name": "MLP 256x4",
                "encoder_parameters": 217344,
                "protocol_sha256": larger_protocol["protocol_sha256"],
                "summary_file_sha256": sha256_file(args.larger_dir / "summary.json"),
            },
            "parameter_ratio": 217344 / 26496,
        },
        "matched_design": design,
        "inference_policy": {
            "delta": "larger_minus_baseline",
            "confidence_interval": "two-sided 95% Student-t CI on five paired seed deltas, df=4",
            "paired_t": "two-sided paired t-test; Holm adjusted within each five-metric view family",
            "wilcoxon": "two-sided exact signed-rank test with zero deltas discarded",
            "n5_limit": "With five non-zero pairs, the minimum attainable two-sided exact Wilcoxon p-value is 0.0625.",
        },
        "views": views,
    }
    deterministic_fields = {
        key: value for key, value in report.items() if key != "created_utc"
    }
    report["deterministic_result_sha256"] = canonical_sha256(
        deterministic_fields
    )
    report["canonical_report_sha256"] = canonical_sha256(report)
    _atomic_write_json(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "deterministic_result_sha256": report[
                    "deterministic_result_sha256"
                ],
                "canonical_report_sha256": report["canonical_report_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
