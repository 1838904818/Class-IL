#!/usr/bin/env python3
"""Validate and summarize a paired five-seed checkpoint-selection campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from scipy import stats


METRICS = (
    "average_task_accuracy",
    "average_forgetting",
    "final_overall_accuracy",
    "final_macro_f1",
    "final_balanced_accuracy",
    "final_attack_detection_recall",
    "final_benign_false_positive_rate",
)
METRIC_BOUNDS = {
    "average_task_accuracy": (0.0, 1.0),
    "average_forgetting": (-1.0, 1.0),
    "final_overall_accuracy": (0.0, 1.0),
    "final_macro_f1": (0.0, 1.0),
    "final_balanced_accuracy": (0.0, 1.0),
    "final_attack_detection_recall": (0.0, 1.0),
    "final_benign_false_positive_rate": (0.0, 1.0),
}
PRIMARY_METRICS = ("final_macro_f1", "average_forgetting")
EXPECTED_ROUTE = ("official", "joint_cap3000")
EXPECTED_CALIBRATION_AUDIT_SHA256 = (
    "697750d599f448ba1120ba60decac98abb607204eec1a431073673cc3b124b9b"
)
T_CRITICAL_95_DF4 = 2.7764451051977987


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"invalid SHA-256 for {label}: {text!r}")
    return text


def _bounded_metric(value: Any, metric: str, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite value for {label}: {value!r}")
    lower, upper = METRIC_BOUNDS[metric]
    if number < lower or number > upper:
        raise ValueError(
            f"out-of-range value for {label}: {number}; expected [{lower}, {upper}]"
        )
    return number


def _arm_seed_dir(root: Path, arm: str, seed: int) -> Path:
    return root / arm / f"seed_{seed}"


def _protocol_control_fingerprint(protocol: dict[str, Any]) -> dict[str, Any]:
    config = dict(protocol["config"])
    config.pop("family_checkpoint_selection", None)
    config.pop("family_checkpoint_min_macro_f1_gain", None)
    config.pop("family_checkpoint_max_positive_recall_drop", None)
    config.pop("family_checkpoint_max_negative_fpr_increase", None)
    return {
        "manifest_sha256": protocol["manifest_sha256"],
        "dataset": protocol["dataset"],
        "feature_dim": protocol["feature_dim"],
        "tasks": protocol["tasks"],
        "class_names": protocol["class_names"],
        "normal_class": protocol["normal_class"],
        "normalization_scope": protocol["normalization_scope"],
        "prediction_arms": protocol["prediction_arms"],
        "config_except_checkpoint_selection": config,
    }


def _validate_one(
    directory: Path,
    *,
    seed: int,
    expected_selection: str,
) -> dict[str, Any]:
    protocol_path = directory / "protocol.json"
    result_path = directory / f"result_seed_{seed}.json"
    summary_path = directory / "summary.json"
    for path in (protocol_path, result_path, summary_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing non-empty campaign artifact: {path}")

    protocol = _load_json(protocol_path)
    result = _load_json(result_path)
    summary = _load_json(summary_path)
    if protocol["seeds"] != [seed]:
        raise ValueError(f"seed contract mismatch in {protocol_path}")
    if int(result["seed"]) != seed:
        raise ValueError(f"result seed mismatch in {result_path}")
    if result["protocol_sha256"] != protocol["protocol_sha256"]:
        raise ValueError(f"result/protocol SHA mismatch in {directory}")
    if summary["protocol_sha256"] != protocol["protocol_sha256"]:
        raise ValueError(f"summary/protocol SHA mismatch in {directory}")
    if summary["aggregate"]["seeds"] != [seed]:
        raise ValueError(f"summary seed mismatch in {summary_path}")
    if protocol["config"]["family_checkpoint_selection"] != expected_selection:
        raise ValueError(f"checkpoint-selection mismatch in {protocol_path}")
    calibration_protocol = protocol["training_calibration"]
    calibration_audit_sha256: str | None = None
    if expected_selection == "last":
        if calibration_protocol != {"enabled": False}:
            raise ValueError(
                f"last-epoch control unexpectedly received calibration: {protocol_path}"
            )
    elif expected_selection in {
        "training_only_calibration_macro_f1",
        "training_only_calibration_macro_f1_recall_fpr_guard",
    }:
        if calibration_protocol.get("enabled") is not True:
            raise ValueError(f"training calibration is disabled: {protocol_path}")
        if calibration_protocol.get("official_test_used") is not False:
            raise ValueError(f"official test entered checkpoint selection: {protocol_path}")
        if calibration_protocol.get("future_class_policy") != (
            "only classes seen through the current task"
        ):
            raise ValueError(f"future-class policy mismatch: {protocol_path}")
        calibration_audit_sha256 = _require_hash(
            calibration_protocol.get("audit_sha256"),
            f"training calibration audit seed {seed}",
        )
        if calibration_audit_sha256 != EXPECTED_CALIBRATION_AUDIT_SHA256:
            raise ValueError(f"training calibration audit mismatch: {protocol_path}")
    else:
        raise ValueError(f"unsupported checkpoint-selection mode: {expected_selection}")

    deterministic_hash = _require_hash(
        result["deterministic_result_sha256"], f"result seed {seed}"
    )
    if summary["deterministic_result_sha256"][str(seed)] != deterministic_hash:
        raise ValueError(f"deterministic result hash mismatch in {directory}")

    view = result["summary"]["views"][EXPECTED_ROUTE[0]][EXPECTED_ROUTE[1]]
    metrics = {
        metric: _bounded_metric(
            view[metric], metric, f"seed={seed} metric={metric}"
        )
        for metric in METRICS
    }
    records = result["training_exposure_records"]
    if set(records) != {str(value) for value in range(8)}:
        raise ValueError(f"unexpected class set in {result_path}")
    selected_epochs: dict[str, int] = {}
    selection_reasons: dict[str, str] = {}
    if expected_selection in {
        "training_only_calibration_macro_f1",
        "training_only_calibration_macro_f1_recall_fpr_guard",
    }:
        for class_id, record in sorted(records.items(), key=lambda item: int(item[0])):
            selection = record["checkpoint_selection"]
            if selection.get("mode") != expected_selection:
                raise ValueError(f"invalid calibrated record for class {class_id}")
            if selection["official_test_used"] is not False:
                raise ValueError(f"official test used for class {class_id}, seed {seed}")
            if selection.get("future_classes_used", False) is not False:
                raise ValueError(f"future class used for class {class_id}, seed {seed}")
            epoch = int(selection["selected_epoch"])
            if epoch not in range(1, 11):
                raise ValueError(f"invalid selected epoch {epoch}, class {class_id}")
            selected_epochs[class_id] = epoch
            selection_reasons[class_id] = str(selection["reason"])
            if expected_selection.endswith("recall_fpr_guard") and selection["applied"]:
                decision = selection.get("guard_decision", {})
                if decision.get("decision") not in {
                    "restore_candidate",
                    "retain_last_epoch",
                }:
                    raise ValueError(f"invalid guard decision for class {class_id}")
    else:
        for class_id, record in sorted(records.items(), key=lambda item: int(item[0])):
            selection = record["checkpoint_selection"]
            if selection.get("mode") != "last":
                raise ValueError(f"invalid last-epoch record for class {class_id}")
            if selection.get("applied") is not False:
                raise ValueError(f"last-epoch selection was unexpectedly applied: {class_id}")
            if selection.get("reason") != "last_epoch_control":
                raise ValueError(f"invalid last-epoch reason for class {class_id}")

    return {
        "seed": seed,
        "metrics": metrics,
        "selected_epochs": selected_epochs,
        "selection_reasons": selection_reasons,
        "protocol_sha256": _require_hash(protocol["protocol_sha256"], "protocol"),
        "deterministic_result_sha256": deterministic_hash,
        "protocol_file_sha256": _file_sha256(protocol_path),
        "result_file_sha256": _file_sha256(result_path),
        "summary_file_sha256": _file_sha256(summary_path),
        "control_fingerprint": _protocol_control_fingerprint(protocol),
        "training_calibration_audit_sha256": calibration_audit_sha256,
    }


def _holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def _metric_statistics(
    baseline: Iterable[float], candidate: Iterable[float]
) -> dict[str, Any]:
    baseline_values = [float(value) for value in baseline]
    candidate_values = [float(value) for value in candidate]
    if len(baseline_values) != 5 or len(candidate_values) != 5:
        raise ValueError("paired campaign requires exactly five values per arm")
    differences = [
        candidate_value - baseline_value
        for baseline_value, candidate_value in zip(baseline_values, candidate_values)
    ]
    difference_mean = statistics.mean(differences)
    difference_sd = statistics.stdev(differences)
    standard_error = difference_sd / math.sqrt(len(differences))
    ci_low = difference_mean - T_CRITICAL_95_DF4 * standard_error
    ci_high = difference_mean + T_CRITICAL_95_DF4 * standard_error
    if difference_sd == 0.0:
        # A constant non-zero paired delta implies an unbounded t statistic and
        # standardized effect.  Keep the JSON standards-compliant instead of
        # serializing non-standard Infinity tokens.
        t_statistic = None if difference_mean != 0.0 else 0.0
        t_pvalue = 0.0 if difference_mean != 0.0 else 1.0
        effect_dz = None if difference_mean != 0.0 else 0.0
    else:
        test = stats.ttest_rel(candidate_values, baseline_values)
        t_statistic = float(test.statistic)
        t_pvalue = float(test.pvalue)
        effect_dz = difference_mean / difference_sd
    try:
        wilcoxon = stats.wilcoxon(
            candidate_values,
            baseline_values,
            alternative="two-sided",
            method="auto",
        )
        wilcoxon_statistic = float(wilcoxon.statistic)
        wilcoxon_pvalue = float(wilcoxon.pvalue)
    except ValueError:
        wilcoxon_statistic = 0.0
        wilcoxon_pvalue = 1.0
    return {
        "baseline_per_seed": baseline_values,
        "candidate_per_seed": candidate_values,
        "paired_delta_per_seed": differences,
        "baseline_mean": statistics.mean(baseline_values),
        "baseline_sample_std": statistics.stdev(baseline_values),
        "candidate_mean": statistics.mean(candidate_values),
        "candidate_sample_std": statistics.stdev(candidate_values),
        "paired_delta_mean": difference_mean,
        "paired_delta_sample_std": difference_sd,
        "paired_delta_95ci": [ci_low, ci_high],
        "paired_t_statistic": t_statistic,
        "paired_t_pvalue_raw": t_pvalue,
        "cohen_dz": effect_dz,
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_pvalue_raw": wilcoxon_pvalue,
    }


def build_report(
    campaign_root: Path,
    *,
    baseline_arm: str,
    candidate_arm: str,
    seeds: list[int],
) -> dict[str, Any]:
    if seeds != [1, 2, 3, 4, 42]:
        raise ValueError("formal seed contract must be [1, 2, 3, 4, 42]")
    baseline_records = [
        _validate_one(
            _arm_seed_dir(campaign_root, baseline_arm, seed),
            seed=seed,
            expected_selection="last",
        )
        for seed in seeds
    ]
    candidate_selection = (
        "training_only_calibration_macro_f1_recall_fpr_guard"
        if candidate_arm == "guarded_checkpoint"
        else "training_only_calibration_macro_f1"
    )
    candidate_records = [
        _validate_one(
            _arm_seed_dir(campaign_root, candidate_arm, seed),
            seed=seed,
            expected_selection=candidate_selection,
        )
        for seed in seeds
    ]
    fingerprints = {
        json.dumps(record["control_fingerprint"], sort_keys=True)
        for record in baseline_records + candidate_records
    }
    if len(fingerprints) != 1:
        raise ValueError("paired arms differ beyond checkpoint-selection policy")

    metrics: dict[str, Any] = {}
    for metric in METRICS:
        metrics[metric] = _metric_statistics(
            [record["metrics"][metric] for record in baseline_records],
            [record["metrics"][metric] for record in candidate_records],
        )
    primary_raw = {
        metric: metrics[metric]["paired_t_pvalue_raw"]
        for metric in PRIMARY_METRICS
    }
    primary_holm = _holm_adjust(primary_raw)
    for metric in PRIMARY_METRICS:
        metrics[metric]["paired_t_pvalue_holm_primary_two"] = primary_holm[metric]

    epoch_counts: dict[str, dict[str, int]] = {str(value): {} for value in range(8)}
    fallback_counts: dict[str, int] = {str(value): 0 for value in range(8)}
    for record in candidate_records:
        for class_id, epoch in record["selected_epochs"].items():
            key = str(epoch)
            epoch_counts[class_id][key] = epoch_counts[class_id].get(key, 0) + 1
            if record["selection_reasons"][class_id] != (
                "sufficient_training_only_calibration_support"
            ):
                fallback_counts[class_id] += 1

    return {
        "schema_version": 1,
        "campaign": (
            "replayids-d2-checkpoint-recall-guard-paired-five-seed-v1"
            if candidate_arm == "guarded_checkpoint"
            else "replayids-d2-checkpoint-selection-paired-five-seed-v3"
        ),
        "experimental_role": "paired_guarded_checkpoint_confirmation",
        "seeds": seeds,
        "arms": {
            "baseline": baseline_arm,
            "candidate": candidate_arm,
        },
        "primary_route": "/".join(EXPECTED_ROUTE),
        "primary_metrics_for_inference": list(PRIMARY_METRICS),
        "other_metrics_are_descriptive": True,
        "metrics": metrics,
        "selected_epoch_counts_by_class": epoch_counts,
        "low_support_fallback_counts_by_class": fallback_counts,
        "records": {
            "baseline": baseline_records,
            "candidate": candidate_records,
        },
        "limitations": [
            "five paired training seeds share one fixed data split and are not five independent datasets",
            "n=5 limits distributional diagnostics and exact Wilcoxon power",
            "Heartbleed has one training-calibration row and therefore uses the registered last-epoch fallback",
        ],
    }


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ReplayIDS D2 paired five-seed guarded-checkpoint result",
        "",
        "Seeds: `1, 2, 3, 4, 42`. Primary route: `official/joint_cap3000`.",
        "",
        "| Metric | Last epoch mean +/- SD | Guarded checkpoint mean +/- SD | Paired delta | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "average_task_accuracy": "Average task accuracy",
        "average_forgetting": "Average forgetting",
        "final_overall_accuracy": "Final accuracy",
        "final_macro_f1": "Final Macro-F1",
        "final_balanced_accuracy": "Final balanced accuracy",
        "final_attack_detection_recall": "Attack recall",
        "final_benign_false_positive_rate": "Benign FPR",
    }
    for metric in METRICS:
        values = report["metrics"][metric]
        ci_low, ci_high = values["paired_delta_95ci"]
        lines.append(
            "| {label} | {bmean} +/- {bsd} | {cmean} +/- {csd} | {delta} | [{lo}, {hi}] |".format(
                label=labels[metric],
                bmean=_percent(values["baseline_mean"]),
                bsd=_percent(values["baseline_sample_std"]),
                cmean=_percent(values["candidate_mean"]),
                csd=_percent(values["candidate_sample_std"]),
                delta=_percent(values["paired_delta_mean"]),
                lo=_percent(ci_low),
                hi=_percent(ci_high),
            )
        )
    lines.extend(
        [
            "",
            "## Inference boundary",
            "",
            "Paired t-tests are designated only for Macro-F1 and forgetting and are Holm-adjusted across those two outcomes. Other metrics are descriptive.",
            "",
        ]
    )
    for metric in PRIMARY_METRICS:
        values = report["metrics"][metric]
        effect_dz = values["cohen_dz"]
        effect_text = (
            "undefined (zero paired-difference variance)"
            if effect_dz is None
            else f"{effect_dz:.4g}"
        )
        lines.append(
            f"- `{metric}`: raw paired-t p={values['paired_t_pvalue_raw']:.6g}; Holm p={values['paired_t_pvalue_holm_primary_two']:.6g}; Wilcoxon p={values['wilcoxon_pvalue_raw']:.6g}; Cohen dz={effect_text}."
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}." for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--baseline-arm", default="last_epoch")
    parser.add_argument("--candidate-arm", default="training_only_calibration")
    parser.add_argument("--expected-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        args.campaign_root,
        baseline_arm=args.baseline_arm,
        candidate_arm=args.candidate_arm,
        seeds=list(args.expected_seeds),
    )
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print("PAIRED_CHECKPOINT_SELECTION_REPORT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
