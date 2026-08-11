from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import stats

from .data import canonical_sha256, resolve_manifest_semantics
from .exposure_preflight import validate_exposure_preflight
from .validation import (
    ARMS,
    DIAGNOSTIC_ARMS,
    EXPOSURE_PRIOR_FORMULA_REGISTRY,
    OUTPUT_DIRECTORY_LOCK_CONTRACT,
    PRIMARY_ARMS,
    _aggregate,
    _atomic_write_json,
    _load_json_object,
    _validate_protocol_file,
    _validate_result_file,
)


METRICS: dict[str, str] = {
    "average_task_accuracy": "higher_is_better",
    "average_forgetting": "lower_is_better",
    "final_overall_accuracy": "higher_is_better",
    "final_macro_f1": "higher_is_better",
    "final_balanced_accuracy": "higher_is_better",
    "final_benign_false_positive_rate": "lower_is_better",
    "final_attack_detection_recall": "higher_is_better",
}

PRIMARY_COMPARISONS: dict[str, tuple[str, str]] = {
    "joint_uncapped_minus_joint_cap3000": ("joint_cap3000", "joint_uncapped"),
    "router_uncapped_minus_router_cap3000": (
        "router_only_cap3000",
        "router_only_uncapped",
    ),
    "router_cap3000_minus_head": ("head_only", "router_only_cap3000"),
    "joint_cap3000_minus_head": ("head_only", "joint_cap3000"),
    "joint_cap3000_minus_router_cap3000": (
        "router_only_cap3000",
        "joint_cap3000",
    ),
    "router_uncapped_minus_head": ("head_only", "router_only_uncapped"),
    "joint_uncapped_minus_head": ("head_only", "joint_uncapped"),
    "joint_uncapped_minus_router_uncapped": (
        "router_only_uncapped",
        "joint_uncapped",
    ),
}

DIAGNOSTIC_COMPARISONS: dict[str, tuple[str, str]] = {
    "head_exposure_prior_corrected_minus_head": (
        "head_only",
        "head_only_exposure_prior_corrected",
    ),
    "joint_cap_exposure_prior_corrected_minus_joint_cap": (
        "joint_cap3000",
        "joint_cap3000_exposure_prior_corrected",
    ),
    "joint_uncapped_exposure_prior_corrected_minus_joint_uncapped": (
        "joint_uncapped",
        "joint_uncapped_exposure_prior_corrected",
    ),
}


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def exact_wilcoxon(deltas: np.ndarray) -> dict:
    deltas = np.asarray(deltas, dtype=np.float64)
    nonzero = deltas[deltas != 0.0]
    effective = len(nonzero)
    if effective == 0:
        return {
            "statistic": 0.0,
            "p_value_raw": 1.0,
            "n_effective": 0,
            "zero_method": "discard",
            "method": "exact_sign_enumeration_with_average_tie_ranks",
            "minimum_attainable_two_sided_p": 1.0,
        }
    ranks = _average_ranks(np.abs(nonzero))
    observed_plus = float(ranks[nonzero > 0].sum())
    total_rank = float(ranks.sum())
    possible = []
    for signs in itertools.product((0, 1), repeat=effective):
        possible.append(float(ranks[np.asarray(signs, dtype=bool)].sum()))
    possible_array = np.asarray(possible, dtype=np.float64)
    lower = float(np.mean(possible_array <= observed_plus + 1e-12))
    upper = float(np.mean(possible_array >= observed_plus - 1e-12))
    p_value = min(1.0, 2.0 * min(lower, upper))
    minimum = min(1.0, 2.0 / (2**effective))
    return {
        "statistic": float(min(observed_plus, total_rank - observed_plus)),
        "p_value_raw": p_value,
        "n_effective": effective,
        "zero_method": "discard",
        "method": "exact_sign_enumeration_with_average_tie_ranks",
        "minimum_attainable_two_sided_p": minimum,
    }


def _paired_t_and_ci(deltas: np.ndarray) -> tuple[dict, list[float]]:
    deltas = np.asarray(deltas, dtype=np.float64)
    count = len(deltas)
    mean = float(deltas.mean())
    sample_sd = float(deltas.std(ddof=1))
    if sample_sd == 0.0:
        if mean == 0.0:
            test = {"statistic": 0.0, "p_value_raw": 1.0, "degenerate": True}
        else:
            test = {
                "statistic": None,
                "statistic_text": "+infinity" if mean > 0 else "-infinity",
                "p_value_raw": 0.0,
                "degenerate": True,
            }
        return test, [mean, mean]
    statistic = mean / (sample_sd / math.sqrt(count))
    p_value = float(2.0 * stats.t.sf(abs(statistic), df=count - 1))
    half_width = float(stats.t.ppf(0.975, df=count - 1) * sample_sd / math.sqrt(count))
    return {
        "statistic": float(statistic),
        "p_value_raw": p_value,
        "degenerate": False,
    }, [mean - half_width, mean + half_width]


def _holm_adjust(values: Sequence[float]) -> list[float]:
    count = len(values)
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="stable")
    adjusted = np.empty(count, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(values[int(index)]))
        running = max(running, candidate)
        adjusted[int(index)] = running
    return adjusted.tolist()


def _validate_dataset_directory(
    protocol_path: Path,
    expected_seeds: list[int],
) -> tuple[dict, list[dict]]:
    protocol_raw = _load_json_object(protocol_path)
    protocol_sha256 = protocol_raw.get("protocol_sha256")
    if not isinstance(protocol_sha256, str):
        raise RuntimeError(f"protocol lacks SHA-256: {protocol_path}")
    protocol = _validate_protocol_file(protocol_path, protocol_sha256)
    if protocol.get("schema_version") != 2:
        raise RuntimeError(f"formal protocol is not schema version 2: {protocol_path}")
    if (
        protocol.get("prediction_arms") != ARMS
        or protocol.get("primary_prediction_arms") != list(PRIMARY_ARMS)
        or protocol.get("diagnostic_prediction_arms") != list(DIAGNOSTIC_ARMS)
        or protocol.get("output_directory_lock") != OUTPUT_DIRECTORY_LOCK_CONTRACT
    ):
        raise RuntimeError(f"formal protocol arm/lock registry mismatch: {protocol_path}")
    normal_class = protocol.get("normal_class")
    if not isinstance(normal_class, dict):
        raise RuntimeError(f"formal protocol lacks normal-class semantics: {protocol_path}")
    semantic_record = {
        "problem_type": protocol.get("problem_type"),
        "metric_profile": protocol.get("metric_profile"),
        "task_semantics": protocol.get("task_semantics"),
        "normal_class_id": normal_class.get("class_id"),
    }
    try:
        problem_type, metric_profile, task_semantics, normal_class_id = (
            resolve_manifest_semantics(
                semantic_record, context="formal protocol"
            )
        )
    except ValueError as error:
        raise RuntimeError(
            f"formal protocol semantic mode is invalid: {protocol_path}"
        ) from error
    class_names = protocol.get("class_names")
    expected_normal_name = None
    if normal_class_id is not None and isinstance(class_names, dict):
        expected_normal_name = class_names.get(
            str(normal_class_id), class_names.get(normal_class_id)
        )
    expected_normal_source = (
        "explicit manifest normal_class_id"
        if normal_class_id is not None
        else "not applicable for application_classification"
    )
    if (
        normal_class.get("class_name") != expected_normal_name
        or normal_class.get("source") != expected_normal_source
    ):
        raise RuntimeError(
            f"formal protocol normal-class record is invalid: {protocol_path}"
        )
    exposure_formula = protocol.get("exposure_prior_diagnostic")
    if not isinstance(exposure_formula, dict) or any(
        exposure_formula.get(key) != value
        for key, value in EXPOSURE_PRIOR_FORMULA_REGISTRY.items()
    ):
        raise RuntimeError(f"formal protocol exposure-prior formula mismatch: {protocol_path}")
    try:
        exposure_preflight = validate_exposure_preflight(
            protocol.get("exposure_preflight")
        )
    except ValueError as error:
        raise RuntimeError(
            f"formal protocol exposure preflight is invalid: {protocol_path}"
        ) from error
    if (
        exposure_preflight.get("dataset") != protocol.get("dataset")
        or exposure_preflight.get("streaming_manifest_sha256")
        != protocol.get("manifest_sha256")
        or exposure_preflight.get("run_config") != protocol.get("config")
        or exposure_preflight.get("problem_type") != problem_type
        or exposure_preflight.get("metric_profile") != metric_profile
        or exposure_preflight.get("task_semantics") != task_semantics
        or exposure_preflight.get("normal_class_id") != normal_class_id
    ):
        raise RuntimeError(f"formal protocol exposure preflight binding mismatch: {protocol_path}")
    evaluation = protocol.get("evaluation")
    view_order = evaluation.get("view_order") if isinstance(evaluation, dict) else None
    if (
        not isinstance(view_order, list)
        or not view_order
        or view_order[0] != "official"
        or len(set(view_order)) != len(view_order)
        or list(evaluation.get("views", {})) != view_order
        or evaluation.get("algorithm") != "single_forward_mask_projection_v1"
        or evaluation.get("empty_retained_class_policy") != "error_before_training"
    ):
        raise RuntimeError(f"formal protocol evaluation-view registry mismatch: {protocol_path}")
    if protocol.get("seeds") != expected_seeds:
        raise RuntimeError(
            f"seed contract mismatch in {protocol_path}: {protocol.get('seeds')} != {expected_seeds}"
        )
    dataset = protocol.get("dataset")
    if not isinstance(dataset, str):
        raise RuntimeError(f"protocol lacks dataset identity: {protocol_path}")
    expected_names = {f"result_seed_{seed}.json" for seed in expected_seeds}
    actual_names = {path.name for path in protocol_path.parent.glob("result_seed_*.json")}
    if actual_names != expected_names:
        raise RuntimeError(
            f"result file set mismatch for {dataset}: {actual_names} != {expected_names}"
        )
    results = [
        _validate_result_file(
            protocol_path.parent / f"result_seed_{seed}.json",
            dataset=dataset,
            seed=seed,
            protocol_sha256=protocol_sha256,
            expected_problem_type=problem_type,
            expected_metric_profile=metric_profile,
            expected_task_semantics=task_semantics,
            expected_normal_class_id=normal_class_id,
            expected_view_order=protocol["evaluation"]["view_order"],
            expected_exposure_preflight=protocol["exposure_preflight"],
        )
        for seed in expected_seeds
    ]
    expected_views = protocol.get("evaluation", {}).get("view_order")
    for result in results:
        if list(result["summary"]["views"]) != expected_views:
            raise RuntimeError(
                f"result evaluation-view order mismatch: seed={result['seed']}"
            )
    summary_path = protocol_path.parent / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"missing summary.json beside {protocol_path}")
    summary = _load_json_object(summary_path)
    recomputed = _aggregate(results)
    expected_hashes = {
        str(result["seed"]): result["deterministic_result_sha256"] for result in results
    }
    if (
        summary.get("schema_version") != 2
        or summary.get("dataset") != dataset
        or summary.get("problem_type") != problem_type
        or summary.get("metric_profile") != metric_profile
        or summary.get("task_semantics") != task_semantics
        or summary.get("normal_class_id") != normal_class_id
        or summary.get("protocol_sha256") != protocol_sha256
        or summary.get("result_files")
        != [f"result_seed_{seed}.json" for seed in expected_seeds]
        or summary.get("deterministic_result_sha256") != expected_hashes
        or canonical_sha256(summary.get("aggregate")) != canonical_sha256(recomputed)
    ):
        raise RuntimeError(f"summary does not trace to validated result JSON: {summary_path}")
    return protocol, results


def _paired_metric_record(
    results: list[dict],
    directory: Path,
    values,
    *,
    preference: str,
) -> dict:
    pairs = []
    deltas = []
    for result in results:
        reference, treatment = values(result)
        reference = float(reference)
        treatment = float(treatment)
        delta = treatment - reference
        deltas.append(delta)
        pairs.append(
            {
                "seed": int(result["seed"]),
                "reference": reference,
                "treatment": treatment,
                "delta": delta,
                "source_json": str(
                    (directory / f"result_seed_{result['seed']}.json").resolve()
                ),
                "deterministic_result_sha256": result[
                    "deterministic_result_sha256"
                ],
            }
        )
    delta_array = np.asarray(deltas, dtype=np.float64)
    paired_t, confidence_interval = _paired_t_and_ci(delta_array)
    return {
        "preference": preference,
        "delta_definition": "treatment_minus_reference",
        "n_pairs": len(delta_array),
        "pairs": pairs,
        "mean_delta": float(delta_array.mean()),
        "sample_sd_delta": float(delta_array.std(ddof=1)),
        "t_confidence_interval_95": confidence_interval,
        "paired_t": paired_t,
        "wilcoxon_exact": exact_wilcoxon(delta_array),
    }


def _apply_holm(
    records: dict[str, dict],
    registry: dict[str, tuple[str, str]],
    *,
    family_prefix: str,
) -> None:
    metric_names = list(next(iter(records.values()))["metrics"])
    if any(list(record["metrics"]) != metric_names for record in records.values()):
        raise RuntimeError("comparison metric registries are inconsistent")
    for metric in metric_names:
        metric_records = [records[name]["metrics"][metric] for name in registry]
        for test_name in ("paired_t", "wilcoxon_exact"):
            adjusted = _holm_adjust(
                [record[test_name]["p_value_raw"] for record in metric_records]
            )
            for record, value in zip(metric_records, adjusted):
                record[test_name]["p_value_holm"] = value
                record[test_name]["holm_family"] = (
                    f"{family_prefix};metric={metric};test={test_name};"
                    f"m={len(registry)}"
                )


def _arm_comparisons(
    protocol: dict,
    results: list[dict],
    directory: Path,
    view_name: str,
    registry: dict[str, tuple[str, str]],
    family_name: str,
) -> dict[str, dict]:
    available_metrics = {
        metric: preference
        for metric, preference in METRICS.items()
        if metric in results[0]["summary"]["views"][view_name][next(iter(ARMS))]
    }
    comparisons: dict[str, dict] = {}
    for name, (reference_arm, treatment_arm) in registry.items():
        comparisons[name] = {
            "reference_arm": reference_arm,
            "treatment_arm": treatment_arm,
            "metrics": {
                metric: _paired_metric_record(
                    results,
                    directory,
                    lambda result, m=metric, r=reference_arm, t=treatment_arm: (
                        result["summary"]["views"][view_name][r][m],
                        result["summary"]["views"][view_name][t][m],
                    ),
                    preference=preference,
                )
                for metric, preference in available_metrics.items()
            },
        }
    _apply_holm(
        comparisons,
        registry,
        family_prefix=(
            f"dataset={protocol['dataset']};view={view_name};family={family_name}"
        ),
    )
    return comparisons


def _view_sensitivity(
    protocol: dict,
    results: list[dict],
    directory: Path,
    view_name: str,
    arms: list[str],
    family_name: str,
) -> dict[str, dict]:
    contrast = f"{view_name}_minus_official"
    available_metrics = {
        metric: preference
        for metric, preference in METRICS.items()
        if metric
        in results[0]["summary"]["sensitivity"][contrast][arms[0]]
    }
    registry = {f"{arm}_view_delta": (arm, arm) for arm in arms}
    records: dict[str, dict] = {}
    for name, (arm, _) in registry.items():
        records[name] = {
            "arm": arm,
            "reference_view": "official",
            "treatment_view": view_name,
            "metrics": {
                metric: _paired_metric_record(
                    results,
                    directory,
                    lambda result, m=metric, a=arm: (
                        result["summary"]["views"]["official"][a][m],
                        result["summary"]["views"][view_name][a][m],
                    ),
                    preference="diagnostic_support_shift_no_improvement_claim",
                )
                for metric in available_metrics
            },
        }
    _apply_holm(
        records,
        registry,
        family_prefix=(
            f"dataset={protocol['dataset']};contrast={view_name}-official;"
            f"family={family_name}"
        ),
    )
    return records


def _dataset_statistics(protocol: dict, results: list[dict], directory: Path) -> dict:
    if protocol.get("schema_version") != 2:
        raise RuntimeError("formal paired inference requires protocol schema 2")
    view_names = protocol["evaluation"]["view_order"]
    view_statistics = {}
    for view_name in view_names:
        view_statistics[view_name] = {
            "primary_arm_comparisons": _arm_comparisons(
                protocol,
                results,
                directory,
                view_name,
                PRIMARY_COMPARISONS,
                "primary_m8",
            ),
            "exposure_prior_diagnostic_comparisons": _arm_comparisons(
                protocol,
                results,
                directory,
                view_name,
                DIAGNOSTIC_COMPARISONS,
                "diagnostic_m3",
            ),
        }
    sensitivity = {}
    for view_name in view_names:
        if view_name == "official":
            continue
        sensitivity[f"{view_name}_minus_official"] = {
            "primary_arms": _view_sensitivity(
                protocol,
                results,
                directory,
                view_name,
                list(PRIMARY_ARMS),
                "primary_m5",
            ),
            "diagnostic_arms": _view_sensitivity(
                protocol,
                results,
                directory,
                view_name,
                list(DIAGNOSTIC_ARMS),
                "diagnostic_m3",
            ),
            "interpretation": (
                "test-support sensitivity on the identical trained model and "
                "single forward predictions; not a model-improvement estimate"
            ),
        }
    return {
        "protocol_path": str((directory / "protocol.json").resolve()),
        "protocol_sha256": protocol["protocol_sha256"],
        "manifest_sha256": protocol["manifest_sha256"],
        "input_source_provenance": protocol.get("input_source_provenance"),
        "problem_type": protocol["problem_type"],
        "metric_profile": protocol["metric_profile"],
        "task_semantics": protocol["task_semantics"],
        "metric_registry": {
            metric: preference
            for metric, preference in METRICS.items()
            if metric
            in results[0]["summary"]["views"]["official"][next(iter(ARMS))]
        },
        "evaluation": protocol["evaluation"],
        "seeds": protocol["seeds"],
        "validated_result_sha256": {
            str(result["seed"]): result["deterministic_result_sha256"]
            for result in results
        },
        "descriptive_aggregate_recomputed": _aggregate(results),
        "views": view_statistics,
        "view_sensitivity": sensitivity,
    }


def summarize_campaign(
    input_root: str | Path,
    output_path: str | Path,
    *,
    expected_seeds: Sequence[int],
    expected_datasets: Sequence[str] | None = None,
) -> dict:
    root = Path(input_root).resolve()
    seeds = [int(seed) for seed in expected_seeds]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("formal paired aggregation requires exactly five unique seeds")
    protocol_paths = sorted(root.rglob("protocol.json"))
    if not protocol_paths:
        raise FileNotFoundError(f"no protocol.json found below {root}")
    datasets: dict[str, dict] = {}
    config_hash: str | None = None
    source_hash: str | None = None
    environment_hash: str | None = None
    shared_view_order: list[str] | None = None
    for protocol_path in protocol_paths:
        protocol, results = _validate_dataset_directory(protocol_path, seeds)
        dataset = protocol["dataset"]
        if dataset in datasets:
            raise RuntimeError(f"duplicate dataset output directories for {dataset}")
        current_config = canonical_sha256(protocol["config"])
        current_source = protocol["source"]["manifest_sha256"]
        current_environment = canonical_sha256(protocol["environment"])
        current_view_order = list(protocol["evaluation"]["view_order"])
        if config_hash is None:
            config_hash = current_config
            source_hash = current_source
            environment_hash = current_environment
            shared_view_order = current_view_order
        elif (
            current_config != config_hash
            or current_source != source_hash
            or current_environment != environment_hash
            or current_view_order != shared_view_order
        ):
            raise RuntimeError(
                "campaign protocols do not share identical config, source, environment, "
                "and evaluation-view order"
            )
        datasets[dataset] = _dataset_statistics(protocol, results, protocol_path.parent)
    if expected_datasets is not None and set(datasets) != set(expected_datasets):
        raise RuntimeError(
            f"dataset contract mismatch: {sorted(datasets)} != {sorted(expected_datasets)}"
        )
    report = {
        "schema_version": 2,
        "report": "ofra_streaming_full_paired_campaign_v2_masked_views",
        "input_root": str(root),
        "expected_seeds": seeds,
        "datasets": datasets,
        "comparison_registry": {
            "primary_m8": {
                name: {"reference_arm": arms[0], "treatment_arm": arms[1]}
                for name, arms in PRIMARY_COMPARISONS.items()
            },
            "exposure_prior_diagnostic_m3": {
                name: {"reference_arm": arms[0], "treatment_arm": arms[1]}
                for name, arms in DIAGNOSTIC_COMPARISONS.items()
            },
            "view_sensitivity": {
                "primary_arms_m5": list(PRIMARY_ARMS),
                "diagnostic_arms_m3": list(DIAGNOSTIC_ARMS),
            },
        },
        "metric_registry": METRICS,
        "inference_policy": {
            "delta": "treatment_minus_reference",
            "confidence_interval": "two-sided 95% Student-t CI on paired deltas, df=4",
            "paired_t": "two-sided one-sample t-test of paired deltas against zero",
            "wilcoxon": "two-sided exact sign enumeration; zero deltas discarded",
            "n5_wilcoxon_note": (
                "With five non-zero pairs, the smallest attainable two-sided exact "
                "Wilcoxon p-value is 0.0625."
            ),
            "holm": (
                "separate pre-registered Holm families: within-view primary m=8; "
                "within-view exposure-prior diagnostic m=3; retained-vs-official "
                "primary arms m=5; retained-vs-official diagnostic arms m=3"
            ),
        },
        "shared_config_sha256": config_hash,
        "shared_source_manifest_sha256": source_hash,
        "shared_environment_sha256": environment_hash,
        "shared_evaluation_view_order": shared_view_order,
    }
    report["campaign_report_sha256"] = canonical_sha256(report)
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and aggregate five-seed streaming_full result JSON files."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, nargs=5, required=True)
    parser.add_argument("--expected-datasets", nargs="+")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = summarize_campaign(
        args.input_root,
        args.output,
        expected_seeds=args.expected_seeds,
        expected_datasets=args.expected_datasets,
    )
    print(
        f"wrote {args.output.resolve()} "
        f"sha256={report['campaign_report_sha256']} datasets={len(report['datasets'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
