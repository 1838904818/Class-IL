#!/usr/bin/env python3
"""Verify and aggregate five hash-bound CPU-reconstruction attribution/ETG artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from itertools import combinations
from pathlib import Path
from typing import Any

from formal_v2_explanation_etg.attribution_scope import (
    PREDICTIVE_METRIC_KEYS,
    attribution_scope_contract,
    validate_attribution_scope,
)


EXPECTED_SEEDS = (1, 2, 3, 4, 42)
METHODS = ("expected_gradients", "feature_ablation", "gradient_x_input")
SCHEMAS = {"ofra_attribution_robustness_v3"}
OUTPUT_SCHEMA = "ofra_attribution_robustness_five_seed_v2"
T975_DF4 = 2.7764451051977987


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def jaccard(left: list[int], right: list[int]) -> float:
    a, b = set(map(int, left)), set(map(int, right))
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _row_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["checkpoint"]), int(row["class_id"])


def _transition_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return int(row["from_checkpoint"]), int(row["to_checkpoint"]), int(row["class_id"])


def _maps(
    artifact: dict[str, Any], seed: int
) -> tuple[dict[str, dict[tuple[int, int], dict[str, Any]]], dict[str, dict[tuple[int, int, int], dict[str, Any]]]]:
    row_maps: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    transition_maps: dict[str, dict[tuple[int, int, int], dict[str, Any]]] = {}
    for method in METHODS:
        rows = artifact.get("checkpoint_rows", {}).get(method)
        transitions = artifact.get("transition_rows", {}).get(method)
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(f"seed {seed}: missing checkpoint rows for {method}")
        if not isinstance(transitions, list) or not transitions:
            raise RuntimeError(f"seed {seed}: missing transition rows for {method}")
        if any(int(row.get("seed", -1)) != seed for row in rows):
            raise RuntimeError(f"seed {seed}: checkpoint row scope mismatch for {method}")
        row_map = {_row_key(row): row for row in rows}
        transition_map = {_transition_key(row): row for row in transitions}
        if len(row_map) != len(rows) or len(transition_map) != len(transitions):
            raise RuntimeError(f"seed {seed}: duplicate evidence key for {method}")
        row_maps[method] = row_map
        transition_maps[method] = transition_map
    row_keys = set(row_maps[METHODS[0]])
    transition_keys = set(transition_maps[METHODS[0]])
    if any(set(row_maps[m]) != row_keys for m in METHODS[1:]):
        raise RuntimeError(f"seed {seed}: methods do not share checkpoint/class scope")
    if any(set(transition_maps[m]) != transition_keys for m in METHODS[1:]):
        raise RuntimeError(f"seed {seed}: methods do not share transition scope")
    return row_maps, transition_maps


def compute_agreement(artifact: dict[str, Any], seed: int) -> dict[str, Any]:
    row_maps, transition_maps = _maps(artifact, seed)
    row_keys = sorted(row_maps[METHODS[0]])
    transition_keys = sorted(transition_maps[METHODS[0]])
    pairs: dict[str, Any] = {}
    for left, right in combinations(METHODS, 2):
        overlaps = [
            jaccard(row_maps[left][key]["top15_indices"], row_maps[right][key]["top15_indices"])
            for key in row_keys
        ]
        pairs[f"{left}__vs__{right}"] = {
            "row_count": len(row_keys),
            "top15_jaccard_mean": statistics.fmean(overlaps),
            "top15_jaccard_median": statistics.median(overlaps),
            "top15_jaccard_min": min(overlaps),
            "admission_decision_agreement": statistics.fmean(
                row_maps[left][key]["admitted"] == row_maps[right][key]["admitted"]
                for key in row_keys
            ),
            "etg_state_agreement": statistics.fmean(
                row_maps[left][key]["etg_state"] == row_maps[right][key]["etg_state"]
                for key in row_keys
            ),
            "silent_drift_event_agreement": statistics.fmean(
                transition_maps[left][key]["primary_event"]
                == transition_maps[right][key]["primary_event"]
                for key in transition_keys
            ),
        }
    return {
        "common_checkpoint_class_rows": len(row_keys),
        "common_adjacent_class_transitions": len(transition_keys),
        "pairwise": pairs,
        "all_method_admission_agreement": statistics.fmean(
            len({row_maps[m][key]["admitted"] for m in METHODS}) == 1 for key in row_keys
        ),
        "all_method_etg_state_agreement": statistics.fmean(
            len({row_maps[m][key]["etg_state"] for m in METHODS}) == 1 for key in row_keys
        ),
        "all_method_silent_drift_conclusion_agreement": statistics.fmean(
            len({transition_maps[m][key]["primary_event"] for m in METHODS}) == 1
            for key in transition_keys
        ),
    }


def compute_method_summaries(artifact: dict[str, Any], seed: int) -> dict[str, Any]:
    row_maps, transition_maps = _maps(artifact, seed)
    output: dict[str, Any] = {}
    for method in METHODS:
        rows = list(row_maps[method].values())
        transitions = list(transition_maps[method].values())
        actions = [str(row["etg_action"]) for row in rows]
        output[method] = {
            "checkpoint_class_rows": len(rows),
            "admitted_rows": sum(bool(row["admitted"]) for row in rows),
            "silent_drift_events": sum(bool(row["primary_event"]) for row in transitions),
            "eligible_transitions": sum(bool(row["primary_eligible"]) for row in transitions),
            "certified_admissions": sum(action == "admission_certified" for action in actions),
            "refused_admissions": sum(action.startswith("admission_refused") for action in actions),
            "escalations": sum(action == "human_review_escalation" for action in actions),
            "strict_recertifications": sum(action == "strict_recertified" for action in actions),
            "strict_recertification_failures": sum(
                action.startswith("strict_recertification_failed") for action in actions
            ),
        }
    return output


def _assert_close(actual: Any, stored: Any, context: str) -> None:
    if isinstance(actual, dict):
        if not isinstance(stored, dict):
            raise RuntimeError(f"{context}: stored value is not an object")
        for key, value in actual.items():
            if key not in stored:
                raise RuntimeError(f"{context}: missing stored field {key}")
            _assert_close(value, stored[key], f"{context}.{key}")
    elif isinstance(actual, float):
        if not math.isclose(actual, float(stored), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"{context}: recomputation mismatch")
    elif actual != stored:
        raise RuntimeError(f"{context}: recomputation mismatch")


def validate_artifact(artifact: dict[str, Any], seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if artifact.get("schema_version") not in SCHEMAS:
        raise RuntimeError(f"seed {seed}: unsupported schema")
    stored_hash = artifact.get("canonical_sha256")
    payload = {key: value for key, value in artifact.items() if key != "canonical_sha256"}
    if stored_hash != canonical_sha256(payload):
        raise RuntimeError(f"seed {seed}: canonical hash mismatch")
    if int(artifact.get("seed", -1)) != seed:
        raise RuntimeError(f"seed {seed}: artifact seed mismatch")
    if artifact.get("dataset") != "malaya-network-gt":
        raise RuntimeError(f"seed {seed}: dataset mismatch")
    if artifact.get("score_target") != "joint_cap3000 class margin":
        raise RuntimeError(f"seed {seed}: score target mismatch")
    if artifact.get("status") != "completed_cpu_reconstruction_single_seed_analysis":
        raise RuntimeError(f"seed {seed}: completion status widens the attribution scope")
    validate_attribution_scope(artifact.get("attribution_scope"), context=f"seed {seed}")
    predictive = artifact.get("predictive_metrics")
    if not isinstance(predictive, dict):
        raise RuntimeError(f"seed {seed}: predictive metrics missing")
    expected_predictive_keys = {"evaluation_view", "arm", *PREDICTIVE_METRIC_KEYS}
    if set(predictive) != expected_predictive_keys:
        raise RuntimeError(f"seed {seed}: predictive metric registry mismatch")
    if predictive.get("evaluation_view") != "official" or predictive.get("arm") != "joint_cap3000":
        raise RuntimeError(f"seed {seed}: predictive metric scope mismatch")
    for key in PREDICTIVE_METRIC_KEYS:
        value = float(predictive[key])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise RuntimeError(f"seed {seed}: invalid predictive metric {key}")
    agreement = compute_agreement(artifact, seed)
    summaries = compute_method_summaries(artifact, seed)
    _assert_close(agreement, artifact.get("agreement"), f"seed {seed}.agreement")
    _assert_close(summaries, artifact.get("method_summaries"), f"seed {seed}.method_summaries")
    return agreement, summaries


def describe(values: list[float]) -> dict[str, float | int]:
    if len(values) != len(EXPECTED_SEEDS):
        raise RuntimeError("five-seed descriptive interval requires exactly five values")
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    half = T975_DF4 * sd / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": mean,
        "sample_sd": sd,
        "minimum": min(values),
        "maximum": max(values),
        "t95_ci_lower": max(0.0, mean - half),
        "t95_ci_upper": min(1.0, mean + half),
    }


def aggregate(paths: dict[int, Path]) -> dict[str, Any]:
    if tuple(sorted(paths)) != EXPECTED_SEEDS:
        raise RuntimeError(f"artifacts must be exactly seeds {list(EXPECTED_SEEDS)}")
    artifacts: dict[int, dict[str, Any]] = {}
    agreements: dict[int, dict[str, Any]] = {}
    summaries: dict[int, dict[str, Any]] = {}
    bindings: dict[str, Any] = {}
    threshold_fingerprint: str | None = None
    for seed in EXPECTED_SEEDS:
        path = paths[seed].resolve()
        artifact = load_json(path)
        agreement, method_summary = validate_artifact(artifact, seed)
        fingerprint = canonical_sha256(artifact["thresholds"])
        if threshold_fingerprint is None:
            threshold_fingerprint = fingerprint
        elif fingerprint != threshold_fingerprint:
            raise RuntimeError("seed artifacts do not share identical thresholds")
        artifacts[seed] = artifact
        agreements[seed] = agreement
        summaries[seed] = method_summary
        bindings[str(seed)] = {
            "path": str(path),
            "file_sha256": file_sha256(path),
            "canonical_sha256": artifact["canonical_sha256"],
            "schema_version": artifact["schema_version"],
        }

    pairwise: dict[str, Any] = {}
    for pair in agreements[EXPECTED_SEEDS[0]]["pairwise"]:
        pairwise[pair] = {}
        for metric in (
            "top15_jaccard_mean",
            "admission_decision_agreement",
            "etg_state_agreement",
            "silent_drift_event_agreement",
        ):
            pairwise[pair][metric] = describe(
                [float(agreements[seed]["pairwise"][pair][metric]) for seed in EXPECTED_SEEDS]
            )

    all_method = {
        metric: describe([float(agreements[seed][metric]) for seed in EXPECTED_SEEDS])
        for metric in (
            "all_method_admission_agreement",
            "all_method_etg_state_agreement",
            "all_method_silent_drift_conclusion_agreement",
        )
    }
    method_results: dict[str, Any] = {}
    for method in METHODS:
        seed_rows = []
        for seed in EXPECTED_SEEDS:
            record = dict(summaries[seed][method])
            record["seed"] = seed
            eligible = int(record["eligible_transitions"])
            record["silent_drift_rate"] = (
                float(record["silent_drift_events"]) / eligible if eligible else None
            )
            seed_rows.append(record)
        rates = [float(row["silent_drift_rate"]) for row in seed_rows]
        method_results[method] = {
            "per_seed": seed_rows,
            "silent_drift_rate_seed_summary": describe(rates),
            "pooled_counts": {
                key: sum(int(row[key]) for row in seed_rows)
                for key in summaries[EXPECTED_SEEDS[0]][method]
            },
        }

    predictive_rows = [
        {"seed": seed, **dict(artifacts[seed]["predictive_metrics"])}
        for seed in EXPECTED_SEEDS
    ]
    predictive_statistics = {
        key: describe([float(row[key]) for row in predictive_rows])
        for key in PREDICTIVE_METRIC_KEYS
    }

    result = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "completed_hash_verified_five_seed_robustness",
        "dataset": "malaya-network-gt",
        "seeds": list(EXPECTED_SEEDS),
        "score_target": "joint_cap3000 class margin",
        "attribution_scope": attribution_scope_contract(),
        "methods": list(METHODS),
        "threshold_fingerprint": threshold_fingerprint,
        "artifact_bindings": bindings,
        "pairwise_seed_statistics": pairwise,
        "all_method_seed_statistics": all_method,
        "method_conditioned_results": method_results,
        "predictive_performance": {
            "evaluation_view": "official",
            "arm": "joint_cap3000",
            "per_seed": predictive_rows,
            "seed_statistics": predictive_statistics,
        },
        "interpretation_guardrails": [
            "The five seeds share one fixed data split and are repeated stochastic runs, not five independent datasets.",
            "The analysis measures attribution-method dependence; it does not identify a uniquely correct explainer.",
            "ETG is an offline post-hoc non-suppressing ledger and does not alter OFRA training, routing, or predictions.",
            "The t intervals are descriptive seed-level uncertainty summaries with n=5, not evidence of universal generalisation.",
        ],
    }
    result["canonical_sha256"] = canonical_sha256(result)
    return result


def parse_artifacts(values: list[str]) -> dict[int, Path]:
    output: dict[int, Path] = {}
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator:
            raise RuntimeError("artifact must use SEED=PATH")
        seed = int(seed_text)
        if seed in output:
            raise RuntimeError(f"duplicate artifact seed {seed}")
        output[seed] = Path(path_text)
    return output


def write_csv(result: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for method, block in result["method_conditioned_results"].items():
        for record in block["per_seed"]:
            rows.append({"method": method, **record})
    fields = [
        "method", "seed", "checkpoint_class_rows", "admitted_rows", "silent_drift_events",
        "eligible_transitions", "silent_drift_rate", "certified_admissions", "refused_admissions",
        "escalations", "strict_recertifications", "strict_recertification_failures",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", required=True, help="SEED=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(parse_artifacts(args.artifact))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "attribution_robustness_five_seed.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(result, args.output_dir / "attribution_robustness_five_seed_method_counts.csv")
    print(json.dumps({"status": result["status"], "canonical_sha256": result["canonical_sha256"]}))


if __name__ == "__main__":
    main()
