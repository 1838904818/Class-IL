"""Independently recompute the complete five-seed aggregate's arithmetic."""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path

from validate_semantics import METHODS, SEEDS, THRESHOLDS, canonical, checked_json, finite, keyed, overlap, require

PREDICTIVE = ("average_task_accuracy", "average_forgetting", "final_overall_accuracy", "final_macro_f1", "final_balanced_accuracy")
PAIR_METRICS = ("top15_jaccard_mean", "admission_decision_agreement", "etg_state_agreement", "silent_drift_event_agreement")
ALL_METRICS = ("all_method_admission_agreement", "all_method_etg_state_agreement", "all_method_silent_drift_conclusion_agreement")


def describe(values):
    require(len(values) == 5, "Exactly five seed values are required")
    require(all(value is not None for value in values), "Undefined seed rate: an explicit missingness policy is required")
    values = [finite(value, "seed statistic") for value in values]
    require(all(0 <= value <= 1 for value in values), "Seed statistic outside [0,1]")
    mean = sum(values) / 5
    sd = math.sqrt(sum((value-mean)**2 for value in values) / 4)
    half = 2.7764451051977987 * sd / math.sqrt(5)
    return {"n": 5, "mean": mean, "sample_sd": sd, "minimum": min(values), "maximum": max(values),
            "t95_ci_lower": max(0.0, mean-half), "t95_ci_upper": min(1.0, mean+half)}


def match(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"Field registry mismatch: {label}")
        for key in expected:
            match(actual[key], expected[key], label+"."+key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"Row count mismatch: {label}")
        for index, value in enumerate(expected):
            match(actual[index], value, f"{label}[{index}]")
    elif isinstance(expected, float):
        require(math.isclose(finite(actual, label), expected, rel_tol=0, abs_tol=1e-12), f"Aggregate arithmetic mismatch: {label}")
    else:
        require(type(actual) is type(expected) and actual == expected, f"Aggregate value mismatch: {label}")


def seed_counts(artifact, seed):
    require(artifact.get("seed") == seed and artifact.get("dataset") == "malaya-network-gt", "Seed/dataset mismatch")
    require(artifact.get("status") == "completed_cpu_reconstruction_single_seed_analysis", "Single-seed status mismatch")
    require(artifact.get("score_target") == "joint_cap3000 class margin", "Score target mismatch")
    require(artifact.get("thresholds") == THRESHOLDS, "Frozen thresholds mismatch")
    require(artifact.get("canonical_sha256") == canonical({k: v for k, v in artifact.items() if k != "canonical_sha256"}), "Artifact canonical hash mismatch")
    row_keys = {(t, c) for t in range(5) for c in range(2*(t+1))}
    transition_keys = {(t-1, t, c) for t in range(1, 5) for c in range(2*t)}
    rows, transitions, counts = {}, {}, {}
    for method in METHODS:
        rows[method] = keyed(artifact["checkpoint_rows"][method], ("checkpoint", "class_id"), row_keys, method)
        transitions[method] = keyed(artifact["transition_rows"][method], ("from_checkpoint", "to_checkpoint", "class_id"), transition_keys, method)
        for row in rows[method].values():
            require(type(row["admitted"]) is bool and row["seed"] == seed, "Invalid admission/seed")
        for row in transitions[method].values():
            require(type(row["primary_event"]) is bool and type(row["primary_eligible"]) is bool, "Invalid drift flags")
            require(not row["primary_event"] or row["primary_eligible"], "Event outside eligible cohort")
        actions = [row["etg_action"] for row in rows[method].values()]
        counts[method] = {
            "checkpoint_class_rows": len(row_keys),
            "admitted_rows": sum(row["admitted"] for row in rows[method].values()),
            "silent_drift_events": sum(row["primary_event"] for row in transitions[method].values()),
            "eligible_transitions": sum(row["primary_eligible"] for row in transitions[method].values()),
            "certified_admissions": actions.count("admission_certified"),
            "refused_admissions": sum(action.startswith("admission_refused") for action in actions),
            "escalations": actions.count("human_review_escalation"),
            "strict_recertifications": actions.count("strict_recertified"),
            "strict_recertification_failures": sum(action.startswith("strict_recertification_failed") for action in actions),
        }
    pairs = {}
    for left, right in combinations(METHODS, 2):
        pairs[left+"__vs__"+right] = {
            "top15_jaccard_mean": sum(overlap(rows[left][k]["top15_indices"], rows[right][k]["top15_indices"]) for k in row_keys)/len(row_keys),
            "admission_decision_agreement": sum(rows[left][k]["admitted"] == rows[right][k]["admitted"] for k in row_keys)/len(row_keys),
            "etg_state_agreement": sum(rows[left][k]["etg_state"] == rows[right][k]["etg_state"] for k in row_keys)/len(row_keys),
            "silent_drift_event_agreement": sum(transitions[left][k]["primary_event"] == transitions[right][k]["primary_event"] for k in transition_keys)/len(transition_keys),
        }
    all_methods = {
        ALL_METRICS[0]: sum(len({rows[m][k]["admitted"] for m in METHODS}) == 1 for k in row_keys)/len(row_keys),
        ALL_METRICS[1]: sum(len({rows[m][k]["etg_state"] for m in METHODS}) == 1 for k in row_keys)/len(row_keys),
        ALL_METRICS[2]: sum(len({transitions[m][k]["primary_event"] for m in METHODS}) == 1 for k in transition_keys)/len(transition_keys),
    }
    return counts, pairs, all_methods


def verify(aggregate, artifacts, file_hashes):
    require(tuple(sorted(artifacts)) == tuple(sorted(file_hashes)) == SEEDS, "Exactly registered seeds 1,2,3,4,42 are required")
    require(aggregate.get("schema_version") == "ofra_attribution_robustness_five_seed_v2", "Aggregate schema mismatch")
    require(aggregate.get("status") == "completed_hash_verified_five_seed_robustness", "Aggregate completion label mismatch")
    require(aggregate.get("seeds") == list(SEEDS) and aggregate.get("methods") == list(METHODS), "Aggregate scope mismatch")
    require(aggregate.get("dataset") == "malaya-network-gt" and aggregate.get("score_target") == "joint_cap3000 class margin", "Aggregate dataset/score mismatch")
    require(aggregate.get("canonical_sha256") == canonical({k: v for k, v in aggregate.items() if k != "canonical_sha256"}), "Aggregate canonical hash mismatch")
    require(aggregate.get("threshold_fingerprint") == canonical(THRESHOLDS), "Threshold fingerprint mismatch")
    require(set(aggregate["artifact_bindings"]) == {str(seed) for seed in SEEDS}, "Aggregate input registry mismatch")
    stats = {}
    for seed in SEEDS:
        artifact = artifacts[seed]
        binding = aggregate["artifact_bindings"][str(seed)]
        require(binding["file_sha256"] == file_hashes[seed] and binding["canonical_sha256"] == artifact["canonical_sha256"], "Aggregate-to-seed hash mismatch")
        require(binding["schema_version"] == artifact["schema_version"], "Seed schema binding mismatch")
        scope = artifact["attribution_scope"]
        require(scope.get("target") == "hash-bound checkpoint reconstructed on CPU and evaluated on each fixed true-class probe batch", "Attribution scope mismatch")
        require(scope.get("cross_device_equivalence_claimed") is False and scope.get("batch_partition_equivalence_claimed") is False, "Unsupported numerical equivalence claim")
        require(scope == aggregate["attribution_scope"], "Cross-seed attribution scope mismatch")
        for field in ("streaming_manifest_file_sha256", "feature_schema_file_sha256", "script_sha256"):
            require(artifact["source_bindings"][field] == artifacts[SEEDS[0]]["source_bindings"][field], "Cross-seed input/code binding mismatch")
        stats[seed] = seed_counts(artifact, seed)
    expected_pairwise = {pair: {metric: describe([stats[s][1][pair][metric] for s in SEEDS]) for metric in PAIR_METRICS} for pair in stats[SEEDS[0]][1]}
    match(aggregate["pairwise_seed_statistics"], expected_pairwise, "pairwise")
    match(aggregate["all_method_seed_statistics"], {metric: describe([stats[s][2][metric] for s in SEEDS]) for metric in ALL_METRICS}, "all-method")
    methods = {}
    for method in METHODS:
        rows = []
        for seed in SEEDS:
            count = stats[seed][0][method]
            eligible = count["eligible_transitions"]
            rows.append({**count, "seed": seed, "silent_drift_rate": count["silent_drift_events"]/eligible if eligible else None})
        methods[method] = {"per_seed": rows, "silent_drift_rate_seed_summary": describe([row["silent_drift_rate"] for row in rows]),
                           "pooled_counts": {key: sum(row[key] for row in rows) for key in stats[SEEDS[0]][0][method]}}
    match(aggregate["method_conditioned_results"], methods, "method-conditioned")
    predictive = []
    for seed in SEEDS:
        row = artifacts[seed]["predictive_metrics"]
        require(set(row) == {*PREDICTIVE, "evaluation_view", "arm"}, "Predictive field registry mismatch")
        require(row["evaluation_view"] == "official" and row["arm"] == "joint_cap3000", "Predictive scope mismatch")
        predictive.append({"seed": seed, **row})
    match(aggregate["predictive_performance"], {"evaluation_view": "official", "arm": "joint_cap3000", "per_seed": predictive,
            "seed_statistics": {key: describe([row[key] for row in predictive]) for key in PREDICTIVE}}, "predictive")
    return {"status": "five_seed_aggregate_arithmetic_verified_not_publication_approval", "seeds": list(SEEDS),
            "replication_unit": "seed on one fixed split", "descriptive_interval": "Student t, df=4; displayed bounds clipped to [0,1] as in registered aggregate",
            "limitations": ["Not an explainer accuracy test or evidence of causal architecture benefit.",
                            "Requires separate all-seed semantic, array, source-provenance, and W&B checks.",
                            "Within-seed class transitions are not independent seed replicates."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--aggregate-sha256", required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="JSON list of exactly five seed/path/sha256 records")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.inputs.read_text())
    require(isinstance(registry, list) and len(registry) == 5 and all(type(r.get("seed")) is int for r in registry), "Exactly five input records required")
    require(sorted(r["seed"] for r in registry) == list(SEEDS), "Missing or duplicate seed")
    artifacts, hashes = {}, {}
    for row in registry:
        path = args.inputs.parent / row["path"]
        artifacts[row["seed"]] = checked_json(path, row["sha256"])
        hashes[row["seed"]] = row["sha256"]
    output = verify(checked_json(args.aggregate, args.aggregate_sha256), artifacts, hashes)
    output["aggregate_file_sha256"] = args.aggregate_sha256
    output["artifact_file_sha256"] = {str(seed): hashes[seed] for seed in SEEDS}
    output["validator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output["canonical_sha256"] = canonical(output)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("AGGREGATE_ARITHMETIC=PASS; not full publication approval")


if __name__ == "__main__":
    main()
