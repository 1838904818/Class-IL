"""Independent offline publication checks; never modify the v9 campaign."""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path

METHODS = ("expected_gradients", "feature_ablation", "gradient_x_input")
SEEDS = (1, 2, 3, 4, 42)
THRESHOLDS = {
    "top_k": 15, "silent_drift_jaccard": 0.7, "allowed_recall_drop": 0.05,
    "admission": "selected-feature deletion mass exceeds the same fixed random-control 95th percentile used by Expected Gradients",
}


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value), f"Invalid number: {label}")
    return float(value)


def close(actual, expected, label):
    require(math.isclose(finite(actual, label), expected, rel_tol=0, abs_tol=1e-12),
            f"Recomputation mismatch: {label}")


def flag(actual, expected, label):
    require(type(actual) is bool and actual is expected, f"Invalid decision: {label}")


def overlap(left, right):
    a, b = set(left), set(right)
    return len(a & b) / len(a | b)


def rate(events, eligible):
    require(type(events) is int and type(eligible) is int and 0 <= events <= eligible,
            "Invalid event denominator")
    return events / eligible if eligible else None


def keyed(rows, keys, expected, label):
    require(isinstance(rows, list), f"Missing rows: {label}")
    mapping = {}
    for row in rows:
        require(isinstance(row, dict), f"Invalid row: {label}")
        key = tuple(row.get(k) for k in keys)
        require(all(type(item) is int for item in key), f"Invalid row key: {label}")
        require(key not in mapping, f"Duplicate row: {label}")
        mapping[key] = row
    require(set(mapping) == expected, f"Incomplete source-bound coverage: {label}")
    return mapping


def validate(artifact, training, features):
    seed = training["seed"]
    require(type(seed) is int and seed in SEEDS and artifact.get("seed") == seed, "Seed mismatch")
    require(artifact.get("schema_version") == "ofra_attribution_robustness_v3", "Schema mismatch")
    require(artifact.get("status") == "completed_cpu_reconstruction_single_seed_analysis", "Status mismatch")
    require(all(item.get("dataset") == "malaya-network-gt" for item in (artifact, training, features)), "Dataset mismatch")
    require(artifact.get("score_target") == "joint_cap3000 class margin", "Score target mismatch")
    scope = artifact.get("attribution_scope", {})
    require(scope.get("target") == "hash-bound checkpoint reconstructed on CPU and evaluated on each fixed true-class probe batch", "Attribution target scope mismatch")
    flag(scope.get("cross_device_equivalence_claimed"), False, "cross-device scope")
    flag(scope.get("batch_partition_equivalence_claimed"), False, "batch scope")
    require(artifact.get("canonical_sha256") == canonical({k: v for k, v in artifact.items() if k != "canonical_sha256"}), "Canonical hash mismatch")
    require(artifact.get("thresholds") == THRESHOLDS, "Frozen threshold mismatch")
    names = features["feature_columns"]
    require(features["feature_count"] == len(names) == 77 and len(set(names)) == 77, "Feature registry mismatch")
    checkpoints = training["checkpoints"]
    require(len(checkpoints) == 5, "Expected five source checkpoints")
    expected_rows, expected_transitions, recalls = set(), set(), {}
    for index, checkpoint in enumerate(checkpoints):
        require(checkpoint["checkpoint"] == index, "Checkpoint index mismatch")
        require(checkpoint["seen_classes"] == list(range(2 * (index + 1))), "Registered class stream mismatch")
        for class_id in checkpoint["seen_classes"]:
            expected_rows.add((index, class_id))
        per_class = checkpoint["views"]["official"]["arms"]["joint_cap3000"]["per_class"]
        source = keyed(per_class, ("class_id",), {(c,) for c in checkpoint["seen_classes"]}, "training recall")
        for (class_id,), row in source.items():
            recalls[(index, class_id)] = finite(row["recall"], "training recall")
        if index:
            expected_transitions.update((index-1, index, c) for c in checkpoints[index-1]["seen_classes"])
    predictive = artifact["predictive_metrics"]
    require(predictive.get("evaluation_view") == "official" and predictive.get("arm") == "joint_cap3000", "Predictive metric scope mismatch")
    for key in ("average_task_accuracy", "average_forgetting", "final_overall_accuracy", "final_macro_f1", "final_balanced_accuracy"):
        close(predictive[key], training["summary"]["views"]["official"]["joint_cap3000"][key], key)

    rows_by_method, transitions_by_method, summaries = {}, {}, {}
    for method in METHODS:
        rows = keyed(artifact["checkpoint_rows"][method], ("checkpoint", "class_id"), expected_rows, method)
        transitions = keyed(artifact["transition_rows"][method], ("from_checkpoint", "to_checkpoint", "class_id"), expected_transitions, method)
        for key, row in rows.items():
            require(type(row.get("seed")) is int and row["seed"] == seed and row.get("dataset") == "malaya-network-gt", "Row scope mismatch")
            top = row["top15_indices"]
            require(len(top) == len(set(top)) == 15 and all(type(i) is int and 0 <= i < 77 for i in top), "Invalid top-15 feature set")
            require(row["top15_features"] == [names[i] for i in top], "Feature labels mismatch")
            require(row["class_name"] == features["class_order"][key[1]], "Class label mismatch")
            require(type(row["probe_rows"]) is int and 0 < row["probe_rows"] <= 128, "Invalid probe count")
            require(type(row["background_rows"]) is int and row["background_rows"] > 0, "Invalid background count")
            close(row["recall"], recalls[key], "official class recall")
            mass, null = finite(row["rationale_mass"], "mass"), finite(row["random_null_95"], "null")
            require(-1 <= mass <= 1 and -1 <= null <= 1, "Invalid deletion mass range")
            close(row["mass_margin"], mass-null, "mass margin")
            flag(row["admitted"], mass > null, "admission")
        for (before, after, class_id), row in transitions.items():
            left, right = rows[(before, class_id)], rows[(after, class_id)]
            delta = right["recall"] - left["recall"]
            jaccard = overlap(left["top15_indices"], right["top15_indices"])
            close(row["recall_before"], left["recall"], "recall before")
            close(row["recall_after"], right["recall"], "recall after")
            close(row["delta_recall"], delta, "recall delta")
            close(row["jaccard_top15"], jaccard, "adjacent Jaccard")
            flag(row["primary_eligible"], delta > -0.05, "eligibility")
            flag(row["primary_event"], delta > -0.05 and jaccard < 0.7, "silent drift")
        # Independently replay the state machine from admission and drift facts.
        for class_id in range(10):
            state, reference = "UNCERTIFIED", None
            for checkpoint, _ in sorted(k for k in rows if k[1] == class_id):
                row = rows[(checkpoint, class_id)]
                action = "monitor_no_change"
                if state == "UNCERTIFIED":
                    if row["admitted"]:
                        state, reference, action = "CERTIFIED_STABLE", row["top15_indices"], "admission_certified"
                    else:
                        state, action = "UNEXPLAINABLE", "admission_refused_explanation_alert_withheld"
                elif state == "DRIFTED":
                    if row["admitted"] and overlap(row["top15_indices"], reference) >= 0.7:
                        state, reference, action = "CERTIFIED_STABLE", row["top15_indices"], "strict_recertified"
                    else:
                        state, action = "UNEXPLAINABLE", "strict_recertification_failed_explanation_alert_withheld"
                elif state == "UNEXPLAINABLE":
                    action = "explanation_alert_withheld"
                elif transitions[(checkpoint-1, checkpoint, class_id)]["primary_event"]:
                    state, action = "DRIFTED", "human_review_escalation"
                require(row["etg_state"] == state and row["etg_action"] == action, "ETG ledger mismatch")
        events = sum(t["primary_event"] for t in transitions.values())
        eligible = sum(t["primary_eligible"] for t in transitions.values())
        for key, value in (("checkpoint_class_rows", len(rows)), ("admitted_rows", sum(r["admitted"] for r in rows.values())), ("silent_drift_events", events), ("eligible_transitions", eligible)):
            require(artifact["method_summaries"][method][key] == value, "Method count mismatch")
        summaries[method] = {"events": events, "eligible_transitions": eligible, "rate": rate(events, eligible)}
        rows_by_method[method], transitions_by_method[method] = rows, transitions

    for key in expected_rows:
        ref = rows_by_method[METHODS[0]][key]
        for method in METHODS[1:]:
            for field in ("recall", "random_null_95", "probe_rows", "background_rows"):
                close(rows_by_method[method][key][field], ref[field], "shared " + field)
    agreement = {}
    for left, right in combinations(METHODS, 2):
        eligible_keys = {key for key in expected_transitions if transitions_by_method[left][key]["primary_eligible"]}
        require(eligible_keys == {key for key in expected_transitions if transitions_by_method[right][key]["primary_eligible"]}, "Methods changed the eligible cohort")
        matching = lambda keys: sum(transitions_by_method[left][k]["primary_event"] == transitions_by_method[right][k]["primary_event"] for k in keys)
        agreement[left + "__vs__" + right] = {
            "all_transitions": {"agreements": matching(expected_transitions), "denominator": len(expected_transitions)},
            "eligible_transitions": {"agreements": matching(eligible_keys), "denominator": len(eligible_keys), "rate": rate(matching(eligible_keys), len(eligible_keys))},
        }
    return {"status": "semantic_checks_passed_single_seed_not_campaign_completion", "seed": seed,
            "checkpoint_class_rows_per_method": len(expected_rows), "transitions_per_method": len(expected_transitions),
            "method_rates": summaries, "drift_agreement_denominators": agreement,
            "limitations": ["No attribution recomputation or array-to-ranking verification.", "Probe and background identities require their separately verified manifests.", "Five-seed aggregation, source provenance, and predictive performance validation remain separate gates."]}


def checked_json(path, expected):
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, "Input file SHA-256 mismatch")
    return json.loads(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact", "training", "features"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {name: checked_json(getattr(args, name), getattr(args, name + "_sha256")) for name in ("artifact", "training", "features")}
    bindings = inputs["artifact"]["source_bindings"]
    require(bindings["training_result_file_sha256"] == args.training_sha256 and bindings["feature_schema_file_sha256"] == args.features_sha256, "Artifact input binding mismatch")
    output = validate(**inputs)
    output["input_sha256"] = {name: getattr(args, name + "_sha256") for name in inputs}
    output["validator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output["canonical_sha256"] = canonical(output)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("SEMANTIC_CHECKS=PASS; validation only, not campaign completion")


if __name__ == "__main__":
    main()
