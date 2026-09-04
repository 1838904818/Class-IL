import copy
import json
import tempfile
import unittest
from pathlib import Path

from formal_v2_explanation_etg.attribution_scope import attribution_scope_contract
from formal_v2_explanation_etg.aggregate_attribution_robustness import (
    EXPECTED_SEEDS,
    METHODS,
    aggregate,
    canonical_sha256,
    compute_agreement,
    compute_method_summaries,
    validate_artifact,
)


def artifact(seed: int) -> dict:
    rows = {}
    transitions = {}
    for index, method in enumerate(METHODS):
        rows[method] = [
            {
                "seed": seed, "checkpoint": 0, "class_id": 0,
                "top15_indices": [0, 1 + index], "admitted": index != 2,
                "etg_state": "certified" if index != 2 else "refused",
                "etg_action": "admission_certified" if index != 2 else "admission_refused_null",
            },
            {
                "seed": seed, "checkpoint": 1, "class_id": 0,
                "top15_indices": [0, 2], "admitted": True,
                "etg_state": "certified", "etg_action": "strict_recertified",
            },
        ]
        transitions[method] = [
            {
                "from_checkpoint": 0, "to_checkpoint": 1, "class_id": 0,
                "primary_event": index == 0, "primary_eligible": True,
            }
        ]
    value = {
        "schema_version": "ofra_attribution_robustness_v3",
        "status": "completed_cpu_reconstruction_single_seed_analysis",
        "dataset": "malaya-network-gt",
        "seed": seed,
        "score_target": "joint_cap3000 class margin",
        "attribution_scope": attribution_scope_contract(),
        "predictive_metrics": {
            "evaluation_view": "official",
            "arm": "joint_cap3000",
            "average_task_accuracy": 0.30 + seed / 1000,
            "average_forgetting": 0.04,
            "final_overall_accuracy": 0.55,
            "final_macro_f1": 0.21,
            "final_balanced_accuracy": 0.24,
        },
        "thresholds": {"top_k": 15, "silent_drift_jaccard": 0.7},
        "checkpoint_rows": rows,
        "transition_rows": transitions,
    }
    value["agreement"] = compute_agreement(value, seed)
    value["method_summaries"] = compute_method_summaries(value, seed)
    value["canonical_sha256"] = canonical_sha256(value)
    return value


class AggregateTests(unittest.TestCase):
    def test_five_seed_aggregate_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for seed in EXPECTED_SEEDS:
                path = Path(directory) / f"seed-{seed}.json"
                path.write_text(json.dumps(artifact(seed)), encoding="utf-8")
                paths[seed] = path
            result = aggregate(paths)
            self.assertEqual(result["seeds"], list(EXPECTED_SEEDS))
            self.assertEqual(result["status"], "completed_hash_verified_five_seed_robustness")
            self.assertEqual(result["attribution_scope"], attribution_scope_contract())
            self.assertEqual(result["predictive_performance"]["evaluation_view"], "official")
            self.assertEqual(len(result["predictive_performance"]["per_seed"]), 5)

    def test_missing_seed_fails(self):
        with self.assertRaisesRegex(RuntimeError, "exactly seeds"):
            aggregate({})

    def test_tampered_canonical_hash_fails(self):
        value = artifact(1)
        value["checkpoint_rows"][METHODS[0]][0]["admitted"] = False
        with self.assertRaisesRegex(RuntimeError, "canonical hash mismatch"):
            validate_artifact(value, 1)

    def test_tampered_stored_agreement_fails(self):
        value = artifact(1)
        value["agreement"]["all_method_admission_agreement"] = 0.123
        value["canonical_sha256"] = canonical_sha256(
            {key: item for key, item in value.items() if key != "canonical_sha256"}
        )
        with self.assertRaisesRegex(RuntimeError, "recomputation mismatch"):
            validate_artifact(value, 1)

    def test_cross_seed_row_fails(self):
        value = artifact(1)
        value["checkpoint_rows"][METHODS[1]][0]["seed"] = 2
        value["canonical_sha256"] = canonical_sha256(
            {key: item for key, item in value.items() if key != "canonical_sha256"}
        )
        with self.assertRaisesRegex(RuntimeError, "row scope mismatch"):
            validate_artifact(value, 1)

    def test_threshold_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for seed in EXPECTED_SEEDS:
                value = artifact(seed)
                if seed == 42:
                    value["thresholds"]["silent_drift_jaccard"] = 0.6
                    value["canonical_sha256"] = canonical_sha256(
                        {key: item for key, item in value.items() if key != "canonical_sha256"}
                    )
                path = Path(directory) / f"seed-{seed}.json"
                path.write_text(json.dumps(value), encoding="utf-8")
                paths[seed] = path
            with self.assertRaisesRegex(RuntimeError, "identical thresholds"):
                aggregate(paths)

    def test_equivalence_claim_fails(self):
        value = artifact(1)
        value["attribution_scope"]["cross_device_equivalence_claimed"] = True
        value["canonical_sha256"] = canonical_sha256(
            {key: item for key, item in value.items() if key != "canonical_sha256"}
        )
        with self.assertRaisesRegex(RuntimeError, "equivalence claim"):
            validate_artifact(value, 1)

    def test_predictive_metric_scope_fails(self):
        value = artifact(1)
        value["predictive_metrics"]["evaluation_view"] = "duplicate_excluded"
        value["canonical_sha256"] = canonical_sha256(
            {key: item for key, item in value.items() if key != "canonical_sha256"}
        )
        with self.assertRaisesRegex(RuntimeError, "predictive metric scope"):
            validate_artifact(value, 1)


if __name__ == "__main__":
    unittest.main()
