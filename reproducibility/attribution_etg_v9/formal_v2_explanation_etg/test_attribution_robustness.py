import copy
import unittest

from formal_v2_explanation_etg.attribution_scope import (
    attribution_scope_contract,
    extract_predictive_metrics,
    validate_declared_scope,
)
from formal_v2_explanation_etg.attribution_robustness import build_secondary_profile_record


class DeclaredScopeTests(unittest.TestCase):
    def setUp(self):
        self.result = {"seed": 2}
        self.protocol = {"seeds": [2]}
        self.analysis = {
            "dataset": "malaya-network-gt",
            "seed": 2,
            "checkpoint_rows": [{"seed": 2, "checkpoint": 0, "class_id": 0}],
            "attribution_scope": attribution_scope_contract(),
            "cpu_reload_vs_saved_gpu_numerical_audit": {"equivalence_claimed": False},
            "batch_partition_numerical_sensitivity": {"equivalence_claimed": False},
        }

    def validate(self, result=None, protocol=None, analysis=None):
        validate_declared_scope(
            result or self.result,
            protocol or self.protocol,
            analysis or self.analysis,
            seed=2,
            dataset="malaya-network-gt",
        )

    def test_matching_scope_passes(self):
        self.validate()

    def test_training_result_seed_mismatch_fails(self):
        with self.assertRaisesRegex(RuntimeError, "training result/declared seed"):
            self.validate(result={"seed": 1})

    def test_protocol_seed_mismatch_fails(self):
        with self.assertRaisesRegex(RuntimeError, "absent from the training protocol"):
            self.validate(protocol={"seeds": [1]})

    def test_expected_analysis_seed_mismatch_fails(self):
        analysis = copy.deepcopy(self.analysis)
        analysis["seed"] = 1
        with self.assertRaisesRegex(RuntimeError, "analysis seed mismatch"):
            self.validate(analysis=analysis)

    def test_expected_row_seed_mismatch_fails(self):
        analysis = copy.deepcopy(self.analysis)
        analysis["checkpoint_rows"][0]["seed"] = 1
        with self.assertRaisesRegex(RuntimeError, "checkpoint row seed mismatch"):
            self.validate(analysis=analysis)

    def test_dataset_mismatch_fails(self):
        analysis = copy.deepcopy(self.analysis)
        analysis["dataset"] = "wrong-dataset"
        with self.assertRaisesRegex(RuntimeError, "dataset mismatch"):
            self.validate(analysis=analysis)

    def test_missing_checkpoint_rows_fails(self):
        analysis = copy.deepcopy(self.analysis)
        analysis["checkpoint_rows"] = []
        with self.assertRaisesRegex(RuntimeError, "checkpoint rows are missing"):
            self.validate(analysis=analysis)

    def test_cross_device_equivalence_claim_fails(self):
        analysis = copy.deepcopy(self.analysis)
        analysis["attribution_scope"]["cross_device_equivalence_claimed"] = True
        analysis["cpu_reload_vs_saved_gpu_numerical_audit"]["equivalence_claimed"] = True
        with self.assertRaisesRegex(RuntimeError, "equivalence claim"):
            self.validate(analysis=analysis)

    def test_batch_partition_equivalence_claim_fails(self):
        analysis = copy.deepcopy(self.analysis)
        analysis["attribution_scope"]["batch_partition_equivalence_claimed"] = True
        analysis["batch_partition_numerical_sensitivity"]["equivalence_claimed"] = True
        with self.assertRaisesRegex(RuntimeError, "equivalence claim"):
            self.validate(analysis=analysis)

    def test_extracts_only_official_joint_cap3000_metrics(self):
        result = {
            "summary": {"views": {"official": {"joint_cap3000": {
                "average_task_accuracy": 0.31,
                "average_forgetting": 0.04,
                "final_overall_accuracy": 0.55,
                "final_macro_f1": 0.21,
                "final_balanced_accuracy": 0.24,
            }}}}
        }
        metrics = extract_predictive_metrics(result)
        self.assertEqual(metrics["evaluation_view"], "official")
        self.assertEqual(metrics["arm"], "joint_cap3000")
        self.assertEqual(metrics["final_macro_f1"], 0.21)

    def test_secondary_profile_excludes_scientific_results(self):
        record = build_secondary_profile_record(
            dataset="malaya-network-gt",
            seed=1,
            source_training_job=388991,
            row_counts={"feature_ablation": 30, "gradient_x_input": 30},
            source_bindings={"training_result_file_sha256": "a" * 64},
        )
        self.assertEqual(record["status"], "completed_secondary_profile_only")
        self.assertFalse(record["scientific_use_allowed"])
        self.assertFalse(record["contains_attribution_values"])
        self.assertFalse(record["contains_agreement_results"])
        self.assertFalse(record["contains_etg_results"])
        self.assertNotIn("agreement", record)
        self.assertNotIn("checkpoint_rows", record)
        self.assertNotIn("transition_rows", record)


if __name__ == "__main__":
    unittest.main()
