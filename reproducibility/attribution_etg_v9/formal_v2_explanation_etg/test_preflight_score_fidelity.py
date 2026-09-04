import copy
import json
import tempfile
import unittest
from pathlib import Path

from streaming_full.data import canonical_sha256

from formal_v2_explanation_etg.preflight_score_fidelity import (
    ATTRIBUTION_TARGET_SCOPE,
    BATCH_PARTITION_POLICY,
    CONFIRMATION,
    CROSS_DEVICE_POLICY,
    EXPECTED_CLASS_COUNTS,
    EXPECTED_RECONSTRUCTION_ARRAY_TOLERANCES,
    PASS_SCOPE,
    SCHEMA_VERSION,
    validate_report,
)
from formal_v2_explanation_etg.analyze import (
    CROSS_DEVICE_REFERENCE_TOLERANCES,
    SURROGATE_DISTANCE_SQUARED_FLOOR,
)


class ScoreFidelityPreflightTests(unittest.TestCase):
    @staticmethod
    def bindings():
        return {
            "bindings_file_sha256": "a" * 64,
            "method_protocol_file_sha256": "b" * 64,
            "analyzer_file_sha256": "c" * 64,
            "preflight_file_sha256": "d" * 64,
        }

    @classmethod
    def valid_report(cls):
        fidelity_rows = []
        partition_rows = []
        gradient_rows = []
        reconstruction_rows = []
        for seed in (1, 2, 3, 4, 42):
            for checkpoint, class_count in enumerate(EXPECTED_CLASS_COUNTS):
                for array, tolerance in EXPECTED_RECONSTRUCTION_ARRAY_TOLERANCES.items():
                    row = {
                        "seed": seed,
                        "checkpoint": checkpoint,
                        "array": array,
                        "max_abs_error": 0.0,
                        "p99_abs_error": 0.0,
                        "reference_absolute_tolerance": tolerance,
                        "within_reference_tolerance": True,
                        "reference_exceedance_count": 0,
                        "exact_match": True,
                    }
                    if array == "predicted_class_id":
                        row.update(
                            {
                                "decision_count": 10,
                                "mismatch_count": 0,
                                "equivalence_claimed": False,
                                "maximum_row_joint_score_error_at_mismatch": 0.0,
                                "maximum_saved_winner_gap": 0.0,
                                "maximum_reconstructed_winner_gap": 0.0,
                            }
                        )
                    reconstruction_rows.append(row)
                for class_id in range(class_count):
                    fidelity_rows.append(
                        {
                            "seed": seed,
                            "checkpoint": checkpoint,
                            "class_id": class_id,
                            "joint_score_max_abs_error": 0.0,
                            "max_abs_error": 0.0,
                        }
                    )
                    partition_rows.append(
                        {
                            "seed": seed,
                            "checkpoint": checkpoint,
                            "class_id": class_id,
                            "class_vs_full_batch_joint_score_max_abs_difference": 0.0005,
                            "joint_score_reference_absolute_tolerance": 0.001,
                            "joint_score_within_reference_tolerance": True,
                            "joint_score_reference_exceedance_count": 0,
                            "class_vs_full_batch_margin_max_abs_difference": 0.001,
                            "margin_reference_absolute_tolerance": 0.002,
                            "margin_within_reference_tolerance": True,
                            "margin_reference_exceedance_count": 0,
                            "exact_match": True,
                            "decision_count": 10,
                            "mismatch_count": 0,
                            "equivalence_claimed": False,
                            "maximum_row_joint_score_error_at_mismatch": 0.0,
                            "maximum_saved_winner_gap": 0.0,
                            "maximum_reconstructed_winner_gap": 0.0,
                        }
                    )
                    gradient_rows.append(
                        {
                            "seed": seed,
                            "checkpoint": checkpoint,
                            "class_id": class_id,
                            "probe_rows": 10,
                            "feature_dim": 77,
                            "output_max_abs": 1.0,
                            "gradient_max_abs": 0.5,
                            "gradient_nonfinite_count": 0,
                            "surrogate_distance_squared_floor": SURROGATE_DISTANCE_SQUARED_FLOOR,
                            "exact_forward_value_changed": False,
                        }
                    )
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "dataset": "malaya-network-gt",
            "seeds": [1, 2, 3, 4, 42],
            "checkpoint_count": 25,
            "class_checkpoint_count": len(fidelity_rows),
            "attribution_gradient_target_count": len(gradient_rows),
            "attribution_gradient_nonfinite_count": 0,
            "surrogate_distance_squared_floor": SURROGATE_DISTANCE_SQUARED_FLOOR,
            "artifact_reconstruction_row_count": len(reconstruction_rows),
            "artifact_reconstruction_max_abs_error_by_array": {
                array: 0.0 for array in EXPECTED_RECONSTRUCTION_ARRAY_TOLERANCES
            },
            "artifact_reconstruction_p99_abs_error_max_by_array": {
                array: 0.0 for array in EXPECTED_RECONSTRUCTION_ARRAY_TOLERANCES
            },
            "artifact_reconstruction_reference_exceedance_count_by_array": {
                array: 0 for array in EXPECTED_RECONSTRUCTION_ARRAY_TOLERANCES
            },
            "cross_device_reference_absolute_tolerances": CROSS_DEVICE_REFERENCE_TOLERANCES,
            "cross_device_policy": CROSS_DEVICE_POLICY,
            "cross_device_equivalence_claimed": False,
            "pass_scope": PASS_SCOPE,
            "attribution_target_scope": ATTRIBUTION_TARGET_SCOPE,
            "cross_device_decision_count": 250,
            "cross_device_prediction_mismatch_count": 0,
            "cross_device_prediction_mismatch_rate": 0.0,
            "same_batch_absolute_tolerance": 1e-6,
            "same_batch_max_abs_error": 0.0,
            "batch_partition_joint_score_reference_tolerance": 1e-3,
            "batch_partition_joint_score_max_abs_difference": 0.0005,
            "batch_partition_margin_reference_tolerance": 0.002,
            "batch_partition_margin_max_abs_difference": 0.001,
            "batch_partition_joint_score_reference_exceedance_count": 0,
            "batch_partition_margin_reference_exceedance_count": 0,
            "batch_partition_decision_count": 1500,
            "batch_partition_prediction_mismatch_count": 0,
            "batch_partition_prediction_mismatch_rate": 0.0,
            "batch_partition_policy": BATCH_PARTITION_POLICY,
            "batch_partition_equivalence_claimed": False,
            "execution_order": "completed before any attribution call",
            "bindings": cls.bindings(),
            "artifact_reconstruction_rows": reconstruction_rows,
            "fidelity_rows": fidelity_rows,
            "batch_partition_rows": partition_rows,
            "attribution_gradient_rows": gradient_rows,
        }
        report["canonical_sha256"] = canonical_sha256(report)
        return report

    @staticmethod
    def write_report(path, report):
        report = copy.deepcopy(report)
        report.pop("canonical_sha256", None)
        report["canonical_sha256"] = canonical_sha256(report)
        path.write_text(json.dumps(report), encoding="utf-8")

    def test_sbatch_runs_all_seed_preflight_before_attribution(self):
        operation_root = Path(__file__).resolve().parents[1]
        text = (operation_root / "scheduler_contract.sbatch.txt").read_text(
            encoding="utf-8"
        )
        preflight = "-m formal_v2_explanation_etg.preflight_score_fidelity"
        attribution = "-m formal_v2_explanation_etg.analyze"
        self.assertEqual(text.count(preflight), 1)
        self.assertGreaterEqual(text.count(attribution), 1)
        self.assertLess(text.index(preflight), text.index(attribution))
        self.assertIn(f"--confirm {CONFIRMATION}", text)

    def test_completed_report_is_self_hashed_and_source_bound(self):
        bindings = self.bindings()
        report = self.valid_report()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preflight.json"
            self.write_report(path, report)
            self.assertEqual(validate_report(path, bindings)["status"], "passed")
            with self.assertRaisesRegex(RuntimeError, "source bindings"):
                validate_report(path, {**bindings, "analyzer_file_sha256": "e" * 64})

    def test_rejects_missing_and_duplicate_reconstruction_rows(self):
        report = self.valid_report()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preflight.json"
            missing = copy.deepcopy(report)
            missing["artifact_reconstruction_rows"].pop()
            self.write_report(path, missing)
            with self.assertRaisesRegex(RuntimeError, "rows are incomplete"):
                validate_report(path, self.bindings())
            duplicate = copy.deepcopy(report)
            duplicate["artifact_reconstruction_rows"][-1] = copy.deepcopy(
                duplicate["artifact_reconstruction_rows"][0]
            )
            self.write_report(path, duplicate)
            with self.assertRaisesRegex(RuntimeError, "duplicated"):
                validate_report(path, self.bindings())

    def test_rejects_unknown_reconstruction_array(self):
        report = self.valid_report()
        report["artifact_reconstruction_rows"][0]["array"] = "unknown_scores"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preflight.json"
            self.write_report(path, report)
            with self.assertRaisesRegex(RuntimeError, "array domain"):
                validate_report(path, self.bindings())

    def test_rejects_non_finite_values(self):
        report = self.valid_report()
        report["fidelity_rows"][0]["max_abs_error"] = "nan"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preflight.json"
            self.write_report(path, report)
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                validate_report(path, self.bindings())

    def test_accepts_cross_device_reference_exceedance_without_decision_flip(self):
        report = self.valid_report()
        row = next(
            item
            for item in report["artifact_reconstruction_rows"]
            if item["array"] == "joint_scores"
        )
        row["max_abs_error"] = 0.003
        row["p99_abs_error"] = 0.0012
        row["within_reference_tolerance"] = False
        row["reference_exceedance_count"] = 25
        row["exact_match"] = False
        report["artifact_reconstruction_max_abs_error_by_array"]["joint_scores"] = 0.003
        report["artifact_reconstruction_p99_abs_error_max_by_array"]["joint_scores"] = 0.0012
        report["artifact_reconstruction_reference_exceedance_count_by_array"]["joint_scores"] = 25
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preflight.json"
            self.write_report(path, report)
            self.assertEqual(validate_report(path, self.bindings())["status"], "passed")

    def test_rejects_cross_device_equivalence_claim(self):
        report = self.valid_report()
        row = next(
            item
            for item in report["artifact_reconstruction_rows"]
            if item["array"] == "predicted_class_id"
        )
        row["equivalence_claimed"] = True
        report["cross_device_equivalence_claimed"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preflight.json"
            self.write_report(path, report)
            with self.assertRaisesRegex(RuntimeError, "equivalence claim"):
                validate_report(path, self.bindings())

    def test_accepts_partition_reference_exceedance_when_decisions_are_audited(self):
        report = self.valid_report()
        row = report["batch_partition_rows"][0]
        row["class_vs_full_batch_joint_score_max_abs_difference"] = 0.0011
        row["joint_score_within_reference_tolerance"] = False
        row["joint_score_reference_exceedance_count"] = 2
        report["batch_partition_joint_score_max_abs_difference"] = 0.0011
        report["batch_partition_joint_score_reference_exceedance_count"] = 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preflight.json"
            self.write_report(path, report)
            self.assertEqual(validate_report(path, self.bindings())["status"], "passed")

    def test_rejects_batch_partition_equivalence_claim(self):
        report = self.valid_report()
        report["batch_partition_rows"][0]["equivalence_claimed"] = True
        report["batch_partition_equivalence_claimed"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preflight.json"
            self.write_report(path, report)
            with self.assertRaisesRegex(RuntimeError, "equivalence claim"):
                validate_report(path, self.bindings())

    def test_rejects_incomplete_fidelity_rows(self):
        report = self.valid_report()
        report["fidelity_rows"].pop()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preflight.json"
            self.write_report(path, report)
            with self.assertRaisesRegex(RuntimeError, "coverage"):
                validate_report(path, self.bindings())

    def test_rejects_non_finite_attribution_gradient(self):
        report = self.valid_report()
        report["attribution_gradient_rows"][0]["gradient_nonfinite_count"] = 1
        report["attribution_gradient_nonfinite_count"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preflight.json"
            self.write_report(path, report)
            with self.assertRaisesRegex(RuntimeError, "non-finite attribution gradients"):
                validate_report(path, self.bindings())


if __name__ == "__main__":
    unittest.main()
