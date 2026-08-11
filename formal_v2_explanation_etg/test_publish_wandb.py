from __future__ import annotations

import json
import hashlib
import math
import tempfile
import unittest
from pathlib import Path

from publish_wandb import (
    build_config,
    combined_history_rows,
    history_rows,
    load_governance,
    load_training_result,
    performance_rows,
    summary_payload,
    table_payloads,
    table_specs,
    validate_outbound_payload,
    wandb_settings_kwargs,
)


class FakeTable:
    def __init__(self, *, columns, data):
        self.columns = columns
        self.data = data


class FakeWandb:
    Table = FakeTable


class PublishWandbTests(unittest.TestCase):
    @staticmethod
    def governance() -> dict:
        return json.loads(
            Path(__file__).with_name("MALAYANETWORK_GT_DATA_GOVERNANCE.json").read_text(
                encoding="utf-8"
            )
        )

    @staticmethod
    def complete_analysis() -> dict:
        return {
            "dataset": "malaya-network-gt",
            "seed": 1,
            "schema_version": "formal_v2",
            "analysis_protocol_sha256": "a" * 64,
            "canonical_sha256": "b" * 64,
            "checkpoint_rows": [],
            "transition_rows": [],
            "threshold_sensitivity": [],
            "etg_ledger": [],
            "etg_summary": {"final_states": {}, "event_count": 0},
            "primary_silent_explanation_drift": {
                "k": 15,
                "jaccard_threshold": 0.7,
                "allowed_recall_drop": 0.01,
                "events": 0,
                "eligible_transitions": 0,
                "rate": 0.0,
            },
        }

    @staticmethod
    def training_result() -> dict:
        task_rows = [
            {"0": 0.9},
            {"0": 0.8, "1": 0.7},
            {"0": 0.7, "1": 0.6, "2": 0.5},
        ]
        checkpoints = []
        for checkpoint, tasks in enumerate(task_rows):
            checkpoints.append({
                "checkpoint": checkpoint,
                "views": {"official": {"arms": {"joint_cap3000": {
                    "accuracy": [0.8, 0.7, 0.55][checkpoint],
                    "macro_f1": [0.75, 0.65, 0.5][checkpoint],
                    "balanced_accuracy": [0.78, 0.68, 0.52][checkpoint],
                    "task_accuracy": tasks,
                }}}},
            })
        return {
            "dataset": "malaya-network-gt",
            "seed": 1,
            "checkpoints": checkpoints,
            "summary": {"views": {"official": {"joint_cap3000": {
                "task_accuracy_matrix": [
                    [0.9, None, None],
                    [0.8, 0.7, None],
                    [0.7, 0.6, 0.5],
                ],
                "average_task_accuracy": 0.6,
                "average_forgetting": 0.15,
                "final_overall_accuracy": 0.55,
                "final_macro_f1": 0.5,
                "final_balanced_accuracy": 0.52,
            }}}},
        }
    def test_history_uses_checkpoint_axis_and_cumulative_denominator(self):
        analysis = {
            "transition_rows": [
                {"to_checkpoint": 1, "jaccard_top15": 0.5, "primary_event": True, "primary_eligible": True},
                {"to_checkpoint": 1, "jaccard_top15": 0.9, "primary_event": False, "primary_eligible": True},
                {"to_checkpoint": 2, "jaccard_top15": 0.8, "primary_event": False, "primary_eligible": False},
            ]
        }
        rows = history_rows(analysis)
        self.assertEqual([row["explanation/checkpoint"] for row in rows], [1, 2])
        self.assertAlmostEqual(rows[0]["explanation/mean_top15_jaccard"], 0.7)
        self.assertEqual(rows[-1]["explanation/silent_drift_events_cumulative"], 1)
        self.assertEqual(rows[-1]["explanation/eligible_transitions_cumulative"], 2)

    def test_tables_keep_exact_rows_and_readable_feature_names(self):
        analysis = {
            "checkpoint_rows": [{
                "dataset": "demo", "seed": 1, "checkpoint": 0, "class_id": 0,
                "class_name": "Normal", "top15_features": ["a", "b"],
            }],
            "transition_rows": [],
            "threshold_sensitivity": [],
            "etg_ledger": [{"action": "admission_certified"}],
            "etg_summary": {"final_states": {"CERTIFIED_STABLE": 1}},
        }
        payloads = table_payloads(FakeWandb(), analysis)
        self.assertEqual(len(payloads), 6)
        checkpoint = payloads["results/shap_checkpoint_metrics"]
        self.assertEqual(checkpoint.data[0][-1], "a, b")

    def test_governance_is_bound_to_dataset_destination_and_public_safe_mode(self):
        governance = self.governance()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "governance.json"
            path.write_text(json.dumps(governance), encoding="utf-8")
            loaded = load_governance(
                path,
                runtime_dataset_id="malaya-network-gt",
                destination="csnet/ofra-etg-leon-hpc",
            )
            self.assertEqual(loaded["license"], "CC-BY-4.0")
            with self.assertRaisesRegex(RuntimeError, "runtime dataset mismatch"):
                load_governance(
                    path,
                    runtime_dataset_id="unapproved-dataset-id",
                    destination="csnet/ofra-etg-leon-hpc",
                )
            with self.assertRaisesRegex(RuntimeError, "destination mismatch"):
                load_governance(
                    path,
                    runtime_dataset_id="malaya-network-gt",
                    destination="someone-else/project",
                )

    def test_governance_rejects_missing_allowlist(self):
        governance = self.governance()
        del governance["outbound_policy"]["allowed_history_keys"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "governance.json"
            path.write_text(json.dumps(governance), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "allow/deny"):
                load_governance(
                    path,
                    runtime_dataset_id="malaya-network-gt",
                    destination="csnet/ofra-etg-leon-hpc",
                )

    def test_wandb_automatic_metadata_and_telemetry_are_disabled(self):
        settings = wandb_settings_kwargs()
        self.assertTrue(settings["disable_git"])
        self.assertTrue(settings["disable_code"])
        self.assertTrue(settings["disable_job_creation"])
        self.assertTrue(settings["x_disable_meta"])
        self.assertTrue(settings["x_disable_stats"])
        self.assertTrue(settings["x_disable_machine_info"])
        self.assertFalse(settings["x_save_requirements"])

    def test_payload_is_hash_only_and_rejects_absolute_paths(self):
        governance = self.governance()
        analysis = self.complete_analysis()
        manifest = {"canonical_sha256": "c" * 64}
        config = build_config(
            analysis,
            manifest,
            governance,
            submission_bindings_sha256="d" * 64,
            training_result_sha256="e" * 64,
        )
        self.assertNotIn("input_bindings", config)
        self.assertEqual(config["dataset"], "MalayaNetwork_GT")
        self.assertEqual(config["runtime_dataset_id"], "malaya-network-gt")
        self.assertEqual(config["submission_bindings_sha256"], "d" * 64)
        training_result = self.training_result()
        history = combined_history_rows(analysis, training_result)
        tables = table_specs(analysis, training_result)
        summary = summary_payload(analysis, manifest, training_result)
        validate_outbound_payload(
            governance, config=config, history=history, tables=tables, summary=summary
        )
        config["data_governance"]["derivative_description"] = "/tmp/example/raw.csv"
        with self.assertRaisesRegex(RuntimeError, "absolute path is forbidden"):
            validate_outbound_payload(
                governance,
                config=config,
                history=history,
                tables=tables,
                summary=summary,
            )

    def test_training_result_is_bound_to_the_validated_runtime_dataset_id(self):
        result = self.training_result()
        result["dataset"] = "malaya-network-gt"
        with tempfile.TemporaryDirectory() as temporary:
            result_path = Path(temporary) / "result_seed_1.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            bindings_path = Path(temporary) / "bindings.json"
            bindings_path.write_text(
                json.dumps({"files": {"training_result": {
                    "path": str(result_path),
                    "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                }}}),
                encoding="utf-8",
            )
            loaded, _ = load_training_result(
                result_path,
                bindings_path,
                expected_dataset="malaya-network-gt",
                expected_seed=1,
            )
            self.assertEqual(loaded["dataset"], "malaya-network-gt")
            with self.assertRaisesRegex(RuntimeError, "differs from validated analysis"):
                load_training_result(
                    result_path,
                    bindings_path,
                    expected_dataset="unapproved-dataset-id",
                    expected_seed=1,
                )

    def test_primary_performance_metrics_follow_bound_formulas(self):
        rows = performance_rows(self.training_result())
        self.assertEqual([row["checkpoint/index"] for row in rows], [0, 1, 2])
        self.assertAlmostEqual(rows[1]["performance/average_task_accuracy"], 0.75)
        self.assertAlmostEqual(rows[1]["performance/average_forgetting"], 0.1)
        self.assertAlmostEqual(rows[2]["performance/average_task_accuracy"], 0.6)
        self.assertAlmostEqual(rows[2]["performance/average_forgetting"], 0.15)
        specs = table_specs(self.complete_analysis(), self.training_result())
        matrix = specs["results/primary_task_accuracy_matrix"]
        self.assertEqual(matrix["columns"], ["checkpoint", "task", "task_accuracy"])
        self.assertEqual(len(matrix["data"]), 6)

    def test_task_accuracy_matrix_must_match_checkpoint_records(self):
        result = self.training_result()
        result["summary"]["views"]["official"]["joint_cap3000"][
            "task_accuracy_matrix"
        ][1][0] = 0.79
        with self.assertRaisesRegex(RuntimeError, "differs from checkpoint data"):
            performance_rows(result)

    def test_task_accuracy_values_must_be_finite_unit_interval(self):
        for invalid in (math.nan, math.inf, -0.01, 1.01):
            with self.subTest(invalid=invalid):
                result = self.training_result()
                result["checkpoints"][0]["views"]["official"]["arms"][
                    "joint_cap3000"
                ]["task_accuracy"]["0"] = invalid
                with self.assertRaisesRegex(RuntimeError, "finite and within"):
                    performance_rows(result)


if __name__ == "__main__":
    unittest.main()
