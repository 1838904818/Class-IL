import json
import tempfile
import unittest
from pathlib import Path

from streaming_full.runner import build_parser
from streaming_full.wandb_tracking import (
    WandbSeedTracker,
    confusion_table_rows,
    flatten_checkpoint_event,
    summary_table_rows,
    task_accuracy_table_rows,
)
from streaming_full.validation import _training_rows_processed


class FakeTable:
    def __init__(self, *, columns, data):
        self.columns = columns
        self.data = data


class FakeRun:
    def __init__(self):
        self.summary = {}
        self.logged = []
        self.finished = None

    def log(self, payload, step=None):
        self.logged.append((payload, step))

    def finish(self, exit_code=0):
        self.finished = exit_code


class FakeWandb:
    Table = FakeTable

    @staticmethod
    def Settings(**kwargs):
        return kwargs

    def __init__(self):
        self.init_kwargs = []
        self.runs = []

    def init(self, **kwargs):
        self.init_kwargs.append(kwargs)
        run = FakeRun()
        self.runs.append(run)
        return run


def result_record():
    metrics = {
        "confusion_matrix": [[8, 2], [1, 9]],
        "per_class": [
            {"class_id": 0, "class_name": "Benign"},
            {"class_id": 1, "class_name": "Attack"},
        ],
    }
    summary = {
        "average_task_accuracy": 0.85,
        "average_forgetting": 0.05,
        "final_overall_accuracy": 0.85,
        "final_macro_f1": 0.8496,
        "final_balanced_accuracy": 0.85,
        "final_benign_false_positive_rate": 0.2,
        "final_attack_detection_recall": 0.9,
        "task_accuracy_matrix": [[0.9, None], [0.8, 0.9]],
    }
    return {
        "seed": 1,
        "deterministic_result_sha256": "a" * 64,
        "model_parameters": {
            "encoder_parameters": 1000,
            "family_heads_total_parameters": 100,
            "encoder_plus_family_heads_parameters": 1100,
        },
        "timing": {
            "total_seconds": 2.0,
            "workload": {
                "training_rows_processed": 200,
                "official_evaluation_rows_processed": 40,
                "training_rows_per_second_end_to_end": 100.0,
            },
            "cuda": {
                "device_total_memory_bytes": 10000,
                "peak_allocated_bytes": 5000,
                "peak_reserved_bytes": 6000,
                "peak_allocated_fraction": 0.5,
                "peak_reserved_fraction": 0.6,
            },
        },
        "summary": {"views": {"official": {"joint_cap3000": summary}}},
        "checkpoints": [
            {"seen_classes": [0]},
            {
                "seen_classes": [0, 1],
                "views": {"official": {"arms": {"joint_cap3000": metrics}}},
            },
        ],
        "monitoring": {"enabled": False},
    }


class WandbTrackingTests(unittest.TestCase):
    def test_training_row_telemetry_includes_pretrain_and_binary_head_rows(self):
        pretrain = [{"rows": 100}, {"rows": 100}]
        training_history = {
            "0": [{"positive_rows": 20, "negative_rows": 60}],
            "1": [
                {"positive_rows": 10, "negative_rows": 30},
                {"positive_rows": 10, "negative_rows": 30},
            ],
        }
        self.assertEqual(
            _training_rows_processed(pretrain, training_history),
            360,
        )

    def test_cli_has_no_api_key_argument(self):
        parser = build_parser()
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertIn("--wandb-project", option_strings)
        self.assertNotIn("--wandb-api-key", option_strings)

    def test_checkpoint_metric_names_are_clear_and_checkpoint_indexed(self):
        event = {
            "checkpoint": 2,
            "seen_classes": [0, 1, 2],
            "elapsed_seconds": 12.5,
            "views": {
                "official": {
                    "joint_cap3000": {
                        "final_overall_accuracy": 0.8,
                        "average_forgetting": 0.1,
                    }
                }
            },
        }
        flattened = flatten_checkpoint_event(event)
        self.assertEqual(
            set(flattened),
            {
                "checkpoint/index",
                "official/accuracy/joint_cap3000",
                "official/average_forgetting/joint_cap3000",
            },
        )
        self.assertEqual(flattened["checkpoint/index"], 2)

    def test_result_tables_are_long_form_and_non_graphical(self):
        result = result_record()
        self.assertEqual(len(summary_table_rows(result)), 1)
        self.assertEqual(len(task_accuracy_table_rows(result)), 3)
        confusion = confusion_table_rows(result)
        self.assertEqual(len(confusion), 4)
        self.assertAlmostEqual(confusion[0][-1] + confusion[1][-1], 1.0)

    def test_one_wandb_run_per_seed_without_secret_in_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "protocol.json").write_text(
                json.dumps(
                    {
                        "config": {
                            "encoder_type": "ft_transformer",
                            "d_model": 64,
                            "n_layers": 4,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (output / "result_seed_1.json").write_text(
                json.dumps(result_record()), encoding="utf-8"
            )
            fake = FakeWandb()
            tracker = WandbSeedTracker(
                project="ofra-etg-leon-hpc",
                entity="csnet",
                output_dir=output,
                wandb_module=fake,
            )
            common = {
                "dataset": "NSL-KDD",
                "seed": 1,
                "protocol_sha256": "b" * 64,
                "elapsed_seconds": 0.0,
            }
            tracker({"event": "start", **common})
            tracker(
                {
                    "event": "checkpoint",
                    **common,
                    "checkpoint": 0,
                    "seen_classes": [0],
                    "views": {
                        "official": {
                            "joint_cap3000": {"final_overall_accuracy": 0.8}
                        }
                    },
                }
            )
            tracker({"event": "end", **common})
            self.assertEqual(len(fake.runs), 1)
            self.assertEqual(fake.init_kwargs[0]["entity"], "csnet")
            self.assertEqual(fake.runs[0].finished, 0)
            config_text = json.dumps(fake.init_kwargs[0]["config"])
            self.assertNotIn("WANDB_API_KEY", config_text)
            self.assertNotIn("api_key", config_text.lower())
            self.assertEqual(
                fake.runs[0].summary["governance/shap_status"], "not_computed"
            )
            self.assertEqual(
                fake.runs[0].summary["governance/etg_status"], "not_computed"
            )
            settings = fake.init_kwargs[0]["settings"]
            self.assertEqual(
                settings,
                {
                    "x_disable_meta": True,
                    "x_disable_stats": True,
                    "x_disable_machine_info": True,
                    "x_save_requirements": False,
                    "disable_code": True,
                    "disable_git": True,
                    "disable_job_creation": True,
                    "save_code": False,
                    "console": "off",
                },
            )
            self.assertNotIn("slurm_job_id", config_text)

    def test_diagnostic_views_and_arms_are_not_published(self):
        event = {
            "checkpoint": 0,
            "seen_classes": [0],
            "elapsed_seconds": 1.0,
            "views": {"official": {"z_only": {"final_overall_accuracy": 0.5}}},
        }
        with self.assertRaises(ValueError):
            flatten_checkpoint_event(event)
        event["views"] = {
            "official": {"head_only": {"final_overall_accuracy": 0.5}},
            "diagnostic": {},
        }
        flattened = flatten_checkpoint_event(event)
        self.assertNotIn("diagnostic", json.dumps(flattened))

    def test_failure_event_never_uploads_exception_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "protocol.json").write_text(
                json.dumps({"config": {"encoder_type": "ft_transformer"}}),
                encoding="utf-8",
            )
            fake = FakeWandb()
            tracker = WandbSeedTracker(
                project="ofra-etg-leon-hpc",
                entity="csnet",
                output_dir=output,
                wandb_module=fake,
            )
            common = {"dataset": "NSL-KDD", "seed": 1, "protocol_sha256": "b" * 64}
            tracker({"event": "start", **common})
            tracker({"event": "fail", "error_type": "RuntimeError", "error": "C:/secret/path", **common})
            summary_text = json.dumps(fake.runs[0].summary)
            self.assertNotIn("secret", summary_text)
            self.assertNotIn("RuntimeError", summary_text)


if __name__ == "__main__":
    unittest.main()
