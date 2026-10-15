"""Unit fixtures only; these tests are not scientific evidence."""
import unittest
import tempfile
import json
import copy
from pathlib import Path
from ncd.analyze_problem_scout import SEEDS, analyze, best_comparison, checkpoint_costs, count_bucket, error_flow, load_cells, validate_confusion


class ScoutAnalysisTests(unittest.TestCase):
    @staticmethod
    def cost_cell():
        metrics = {"confusion_matrix": [[8, 2], [1, 9]], "evaluated_rows": 20,
                   "per_class": [{"class_id": 0, "support": 10, "recall": .8, "f1": 16/19},
                                 {"class_id": 1, "support": 10, "recall": .9, "f1": 18/21}],
                   "macro_f1": (16/19+18/21)/2, "balanced_accuracy": .85,
                   "attack_recall": .9, "normal_fpr": .2}
        return {"identity": {"dataset": "NSL-KDD", "method": "replay", "seed": 1},
                "completed": True, "class_order": ["Normal", "Attack"], "tasks": [[0], [1]],
                "train_counts": {"0": 100, "1": 100}, "training_seconds_total": 5.0,
                "elapsed_seconds_this_attempt": 999.0,
                "checkpoints": [
                    {"task_index": 0, "training_seconds": 2.0, "evaluation_seconds": .5,
                     "buffer_selection_seconds": .25, "peak_cuda_allocated_bytes": 10,
                     "peak_cuda_reserved_bytes": 20, "replay_bytes_after": 100,
                     "replay_allocated_bytes_after": 400, "id": copy.deepcopy(metrics)},
                    {"task_index": 1, "training_seconds": 3.0, "evaluation_seconds": .75,
                     "buffer_selection_seconds": .125, "peak_cuda_allocated_bytes": 15,
                     "peak_cuda_reserved_bytes": 30, "replay_bytes_after": 200,
                     "replay_allocated_bytes_after": 400, "id": copy.deepcopy(metrics)}]}

    def test_same_method_across_seeds_not_per_seed_oracle(self):
        values = {"joint": dict.fromkeys(SEEDS, .9),
                  "a": dict(zip(SEEDS, [.9, .5, .9, .5, .9])),
                  "b": dict.fromkeys(SEEDS, .8)}
        result = best_comparison(values)
        self.assertEqual(result["best_cil_method"], "b")
        self.assertAlmostEqual(result["gap_pp"]["mean"], 10)
        self.assertTrue(result["gap_positive_all_five_seeds"])

    def test_missing_is_not_zero_and_lower_fpr_is_better(self):
        self.assertIsNone(best_comparison({"joint": {1: .9}, "a": dict.fromkeys(SEEDS, .5)}))
        result = best_comparison({"joint": dict.fromkeys(SEEDS, .01), "a": dict.fromkeys(SEEDS, .1),
                                  "b": dict.fromkeys(SEEDS, .2)}, False)
        self.assertEqual(result["best_cil_method"], "a")
        self.assertAlmostEqual(result["gap_pp"]["mean"], 9)

    def test_mutually_exclusive_error_destinations(self):
        matrix = [[9, 0, 0, 1], [2, 5, 1, 2], [0, 0, 9, 1], [0, 0, 0, 10]]
        result = error_flow(matrix, 1, [[0, 1], [2], [3]])
        self.assertEqual(result["counts"], {"normal": 2, "newer_classes": 3, "other_old_or_same_task": 0})
        self.assertEqual(result["errors"], 5)

    def test_boundary_bins_and_empty_matrix_pending(self):
        self.assertEqual([count_bucket(n) for n in (999, 1000, 10000, 10001)],
                         ["<1k", "1k-10k", "1k-10k", ">10k"])
        result = analyze([])
        self.assertEqual(result["execution_status"], "partial")
        self.assertEqual(result["expected_cells"], 120)
        self.assertEqual(result["candidates"], [])

    def test_multiple_matrix_roots_exclude_indexes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("tabular", "cic"):
                (root / name / "dataset" / "method").mkdir(parents=True)
                (root / name / "result.json").write_text(json.dumps({"cells": [], "expected": {}}))
                (root / name / "dataset" / "method" / "result.json").write_text(json.dumps({"identity": {"dataset": name}}))
            paths, cells = load_cells(root)
            self.assertEqual(len(paths), 2)
            self.assertEqual({c["identity"]["dataset"] for c in cells}, {"cic", "tabular"})

    def test_independent_confusion_recomputation(self):
        metrics = {"confusion_matrix": [[8, 2], [1, 9]], "evaluated_rows": 20,
                   "per_class": [{"class_id": 0, "support": 10, "recall": .8, "f1": 16/19},
                                 {"class_id": 1, "support": 10, "recall": .9, "f1": 18/21}],
                   "macro_f1": (16/19+18/21)/2, "balanced_accuracy": .85,
                   "attack_recall": .9, "normal_fpr": .2}
        validate_confusion(metrics)
        metrics["normal_fpr"] = .1
        with self.assertRaises(ValueError):
            validate_confusion(metrics)

    def test_g7_sums_all_successful_checkpoint_stages_without_walltime(self):
        result = analyze([self.cost_cell()])
        row = result["G7"][0]
        self.assertEqual(row["training_seconds_total"], 5.0)
        self.assertEqual(row["evaluation_seconds_total"], 1.25)
        self.assertEqual(row["buffer_selection_seconds_total"], .375)
        self.assertEqual(row["peak_cuda_allocated_bytes"], 15)
        self.assertEqual(row["peak_cuda_reserved_bytes"], 30)
        self.assertEqual(row["maximum_replay_bytes"], 200)
        self.assertEqual(row["maximum_replay_allocated_bytes"], 400)
        self.assertNotIn("elapsed_seconds_this_attempt", row)
        limits = result["G7_measurement_limits"]
        self.assertIn("DER++", limits["buffer_selection_seconds_total"])
        self.assertIn("before selection/evaluation", limits["cuda_memory"])
        self.assertIn("failed/cancelled", limits["walltime"])
        self.assertFalse(result["scientific_comparisons_final"])

    def test_g7_missing_costs_are_errors_not_zero_or_payload_fallback(self):
        checkpoint_fields = ("training_seconds", "evaluation_seconds", "buffer_selection_seconds",
                             "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes",
                             "replay_bytes_after", "replay_allocated_bytes_after")
        for field in checkpoint_fields:
            with self.subTest(field=field):
                cell = self.cost_cell()
                del cell["checkpoints"][1][field]
                with self.assertRaisesRegex(ValueError, f"Missing G7 cost field {field}"):
                    analyze([cell])
        cell = self.cost_cell()
        del cell["training_seconds_total"]
        with self.assertRaisesRegex(ValueError, "Missing G7 cost field training_seconds_total"):
            analyze([cell])

    def test_g7_nonfinite_negative_or_nonnumeric_costs_are_errors(self):
        for field in ("training_seconds", "evaluation_seconds", "buffer_selection_seconds",
                      "replay_bytes_after", "replay_allocated_bytes_after"):
            for invalid in (None, float("nan"), float("inf"), -float("inf"), -1, True, "1"):
                with self.subTest(field=field, invalid=invalid):
                    cell = self.cost_cell()
                    cell["checkpoints"][0][field] = invalid
                    with self.assertRaisesRegex(ValueError, f"Invalid G7 cost field {field}"):
                        checkpoint_costs(cell)
        for invalid in (None, float("nan"), float("inf"), -1, True, "5"):
            with self.subTest(cell_training=invalid):
                cell = self.cost_cell()
                cell["training_seconds_total"] = invalid
                with self.assertRaisesRegex(ValueError, "Invalid G7 cost field training_seconds_total"):
                    checkpoint_costs(cell)

    def test_g7_unmeasured_cuda_remains_null_but_invalid_values_fail(self):
        cell = self.cost_cell()
        for checkpoint in cell["checkpoints"]:
            checkpoint["peak_cuda_allocated_bytes"] = None
            checkpoint["peak_cuda_reserved_bytes"] = None
        result = checkpoint_costs(cell)
        self.assertIsNone(result["peak_cuda_allocated_bytes"])
        self.assertIsNone(result["peak_cuda_reserved_bytes"])
        cell["checkpoints"][1]["peak_cuda_allocated_bytes"] = 15
        self.assertEqual(checkpoint_costs(cell)["peak_cuda_allocated_bytes"], 15)
        for field in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"):
            for invalid in (float("nan"), float("inf"), -1, True):
                with self.subTest(field=field, invalid=invalid):
                    invalid_cell = self.cost_cell()
                    invalid_cell["checkpoints"][0][field] = invalid
                    with self.assertRaisesRegex(ValueError, f"Invalid G7 cost field {field}"):
                        checkpoint_costs(invalid_cell)

    def test_g7_measured_zero_is_valid(self):
        cell = self.cost_cell()
        cell["training_seconds_total"] = 0
        for checkpoint in cell["checkpoints"]:
            for field in ("training_seconds", "evaluation_seconds", "buffer_selection_seconds",
                          "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes",
                          "replay_bytes_after", "replay_allocated_bytes_after"):
                checkpoint[field] = 0
        result = checkpoint_costs(cell)
        for field, value in result.items():
            if field != "cost_scope":
                self.assertEqual(value, 0)

    def test_g7_rejects_inconsistent_training_total_and_overflow(self):
        cell = self.cost_cell()
        cell["training_seconds_total"] = 6
        with self.assertRaisesRegex(ValueError, "differs from successful checkpoint training sum"):
            checkpoint_costs(cell)
        cell = self.cost_cell()
        for checkpoint in cell["checkpoints"]:
            checkpoint["evaluation_seconds"] = 1e308
        with self.assertRaisesRegex(ValueError, "Invalid G7 cost field evaluation_seconds"):
            checkpoint_costs(cell)

    def test_g7_rejects_empty_checkpoint_costs(self):
        cell = self.cost_cell()
        cell["checkpoints"] = []
        with self.assertRaisesRegex(ValueError, "No successful checkpoints for G7 costs"):
            checkpoint_costs(cell)


if __name__ == "__main__":
    unittest.main()
