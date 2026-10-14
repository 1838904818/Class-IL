"""Unit fixtures only; these tests are not scientific evidence."""
import unittest
import tempfile
import json
from pathlib import Path
from ncd.analyze_problem_scout import SEEDS, analyze, best_comparison, count_bucket, error_flow, load_cells, validate_confusion


class ScoutAnalysisTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
