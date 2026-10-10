from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ncd.evaluate_exp3_performance import (
    collect_training_runs,
    metrics_from_predictions,
    summarize_forgetting,
)


class NcdEvaluateExp3PerformanceTest(unittest.TestCase):
    def test_collects_direct_training_results_and_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "replay.json"
            second = root / "icarl.json"
            first.write_text(
                json.dumps({"dataset": "NSL-KDD", "method": "replay", "seed": 1}),
                encoding="utf-8",
            )
            second.write_text(
                json.dumps({"dataset": "NSL-KDD", "method": "icarl", "seed": 1}),
                encoding="utf-8",
            )
            dataset, runs, provenance = collect_training_runs(
                root, None, [first, second]
            )
            self.assertEqual(dataset, "NSL-KDD")
            self.assertEqual(
                [(run["method"], run["seed"]) for run in runs],
                [("replay", 1), ("icarl", 1)],
            )
            self.assertEqual(len(provenance["direct_training_results"]), 2)
            with self.assertRaisesRegex(ValueError, "duplicate method/seed"):
                collect_training_runs(root, None, [first, first])

    def test_metrics_and_forgetting(self) -> None:
        metrics = metrics_from_predictions(
            np.array([0, 0, 1, 1, 2], dtype=np.int64),
            np.array([0, 1, 1, 2, 2], dtype=np.int64),
            [0, 1],
            3,
        )
        self.assertEqual(metrics["seen_test_rows"], 4)
        self.assertEqual(metrics["accuracy_seen"], 0.5)
        self.assertEqual(metrics["balanced_accuracy_seen"], 0.5)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1, 0], [0, 1, 1], [0, 0, 0]])

        summary = summarize_forgetting(
            [
                {"per_class_recall": {"0": 0.9}, "accuracy_seen": 0.9,
                 "balanced_accuracy_seen": 0.9, "macro_f1_seen": 0.9,
                 "confusion_matrix": [[9, 1], [0, 0]]},
                {"per_class_recall": {"0": 0.7, "1": 0.8}, "accuracy_seen": 0.75,
                 "balanced_accuracy_seen": 0.75, "macro_f1_seen": 0.74,
                 "confusion_matrix": [[7, 3], [2, 8]]},
            ]
        )
        self.assertAlmostEqual(summary["average_forgetting_prior_classes"], 0.2)
        self.assertAlmostEqual(summary["forgetting_by_class"]["0"], 0.2)
        self.assertEqual(summary["final_normal_recall"], 0.7)


if __name__ == "__main__":
    unittest.main()
