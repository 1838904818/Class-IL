from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_open_set import (
    average_precision,
    empirical_anomaly_scores,
    rejection_metrics,
    roc_auc,
)


class OpenSetMetricTests(unittest.TestCase):
    def test_perfect_ranking(self) -> None:
        labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
        scores = np.asarray([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(roc_auc(labels, scores), 1.0)
        self.assertAlmostEqual(average_precision(labels, scores), 1.0)

    def test_empirical_joint_is_larger_for_low_confidence_far_distance(self) -> None:
        calibration_p = np.asarray([0.60, 0.70, 0.80, 0.90])
        calibration_d = np.asarray([1.0, 2.0, 3.0, 4.0])
        scores = empirical_anomaly_scores(
            calibration_p,
            calibration_d,
            np.asarray([0.95, 0.20]),
            np.asarray([0.5, 9.0]),
        )
        self.assertLess(scores[0], scores[1])

    def test_rejection_metrics(self) -> None:
        metrics = rejection_metrics(
            np.asarray([False, False, True, False]),
            np.asarray([True, True, False, True]),
        )
        self.assertAlmostEqual(metrics["unknown_recall"], 0.75)
        self.assertAlmostEqual(metrics["known_false_unknown_rate"], 0.25)
        self.assertAlmostEqual(metrics["open_set_balanced_accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()
