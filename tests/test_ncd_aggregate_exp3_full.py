from __future__ import annotations

import unittest

from ncd.aggregate_exp3_full import check_coverage, mean_std, summarize_group


class NcdAggregateExp3FullTest(unittest.TestCase):
    def test_mean_std_and_weighted_summary(self) -> None:
        drift = [
            {
                "seed": 1,
                "silent_units_5pp": 2,
                "b1_consensus_drift_units": 1,
                "b2_consensus_drift_units": 0,
                "ncd_consensus_drift_units": 2,
                "b1_all_explainer_agreement_rate": 0.5,
                "b2_all_explainer_agreement_rate": 1.0,
                "ncd_all_explainer_agreement_rate": 1.0,
            },
            {
                "seed": 2,
                "silent_units_5pp": 1,
                "b1_consensus_drift_units": 0,
                "b2_consensus_drift_units": 1,
                "ncd_consensus_drift_units": 1,
                "b1_all_explainer_agreement_rate": 1.0,
                "b2_all_explainer_agreement_rate": 0.0,
                "ncd_all_explainer_agreement_rate": 1.0,
            },
        ]
        performance = [
            {"final_accuracy_seen": 0.6},
            {"final_accuracy_seen": 0.8},
        ]
        consensus = [
            {"silent_at_5pp": "True", "future_recall_delta": "-0.1", "ncd_consensus_drift": "True"},
            {"silent_at_5pp": "True", "future_recall_delta": "0.2", "ncd_consensus_drift": "False"},
        ]
        costs = [
            {"training_seconds": 1, "explanation_seconds": 2, "fixed_rule_seconds": 0.5, "ncd_permutation_seconds": 5},
            {"training_seconds": 3, "explanation_seconds": 4, "fixed_rule_seconds": 0.5, "ncd_permutation_seconds": 5},
        ]
        summary = summarize_group(drift, performance, consensus, costs)
        self.assertEqual(summary["silent_units_5pp"], 3)
        self.assertAlmostEqual(summary["b1_drift_rate"], 1 / 3)
        self.assertAlmostEqual(summary["b1_all_explainer_agreement_rate"], 2 / 3)
        self.assertAlmostEqual(summary["final_accuracy_seen_mean"], 0.7)
        self.assertAlmostEqual(summary["ncd_to_fixed_rule_time_ratio"], 10.0)
        self.assertEqual(summary["median_future_recall_delta_after_ncd_drift"], -0.1)
        self.assertEqual(mean_std([1.0, 3.0]), (2.0, 2 ** 0.5))

    def test_partial_coverage_requires_opt_in(self) -> None:
        key = ("NSL-KDD", "replay", 1)
        row = {key: {}}
        with self.assertRaisesRegex(ValueError, "incomplete Exp 3 matrix"):
            check_coverage(row, row, allow_partial=False)
        coverage = check_coverage(row, row, allow_partial=True)
        self.assertFalse(coverage["complete"])
        self.assertEqual(coverage["observed_runs"], 1)

    def test_drift_and_performance_identities_must_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            check_coverage({("NSL-KDD", "replay", 1): {}}, {}, allow_partial=True)


if __name__ == "__main__":
    unittest.main()
