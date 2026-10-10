from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ncd.aggregate_exp3_full import (
    check_coverage,
    fixed_consensus_sensitivity_rows,
    mean_std,
    ncd_q_sensitivity_rows,
    portable_source_id,
    provenance_record,
    summarize_group,
)


class NcdAggregateExp3FullTest(unittest.TestCase):
    def test_provenance_is_hash_bound_and_portable(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "ncd" / "results" / "exp3" / "slice" / "exp3_result.json"
            source.parent.mkdir(parents=True)
            source.write_text("{}\n", encoding="utf-8")
            self.assertEqual(
                portable_source_id(source, root),
                "ncd/results/exp3/slice/exp3_result.json",
            )
            record = provenance_record(
                "drift",
                source,
                {
                    "job_id": "483686",
                    "source_commit": "d3a9db2",
                    "run_summaries": [{"dataset": "NSL-KDD"}],
                },
                root,
            )
            self.assertEqual(record["source"], "ncd/results/exp3/slice/exp3_result.json")
            self.assertEqual(record["datasets"], ["NSL-KDD"])
            self.assertEqual(record["job_id"], "483686")
            self.assertEqual(len(record["sha256"]), 64)
            self.assertNotIn(str(root), str(record))

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

    def test_ncd_q_sensitivity_recomputes_cross_explainer_consensus(self) -> None:
        rows = []
        for class_id, recall_delta, q_values in (
            (0, -0.01, (0.001, 0.02, 0.20)),
            (1, -0.07, (0.001, 0.002, 0.003)),
        ):
            for index, q_value in enumerate(q_values):
                rows.append(
                    {
                        "dataset": "NSL-KDD",
                        "method": "replay",
                        "seed": "1",
                        "checkpoint_before": "0",
                        "checkpoint_after": "1",
                        "class_id": str(class_id),
                        "explainer": f"e{index}",
                        "recall_delta": str(recall_delta),
                        "q_value": str(q_value),
                        "ncd_effect": "0.1",
                    }
                )
        result = ncd_q_sensitivity_rows(rows)
        primary = next(
            row
            for row in result
            if row["silence_tolerance"] == 0.05 and row["q_threshold"] == 0.025
        )
        self.assertEqual(primary["silent_units"], 1)
        self.assertEqual(primary["ncd_consensus_drift_units"], 1)
        self.assertEqual(primary["ncd_consensus_drift_rate"], 1.0)
        self.assertEqual(primary["ncd_all_explainer_agreement_rate"], 0.0)

    def test_fixed_consensus_sensitivity_uses_two_of_three_vote(self) -> None:
        rows = []
        for index, b1 in enumerate((0.4, 0.6, 0.9)):
            row = {
                "dataset": "NSL-KDD",
                "method": "replay",
                "seed": "1",
                "checkpoint_before": "0",
                "checkpoint_after": "1",
                "class_id": "0",
                "explainer": f"e{index}",
                "recall_delta": "-0.01",
                "q_value": "0.1",
                "ncd_effect": "0.1",
            }
            for top_k in (5, 10, 15, 20):
                row[f"b1_jaccard_k{top_k}"] = str(b1)
                row[f"b2_jaccard_k{top_k}"] = str(1.0 - b1)
            rows.append(row)
        result = fixed_consensus_sensitivity_rows(rows)
        primary = next(
            row
            for row in result
            if row["silence_tolerance"] == 0.05
            and row["top_k"] == 15
            and row["jaccard_threshold"] == 0.7
        )
        self.assertEqual(primary["silent_units"], 1)
        self.assertEqual(primary["b1_consensus_drift_units"], 1)
        self.assertEqual(primary["b2_consensus_drift_units"], 1)


if __name__ == "__main__":
    unittest.main()
