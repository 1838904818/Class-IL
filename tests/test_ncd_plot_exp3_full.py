from __future__ import annotations

import unittest

from ncd.plot_exp3_full import (
    DATASETS,
    METHODS,
    SEEDS,
    merge_performance_payloads,
    pooled_dataset_confusion,
    pooled_dataset_curves,
)


class NcdPlotExp3FullTest(unittest.TestCase):
    @staticmethod
    def _payload(dataset: str, methods: tuple[str, ...]) -> dict[str, object]:
        runs = []
        for method in methods:
            for seed in SEEDS:
                runs.append(
                    {
                        "method": method,
                        "seed": seed,
                        "final_confusion_matrix": [[seed, seed], [0, 2 * seed]],
                    }
                )
        return {"dataset": dataset, "class_order": ["A", "B"], "runs": runs}

    def test_merges_split_packages_and_requires_complete_matrix(self) -> None:
        payloads = []
        for dataset in DATASETS:
            payloads.append(self._payload(dataset, METHODS[:2]))
            payloads.append(self._payload(dataset, METHODS[2:]))
        merged = merge_performance_payloads(payloads)
        self.assertEqual([payload["dataset"] for payload in merged], list(DATASETS))
        self.assertTrue(all(len(payload["runs"]) == 35 for payload in merged))

    def test_incomplete_matrix_fails_closed(self) -> None:
        payloads = [self._payload(dataset, METHODS) for dataset in DATASETS]
        payloads[0]["runs"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete performance matrix"):
            merge_performance_payloads(payloads)

    def test_pools_all_runs_before_row_normalizing(self) -> None:
        payload = self._payload("NSL-KDD", METHODS)
        labels, pooled, normalized = pooled_dataset_confusion(payload)
        self.assertEqual(labels, ["A", "B"])
        self.assertEqual(pooled, [[364, 364], [0, 728]])
        self.assertEqual(normalized, [[50.0, 50.0], [0.0, 100.0]])

    def test_sensitivity_pools_counts_by_dataset(self) -> None:
        rows = []
        for dataset in DATASETS:
            rows.extend(
                [
                    {
                        "dataset": dataset,
                        "silence_tolerance": "0.05",
                        "q_threshold": "0.01",
                        "silent_units": "1",
                        "ncd_consensus_drift_units": "1",
                    },
                    {
                        "dataset": dataset,
                        "silence_tolerance": "0.05",
                        "q_threshold": "0.01",
                        "silent_units": "9",
                        "ncd_consensus_drift_units": "0",
                    },
                ]
            )
        curves = pooled_dataset_curves(
            rows, "q_threshold", "ncd_consensus_drift_units"
        )
        self.assertEqual(curves["NSL-KDD"][0.01], 10.0)


if __name__ == "__main__":
    unittest.main()
