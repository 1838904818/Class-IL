from __future__ import annotations

import unittest

from ncd.plot_exp3_partial_confusion import pooled_row_normalized_confusion


class NcdPlotExp3PartialConfusionTest(unittest.TestCase):
    def test_pools_counts_before_row_normalizing(self) -> None:
        runs = []
        for seed in [1, 2, 3, 4, 42]:
            runs.append(
                {
                    "method": "replay",
                    "seed": seed,
                    "final_confusion_matrix": [[seed, seed], [0, 2 * seed]],
                }
            )
        payload = {"dataset": "Toy", "class_order": ["A", "B"], "runs": runs}
        labels, seeds, pooled, normalized = pooled_row_normalized_confusion(
            payload, "replay"
        )
        self.assertEqual(labels, ["A", "B"])
        self.assertEqual(seeds, [1, 2, 3, 4, 42])
        self.assertEqual(pooled, [[52, 52], [0, 104]])
        self.assertEqual(normalized, [[50.0, 50.0], [0.0, 100.0]])

    def test_requires_complete_five_seed_slice(self) -> None:
        payload = {
            "dataset": "Toy",
            "class_order": ["A"],
            "runs": [
                {"method": "icarl", "seed": 1, "final_confusion_matrix": [[1]]}
            ],
        }
        with self.assertRaisesRegex(ValueError, "requires seeds"):
            pooled_row_normalized_confusion(payload, "icarl")


if __name__ == "__main__":
    unittest.main()
