from __future__ import annotations

import unittest

from ncd.plot_exp3_partial_sensitivity import pooled_curves


class NcdPlotExp3PartialSensitivityTest(unittest.TestCase):
    def test_pools_counts_instead_of_averaging_seed_rates(self) -> None:
        rows = []
        for dataset, method in (
            ("NSL-KDD", "replay"),
            ("NSL-KDD", "icarl"),
            ("UNSW-NB15", "replay"),
            ("UNSW-NB15", "icarl"),
        ):
            rows.extend(
                [
                    {
                        "dataset": dataset,
                        "method": method,
                        "silence_tolerance": "0.05",
                        "q_threshold": "0.01",
                        "silent_units": "1",
                        "ncd_consensus_drift_units": "1",
                    },
                    {
                        "dataset": dataset,
                        "method": method,
                        "silence_tolerance": "0.05",
                        "q_threshold": "0.01",
                        "silent_units": "9",
                        "ncd_consensus_drift_units": "0",
                    },
                ]
            )
        curves = pooled_curves(rows, "q_threshold", "ncd_consensus_drift_units")
        self.assertEqual(curves[("NSL-KDD", "replay")][0.01], 10.0)


if __name__ == "__main__":
    unittest.main()
