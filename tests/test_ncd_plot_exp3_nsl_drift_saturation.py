from __future__ import annotations

import unittest

from ncd.plot_exp3_nsl_drift_saturation import chart_rows, q_curve


class PlotExp3NslDriftSaturationTest(unittest.TestCase):
    def payload(self):
        methods = []
        for method, silent, fixed, ncd in (
            ("naive", 35, 9, 34),
            ("ewc", 35, 8, 34),
            ("lwf", 35, 13, 34),
            ("replay-herding", 23, 10, 23),
            ("replay-dpmeans", 29, 12, 29),
        ):
            methods.append(
                {
                    "method": method,
                    "silent_units": silent,
                    "b1_drift_units": fixed,
                    "b2_drift_units": fixed,
                    "ncd_drift_units": ncd,
                }
            )
        return {
            "dataset": "NSL-KDD",
            "scope": {"runs": 25},
            "pooled_result": {
                "silent_units": 157,
                "b1_drift_units": 52,
                "b2_drift_units": 52,
                "ncd_drift_units": 154,
            },
            "method_summaries": methods,
            "ncd_q_sensitivity": [
                {"q": q, "drift_units": drift, "silent_units": 157, "drift_rate": drift / 157}
                for q, drift in ((0.01, 147), (0.025, 150), (0.05, 154), (0.1, 156))
            ],
        }

    def test_reconciles_method_counts_and_orders_rows(self) -> None:
        rows = chart_rows(self.payload())
        self.assertEqual([row["method"] for row in rows], ["naive", "ewc", "lwf", "replay-herding", "replay-dpmeans"])
        self.assertAlmostEqual(rows[0]["fixed_rate"], 100 * 9 / 35)
        self.assertAlmostEqual(rows[-1]["ncd_rate"], 100.0)

    def test_q_curve_recomputes_display_rates(self) -> None:
        rows = q_curve(self.payload())
        self.assertEqual([row[0] for row in rows], [0.01, 0.025, 0.05, 0.1])
        self.assertAlmostEqual(rows[2][3], 100 * 154 / 157)

    def test_rejects_nonidentical_b1_b2_claim(self) -> None:
        payload = self.payload()
        payload["method_summaries"][0]["b2_drift_units"] = 8
        with self.assertRaisesRegex(ValueError, "B1 and B2 are not identical"):
            chart_rows(payload)


if __name__ == "__main__":
    unittest.main()
