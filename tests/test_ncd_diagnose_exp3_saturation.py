from __future__ import annotations

import unittest

from ncd.diagnose_exp3_saturation import (
    effect_gate_vote,
    lower_order_quantile,
    nearest_rank,
    noise_thresholds,
    positive_anchor_thresholds,
)


def row(
    explainer: str,
    effect: float,
    within: float,
    condition: str,
    *,
    truth_positive: bool = False,
    silent_eligible: bool = False,
    q_value: float = 0.01,
) -> dict[str, object]:
    return {
        "explainer": explainer,
        "ncd_effect": effect,
        "within_median_1_minus_spearman": within,
        "condition": condition,
        "strength": 1.0,
        "truth_positive": str(truth_positive),
        "silent_eligible": str(silent_eligible),
        "ncd_q_value": q_value,
    }


class NcdSaturationDiagnosticTest(unittest.TestCase):
    def test_quantiles_are_deterministic_order_statistics(self) -> None:
        values = list(range(1, 21))
        self.assertEqual(nearest_rank(values, 0.95), 19)
        self.assertEqual(lower_order_quantile(values, 0.05), 1)

    def test_noise_thresholds_use_only_negative_controls(self) -> None:
        rows = []
        for explainer in ("expected_gradients", "feature_ablation", "grad_x_input"):
            rows.extend(
                [
                    row(explainer, 0.1, 1.0, "N1_same_checkpoint"),
                    row(explainer, 0.2, 1.0, "N2_function_preserving"),
                    row(
                        explainer,
                        10.0,
                        1.0,
                        "P1_shortcut",
                        truth_positive=True,
                        silent_eligible=True,
                    ),
                ]
            )
        thresholds = noise_thresholds(rows, "absolute")
        self.assertEqual(set(thresholds.values()), {0.2})

    def test_positive_anchor_is_not_below_noise_threshold(self) -> None:
        rows = []
        for explainer in ("expected_gradients", "feature_ablation", "grad_x_input"):
            rows.extend(
                [
                    row(explainer, 0.1, 1.0, "N1_same_checkpoint"),
                    row(explainer, 0.2, 1.0, "N2_function_preserving"),
                    row(
                        explainer,
                        0.15,
                        1.0,
                        "P1_shortcut",
                        truth_positive=True,
                        silent_eligible=True,
                    ),
                ]
            )
        thresholds = positive_anchor_thresholds(rows, "absolute")
        self.assertEqual(set(thresholds.values()), {0.2})

    def test_effect_gate_requires_two_explainers_and_significance(self) -> None:
        explainers = ("expected_gradients", "feature_ablation", "grad_x_input")
        rows = [row(explainer, effect, 1.0, "P1_shortcut") for explainer, effect in zip(explainers, (0.5, 0.5, 0.1))]
        thresholds = {explainer: 0.2 for explainer in explainers}
        self.assertTrue(effect_gate_vote(rows, "absolute", thresholds, inclusive=False))
        rows[1]["ncd_q_value"] = 0.5
        self.assertFalse(effect_gate_vote(rows, "absolute", thresholds, inclusive=False))


if __name__ == "__main__":
    unittest.main()
