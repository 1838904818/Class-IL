"""Synthetic checks only: no real classifier, labels or attribution extraction."""
import unittest
import numpy as np
from pilot_core import partition_indices, explanation_signal, choose, empirical_accept


class PilotTests(unittest.TestCase):
    def test_disjoint_repeatable_and_input_order_invariant(self):
        counts = dict(trigger=4, fit=8, acceptance=8, evaluation=8)
        a = partition_indices(range(40), range(40, 60), 0, "fixed", counts)
        b = partition_indices(reversed(range(40)), range(40, 60), 0, "fixed", counts)
        self.assertEqual(a, b)
        self.assertEqual(len(set(sum(a.values(), []))), 28)

    def test_reject_training_overlap(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            partition_indices(range(40), [1], 0, "x", dict(trigger=4, fit=8, acceptance=8, evaluation=8))

    def test_reject_rare_class_instead_of_resampling(self):
        with self.assertRaisesRegex(ValueError, "insufficient"):
            partition_indices([2], [0, 1], 6, "x", dict(trigger=1, fit=1, acceptance=1, evaluation=1))

    def test_stable_explanations(self):
        a = np.tile([1., -1.], (8, 3, 1))
        result = explanation_signal(a, a.copy())
        self.assertEqual(result["excess"], 0)
        self.assertFalse(result["statistical_certificate"])

    def test_stable_between_change_detected(self):
        a = np.tile([1., 0.], (8, 3, 1))
        b = np.tile([0., 1.], (8, 3, 1))
        self.assertEqual(explanation_signal(a, b)["excess"], 1)

    def test_method_noise_cannot_be_called_change(self):
        a = np.tile([1., 0.], (8, 3, 1))
        a[1::2] = [0., 1.]
        b = a[::-1].copy()
        result = explanation_signal(a, b)
        self.assertEqual(result["between"], 1)
        self.assertEqual(result["excess"], 0)

    def test_degenerate_or_nonfinite_not_zero_drift(self):
        a = np.tile([1., 0.], (8, 3, 1))
        for b in [np.zeros_like(a), np.full_like(a, np.nan), a[:4]]:
            with self.assertRaises(ValueError):
                explanation_signal(a, b)

    def test_noise_gate_only_changes_eligibility(self):
        c = [
            dict(class_id=0, error_increase=.2, attribution_validated=True, between=.8, excess=.0),
            dict(class_id=1, error_increase=.1, attribution_validated=True, between=.7, excess=.5)]
        self.assertEqual(choose(c, "error-only", "x")["target"], 0)
        self.assertEqual(choose(c, "raw-drift", "x")["target"], 0)
        self.assertEqual(choose(c, "noise-aware", "x")["target"], 1)

    def test_no_harm_means_audit_only_for_silent_drift(self):
        c = [dict(class_id=0, error_increase=0, attribution_validated=True, between=1, excess=1)]
        self.assertEqual(choose(c, "noise-aware", "x")["status"], "abstain")

    def test_missing_explanation_does_not_block_performance_control(self):
        c = [dict(class_id=0, error_increase=.2, attribution_validated=False)]
        self.assertEqual(choose(c, "error-only", "x")["target"], 0)
        self.assertEqual(choose(c, "noise-aware", "x")["status"], "unavailable-attribution")

    def test_threshold_is_strict_and_tie_is_deterministic(self):
        c = [dict(class_id=i, error_increase=.2, attribution_validated=True, between=.05, excess=.05) for i in [1, 0]]
        self.assertEqual(choose(c, "noise-aware", "x")["status"], "abstain")
        self.assertEqual(choose(c, "error-only", "x")["target"], 0)
        self.assertEqual(choose(c, "random-harmful", "x"), choose(c[::-1], "random-harmful", "x"))

    def test_empirical_acceptance_not_certificate(self):
        y = [0, 0, 1, 1, 2, 2, 3, 3]
        a = [1, 0, 1, 1, 2, 2, 3, 3]
        b = y
        result = empirical_accept(y, a, b, 0, [0, 1, 2, 3], [1, 2, 3])
        self.assertTrue(result["accepted_empirically"])
        self.assertFalse(result["statistical_certificate"])
        self.assertFalse(empirical_accept(y, a, a, 0, [0, 1, 2, 3], [1, 2, 3])["accepted_empirically"])

    def test_target_gain_cannot_hide_other_class_harm(self):
        y = [0, 0, 1, 1, 2, 2, 3, 3]
        a = [1, 0, 1, 1, 2, 2, 3, 3]
        b = [0, 0, 0, 1, 2, 2, 3, 3]
        self.assertFalse(empirical_accept(y, a, b, 0, [0, 1, 2, 3], [1, 2, 3])["accepted_empirically"])

    def test_unknown_class_is_blocked(self):
        with self.assertRaises(ValueError):
            empirical_accept([0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 3], 0, [0, 1, 2, 3], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
