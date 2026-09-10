"""Synthetic selector contract tests, not real-data or statistical efficacy tests.

No training, fitting, attribution extraction, network calls or persistent ledger
writes. Repeated artificial vectors exercise branch behavior only.
"""
import copy
import unittest

import numpy as np

from make_demo import prospective_policy
from score_core import candidate_pool


class MinimalETGContrastTests(unittest.TestCase):
    def setUp(self):
        groups = 128
        self.classes = ["application-a", "application-b", "application-c"]
        n = 3 * groups
        self.manifest = {"classes": self.classes, "old_classes": self.classes[:2], "increment": 1}
        self.policy = prospective_policy(groups, n)  # Software fixture, not real-study settings.
        a = np.tile([1.0, 0.0], (4, n, 1)).tolist()
        d = np.tile([0.0, 1.0], (4, n, 1)).tolist()
        self.part = {
            "labels": self.classes * groups,
            "row_ids": [f"row-{i}" for i in range(n)],
            "group_ids": [f"synthetic-group-{i}" for i in range(groups) for _ in self.classes],
            "old_head_logits": [[4., -4.], [-4., 4.], [0., 0.]] * groups,
            "new_head_logits": [[-4., -4., 4.]] * n,
            "old_router_raw": [[0., 0.]] * n,
            "new_router_raw": [[0., 0., 0.]] * n,
            "explanations": {
                c: {"method": self.policy["trigger"]["explanation_method"],
                    "independent_draws_by_group": True, "A": a, "D": d}
                for c in self.classes[:2]
            },
        }

    def pool(self, arm):
        return candidate_pool(self.part, self.manifest, self.policy, arm)

    def test_same_harm_order_when_explanation_gate_passes(self):
        control, candidate = self.pool("error-only"), self.pool("expansion-w0")
        self.assertEqual(control["ordered_targets"], self.classes[:2])
        self.assertEqual(control["ordered_targets"], candidate["ordered_targets"])
        self.assertEqual([x["priority"] for x in control["candidates"]],
                         [x["priority"] for x in candidate["candidates"]])

    def test_no_detectable_explanation_change_abstains_without_altering_control(self):
        for info in self.part["explanations"].values():
            info["D"] = copy.deepcopy(info["A"])
        self.assertEqual(self.pool("expansion-w0")["ordered_targets"], [])
        self.assertEqual(self.pool("error-only")["ordered_targets"], self.classes[:2])

    def test_explanation_change_without_harm_does_not_trigger_repair(self):
        self.part["new_head_logits"] = [[4., -4., -8.], [-4., 4., -8.], [-4., -4., 4.]] * 128
        self.assertEqual(self.pool("error-only")["ordered_targets"], [])
        self.assertEqual(self.pool("expansion-w0")["ordered_targets"], [])

    def test_missing_explanations_fail_closed_only_for_candidate(self):
        del self.part["explanations"]
        candidate = self.pool("expansion-w0")
        self.assertEqual(candidate["ordered_targets"], [])
        self.assertTrue(all("missing" in x["reason"] for x in candidate["candidates"]))
        self.assertEqual(self.pool("error-only")["ordered_targets"], self.classes[:2])

    def test_method_mismatch_is_not_silent_fallback(self):
        for info in self.part["explanations"].values():
            info["method"] = "different-software-fixture"
        candidate = self.pool("expansion-w0")
        self.assertEqual(candidate["ordered_targets"], [])
        self.assertTrue(all("mismatch" in x["reason"] for x in candidate["candidates"]))

    def test_decomposition_weight_does_not_affect_primary_candidate(self):
        original = self.pool("expansion-w0")
        self.policy["trigger"]["state_priority_weight"] = 1000.0
        changed = self.pool("expansion-w0")
        self.assertEqual(original["ordered_targets"], changed["ordered_targets"])
        self.assertEqual([x["priority"] for x in original["candidates"]],
                         [x["priority"] for x in changed["candidates"]])

    def test_selectors_do_not_mutate_scores_or_explanations(self):
        original = copy.deepcopy(self.part)
        for arm in ("audit-only", "error-only", "expansion-w0"):
            self.pool(arm)
        self.assertEqual(original, self.part)


if __name__ == "__main__":
    unittest.main()
