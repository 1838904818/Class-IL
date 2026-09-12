"""Synthetic metadata tests, not a scientific experiment."""
import copy
import hashlib
import unittest
from action_diagnostic import ARMS, BINDINGS, EVENTS, audit_locked_choices


def sha(name):
    return hashlib.sha256(name.encode()).hexdigest()


def fixture():
    arms = []
    for arm in ARMS:
        active = arm != "audit-only"
        arms.append({"arm": arm, "choice_status": "attempt" if active else "audit-only",
                     "target": 0 if active else None,
                     "fit_status": "complete" if active else "not-attempted",
                     "decision_status": "complete", "accepted_empirically": active,
                     "fit_semantics_sha256": sha("same-fit") if active else None,
                     "decision_semantics_sha256": sha("same-decision") if active else sha("baseline"),
                     "bindings": {name: sha(name) for name in BINDINGS}})
    return {"schema": "etg-action-lock-metadata-v1", "source_kind": "synthetic",
            "locked_stage": "DECISIONS_LOCKED", "events_sha256": {e: sha(e) for e in EVENTS},
            "arms": arms}


def noise(snapshot):
    return next(a for a in snapshot["arms"] if a["arm"] == "noise-aware")


def primary(snapshot):
    return audit_locked_choices(snapshot)["primary"]["status"]


class DiagnosticTests(unittest.TestCase):
    def test_same_action_is_not_explanation_gain(self):
        snapshot = fixture(); original = copy.deepcopy(snapshot)
        result = audit_locked_choices(snapshot)
        self.assertEqual(result["primary"]["status"], "SAME_ACTION_NO_SELECTION_GAIN")
        self.assertEqual(len(result["pairs"]), 10)
        self.assertFalse(result["scientific_efficacy_established"])
        self.assertTrue(result["external_hash_verification_required"])
        self.assertEqual(snapshot, original)

    def test_different_target_needs_unopened_heldout(self):
        s = fixture(); r = noise(s)
        r.update(target=1, fit_semantics_sha256=sha("other-fit"), decision_semantics_sha256=sha("other-decision"))
        self.assertEqual(primary(s), "DIFFERENT_ACTION_REQUIRES_HELDOUT")
        self.assertFalse(audit_locked_choices(s)["primary"]["different_heldout_predictions_established"])

    def test_same_target_different_fit_is_not_comparable_gain(self):
        s = fixture(); noise(s)["fit_semantics_sha256"] = sha("walltime-or-real-param-difference")
        self.assertEqual(primary(s), "UNRESOLVED_SAME_TARGET")

    def test_same_fit_conflicting_decision_metadata(self):
        s = fixture(); noise(s)["decision_semantics_sha256"] = sha("different-diagnostics")
        self.assertEqual(primary(s), "INCONSISTENT_DECISION_SEMANTICS")

    def test_same_target_acceptance_disagreement_cannot_look_like_selection_gain(self):
        s = fixture(); noise(s)["accepted_empirically"] = False
        self.assertEqual(primary(s), "INCONSISTENT_DECISION_SEMANTICS")
        noise(s)["fit_semantics_sha256"] = sha("different-parameters")
        self.assertEqual(primary(s), "UNRESOLVED_SAME_TARGET")

    def test_different_targets_cannot_share_target_bound_hash(self):
        s = fixture(); noise(s)["target"] = 1
        self.assertEqual(primary(s), "INCONSISTENT_FIT_SEMANTICS")

    def test_both_rejected_retain_baseline_even_if_targets_differ(self):
        s = fixture()
        for r in s["arms"]:
            r["accepted_empirically"] = False
        noise(s).update(target=1, fit_semantics_sha256=sha("other-fit"), decision_semantics_sha256=sha("other-rejection"))
        self.assertEqual(primary(s), "SAME_ACTION_NO_SELECTION_GAIN")

    def test_abstention_differs_from_accepted_correction(self):
        s = fixture(); noise(s).update(choice_status="abstain", target=None,
                                      fit_status="not-attempted", fit_semantics_sha256=None,
                                      accepted_empirically=False, decision_semantics_sha256=sha("abstain"))
        self.assertEqual(primary(s), "DIFFERENT_ACTION_REQUIRES_HELDOUT")

    def test_missing_attribution_is_not_abstention(self):
        s = fixture(); noise(s).update(choice_status="unavailable-attribution", target=None,
                                      fit_status="not-attempted", fit_semantics_sha256=None,
                                      accepted_empirically=False, decision_semantics_sha256=sha("unavailable"))
        self.assertEqual(primary(s), "UNAVAILABLE_ATTRIBUTION")

    def test_resource_failure_is_not_zero_gain(self):
        s = fixture(); noise(s).update(fit_status="resource-failure", fit_semantics_sha256=None,
                                      decision_status="incomplete", accepted_empirically=None,
                                      decision_semantics_sha256=None)
        self.assertEqual(primary(s), "UNAVAILABLE_RESOURCE")

    def test_other_incomplete_stage_is_unavailable(self):
        for status in ("failed", "incomplete"):
            with self.subTest(status=status):
                s = fixture(); noise(s).update(fit_status=status, fit_semantics_sha256=None,
                                               decision_status="incomplete", accepted_empirically=None,
                                               decision_semantics_sha256=None)
                self.assertEqual(primary(s), "UNAVAILABLE_INCOMPLETE")

    def test_each_context_binding_is_material(self):
        for binding in BINDINGS:
            with self.subTest(binding=binding):
                s = fixture(); noise(s)["bindings"][binding] = sha("different")
                self.assertEqual(primary(s), "INCOMPARABLE_BINDINGS")

    def test_only_closed_hash_metadata_allowed(self):
        mutations = [lambda s: s.update(labels=[0]),
                     lambda s: noise(s).update(accuracy=1.0),
                     lambda s: noise(s).update(params=[1.0, 0.0]),
                     lambda s: s.update(locked_stage="FITS_LOCKED"),
                     lambda s: s["events_sha256"].pop("DECISIONS_LOCKED"),
                     lambda s: noise(s).update(target=True),
                     lambda s: noise(s).update(target=2),
                     lambda s: noise(s).update(fit_semantics_sha256="x" * 64),
                     lambda s: noise(s).update(accepted_empirically=1),
                     lambda s: s["arms"].pop(),
                     lambda s: noise(s).update(arm="error-only"),
                     lambda s: noise(s).update(fit_status="failed")]
        for change in mutations:
            with self.subTest(change=change):
                s = fixture(); change(s)
                with self.assertRaises(ValueError):
                    audit_locked_choices(s)

    def test_real_input_label_does_not_establish_verification(self):
        s = fixture(); s["source_kind"] = "real-inputs"
        result = audit_locked_choices(s)
        self.assertTrue(result["external_hash_verification_required"])
        self.assertFalse(result["scientific_efficacy_established"])
        self.assertFalse(result["cost_comparison_established"])


if __name__ == "__main__":
    unittest.main()
