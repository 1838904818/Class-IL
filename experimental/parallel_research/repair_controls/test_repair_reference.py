"""Fabricated fixtures test software invariants only, never model performance."""
from dataclasses import replace
import math
import unittest

from repair_reference import (
    AcceptanceEvidence, AcceptancePolicy, Arm, Binding, BudgetLedger, BudgetPlan,
    Candidate, FitEvidence, GateError, Interval, MetricEvidence, Partitions,
    RepairPackage, Sample, Semantics, TriggerPolicy, acceptance_gate,
    compare_active_budgets, rank_candidates,
)


SHA = "a" * 64  # Deliberately fabricated fixture digest; not an experiment.


def partitions():
    def rows(role):
        return tuple(Sample(f"{role}-{c}-{i}", f"{role}-group-{c}-{i}", 1, 1, c)
                     for c in ("old-a", "old-b", "new-c") for i in range(2))
    return Partitions(1, rows("trigger"), rows("fit"), rows("acceptance"),
                      (Sample("future-1", "future-group-1", 2, None, None),))


def trigger_policy():
    # Toy boundary values chosen solely to exercise branches, not study defaults.
    return TriggerPolicy(2, .05, .1, .02, 1., 1e-12, 2, 1, 17, "toy-distance", ("old-a", "old-b"))


def candidate(class_id="old-a", **changes):
    p = partitions()
    c = Candidate(class_id, class_id, 1, p.support("trigger", class_id), p.fingerprint(),
                  Binding(SHA, SHA, SHA, SHA, SHA, SHA), Interval(.1, .2), Interval(.2, .3),
                  Interval(.4, .5), Interval(.1, .2), "toy-distance", True, True, True,
                  -.2, 0., -.1, -.3)
    return replace(c, **changes)


def plan(**changes):
    return replace(BudgetPlan(18, 20, 1, 10, 12, RepairPackage("toy-calibration-v1", 10, 12),
                              "toy fixed optimiser steps plus forward rows; not measured time"), **changes)


def charged(arm=Arm.EXPANSION_AWARE, budget=None):
    ledger = BudgetLedger(arm, budget or plan(), partitions())
    ledger.charge_trigger([r.row_id for r in partitions().trigger], 5)
    return ledger


def acceptance_policy(semantics=Semantics.APPLICATION, **changes):
    attack = semantics == Semantics.ATTACK_BINARY
    return replace(AcceptancePolicy(semantics, ("old-a", "old-b"), ("new-c",), 2, 2, 2,
                                   .05, .02, .03, .01 if attack else None, .01 if attack else None,
                                   ("old-a", "new-c") if attack else (), ("old-b",) if attack else (), SHA), **changes)


def acceptance_evidence(attack=False, **changes):
    p = partitions()
    metrics = [MetricEvidence("class_error_delta:old-a", Interval(-.2, -.1), p.support("acceptance", "old-a")),
               MetricEvidence("class_error_delta:old-b", Interval(-.01, .01), p.support("acceptance", "old-b")),
               MetricEvidence("class_error_delta:new-c", Interval(-.01, .02), p.support("acceptance", "new-c")),
               MetricEvidence("target_error_improvement", Interval(.1, .2), p.support("acceptance", "old-a"))]
    if attack:
        metrics.extend([
            MetricEvidence("missed_attack_rate_delta", Interval(-.1, 0.),
                           p.support("acceptance", "old-a") + p.support("acceptance", "new-c")),
            MetricEvidence("false_positive_rate_delta", Interval(-.01, .005), p.support("acceptance", "old-b")),
        ])
    return replace(AcceptanceEvidence("old-a", "acceptance", p.fingerprint(), SHA, "b" * 64, SHA,
                                      True, tuple(metrics), True, False), **changes)


def fit_evidence(**changes):
    p = partitions()
    return replace(FitEvidence("old-a", "toy-calibration-v1", p.fingerprint(),
                               tuple(r.row_id for r in p.fit), SHA, "b" * 64, SHA), **changes)


def decide(evidence=None, policy=None, fitted=None, c=None):
    ledger = charged()
    reservation = ledger.reserve_repair(candidate(), "toy-calibration-v1", trigger_policy(), policy or acceptance_policy())
    return acceptance_gate(c or candidate(), reservation, fitted or fit_evidence(),
                           evidence or acceptance_evidence(), partitions(), policy or acceptance_policy(), trigger_policy(),
                           Arm.EXPANSION_AWARE)


class PartitionTests(unittest.TestCase):
    def test_valid_available_disjoint_partitions(self):
        self.assertIsInstance(partitions().validate(), Partitions)

    def test_row_overlap_fails(self):
        p = partitions()
        with self.assertRaises(GateError):
            replace(p, fit=p.trigger).validate()

    def test_duplicate_within_partition_fails(self):
        p = partitions()
        with self.assertRaises(GateError):
            replace(p, fit=(p.fit[0], p.fit[0])).validate()

    def test_capture_group_leakage_fails(self):
        p = partitions()
        with self.assertRaises(GateError):
            replace(p, acceptance=(replace(p.acceptance[0], leakage_group=p.fit[0].leakage_group),)).validate()

    def test_future_training_labels_fail(self):
        p = partitions()
        with self.assertRaises(GateError):
            replace(p, fit=(replace(p.fit[0], label_available_increment=2),)).validate()

    def test_no_labels_no_supervised_fit(self):
        p = partitions()
        with self.assertRaises(GateError):
            replace(p, fit=(replace(p.fit[0], class_id=None, label_available_increment=None),)).validate()

    def test_future_evaluation_labels_cannot_enter(self):
        p = partitions()
        with self.assertRaises(GateError):
            replace(p, subsequent=(replace(p.subsequent[0], class_id="old-a", label_available_increment=2),)).validate()

    def test_evaluation_must_be_subsequent(self):
        p = partitions()
        with self.assertRaises(GateError):
            replace(p, subsequent=(replace(p.subsequent[0], observed_increment=1),)).validate()


class RankingTests(unittest.TestCase):
    def rank(self, arm=Arm.EXPANSION_AWARE, cs=None, policy=None):
        return rank_candidates(arm, cs if cs is not None else [candidate()], partitions(), policy or trigger_policy())

    def test_all_six_arms_have_defined_reference_behavior(self):
        for arm in Arm:
            result = self.rank(arm)
            self.assertEqual(result.ordered_ids, () if arm == Arm.AUDIT_ONLY else ("old-a",))

    def test_explanation_only_change_never_implies_harm(self):
        self.assertEqual(self.rank(cs=[candidate(error_increase=Interval(-.1, .1))]).ordered_ids, ())

    def test_noise_overlap_abstains(self):
        self.assertEqual(self.rank(cs=[candidate(within_method_noise=Interval(.3, .5))]).ordered_ids, ())

    def test_cross_method_uncertainty_is_blocked(self):
        result = self.rank(cs=[candidate(method="different-method")])
        self.assertEqual(result.ordered_ids, ())
        self.assertIn("method-matched", result.blocked[0][1])

    def test_missing_fidelity_blocks(self):
        for field in ("score_validated", "attribution_validated", "within_method_estimated"):
            self.assertEqual(self.rank(cs=[candidate(**{field: False})]).ordered_ids, ())

    def test_tiny_support_blocks(self):
        self.assertTrue(self.rank(policy=replace(trigger_policy(), min_trigger_support=3)).blocked)

    def test_nonfinite_and_bad_identity_block(self):
        for c in (candidate(state_effect=math.nan), candidate(total_margin_change=1), candidate(new_rival_effect=.1)):
            self.assertTrue(self.rank(cs=[c]).blocked)

    def test_head_creation_is_blocked(self):
        self.assertTrue(self.rank(cs=[candidate(requests_new_head=True)]).blocked)

    def test_new_class_cannot_be_selected_as_old_repair_target(self):
        self.assertTrue(self.rank(Arm.RANDOM, [candidate("new-c")]).blocked)

    def test_future_or_foreign_evidence_is_blocked(self):
        for c in (candidate(increment=2), candidate(partition_sha256="b" * 64),
                  candidate(trigger_row_ids=("future-1",)), candidate(binding=replace(candidate().binding, code_sha256="missing"))):
            self.assertTrue(self.rank(cs=[c]).blocked)

    def test_harmful_pure_expansion_is_not_filtered_out(self):
        result = self.rank(cs=[candidate(state_effect=0., normalization_effect=-.2, new_rival_effect=-.1)])
        self.assertEqual(result.ordered_ids, ("old-a",))

    def test_decomposition_changes_priority_and_zero_weight_ablates_it(self):
        expansion = candidate("old-a", state_effect=0., normalization_effect=-.2)
        state = candidate("old-b")
        self.assertEqual(self.rank(cs=[expansion, state]).ordered_ids, ("old-b", "old-a"))
        self.assertEqual(self.rank(cs=[expansion, state], policy=replace(trigger_policy(), state_priority_weight=0)).ordered_ids,
                         ("old-a", "old-b"))

    def test_random_is_seeded_and_input_order_invariant(self):
        cs = [candidate(), candidate("old-b")]
        self.assertEqual(self.rank(Arm.RANDOM, cs).ordered_ids, self.rank(Arm.RANDOM, list(reversed(cs))).ordered_ids)

    def test_periodic_does_not_fire_off_schedule(self):
        self.assertEqual(self.rank(Arm.PERIODIC, policy=replace(trigger_policy(), periodic_phase=0)).ordered_ids, ())

    def test_simple_controls_do_not_use_explanation_threshold(self):
        c = candidate(explanation_drift=Interval(0, 0), attribution_validated=False)
        for arm in (Arm.ERROR_ONLY, Arm.DISAGREEMENT_ONLY, Arm.RANDOM):
            self.assertEqual(self.rank(arm, [c]).ordered_ids, ("old-a",))

    def test_simple_controls_need_no_attribution_or_decomposition(self):
        c = candidate(binding=replace(candidate().binding, attribution_validation_sha256=None,
                                       uncertainty_validation_sha256=None), explanation_drift=None,
                      within_method_noise=None, method=None, attribution_validated=False,
                      within_method_estimated=False, state_effect=None, normalization_effect=None,
                      new_rival_effect=None, total_margin_change=None)
        for arm in (Arm.RANDOM, Arm.PERIODIC, Arm.ERROR_ONLY, Arm.DISAGREEMENT_ONLY):
            self.assertEqual(self.rank(arm, [c]).ordered_ids, ("old-a",))
        self.assertTrue(self.rank(Arm.EXPANSION_AWARE, [c]).blocked)

    def test_random_and_periodic_do_not_need_error_or_disagreement(self):
        c = candidate(error_increase=None, disagreement=None)
        for arm in (Arm.RANDOM, Arm.PERIODIC):
            self.assertEqual(self.rank(arm, [c]).ordered_ids, ("old-a",))
        self.assertTrue(self.rank(Arm.ERROR_ONLY, [c]).blocked)
        self.assertTrue(self.rank(Arm.DISAGREEMENT_ONLY, [c]).blocked)

    def test_duplicate_target_and_unknown_arm_fail(self):
        with self.assertRaises(GateError):
            self.rank(cs=[candidate(), candidate()])
        with self.assertRaises(GateError):
            self.rank("not-an-arm")

    def test_thresholds_are_required_and_validated(self):
        with self.assertRaises(TypeError):
            TriggerPolicy()
        with self.assertRaises(GateError):
            self.rank(policy=replace(trigger_policy(), state_priority_weight=math.inf))


class BudgetTests(unittest.TestCase):
    def test_unique_labels_not_double_charged(self):
        ledger = charged()
        ledger.charge_trigger([r.row_id for r in partitions().trigger], 0)
        self.assertEqual(ledger.usage()["unique_labels"], 6)

    def test_atomic_failed_charge_does_not_mutate(self):
        ledger = charged()
        before = ledger.usage()
        with self.assertRaises(GateError):
            ledger.charge_trigger([], 100)
        self.assertEqual(ledger.usage(), before)

    def test_atomic_failed_reservation_does_not_mutate(self):
        for budget in (plan(unique_label_cap=17), plan(max_repairs=0), plan(fit_step_cap=9), plan(acceptance_row_cap=11)):
            ledger = charged(budget=budget)
            before = ledger.usage()
            with self.assertRaises(GateError):
                ledger.reserve_repair(candidate(), "toy-calibration-v1", trigger_policy(), acceptance_policy())
            self.assertEqual(ledger.usage(), before)

    def test_no_free_trigger_labels(self):
        ledger = BudgetLedger(Arm.RANDOM, plan(), partitions())
        with self.assertRaises(GateError):
            ledger.reserve_repair(candidate(), "toy-calibration-v1", trigger_policy(), acceptance_policy())

    def test_uncharged_evaluation_labels_cannot_be_trigger_labels(self):
        with self.assertRaises(GateError):
            charged().charge_trigger(["future-1"], 0)

    def test_audit_only_cannot_repair(self):
        with self.assertRaises(GateError):
            charged(Arm.AUDIT_ONLY).reserve_repair(candidate(), "toy-calibration-v1", trigger_policy(), acceptance_policy())

    def test_operation_mismatch_fails(self):
        with self.assertRaises(GateError):
            charged().reserve_repair(candidate(), "different-repair", trigger_policy(), acceptance_policy())

    def test_rejected_or_failed_attempt_cannot_reuse_acceptance(self):
        ledger = charged(budget=plan(max_repairs=2, fit_step_cap=20, acceptance_row_cap=24))
        ledger.reserve_repair(candidate(), "toy-calibration-v1", trigger_policy(), acceptance_policy())
        with self.assertRaises(GateError):
            ledger.reserve_repair(candidate("old-b"), "toy-calibration-v1", trigger_policy(), acceptance_policy())
        self.assertEqual(ledger.usage()["repairs"], 1)

    def test_undercharged_acceptance_scoring_fails(self):
        ledger = charged(budget=plan(package=RepairPackage("toy-calibration-v1", 10, 6)))
        with self.assertRaises(GateError):
            ledger.reserve_repair(candidate(), "toy-calibration-v1", trigger_policy(), acceptance_policy())

    def test_active_matching_excludes_audit_and_disclaims_measured_compute(self):
        ledgers = [charged(arm) for arm in Arm]
        for ledger in ledgers:
            if ledger.arm != Arm.AUDIT_ONLY:
                ledger.reserve_repair(candidate(), "toy-calibration-v1", trigger_policy(), acceptance_policy())
        report = compare_active_budgets(ledgers)
        self.assertTrue(report["active_intervention_ledger_matched"])
        self.assertFalse(report["actual_compute_verified"])
        self.assertFalse(report["audit_only_active_compute_matched"])

    def test_equal_caps_unequal_use_is_not_matching(self):
        ledgers = [charged(arm) for arm in Arm if arm != Arm.AUDIT_ONLY]
        ledgers[0].reserve_repair(candidate(), "toy-calibration-v1", trigger_policy(), acceptance_policy())
        with self.assertRaises(GateError):
            compare_active_budgets(ledgers)

    def test_missing_arm_is_not_full_control_comparison(self):
        with self.assertRaises(GateError):
            compare_active_budgets([charged()])

    def test_diagnostic_overhead_is_explicit_not_forced_equal(self):
        ledgers = [charged(arm) for arm in Arm if arm != Arm.AUDIT_ONLY]
        for ledger in ledgers:
            ledger.reserve_repair(candidate(), "toy-calibration-v1", trigger_policy(), acceptance_policy())
        ledgers[-1].charge_trigger([], 2)
        report = compare_active_budgets(ledgers)
        self.assertTrue(report["active_intervention_ledger_matched"])
        self.assertFalse(report["equal_diagnostic_ledger_units"])
        self.assertEqual(report["diagnostic_units_by_arm"][Arm.EXPANSION_AWARE.value], 7)

    def test_zero_repairs_is_not_an_active_intervention_comparison(self):
        with self.assertRaises(GateError):
            compare_active_budgets([charged(arm) for arm in Arm if arm != Arm.AUDIT_ONLY])

    def test_blocked_expansion_trigger_cannot_bypass_ranking_via_reservation(self):
        c = candidate(attribution_validated=False, within_method_estimated=False,
                      error_increase=Interval(-.1, 0.))
        self.assertTrue(rank_candidates(Arm.EXPANSION_AWARE, [c], partitions(), trigger_policy()).blocked)
        ledger = charged()
        before = ledger.usage()
        with self.assertRaises(GateError):
            ledger.reserve_repair(c, "toy-calibration-v1", trigger_policy(), acceptance_policy())
        self.assertEqual(ledger.usage(), before)

    def test_periodic_off_schedule_cannot_reserve(self):
        with self.assertRaises(GateError):
            charged(Arm.PERIODIC).reserve_repair(candidate(), "toy-calibration-v1",
                                                replace(trigger_policy(), periodic_phase=0), acceptance_policy())

    def test_nonharmful_expansion_and_error_candidates_cannot_reserve(self):
        for arm in (Arm.EXPANSION_AWARE, Arm.ERROR_ONLY):
            with self.assertRaises(GateError):
                charged(arm).reserve_repair(candidate(error_increase=Interval(-.1, 0.)),
                                            "toy-calibration-v1", trigger_policy(), acceptance_policy())


class AcceptanceTests(unittest.TestCase):
    def test_valid_fixture_pass_is_not_model_repair_or_real_utility(self):
        result = decide()
        self.assertEqual(result["decision"], "ACCEPT_FOR_RESEARCH_ONLY")
        self.assertFalse(result["repair_applied"])
        self.assertFalse(result["real_data_utility_established"])

    def test_insufficient_fit_and_acceptance_support_fail_closed(self):
        for changes in ({"min_fit_per_class": 3}, {"min_acceptance_per_class": 3}, {"min_metric_support": 3}):
            with self.assertRaises(GateError):
                decide(policy=acceptance_policy(**changes))

    def test_later_test_outcomes_cannot_accept_candidate(self):
        with self.assertRaises(GateError):
            decide(evidence=acceptance_evidence(partition_role="subsequent"))

    def test_candidate_cannot_change_after_fit_or_reservation(self):
        with self.assertRaises(GateError):
            decide(evidence=acceptance_evidence(candidate_id="old-b"))
        with self.assertRaises(GateError):
            decide(fitted=fit_evidence(candidate_id="old-b"))

    def test_same_id_mutated_candidate_and_stale_increment_fail(self):
        with self.assertRaises(GateError):
            decide(c=candidate(error_increase=Interval(.3, .4)))
        with self.assertRaises(GateError):
            decide(c=candidate(increment=2))

    def test_policy_changes_after_reservation_fail(self):
        c = candidate()
        reservation = charged().reserve_repair(c, "toy-calibration-v1", trigger_policy(), acceptance_policy())
        with self.assertRaises(GateError):
            acceptance_gate(c, reservation, fit_evidence(), acceptance_evidence(), partitions(),
                            acceptance_policy(), replace(trigger_policy(), min_error_increase=.01), Arm.EXPANSION_AWARE)

    def test_acceptance_rechecks_ineligible_content_even_with_rewritten_hash(self):
        c = candidate(attribution_validated=False, within_method_estimated=False, error_increase=Interval(-.1, 0.))
        reservation = charged().reserve_repair(candidate(), "toy-calibration-v1", trigger_policy(), acceptance_policy())
        rewritten = replace(reservation, candidate_sha256=c.fingerprint())
        with self.assertRaises(GateError):
            acceptance_gate(c, rewritten, fit_evidence(), acceptance_evidence(), partitions(),
                            acceptance_policy(), trigger_policy(), Arm.EXPANSION_AWARE)

    def test_acceptance_policy_cannot_be_loosened_after_reservation(self):
        c = candidate()
        reservation = charged().reserve_repair(c, "toy-calibration-v1", trigger_policy(), acceptance_policy())
        with self.assertRaises(GateError):
            acceptance_gate(c, reservation, fit_evidence(), acceptance_evidence(), partitions(),
                            acceptance_policy(min_target_improvement=0), trigger_policy(), Arm.EXPANSION_AWARE)

    def test_arm_binding_cannot_change_between_reserve_and_accept(self):
        c = candidate()
        reservation = charged().reserve_repair(c, "toy-calibration-v1", trigger_policy(), acceptance_policy())
        with self.assertRaises(GateError):
            acceptance_gate(c, replace(reservation, arm=Arm.RANDOM), fit_evidence(), acceptance_evidence(), partitions(),
                            acceptance_policy(), trigger_policy(), Arm.EXPANSION_AWARE)

    def test_future_or_acceptance_labels_cannot_fit(self):
        with self.assertRaises(GateError):
            decide(fitted=fit_evidence(fit_row_ids=("acceptance-old-a-0",)))

    def test_state_hashes_and_receipt_are_required(self):
        with self.assertRaises(GateError):
            decide(fitted=fit_evidence(fit_record_sha256="missing"))
        with self.assertRaises(GateError):
            decide(evidence=acceptance_evidence(repaired_state_sha256="c" * 64))

    def test_no_alert_suppression_or_new_heads(self):
        for changes in ({"preserves_raw_alerts": False}, {"creates_new_head": True}):
            with self.assertRaises(GateError):
                decide(evidence=acceptance_evidence(**changes))

    def test_uncorrected_or_changed_uncertainty_is_blocked(self):
        for changes in ({"simultaneous_intervals": False}, {"uncertainty_protocol_sha256": "b" * 64}):
            with self.assertRaises(GateError):
                decide(evidence=acceptance_evidence(**changes))

    def test_new_class_harm_rejects_despite_target_benefit(self):
        e = acceptance_evidence()
        metrics = tuple(replace(m, interval=Interval(.1, .2)) if m.name == "class_error_delta:new-c" else m for m in e.metrics)
        result = decide(evidence=replace(e, metrics=metrics))
        self.assertEqual(result["decision"], "REJECT")
        self.assertIn("class_error_delta:new-c", result["failed_constraints"])

    def test_contradictory_target_benefit_is_blocked(self):
        e = acceptance_evidence()
        metrics = tuple(replace(m, interval=Interval(.5, .6)) if m.name == "target_error_improvement" else m for m in e.metrics)
        with self.assertRaises(GateError):
            decide(evidence=replace(e, metrics=metrics))

    def test_malaya_application_labels_cannot_report_attack_metrics(self):
        with self.assertRaises(GateError):
            decide(evidence=acceptance_evidence(attack=True))
        with self.assertRaises(GateError):
            decide(policy=acceptance_policy(max_fpr_increase=.01))

    def test_binary_attack_mapping_requires_both_constraints(self):
        result = decide(evidence=acceptance_evidence(attack=True), policy=acceptance_policy(Semantics.ATTACK_BINARY))
        self.assertEqual(result["decision"], "ACCEPT_FOR_RESEARCH_ONLY")
        with self.assertRaises(GateError):
            decide(policy=acceptance_policy(Semantics.ATTACK_BINARY))

    def test_attack_mapping_support_cannot_include_benign(self):
        e = acceptance_evidence(attack=True)
        metrics = tuple(replace(m, support_ids=partitions().support("acceptance", "old-b"))
                        if m.name == "missed_attack_rate_delta" else m for m in e.metrics)
        with self.assertRaises(GateError):
            decide(evidence=replace(e, metrics=metrics), policy=acceptance_policy(Semantics.ATTACK_BINARY))

    def test_missing_duplicate_nonfinite_and_foreign_metrics_fail(self):
        e = acceptance_evidence()
        bad_metrics = (e.metrics[:-1], e.metrics + (e.metrics[0],),
                       (replace(e.metrics[0], interval=Interval(math.nan, 0)),) + e.metrics[1:],
                       (replace(e.metrics[0], support_ids=("future-1", "foreign")),) + e.metrics[1:])
        for metrics in bad_metrics:
            with self.assertRaises(GateError):
                decide(evidence=replace(e, metrics=metrics))


if __name__ == "__main__":
    unittest.main()
