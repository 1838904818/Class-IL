"""Research-only control/partition/budget gates; no model fitting or deployment.

All scientific thresholds are caller-supplied prospective choices. Evidence
summaries and digest provenance must be established outside this reference.
"""
from dataclasses import dataclass, fields
from enum import Enum
import hashlib
import math
import random
import re


class GateError(ValueError):
    """Malformed or insufficient evidence; the caller must abstain."""


class Arm(str, Enum):
    RANDOM = "random"
    PERIODIC = "periodic"
    ERROR_ONLY = "error-only"
    DISAGREEMENT_ONLY = "disagreement-only"
    EXPANSION_AWARE = "expansion-aware-candidate"
    AUDIT_ONLY = "audit-only"


class Semantics(str, Enum):
    APPLICATION = "application"
    ATTACK_BINARY = "attack-binary"


def require(condition, message):
    if not condition:
        raise GateError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def natural(value, positive=False):
    return isinstance(value, int) and not isinstance(value, bool) and value >= int(positive)


def unique(values):
    return len(values) == len(set(values))


@dataclass(frozen=True)
class Sample:
    row_id: str
    leakage_group: str
    observed_increment: int
    label_available_increment: int | None
    class_id: str | None


@dataclass(frozen=True)
class Partitions:
    increment: int
    trigger: tuple[Sample, ...]
    fit: tuple[Sample, ...]
    acceptance: tuple[Sample, ...]
    subsequent: tuple[Sample, ...]

    def rows(self, role):
        require(role in ("trigger", "fit", "acceptance", "subsequent"), "unknown partition")
        return getattr(self, role)

    def validate(self):
        require(natural(self.increment), "invalid increment")
        seen_rows, seen_groups = set(), set()
        for role in ("trigger", "fit", "acceptance", "subsequent"):
            rows = self.rows(role)
            require(bool(rows), "every partition must be nonempty")
            ids = [r.row_id for r in rows]
            groups = {r.leakage_group for r in rows}
            require(all(isinstance(x, str) and x for x in ids), "missing row identity")
            require(all(isinstance(x, str) and x for x in groups), "missing leakage-group identity")
            require(unique(ids) and not seen_rows.intersection(ids), "row overlap or duplicates")
            require(not seen_groups.intersection(groups), "cross-partition leakage group")
            seen_rows.update(ids)
            seen_groups.update(groups)
            for row in rows:
                require(natural(row.observed_increment), "invalid observation increment")
                if role == "subsequent":
                    require(row.observed_increment > self.increment, "evaluation must be subsequent")
                    require(row.class_id is None and row.label_available_increment is None,
                            "evaluation labels must be sealed, not supplied to this gate")
                else:
                    require(row.observed_increment <= self.increment, "future observation leakage")
                    require(natural(row.label_available_increment), "available labels required")
                    require(row.observed_increment <= row.label_available_increment <= self.increment,
                            "future or impossible label availability")
                    require(isinstance(row.class_id, str) and bool(row.class_id), "semantic label required")
        return self

    def support(self, role, class_id):
        return tuple(r.row_id for r in self.rows(role) if r.class_id == class_id)

    def fingerprint(self):
        self.validate()
        # This identifies supplied metadata, not raw-file integrity or provenance.
        return hashlib.sha256(repr(self).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Binding:
    code_sha256: str
    data_sha256: str
    protocol_sha256: str
    score_validation_sha256: str
    attribution_validation_sha256: str | None
    uncertainty_validation_sha256: str | None

    def validate(self, expansion=False):
        selected = fields(self) if expansion else fields(self)[:4]
        for field in selected:
            require(isinstance(getattr(self, field.name), str) and
                    re.fullmatch(r"[0-9a-f]{64}", getattr(self, field.name)) is not None,
                    "missing or malformed evidence digest")


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float

    def validate(self, probability=False):
        require(finite(self.lower) and finite(self.upper) and self.lower <= self.upper,
                "invalid uncertainty interval")
        if probability:
            require(-1 <= self.lower <= self.upper <= 1, "rate-change interval out of range")


@dataclass(frozen=True)
class TriggerPolicy:
    min_trigger_support: int
    min_error_increase: float
    min_disagreement: float
    min_drift_excess: float
    state_priority_weight: float
    identity_atol: float
    periodic_every: int
    periodic_phase: int
    random_seed: int
    explanation_method: str
    old_classes: tuple[str, ...]

    def fingerprint(self):
        self.validate()
        return hashlib.sha256(repr(self).encode("utf-8")).hexdigest()

    def validate(self):
        require(natural(self.min_trigger_support, True), "support threshold must be explicit and positive")
        for name in ("min_error_increase", "min_disagreement", "min_drift_excess", "state_priority_weight", "identity_atol"):
            require(finite(getattr(self, name)) and getattr(self, name) >= 0, "invalid prospective threshold")
        require(self.min_error_increase <= 1 and self.min_disagreement <= 1, "rate threshold out of range")
        require(natural(self.periodic_every, True) and natural(self.periodic_phase) and
                self.periodic_phase < self.periodic_every, "invalid frozen periodic schedule")
        require(natural(self.random_seed) and bool(self.explanation_method), "seed and method are required")
        require(bool(self.old_classes) and unique(self.old_classes) and
                all(isinstance(x, str) and x for x in self.old_classes), "old target class set must be frozen")


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    class_id: str
    increment: int
    trigger_row_ids: tuple[str, ...]
    partition_sha256: str
    binding: Binding
    error_increase: Interval | None
    disagreement: Interval | None
    explanation_drift: Interval | None
    within_method_noise: Interval | None
    method: str | None
    score_validated: bool
    attribution_validated: bool
    within_method_estimated: bool
    state_effect: float | None
    normalization_effect: float | None
    new_rival_effect: float | None
    total_margin_change: float | None
    requests_new_head: bool = False

    def fingerprint(self):
        return hashlib.sha256(repr(self).encode("utf-8")).hexdigest()

    def validate(self, partitions, policy, arm):
        partitions.validate()
        policy.validate()
        self.binding.validate(expansion=arm == Arm.EXPANSION_AWARE)
        require(bool(self.candidate_id) and bool(self.class_id), "candidate identity required")
        require(self.class_id in policy.old_classes, "repair target must be a registered old class")
        require(self.increment == partitions.increment, "stale or future candidate")
        require(self.partition_sha256 == partitions.fingerprint(), "partition binding mismatch")
        require(unique(self.trigger_row_ids) and set(self.trigger_row_ids) ==
                set(partitions.support("trigger", self.class_id)), "trigger support must match frozen class rows")
        require(len(self.trigger_row_ids) >= policy.min_trigger_support, "insufficient trigger support")
        require(not self.requests_new_head, "this study never creates supervised heads")
        require(self.score_validated is True, "numerical-fidelity gate missing")
        if arm in (Arm.ERROR_ONLY, Arm.EXPANSION_AWARE):
            require(isinstance(self.error_increase, Interval), "error-only/candidate trigger needs error evidence")
            self.error_increase.validate(probability=True)
        if arm == Arm.DISAGREEMENT_ONLY:
            require(isinstance(self.disagreement, Interval), "disagreement trigger needs disagreement evidence")
            self.disagreement.validate(probability=True)
            require(self.disagreement.lower >= 0, "disagreement is a nonnegative rate")
        if arm != Arm.EXPANSION_AWARE:
            return
        require(self.method == policy.explanation_method and self.attribution_validated is True and
                self.within_method_estimated is True, "method-matched attribution/uncertainty gate missing")
        require(isinstance(self.explanation_drift, Interval) and isinstance(self.within_method_noise, Interval),
                "expansion trigger needs explanation and within-method noise intervals")
        self.explanation_drift.validate()
        self.within_method_noise.validate()
        require(self.explanation_drift.lower >= 0 and self.within_method_noise.lower >= 0,
                "drift and noise must use the same nonnegative distance")
        for value in (self.state_effect, self.normalization_effect, self.new_rival_effect, self.total_margin_change):
            require(finite(value), "nonfinite decomposition")
        require(abs(self.state_effect + self.normalization_effect + self.new_rival_effect -
                    self.total_margin_change) <= policy.identity_atol, "non-telescoping registered path")
        require(self.new_rival_effect <= policy.identity_atol, "new-rival monotonicity violation")


@dataclass(frozen=True)
class Ranking:
    ordered_ids: tuple[str, ...]
    blocked: tuple[tuple[str, str], ...]
    audit_ids: tuple[str, ...]


def rank_candidates(arm, candidates, partitions, policy):
    """Order targets only. An ID is NOT a repair or acceptance authorization.

    Error/disagreement arms use only their named signal for ordering. The
    expansion arm requires harmful error change AND drift beyond within-method
    noise; harmful pure expansion remains eligible. Random is seeded and input
    order invariant. Periodic rotates lexical class order on frozen due dates.
    """
    require(isinstance(arm, Arm), "unknown arm")
    candidates = tuple(candidates)
    partitions.validate()
    policy.validate()
    require(unique([c.candidate_id for c in candidates]), "duplicate candidates")
    require(unique([c.class_id for c in candidates]), "one candidate per class per increment")
    eligible, blocked = [], []
    for candidate in candidates:
        try:
            candidate.validate(partitions, policy, arm)
            eligible.append(candidate)
        except GateError as exc:
            blocked.append((candidate.candidate_id, str(exc)))
    eligible.sort(key=lambda c: c.candidate_id)
    if arm == Arm.AUDIT_ONLY:
        return Ranking((), tuple(blocked), tuple(c.candidate_id for c in eligible))
    if arm == Arm.RANDOM:
        random.Random(f"{policy.random_seed}:{partitions.increment}").shuffle(eligible)
    elif arm == Arm.PERIODIC:
        if partitions.increment % policy.periodic_every != policy.periodic_phase:
            eligible = []
        elif eligible:
            offset = (partitions.increment // policy.periodic_every) % len(eligible)
            eligible = eligible[offset:] + eligible[:offset]
    elif arm == Arm.ERROR_ONLY:
        eligible = [c for c in eligible if c.error_increase.lower > policy.min_error_increase]
        eligible.sort(key=lambda c: (-c.error_increase.lower, c.candidate_id))
    elif arm == Arm.DISAGREEMENT_ONLY:
        eligible = [c for c in eligible if c.disagreement.lower > policy.min_disagreement]
        eligible.sort(key=lambda c: (-c.disagreement.lower, c.candidate_id))
    else:
        scored = []
        for c in eligible:
            excess = c.explanation_drift.lower - c.within_method_noise.upper
            if c.error_increase.lower <= policy.min_error_increase or excess <= policy.min_drift_excess:
                continue
            mass = abs(c.state_effect) + abs(c.normalization_effect) + abs(c.new_rival_effect)
            state_share = max(0.0, -c.state_effect) / mass if mass else 0.0
            priority = c.error_increase.lower * (1.0 + policy.state_priority_weight * state_share)
            require(finite(priority), "nonfinite priority")
            scored.append((priority, excess, c.candidate_id, c))
        eligible = [x[3] for x in sorted(scored, key=lambda x: (-x[0], -x[1], x[2]))]
    return Ranking(tuple(c.candidate_id for c in eligible), tuple(blocked), ())


@dataclass(frozen=True)
class RepairPackage:
    operation_id: str
    fit_steps: int
    acceptance_score_rows: int

    def validate(self):
        require(bool(self.operation_id) and natural(self.fit_steps, True) and
                natural(self.acceptance_score_rows, True), "fixed active repair package required")


@dataclass(frozen=True)
class BudgetPlan:
    unique_label_cap: int
    diagnostic_unit_cap: int
    max_repairs: int
    fit_step_cap: int
    acceptance_row_cap: int
    package: RepairPackage
    compute_unit_definition: str

    def validate(self):
        self.package.validate()
        for name in ("unique_label_cap", "diagnostic_unit_cap", "max_repairs", "fit_step_cap", "acceptance_row_cap"):
            require(natural(getattr(self, name)), "invalid budget cap")
        require(bool(self.compute_unit_definition), "compute accounting unit must be frozen")


@dataclass(frozen=True)
class Reservation:
    candidate_id: str
    candidate_sha256: str
    arm: Arm
    trigger_policy_sha256: str
    acceptance_policy_sha256: str
    partition_sha256: str
    operation_id: str
    ordinal: int


class BudgetLedger:
    """Atomic in-memory reservation accounting, NOT a measured resource monitor.

    Rejected/failed attempts remain charged. A candidate cannot be retried on
    the same acceptance partition; no refund API is provided. Every active
    attempt consumes the same registered fit and acceptance label pools.
    """
    def __init__(self, arm, plan, partitions):
        require(isinstance(arm, Arm), "unknown arm")
        plan.validate()
        partitions.validate()
        self.arm, self.plan, self.partitions = arm, plan, partitions
        self.labels = frozenset()
        self.diagnostic_units = self.repairs = self.fit_steps = self.acceptance_rows = 0
        self._acceptance_consumed = False

    def charge_trigger(self, label_ids, diagnostic_units):
        require(natural(diagnostic_units), "invalid diagnostic charge")
        label_ids = tuple(label_ids)
        require(set(label_ids).issubset({r.row_id for r in self.partitions.trigger}), "non-trigger labels charged as trigger")
        labels = self.labels.union(label_ids)
        require(len(labels) <= self.plan.unique_label_cap, "label cap exceeded")
        require(self.diagnostic_units + diagnostic_units <= self.plan.diagnostic_unit_cap, "diagnostic cap exceeded")
        self.labels = labels
        self.diagnostic_units += diagnostic_units

    def reserve_repair(self, candidate, operation_id, trigger_policy, acceptance_policy):
        require(self.arm != Arm.AUDIT_ONLY, "audit-only never repairs")
        require(isinstance(candidate, Candidate) and bool(candidate.candidate_id), "candidate must be locked before acceptance")
        require(candidate.partition_sha256 == self.partitions.fingerprint() and
                candidate.increment == self.partitions.increment, "reservation candidate binding mismatch")
        eligibility = rank_candidates(self.arm, (candidate,), self.partitions, trigger_policy)
        require(eligibility.ordered_ids == (candidate.candidate_id,),
                "candidate fails frozen arm eligibility or periodic schedule")
        acceptance_policy.validate_partitions(self.partitions, candidate.class_id)
        require(set(trigger_policy.old_classes) == set(acceptance_policy.old_classes), "old class policies disagree")
        require(operation_id == self.plan.package.operation_id, "repair operation mismatch")
        require(not self._acceptance_consumed, "acceptance partition already spent; no repeated testing")
        require({r.row_id for r in self.partitions.trigger}.issubset(self.labels),
                "all shared trigger labels must be charged before candidate ranking/repair")
        new_labels = {r.row_id for r in self.partitions.fit + self.partitions.acceptance}
        labels = self.labels.union(new_labels)
        package = self.plan.package
        require(len(labels) <= self.plan.unique_label_cap, "label cap exceeded")
        require(self.repairs + 1 <= self.plan.max_repairs, "repair-count cap exceeded")
        require(self.fit_steps + package.fit_steps <= self.plan.fit_step_cap, "fit-step cap exceeded")
        require(self.acceptance_rows + package.acceptance_score_rows <= self.plan.acceptance_row_cap,
                "acceptance-compute cap exceeded")
        require(package.acceptance_score_rows >= 2 * len(self.partitions.acceptance),
                "package must account for baseline and repaired acceptance scoring")
        self.labels = labels
        self.repairs += 1
        self.fit_steps += package.fit_steps
        self.acceptance_rows += package.acceptance_score_rows
        self._acceptance_consumed = True
        return Reservation(candidate.candidate_id, candidate.fingerprint(), self.arm,
                           trigger_policy.fingerprint(), acceptance_policy.fingerprint(),
                           self.partitions.fingerprint(), operation_id, self.repairs)

    def usage(self):
        return {"unique_labels": len(self.labels), "diagnostic_units": self.diagnostic_units,
                "repairs": self.repairs, "fit_steps": self.fit_steps, "acceptance_score_rows": self.acceptance_rows}


def compare_active_budgets(ledgers):
    """Fail unless all five intervention arms share plan and intervention usage.

    Equal counters only establish reference-ledger equality, not actual compute
    equality. Diagnostic costs may differ and are never hidden or padded with
    compulsory SHAP. Audit-only is separate, with its unused repair budget.
    """
    ledgers = tuple(ledgers)
    require(unique([x.arm for x in ledgers]), "duplicate arm ledger")
    active = [x for x in ledgers if x.arm != Arm.AUDIT_ONLY]
    require({x.arm for x in active} == set(Arm) - {Arm.AUDIT_ONLY}, "five active arms required")
    first = active[0]
    require(first.repairs > 0, "zero repairs cannot establish an active-intervention comparison")
    for ledger in active[1:]:
        require(ledger.plan == first.plan and ledger.partitions == first.partitions, "unmatched frozen budget or partitions")
        require(all(ledger.usage()[k] == first.usage()[k] for k in
                    ("unique_labels", "repairs", "fit_steps", "acceptance_score_rows")),
                "realized intervention usage differs; do not claim intervention matching")
    return {"active_intervention_ledger_matched": True, "actual_compute_verified": False,
            "equal_diagnostic_ledger_units": len({x.diagnostic_units for x in active}) == 1,
            "diagnostic_units_by_arm": {x.arm.value: x.diagnostic_units for x in ledgers},
            "audit_only_active_compute_matched": False, "reference_usage": first.usage()}


@dataclass(frozen=True)
class AcceptancePolicy:
    semantics: Semantics
    old_classes: tuple[str, ...]
    new_classes: tuple[str, ...]
    min_fit_per_class: int
    min_acceptance_per_class: int
    min_metric_support: int
    min_target_improvement: float
    max_old_error_increase: float
    max_new_error_increase: float
    max_missed_attack_increase: float | None
    max_fpr_increase: float | None
    attack_classes: tuple[str, ...]
    benign_classes: tuple[str, ...]
    uncertainty_protocol_sha256: str

    def fingerprint(self):
        self.validate()
        return hashlib.sha256(repr(self).encode("utf-8")).hexdigest()

    def validate_partitions(self, partitions, target_class):
        self.validate()
        require(target_class in self.old_classes, "repair target must be a registered old class")
        classes = self.old_classes + self.new_classes
        for role, minimum in (("fit", self.min_fit_per_class), ("acceptance", self.min_acceptance_per_class)):
            require({r.class_id for r in partitions.rows(role)} == set(classes), "unregistered or missing class labels")
            for class_id in classes:
                require(len(partitions.support(role, class_id)) >= minimum, "insufficient per-class support")

    def validate(self):
        require(isinstance(self.semantics, Semantics), "label semantics required")
        require(bool(self.old_classes) and bool(self.new_classes) and
                unique(self.old_classes + self.new_classes), "disjoint old/new classes required")
        for value in (self.min_fit_per_class, self.min_acceptance_per_class, self.min_metric_support):
            require(natural(value, True), "positive support requirements must be registered")
        for value in (self.min_target_improvement, self.max_old_error_increase, self.max_new_error_increase):
            require(finite(value) and 0 <= value <= 1, "invalid acceptance tolerance")
        attack = (self.max_missed_attack_increase, self.max_fpr_increase)
        if self.semantics == Semantics.APPLICATION:
            require(all(x is None for x in attack), "application labels do not define attack recall/FPR")
            require(not self.attack_classes and not self.benign_classes, "application labels cannot be relabeled as attacks")
        else:
            require(all(finite(x) and 0 <= x <= 1 for x in attack), "attack constraints required")
            require(bool(self.attack_classes) and bool(self.benign_classes) and
                    unique(self.attack_classes + self.benign_classes) and
                    set(self.attack_classes + self.benign_classes) == set(self.old_classes + self.new_classes),
                    "explicit authorized attack/benign class mapping required")
        require(isinstance(self.uncertainty_protocol_sha256, str) and
                re.fullmatch(r"[0-9a-f]{64}", self.uncertainty_protocol_sha256) is not None,
                "simultaneous uncertainty procedure must be frozen")


@dataclass(frozen=True)
class MetricEvidence:
    name: str
    interval: Interval
    support_ids: tuple[str, ...]


@dataclass(frozen=True)
class AcceptanceEvidence:
    candidate_id: str
    partition_role: str
    partition_sha256: str
    baseline_state_sha256: str
    repaired_state_sha256: str
    uncertainty_protocol_sha256: str
    simultaneous_intervals: bool
    metrics: tuple[MetricEvidence, ...]
    preserves_raw_alerts: bool
    creates_new_head: bool


@dataclass(frozen=True)
class FitEvidence:
    candidate_id: str
    operation_id: str
    partition_sha256: str
    fit_row_ids: tuple[str, ...]
    baseline_state_sha256: str
    repaired_state_sha256: str
    fit_record_sha256: str


def acceptance_gate(candidate, reservation, fit, evidence, partitions, policy, trigger_policy, expected_arm):
    """Return research ACCEPT/REJECT, never apply a repair or suppress alerts.

    All old/new class constraints are required separately. Intervals are
    externally computed, paired baseline/repair estimates on acceptance only.
    This function cannot authenticate a claimed interval or uncertainty method.
    """
    partitions.validate()
    policy.validate()
    candidate.binding.validate()
    require(candidate.partition_sha256 == partitions.fingerprint() and candidate.score_validated is True and
            candidate.requests_new_head is False and candidate.increment == partitions.increment,
            "candidate safety or binding gate missing")
    require(reservation.candidate_id == candidate.candidate_id == evidence.candidate_id, "candidate changed after reservation")
    require(reservation.candidate_sha256 == candidate.fingerprint() and isinstance(reservation.arm, Arm) and
            reservation.arm != Arm.AUDIT_ONLY and reservation.arm == expected_arm, "immutable candidate or arm binding changed")
    require(reservation.trigger_policy_sha256 == trigger_policy.fingerprint(), "frozen trigger policy changed")
    require(reservation.acceptance_policy_sha256 == policy.fingerprint(), "frozen acceptance policy changed")
    require(set(trigger_policy.old_classes) == set(policy.old_classes), "old class policies disagree")
    require(rank_candidates(reservation.arm, (candidate,), partitions, trigger_policy).ordered_ids ==
            (candidate.candidate_id,), "reserved candidate fails recomputed arm eligibility")
    require(fit.candidate_id == candidate.candidate_id and fit.operation_id == reservation.operation_id and
            fit.partition_sha256 == partitions.fingerprint(), "fit receipt does not match locked candidate")
    require(unique(fit.fit_row_ids) and set(fit.fit_row_ids) == {r.row_id for r in partitions.fit},
            "fit receipt used a different or non-fit label pool")
    for digest in (fit.baseline_state_sha256, fit.repaired_state_sha256, fit.fit_record_sha256):
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                "bound external fit record and states required")
    require(evidence.baseline_state_sha256 == fit.baseline_state_sha256 and
            evidence.repaired_state_sha256 == fit.repaired_state_sha256,
            "acceptance scores are not bound to the fitted and baseline states")
    require(reservation.partition_sha256 == evidence.partition_sha256 == partitions.fingerprint(), "acceptance binding mismatch")
    require(evidence.partition_role == "acceptance", "future evaluation cannot accept or select a repair")
    require(evidence.uncertainty_protocol_sha256 == policy.uncertainty_protocol_sha256 and
            evidence.simultaneous_intervals is True, "uncorrected or mismatched repeated/multiple inference")
    require(evidence.preserves_raw_alerts is True and evidence.creates_new_head is False,
            "alert suppression or semantic head creation forbidden")
    classes = policy.old_classes + policy.new_classes
    policy.validate_partitions(partitions, candidate.class_id)
    expected = {f"class_error_delta:{c}" for c in classes} | {"target_error_improvement"}
    if policy.semantics == Semantics.ATTACK_BINARY:
        expected |= {"missed_attack_rate_delta", "false_positive_rate_delta"}
    names = [m.name for m in evidence.metrics]
    require(unique(names) and set(names) == expected, "missing, duplicate, or label-incompatible acceptance metrics")
    by_name = {m.name: m for m in evidence.metrics}
    delta = by_name[f"class_error_delta:{candidate.class_id}"].interval
    benefit = by_name["target_error_improvement"].interval
    require(benefit.lower == -delta.upper and benefit.upper == -delta.lower,
            "target benefit must be the sign reversal of the same paired class error interval")
    all_acceptance = {r.row_id for r in partitions.acceptance}
    rejected = []
    for metric in evidence.metrics:
        metric.interval.validate(probability=True)
        require(unique(metric.support_ids) and len(metric.support_ids) >= policy.min_metric_support and
                set(metric.support_ids).issubset(all_acceptance), "insufficient or foreign metric support")
        if metric.name == "target_error_improvement":
            require(set(metric.support_ids) == set(partitions.support("acceptance", candidate.class_id)), "target support mismatch")
            if metric.interval.lower < policy.min_target_improvement:
                rejected.append(metric.name)
        elif metric.name.startswith("class_error_delta:"):
            class_id = metric.name.split(":", 1)[1]
            require(set(metric.support_ids) == set(partitions.support("acceptance", class_id)), "class support mismatch")
            maximum = policy.max_old_error_increase if class_id in policy.old_classes else policy.max_new_error_increase
            if metric.interval.upper > maximum:
                rejected.append(metric.name)
        else:
            mapped = policy.attack_classes if metric.name == "missed_attack_rate_delta" else policy.benign_classes
            require(set(metric.support_ids) == {r.row_id for r in partitions.acceptance if r.class_id in mapped},
                    "attack/benign metric support does not match the registered label mapping")
            maximum = policy.max_missed_attack_increase if metric.name == "missed_attack_rate_delta" else policy.max_fpr_increase
            if metric.interval.upper > maximum:
                rejected.append(metric.name)
    return {"decision": "REJECT" if rejected else "ACCEPT_FOR_RESEARCH_ONLY", "failed_constraints": tuple(rejected),
            "repair_applied": False, "real_data_utility_established": False}
