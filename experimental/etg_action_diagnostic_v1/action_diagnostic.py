"""Metadata-only, conditional action-identifiability diagnostic.

No file, network, label, model, attribution or evaluation access. This utility
does not verify the origin of caller-supplied hashes and cannot establish an
experimental result. See README.md for the exact semantic-hash contract.
"""
from itertools import combinations

ARMS = ("audit-only", "error-only", "raw-drift", "noise-aware", "random-harmful")
BINDINGS = ("protocol", "native_baseline", "fit_input", "acceptance_input",
            "settings", "correction_code", "acceptance_rule", "numeric_environment")
EVENTS = ("CANDIDATES_LOCKED", "FITS_LOCKED", "DECISIONS_LOCKED")
ARM_KEYS = {"arm", "choice_status", "target", "fit_status", "decision_status",
            "accepted_empirically", "fit_semantics_sha256",
            "decision_semantics_sha256", "bindings"}
CHOICES = {"attempt", "audit-only", "abstain", "unavailable-attribution"}
FIT_STATES = {"complete", "not-attempted", "resource-failure", "failed", "incomplete"}
DECISION_STATES = {"complete", "failed", "incomplete"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value):
    return (type(value) is str and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _validate_arm(record):
    _require(type(record) is dict and set(record) == ARM_KEYS, "arm metadata keys")
    _require(record["arm"] in ARMS, "unknown arm")
    _require(record["choice_status"] in CHOICES, "unknown choice status")
    _require(record["fit_status"] in FIT_STATES, "unknown fit status")
    _require(record["decision_status"] in DECISION_STATES, "unknown decision status")
    target = record["target"]
    _require(target is None or (type(target) is int and target in (0, 1)),
             "only the registered two-old-class pilot is supported")
    attempt = record["choice_status"] == "attempt"
    _require((target is not None) == attempt, "target/choice mismatch")
    _require((record["arm"] == "audit-only") == (record["choice_status"] == "audit-only"),
             "audit-only arm mismatch")
    accepted = record["accepted_empirically"]
    _require(accepted is None or type(accepted) is bool, "acceptance must be bool/null")
    bindings = record["bindings"]
    _require(type(bindings) is dict and set(bindings) == set(BINDINGS)
             and all(_sha(v) for v in bindings.values()), "binding hashes incomplete")
    for key in ("fit_semantics_sha256", "decision_semantics_sha256"):
        _require(record[key] is None or _sha(record[key]), "invalid semantic hash")
    if record["fit_status"] == "complete":
        _require(attempt and _sha(record["fit_semantics_sha256"]), "completed fit needs semantic hash")
    if not attempt:
        _require(record["fit_status"] == "not-attempted" and record["fit_semantics_sha256"] is None,
                 "non-attempt cannot contain a fit")
    if record["decision_status"] == "complete":
        _require(type(accepted) is bool and _sha(record["decision_semantics_sha256"]),
                 "completed decision needs bool and semantic hash")
        _require(not accepted or (attempt and record["fit_status"] == "complete"),
                 "accepted action requires a completed fit")
        _require(not attempt or record["fit_status"] == "complete",
                 "failed/incomplete attempt cannot become a completed decision")
    else:
        _require(accepted is None and record["decision_semantics_sha256"] is None,
                 "incomplete decision cannot look decided")


def _pair(left, right):
    result = {"left": left["arm"], "right": right["arm"],
              "metadata_only": True, "scientific_efficacy_established": False,
              "external_hash_verification_required": True,
              "different_heldout_predictions_established": False}

    def emit(status, reason):
        return dict(result, status=status, reason=reason)

    # Failures are not affirmative abstentions or proof of no scientific effect.
    if any(r["choice_status"] == "unavailable-attribution" for r in (left, right)):
        return emit("UNAVAILABLE_ATTRIBUTION", "Attribution availability is not a selector decision.")
    if any(r["fit_status"] == "resource-failure" for r in (left, right)):
        return emit("UNAVAILABLE_RESOURCE", "A resource failure is not a completed comparison.")
    if any(r["fit_status"] in {"failed", "incomplete"}
           or r["decision_status"] != "complete" for r in (left, right)):
        return emit("UNAVAILABLE_INCOMPLETE", "Fit/decision stage did not finish for both arms.")
    if left["bindings"] != right["bindings"]:
        return emit("INCOMPARABLE_BINDINGS", "The shared correction/input/numerical contract differs.")
    a, b = left["accepted_empirically"], right["accepted_empirically"]
    if not a and not b:
        return emit("SAME_ACTION_NO_SELECTION_GAIN", "Both effective policies retain the identical baseline.")
    both_attempt = all(r["choice_status"] == "attempt" for r in (left, right))
    if both_attempt and left["target"] == right["target"]:
        if left["fit_semantics_sha256"] != right["fit_semantics_sha256"]:
            return emit("UNRESOLVED_SAME_TARGET", "Same target alone does not prove identical fitted correction.")
        if a != b or left["decision_semantics_sha256"] != right["decision_semantics_sha256"]:
            return emit("INCONSISTENT_DECISION_SEMANTICS", "Identical proposed action/context has conflicting decision semantics.")
        return emit("SAME_ACTION_NO_SELECTION_GAIN", "Same target, fitted correction, decision and context imply identical effective policy.")
    if both_attempt and left["fit_semantics_sha256"] == right["fit_semantics_sha256"]:
        return emit("INCONSISTENT_FIT_SEMANTICS", "Target-bound semantic hashes cannot coincide for different targets.")
    if a != b:
        return emit("DIFFERENT_ACTION_REQUIRES_HELDOUT", "One policy applies its correction; the other retains baseline.")
    return emit("DIFFERENT_ACTION_REQUIRES_HELDOUT", "Both accepted corrections target different class-score columns.")


def audit_locked_choices(snapshot):
    """Return ten conditional pair diagnoses and the registered primary contrast.

    Caller independently verifies the three event hashes before using this
    result. No hash here proves its own provenance. Extra fields (including
    labels, metrics, attribution arrays or raw fit parameters) are rejected.
    """
    keys = {"schema", "source_kind", "locked_stage", "events_sha256", "arms"}
    _require(type(snapshot) is dict and set(snapshot) == keys, "snapshot metadata keys")
    _require(snapshot["schema"] == "etg-action-lock-metadata-v1", "snapshot schema")
    _require(snapshot["source_kind"] in {"synthetic", "real-inputs"}, "explicit source kind required")
    _require(snapshot["locked_stage"] in {"DECISIONS_LOCKED", "EVALUATION_SPENT", "COMPLETE"},
             "decisions must already be durably locked")
    events = snapshot["events_sha256"]
    _require(type(events) is dict and set(events) == set(EVENTS)
             and all(_sha(v) for v in events.values()), "three locked event hashes required")
    arms = snapshot["arms"]
    _require(type(arms) is list and len(arms) == len(ARMS), "all five arms required")
    for arm in arms:
        _validate_arm(arm)
    _require({a["arm"] for a in arms} == set(ARMS), "duplicate or missing arm")
    by_name = {a["arm"]: a for a in arms}
    pairs = [_pair(by_name[a], by_name[b]) for a, b in combinations(ARMS, 2)]
    return {"schema": "etg-action-identifiability-diagnostic-v1",
            "source_kind": snapshot["source_kind"], "locked_stage": snapshot["locked_stage"],
            "source_event_hashes": dict(events), "metadata_only": True,
            "scientific_efficacy_established": False,
            "external_hash_verification_required": True,
            "primary": _pair(by_name["noise-aware"], by_name["error-only"]),
            "pairs": pairs,
            "cost_comparison_established": False,
            "interpretation": "Conditional action identity only; no labels were accessed or efficacy estimated."}
