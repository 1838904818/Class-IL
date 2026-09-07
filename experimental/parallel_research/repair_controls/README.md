# Repair-control preparation (research only)

This directory contains an English prospective protocol, a pure-standard-library
reference and synthetic invariant tests. It does **not** contain a trained
repair model, optimizer, dataset loader, explainer, confidence-interval estimator,
actual compute meter, deployment hook, cluster script or new experiment result.

- [Protocol](PROTOCOL.md): exact proposed controls, partitions, bounded-repair
  interface, evidence conditions, cost distinctions and falsification.
- [Reference](repair_reference.py): gates and deterministic target ordering.
- [Tests](test_repair_reference.py): fabricated fixtures, not measured utility.
- [Verification and remaining work](VERIFICATION.md): local result and limits.

## Local check

Run from this directory, with Python 3.10 or newer:

```text
python -B -m unittest discover -s . -p test_repair_reference.py -v
```

No training, network call, GPU process or external package is needed. All test
scores, thresholds, sample IDs and digests are deliberately fabricated. Passing
tests does not validate their numerical choices or establish a research result.

## Available interfaces

| Interface | What it does | What it does not prove |
|---|---|---|
| `Partitions.validate()` | Enforces supplied sample/group separation, label availability and sealed later labels | Correct raw grouping, acquisition provenance or absence of hidden duplicates |
| `Candidate.validate(..., arm)` | Common score/partition checks and only the chosen arm's required signals | Authentic hashes, true numerical fidelity or valid uncertainty estimates |
| `rank_candidates(...)` | Freezes six control behaviors and orders candidate IDs; logs blocked evidence | Fit, select a recipe, accept or deploy a model |
| `BudgetLedger.charge_trigger(...)` | Charges acquired trigger labels and declared diagnosis units | Actual work performed or all undeclared costs |
| `BudgetLedger.reserve_repair(...)` | Rechecks arm eligibility/support; charges one fixed package; binds candidate, arm, both policies and partition; spends acceptance | Authentic runtime metering or cryptographic authorization |
| `compare_active_budgets(...)` | Checks active intervention-ledger matching; exposes diagnosis differences; separates audit-only | Total measured compute equivalence |
| `acceptance_gate(...)` | Rechecks arm eligibility/policy bindings and external fit/state receipts, paired constraints, support and semantics | Model improvement, real provenance, production safety or applying repair |

Provide all `TriggerPolicy`, `AcceptancePolicy` and `BudgetPlan` fields from a
versioned prospective registration. There are no production scientific defaults.
The fixed no-head/no-suppression behavior is a safety boundary, not a fitted
research threshold. Triggering and acceptance have different labeled partitions.

The future driver must execute in this order:

```text
validate partitions and external evidence
  -> charge actual arm-specific diagnosis and shared acquired trigger labels
  -> rank candidates and lock one candidate's full content
  -> reserve the fixed package (failed/rejected attempts stay charged)
  -> EXTERNAL R1 fitting on fit-only labels; hash fit record and model states
  -> acceptance_gate on acceptance-only paired predictions and intervals
  -> retain baseline or record accepted research variant, without alert suppression
  -> lock decisions and unseal subsequent evaluation (never feed back into selection)
```

The uppercase external step does not exist in this package. `FitEvidence` and
`AcceptanceEvidence` are receipt schemas, not a trainer or estimator. Simple
controls can omit all attribution/noise/decomposition fields. Random and
periodic can also omit error and disagreement evidence. `audit-only` never
reserves repair; if it computes extra flags, account for that diagnosis cost.

The reservation and acceptance call signatures are:

```text
ledger.reserve_repair(candidate, operation_id, trigger_policy, acceptance_policy)
acceptance_gate(candidate, reservation, fit_evidence, acceptance_evidence,
                partitions, acceptance_policy, trigger_policy, expected_arm)
```

Both operations recompute that arm's eligibility, including periodic due dates.
The reservation hashes both policies before fitting; changing either policy,
the candidate content, partition or expected arm invalidates acceptance.

## Important boundary

These are in-memory research references, not security or institutional approval
controls. Digest-format checks do not read or authenticate source files.
Content fingerprints detect changed supplied records, not dishonesty. Dataclass
receipts can be constructed directly and a new ledger resets memory; a real
driver needs authenticated file bindings, a persisted one-attempt ledger,
measured runtime accounting and cross-increment label-use checks. External
code must ensure only the locked top-ranked eligible candidate from the complete
registered candidate pool reaches fitting. Internal single-candidate eligibility
checks prevent bypassing a blocked trigger but do not reconstruct a missing
full candidate pool or authenticate a claimed random selection.
No reference output replaces independent review or user confirmation.

There is no automatic success fixture or artificial improvement plot. Tests
cover both software acceptance and rejection using invented values; neither is
a real repair outcome. Application-label data such as Malaya cannot acquire
attack/FPR meaning through a flag or renamed output column.
