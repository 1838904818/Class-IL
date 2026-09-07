# Local verification and remaining work

7 September 2026. Research preparation only.

## Observed local verification

Command run from the workspace root with Python 3.11.9:

```text
python -B -m unittest discover -s work/parallel_research_20260907/repair_controls -p test_repair_reference.py -q
```

Observed: **63 tests passed**. The fixture suite exercises partition/capture
separation, temporal label availability, no-label blocking, all six controls,
simple arms without attribution, per-class support, nonfinite inputs, paired
score identity, within-method noise, harmful pure expansion, the weight-zero
ablation, deterministic random/scheduled ordering, cost limits and atomic
failure, audit-only non-intervention, strict active-intervention accounting,
explicit diagnosis overhead, one-spend acceptance, all-class retention/new-class
constraints, application/attack semantics, fit/model-state binding, no alert
suppression, and no semantic head creation.

Regression coverage also rejects the formerly possible path from a blocked
expansion trigger directly into repair reservation. Reservation now checks arm
eligibility itself. Acceptance checks it again; tests include a hand-rewritten
candidate digest, off-schedule periodic execution, changed candidate content,
changed arm, stale increment, changed trigger thresholds and loosened acceptance
tolerances. Both policies are content-bound before fitting.

These are tests of gates on fabricated metadata and intervals. They are not
real-data statistical tests, a numerical-fidelity experiment, a fitted repair,
or evidence of a mechanism's effectiveness. A toy ACCEPT outcome explicitly
returns repair_applied = false and real_data_utility_established = false.

## Required before a real experiment

1. Close source-bound deployed-score fidelity and method-specific attribution
   faithfulness, including the outstanding historical reconstruction gate.
2. Freeze raw-file/class/row/capture provenance, authorized label semantics and
   independent future evaluation; verify that declared groups cover actual
   leakage dependencies and that all required class supports are sufficient.
3. Select and justify every scientific threshold, budget and uncertainty method
   on development-only evidence, then hash a complete prospective registration.
   The toy test settings must not be copied as validated study settings.
4. Implement and validate the actual bounded R1 fitter and state rollback. Bind
   the full candidate pool, top-target choice, objective, parameter bounds,
   optimizer, row exposure, code, environment and checkpoint/fit receipts.
5. Implement paired simultaneous uncertainty on the proper independent sampling
   unit; validate interval/support correctness rather than supplying unchecked
   summaries. Seal subsequent labels from this entire decision path.
6. Add durable one-attempt/cross-increment label-use accounting and a complete
   immutable candidate-pool ledger. The in-memory object and receipt hashes are
   reference integrity checks, not authentic storage or authorization controls.
7. Measure actual diagnosis, fit, prediction, load and memory costs. Equal caps
   or equal reference counters do not prove total compute matching. Preserve
   abstentions and unused audit-only budget instead of padding or dropping them.
8. Complete novelty/control review and freeze independent seeds/capture design,
   contrasts, multiplicity/stopping rules and a plan to report disconfirmation.
9. Prepare any eventual executable compute stage separately with exact bindings,
   measured resources, current institutional evidence, required independent
   approval and current one-command user confirmation. None is supplied here.

No Git operation, remote action, training run, job submission, experiment
promotion or protected-result modification is part of this local preparation.
