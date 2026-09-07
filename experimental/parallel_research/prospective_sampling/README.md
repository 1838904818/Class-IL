# Prospective train-sampling reference

Status: **local synthetic reference only**. No real dataset was read or derived,
no model was trained, and no HPC operation was performed for this package.
This is a follow-up design informed by inspected historical evidence, not a
preregistration of the historical study or a claim of an optimal sampler.

The minimum candidate freezes the normal-class cap from **Task-0-available
attack fitting counts**, after a deterministic train-only calibration split.
It never enlarges or recomputes that cap after new classes arrive. A separately
predeclared fixed-cap arm is also supported. A missing Task-0 attack fit pool
stops the Task-0 arm; it does not trigger an outcome-dependent fallback.

## What the historical audit establishes

Two distinct data-construction dependencies were verified in the exact D2 v2
builder matching the saved audit:

1. The normal cap takes the largest attack fit count over every class in the
   offline source manifest, including later increments: 124,780 rows, from
   DoS Hulk (Task 1), while Task 0 contains Benign and DoS GoldenEye.
2. Each class's permutation seed includes the SHA-256 of the **entire source
   manifest**. Future-class or official-test metadata changes can therefore
   change the early sampling seed without changing early training rows. This
   is manifest-wide metadata coupling; it is not evidence that test labels
   were used to fit a model or optimize a cap.

These are separate from Task-0-only numerical scaling and vocabulary fitting.
Correct Task-0 transforms do not remove cap lookahead or manifest-seed coupling.
The new code removes both dependencies, so a direct legacy/new comparison
cannot attribute every change to the cap. See the matched bridge controls in
[PROTOCOL.md](PROTOCOL.md) and exact source lines/hashes in
[SOURCE_AUDIT.md](SOURCE_AUDIT.md).

Historical-count arithmetic makes the tradeoff concrete: Task-0 GoldenEye
has 6,176 source training rows, 617 calibration rows and 5,559 fit rows.
A cap frozen at 5,559 would remove 119,221 normal fit rows compared with
the offline cap of 124,780. Estimated total fitting support becomes 149,476
instead of 268,697, assuming the same support counts. This is arithmetic,
not data derivation. Less normal training may hurt accuracy or increase FPR;
check support adequacy and measured cost before a large comparison. No
lookahead is an information constraint, not an accuracy-improvement promise.

## Files and local check

- `prospective_sampling.py`: pure, immutable train-row-ID selection; no file,
  feature matrix, test-set, future-count, network, or training interface.
- `test_prospective_sampling.py`: synthetic prefix invariance, determinism,
  disjointness, rare/empty-class behavior, and rejection tests.
- `PROTOCOL.md`: information boundary, comparison accounting, and stage gates.
- `SOURCE_AUDIT.md`: verified historical-source bindings; no real-array claim.
- `RUN_REPORT.md`: local test evidence and remaining boundaries.

From this directory, run:

```console
python -B -m unittest discover -s . -p "test_*.py" -v
```

The reference refuses more than 10,000 row IDs across all increments. This is
an accidental-use guard, not a production memory/performance guarantee.

## Function contract

Create `initial_state(Policy(...))`, then call
`append_increment(state, increment=t, batches=(...))` once per increment.
Each `TrainBatch` describes one newly released class and contains only
`RowRef` entries from source training data available by that increment. Supply
no all-class registry, test labels/hashes, or future counts. Full-world inputs
are rejected, not filtered inside the selector.

Every `ClassSelection` returns sorted fitting, calibration and omitted row IDs,
and records the current cohort's train-only digest as provenance. Digests do
not seed sampling. The fixed sampling domain, integer seed, class ID, stable
row ID and split stage determine hash rankings. Input row/class order has no
effect. IDs must not themselves be constructed from future/test metadata.

Only new-class cohort arrival is implemented. Updating an old cohort or
adding later rows to an old class is rejected. Historical selections are
immutable tuples, not rewritten files. The reference does not validate actual
features, semantic duplicate leakage, true release times, digest provenance,
or trainer compatibility; these remain production-ingestion gates.
The production builder, loader adapter and real-array validation have not
been implemented in this package.
