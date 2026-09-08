# Checkpoint metrics contract

This module constructs aggregate payloads only. It does not initialize W&B,
read credentials, send data, or certify a remote dashboard. It is not yet wired
into the training runner. Do not present passing unit tests as online delivery.

Call `checkpoint_payload` with the current evaluation, the loss arm, the digest
of its committed checkpoint, and the complete contiguous prior evaluation
history. Before calling, the integration must verify that both current and
historical evaluations come from the same input/protocol/arm and hash-verified
committed files. Dictionary validation cannot establish file provenance.

Outputs include overall accuracy, Macro-F1, balanced accuracy, signed mean
recall forgetting, per-class support/precision/recall/F1, and confusion counts.
All performance measures are fractions, not percentages. Forgetting is signed:
historical best recall minus current recall, so improvements may be negative.
It is not clipped and must not be mixed with a clipped forgetting statistic.

For this official-test protocol every seen class must have positive support.
Zero support is rejected, not assigned a zero performance score. Task 0 has no
historical forgetting: its list must be empty and mean must be null. Subsequent
per-class and mean forgetting are checked against the historical best recall.
Numerical reconciliation uses zero relative tolerance and absolute tolerance
1e-12. Source metrics are reconciled against the confusion matrix.

`exposure_plan.plan` calculates the existing L1 schedule without training. It
does not implement equal-exposure sampling. Its synthetic test compares counts
with the actual batch generator, while its fixed real-count regression checks
the disclosed P/O schedule. A planned count is never proof of work executed.

Local regression on 9 September 2026: 62 tests passed, including the existing
runner/pretraining suite and the new payload/exposure tests. Independent review
and an explicit destination-bound integration remain required before live use.
