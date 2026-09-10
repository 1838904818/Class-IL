# Implementation stage 2: native target and one-use execution

11 September 2026. **Software tests, not an intervention result.**

## What is now implemented

- `native_loader.py` checks the historical source tree and checkpoint bindings
  before importing/loading a model. The original loader reconstructs the
  checkpoint; the new code does not substitute a different FT adapter.
- `native_target.py` scores parent classes/shards in their original 512-row
  contexts before selecting output rows. For attribution it replaces one row
  at a time, leaving every companion row fixed. It validates native float32
  head/Router/joint values, class order and argmax.
- `capture_native` captures the binary logits from the same native forward call;
  temporary forward hooks are removed on success and failure.
- `permutation_repeats` uses the published SHAP package, version 0.51.0, not an
  invented attribution estimator. The Independent masker is subclassed only
  to use exact value equality for invariance. The default tolerance-based
  skipping could omit a distinct float32 value near a decision boundary.
- SHAP delta masking can promote float32 data to float64. The adapter permits
  only lossless round trips back to float32; it rejects genuinely changed
  numerical values instead of silently rounding them.
- `native_r1.py` retains the original float32 joint scores byte-for-byte at
  identity. A nonidentity correction changes one target column only. Original
  Router outputs and non-target columns are never reconstructed in float64.
- `one_use.py` computes the trigger from actual provided old/new score arrays,
  claims row identities in SQLite, reserves all fitting attempts, locks every
  fitted state before acceptance access, then locks all decisions before
  evaluation access. Failures and reused row identities remain spent across
  restart/study rename. No reset/retry command is provided.

The exposure guard is operational program order, not cryptographic blinding:
a class-stratified retrospective data layout can already reveal label identity.
It cannot undo previous calibration use or establish capture independence.

The explained output is the old class's margin over its strongest currently
available rival. The rival set grows between checkpoints. Measured change thus
includes new-class competition, not just change inside the old head; explanation
distance alone is not evidence of damaging drift or forgetting.

## Numerical definition of the correction

The inherited R1 mathematical action is unchanged: a*z+b for one old class.
Its implementation is separately native-anchored here. The optimizer uses a
smooth float64 surrogate while retaining frozen native non-target scores.
Evaluation quantizes the corrected probability to float32 and applies the
native float32 Router addition. Exact identity bypasses reconstruction.
Thus the optimization surrogate and final quantized evaluation are explicitly
different; acceptance always checks the latter. There is no claim of bit-exact
equivalence between float64 sigmoid and the archived float32 softmax.

All five selectors use the same fitter and numerical contract. Synthetic test
settings (five iterations, specific bounds/rate) are fixtures, not adopted
scientific hyperparameters. Final R1 settings remain to be frozen in a reviewed
candidate before real fitting.

## New failure handling

A missing/invalid explanation is not a valid zero-drift measurement.
When it makes the explanation arm unavailable, the primary comparison is
`null` with `UNAVAILABLE_ATTRIBUTION`; preserved baseline output is not
misrepresented as a tested explanation-guided policy. Legitimate abstention
is recorded separately.

The native call budget counts failed calls and all context rows. A fit that
fails remains a reserved/spent attempt. The engine preserves proposed versus
effective outcomes, per-class confusion/recall/precision/F1, accuracy,
Macro-F1, balanced accuracy, old-class error, attack recall, Benign FPR,
actual successful-fit time/steps and label accesses. Attribution cost is
explicitly marked not yet included in standalone engine output; no compute
efficiency claim is allowed from that output.

## Metadata lineage and previous data use

[The metadata receipt](LINEAGE_METADATA.json) checks the archived registry,
the last-epoch protocol, both checkpoint bindings and all 16 files in an
available historical runtime tree. Five files in the current repository's
general runtime differ from that protocol, including the FT adapter; use the
verified historical tree, not the latest checkout by assumption. This is source
and artifact provenance, not measured inference fidelity.

A related training-only calibration selection protocol is bound to the same
native D2 manifest. The new pilot is consequently retrospective, not a fresh
confirmation. Exact prior per-row accesses were not reconstructed. The
previous no-capture-independence limitation remains.

## Cost planning: measure before full extraction

With 78 features, 32 references, eight repeats, 64 target rows and two
checkpoints, a straightforward no-invariant-skip evaluation has approximately

```text
64 * 2 * (1 + 8 * 32 * (2 * 78 + 1)) = 5,144,704 native batch calls.
```

This is planning arithmetic, not a runtime measurement or a promise that this
many calls will occur. SHAP may skip exact invariants. Each call retains up to
512 context rows; reducing the displayed target count does not make these
calls singleton inference. Do not start the full workload on this calculation.
A separately reviewed label-blind, bounded profile must establish elapsed time,
actual call counts, peak memory and source/perturbation fidelity first.
Changing reference effort or method afterward needs an explicit new binding,
not an unrecorded fallback chosen from evaluation performance.

## Evidence boundary and next execution gate

Current verification uses tiny synthetic arrays, synthetic neural heads and
the real SHAP package. No project feature/checkpoint tensors were loaded.
The implemented orchestration accepts callbacks but does not itself verify
an external review, construct a genuine data-governance approval or authorize
HPC use. No real-data launcher is supplied. The provided source-kind string
cannot confer authorization.

Before a real profile or study: complete the hash-bound bundle/loader,
finalize reproducible repeat RNG/settings, measure safe resources, verify
native and masked targets and obtain independent execution review. Keep
official DICC submission controls separate from local synthetic tests.
The existing strict experiment is not modified.

Sources for the implemented attribution interface:
[PermutationExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html)
and [Independent masker documentation](https://shap.readthedocs.io/en/latest/generated/shap.maskers.Independent.html).
The context adapter and precision checks are engineering controls, not the
candidate innovation; explanation-informed selection utility remains untested.
