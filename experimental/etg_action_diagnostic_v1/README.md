# Two-old-class action-identifiability audit

12 September 2026. Metadata-only diagnostic library. No real experiment result,
fit or held-out evaluation is supplied. The frozen five-arm protocol and the
separately approved extraction candidate are not modified.

## Source facts and the identification bottleneck

The audited source is the current supervisor-facing
`experimental/etg_exploratory_v1/{pilot_core,one_use,native_r1}.py`.

Source SHA-256 at inspection:

| Source | SHA-256 |
|---|---|
| pilot_core.py | `122371c1b52caf1acc1a91ce7383fd90f4493ca743bf28cb6eb8838eec656839` |
| one_use.py | `d8b70d31e6a2dac0a99fcfb09b219ff7576d1f563d2e236b0c5358fb6753d10b` |
| native_r1.py | `2ba31a2a1769c836bf59b96fb5e7f7b8a6843e1d958c0910f113244d875e4c1e` |

- `one_use.derive_candidates` creates candidates only for old classes 0 and 1.
- `pilot_core.choose` first requires error increase strictly above zero.
  Error-only selects the largest error increase, breaking ties by class id.
  Raw-drift and noise-aware retain exactly the same ranking; they only filter
  eligibility using `between > 0.05` or `excess > 0.05` respectively. The 0.05
  value is the existing default, not introduced by this diagnostic.
- `one_use.run` supplies all active arms the same fit population and fixed R1
  settings. `native_r1.fit` starts from `[1,0]`, performs a fixed number of
  projected gradient steps, and uses no arm name or explanation in its loss.
- `native_r1.adjust` changes only the selected class-score column. Identity
  returns the original native float32 joint score directly. The empirical
  acceptance rule is shared. A rejected arm's effective policy is baseline.

Let H be the harmful old classes and let h* be their common rank winner. For
an explanation gate G, the selected target equals h* precisely when h* is in
G, assuming H is nonempty and attribution for every harmful candidate is valid.
If H is empty both selectors abstain. If H contains one class, gating can only
retain or abstain. If H contains two classes, it can retain h*, select the other
eligible class when h* is excluded, or abstain. If required attribution is
unavailable the comparison is unavailable, not an abstention or zero drift.

The two explanation gates are nested for the actual signal implementation:
each row's noise floor is nonnegative, hence `median(between_row-noise_row)` is
at most `median(between_row)`. With their shared threshold, the noise-aware
eligible set is a subset of raw-drift's eligible set, itself a subset of H.
This restricts possible actions; it does not establish that the stricter gate
is better. The random-harmful arm uses one fixed hash ranking over H; one
salt/checkpoint comparison is not repeated statistical evidence about random
selection performance.

For the same target c, let the common frozen inputs be D_fit, settings s and
native numerical contract n. When both executions complete all fixed steps
successfully with identical numerical behavior, their fitted parameters obey

`theta_A = Fit(c, D_fit, s, n) = theta_B`.

With the same acceptance inputs/rule, acceptance decisions are identical. Hence
for any evaluation row x under the same native context,

`effective_A(x) = effective_B(x)`.

Their effective accuracy/error/recall/F1 and old-class error are therefore
identical on any identical evaluated population. Any improvement over baseline
belongs to the common R1 correction, not to an incremental explanation-based
selection benefit. This is a mathematical conditional statement, NOT a claim
that the current real experiment has already selected the same class.

Both arms rejecting (or validly abstaining) also yields baseline-versus-baseline
effective identity, even when their proposed corrections differ. That case
does not rule out useful differences in attempts avoided, rejection rate or
cost; such quantities need separately complete accounting, including SHAP.

Different targets or apply-versus-baseline makes a different locked policy
possible, but does not prove predictions differ on held-out rows or that the
explanation arm improves them. A single pair is descriptive, not general
efficacy, novelty, population risk control or a full-stream forgetting result.

## Why target equality alone is insufficient

Walltime can interrupt fixed-step fits. A failed arm is not a zero-effect arm.
Floating-point environment, source order or native batch/context mismatch may
break determinism. `fit` returns wall seconds, so hashing the whole fit object
can differ even for identical parameters. Conversely a caller-supplied hash
or a `real-inputs` label does not verify provenance or equality on its own.

The diagnostic therefore requires externally verified locked event hashes plus
per-arm semantic hashes. For accepted same-target arms, a differing fit hash is
`UNRESOLVED_SAME_TARGET`, not evidence that the explanation mechanism improved
the correction. Both completed corrections retain identical shared binding
hashes for protocol, baseline, fit input, acceptance input, settings, correction
code, acceptance rule and numerical environment before comparison is possible.

## Metadata contract

`audit_locked_choices(snapshot)` is a pure function. It accepts only a closed
metadata schema and returns ten arm-pair diagnoses plus the preregistered
noise-aware versus error-only comparison. It performs no reads or writes.
Labels, scores, attribution arrays, metrics and fitted parameter arrays are
rejected as extra fields. Tests contain synthetic hash strings only.

The caller must first verify the immutable `CANDIDATES_LOCKED`, `FITS_LOCKED`
and `DECISIONS_LOCKED` events against their real stored bytes. Then it projects
metadata without opening the evaluation role. This utility does not implement
or authorize that projection, reopen an experiment, or independently verify
the supplied event hashes. Every report keeps
`external_hash_verification_required=true` and
`scientific_efficacy_established=false`.

The auxiliary **fit semantic hash** must canonically bind target, exact fitted
parameter values, completed step count, correction algorithm/settings and
native numerical/input identity. Exclude elapsed seconds, arm name, paths,
timestamps and other telemetry. Retain the full original fit-event hash as
separate immutable provenance; never replace it with a telemetry-stripped hash.
The **decision semantic hash** binds target, acceptance Boolean, exact
class-correct-count changes, attack-miss and Benign-FP changes and the shared
acceptance rule/population; exclude arm label, paths and telemetry. These are
new diagnostic metadata projections, NOT edits to registered output schemas.

`source_kind` is a caller label (`synthetic` or `real-inputs`), never proof of
real provenance. `locked_stage` is DECISIONS_LOCKED, EVALUATION_SPENT or COMPLETE.
Resource/fit failures in the receipt yield UNAVAILABLE classifications, and
must still be reconciled with the registered pipeline: current `one_use.run`
fails the overall study when a fit raises. A partial failed study is not made
complete or authorized to evaluate by this diagnostic.

Status meanings:

| Status | Permitted interpretation |
|---|---|
| SAME_ACTION_NO_SELECTION_GAIN | Conditional identical effective policy for this comparison; not evidence of incremental explanation selection gain. |
| DIFFERENT_ACTION_REQUIRES_HELDOUT | Locked policies differ; held-out performance, uncertainty and full cost remain unmeasured here. |
| UNAVAILABLE_ATTRIBUTION / RESOURCE / INCOMPLETE | Missing scientific comparison, not a negative efficacy result. |
| INCOMPARABLE_BINDINGS | Shared correction/input contract does not match. |
| UNRESOLVED_SAME_TARGET | Need independently verified correction semantics before claiming action identity. |
| INCONSISTENT_FIT_SEMANTICS / DECISION_SEMANTICS | Metadata conflicts with its declared hash contract; investigate, do not publish a gain. |

The statuses must never be used to stop or change predeclared held-out access,
retune thresholds after seeing labels, choose another checkpoint from results,
skip controls, or modify the approved extraction job. Any future policy change
is a separately reviewed protocol, not a rescue interpretation of this pilot.

## Pure software tests

```powershell
python -B -m unittest discover -s experimental/etg_action_diagnostic_v1 -p 'test_*.py' -v
```

These tests exercise only metadata and conditional logic. They establish no
model fidelity, extraction fidelity, SHAP stability, ETG benefit or innovation.

Independent local verification on 12 September 2026: 14 synthetic metadata
tests passed. The diagnostic does not verify real artifact origin, read held-out
labels, compute efficacy or authorize a future experiment.
