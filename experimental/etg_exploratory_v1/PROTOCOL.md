# Early-checkpoint descriptive ETG pilot, version 1

11 September 2026. Preparation scope accepted; experimental benefit untested.
The original ten-arm, independent-group protocol is unchanged. This is an
explicitly different **row-level exploratory study without risk certificates**,
not a workaround that relabels correlated rows as independent captures.

## Small question, fixed architecture

For one fixed OFRA transition, does a method-noise-adjusted explanation gate
choose a more useful bounded calibration action than the same performance-only
selector? The primary comparison is `noise-aware` versus `error-only`.
No new classifier, router, LoRA, head, fusion coefficient or training loss.
This evaluates explanation-informed action selection, not whether explanations
are true, whether people understand them, or whether pure silent drift is fixed.

The chronology rule fixes native D2 training seed 1, checkpoint 000 to 001:
Benign / DoS GoldenEye become old classes; DoS Hulk / DoS Slowhttptest arrive.
This is the first covered transition, not the best observed transition.
Later tasks, especially the one-row Heartbleed calibration class, are outside
this pilot. Their exclusion cannot be described as solving full-stream coverage.
Two old-class candidates and one transition cannot validate efficacy, variability,
rare-attack safety, multi-seed robustness or a complete forgetting trajectory.

## Bound resources and disjoint roles

Use only the **original native D2 calibration indices**, reconstructed against
their archived hashes. Do not use the new P/O calibration pool or its embeddings.
Hash-rank offsets independently within each class with a fixed salt, then
allocate 32 trigger, 64 fit, 64 acceptance and 64 evaluation rows per class.
That is 896 role rows; only 64 old-class trigger rows need paired attribution.
The smallest included class has 329 native calibration rows, exceeding the 224
required. All unselected rows remain unused, not a retry reserve. These counts
are engineering choices for feasibility, not a power calculation or a literature
requirement. The resulting balanced sample is not the original class prevalence.

Use 16 native Task-0 fitting rows per old class as a fixed explanation reference
pool, separate from all 896 role rows. Both checkpoints share those references.
Training seed 1, original data split seed 42 and the pilot's hash salt have distinct
roles. Existing calibration label use must be registered before real execution:
new row assignments do not erase prior tuning or make this a fresh confirmation.

Record source offsets, native-calibration ordinals and original class-batch
contexts. First score complete parent calibration classes in their recorded
order with the bound 512-row native call convention; then select role rows.
Never change Router-call boundaries merely because the displayed subset is
small. No official test row can select, fit or accept an action. The new
evaluation subset is still a retrospective, same-pool diagnostic, not a future
traffic sample, an independent capture study or a full-test result.

## Candidate rule and ablations

For a fixed old class c, compute the increase in empirical error on its trigger
rows, h_c = error_new(c) - error_old(c). Every nonrandom active selector uses
exactly the same descending h_c priority, class ID breaking ties.

For each old/new checkpoint and identical row/target, obtain eight independent
stochastic repeats from one frozen explanation method. Normalize each signed
attribution vector by its L1 norm. Let d be half-L1 distance.

- Per row, between change is the median of the eight matched old/new distances.
- Per row, noise is the larger of the old and new 90th percentiles of four
  disjoint within-checkpoint repeat-pair distances.
- Class excess is the median across its rows of between minus noise.

The 90th percentile is a small empirical noise diagnostic, **not a confidence
bound, hypothesis test, or correction of systematic attribution bias**.
All-zero, nonfinite, unverified or incompatible attribution makes the
explanation arm unavailable; it is not zero drift. Keep signed vectors;
a top-15 Jaccard threshold is not used.

| Arm | Eligibility and action budget |
|---|---|
| Audit only | No action; preserve original alerts and ledger. |
| Error only | h_c > 0; highest-harm target; at most one attempt. |
| Raw drift | Same, additionally median between > 0.05. |
| Noise aware | Same, additionally class excess > 0.05. |
| Random harmful | One fixed hash-random target among h_c > 0; same attempt cap. |

Eight repeats, quantile 0.9 and threshold 0.05 are explicit project design
choices, not numbers from SHAP, CLEX or DICC. Do not sweep them against
evaluation outcomes or use a favorable explainer after looking at the result.
The subtraction is a candidate noise gate, not an invented Shapley estimator.
Raw-drift is the ablation for the noise subtraction; error-only is the ablation
for explanation information; random-harmful is a low-information selector.
If they choose the same target, there is no demonstrated selection advantage.
If a gate abstains, compare actual spending: equal caps are not equal compute.
No performance harm means no action, even with large explanation change.
Such silent-drift events remain in the audit ledger.

## Same correction, empirical rollback

Each attempted arm must use exactly the same existing R1 operation: a positive
bounded affine transform a*z_c+b on one old head's binary logit margin, keeping
other heads, native router terms, encoder and raw alert output unchanged.
R1 is prior-style calibration, not the contribution. Fit one shared pair using
fit rows only; numerical bounds, optimizer steps and time limits must be frozen
in the independent execution candidate, not copied from a software demo.

On the acceptance sample, require at least one extra correctly classified target
row, no loss of correct-count in any of the four classes, no additional attack
miss and no additional Benign false positive. Otherwise revert to identity.
This is an empirical guard over 64 rows per class: one row is 1.5625 percentage
points. It does not certify population non-harm. Do not retry rejected targets.

The future driver must freeze all candidates before fitting; persist row-use,
attempt, fitted-state and decision hashes; count failed attempts as spent; lock
every decision before exposing the evaluation labels. Identity is a valid result.
The current pure predicates do not implement or substitute for that driver.

## Real target and computational gate

Permutation SHAP is the proposed primary model-agnostic method, because its
model callable can use the actual scorer instead of an unverified surrogate
gradient. This is a proposal, **not a completed faithful explanation exporter**.
Explain the old class's native fused-score margin over its strongest available
rival at each checkpoint. Bind class order and tie conventions.

Because historical Router evaluation can depend on call context, each feature
mask must replace only the designated row in a frozen parent native batch.
All companion rows remain fixed; never batch unrelated masks as if they were
native flow companions. This defines a context-conditioned explanation, not an
intrinsic explanation independent of batch context. Off-distribution masks and
the chosen reference distribution remain explicit limitations.

Before 64-row extraction, an independently reviewed, label-blind feasibility
profile must check the actual native target under unmasked AND masked inputs,
deterministic replay, reconstruction, signed-vector validity, runtime and memory.
Freeze SHAP version, masker, fixed references, permutation effort, RNG schedules
and additive reconstruction diagnostic. Forward equality and additivity alone
do not certify explanation truth or absence of bias. No GPU/CPU resource request
or real profile execution is authorized by the metadata planner.

## Results and failure criteria

Primary endpoint: difference in balanced old-class error on the 128 old-class
evaluation rows (64 each), noise-aware minus error-only; lower is better.
Report absolute counts, proposed AND effective outcomes, all four class recalls,
precision/F1/support, confusion matrices, balanced accuracy, Macro-F1, Benign FPR,
attack recall, attempts/abstentions, labels/steps/walltime/memory and attribution
overhead. Sample accuracy cannot be advertised as full-population accuracy.

Any accepted change is only an offline counterfactual; do not replace checkpoints,
suppress production alerts or claim a human reviewed it. Old-class recall change
is not a multi-checkpoint forgetting rate.

Reject the proposed utility claim if explanation adds no effective advantage,
if gains require moving thresholds/methods after results, or if protected class
harm appears. A less costly outcome counts as useful only after attribution
overhead and failed/rejected fits are included. Pure abstention or infeasible
inputs give limited/inconclusive evidence, not proof of effective governance.
A positive pilot warrants a new preregistered multi-transition, multi-seed study;
it does not establish novelty, generalization or publication readiness.
