# Expansion-aware repair controls: prospective local preparation

Status: research-only protocol and software-invariant reference, 7 September 2026.
No model repair, training run, real-data improvement, deployment, new human
review action, or algorithmic priority is established by this directory.
No cluster script or submission is included. Protected historical results,
submitted manuscripts, and already submitted jobs are outside this change.

## 1. Question and falsification

Does expansion-aware prioritization identify useful old-class repair targets
more efficiently than simple triggers when all active arms have the same
available labels, the same bounded repair recipe and the same intervention
budget? A second, stricter question is whether any advantage remains after
charging the candidate's extra diagnosis cost. Explanation change alone is not
harm, and a pure normalization or new-rival effect can still cause real harm.

The proposed contribution is narrower than explanation drift, adding actions
to a drift monitor, a four-term telescoping identity, or a new acronym. Those
are not novelty claims. This protocol tests a particular, order-dependent
diagnostic priority rule and its possible utility. The separate prior-art work
must determine overlap and missing controls; the existing research plan already
identifies CLEX and Drift2Act as novelty risks. No additional literature claim
has been verified by this implementation task.

Reject the operational-contribution claim if benefit is absent, unsafe,
unreliable across independent runs/captures, explainable by extra resources, or
not distinguishable from the simple controls. Retain the historical ledger as
supplementary auditing, or omit it from the main method. A rejected repair or
negative experiment remains a result, not an invitation to retune on evaluation.

## 2. Entry dependencies and numerical targets

Before any real-data mechanism experiment, close the numerical and attribution
gates in the existing experiment-gates document. In particular, reconstruction
of historical deployed scores is distinct from a float64 arithmetic reference,
and forward equality is distinct from faithful derivatives. This directory
does not close either dependency.

Use the registered A/B/C/D path on identical ordered rows, fixed old target c,
and verified independent family-head scores/raw router affinities:

- A: old checkpoint, old normalization classes and old rivals.
- B: new checkpoint, old normalization classes and old rivals.
- C: new checkpoint, full normalization classes and old rivals.
- D: new checkpoint, full normalization classes and full rivals.

D remains the deployed classifier score; counterfactual B/C never replace
headline predictions. Bind checkpoint/code/data/row hashes, class axes, dtype,
batching and normalization semantics. Let s, n, r be the trigger-row means of
B-A, C-B and D-C. Their sum equals mean(D-A) up to the registered arithmetic
tolerance. This check is not causal identification or numerical fidelity.
The convention is order-dependent and allows cancellation among components.

Explain registered A and D with the same validated explanation method, target,
feature space, background policy and mask policy. Freeze a nonnegative distance.
Estimate its uncertainty and method-internal stochastic floor from repeated
background/mask draws and appropriately blocked rows, on development/trigger
data only. Pair repeated draws where justified. The floor must come from the
same method, not cross-method disagreement. If making later attribution claims
about B/C, explain B/C faithfully too; this reference does not implement that.

## 3. Freeze the arms before outcomes

All arms start each paired one-step comparison from the identical unrepaired
baseline checkpoint. They see the same increment-available labeled pools,
ordered old-class target pool, split/capture grouping and common numerical
validity gate. No arm can select a target without sufficient registered support.
No automatic new supervised head is permitted.

| Arm | Frozen target selection | Active intervention |
|---|---|---|
| random | Seeded permutation of eligible old classes; independent of signal magnitudes | Same R1 package when allocated |
| periodic | Fixed increment schedule and rotating canonical target order; not outcome-derived | Same R1 package on due opportunities |
| error-only | Rank by lower uncertainty bound of old-class error increase, above registered harm threshold | Same R1 package |
| disagreement-only | Rank by lower bound of prediction disagreement rate, above registered threshold | Same R1 package |
| expansion-aware candidate | Harm evidence plus method-internal drift evidence; priority below | Same R1 package |
| audit-only | Store immutable flags, blocked reasons and costs; no action | None; report unused repair budget |

Simple arms need no attribution, within-method noise or four-score decomposition
input. Random/periodic also need no error or disagreement estimate. The reference
supports those fields being absent. Its audit output lists common-validity
records, not a newly implemented historical ETG-state algorithm. If the audit
arm computes the expansion candidate's flags, it calls that diagnostic separately,
records the incurred cost, and still never calls repair reservation.

The expansion candidate is eligible only when L_h > h_min and
L_d - U_noise > d_min. L_h is a lower bound for the paired increase in target
classification error under deployed D versus the registered baseline; L_d is a
lower bound for the A/D explanation distance; U_noise is the corresponding
method-internal upper noise bound. All use trigger data and a registered
simultaneous uncertainty procedure. Passing these tests indicates evidence for
investigation, not that repair will help.

For eligible candidates, define

```text
state_share = max(0, -s) / (abs(s) + abs(n) + abs(r))
              or 0 when the denominator is exactly 0
priority = L_h * (1 + w * state_share)
```

Order by descending priority, then descending drift excess, then immutable
candidate ID. w is nonnegative and selected using development data only. This
is an intentionally falsifiable prospective heuristic, not a calibrated risk
score, estimated causal responsibility or validated optimum. Harmful pure
expansion retains positive priority; it is not filtered out merely because
s = 0. w = 0 is a required decomposition-removal ablation with all other gates
unchanged. Additional harm-only and noise-gate-removal ablations isolate whether
any value comes from the proposed component or from ordinary supervised risk.

## 4. Repair recipe and label partitions

The first proposed recipe, R1, is bounded affine/temperature calibration of an
existing target-family score under the deployed fusion rule. Encoder, router,
other family heads, class identities and feature transforms are frozen. Freeze
its exact parameterization, allowed parameter domain, differentiable objective,
optimizer, initial state, fit steps, row exposure, negative sampling and rollback
before testing. The code here DOES NOT implement R1, select its parameter bounds,
fit it, compute repaired predictions, or install a repaired state. A bounded
family-head update is an alternative future recipe and requires a separate
registered experiment; do not mix it into R1 or choose between recipes using
acceptance/evaluation outcomes.

For increment t, separate four partitions before accessing outcome labels:

| Partition | Label availability | Allowed use |
|---|---|---|
| trigger | Observed and labels available by t | Support checks, frozen triggers and priority |
| fit | Observed and labels available by t | Fit one locked R1 candidate |
| acceptance | Observed and labels available by t; held from fitting and ranking | One accept/reject evaluation of the locked candidate |
| subsequent | Observed after t; labels sealed from this gate | Estimate intervention effects only after decisions are locked |

Partition membership is disjoint by row and by leakage group. A leakage group
must cover duplicates, shared flow/session, host/capture and other relevant
dependencies; string uniqueness alone does not establish that grouping was
correct. Register the grouping design and independent-capture holdout. All fit
and acceptance pools need adequate old/new class support, not just aggregate
sample count. Trigger support is required per candidate target. Missing or
delayed labels mean abstain; never invent semantic labels for an unlabeled
cluster or use subsequent data as substitute support.

Rank the target, freeze its content fingerprint, arm and BOTH trigger/acceptance
policy fingerprints, reserve a package,
fit using fit-only labels, then hash the baseline/repaired states and fit record
before acceptance. There is one attempt per acceptance partition in this
reference. Failure and rejection spend the partition and the reserved cost;
never try another target or hyperparameter on that same acceptance set until
one passes. An extension with multiple candidates needs a separately justified
sequential/multiplicity protocol and new code, not a loop around this gate.
Cross-increment partition reuse must be tracked persistently by the future
driver; creating a fresh in-memory ledger is not permission to reuse labels.

## 5. Acceptance is not subsequent evaluation

Use paired baseline/repaired predictions on exactly the same acceptance rows
and the complete deployed class set. The externally implemented uncertainty
procedure must account for all registered constraints/classes and the actual
sampling unit. Report the interval method, confidence level, grouping/resampling
policy and any sequential allocation. Do not pool unstable classes or pretend
that a small nominal confidence interval solves lack of independent support.

Require a lower bound on target error improvement above the registered minimum
and an upper bound on error increase below the registered tolerance for EACH
old class and EACH new class. Target improvement is the exact sign reversal of
the same paired target error-delta interval, not a separately chosen favorable
statistic. Missing metrics, inconsistent support, nonfinite intervals, wrong
state/partition fingerprints or insufficient support fail closed.

For data with an authorized explicit binary attack/benign class mapping, ALSO
require upper bounds for missed-attack-rate change and false-positive-rate
change. The first uses all acceptance attack rows and the second all acceptance
benign rows. Record confusion counts and the exact mapping. These safety checks
are not guarantees about deployment populations.

Malaya application labels do NOT support attack recall, missed attacks or FPR.
For Malaya use application-class error/recall, macro measures, confusion counts,
old-class retention and new-class learning. The reference rejects attack metric
names and attack tolerances in application mode. Changing an enum does not
authorize changing the meaning of dataset labels.

Accepted means eligible for the predefined research comparison only. Do not
suppress or delete raw alerts. Preserve the original baseline scores/alerts
and separately log proposed repaired scores and research decisions. Neither
an accepted candidate nor this code changes a production classifier.

## 6. Budgets and what matching actually means

Freeze identical caps across active arms for unique acquired labels, active
repair attempts, optimizer steps, acceptance scoring and diagnosis. All active
attempts use the same R1 package, common fit/acceptance label pools and the same
baseline/repaired acceptance-row accounting. Shared trigger labels count once
per arm as acquired information even if random/periodic do not inspect their
values; record both acquired and actually used labels. Charge rejected, failed
and rolled-back attempts, repeat model/attribution evaluations, model load cost,
checkpoint I/O and peak memory. No free discarded candidates.

The in-memory reference accounts for unique row-label IDs, diagnostic units,
fixed fit steps, paired acceptance forward rows and attempts. It is not a GPU
meter, cannot infer FLOPs from steps, and does not verify a caller's charges.
Choose interpretable accounting units before profiling. Collect actual
train/load/prediction/attribution time, device characteristics, optimization
steps/row exposure, processed attribution masks and peak CPU/GPU bytes in the
future measured experiment. A compute envelope is not proof of actual matching.

Do NOT force simple controls to compute SHAP solely to hide candidate overhead.
Report diagnostic cost separately and in total cost. The comparison helper
checks equality of realized reference intervention counters and labels for all
five active arms, but allows and exposes differing diagnostic units. It always
returns actual_compute_verified = false. If diagnosis differs, do not claim
total compute matching; test budget-utility curves under a common total ceiling
or run a separately registered total-cost-matched comparison using measured cost.

Trigger abstention, an off-schedule periodic arm, support failure or cost limits
can cause unequal realized intervention use. Keep those outcomes in the
intention-to-treat ledger. Do not discard those increments, force an unsafe
attempt, fabricate a repair or pad dummy work to manufacture matching. Report
same AVAILABLE budget, actual costs and abstentions. A stricter same-active-
budget claim is unavailable until its realized matching conditions are met.
The code's equality check fails when active intervention counters differ.

Audit-only does not repair. Its unused active budget must be reported and it
must NEVER be called an active compute-matched repair arm. Compare it as an
alarm-only/no-intervention outcome and cost reference.

## 7. Evaluation, uncertainty and stopping

Lock decisions, code/config/model/data/environment hashes and all exclusions
before unsealing subsequent labels. On later held-out rows, compare repaired
and unrepaired branches with the same labels/class set. Include accuracy,
Macro-F1, balanced accuracy, per-class support/confusion counts, signed
forgetting, old/new class behavior, accepted/rejected/abstained counts and the
binary attack measures only where semantics permit. Report all registered
seeds/captures, including failures and costs, not a best run.

Use a prospective independent evaluation sequence; previously inspected
historical tests are not unseen confirmation. A one-step paired study isolates
the immediate intervention; cumulative repair trajectories introduce altered
future checkpoints and require a separately frozen longitudinal experiment.
Do not pool these estimands. Bound the number of comparisons and report paired
uncertainty at the independent unit, not just flow-level resampling. No outcome
from subsequent evaluation can tune a trigger, target, repair recipe, threshold,
partition, checkpoint or stopping rule in this study.

Every scientific threshold and design parameter in this document or reference
is a PROSPECTIVE DESIGN CHOICE, NOT VALIDATED. Production defaults are absent.
Select and justify them on training-only development before the experiment:
support minima, harm/disagreement/drift thresholds, explanation distance,
confidence/multiplicity method, w, numerical tolerance, periodic schedule,
random seed, parameter bounds, fit steps and all safety tolerances/budgets.
The numerical values in unit tests are fabricated software fixtures only.

Stop before real experiments if any dependency, label governance, partition
provenance, fidelity, uncertainty, capacity, measured resource or safety gate
is unresolved. Any eventual cluster stage separately requires current DICC
rules/live evidence, exact immutable bindings, preflight, independent approval
and one exact-command user confirmation. This protocol authorizes none of those
remote actions and does not report current cluster status.

## 8. Truthful research record

Describe this artifact as a prospective control protocol and reference gates.
Do not say a model was repaired, a human reviewed a case, a trained controller
exists, or a mechanism improved IDS performance on the basis of synthetic
fixtures. No author motivation, intellectual history or simulated analyst
decision is manufactured here. Record actual researcher choices when made,
and disclose research assistance according to applicable publication rules.
