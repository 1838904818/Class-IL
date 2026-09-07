# Prospective executable protocol and mechanism-matching control

All scientific thresholds, schedules, bounds, resource ceilings and alpha
levels are prospective design choices, not validated operating settings. The
earlier reference protocol remains unchanged; this separately versioned runner
implements its missing score-level fitting, inference and durable accounting.

## R1: exact operation

For class c, let z_c be its native two-logit margin (positive minus negative),
r be the vector of raw router affinities and C the entire seen class set. The
registered score is

```text
S_c = sigmoid(z_c) + lambda * (r_c - mean_C(r)) / (population_std_C(r) + epsilon).
```

One R1 action replaces ONLY z_c with a*z_c+b. a is bounded and strictly positive;
its reciprocal is a temperature. b is bounded. Both intervals include identity
(a=1,b=0). Every other head, all raw affinities, the normalization class set,
fusion settings, feature transforms and class identities remain fixed. One
shared parameter pair applies to every row. No new class or head is created.

Fit with the same registered full-batch projected gradient-descent steps for
every active arm. The objective is class-balanced multiclass softmax cross
entropy on the fused scores plus the registered squared distance to identity.
Both optimizer derivatives are analytic. There is no early stopping or candidate
selection from acceptance labels. A runtime limit causes failure/rollback, not
an unrecorded shorter successful fit. Fixed-step costs remain reserved on failure.

This is an actual score-postprocessing calibration fit, not a full neural-model
repair, a new explainer, a checkpoint weight update or a deployment action.

## A/B/C/D and the limits of state_share

On the same ordered rows and old target, A uses old checkpoint/old normalization/
old rivals; B uses new checkpoint/old normalization/old rivals; C uses new
checkpoint/full normalization/old rivals; D uses new checkpoint/full
normalization/full rivals. D is the prediction target. The reference computes
B-A, C-B and D-C directly from independent native logits and raw affinities and
checks telescoping and new-rival monotonicity. It does not invert standardized
scores. Missing old raw inputs leave the decomposition unavailable.

The expansion-aware priority retains the earlier hypothesis:

```text
harm = lower simultaneous bound on old-class error increase
state_share = max(0,-mean(B-A)) / sum(abs(mean(each path effect)))
priority = harm * (1 + w*state_share)
```

The denominator-zero case has share 0. Eligibility also requires harm above its
registered threshold and A/D explanation distance above the method's internal
noise bound. Harmful pure expansion is not automatically discarded. w=0 removes
decomposition weighting; the no-attribution arm removes the explanation gate.
The score-effect summary uses the registered class-row mean, while error
uncertainty uses the separately declared group-equal estimand. This distinction
must be retained in reporting, not hidden behind a causal interpretation.

state_share is NOT an R1-response model. State change can be heterogeneous or
reverse row ordering, which no shared positive affine logit transform fixes.
Conversely, a pure rival/normalization effect may be correctable within R1's
bounds. The runner therefore includes action-matching controls:

- R1-reachability: harmful-error evidence, ranked by harm times the fraction of
  currently wrong target rows with a positive pointwise attainable margin bound.
- Expansion-R1-reachability: the same action-specific ranking plus the
  method-internal attribution gate. It does not use state_share weighting.

For each row, the maximum target logit is
max(a_min*z_c, a_max*z_c)+b_max, while rivals remain fixed. A bound that cannot
cross the winning rival rules out that row under R1. A bound that CAN cross is
only a necessary possibility, not a jointly attainable improvement certificate.
The optimizer must still use one shared (a,b), and all-class acceptance remains
mandatory. The bound may legitimately predict no actionable repair.

An explicit counterexample is tested. Two wrong target rows have z=1 and z=-1,
with rival scores .95 and .8. Under a in [.5,2], b in [-2,2], each row's separate
upper bound crosses its rival. But joint success would require simultaneously
a > logit(.95)-2 > .944 and a < 2-logit(.8) < .614. No shared parameter pair can
satisfy both. These numbers define a mathematical software counterexample,
not a empirical traffic finding or a validated parameter recommendation.

## Partitions, selection and durable single use

Trigger, fit and acceptance rows/labels must be increment-available training
resources. They are disjoint from each other and from subsequent evaluation by
both row identity and leakage/capture group. The manifest carries identities
and availability metadata without future labels. The native adapter rejects
official-test origins for all decision partitions and binds separate label files.

Before any labels are loaded, a SQLite transaction freezes source, policy,
source/runtime identity and row/group claims. Claims cover all four partitions
and survive process restart. All ten candidate pools lock before fitting begins.
The runner re-loads hash-bound trigger inputs and independently recomputes a
pool before reserving the chosen top target. Reordering or fabricating a pool
cannot bypass eligibility. Both policies and the R1 operation are bound before
fitting; the fitted state locks before acceptance is loaded.

One repair attempt is available per arm per study. Audit-only has zero attempts.
No rejected candidate is replaced using the same acceptance labels. Any later
study reusing a row or group is rejected by the same persistent ledger, even if
its increment or study name changes. A crash leaves the attempt and claims
spent. No CLI offers reset/retry-until-success. Preserve the ledger; deleting it
or renaming identities would violate the protocol, not create new evidence.

Subsequent labels are read ONLY in a distinct evaluate stage, after all arm
decisions durably lock. The check precedes even hashing the labels file. A
failed label read/evaluation also remains spent. A later outcome cannot change
the already locked target, parameters, gates or result selection.

## Paired group-level simultaneous uncertainty

For each registered metric, compute per-row paired differences between the
proposed and baseline predictions. Average within each represented independent
capture/group, then weight the G represented groups equally. This is explicitly
not a row-weighted population estimand. Arbitrary dependence inside a group is
allowed; independence between groups and correct grouping are required.

For group values bounded by [l,u], the implemented interval is the sample group
mean plus/minus

```text
(u-l) * sqrt(log(2*M/alpha)/(2*G)), clipped to [l,u].
```

Hoeffding's bound for independent bounded group values followed by a union bound
gives simultaneous coverage for at most M registered contrasts under these
assumptions. The estimator does not claim independence for duplicated flows.
Repeating rows inside an unchanged group does not narrow the interval. M covers
all registered trigger classes/arms, all acceptance class/aggregate metrics,
and both proposed and effective subsequent contrasts. No bootstrap normality,
asymptotic small-support approximation or iterative testing-until-pass is used.

For explanations, normalized signed attribution vectors have L1 norm 1 and a
half-L1 distance in [0,1]. Repeated matched A/D draws estimate change; disjoint
repeat pairs within A and D estimate internal noise, taking the larger floor.
The data contract requires independent random draws by group. Shared unmodeled
background randomness across groups would invalidate the stated coverage and
must be addressed by the real attribution experiment, not ignored.

Acceptance requires lower-bound target benefit and upper-bound error-increase
limits for EVERY old and new class. With authorized attack/benign semantics,
also constrain missed-attack and false-positive-rate changes using their proper
supports. Malaya application labels never produce attack recall/FPR. Small
independent-group support, missing classes, malformed evidence or overly wide
intervals lead to rejection/abstention, not waived constraints.

Subsequent output contains actual row confusion counts, per-class precision/
recall/F1/support, accuracy, Macro-F1, balanced accuracy and signed paired error
changes. It separates rejected proposed counterfactuals from the effective
baseline policy. A good rejected proposal on later data does not retroactively
become an accepted intervention.

## Cost and failure accounting

Common active budgets cover unique acquired trigger/fit/acceptance labels, one
attempt, fixed optimizer steps and paired acceptance scoring. Every active
attempt uses the same fit/acceptance pools, recipe and step cap. Failed/rejected
attempts remain charged; no candidate is free because it failed. Diagnosis
revalidation is charged in addition to initial diagnosis. No simple arm is
forced to execute or load SHAP to appear compute-matched.

Measured quantities include fit wall/CPU seconds, actual completed optimizer
steps and processed fit rows, traced allocation peak, explicit input-array
bytes, diagnostic processing/time and paired scoring/time. Tracemalloc peak is
not process RSS or GPU memory, and input arrays allocated before tracing are
listed separately. Diagnostic cells are an accounting proxy, not FLOPs.
Precomputed checkpoint inference and SHAP costs are UPSTREAM costs: they must
be measured from their bound export/attribution receipts in a real comparison.

Equal caps do not mean equal actual intervention use, total time or compute.
Report abstentions, off-schedule opportunities, failures, overhead and unused
audit-only repair budget. The runner never labels audit-only an active
compute-matched arm and never automatically claims total compute matching.

## Scientific claim boundary

Score-level software closure does not close historical GPU fidelity, training
lineage, faithful explanation extraction, dataset governance, independent
capture validation, real resource profiling, or evidence of an added mechanism.
Those require the actual experiments/reviews. Existing expansion-routing and
drift-to-action literature is not displaced by this implementation. No first-
of-its-kind claim, human-review action or favorable outcome is manufactured.
