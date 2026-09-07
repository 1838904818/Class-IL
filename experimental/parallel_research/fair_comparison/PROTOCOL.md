# Prospective, falsifiable comparison ladder

Status: unbound research preparation, 7 September 2026. This is not a retroactive
preregistration or a claim that earlier unequal designs were component controls.

## Immutable common contract

Start with one dataset chosen before any new result. The dataset/split manifest
must bind raw-row identity, duplicate/capture grouping, train/calibration/test
partition, feature order/units, task order and first labelled availability.
Task-0-only transformations include categorical vocabulary. A separate cap or
sampling rule must use only increment-available information; historical ReplayIDS
D2 offline normal-cap construction must be labelled retrospective unless changed
and newly trained. Official test rows never select a model, budget, arm or seed.

Lock encoder architecture (including tokenization, width, depth, heads, dropout),
initial tensor hashes per seed, optimizer/scheduler and numerical environment.
Equal seeds or dimensions do not imply equal initial tensors. Run common Task-0
pretraining once per seed, bind its encoder checkpoint, and load that same tensor
state into every arm; account its actual cost separately and attribute the same
one-time cost to every method. Do not silently omit it from frozen-NME totals.
Arm-specific initial heads require separate hashes and a registered initialization
rule; changes cannot be concealed under a common encoder hash.

Use an explicitly registered fixed final update checkpoint for the first matched
contrast. Later calibration-selected experiments require the same disjoint,
increment-available selection partition, metric, tie-break and stopping rule for
all eligible arms. Never mix an OFRA guarded checkpoint with an unguarded baseline
and describe the comparison as checkpoint-matched. Freeze official evaluation
support, batch/numerical scorer, and all seed identities before comparison.

## Minimum ladder and disconfirming outcomes

| Stage | Smallest contrast and controlled variables | Falsifiable question / stopping rule |
| --- | --- | --- |
| L0: measurement and reference | Shared Task-0 checkpoint; frozen nearest-exemplar-mean, shared-model bounded replay, historical OFRA recipe. All have exact data/label bindings, but losses and exposure are disclosed as different. | Does the existing headline survive an explicitly heterogeneous method comparison? No architecture-only conclusion is allowed, whatever the result. |
| L1: objective control | On the same frozen encoder and identical stored exemplar identities, compare independent binary heads under plain CE versus the registered conditional focal weighting. Keep target assignment, row order, updates, head initialization, parameter budget and checkpoint rule fixed. | Does focal weighting alone explain the gain/loss? If so, an architecture-only explanation is rejected. No router change in this contrast. |
| L2: representation update control | Shared multiclass replay model frozen versus trainable encoder; same raw new/replay draws, CE, classifier initialization and update count. The frozen-NME arm remains a non-gradient reference, not an equal-step arm. | Is continuing encoder training sufficient to change the result? Extra optimizer/activation memory of the trainable arm is disclosed; enforce the preregistered byte ceilings. |
| L3: head/scoring control | Under one frozen encoder, one binary loss and the same raw-row/target schedule, compare a capacity-controlled shared binary head system with independent low-rank heads. Compare head-only, router-only and joint scores on each identical trained checkpoint separately. | Does the head allocation effect survive matched loss/exposure and total-state bounds? Same-checkpoint scoring ablations alone cannot answer the training question. |
| L4: capacity allocation, only if justified | One factor at a time: rank or shared capacity. Any adaptive-rank arm needs fixed-rank, fixed-total-parameter and shuffled/fixed allocation controls under the same total-byte and update ceilings. | If fixed allocation achieves the same result, adaptive allocation has no demonstrated incremental value. This stage has no implementation or chosen ranks here. |

L1/L2/L3 are distinct estimands, not an omnibus claim that every budget can be
identically matched. Binary target multiplicity can differ while raw exposure is
equal. A strict equality contract is admitted only when measured records satisfy
it. Otherwise report a bound/Pareto comparison with all residual inequalities;
do not add fake updates to make NME look matched. Frozen-NME still reads candidate
rows, computes embeddings, forms means and consumes memory/time.

Before each contrast, register its primary changed factor, controlled fields,
exact-match axes, bounded axes and residual confounds. Numerical ceilings must
come from actual small profiling runs, not historical allocation sizes or head
parameter arithmetic. Stop if full matching would require a materially different
question; the author chooses and locks the revised estimand before execution.

## Accounting and outcomes

Measure optimizer updates at successful step calls, raw-row presentations at
consumption, unique raw rows across aliases/copies, old/new presentations, binary
label decisions and multiclass targets separately. Log resumption/retried work
without double-counting committed scientific steps: consumed compute from failed
attempts is a separate cost ledger, never discarded. Audit gradients, candidate
embedding passes, prototype/router fitting and shared pretraining separately.

Inventory all encoder/head parameters, centroids/counts, replay values/labels/
indices, optimizer state, normalization, buffers and metadata. Count live unique
storages once per snapshot, including resident teacher/common copies. Report
checkpoint-retained bytes, optimizer-state bytes, host process-tree peak RSS,
device allocated/reserved peaks, and serialized checkpoint bytes separately;
none is a substitute for another. Include activation/workspace peaks and data
pipeline processes in profiling. Record preprocessing/load/train/prediction/
attribution timings separately with hardware, warmup, batch size and sampling.

Report every registered seed, confusion counts and per-class support at every
checkpoint, accuracy, Macro-F1, balanced accuracy and signed forgetting. Attack
recall and benign FPR apply only to manifests with attack semantics, not Malaya
application labels. Pair seeds, disclose screening/selection history, show
uncertainty and multiplicity families, and do not treat classes/rows/thresholds
as independent training replications. Negative findings remain reportable.

## Dependencies and decision gates

Historical score reconstruction and attribution-target faithfulness remain
separate unresolved gates; mechanism/repair claims cannot pass around them. The
fairness preparation can proceed independently, but a new training run still
requires exact code/data/environment and numerical-target bindings. Human choices
still pending are dataset/split, primary contrast, new seeds, primary metric and
power rationale, byte/update limits after profiling, and checkpoint rule details.
No real field in the accompanying template is filled by synthetic tests.
