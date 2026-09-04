# Matched continual-learning comparisons: design requirements

Status: design checklist, 4 September 2026. This is not a registered candidate,
an approved job, an implementation, or an experimental result.

## Questions that must remain separate

1. **Scoring contribution:** compare head-only, router-only and joint scores
   from the same checkpoint, data and probe batches. This isolates the scoring
   choice conditional on an already trained model; it does not compare OFRA
   with an independently trained continual-learning method.
2. **Backbone contribution:** change the encoder while retaining the OFRA
   heads, replay, routing and evaluation rules. This is a backbone ablation,
   not an independent baseline. Adapter-specific changes must be explicit.
3. **Continual-learning method comparison:** compare separately implemented
   methods with the same permitted data access and declared resource budgets.
   Cumulative retraining on all seen-class rows is a headroom diagnostic, not
   a bounded-replay comparator. It must remain in a separate result group.

## Controlling protocol and reference

The existing D2 checkpoint-selection reference is documented in the
[paired five-seed package](../results/replayids-d2-checkpoint-selection-paired5/README.md).
Its primary remains last epoch, not the retrospectively most favourable
checkpoint-selection arm. The
[guarded protocol](../results/replayids-d2-checkpoint-recall-guard-paired5/D2_CHECKPOINT_RECALL_GUARD_PAIRED5_PROTOCOL.md)
is a separate ablation, not permission to change the primary.

A new comparison must have its own versioned protocol. Before any training,
bind the exact data/shard hashes, Task-0-only preprocessing and feature order,
class/task schedule, calibration exclusions, reference outputs, source and
configuration hashes, and seed registry. Official-test rows and class supports
must be identical across arms; no test balancing or outcome-driven exclusions.

## Fairness must be measured, not inferred

- Match allowed historical-data access. Record replay rows and bytes, selection
  policy, candidate-pool access and any retained statistics or prototypes.
  A nominal replay capacity alone does not establish equal information access.
- Record which parameters are frozen, updated or newly created at each task,
  plus total and trainable parameter counts. Distinguish encoder updates from
  family-head updates and multiclass-output updates.
- Match the intended training exposure and state the sampling/loss differences.
  Report row presentations, optimizer steps, batch sizes, gradient accumulation,
  epochs, wall time and peak memory. Equal epochs are not equal compute,
  particularly when a method trains multiple binary heads.
- Use training-only calibration for tuning, with the same declared search
  budget where applicable. Freeze the primary outcome and decision rule before
  the comparison. Do not silently tune on official-test results.
- Choose resources from a measured pilot and obtain the required independent
  review and action-specific confirmation. This checklist authorizes no upload,
  environment change, resource request or job submission.

## Required output and interpretation

Retain all five paired seeds (1, 2, 3, 4, 42), checkpoint-level confusion
matrices and class supports, per-class precision/recall/F1 and explicitly
defined class forgetting, aggregate Macro-F1 and forgetting, overall and
average-task accuracy, balanced accuracy, attack recall and Benign FPR where
the labels support those definitions. Undefined quantities remain explicit,
not zero-filled. Malaya application labels do not define benign/attack metrics.

Declare the comparison family before testing. Report paired seed effects,
sample standard deviations, uncertainty intervals and the registered
multiple-testing treatment; do not treat classes or transitions as independent
seed replications. Preserve failed and safety-degrading arms, resource costs,
W&B records and protected checksums alongside favourable outcomes.

The fixed official test has already been inspected during earlier experiments.
Train-only checkpoint selection does not erase researcher-level selection
across those experiments. A newly designed follow-up must disclose that
history; five additional seeds on the same split do not make it a fresh,
independent test dataset. External-validity evidence, when available, must be
identified separately rather than implied by seed replication.

Acceptance requires a complete, hash-verified comparison and an interpretation
consistent with its scope and safety trade-offs. Passing a checklist, obtaining
a higher accuracy, or completing the separate attribution/ETG campaign is not
proof of architectural superiority or publication readiness.
