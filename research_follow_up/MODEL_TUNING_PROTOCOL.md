# OFRA model-adjustment protocol

## Objective

The first model-adjustment stage tests whether OFRA's limited replay memory,
rather than FT-Transformer size alone, contributes to low minority-class
retention. It is a controlled diagnostic, not an unrestricted hyperparameter
search.

## Stage M1: replay-capacity screen

The immutable seed-42 ReplayIDS control stores at most 50 exemplars per class.
Two candidates increase that capacity to 500 and 3,000. All other registered
settings are held fixed:

- uncapped expected-contract training data and the full official test split;
- FT-Transformer with width 256 and four layers;
- eight Task-0 pretraining epochs and ten epochs per later task;
- Adam, learning rate 0.001, focal loss and the same batch settings;
- the same DP-Means router and joint-score evaluation views;
- seed 42, deterministic execution and verified input shards.

The run order was replay 500 followed by replay 3,000 in one sequential Slurm
job. Each arm had a separate result directory, recovery directory, W&B run and
protected checksum registry. The two arms did not run concurrently.

## Metrics and selection

The primary metrics are final Macro-F1, balanced accuracy and attack recall.
Average forgetting is the primary continual-learning cost. Overall accuracy,
Benign false-positive rate, runtime, peak GPU memory and replay storage are
reported as secondary outcomes.

A candidate is retained only if it lies on a useful performance-cost frontier.
An accuracy improvement does not compensate automatically for a material loss
in minority-class metrics. A larger replay buffer must also justify its memory
and runtime cost.

## Stage M1 result

DICC Job `414908` completed both candidates for seed 42. Under the registered
official `joint_cap3000` scoring view, replay 500 and replay 3,000 raised final
overall accuracy and reduced Benign false-positive rate relative to replay 50.
Both candidates, however, reduced average task accuracy, balanced accuracy and
attack recall. Replay 3,000 also reduced Macro-F1 by 8.25 percentage points.

Replay 50 therefore remains the reference for the next controlled stage. This
is a diagnostic decision from one seed, not evidence that replay 50 is a
universally optimal memory size. `joint_cap3000` refers to the separate router
sample cap and must not be confused with exemplar capacity 3,000.

## Later stages

Only after M1 is measured will the study change optimiser, learning schedule,
loss weighting or architecture. The intended order is:

1. replay capacity;
2. Benign anchoring and class-aware sampling;
3. loss and optimiser schedule;
4. a small, matched architecture comparison.

Each stage freezes the selected preceding protocol. This prevents a result
from being attributed to several simultaneous changes. Only a frozen selected
configuration is expanded to seeds `{1,2,3,4,42}`.

## Evidence boundary

The configurations, review, execution and protected-output verification are
complete for seed 42. The protected checksum registry has SHA-256
`a088ce4d77daea4ec597f9da0c39059283627bab6a8a70974447909cb9b171f3`.
Multi-seed confirmation and later tuning stages remain incomplete.
