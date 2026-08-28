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

The run order is replay 500 followed by replay 3,000 in one sequential Slurm
job. Each arm has a separate result directory, recovery directory, W&B run and
protected checksum registry. The two arms do not run concurrently.

## Metrics and selection

The primary metrics are final Macro-F1, balanced accuracy and attack recall.
Average forgetting is the primary continual-learning cost. Overall accuracy,
Benign false-positive rate, runtime, peak GPU memory and replay storage are
reported as secondary outcomes.

A candidate is retained only if it lies on a useful performance-cost frontier.
An accuracy improvement does not compensate automatically for a material loss
in minority-class metrics. A larger replay buffer must also justify its memory
and runtime cost.

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

The configurations and local preflight are complete. Replay 500 and replay
3,000 are not results until the exact DICC job has passed independent review,
completed, and produced verified protected outputs.
