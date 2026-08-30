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

## Stage M2 result: D2 AdamW recipe screen

DICC Job `425182` compared the completed D2 Adam control with a candidate that
changed the optimizer recipe to AdamW, learning rate `5e-4` and weight decay
`1e-5`. The data, replay-50 budget, FT256x4 model, epoch budget, loss, router,
seed and `official/joint_cap3000` view were fixed.

The candidate increased final accuracy from 89.08% to 92.10%, increased
Macro-F1 from 55.70% to 59.52%, reduced forgetting from 4.36% to 2.15%, and
reduced Benign FPR from 10.46% to 6.23%. It also reduced average task accuracy
from 91.45% to 81.17%, balanced accuracy from 91.46% to 71.45%, and attack
recall from 96.18% to 87.08%. The candidate is therefore not selected.

Because optimizer family, learning rate and weight decay changed together,
this run does not isolate an AdamW main effect. The next registered diagnostic
keeps Adam and zero weight decay while changing only the learning rate to
`5e-4`. No optimizer arm is expanded to multiple seeds before this ambiguity
is resolved.

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

The replay-capacity and D2 optimizer-recipe screens are complete for seed 42.
The AdamW result package has deterministic result SHA-256
`67250b60ca5b9d2fb1362c9db2ec881c6c18c188bf5c3b7d505f32b5932e03b3`
and protected checksum-registry SHA-256
`92d273d30cf483cd39f2424f1213da3e675eed72ae7fa57fdfbfd84be48993dc`.
The matched learning-rate diagnostic, multi-seed confirmation and later tuning
stages remain incomplete.
