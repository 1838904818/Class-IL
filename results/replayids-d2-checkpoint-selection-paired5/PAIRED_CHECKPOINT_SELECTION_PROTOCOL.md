# ReplayIDS D2 paired five-seed checkpoint-selection protocol

## Research question

Does training-only checkpoint selection improve the accuracy-retention trade-off
relative to retaining epoch 10 when all data, model, optimizer, replay, router,
training budget and stochastic seeds are paired?

## Paired design

- Formal seeds: `1, 2, 3, 4, 42`.
- Baseline arm: `family_checkpoint_selection=last`.
- Candidate arm: `family_checkpoint_selection=training_only_calibration_macro_f1`.
- The two arms run sequentially in one reviewed Slurm job on one GPU.
- Both arms use the same D2 fit and official-test files. Only the checkpoint-
  selection arm consumes the single bound training-calibration audit.
- Every seed-arm run has hash-bound task/epoch recovery and a dedicated W&B run.
- The paired report rejects any protocol difference beyond checkpoint selection.

## No-look-ahead contract

- Calibration consists only of the 68,313 source-training rows held out by
  derivation Job 414907; fit and calibration membership are disjoint.
- Official test rows remain byte-identical and are never used for selection.
- At task t, only classes already introduced through task t may enter the
  calibration pool.
- Positive and negative calibration labels are balanced, capped at 5,000 rows
  per label, with a minimum of 32 rows per label.
- All ten family-head epochs are trained. The candidate retains the earliest
  epoch with maximum binary Macro-F1; low-support labels retain epoch 10.
- The baseline reads the identical fit manifest but receives no calibration
  audit argument and retains epoch 10.
- The baseline runtime protocol must record `training_calibration` as exactly
  `{"enabled": false}`. The calibrated arm must record `enabled=true`,
  `official_test_used=false`, the registered audit SHA-256 and the policy that
  only classes seen through the current task may enter selection. This
  arm-specific protocol shape is part of the registered checkpoint-selection
  treatment rather than an additional training-data change.

## Frozen controls

- Dataset: ReplayIDS CIC-IDS-2017 D2 adaptive normal-only cap.
- Fit/calibration/official-test rows: `268,697 / 68,313 / 227,723`.
- Tasks: `[[0,1],[2,3],[4,5],[6,7]]`.
- Model: FT-Transformer 256x4, 8 heads, dimension per head 32.
- Training: Task-0 8 epochs; later family heads 10 epochs.
- Optimizer: Adam, learning rate `1e-3`, weight decay `0`.
- Loss: focal loss, gamma `2.0`, alpha `0.75`.
- Replay: 50 exemplars per class.
- Router: joint score with cap 3,000 is the primary reporting arm.

## Outcomes and inference

The primary route is `official/joint_cap3000`. The two designated inferential
outcomes are final Macro-F1 and average forgetting. Their paired t-test p-values
are Holm-adjusted across these two outcomes. Average task accuracy, final
accuracy, balanced accuracy, attack recall and Benign FPR are descriptive
secondary outcomes. The report includes paired deltas, sample standard
deviations, 95% paired t intervals, Wilcoxon results and Cohen dz. With n=5,
distributional diagnostics and exact Wilcoxon power are limited.

All reported rate/score fields except forgetting are validated in `[0,1]`.
Average forgetting is validated in `[-1,1]`: a negative value is mathematically
valid under the registered `max prior accuracy - final accuracy` definition and
indicates backward improvement rather than a corrupted rate. SciPy `1.15.3` is
an exact allocation-time dependency because it implements the paired t-test and
Wilcoxon calculations.

## Pilot evidence and remaining limitation

DICC Job 425441 completed the candidate arm for seed 42 and improved the primary
route numerically relative to Job 414989. That result is a pilot, not the
five-seed conclusion. Heartbleed has only one training-calibration row, so its
registered low-support fallback remains epoch 10.

## Execution boundary

The campaign uses one A100, one node, one task, two CPUs, 13 GiB and normal QoS
for at most three hours. It does not use arrays, fan-out, distributed training,
servers, tunnels or login-node computation. Scratch stores active outputs; the
completed result, per-seed protocols, monitoring evidence, W&B run registry and
checksum manifest are copied atomically to protected HOME storage. Submission
requires current live evidence, exact-hash independent approval and one fresh
confirmation in the current conversation.
