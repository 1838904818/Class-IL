# Balanced replay replication protocol

Status (7 September 2026): all five training seeds are complete on both datasets. The six seed-2–4 jobs below completed with exit code 0, in 14m01s–14m45s each, excluding queue time. Protected result/protocol checksums, fixed configuration/runtime/data bindings and all checkpoint confusion/task metrics passed local verification. W&B URLs are recorded, but independent cloud verification remains pending. See the [five-seed results and four-new-seed sensitivity analysis](BALANCED_REPLAY_FIVE_SEED_RESULTS.md). The [seed-1 report](BALANCED_REPLAY_SEED1_RESULTS.md) is retained as a historical checkpoint.

| Training seed | Malaya job | ReplayIDS D2 job |
|---|---:|---:|
| 2 | 451841 | 451842 |
| 3 | 451843 | 451844 |
| 4 | 451845 | 451846 |

Each job requests one A100, 16 GiB host memory, one node/task and a 45-minute short-QoS limit. Malaya uses four CPUs; ReplayIDS D2 uses two. The measured seed-1 elapsed times were 14m49s and 14m03s, excluding queue time. Those measurements are not a scheduling guarantee. Job submission is not evidence that training or W&B synchronization has begun.

## Purpose and scope

The seed-42 diagnostic screened balanced replay against frozen nearest-mean exemplars. Balanced replay had higher final accuracy and Macro-F1 on both MalayaNetwork_GT and ReplayIDS D2 and was retained for replication. This is a custom conventional comparator, not a tuned state-of-the-art baseline.

The completed training seeds are 1, 2, 3, 4 and 42. Seed 42 informed comparator selection; reporting therefore includes a separate sensitivity summary for seeds 1–4. Training and data contracts were unchanged across the replication; no new-seed test outcome selected hyperparameters.

## Fixed training and data contracts

| Item | MalayaNetwork_GT | ReplayIDS D2 |
|---|---|---|
| Backbone | FT-Transformer, width 512, depth 12 | FT-Transformer, width 256, depth 4 |
| Training | Task 0: 8 epochs; later tasks: 10 epochs each | Same |
| Optimizer | Adam, learning rate 0.001, weight decay 0 | Same |
| Replay memory | 50 retained examples per observed class | Same |
| Later-task sampling | Every new row paired with one old-memory draw | Same |
| Tasks / classes | 5 / 10 | 4 / 8 |
| Features / final test rows | 77 / 10,370 | 78 / 227,723 |

Data preprocessing, task order, hyperparameters and implementation remain unchanged from the seed-42 diagnostic. Normalization uses Task-0 training rows. Official test rows are not resampled. ReplayIDS D2 keeps the existing seed-42-derived data partition; changing the training seed does not change the split.

A balanced-replay later epoch contains equal numbers of new-row and replay presentations, so it has twice the row exposure of an epoch visiting only new rows. Equal epoch counts do not mean equal training compute. OFRA also retains router centroids; an equal exemplar count is not a claim of equal total memory.

## Evaluation and reporting

Checkpoint-policy scope (clarified 7 September 2026): the original hash-bound ReplayIDS comparator was Job 426307 guarded selection. The registered primary remains Job 425539 last epoch; that comparison is added retrospectively in the [primary-policy diagnostic](REPLAYIDS_PRIMARY_SCORING_DIAGNOSTIC.md). The same `official/joint_cap3000` arm name does not make these checkpoint policies interchangeable. No original run, binding or result is replaced.

Record every task checkpoint's confusion matrix, per-class precision/recall/F1/support, overall accuracy, Macro-F1, balanced accuracy and forgetting. Record attack recall and benign false-positive rate for ReplayIDS D2. Malaya's application labels do not define a benign/attack partition.

Pair each result with the same training seed of OFRA's joint_cap3000 arm under the corresponding fixed data contract. Preserve the distinction between current diagnostic evidence and five-seed inference; report effect sizes and uncertainty, not a publication-readiness claim.

W&B receives validated aggregate metrics and tables only. Raw feature rows, identifiers and model checkpoints are not uploaded.

## Integrity gates

Verify the configuration, runtime and data bindings before execution. Verify result/protocol checksums, all expected checkpoints and final confusion-matrix row totals before admitting numerical results. Independently verify the W&B record before claiming cloud synchronization is complete. A successful scheduler exit alone is insufficient. For seed 1, local numerical verification is complete but independent W&B cloud verification remains pending.

The seed-1 paired OFRA result SHA-256 values are:

- Malaya: fd532519409bd09f97074805aef924fdcdf4dc68a2038c62125934d8da525026
- ReplayIDS D2: 4676c8f2ac98b73fccc90f8b86630053e93cb0403643e83032b37619db017310

This protocol records replication progress; it does not revise manuscript headline results.
