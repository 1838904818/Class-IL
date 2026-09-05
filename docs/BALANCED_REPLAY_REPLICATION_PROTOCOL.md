# Balanced replay replication protocol

Status (6 September 2026): the seed-1 candidates are prepared; no seed-1 result is claimed here.

## Purpose and scope

The seed-42 diagnostic screened balanced replay against frozen nearest-mean exemplars. Balanced replay had higher final accuracy and Macro-F1 on both MalayaNetwork_GT and ReplayIDS D2 and was retained for replication. This is a custom conventional comparator, not a tuned state-of-the-art baseline.

The planned training seeds are 1, 2, 3, 4 and 42. The next stage runs seed 1 separately on each dataset. Seeds 2–4 remain pending. Seed 42 informed comparator selection; final reporting must disclose that selection and provide a sensitivity summary for the four new seeds separately. No new-seed test outcome will select hyperparameters.

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

Record every task checkpoint's confusion matrix, per-class precision/recall/F1/support, overall accuracy, Macro-F1, balanced accuracy and forgetting. Record attack recall and benign false-positive rate for ReplayIDS D2. Malaya's application labels do not define a benign/attack partition.

Pair each result with the same training seed of OFRA's joint_cap3000 arm under the corresponding fixed data contract. Preserve the distinction between current diagnostic evidence and five-seed inference; report effect sizes and uncertainty, not a publication-readiness claim.

W&B receives validated aggregate metrics and tables only. Raw feature rows, identifiers and model checkpoints are not uploaded.

## Integrity gates

Verify the configuration, runtime and data bindings before execution. Verify result/protocol checksums, all expected checkpoints, final confusion-matrix row totals and the W&B record before admitting a run into the evidence package. A successful scheduler exit alone is insufficient.

The seed-1 paired OFRA result SHA-256 values are:

- Malaya: fd532519409bd09f97074805aef924fdcdf4dc68a2038c62125934d8da525026
- ReplayIDS D2: 4676c8f2ac98b73fccc90f8b86630053e93cb0403643e83032b37619db017310

This protocol adds a replication plan; it does not revise manuscript headline results.
