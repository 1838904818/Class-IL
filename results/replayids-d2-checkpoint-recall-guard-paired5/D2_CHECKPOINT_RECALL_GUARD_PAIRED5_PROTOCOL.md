# ReplayIDS D2 guarded checkpoint five-seed confirmation

## Purpose

Confirm whether the seed-42 improvement from DICC Job 425982 generalises across seeds 1, 2, 3, 4, and 42 under the same official test protocol. This is a confirmation experiment, not a new hyperparameter search.

## Fixed comparison

- Baseline: immutable `last_epoch` results from DICC Job 425539.
- Candidate: `training_only_calibration_macro_f1_recall_fpr_guard`.
- Dataset: ReplayIDS D2 adaptive normal-to-largest-attack training contract.
- Official test shards remain byte-identical, full, and excluded from checkpoint selection.
- Model and training budget: FT-Transformer 256x4, pretrain 8 epochs, 10 epochs per incremental task, Adam 1e-3, Replay50, router cap 3000.
- Seeds: 1, 2, 3, 4, 42.

## Guard rule

The candidate epoch must improve training-only calibration Macro-F1 by at least 0.01 while allowing at most 0.01 positive-class recall loss and at most 0.01 negative-class false-positive-rate increase relative to epoch 10. These thresholds are project-defined operational constraints, not an external standard. A failed guard retains epoch 10.

## Primary outcomes

The registered primary paired outcomes are official/joint_cap3000 Macro-F1 and forgetting. Accuracy, balanced accuracy, attack recall, and Benign FPR are required descriptive safety outcomes. The report includes paired seed values, mean, sample standard deviation, 95% confidence intervals, paired t-tests, Wilcoxon tests, effect sizes, and Holm adjustment for the two primary outcomes.

## Claim boundary

No five-seed claim is permitted unless all five candidate results, the immutable baseline registry, W&B run records, monitoring artifacts, report, and protected checksum registry validate. A statistically weak or safety-degrading result is retained as a negative finding and does not justify selecting this policy as the final method.

## Compute and governance

One A100, one task, two CPUs, 13 GiB RAM, and two hours normal QoS are requested from measured prior runs. Seeds execute sequentially with hash-bound recovery. No job array, fan-out, login-node computation, server, tunnel, credential embedding, or raw-row W&B logging is used. DICC computational resources must be acknowledged in any publication using these results.
