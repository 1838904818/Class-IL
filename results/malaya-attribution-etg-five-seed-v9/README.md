# Malaya attribution and ETG five-seed result

This package records the completed post-hoc attribution-method robustness and
offline ETG analysis for the existing MalayaNetwork_GT FT512x12 checkpoints.
It does not contain raw traffic rows, checkpoint tensors, credentials, or
cluster-local paths.

## Registered scope

- Seeds: 1, 2, 3, 4, 42
- Prediction arm: `joint_cap3000`
- Score explained: winning-class margin
- Attribution methods: Expected Gradients, Feature Ablation, Gradient x Input
- Drift rule: top-15 Jaccard below 0.70 while class recall falls by no more
  than five percentage points
- Governance: offline and non-suppressing; ETG does not alter training,
  routing, replay, predictions, or semantic head creation
- Replication unit: stochastic seed on one fixed data split

The thresholds and top-15 rule are study-defined settings, not externally
validated universal constants.

## Main result

| Method | Silent-drift events | Eligible transitions | Mean seed rate | 95% t interval |
|---|---:|---:|---:|---:|
| Expected Gradients | 61 | 82 | 73.98% | [58.38%, 89.57%] |
| Feature Ablation | 3 | 82 | 3.60% | [0.00%, 10.19%] |
| Gradient x Input | 12 | 82 | 14.34% | [0.00%, 35.46%] |

Across all three methods, mean agreement is 50.0% for admission, 54.0% for
ETG state, and 35.0% for the silent-drift conclusion. The selected explainer
therefore changes the governance result materially. The defensible conclusion
is method-conditioned evidence, not method-independent certification.

## Reproducibility identifiers

- DICC job: `434747`
- W&B run: `attr5-2062659b92fd2e1d`
- W&B URL: <https://wandb.ai/csnet/ofra-etg-leon-hpc/runs/attr5-2062659b92fd2e1d>
- Submitted batch script SHA-256:
  `edea800c779c674b8cf19e362103c9110ba7a2bf8da4393d49bf6b82716dac1c`
- Protected aggregate manifest SHA-256:
  `a975b874c6cea79a63e4e644aa583192df236420727400c006e4fdc6d52cddc1`
- Protected aggregate result SHA-256:
  `ac86da2100bb7ad50ec66979e59e38996b1128116864fbba4859fb37a042782c`
- Canonical aggregate SHA-256:
  `2062659b92fd2e1d4b7ddcba13db44bcb819b2e37f0b402dec3962f783e441cb`
- Independent validator SHA-256:
  `84643f98c6b180654596d1cd887ddb19ccf45a1e7ef736f23c9817b6dabad3c4`

All five seed packages and the aggregate checksum registry passed read-only
verification. Independent semantic, array, and aggregate arithmetic checks
passed on the downloaded evidence.

## Interpretation limits

The underlying prediction result is not produced by this post-hoc job. The
analyzed checkpoints have five-seed means of 54.68% final accuracy, 21.15%
Macro-F1, 23.03% balanced accuracy, and 3.55 percentage points of average
forgetting. The analysis does not identify a uniquely correct explainer, prove
cross-device equivalence, measure analyst benefit, or validate an online ETG
feedback loop.
