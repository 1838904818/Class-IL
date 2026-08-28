# Adaptive training-data protocol v2

## Motivation

The seed-42 fixed-cap pilot raised final accuracy and reduced forgetting, but it
also reduced Macro-F1 and balanced accuracy. The fixed 50,000-row rule removed
training rows from both Benign and DoS Hulk. It therefore mixed two effects:
reducing normal-traffic dominance and discarding attack evidence.

Protocol v2 isolates those effects. It caps only the declared normal class and
preserves every attack row available after the training-only calibration split.

## Registered contract

1. Split every class deterministically into 90% fit and 10% calibration using
   only the source training partition.
2. Let `C` be the largest attack-class fit-pool size.
3. Retain `min(normal_fit_pool, C)` normal-class rows without replacement.
4. Retain every attack-class fit row. Do not oversample or synthesise rows.
5. Preserve every official test shard byte-for-byte. The test partition is not
   used for sampling, normalisation, calibration or model selection.
6. Fit feature normalisation only from Task-0 fit rows and freeze it before
   training begins.

For the current ReplayIDS expected-contract source, `C` is 124,780 (DoS Hulk).
The resulting seed-42 candidate contains 268,697 fit rows, 68,313 disjoint
training-only calibration rows and all 227,723 official test rows. Only Benign
is capped; no attack class is reduced.

## Comparison and decision rule

The first run is a seed-42 screen against both immutable references:

- A0: uncapped training, replay 50;
- A1: fixed 50,000-row cap, replay 50;
- D2: adaptive normal-only cap, replay 50.

The primary decision metrics are Macro-F1, balanced accuracy and attack recall.
Accuracy, forgetting, benign false-positive rate, runtime and memory are also
reported. D2 is not selected merely because accuracy rises. It must recover a
material part of A0's minority-class performance while retaining useful A1
gains in forgetting or benign false-positive rate.

Only a selected frozen protocol is expanded to seeds `{1,2,3,4,42}`. The
seed-42 screen is diagnostic and cannot support a publication-level claim.

## Scope

The policy applies to intrusion-detection datasets with an explicit
`normal_class_id`. A dataset without a normal/attack distinction must use its
own registered contract; this script does not invent a normal class or pool
unrelated datasets.
