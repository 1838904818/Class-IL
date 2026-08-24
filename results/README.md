# Validated result index

This directory contains the completed, source-bound results available on 24
August 2026. Each dataset was trained and evaluated independently. Results
from different model sizes or training schedules remain protocol-separated.

## Current strict five-seed prediction evidence

The completed seed set for the table below is `1, 2, 3, 4, 42`. NSL-KDD,
UNSW-NB15, and CIC-IDS-2017 use FT256x4; MalayaNetwork_GT uses FT512x12.
The table is not a capacity-matched pooled comparison.

| Dataset | Model | Scoring arm | Accuracy | Macro-F1 | Balanced accuracy | Forgetting |
|---|---|---|---:|---:|---:|---:|
| NSL-KDD | FT256x4 | Head only | 56.29% | 31.08% | 36.48% | 8.14 pp |
| NSL-KDD | FT256x4 | Joint cap 3,000 | 71.56% | 41.46% | 42.25% | 1.60 pp |
| UNSW-NB15 | FT256x4 | Head only | 68.78% | 21.16% | 23.52% | 2.68 pp |
| UNSW-NB15 | FT256x4 | Joint cap 3,000 | 61.95% | 23.50% | 28.45% | 8.97 pp |
| CIC-IDS-2017 | FT256x4 | Head only | 64.59% | 21.25% | 25.91% | 9.04 pp |
| CIC-IDS-2017 | FT256x4 | Joint cap 3,000 | 72.57% | 36.95% | 63.13% | 9.63 pp |
| MalayaNetwork_GT | FT512x12 | Head only | 56.03% | 11.77% | 15.05% | 3.16 pp |
| MalayaNetwork_GT | FT512x12 | Joint cap 3,000 | 54.68% | 21.15% | 23.03% | 3.55 pp |

The reproducible derived package is under `formal-five-seed-20260824/`.
It includes source hashes, per-seed metrics, paired differences, per-class
results, pooled confusion matrices, threshold sensitivity, and presentation
figures. The older four-seed aggregate is retained only as a historical
snapshot and is not the current headline evidence.

## Explanation-governance result

The completed MalayaNetwork_GT seed-1 analysis is under `malaya-network-gt/etg-seed1/`. Its primary silent explanation-drift rate is 12/17 class-by-adjacent-checkpoint transitions (70.59%). This is not a packet, flow, sample, or real-world incident rate. ETG actions are simulated governance outcomes, not completed human reviews.

The registered rule uses top-15 Expected-Gradients features, a Jaccard threshold
of 0.7, and an allowed class-recall drop of 5 percentage points. The sensitivity
grid under `formal-five-seed-20260824/` varies top-k, Jaccard threshold, and
allowed recall drop. The third dimension is class recall, not overall accuracy.

The source-bound attribution robustness pilot is under
`malaya-network-gt/attribution-robustness-seed1/`. Its primary comparison uses
Expected Gradients, feature ablation, and Gradient x Input on the same frozen
seed-1 evidence. Integrated Gradients failed their completeness diagnostic and
are excluded from the primary agreement statistics.

## CSE-CIC-IDS2018 status

`cic-ids-2018/ft256x4-1plus1-five-seed/summary.json` records a completed
FT256x4 closure campaign over seeds `1`, `2`, `3`, `4`, and `42`. Its one
Task-0 epoch plus one epoch per later task schedule differs from the main
FT512x12 8/10-epoch protocol and must not be pooled with it. The joint cap-3000
view reports 52.55% +/- 5.79 accuracy, 34.05% +/- 7.54 Macro-F1, 53.93% +/-
7.79 balanced accuracy, and 19.03 +/- 8.62 points of forgetting. The existing
`capacity_profile.json` remains resource-planning evidence only.
