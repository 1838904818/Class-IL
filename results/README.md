# Validated result index

This directory contains the completed, source-bound results available on 19
August 2026. Each dataset was trained and evaluated independently. Results
from different model sizes or training schedules remain protocol-separated.

## Four-seed descriptive aggregate

The completed seed set is `1, 2, 3, 4`. These numbers must not be described as a five-seed result.

| Dataset | Scoring arm | Accuracy | Macro-F1 | Balanced accuracy | Forgetting |
|---|---|---:|---:|---:|---:|
| MalayaNetwork_GT | Head-only | 55.87% +/- 0.33 | 11.81% +/- 0.93 | 15.16% +/- 0.58 | 1.25 +/- 1.19 pp |
| MalayaNetwork_GT | Router cap 3,000 | 40.98% +/- 10.67 | 18.56% +/- 1.51 | 20.68% +/- 1.92 | 8.02 +/- 1.47 pp |
| MalayaNetwork_GT | Joint cap 3,000 | 54.37% +/- 3.02 | 20.70% +/- 3.72 | 22.70% +/- 3.86 | 3.79 +/- 0.64 pp |
| MalayaNetwork_GT | Router full | 46.48% +/- 2.53 | 19.66% +/- 2.62 | 21.61% +/- 2.74 | 7.90 +/- 1.33 pp |
| MalayaNetwork_GT | Joint full | 56.14% +/- 3.00 | 21.04% +/- 3.85 | 22.89% +/- 3.92 | 3.23 +/- 0.88 pp |
| NSL-KDD | Head-only | 61.94% +/- 19.92 | 34.76% +/- 10.29 | 38.12% +/- 6.94 | 5.44 +/- 9.48 pp |
| NSL-KDD | Router cap 3,000 | 67.48% +/- 3.93 | 45.80% +/- 4.99 | 50.83% +/- 6.26 | 9.93 +/- 5.18 pp |
| NSL-KDD | Joint cap 3,000 | 69.07% +/- 3.38 | 38.81% +/- 3.04 | 40.87% +/- 2.96 | 2.38 +/- 1.34 pp |
| NSL-KDD | Router full | 66.72% +/- 3.65 | 44.55% +/- 5.23 | 49.89% +/- 6.04 | 8.92 +/- 4.36 pp |
| NSL-KDD | Joint full | 68.51% +/- 2.87 | 38.32% +/- 2.97 | 40.44% +/- 2.83 | 2.60 +/- 1.15 pp |

Values are mean +/- sample standard deviation across the four completed seeds. The full per-seed records and deterministic hashes are in `aggregate_4seed.json`.

## Explanation-governance result

The completed MalayaNetwork_GT seed-1 analysis is under `malaya-network-gt/etg-seed1/`. Its primary silent explanation-drift rate is 12/17 class-by-adjacent-checkpoint transitions (70.59%). This is not a packet, flow, sample, or real-world incident rate. ETG actions are simulated governance outcomes, not completed human reviews.

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
