# Current formal result audit (24 August 2026)

## Malaya FT512x12 five-seed result

| Arm | Accuracy | Macro-F1 | Balanced accuracy | Forgetting |
|---|---:|---:|---:|---:|
| Head only | 56.03% +/- 0.45% | 11.77% +/- 0.81% | 15.05% +/- 0.55% | 3.16 +/- 4.39 pp |
| Joint cap 3,000 | 54.68% +/- 2.71% | 21.15% +/- 3.37% | 23.03% +/- 3.42% | 3.55 +/- 0.77 pp |

Paired joint-minus-head mean differences:

- Accuracy: -1.34 pp (95% CI -4.71 to +2.02).
- Macro-F1: +9.37 pp (95% CI +4.70 to +14.04).
- Balanced accuracy: +7.97 pp (95% CI +3.72 to +12.23).
- Forgetting: +0.39 pp (95% CI -5.75 to +6.54; negative favours joint).

Malaya is application-traffic classification, so attack recall and benign false-positive rate are not reported as NIDS metrics.

## Current cross-dataset boundary

NSL-KDD, UNSW-NB15, and CIC-IDS-2017 use FT256x4. Malaya uses FT512x12. The table is a dataset-by-dataset audit and is not a capacity-matched pooled superiority test.

| Dataset | Model | Arm | Accuracy | Macro-F1 | Balanced accuracy | Forgetting |
|---|---|---|---:|---:|---:|---:|
| NSL-KDD | FT256x4 | head_only | 56.29% | 31.08% | 36.48% | 8.14 pp |
| NSL-KDD | FT256x4 | joint_cap3000 | 71.56% | 41.46% | 42.25% | 1.60 pp |
| UNSW-NB15 | FT256x4 | head_only | 68.78% | 21.16% | 23.52% | 2.68 pp |
| UNSW-NB15 | FT256x4 | joint_cap3000 | 61.95% | 23.50% | 28.45% | 8.97 pp |
| CIC-IDS-2017 | FT256x4 | head_only | 64.59% | 21.25% | 25.91% | 9.04 pp |
| CIC-IDS-2017 | FT256x4 | joint_cap3000 | 72.57% | 36.95% | 63.13% | 9.63 pp |
| MalayaNetwork_GT | FT512x12 | head_only | 56.03% | 11.77% | 15.05% | 3.16 pp |
| MalayaNetwork_GT | FT512x12 | joint_cap3000 | 54.68% | 21.15% | 23.03% | 3.55 pp |

## Open evidence

- CSE-CIC-IDS2018 Job 402073 is still running; it is excluded from the current four-dataset aggregate.
- Malaya multi-seed Expected-Gradients/ETG analysis is not yet complete.
- Strong external continual-learning baselines are not protocol-matched in this table.
