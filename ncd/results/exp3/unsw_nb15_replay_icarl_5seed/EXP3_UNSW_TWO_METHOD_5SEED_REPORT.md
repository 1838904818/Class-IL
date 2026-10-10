# Exp 3 UNSW-NB15 two-method, five-seed report

## Ran

- Scope: UNSW-NB15; Replay and iCaRL; seeds 1/2/3/4/42; expected gradients, feature ablation, and gradient × input.
- Analysis code: private `27ea6ec5b2812f3acae584f67ddd17f9e90a3dd3`; identical supervisor-facing mirror `c8b5cc681ea0bea88d6e7c01a82a0ac4d7aed41c`.
- Frozen analysis: 1,000 permutations, BH FDR `q=0.05` within each run/explainer, ≥2/3 explainer consensus, Cauchy-combination control, fixed Jaccard@15 `<0.70` controls, and primary silent condition `Δrecall > -0.05`.
- Validation: 10 runs, 600 explainer-level tests, 200 class-transition units, and 1,920 sensitivity cells. Every unit has exactly three explainers; all p/q values are within `[0,1]`; no NaN or infinity was found.

## Numbers

| Method | Silent units | B1 fixed drift | B2 repeat-mean drift | NCD drift | B1 drift rate, mean ± SD | B2 rate | NCD rate | B1 agreement | B2 agreement | NCD agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Replay | 60 | 19/60 | 18/60 | 59/60 | 31.0% ± 19.0% | 29.3% ± 16.2% | 98.3% ± 3.7% | 44.5% | 44.2% | 98.3% |
| iCaRL | 53 | 10/53 | 10/53 | 52/53 | 19.4% ± 15.2% | 19.4% ± 15.2% | 98.3% ± 3.7% | 50.7% | 56.6% | 100.0% |

Among silent units with a following checkpoint, Replay has 28 NCD-positive and one NCD-negative observation; their median future recall changes are +1.91 pp and +1.52 pp. iCaRL has 23 NCD-positive and one NCD-negative observation; their medians are -2.25 pp and -4.18 pp. A single negative comparator per method is insufficient to test prediction or establish separation.

| Method | Training total | Explanation total | Fixed-rule scoring total | NCD permutation total |
|---|---:|---:|---:|---:|
| Replay | 82.52 s | 2,900.66 s | 28.93 ms | 105.87 s |
| iCaRL | 107.71 s | 3,178.60 s | 29.94 ms | 106.86 s |

## Decision

- **This UNSW-NB15 two-method, five-seed subset is ready within its descriptive scope.** The four new seeds used deterministic CUDA training and explanation settings, and all eight new run logs are retained.
- **The full Exp 3 requirement is not met.** Combined with the completed NSL-KDD tranche, current coverage is 2/4 datasets and 2/8 requested methods.
- **NCD does not establish practical silent-drift discrimination here.** Its pooled 111/113 positive rate across the two methods is near-saturated and leaves only one downstream NCD-negative comparator per method.
- **No substantive innovation, causal prediction, or superiority is established.** Exp 2 did not beat B2, and the present analysis is descriptive rather than a validated early-warning result.

## Next

Complete the remaining Phase I methods on NSL-KDD and UNSW-NB15 before adding CICIDS2017 and Edge-IIoTset. Keep the same five seeds, three explainers, fixed controls, 1,000 permutations, and complete logs; do not claim innovation unless a non-saturated mechanism beats the matched controls on its stated endpoint.
