# Exp 3 NSL-KDD two-method, five-seed report

## Ran

- Scope: NSL-KDD; Replay and iCaRL; seeds 1/2/3/4/42; expected gradients, feature ablation, and gradient × input.
- Analysis code: private `0e4a3c969ce1de2d8282872d00edbab4c1b62530`; identical supervisor-facing mirror `1191c0d`.
- Frozen analysis: 1,000 permutations, BH FDR `q=0.05` within each run/explainer, ≥2/3 explainer consensus, Cauchy-combination control, fixed Jaccard@15 `<0.70` controls, and primary silent condition `Δrecall > -0.05`.
- Validation: 10 runs, 270 explainer-level tests, 90 class-transition units, and 1,920 sensitivity cells. Every unit has exactly three explainers; all p/q values are within `[0,1]`; no NaN or infinity was found.

## Numbers

| Method | Silent units | B1 fixed drift | B2 repeat-mean drift | NCD drift | B1 drift rate, mean ± SD | B2 rate | NCD rate | B1 agreement | B2 agreement | NCD agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Replay | 27 | 17/27 | 16/27 | 27/27 | 63.5% ± 6.5% | 61.0% ± 8.9% | 100.0% ± 0.0% | 55.0% | 51.0% | 100.0% |
| iCaRL | 33 | 10/33 | 10/33 | 32/33 | 30.7% ± 3.8% | 30.7% ± 3.8% | 96.7% ± 7.5% | 78.8% | 81.3% | 93.3% |

The one NCD-negative silent unit has no following checkpoint, so it cannot form the required future-recall comparison. Among units with a following checkpoint there are 28 NCD-positive and zero NCD-negative observations. Therefore this subset cannot test whether drift precedes later recall loss.

| Method | Training total | Explanation total | Fixed-rule scoring total | NCD permutation total |
|---|---:|---:|---:|---:|
| Replay | 58.19 s | 1,315.28 s | 11.58 ms | 46.99 s |
| iCaRL | 63.82 s | 1,428.55 s | 11.87 ms | 47.06 s |

## Decision

- **This NSL-KDD two-method, five-seed subset is ready within its reviewed descriptive scope.** The four new seeds used deterministic CUDA training and explanation settings; interrupted pre-fix attempts remain separately logged and are excluded.
- **The full Exp 3 requirement is not met.** This covers 1/4 datasets and 2/8 requested methods.
- **NCD does not establish practical silent-drift discrimination here.** Its 59/60 pooled positive rate increases agreement mainly by near-saturation and leaves no usable downstream non-drift comparison group.
- **No substantive innovation or superiority is established.** Exp 2 did not beat B2, and these real-data measurements do not repair that failed gate.

## Next

Run the same five-seed Replay/iCaRL tranche on UNSW-NB15, then add the remaining Phase I methods and OFRA by dataset. Preserve the deterministic CUDA fields, three explainers, 1,000 permutations, complete logs, and the same fixed-rule controls.
