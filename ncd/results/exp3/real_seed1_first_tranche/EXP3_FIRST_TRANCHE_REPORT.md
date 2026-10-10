# Exp 3 real-data first tranche

## Ran

- Scope: NSL-KDD and UNSW-NB15; Replay and iCaRL; seed 1; expected gradients, feature ablation, and gradient × input.
- Analysis code: private `638673d6e8bbafd23d9fedf390edf821be99865f`; identical supervisor-facing mirror `144d856`.
- Command: `python -I -B ncd/analyze_exp3.py` with the four recorded Exp 0 explanation receipts, 1,000 permutations, BH FDR `q=0.05`, and the primary silent condition `Δrecall > -0.05`.
- Audited output: 174 explainer-level tests, 58 class-transition units, and 768 fixed-threshold sensitivity cells. Every unit has exactly three explainers; p/q values are within `[0,1]`; no NaN or infinity was found.

## Numbers

| Dataset | Method | Silent units | B1 fixed drift | B2 repeat-mean drift | NCD ≥2/3 drift | B1 all-explainer agreement | B2 agreement | NCD agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NSL-KDD | Replay | 8 | 5/8 (62.5%) | 4/8 (50.0%) | 8/8 (100%) | 50.0% | 50.0% | 100% |
| NSL-KDD | iCaRL | 8 | 2/8 (25.0%) | 2/8 (25.0%) | 8/8 (100%) | 75.0% | 87.5% | 100% |
| UNSW-NB15 | Replay | 11 | 1/11 (9.1%) | 1/11 (9.1%) | 11/11 (100%) | 63.6% | 63.6% | 100% |
| UNSW-NB15 | iCaRL | 10 | 1/10 (10.0%) | 1/10 (10.0%) | 10/10 (100%) | 70.0% | 80.0% | 100% |

All 174 raw permutation p-values are at most `0.014985`; 167 are at the Monte Carlo floor `1/1001`. BH-adjusted per-explainer decisions and BH-adjusted Cauchy-combination decisions both mark every primary-silent unit. The fixed-rule sensitivity grid spans 0–100% depending on `k`, Jaccard threshold, and explainer.

| Dataset/method | Explanation time | Fixed-rule scoring | NCD permutation time | Median next recall Δ after NCD drift |
|---|---:|---:|---:|---:|
| NSL-KDD / Replay | 276.74 s | 2.35 ms | 9.49 s | +2.41 pp |
| NSL-KDD / iCaRL | 306.49 s | 2.35 ms | 9.51 s | -0.93 pp |
| UNSW-NB15 / Replay | 630.49 s | 5.86 ms | 21.03 s | +5.28 pp |
| UNSW-NB15 / iCaRL | 665.97 s | 5.78 ms | 20.94 s | -2.39 pp |

## Decision

- **First-tranche measurement is ready within its reviewed descriptive scope.** The artifacts are internally consistent and bound to the committed analysis code.
- **Full Exp 3 is not complete.** This is 2 datasets × 2 methods × 1 seed, not the required 4 datasets × 8 methods × 5 seeds.
- **No forgetting-prediction conclusion is available.** NCD marked every eligible unit, so there is no non-drift comparison group; the next-recall values above are descriptive only.
- **No substantive innovation or NCD superiority is established.** Exp 2 did not beat B2, and this tranche's universal significance may reflect sensitivity to very small checkpoint changes rather than practical drift magnitude.

## Next

Run the remaining seeds for NSL-KDD Replay/iCaRL first, then extend by dataset and Phase I method. Keep the same frozen three explainers, 1,000-permutation test, within-run FDR, fixed-rule controls, full logs, and cost fields.
