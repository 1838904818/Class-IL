# Exp 2 five-seed intervention-ground-truth result

## What ran

NSL-KDD icarl, seeds 1/2/3/4/42, transition 0->1, N1 same-checkpoint controls, N2 function-preserving controls, strong P1 shortcuts (rho >= 0.9), and no P2 detector case because the frozen P2 screen produced no eligible construction. Each detector uses three explainers. Frozen P2 screen status: unavailable_after_frozen_screen; screen evidence: ncd/results/exp2_v2/p2_screen/nsl-kdd/icarl/seed1_dos/p2_screen_result.json.

## Main numbers

| Quantity | NCD | B2 repeat mean |
|---|---:|---:|
| Negative controls | 0/20 false positives (0.0%) | 0/20 (0.0%) |
| Eligible strong P1 | 9/9 (100.0%) | 9/9 (100.0%) |
| Eligible P2 | n/a | n/a |

Across the five seeds, 0 intervention-positive P1 cases were non-silent (recall drop exceeded 5 percentage points), so they are excluded from the silent-positive TPR denominator rather than counted as detector misses. P2 has no detector denominator because the frozen screen found no eligible construction. Maximum N2 logit error was 0.

| Ablation | False positives | Strong P1 detected | P2 detected |
|---|---:|---:|---:|
| Full NCD | 0/20 | 9/9 | n/a |
| No probe bootstrap | 0/20 | 4/9 | n/a |
| No explainer repeats | 0/20 | 9/9 | n/a |
| No FDR | 0/20 | 9/9 | n/a |

## Decision

The frozen point-estimate gate passes: NCD FPR is 0.0%, strong-P1 detection is 9/9, B2 is 9/9, and B1 is 9/9. All six continuous scores are reported as pooled descriptive AUROC values, not external validation. The 95% Wilson upper bound for the observed 0/20 FPR is 16.1%, so these data do not by themselves establish a population FPR below 5%. The pooled ROC is descriptive and not externally calibrated. Removing bootstrap changes P1 detection from 9/9 to 4/9; the remaining ablation counts are reported above.

This is bounded evidence that the available benchmark constructions and silent-drift measurement operate on this slice. It is not evidence of substantive innovation, NCD superiority, broad method generality, or publication readiness.

All five raw result scope labels match their effective seed.

## Next

Complete the remaining frozen dataset/method cells required by the work order and add external threshold calibration. If B2 remains tied, the defensible conclusion is that repeat averaging is sufficient and NCD should not be claimed as the novel mechanism.
