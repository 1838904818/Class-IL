# Exp 2 five-seed intervention-ground-truth result

## What ran

UNSW-NB15 replay, seeds 1/2/3/4/42, transition 0->1, N1 same-checkpoint controls, N2 function-preserving controls, strong P1 shortcuts (rho >= 0.9), and the screened P2 top-3 feature-mask construction. Each detector uses three explainers. P2 candidate selection and held-out evaluation use disjoint row halves, but both come from the same official dataset and do not constitute external validation.

## Main numbers

| Quantity | NCD | B2 repeat mean |
|---|---:|---:|
| Negative controls | 0/30 false positives (0.0%) | 0/30 (0.0%) |
| Eligible strong P1 | 5/5 (100.0%) | 5/5 (100.0%) |
| Eligible P2 | 4/4 (100.0%) | 4/4 (100.0%) |

Across the five seeds, 5 intervention-positive P1 cases were non-silent (recall drop exceeded 5 percentage points), so they are excluded from the silent-positive TPR denominator rather than counted as detector misses. P2 cases failing the joint intervention-positive-and-silent eligibility rule: 1. They were also excluded. Maximum N2 logit error was 0.

| Ablation | False positives | Strong P1 detected | P2 detected |
|---|---:|---:|---:|
| Full NCD | 0/30 | 5/5 | 4/4 |
| No probe bootstrap | 0/30 | 4/5 | 2/4 |
| No explainer repeats | 0/30 | 5/5 | 4/4 |
| No FDR | 0/30 | 5/5 | 4/4 |

## Decision

The point-estimate gate passes: observed NCD FPR is at most 5%, and its strong-P1 TPR is not below B2. However, NCD does not beat B1 or B2: all three are tied at every primary count above. All six continuous scores have pooled descriptive AUROC 1.000, so the ROC also provides no evidence that NCD ranks cases better. The 95% Wilson upper bound for an observed 0/30 FPR is 11.4%, so these data do not establish a population FPR below 5%. The pooled ROC is descriptive and not externally calibrated. Removing bootstrap reduces P1 detection from 5/5 to 4/5 and P2 from 4/4 to 2/4; removing repeats or FDR changes none of the pooled binary outcomes on this slice.

This is evidence that the benchmark constructions and silent-drift measurement work on this bounded slice. It is not evidence of substantive innovation, NCD superiority, broad method generality, or publication readiness.

All five raw result scope labels match their effective seed.

## Next

Complete the remaining frozen dataset/method cells required by the work order and add external threshold calibration. If B2 remains tied, the defensible conclusion is that repeat averaging is sufficient and NCD should not be claimed as the novel mechanism.
