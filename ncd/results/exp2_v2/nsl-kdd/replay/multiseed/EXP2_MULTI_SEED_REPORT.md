# Exp 2 five-seed intervention-ground-truth result

## What ran

NSL-KDD Replay, seeds 1/2/3/4/42, transition 0->1, N1 same-checkpoint controls, N2 function-preserving controls, strong P1 shortcuts (rho >= 0.9), and the screened DoS P2 top-3 feature-mask construction. Each detector uses three explainers. P2 feature selection and evaluation use the same official dataset and are not independent.

## Main numbers

| Quantity | NCD | B2 repeat mean |
|---|---:|---:|
| Negative controls | 0/34 false positives (0.0%) | 0/34 (0.0%) |
| Eligible strong P1 | 8/8 (100.0%) | 8/8 (100.0%) |
| Eligible P2 | 5/5 (100.0%) | 5/5 (100.0%) |

Seed 2 generated two intervention-positive P1 cases, but both were non-silent (recall drop exceeded 5 percentage points), so they are excluded from the silent-positive TPR denominator rather than counted as detector misses. All five P2 cases were intervention-positive and silent. Maximum N2 logit error was 0.

| Ablation | False positives | Strong P1 detected | P2 detected |
|---|---:|---:|---:|
| Full NCD | 0/34 | 8/8 | 5/5 |
| No probe bootstrap | 0/34 | 6/8 | 3/5 |
| No explainer repeats | 0/34 | 8/8 | 5/5 |
| No FDR | 0/34 | 8/8 | 5/5 |

## Decision

The point-estimate gate passes: observed NCD FPR is at most 5%, and its strong-P1 TPR is not below B2. However, NCD does not beat B1 or B2: all three are tied at every primary count above. All six continuous scores have pooled descriptive AUROC 1.000, so the ROC also provides no evidence that NCD ranks cases better. The 95% Wilson upper bound for an observed 0/34 FPR is 10.2%, so these data do not establish a population FPR below 5%. The pooled ROC is descriptive and not externally calibrated. Removing bootstrap reduces P1 detection from 8/8 to 6/8 and P2 from 5/5 to 3/5; removing repeats or FDR changes none of the pooled binary outcomes on this slice.

This is evidence that the benchmark constructions and silent-drift measurement work on this bounded slice. It is not evidence of substantive innovation, NCD superiority, cross-dataset generality, or publication readiness.

The raw seed 2/3/4/42 JSON files retain an incorrect static seed-1 scope label emitted by the runner. Their `effective_config`, config path, command, logs, and tables identify the actual seed. The source label has been fixed for future runs; raw completed artifacts were not rewritten.

## Next

Run the same frozen intervention benchmark on UNSW-NB15 and iCaRL, with an independent calibration/evaluation split. If B2 remains tied, the defensible conclusion is that repeat averaging is sufficient and NCD should not be claimed as the novel mechanism.
