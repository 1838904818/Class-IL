# Recall/FPR-guarded checkpoint selection

This package records the paired five-seed guarded checkpoint experiment from
DICC Job 426307 and its comparison with the immutable Job 425539 last-epoch
baseline. Seeds are `1, 2, 3, 4, 42`; all use one fixed ReplayIDS-aligned D2
split and the registered `official/joint_cap3000` inference arm.

The non-final checkpoint is admitted only when training-only calibration meets
three project-defined conditions: Macro-F1 gain at least `0.01`, positive-class
recall drop at most `0.01`, and negative-class FPR increase at most `0.01`.
These values are study settings, not published universal thresholds. The
official test split is never used for checkpoint selection.

## Result

| Metric | Last epoch | Guarded | Paired change |
|---|---:|---:|---:|
| Final accuracy | 85.51% | 87.10% | +1.59 pp |
| Macro-F1 | 54.40% | 55.93% | +1.53 pp |
| Forgetting | 3.52% | 3.56% | +0.04 pp |
| Attack recall | 85.73% | 83.44% | -2.29 pp |
| Benign FPR | 13.10% | 10.90% | -2.20 pp |

All paired 95% confidence intervals include zero. Macro-F1 has Holm-adjusted
paired-t `p=0.744`; forgetting has `p=0.977`. Only seed 2 dominates its matched
baseline across the jointly inspected endpoint metrics. The guarded rule is
therefore not promoted over last epoch.

## Contents

- `PAIRED_FIVESEED_SUMMARY.json`: registered aggregate evidence.
- `job426307_independent_analysis.json`: independent recomputation.
- `JOB426307_VERIFIED_ANALYSIS.md`: human-readable interpretation.
- `per_seed/`: guarded result and summary for every registered seed.
- `baseline/`: matched last-epoch result for every registered seed.
- `WANDB_RUN_REGISTRY.json`: run identifiers and public experiment metadata.
- `PROTECTED_RESULTS_SHA256SUMS.txt`: immutable protected-output registry.
- `build_scripts/`: registered aggregate and independent analysis checks.

Recompute the published statistics with:

```bash
python results/replayids-d2-checkpoint-recall-guard-paired5/build_scripts/test_release_recompute.py -v
```

Large inference-state NPZ files and institutional job-control files are not
published in Git. Their hashes remain in the protected registry. The public
package contains no credentials, private paths, presentation notes, or internal
review conversations.
