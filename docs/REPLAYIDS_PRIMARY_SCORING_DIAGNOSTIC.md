# ReplayIDS D2: primary checkpoint scope and scoring diagnostic

Updated 7 September 2026. This is a retrospective re-analysis of existing, hash-verified results. No training, parameter tuning or new prediction was performed.

## Correction to the balanced-replay comparison

The preceding balanced-replay report used OFRA results from **Job 426307, guarded checkpoint selection**. Its reported 87.10% accuracy and 83.44% attack recall are numerically valid, but refer to the secondary guarded policy. The registered primary remains **Job 425539, last epoch**. The guarded policy was not promoted by the earlier five-seed comparison. The inference arm `official/joint_cap3000` is common to both; it does not identify the checkpoint-selection policy. Both labels are necessary.

The original protected artifacts and previous comparison exports are preserved. The additional last-epoch comparison below is retrospective, not a newly preregistered trial. It corrects the scope of the current narrative without replacing any run or using test outcomes to select a new rule. Malaya results are not changed by this ReplayIDS-specific correction.

## Same-data comparison

Mean percentages across training seeds. SD, paired deltas, all seven metrics, exact sign-flip/Wilcoxon tests and within-cohort Holm adjustments are provided in the diagnostic JSON. Lower is better for forgetting and FPR only.

### All five seeds: 1, 2, 3, 4, 42

| Metric | Balanced Replay50 | OFRA last epoch (primary) | OFRA guarded (secondary) |
|---|---:|---:|---:|
| Mean final task accuracy | 86.44 | 79.78 | 77.99 |
| Overall accuracy | 70.65 | 85.51 | 87.10 |
| Macro-F1 | 36.88 | 54.40 | 55.93 |
| Balanced accuracy | 87.22 | 77.63 | 76.02 |
| Signed forgetting (lower better) | 5.89 | 3.52 | 3.56 |
| Attack recall | 99.78 | 85.73 | 83.44 |
| Benign FPR (lower better) | 35.61 | 13.10 | 10.90 |

### Seeds 1-4 sensitivity; excludes the comparator-screening seed

| Metric | Balanced Replay50 | OFRA last epoch (primary) | OFRA guarded (secondary) |
|---|---:|---:|---:|
| Mean final task accuracy | 87.54 | 76.86 | 74.06 |
| Overall accuracy | 71.71 | 84.61 | 86.07 |
| Macro-F1 | 36.48 | 54.07 | 54.56 |
| Balanced accuracy | 88.09 | 74.17 | 71.88 |
| Signed forgetting (lower better) | 4.67 | 3.31 | 4.13 |
| Attack recall | 99.77 | 83.11 | 80.31 |
| Benign FPR (lower better) | 34.35 | 13.76 | 11.12 |

The primary has higher overall accuracy and Macro-F1 in 5/5 pairs, and lower FPR in 5/5, but lower attack recall in 5/5. Forgetting improves in only 3/5. This remains a trade-off, not uniform dominance. Exact two-sided tests with five nonzero pairs cannot reach p < 0.05 (minimum 0.0625); for four pairs the minimum is 0.125. Seed-42 selection and repeated use of the same test split require descriptive interpretation.

## Where the scoring trade-off appears

The following inference arms share each seed's last-epoch trained OFRA state. `head_only` is the OFRA family-head score branch, not the shared multiclass classifier trained by Balanced Replay50. The router-only arms use centroid-based scores. Joint arms combine head score and 0.5 times router score. The 0.5 weight and the cap of 3,000 fitting rows are fixed project settings, not universal or newly optimized values.

| Last-epoch scoring arm | Accuracy | Macro-F1 | Forgetting | Attack recall | Benign FPR |
|---|---:|---:|---:|---:|---:|
| head_only | 66.20 | 26.79 | 11.58 | 73.26 | 34.74 |
| router_only_cap3000 | 78.57 | 42.04 | 9.04 | 92.06 | 21.76 |
| joint_cap3000 | 85.51 | 54.40 | 3.52 | 85.73 | 13.10 |
| router_only_uncapped | 78.58 | 42.26 | 7.92 | 89.72 | 21.58 |
| joint_uncapped | 85.22 | 54.79 | 3.31 | 85.88 | 13.49 |
| head_only_exposure_prior_corrected | 62.12 | 27.45 | 12.21 | 70.89 | 37.02 |
| joint_cap3000_exposure_prior_corrected | 84.10 | 51.42 | 3.94 | 84.81 | 12.17 |
| joint_uncapped_exposure_prior_corrected | 84.19 | 51.36 | 3.67 | 85.59 | 12.28 |

Removing the router does not resolve the overall problem: mean head-only Macro-F1 is 26.79%, versus 54.40% for joint. Conversely, router-only cap3000 reaches 92.06% attack recall versus 85.73% for joint, with a higher FPR (21.76% versus 13.10%). Thus combining the scores changes the sensitivity/false-alarm operating point. It is not evidence that routing has no value, nor proof that head calibration is the sole cause.

The existing exposure-prior-corrected joint arm has lower mean Macro-F1 and attack recall than uncorrected joint. Uncapping the router has mixed endpoint changes, not a general cure. These negative diagnostics are retained rather than promoting the best-looking test arm.

## Per-class accounting

For each true attack class, count predictions assigned to Benign. The table sums these counts over five runs of the **same test examples**. It does not contain five times as many independent examples. A positive joint-minus-router count indicates more aggregate misses under joint; it does not identify which individual rows flipped.

| True attack class | Fixed test support per seed | Joint minus router-only cap3000: total Benign predictions | Mean extra miss rate (pp) |
|---|---:|---:|---:|
| DoS GoldenEye | 2059 | +80 | +0.78 |
| DoS Hulk | 46215 | +13372 | +5.79 |
| DoS Slowhttptest | 1100 | +568 | +10.33 |
| DoS slowloris | 1159 | +705 | +12.17 |
| FTP-Patator | 1588 | +1114 | +14.03 |
| Heartbleed | 2 | +2 | +20.00 |
| SSH-Patator | 1179 | +1026 | +17.40 |

DoS Hulk contributes 13,372 of the net 16,867 additional Benign predictions across these five same-test runs. This motivates examining class-dependent head/router margins, but support imbalance partly explains the count concentration. Per-class rates and seed variation must accompany counts. Heartbleed support is two; no rare-class population claim follows.

## Next experiment: admission gates, not executed work

1. Keep last-epoch joint_cap3000 as the registered primary. Retain guarded selection and all eight scoring arms as explicitly labelled secondary diagnostics.
2. Before changing the model, collect or recover head scores, router scores, true-class-versus-Benign margins and immutable row identifiers on the existing training-only calibration partition. Verify that calibration rows were excluded from fitting, and report per-class support. Never fit thresholds or fusion weights on official-test labels.
3. Use the calibration evidence to define one prospective score-combination/calibration candidate and an unchanged comparator. Lock its objective, constraints, rare-class fallback, data/code hashes and evaluation protocol before running it. Existing margins alone cannot establish that any proposed calibration will work.
4. Audit training differences alongside inference: OFRA binary family heads use focal loss and bounded negative sampling, whereas the replay comparator trains a shared multiclass head with balanced old/new exposure. A performance difference is not attributable to architecture alone. Equal epochs and exemplar counts do not match compute or total memory.
5. Prospective runs require a versioned, fixed protocol and institutional resource compliance. Preserve all five seeds and the separate 1-4 sensitivity view; do not increase model size or sweep test-set weights as a substitute for an identified mechanism.

## Verification and reproduction

Ten original result files (five last-epoch and five guarded) are pinned by the existing independent-analysis registry. Result files are checked byte-for-byte. The older registry is additionally bound by an explicit LF-normalized hash because Git normalizes its Windows line endings; its protected original-byte hash is retained separately. All eight scoring arms at all four checkpoints were recomputed from confusion matrices, including per-class metrics, binary recall/FPR, task accuracies and signed forgetting. Balanced Replay50 inputs were rechecked against their export manifest. Normalization, task order and final class supports match across paired inputs.

[POLICY_BINDINGS.json](../results/replayids-primary-diagnostic/POLICY_BINDINGS.json) records checkpoint policy, exact protected protocol/result hashes, data manifest and runtime hashes. Each original protocol was checked against its protected checksum registry before this non-sensitive extraction. Full protocols containing internal paths are not republished in this addendum.

```text
python tools/replayids_primary_diagnostic.py --repo . --output primary_diagnostic.json
python tests/test_replayids_primary_diagnostic.py
```

[DIAGNOSTIC.json](../results/replayids-primary-diagnostic/DIAGNOSTIC.json) contains both cohorts, all arms, per-class net counts and input hashes. The tool is read-only except for the explicitly named output. Statistical intervals inherit the small-sample assumptions of the paired aggregation tool; these diagnostics are not confirmatory multiple-comparison evidence. No new SHAP/ETG experiment or independently verified W&B cloud result is claimed.
