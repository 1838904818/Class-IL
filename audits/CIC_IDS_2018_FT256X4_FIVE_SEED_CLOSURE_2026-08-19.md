# CIC-IDS-2018 FT256x4 five-seed closure

## Scope

This is a completed, protocol-separated closure campaign for CIC-IDS-2018. It
uses an FT-Transformer with width 256 and four transformer layers, one Task-0
pretraining epoch, and one training epoch for each later task. The seeds are
`1`, `2`, `3`, `4`, and `42`.

The campaign must not be pooled with the FT512x12, 8/10-epoch results in the
main manuscript table. It answers a narrower question: whether the smaller,
short-schedule configuration produces the same qualitative result across five
independent initializations.

## Source-bound runs

| Seed | DICC job |
|---:|---:|
| 1 | 395350 |
| 2 | 399060 |
| 3 | 399246 |
| 4 | 399313 |
| 42 | 399593 |

All five jobs completed. The aggregation script verifies equality of dataset,
problem type, metric profile, task semantics, normalization algorithm and
sample count, normalization source classes, model parameterization, and
checkpoint count. Protocol hashes remain seed-specific and are retained in the
machine-readable result.

## Official-test results

Values are mean ± sample standard deviation over five seeds.

| Inference view | Final accuracy (%) | Macro-F1 (%) | Balanced accuracy (%) | Average forgetting (%) |
|---|---:|---:|---:|---:|
| Head only | 69.27 ± 20.34 | 21.81 ± 8.82 | 25.67 ± 6.35 | 7.62 ± 7.54 |
| Router only, cap 3000 | 41.01 ± 6.08 | 31.88 ± 6.06 | 58.50 ± 5.03 | 19.13 ± 6.01 |
| Joint score, cap 3000 | 52.55 ± 5.79 | 34.05 ± 7.54 | 53.93 ± 7.79 | 19.03 ± 8.62 |
| Router only, uncapped | 45.35 ± 4.05 | 33.04 ± 4.57 | 58.20 ± 4.03 | 16.61 ± 6.55 |
| Joint score, uncapped | 53.52 ± 8.36 | 34.33 ± 8.05 | 53.94 ± 8.08 | 18.09 ± 11.08 |

## Interpretation

The joint cap-3000 view improves Macro-F1 by 12.24 percentage points and
balanced accuracy by 28.26 points relative to head-only inference. It also
reduces final overall accuracy by 16.72 points and increases average forgetting
by 11.41 points on average. The head-only accuracy is unstable across seeds
(20.34-point standard deviation), while the joint view is more stable in
accuracy (5.79 points).

The cap itself has little mean effect in this campaign: joint cap-3000 differs
from joint uncapped by -0.98 points in final accuracy, -0.28 points in
Macro-F1, -0.01 points in balanced accuracy, and +0.93 points in forgetting.
Those small five-seed differences do not establish superiority of either cap
setting.

These results therefore support a class-balance trade-off, not a blanket claim
that joint routing improves every metric. The high attack-recall / high benign
false-positive regime and the seed sensitivity of head-only inference remain
operational limitations.

## Reproducibility

- Machine-readable summary:
  `results/cic-ids-2018/ft256x4-1plus1-five-seed/summary.json`
- Aggregation script: `scripts/aggregate_cic18_ft256x4_5seed.py`
- Summary canonical SHA-256:
  `ddf6d0f484e7467381b6f56986b89c8d580bbbd07af17f941a5716ee7d341f6b`
