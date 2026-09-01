# ReplayIDS D2 paired five-seed checkpoint-selection result

Seeds: `1, 2, 3, 4, 42`. Primary route: `official/joint_cap3000`.

| Metric | Last epoch mean +/- SD | Training-only best epoch mean +/- SD | Paired delta | 95% CI |
|---|---:|---:|---:|---:|
| Average task accuracy | 79.78% +/- 6.86% | 77.88% +/- 9.57% | -1.91% | [-7.48%, 3.66%] |
| Average forgetting | 3.52% +/- 2.60% | 5.18% +/- 5.43% | 1.66% | [-3.53%, 6.84%] |
| Final accuracy | 85.51% +/- 6.18% | 87.52% +/- 4.78% | 2.02% | [-1.16%, 5.19%] |
| Final Macro-F1 | 54.40% +/- 5.07% | 56.34% +/- 4.06% | 1.95% | [-3.50%, 7.39%] |
| Final balanced accuracy | 77.63% +/- 12.05% | 77.27% +/- 10.62% | -0.35% | [-6.43%, 5.72%] |
| Attack recall | 85.73% +/- 14.98% | 83.76% +/- 16.72% | -1.97% | [-4.41%, 0.48%] |
| Benign FPR | 13.10% +/- 8.25% | 10.44% +/- 5.05% | -2.66% | [-6.83%, 1.51%] |

## Inference boundary

Paired t-tests are designated only for Macro-F1 and forgetting and are Holm-adjusted across those two outcomes. Other metrics are descriptive.

- `final_macro_f1`: raw paired-t p=0.37758; Holm p=0.755161; Wilcoxon p=0.4375; Cohen dz=0.4434.
- `average_forgetting`: raw paired-t p=0.425207; Holm p=0.755161; Wilcoxon p=0.8125; Cohen dz=0.3967.

## Limitations

- five paired training seeds share one fixed data split and are not five independent datasets.
- n=5 limits distributional diagnostics and exact Wilcoxon power.
- Heartbleed has one training-calibration row and therefore uses the registered last-epoch fallback.
