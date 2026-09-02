# ReplayIDS D2 paired five-seed guarded-checkpoint result

Seeds: `1, 2, 3, 4, 42`. Primary route: `official/joint_cap3000`.

| Metric | Last epoch mean +/- SD | Guarded checkpoint mean +/- SD | Paired delta | 95% CI |
|---|---:|---:|---:|---:|
| Average task accuracy | 79.78% +/- 6.86% | 77.99% +/- 10.01% | -1.79% | [-6.27%, 2.69%] |
| Average forgetting | 3.52% +/- 2.60% | 3.56% +/- 4.71% | 0.04% | [-3.28%, 3.36%] |
| Final accuracy | 85.51% +/- 6.18% | 87.10% +/- 5.19% | 1.59% | [-0.33%, 3.52%] |
| Final Macro-F1 | 54.40% +/- 5.07% | 55.93% +/- 5.02% | 1.53% | [-2.69%, 5.75%] |
| Final balanced accuracy | 77.63% +/- 12.05% | 76.02% +/- 12.81% | -1.60% | [-5.06%, 1.85%] |
| Attack recall | 85.73% +/- 14.98% | 83.44% +/- 17.49% | -2.29% | [-5.91%, 1.33%] |
| Benign FPR | 13.10% +/- 8.25% | 10.90% +/- 6.43% | -2.20% | [-4.87%, 0.47%] |

## Inference boundary

Paired t-tests are designated only for Macro-F1 and forgetting and are Holm-adjusted across those two outcomes. Other metrics are descriptive.

- `final_macro_f1`: raw paired-t p=0.372005; Holm p=0.744009; Wilcoxon p=0.3125; Cohen dz=0.4492.
- `average_forgetting`: raw paired-t p=0.977001; Holm p=0.977001; Wilcoxon p=0.8125; Cohen dz=0.01372.

## Limitations

- five paired training seeds share one fixed data split and are not five independent datasets.
- n=5 limits distributional diagnostics and exact Wilcoxon power.
- Heartbleed has one training-calibration row and therefore uses the registered last-epoch fallback.
