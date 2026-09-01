# ReplayIDS D2 paired five-seed checkpoint-selection result

DICC Job `425539` completed ten matched runs: two checkpoint-selection arms for
seeds `1, 2, 3, 4, 42`. Both arms use the same fixed split, model, optimizer,
replay budget, router, training budget, stochastic seed, and official
`joint_cap3000` evaluation arm. The only treatment is whether each family head
retains epoch 10 or the earliest epoch with the best training-only calibration
Macro-F1.

## Aggregate result

| Metric | Last epoch | Training-only calibration | Paired delta |
|---|---:|---:|---:|
| Final accuracy | 85.51% +/- 6.18% | 87.52% +/- 4.78% | +2.02 pp |
| Final Macro-F1 | 54.40% +/- 5.07% | 56.34% +/- 4.06% | +1.95 pp |
| Average task accuracy | 79.78% +/- 6.86% | 77.88% +/- 9.57% | -1.91 pp |
| Forgetting | 3.52% +/- 2.60% | 5.18% +/- 5.43% | +1.66 pp |
| Attack recall | 85.73% +/- 14.98% | 83.76% +/- 16.72% | -1.97 pp |
| Benign FPR | 13.10% +/- 8.25% | 10.44% +/- 5.05% | -2.66 pp |

Every paired 95% confidence interval crosses zero. The two designated
inferential outcomes, Macro-F1 and forgetting, both have Holm-adjusted
`p=0.755161`. Training-only calibration is therefore retained as an
inconclusive ablation; it does not replace the last-epoch primary protocol.

## Integrity and boundaries

- Source aggregate SHA-256:
  `084b844aabe34310b7e62937f812df5f5923936f8212241d0e91f9503c53c190`.
- Source report SHA-256:
  `a7e89160ba45f2d3b98551f2672f2b6b6ed00b63ce7cf73121004fec10232990`.
- Source W&B registry SHA-256:
  `44880ed50467ca29a02b1e0c8e4eb5e75bbf193c3173ae2dd5d624cb37bebe67`.
- Five seeds share one fixed data split; they are paired training repetitions,
  not five independent datasets.
- Heartbleed has two official-test rows and one calibration row. It uses the
  registered last-epoch fallback and cannot support a strong class-level claim.
- W&B records in this package contain aggregate experiment metadata and run
  URLs only; no raw samples, feature vectors, checkpoints, or credentials are
  published.

The CSV files provide aggregate and per-seed class metrics. The PNG files are
publication figures derived from the same verified result records.

