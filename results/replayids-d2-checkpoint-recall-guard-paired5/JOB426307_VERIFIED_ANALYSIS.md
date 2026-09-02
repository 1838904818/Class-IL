# Job 426307 verified analysis

## Evidence status

- Slurm state: `COMPLETED`, exit code 0
- Runtime: 01:00:23 on one A100 GPU
- Seeds: `1, 2, 3, 4, 42`
- Route: `official/joint_cap3000`
- Baseline: immutable Job 425539 `last_epoch`
- Candidate: training-only Macro-F1 checkpoint selection with project-defined
  positive-recall and negative-FPR guards
- Remote protected SHA-256 registry: `PASS`
- Downloaded protected SHA-256 registry: `PASS`
- Registered versus independent statistical recomputation: `PASS` at `1e-12`

## Main paired result

| Metric | Baseline mean (SD) | Guarded mean (SD) | Delta | 95% CI |
|---|---:|---:|---:|---:|
| Average task accuracy | 79.78% (6.86) | 77.99% (10.01) | -1.79 pp | [-6.27, 2.69] |
| Average forgetting | 3.52% (2.60) | 3.56% (4.71) | +0.04 pp | [-3.28, 3.36] |
| Final accuracy | 85.51% (6.18) | 87.10% (5.19) | +1.59 pp | [-0.33, 3.52] |
| Final Macro-F1 | 54.40% (5.07) | 55.93% (5.02) | +1.53 pp | [-2.69, 5.75] |
| Balanced accuracy | 77.63% (12.05) | 76.02% (12.81) | -1.60 pp | [-5.06, 1.85] |
| Attack recall | 85.73% (14.98) | 83.44% (17.49) | -2.29 pp | [-5.91, 1.33] |
| Benign FPR | 13.10% (8.25) | 10.90% (6.43) | -2.20 pp | [-4.87, 0.47] |

The registered primary outcomes do not reject the paired null after Holm
correction. Macro-F1 has raw paired-t p=0.3720, Holm p=0.7440, Wilcoxon
p=0.3125 and Cohen dz=0.449. Forgetting has raw/Holm p=0.9770, Wilcoxon
p=0.8125 and Cohen dz=0.014. With five pairs, this is evidence of uncertainty,
not evidence that the methods are equivalent.

## Seed consistency

Only seed 2 improves or preserves all five directional objectives: final
accuracy, Macro-F1, forgetting, attack recall and Benign FPR. Seeds 1, 3, 4
and 42 each have at least one adverse objective. This prevents a stable
superiority claim.

## What changed at class level

Five-seed mean guarded-minus-baseline changes:

| Class | Precision | Recall | F1 | Official-test support per seed |
|---|---:|---:|---:|---:|
| Benign | -0.56 pp | +2.20 pp | +1.00 pp | 174,421 |
| DoS GoldenEye | +6.01 pp | -1.48 pp | +3.06 pp | 2,059 |
| DoS Hulk | +6.22 pp | +0.06 pp | +2.25 pp | 46,215 |
| DoS Slowhttptest | -0.05 pp | -0.33 pp | -0.13 pp | 1,100 |
| DoS slowloris | -1.32 pp | +0.07 pp | -1.27 pp | 1,159 |
| FTP-Patator | +9.58 pp | -9.37 pp | +12.08 pp | 1,588 |
| Heartbleed | -0.98 pp | +0.00 pp | -1.50 pp | 2 |
| SSH-Patator | -1.86 pp | -3.99 pp | -3.27 pp | 1,179 |

Heartbleed support is too small for a population claim. FTP-Patator's mean F1
change is nonlinear across seeds and must be read together with the large recall
loss and seed-level variation.

## Confusion redistribution

The largest five-seed summed off-diagonal changes show that the selector moves
errors rather than uniformly removing them:

| Actual class | Predicted class | Guarded minus baseline count |
|---|---|---:|
| Benign | FTP-Patator | -20,492 |
| Benign | SSH-Patator | +15,876 |
| Benign | DoS Hulk | -14,827 |
| DoS Hulk | FTP-Patator | -8,599 |
| DoS Hulk | Benign | +4,854 |
| DoS Hulk | SSH-Patator | +4,455 |
| FTP-Patator | SSH-Patator | +735 |
| FTP-Patator | Benign | +542 |
| SSH-Patator | Benign | +237 |

This explains the aggregate trade-off: fewer benign samples are called
FTP-Patator or Hulk, but many are redirected to SSH-Patator, while more Hulk,
FTP and SSH attacks are redirected to Benign or another attack family.

## Checkpoint-selection behavior

The guard retained epoch 10 for all seeds in Benign, GoldenEye,
Slowhttptest and Heartbleed; for four of five seeds in slowloris and
SSH-Patator; and for three of five seeds in FTP-Patator. Hulk selected epochs
3, 4 or 6 in every seed. Heartbleed always fell back because its training
calibration support is one row.

## Decision

Do not replace the paper's registered last-epoch baseline with this guarded
selector as a universally better method. Retain it as a valuable negative and
trade-off experiment:

> A training-only checkpoint guard recovered the earlier selector's forgetting
> penalty and reduced benign false positives, but its official-test attack
> recall was lower and no primary paired effect was statistically resolved.

The next experiment should target the specific FTP/SSH/Hulk confusion
redistribution using training-only calibration, rather than adding another
global scalar threshold. Any candidate must remain fixed-protocol, paired and
review-gated before DICC submission.

## Immutable identifiers

- Protected summary SHA-256: `dde3ea57a5a0edfdadefe9f8914f4a0a68b6b38405ac6755805d39773d064e20`
- Bindings SHA-256: `6cdcc1983eb0720820cbf798f69f0b08f378c2083f11b68817a45f5713a8f15e`
- W&B registry SHA-256: `fa84ee728904f1df75532c7f23be6727f7973614e545f5e2ccc7027a74610347`
- Protected registry SHA-256: `5eab4f01db7e699da66b7503425af347b47595245ea65bf76e129756f3e12fbb`
- Independent analysis SHA-256: `c46d3d25e4a9c839d0f653f4a45e58cfa5c7a5302df8b9c89e00d27c231edbe6`
