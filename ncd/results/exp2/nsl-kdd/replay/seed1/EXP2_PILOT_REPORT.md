# Exp 2 intervention-ground-truth pilot

## What ran

NSL-KDD, Replay, seed 1, Normal target, transition 0 to 1.  P1 used four shortcut strengths; P2 zeroed the checkpoint-0 top-three feature-ablation features in 25%–100% of target-class replay rows.  Each scenario used N1, exact-logit N2, three explainers, 10 repeats, 10 probe bootstraps, 1,000 permutations, CLEX ED, matched-FPR/AUROC comparisons, and three NCD ablations.

- Code commit: `0c810e48adaa6ea8406025c7760a4c09008e5858`
- Command: `python -I -B ncd/groundtruth.py --config ncd/configs/exp2_nsl_replay_seed1.json`
- Full log: retained in the private research repository and omitted here because it contains local execution paths
- Elapsed: 635.8 seconds on CUDA

## Numbers

| P1 rho | checkpoint-0 effect | checkpoint-1 effect | recall change | true change | silent | NCD | B2 |
|---:|---:|---:|---:|:---:|:---:|:---:|:---:|
| 0.5 | 0.42 pp | 4.52 pp | -8.09 pp | no | no | yes | yes |
| 0.7 | 0.45 pp | 6.45 pp | -5.66 pp | yes | no | yes | yes |
| 0.9 | 0.61 pp | 11.95 pp | -2.73 pp | yes | yes | yes | yes |
| 1.0 | 0.70 pp | 48.89 pp | +0.13 pp | yes | yes | yes | yes |

| P2 masked replay fraction | intervention-effect change | recall change | true change | silent | NCD | B2 |
|---:|---:|---:|:---:|:---:|:---:|:---:|
| 0.25 | 3.56 pp | -11.36 pp | no | no | yes | yes |
| 0.50 | 3.49 pp | -10.58 pp | no | no | yes | yes |
| 0.75 | 2.74 pp | -10.76 pp | no | no | yes | yes |
| 1.00 | 2.57 pp | -9.88 pp | no | no | yes | yes |

| Predeclared check | NCD | B2 repeat-mean J@15 | Decision |
|---|---:|---:|---|
| N1/N2 negative-control false positives | 0/16 (0%) | 0/16 (0%) | NCD passes the <=5% requirement |
| Silent P1 positives at rho >= 0.9 | 2/2 (100%) | 2/2 (100%) | NCD ties rather than beats B2 |
| N2 maximum absolute logit difference | 0 | n/a | passes `< 1e-5` |
| Threshold-negative interventions (P1 rho 0.5 + all P2) | 5/5 signalled | 5/5 signalled | both rules over-signal these interventions |

All six continuous metrics—B1, B2, adapted CLEX ED, cosine, Kendall, and NCD—have AUROC 1.0 on the same 16 negative controls versus only the two strong silent P1 positives, with matched empirical FPR 0% and TPR 100%.  This tiny in-sample separation cannot distinguish the methods.  Removing probe bootstrap reduces NCD's strong-P1 consensus TPR from 100% to 50%; removing repeats or FDR changes neither FPR nor TPR in this slice.

## Decision and next step

The P1 success criterion is met, but the pilot does **not** establish an NCD advantage or substantive innovation: B2 ties NCD, CLEX and B4 also separate the two strong positives, and repeated explanation sampling adds no measured benefit here.  P2 produced no eligible positive and exposed a more serious calibration problem: attribution drift was detected even when the intervention-defined dependency change stayed below 5 pp and performance was not silent.

Next: redesign P2 to create silent intervention-grounded positives, reserve separate controls for threshold calibration, and rerun across seeds before any novelty claim.  Until then, the defensible result is a validated P1 benchmark plus a negative result on method superiority.

## Evidence hashes

| Artifact | SHA-256 |
|---|---|
| `exp2_groundtruth_result.json` | `c817029fb4a128ece473ea3220b0354ab42a884ae72d127d3b9f5220a290a5b5` |
| `exp2_groundtruth_table.csv` | `ef07322fced3c11e74621b3febaacd62a8a4eff6ee7f1068653c65467b183438` |
| `exp2_groundtruth_consensus.csv` | `7beb62aaa60c51d2ec165e499ea00e1c918f8563d265db6f1ac68e7ff83cf765` |
| `exp2_groundtruth_discrimination.csv` | `63dfca946f33d9bc9f43fb0303ce2bac927ca7d488c2058bc8c8f6d2ae43be19` |
| `exp2_groundtruth_compact.npz` | `4759f7a45933bee2ca11e717b2d317001564e757ef4a1e5ff099f416303d47c0` |
| Private full log | `23514aa90d6a35c8bcb3db849986a0c93f25e6b2a85a0ebda5f7e14aa2c7bcdf` |
