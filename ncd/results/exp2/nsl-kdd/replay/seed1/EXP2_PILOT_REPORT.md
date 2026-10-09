# Exp 2 intervention-ground-truth pilot

## What ran

NSL-KDD, Replay, seed 1, Normal as the old target class, transition 0 to 1.  The run used four shortcut strengths, three explainers, 10 explanation repeats, 10 probe bootstraps, and 1,000 label permutations.  N1 repeated the same checkpoint; N2 applied an exact positive ReLU rescaling and verified the logits; P1 injected four synthetic marker columns and measured their causal effect by permuting them at test time.

- Code commit: `f60f0d22d2bb92f9a228aefc1349e932ac0355c9`
- Command: `python -I -B ncd/groundtruth.py --config ncd/configs/exp2_nsl_replay_seed1.json`
- Full log: retained in the private research repository; omitted here to avoid publishing local paths
- Elapsed time: 325.9 seconds on CUDA

## Numbers

| rho | checkpoint-0 marker effect | checkpoint-1 marker effect | Normal recall change | true dependence change | silent | NCD consensus | repeat-mean J@15 consensus |
|---:|---:|---:|---:|:---:|:---:|:---:|:---:|
| 0.5 | 0.42 pp | 4.52 pp | -8.09 pp | no | no | yes | yes |
| 0.7 | 0.45 pp | 6.45 pp | -5.66 pp | yes | no | yes | yes |
| 0.9 | 0.61 pp | 11.95 pp | -2.73 pp | yes | yes | yes | yes |
| 1.0 | 0.70 pp | 48.89 pp | +0.13 pp | yes | yes | yes | yes |

| Predeclared check | NCD | repeat-mean J@15 | Decision |
|---|---:|---:|---|
| N1/N2 negative-control false positives | 0/8 (0%) | 0/8 (0%) | NCD passes the <=5% requirement |
| Silent P1 positives at rho >= 0.9 | 2/2 (100%) | 2/2 (100%) | NCD meets, but does not exceed, the baseline |
| N2 maximum absolute logit difference | 0 | n/a | passes `< 1e-5` |

All three explainers voted positive for every P1 strength after within-explainer Benjamini-Hochberg correction (`q = 0.002997`).  The compact evidence contains 144 finite arrays and is 740,271 bytes.

## Decision and next step

This slice passes the two predeclared Exp 2 criteria, but it does **not** yet show an advantage over simply averaging repeated attributions: NCD and B2 tie on both strong silent positives.  It is evidence that the intervention creates measurable silent dependency change and that NCD can detect it without firing on N1/N2 in this slice; it is not publication-level TPR/FPR evidence and does not establish novelty.

Next: add P2, CLEX/B3, matched-FPR calibration, AUROC, and the required ablations before expanding across seeds and datasets.

## Evidence hashes

| Artifact | SHA-256 |
|---|---|
| `exp2_groundtruth_result.json` | `6e9d95cc6832c291356bb7513b0b4f51925a927c057c12fed253947ae06ac39e` |
| `exp2_groundtruth_table.csv` | `96e6d562c69f03d82a33d31312ac4b10434a852f5509ca1d0b768c299cd5db2b` |
| `exp2_groundtruth_consensus.csv` | `e28791356e03993adfd7a6984f77b6891c104ed2b09bede73ec71db547a3adc0` |
| `exp2_groundtruth_compact.npz` | `fec2d98db5412f53362253f84f6d941cb21c526320706349f9c75359ac0a1f40` |
| Private full log | `8e52764e84b62deb48ea7e86a885d8eecf5a364a4dbaab50023e47cbc933cfe9` |
