# Balanced Replay50 versus OFRA: completed paired replication

**Checkpoint-scope correction (7 September 2026): ReplayIDS OFRA values below refer to Job 426307 guarded checkpoint selection, a secondary policy, not the registered last-epoch primary.** The original numbers and exports are retained. See the [last-epoch comparison and scoring diagnostic](REPLAYIDS_PRIMARY_SCORING_DIAGNOSTIC.md) for the primary result (85.51% accuracy, 54.40% Macro-F1, 85.73% attack recall, 13.10% FPR). Malaya is unaffected by this correction. `joint_cap3000` identifies the scoring arm, not the checkpoint-selection policy.

Updated 7 September 2026. Both datasets have completed training seeds 1, 2, 3, 4 and 42. This report concerns prediction/retention only; it does not add SHAP or ETG evidence.

## Conclusion

OFRA has higher final overall accuracy in every paired seed on both datasets. Malaya shows substantially lower forgetting, but little new-seed Macro-F1 improvement. ReplayIDS D2 shows higher Macro-F1 and lower benign false-positive rates, alongside lower attack recall and substantial seed sensitivity. These are protocol-specific trade-offs, not uniform superiority.

## Interpretation and experimental boundary

- Balanced Replay50 updates a shared FT-Transformer and multiclass head. Later-task training pairs each new row with one old-memory draw; 50 herded examples are retained per observed class.
- OFRA uses the existing same-seed official-test joint_cap3000 arm: a frozen Task-0 encoder, class-specific LoRA heads and centroid routing. This comparison does not isolate which component causes a difference.
- The fixed preprocessing/task/data contract, Task-0-only normalization and matching class supports were verified. Malaya and ReplayIDS D2 are run independently, not mixed.
- See the [replication protocol](BALANCED_REPLAY_REPLICATION_PROTOCOL.md) for capacities, epochs and sampling. Equal epoch counts do not mean equal compute: replay doubles later-task row exposure. OFRA retains centroids in addition to exemplars, so total memory is not matched.
- Seed 42 selected balanced replay over frozen NME in a diagnostic screen. All-five results are therefore descriptive; the four new seeds are reported separately without changing the method. They vary training randomness on the same split and class order, not independent data collection or unseen domains.

## Metric definitions

Overall accuracy weights each final test row equally. Mean final task accuracy gives each task equal weight at the final checkpoint; it is not an average over training epochs. Balanced accuracy averages class recall; Macro-F1 averages the per-class harmonic mean of precision and recall. These weightings can rank methods differently.

Signed forgetting is the mean, over old tasks, of best accuracy before the final checkpoint minus final accuracy. Negative values mean backward improvement and are not clipped. Low forgetting can also reflect weak initial learning; it must be interpreted with absolute performance.

Attack detection recall counts any attack label as an attack, even if the attack family is misidentified. Benign FPR is the fraction of benign traffic incorrectly flagged as any attack. Neither binary metric is defined for Malaya application-identity labels. Per-class recall is the fraction correctly classified within that true class; it is not one-versus-rest accuracy inflated by true negatives.

## Paired results

Entries are mean +/- sample standard deviation in percent. Delta is OFRA minus Replay50 in percentage points; a negative delta is favorable only for forgetting and FPR. W/L counts use the favorable direction.

### MalayaNetwork_GT

#### All five seeds (includes screening seed 42)

| Metric | Balanced Replay50 | OFRA joint_cap3000 | Delta (pp) | OFRA W/L |
|---|---:|---:|---:|---:|
| Mean final task accuracy | 32.63 +/- 6.61 | 30.95 +/- 4.77 | -1.68 | 1/4 |
| Final overall accuracy | 40.14 +/- 13.17 | 54.68 +/- 2.71 | +14.54 | 5/0 |
| Macro-F1 | 18.72 +/- 4.42 | 21.15 +/- 3.37 | +2.43 | 4/1 |
| Balanced accuracy | 22.13 +/- 3.70 | 23.03 +/- 3.42 | +0.90 | 4/1 |
| Signed average forgetting (lower better) | 43.80 +/- 4.66 | 3.55 +/- 0.77 | -40.25 | 5/0 |

#### Seeds 1-4 sensitivity (excludes screening seed 42)

| Metric | Balanced Replay50 | OFRA joint_cap3000 | Delta (pp) | OFRA W/L |
|---|---:|---:|---:|---:|
| Mean final task accuracy | 35.15 +/- 4.00 | 30.56 +/- 5.42 | -4.58 | 0/4 |
| Final overall accuracy | 45.49 +/- 6.36 | 54.37 +/- 3.02 | +8.87 | 4/0 |
| Macro-F1 | 20.47 +/- 2.36 | 20.70 +/- 3.72 | +0.23 | 3/1 |
| Balanced accuracy | 23.57 +/- 2.09 | 22.70 +/- 3.86 | -0.87 | 3/1 |
| Signed average forgetting (lower better) | 43.19 +/- 5.15 | 3.79 +/- 0.64 | -39.40 | 4/0 |

#### Individual seeds

| Seed | Replay accuracy | OFRA accuracy | Replay Macro-F1 | OFRA Macro-F1 | Replay forgetting | OFRA forgetting |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 51.67 | 53.77 | 21.63 | 24.17 | 36.73 | 3.77 |
| 2 | 36.64 | 54.02 | 17.10 | 18.75 | 49.28 | 3.00 |
| 3 | 47.67 | 51.20 | 22.45 | 16.44 | 42.80 | 3.82 |
| 4 | 46.00 | 58.48 | 20.70 | 23.46 | 43.93 | 4.56 |
| 42 | 18.75 | 55.94 | 11.71 | 22.91 | 46.23 | 2.59 |

### ReplayIDS D2

#### All five seeds (includes screening seed 42)

| Metric | Balanced Replay50 | OFRA joint_cap3000 | Delta (pp) | OFRA W/L |
|---|---:|---:|---:|---:|
| Mean final task accuracy | 86.44 +/- 4.63 | 77.99 +/- 10.01 | -8.45 | 1/4 |
| Final overall accuracy | 70.65 +/- 4.98 | 87.10 +/- 5.19 | +16.45 | 5/0 |
| Macro-F1 | 36.88 +/- 4.85 | 55.93 +/- 5.02 | +19.05 | 5/0 |
| Balanced accuracy | 87.22 +/- 5.22 | 76.02 +/- 12.81 | -11.20 | 2/3 |
| Signed average forgetting (lower better) | 5.89 +/- 8.69 | 3.56 +/- 4.71 | -2.34 | 3/2 |
| Attack detection recall | 99.78 +/- 0.16 | 83.44 +/- 17.49 | -16.34 | 0/5 |
| Benign false-positive rate (lower better) | 35.61 +/- 5.63 | 10.90 +/- 6.43 | -24.71 | 5/0 |

#### Seeds 1-4 sensitivity (excludes screening seed 42)

| Metric | Balanced Replay50 | OFRA joint_cap3000 | Delta (pp) | OFRA W/L |
|---|---:|---:|---:|---:|
| Mean final task accuracy | 87.54 +/- 4.53 | 74.06 +/- 5.53 | -13.48 | 0/4 |
| Final overall accuracy | 71.71 +/- 5.05 | 86.07 +/- 5.37 | +14.36 | 4/0 |
| Macro-F1 | 36.48 +/- 5.51 | 54.56 +/- 4.60 | +18.07 | 4/0 |
| Balanced accuracy | 88.09 +/- 5.60 | 71.88 +/- 10.20 | -16.21 | 1/3 |
| Signed average forgetting (lower better) | 4.67 +/- 9.52 | 4.13 +/- 5.23 | -0.54 | 2/2 |
| Attack detection recall | 99.77 +/- 0.18 | 80.31 +/- 18.52 | -19.46 | 0/4 |
| Benign false-positive rate (lower better) | 34.35 +/- 5.62 | 11.12 +/- 7.40 | -23.23 | 4/0 |

#### Individual seeds

| Seed | Replay accuracy | OFRA accuracy | Replay Macro-F1 | OFRA Macro-F1 | Replay forgetting | OFRA forgetting |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 67.48 | 80.07 | 33.97 | 54.05 | 15.09 | 11.95 |
| 2 | 74.03 | 93.02 | 38.29 | 59.74 | 9.94 | 1.04 |
| 3 | 67.60 | 84.65 | 30.46 | 55.79 | -5.67 | 1.93 |
| 4 | 77.74 | 86.54 | 43.21 | 48.66 | -0.69 | 1.58 |
| 42 | 66.40 | 91.23 | 38.46 | 61.41 | 10.81 | 1.27 |

## Negative findings and claim limits

- Malaya seeds 1-4: Macro-F1 is 20.70% for OFRA versus 20.47% for replay, only +0.23 pp on average. OFRA loses Macro-F1 on seed 3; both methods remain weak across the ten application classes. Mean final task accuracy favors replay in all four new seeds.
- ReplayIDS D2: OFRA attack recall is lower in all five seeds. OFRA seed 3 recalls 69.52% of attacks and seed 4 recalls 59.79%; corresponding replay values are 99.90% and 99.52%. Lower benign FPR must not conceal these missed attacks. The fixed test set contains 174,421 benign rows out of 227,723 (76.59%), so overall accuracy alone is insufficient.
- ReplayIDS forgetting is lower for OFRA in only three of five seeds, and two of four new seeds. Replay seeds 3 and 4 have negative signed forgetting; these valid backward improvements are retained.
- ReplayIDS Heartbleed has only two test examples. High recall on two examples does not establish rare-attack generalization; precision, predicted count and support must be inspected in the exported confusion matrices.
- The stronger replay comparison narrows claims from the earlier collapsed-baseline screen. It is not a tuned state-of-the-art benchmark, deployment approval, or evidence of journal acceptance.

## Uncertainty and statistical limits

Machine-readable aggregates include paired deltas, approximate paired Hedges g_z, descriptive Student-t intervals, exact two-sided sign-flip tests and exact Wilcoxon signed-rank enumeration (average tied ranks; zero differences omitted). Holm correction is applied within each dataset/cohort across every reported metric, separately for each test. This is not a study-wide confirmatory testing family.

With five nonzero pairs, the smallest attainable two-sided exact p-value is 0.0625; with four, it is 0.125. No p < 0.05 superiority claim is supported. Normality for t intervals and sign symmetry/exchangeability for the enumeration tests are assumptions, not established properties at this sample size. Seed-42 selection and repeated use of the same test partition further limit confirmatory interpretation.

## Evidence, per-class results and reproduction

The [export package](../results/balanced-replay-five-seed/) contains 10 paired comparisons, 10 baseline audits, 10 exact baseline protocol files, and 10 evaluation exports containing both methods at every checkpoint (confusion matrices, per-class precision/recall/F1/support, task accuracies and summary matrices). PROVENANCE.json binds the original result, code/config/data and protocol hashes; EXPORT_SHA256.json binds every JSON export.

Full protected result containers were verified locally, but are not republished wholesale. The exports omit raw traffic, model weights, internal storage paths and credentials. Re-running aggregation uses only the published pair/audit exports; full training reproduction requires the separately governed dataset and the exact runtime hashes in the protocol.

```text
python tools/analyze_balanced_replay_replication.py results/balanced-replay-five-seed/malaya_seed*_comparison.json --output malaya_aggregate.json
python tools/analyze_balanced_replay_replication.py results/balanced-replay-five-seed/replayids_d2_seed*_comparison.json --output replayids_d2_aggregate.json
python tests/test_balanced_replay_replication_analysis.py
```

The tool accepts wildcard patterns or the five exact comparison filenames. All ten baseline runs have W&B URLs recorded in their protected result records. Cloud panel/table completeness remains independently unverified; numerical verification uses local immutable evidence.

| Dataset / seed | Completed Slurm job | Recorded W&B run |
|---|---:|---|
| malaya / 1 | 451049 | [e4gc1uvh](https://wandb.ai/csnet/ofra-etg-leon-hpc/runs/e4gc1uvh) |
| malaya / 2 | 451841 | [ik0kgdaw](https://wandb.ai/csnet/ofra-etg-leon-hpc/runs/ik0kgdaw) |
| malaya / 3 | 451843 | [gtsseoz7](https://wandb.ai/csnet/ofra-etg-leon-hpc/runs/gtsseoz7) |
| malaya / 4 | 451845 | [3qmjylo3](https://wandb.ai/csnet/ofra-etg-leon-hpc/runs/3qmjylo3) |
| malaya / 42 | See screening record | [h6auw0qi](https://wandb.ai/csnet/ofra-etg-leon-hpc/runs/h6auw0qi) |
| replayids_d2 / 1 | 451050 | [349vilce](https://wandb.ai/csnet/ofra-etg-leon-hpc/runs/349vilce) |
| replayids_d2 / 2 | 451842 | [a0bzg1fz](https://wandb.ai/csnet/ofra-etg-leon-hpc/runs/a0bzg1fz) |
| replayids_d2 / 3 | 451844 | [fq6htiul](https://wandb.ai/csnet/ofra-etg-leon-hpc/runs/fq6htiul) |
| replayids_d2 / 4 | 451846 | [h0p8z73v](https://wandb.ai/csnet/ofra-etg-leon-hpc/runs/h0p8z73v) |
| replayids_d2 / 42 | See screening record | [nhreghd8](https://wandb.ai/csnet/ofra-etg-leon-hpc/runs/nhreghd8) |

No experiment, checkpoint choice, data split, or prediction rule was changed during this local audit.
