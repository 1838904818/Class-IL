# Balanced Replay50 versus OFRA: seed-1 paired evidence

Updated 7 September 2026. These are individual-seed comparisons, not five-seed conclusions. OFRA denotes the existing same-seed `official / joint_cap3000` arm. Dataset contracts and task order are fixed, as specified in the [replication protocol](BALANCED_REPLAY_REPLICATION_PROTOCOL.md).

## Results

All entries are percentages; differences should be expressed in percentage points. Forgetting and benign false-positive rate are lower-is-better. All other listed metrics are higher-is-better.

| Metric | Malaya: Replay50 | Malaya: OFRA | ReplayIDS D2: Replay50 | ReplayIDS D2: OFRA |
|---|---:|---:|---:|---:|
| Mean final accuracy across tasks | 39.78 | 35.15 | 87.45 | 71.09 |
| Final overall accuracy | 51.67 | 53.77 | 67.48 | 80.07 |
| Final Macro-F1 | 21.63 | 24.17 | 33.97 | 54.05 |
| Final balanced accuracy | 24.97 | 26.13 | 89.31 | 77.77 |
| Average forgetting | 36.73 | 3.77 | 15.09 | 11.95 |
| Attack detection recall | Not defined | Not defined | 99.91 | 96.50 |
| Benign false-positive rate | Not defined | Not defined | 42.03 | 21.82 |

The mean task metric averages final-checkpoint task accuracies with equal task weight. Overall accuracy instead weights every final test row equally; with unequal task sizes these need not rank methods identically. Balanced accuracy averages class recall equally. Macro-F1 also penalizes poor precision and therefore need not track balanced accuracy.

## Interpretation

- Malaya: OFRA has 2.10 points higher final accuracy, 2.54 points higher Macro-F1 and 32.96 points lower forgetting. Replay50 has 4.63 points higher mean final task accuracy. Neither model performs strongly across all application classes.
- ReplayIDS D2: OFRA has 12.60 points higher final accuracy, 20.07 points higher Macro-F1 and 3.13 points lower forgetting. Replay50 has higher balanced accuracy and attack recall, but its benign false-positive rate is 42.03%, versus 21.82% for OFRA. These are material operational weaknesses, not deployment-ready performance.
- Heartbleed has only two test examples. Replay50 recalls both but has precision about 0.055%; this is not reliable evidence of rare-attack generalization. Per-class precision, recall and support must be considered together.
- Malaya labels are application identities, not a benign-versus-attack taxonomy. Attack recall and benign FPR are not defined for this contract.

## Verification and provenance

| Item | Malaya | ReplayIDS D2 |
|---|---|---|
| Baseline Slurm job | 451049 | 451050 |
| Scheduler result | COMPLETED, exit 0 | COMPLETED, exit 0 |
| Elapsed time, excluding queue | 14m49s | 14m03s |
| Checkpoints / final test rows | 5 / 10,370 | 4 / 227,723 |
| Recorded W&B run | [e4gc1uvh](https://wandb.ai/csnet/ofra-etg-leon-hpc/runs/e4gc1uvh) | [349vilce](https://wandb.ai/csnet/ofra-etg-leon-hpc/runs/349vilce) |

Protected result/protocol hashes were checked against their checksum lists. The local result audit recomputed metrics from confusion matrices and task accuracy matrices. The paired comparison additionally checked exact OFRA hashes, matching dataset/seed/checkpoint counts, and matching final class identities, order and supports. Matching supports alone is not a row-identity proof: the existing immutable data contracts bind the split.

Machine-readable aggregate results and per-class comparisons are in `results/balanced-replay-seed1/`. The comparison utility is `tools/compare_verified_replay.py`; it requires the original protected baseline result, its companion checksum-bound files, the verified audit and the exact OFRA result as inputs. Aggregate exports alone do not contain those full original artifacts. Example invocation:

```text
python tools/compare_verified_replay.py AUDIT.json OFRA_RESULT.json --ofra-sha256 EXPECTED_HASH --output COMPARISON.json
```

Independent W&B cloud verification is pending; the run links above come from the protected run records and do not prove that every cloud panel or table is available. Local numerical verification does not depend on successful cloud access.

## Remaining limitations and next step

Keep the selected method, preprocessing, optimizer, model sizes and epoch counts unchanged for seeds 2–4. Combine all five paired seeds only after verification, and report seeds 1–4 separately because seed 42 selected the comparator. No confidence interval, p-value, state-of-the-art or publication-readiness conclusion is claimed from seed 1.

Balanced replay presents one replay row per new row in later epochs, so equal epoch counts do not imply equal compute. OFRA also retains router centroids, so equal exemplar counts do not imply equal total memory. This is a method-level comparison, not an isolated proof that any one OFRA component causes the difference. It adds no new SHAP/ETG result and does not replace existing manuscript headline evidence.
