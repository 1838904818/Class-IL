# ReplayIDS D2: holdout and inference-state readiness

Updated 7 September 2026. This is a verified input-readiness audit, **not a new model result or successful calibration experiment**. The relevant primary remains last-epoch OFRA from Job 425539, as distinguished from guarded checkpoint selection in the [scoring diagnostic](REPLAYIDS_PRIMARY_SCORING_DIAGNOSTIC.md).

## Verified inputs

- The fixed D2 manifest, sampling audit, build summary, derived checksum registry, original v2 builder and its helper match the previously bound SHA-256 values.
- All eight class-specific random permutations were independently regenerated from the original master seed, source-manifest hash and class ID. Fit and calibration index digests match the recorded digests exactly, and every index intersection is empty.
- All eight materialized calibration arrays were retrieved read-only and match their recorded hashes, row counts and 78-feature shapes. Values are finite.
- The training manifest includes only `train_fit.npy`. All five primary run protocols list exactly the corresponding eight fit-shard hashes and have checkpoint selection `last` with training calibration disabled. Existence of a reserved calibration partition is not evidence that a calibrator was previously fitted.
- Final checkpoint 003 for seeds 1, 2, 3, 4 and 42 was retrieved read-only from protected storage. All ten state/manifest files match the original protected registry. Every stored tensor is numeric and finite, normalization scales are positive, and the state members agree with the manifest schema.

## Exact dataset split

Each dataset is processed independently. The table below concerns only the ReplayIDS-aligned CIC-IDS-2017 D2 derivative. Training seed changes do not resample this fixed seed-42 data split.

| Class | Fit rows | Calibration rows | Official test rows |
|---|---:|---:|---:|
| Benign | 124,780 | 52,326 | 174,421 |
| DoS GoldenEye | 5,559 | 617 | 2,059 |
| DoS Hulk | 124,780 | 13,864 | 46,215 |
| DoS Slowhttptest | 2,970 | 329 | 1,100 |
| DoS slowloris | 3,130 | 347 | 1,159 |
| FTP-Patator | 4,287 | 476 | 1,588 |
| Heartbleed | 6 | 1 | 2 |
| SSH-Patator | 3,185 | 353 | 1,179 |
| Total | 268,697 | 68,313 | 227,723 |

The builder first reserves approximately 10% of each source-training class for calibration, with a minimum of one held-out row when a class has more than one source row. Only the remaining Benign fit pool is capped, to the largest attack fit pool (124,780). Attack fit pools are not capped or oversampled. The official test partition is not balanced or sampled; the unchanged manifest and historical materialization audit bind its original shard hashes. This check did not reread all official-test arrays.

This is an **offline benchmark construction**: the adaptive cap uses all attack-class counts. Task-0-only normalization does not imply that the entire dataset-construction policy is a strictly prospective online algorithm. Disjoint source indices also do not prove absence of duplicate feature content across partitions. Neither distinction should be omitted from methodology claims.

## What can and cannot be reused

The five final snapshots include encoder weights, eight family heads, normalization state and cap3000 centroids/counts. They support investigating reconstruction of the existing head/router/joint scorer without first retraining the models. Approximately 88 MB of state files were retrieved in total; raw traffic and weights are not included in the public evidence package.

File integrity is not proof of forward-score equivalence. The following gates remain open:

1. Reconstruct the registered scorer and compare against checksum-bound saved probe scores using the same inputs and explicit numerical tolerances. Do not fit calibration until this parity check passes.
2. Recover or compute head/router scores and immutable row identifiers on the training-only calibration arrays. No such new score tables were generated in this audit.
3. For class-incremental evaluation, restrict each checkpoint to classes already observed. Future-class calibration rows must not tune earlier checkpoints.
4. Retrieve and verify checkpoints 000-002 as well before reporting any revised forgetting trajectory. Only checkpoint 003 was retrieved and tensor-checked in this audit.
5. Specify a rare-class fallback and report calibration support. One Heartbleed calibration row cannot establish per-class calibration or a reliable threshold; repeated seeds do not supply additional independent rows.
6. Lock the fitting objective and safety constraints using training-only information before accessing official-test labels for a new candidate. Previously reported test diagnostics cannot turn a subsequent fitted candidate into a preregistered experiment retrospectively.

## Reproduction and evidence boundary

The public [READINESS.json](../results/replayids-score-readiness/READINESS.json) contains counts, hashes, tensor shapes and explicit incomplete gates. The audit utility performs no model inference or training, and makes no network calls. It requires separately authorized local copies of the original protected state files, calibration arrays, sampling metadata and run protocols:

```text
python tools/audit_replayids_score_readiness.py --input INPUT_DIR --protected-registry REGISTRY_FILE --protocol-root PRIMARY_PROTOCOL_DIR --output readiness.json
python tests/test_replayids_score_readiness.py
```

The lightweight tests reproduce published index digests and check split edge cases without requiring raw traffic or model weights. They do not replace the full input audit. No new accuracy, Macro-F1, forgetting, SHAP, ETG, or W&B cloud-completeness result follows from this readiness report.
