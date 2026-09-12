# ETG extraction status and resource-profile protocol

Status as of 12 September 2026: **first extraction chunk verified; governance efficacy remains unevaluated.**

## Verified extraction scope

The first scheduled chunk (Job 456373) completed with exit code 0 in 2:19:38. Its protected manifest covers 585 files; all listed hashes, 64 record/commit pairs, and completion links were checked. The full copied evidence contains 588 files; the additional three are the manifest, finalization, and authoritative completion marker.

| Scope | Value |
|---|---|
| Training seed | 1 |
| Old classes represented | Benign; DoS GoldenEye |
| Distinct target rows in the completed chunk | 4 (two per class) |
| Checkpoints per target | 0 and 1 |
| Attribution permutation seeds per target/checkpoint | 20270000–20270007 |
| Completed explanation records | 4 × 2 × 8 = 64 |
| Full planned extraction | 64 targets × 2 checkpoints × 8 seeds = 1,024 |
| Remaining records | 960 |
| Features | 78 |
| Fixed reference pool | 32 Task-0 fitting rows |
| SHAP method | PermutationExplainer 0.51.0, exact-equality masker |
| max_evals / SHAP batch_size | 157 / 1 |
| Native scoring context | Original 512-row batch or 105-row tail batch |

The eight attribution seeds are **not eight independent training seeds**. All 64 stored vectors are finite; additive reconstruction is exact for these records. Reconstruction alone does not establish explanation quality, historical full-population scorer equivalence, a statistical certificate, or governance effectiveness.

The historical batch-dependent scorer and original parent context remain unchanged. Full-population parity is not asserted by this extraction. No ETG action has been fitted or evaluated in this chunk.

## Resource observations and the next limited test

Slurm reports 7.965 GiB peak memory against 8 GiB requested. Worker-only sampled RSS peaks at 0.844 GiB and does not cover whole-step accounting. No cgroup memory fields were present in the old telemetry. The source of the accounting peak is unresolved.

Of 549 valid GPU samples, 200 are below 10%. The low observations form short repeated segments; approximately 31.36% of the extraction-stage elapsed time is outside each record's extract_one timer. This supports investigating repeated initialization, but does not identify import, hashing, data reconstruction, and model-loading costs separately. The timers do not measure pure GPU compute time.

A new **unexecuted** short profile is prepared to replay only ordinals 0, 1, 8, 9, 48, 49, 56, and 57. These cover both old classes, checkpoints 0/1, two attribution seeds, and full/tail contexts. It uses one sequential persistent interpreter while recreating the model, target, budget, and explainer for every record. Python/NumPy/Torch RNGs are reset per record. The process-start Python hash seed cannot be reset; that lifecycle change is explicitly under test.

All scientific payload fields must match their archived counterparts exactly. Only elapsed time and two CUDA allocator peak fields may differ. A mismatch stops the profile and is retained for inspection. Replayed records will not increase the 64/1,024 coverage count or replace originals.

## New monitoring controls

The pure resource-monitoring implementation and synthetic tests are in experimental/etg_chunk_extraction_v1/resource_guard.py and test_resource_guard.py.

- Read only the proven current Slurm step's cgroup v1/v2, including descendants; missing or ambiguous whole-step evidence fails closed.
- Stop at 90% of the bound memory allocation.
- Retain rolling sustained-low checks and add independent cumulative CPU/GPU/memory checks.
- After 300 seconds and at least 240 measured seconds, potentially low utilization of 30% or more stops the profile.
- Startup unknown time, bounded to at most 60 seconds, contributes conservatively to the low-fraction upper bound; it is not reported as observed utilization.
- Subsequent measurement gaps over 30 seconds fail closed.
- Final resource checks and hash-linked tail evidence are required before the completion marker.

These numerical guards are project implementation choices. They supplement the published [DICC resource policy](https://www.dicc.um.edu.my/news/resource-utilization-fairshare-policies), which permits cancellation for underutilization. They do not establish policy compliance merely by passing unit tests.

The proposed profile requests one A100, two CPU cores, 12 GiB host memory, and at most 30 minutes. The memory request is a conservative profile margin based on the observed peak, not a measured permanent requirement. Neither performance improvement nor real cgroup compatibility has yet been demonstrated.

## Integrity identifiers

| Artifact | SHA-256 |
|---|---|
| Original SCIENCE.json | 1855e2feedd068dd56ca8ca68247519ecd34cbe46b38a5214736de80c75177e5 |
| Original TARGETS.json | 459d960f3bf8876c856b09fa933e36b7d8dfcd0b23cafecb80a93f0b0fe26e2a |
| First-chunk protected manifest | 2e97498c2542d2f1c4f2767e436d39a2d0359fbcfab196821babf0b823f80765 |
| First-chunk completion marker | febf292e1b2c3873bcba497baa29b4620c4b2de18047f62cda2dcff7075ee119 |

This is an extraction and reproducibility update. It does not establish a new algorithmic contribution, attribution robustness, or ETG intervention benefit.
