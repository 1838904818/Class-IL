# HPC migration and full ETG pilot execution requirements

Updated 12 September 2026. Status: RESOURCE_PROFILE_VERIFIED; FIRST_EXTRACTION_CHUNK_UPLOADED_NOT_SUBMITTED. Full paired extraction and intervention efficacy remain unexecuted.

## Scope retained

Move execution to scheduled HPC resources; do not change classifier capacity or the scientific estimand merely because a larger GPU is available. Preserve native D2 ReplayIDS-aligned CIC-IDS-2017 training seed 1, checkpoint 000 to 001, the original 512-row class/shard contexts, float32 joint scoring and all 78 features.

The full descriptive experiment retains 896 role rows (32 trigger, 64 fit, 64 acceptance and 64 evaluation per class), 64 old-class explanation targets, 32 fixed Task-0 references and 8 attribution repeats per target/checkpoint. Keep all five selectors: audit-only, error-only, raw-drift, noise-aware and random-harmful, with a shared one-attempt calibration cap.

Primary outcome remains noise-aware minus error-only balanced old-class error on the 128 old-class evaluation rows. Preserve per-class confusion/recall/precision/F1/support, accuracy, Macro-F1, balanced accuracy, attack recall and Benign false positives. Old-class recall change over this single transition is not a full-stream forgetting estimate.

The candidate contribution is whether noise-adjusted explanation information improves selection of a fixed corrective action. SHAP, a ledger, detecting drift and ordinary calibration are not new algorithms by themselves. A negative result is retained. This retrospective calibration-pool pilot is not an independent capture or five-seed confirmation.

## Execution stages

1. Bind the existing remote data, checkpoint manifests, historical runtime and installed environment. Run full data hashes and model imports only inside an allocation. Login-node checks remain lightweight file metadata and environment metadata.
2. Submit a bounded GPU SHAP resource/fidelity profile under its own reviewed candidate. Preserve complete reference and feature budgets. Measure actual calls, reconstruction, CUDA/RSS, CPU/GPU utilization and walltime. A local 84-call component test cannot predict complete SHAP throughput.
3. On measured feasibility, freeze the full extraction resource request, repeat seed schedule and R1 numerical hyperparameters. Implement append-only per-target/checkpoint/repeat completion records and exact-hash resume; no repeated fitting/acceptance/evaluation access. A completed extraction chunk is reused only after independent verification, not by existence alone.
4. Fit the same R1 action for each registered selector; lock all fits before acceptance and all acceptance decisions before evaluation. Missing/invalid attribution yields unavailable comparison, not zero drift. Record rejected/failed attempts and actual spending.
5. Verify immutable raw outputs, W&B records and protected output checksums. Update paper/technical documentation and both verified repositories only with supported claims.

A same-batch continuation is permissible only if its complete implementation, resource/time gates and full-scope bindings have been reviewed in advance. This plan does not authorize recursive submission or let a successful profile silently trigger unreviewed stages.

## Resource position

### Verified scheduled profile

Job 456207 completed with exit 0 in 209 seconds. The one-target checkpoint-0
SHAP estimate took 112.80691397469491 seconds: 78 finite signed attributions,
63 nonzero, 3,088 native calls and 1,581,056 context-row forwards. The native
margin was 1.999900460243225, base value -0.18956404738128185 and reported
additive residual zero. This is one feasibility estimate, not an ETG efficacy
result, a full-population parity test or an independent scientific confirmation.

Peak CUDA allocation was 1,207,541,760 bytes; reservation 1,608,515,584 bytes.
Slurm step MaxRSS was 4,156,264 KiB (3.96 GiB), materially larger than the
853,176,320-byte polled worker RSS. Budget from the allocation-level evidence,
not worker RSS alone. CPU efficiency was 59.33%; later active GPU samples
were approximately 87-90%, not a whole-job average. W&B was disabled.

All nine protected artifact hashes matched their manifest. Result SHA-256:
`a7c4ddba4acdf5acc6c979efd0314770bc67159741a306e065936e1928880d51`.
Protected manifest SHA-256:
`20a9bd22db1ea93a822afdc1da59c5b000257e870faca8d3cec5bd71ea96d8ed`.

The first extraction chunk, candidate v2, is uploaded with all 21 files verified
remotely. Read-only validation passed on 12 September 2026 at 14:44 MYT, followed
by exact-candidate submission review approval. No job has been submitted; a
separate single-use execution confirmation and unchanged live conditions are
required. The allocation is one A100, two CPUs, 8 GiB and at most eight hours.

This chunk covers four predetermined targets at both checkpoints with eight
shared repeats (64 estimates). The full protocol remains 64 targets x two
checkpoints x eight repeats (1,024 estimates). The profile estimate is not reused
as a paired repeat. Exclusive records, independent commit verification and a
protected completion marker written last preserve incomplete work distinctly.

Local verification recorded 55 passed tests and one Windows symlink-permission
skip. Platform and tracking services were mocked. Upload, review and validation
do not establish runtime integration, native full-population parity, extraction
completeness or ETG efficacy. No full-extraction ETA is inferred from one target.

Candidate sbatch SHA-256:
`2c724a3e5f729c1818266ffbbdde29775794e7f151612f164d400eb64dea5fb2`.
Operation SHA-256:
`4bafc9589071ae7f0e76ae44ceebcf473841fa62ed87fcd76553a02a144f0af8`.
Upload manifest SHA-256:
`8d2abcd52f708d6d099c8e1ebd6af3ca25deed6e568297cb945f726c31159401`.

### Earlier local resource evidence

Use one node, one task and initially one GPU. A larger allocation is not justified by availability alone. The local component test used 1.126 GiB peak CUDA allocation, 1.498 GiB reservation and 0.929 GiB polled process RSS; it completed 84 calls in a 17.766-second supervised window. These establish a starting resource scale, not A100 throughput, SHAP peak memory or an approved Slurm request.

The prospective 5,144,704-call full extraction count is planning arithmetic, not a runtime forecast. Do not reserve a multi-day GPU job from that number without a real profile and reliable resume. Resource thresholds must follow current DICC publications and live limits; the former local desktop idle rule is not an HPC allocation rule.

## Historical resource-profile preparation record

The paragraphs below retain the preparation/submission history. The pending
snapshot is superseded by the verified completion above. Live account and
resource limits must still be rechecked before any subsequent submission.

Read-only SSH authenticated as the user's own account; account status FULL, walltime limit 3 days, association free with short/normal/long QoS, own queue empty at the check. These are snapshots and must be refreshed before submission.

Existing HPC environment metadata: SHAP 0.51.0, NumPy 2.2.6, PyTorch 2.6.0+cu118, tab-transformer-pytorch 0.6.1, W&B 0.23.0. psutil was absent from the inspected environment. Do not copy the Windows/psutil supervisor unchanged. No installation was performed.

Own native D2/source manifests, historical runtime and last-epoch seed-1 checkpoint directories exist. Directory existence and metadata do not establish full byte identity; scheduled validation remains required. No remote data were modified.

The Linux Slurm worker/supervisor and 35-file bound package are now prepared. Nineteen driver CPU tests and nine synthetic SHAP helper tests pass; Bash syntax and static preflight pass. The resource-only candidate preserves one 512-row native context, 78 features and 32 references. It requests one A100, two CPUs, 4 GiB host RAM and at most 20 minutes; these are bounded feasibility resources, not measured A100 throughput. W&B is disabled for this profile.

The candidate now normalizes PyTorch 2.6 GPU UUIDs, enforces the complete result contract, and handles TERM/INT with owned-worker cleanup and failure preservation. SIGKILL, node loss and unavailable storage cannot guarantee archival. Mocked CPU lifecycle tests do not establish Slurm integration or scientific validity.

Preparation review passed for candidate SHA-256 dd9ad6856378a0408a651f68ad48b3a4fa3e08e6807977fb2e79d08e6b9f1dc2. Following explicit upload authorization, all 35 files were transferred and verified remotely; the four dedicated directories are owned by the account holder with mode 700. Live checks on 11 September 2026 at 10:44 UTC showed FULL status, the expected short QoS and GPU partition capability, and an empty own queue. Remote static preflight and validation-only passed. After final review and explicit confirmation, Job 456207 was submitted once; immediate checks show PENDING (Resources). No model computation or scientific result is yet verified. Scheduled feasibility must still complete before full extraction and intervention evaluation.

## Current official policy source

DICC's July 24 resource-utilization notice is now available at:
https://www.dicc.um.edu.my/news/resource-utilization-fairshare-policies

It describes monitoring resource use and possible cancellation below 10 percent for one third of walltime; this is not a rule requiring a user's desktop GPU to be idle. Apply current policy conservatively and re-check the exact resource request before submission.
