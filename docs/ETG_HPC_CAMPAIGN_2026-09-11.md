# HPC migration and full ETG pilot execution requirements

Prepared 11 September 2026. Status: PREPARATION_ONLY; no upload or job submission.

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

Use one node, one task and initially one GPU. A larger allocation is not justified by availability alone. The local component test used 1.126 GiB peak CUDA allocation, 1.498 GiB reservation and 0.929 GiB polled process RSS; it completed 84 calls in a 17.766-second supervised window. These establish a starting resource scale, not A100 throughput, SHAP peak memory or an approved Slurm request.

The prospective 5,144,704-call full extraction count is planning arithmetic, not a runtime forecast. Do not reserve a multi-day GPU job from that number without a real profile and reliable resume. Resource thresholds must follow current DICC publications and live limits; the former local desktop idle rule is not an HPC allocation rule.

## Current live checks and unresolved items

Read-only SSH authenticated as the user's own account; account status FULL, walltime limit 3 days, association free with short/normal/long QoS, own queue empty at the check. These are snapshots and must be refreshed before submission.

Existing HPC environment metadata: SHAP 0.51.0, NumPy 2.2.6, PyTorch 2.6.0+cu118, tab-transformer-pytorch 0.6.1, W&B 0.23.0. psutil was absent from the inspected environment. Do not copy the Windows/psutil supervisor unchanged. No installation was performed.

Own native D2/source manifests, historical runtime and last-epoch seed-1 checkpoint directories exist. Directory existence and metadata do not establish full byte identity; scheduled validation remains required. No remote data were modified.

Still required before a submission-ready candidate: a Linux Slurm worker/supervisor and tests; mapped input/hash closure; exact measured profile resource request; reliable telemetry, failure preservation and protected-copy validation; W&B governance/record scope; local static preflight; current policy/live checks; exact original-gate approval and final user confirmation.

## Current official policy source

DICC's July 24 resource-utilization notice is now available at:
https://www.dicc.um.edu.my/news/resource-utilization-fairshare-policies

It describes monitoring resource use and possible cancellation below 10 percent for one third of walltime; this is not a rule requiring a user's desktop GPU to be idle. Apply current policy conservatively and re-check the exact resource request before submission.
