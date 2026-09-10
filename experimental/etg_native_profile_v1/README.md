# Native-context component profile, version 1

Prepared 11 September 2026. **Synthetic-tested candidate, not a real-data run or
independent execution approval.** This is a feasibility gate for the separately
versioned [descriptive ETG pilot](../etg_exploratory_v1/PROTOCOL.md), not its full
five-arm study or a new scientific result.

## Fixed workload

Select the first Benign and last DoS GoldenEye trigger row from the frozen native
D2 plan, by identity rather than performance. Their original calibration contexts
contain 512 and 105 rows. At both seed-1 checkpoints 000 and 001:

1. Check two identical native calls and same-forward binary-logit capture.
2. Check six replacements: unchanged row, all-first-reference, all-last-reference,
   first feature, last feature, and alternating features. Each has two direct
   calls and one wrapper call; all companion rows remain fixed.

Total: 84 native calls and 24 perturbation records. Use the frozen 16 native
Task-0 fitting references per old class, shared across checkpoints. Read-only
memory maps verify each selected whole context against reconstructed source
offsets. This does not verify every row or create a new dataset.

## Evidence boundary

No SHAP values, drift rates, calibration parameters, ETG decisions, accuracy,
forgetting or governance benefit are computed. The original checkpoint loader
validates archived probe files internally; these do not select targets or tune
the profile. Companion features can belong to other registered roles, and
class-stratified files reveal labels. This is outcome-unused engineering work,
not cryptographic blinding or fresh confirmation.

Historical code is bound, but local packages/GPU differ from the historical A100
environment. Exact direct-repeat/adapter agreement is **not archived-score
parity or full-population parity**. No tolerance relaxation, alternate adapter,
dataset substitution or automatic CPU fallback is allowed. First mismatch stops
the profile and is recorded. Full SHAP extraction needs its own subsequent gate.

## Local abort controls

| Setting | Fixed limit |
|---|---|
| Native calls / native window | 96 / 120 seconds |
| Supervised worker lifetime | 180 seconds |
| Polled worker-tree RSS | 8 GiB |
| CUDA caching allocator | 3 GiB; driver/context memory is additional |
| GPU start conditions | GPU 0 free memory at least 4096 MiB; utilization at most 10% |
| CPU threads | 2 BLAS/PyTorch; 1 inter-op; one model resident at a time |

These are local abort settings, not measurements, scientific hyperparameters or
DICC allocation rules. RSS is polled, not an OS hard limit. A direct child
`conhost.exe` from the Windows system directory is counted as console overhead;
any other spawned descendant stops the worker tree. The supervisor is bounded,
not a persistent agent. Host/Slurm guards forbid use as an HPC launcher.

No training, fitting, W&B/network client, remote submission, server, overwrite,
retry or resume is provided. Existing run directories remain spent.

## Build, validate, then obtain review

The builder only reads metadata and hashes files. It prints machine-specific
paths: archive that candidate privately, never in the supervisor-facing repo.
The input layout refers to archived research files, not a new public raw-data
download supplied by this repository.

```text
python -I -B experimental/etg_native_profile_v1/build_candidate.py --workspace-root <workspace> --output-parent <existing-private-directory>
python -I -B experimental/etg_native_profile_v1/local_profile.py --candidate <candidate.json> --sha256 <candidate-sha256>
```

The second command is validation-only. Execution additionally needs
`--execute --approval <review.json> --approval-sha256 <review-sha256>`.
The actual independent review must bind candidate SHA-256, run ID, output path,
reviewer reference and scope `local-component-profile-only`, with verdict
`APPROVED`. A JSON field is not a digital signature or proof of approval:
the operator must verify the independent review. This code grants no approval.

`PROFILE_TERMINAL.json` is authoritative. A partial result, progress or zero exit
code is insufficient. The terminal record hashes artifacts and preserves
failure, elapsed time and peak polled RSS. Even success does not authorize full
extraction, scientific claims or an HPC submission.

## Software tests

```text
python -B -m unittest discover -s experimental/etg_native_profile_v1 -p test_profile.py
```

Synthetic arrays and bounded synthetic child processes test full/tail contexts,
source-offset integrity, mutation, missing review, partial outputs, exact call
counts, first mismatch, time/RSS aborts and console-helper identification.
No research tensor inference is established by these tests.
