# Historical GPU fidelity: implementation and execution boundary

Status on 7 September 2026: **v6 uploaded, validated, independently approved
and submitted once as Job 453142. First observed state: PENDING (Resources).
No numerical-fidelity result is established**.
See the [bounded profile specification](REPLAYIDS_FIDELITY_PROFILE.md) for the
later tracking, failure-preservation and resource controls. The 91 remote input
hashes now match the same binding, as recorded in the
[read-only closure check](../results/replayids-gpu-fidelity/REMOTE_INPUT_CLOSURE_20260907.json).
Input identity is not a score-fidelity result.

## Why this check is needed

The [canonical CPU reconstruction](REPLAYIDS_CANONICAL_FIDELITY.md) checked all
20 snapshots, but 15 later-checkpoint Router/Joint comparisons exceeded the fixed
score tolerance. Two prediction changes were the same sample at two checkpoints.
These are numerical reconstruction findings, not new accuracy, forgetting,
attribution drift or ETG results. Historical primary results remain unchanged.

The original protocol records PyTorch 2.6.0+cu118, NumPy 2.2.6, Python 3.11.9,
CUDA 11.8, cuDNN 90100 and an A100-SXM4-80GB. Read-only package metadata inspection
still found the first two versions and tab-transformer-pytorch 0.6.1 in the
research environment. Installed metadata alone does not prove which binaries
will load, that CUDA works, or that the historical computation is reproduced.

## Newly recovered numerical controls

The original hash-bound `streaming_full/validation.py` implements `_seed_process`.
With deterministic execution enabled it explicitly applies all of the following:

| Control | Historical source setting |
|---|---|
| Deterministic algorithms | enabled |
| cuDNN deterministic mode | enabled |
| cuDNN benchmark/autotuning | disabled |
| CUDA matrix-multiplication TF32 | disabled |
| cuDNN TF32 | disabled |
| Float32 matrix-multiplication precision | highest |
| CUBLAS workspace configuration | `:4096:8`, recorded by the protocol |
| OMP/MKL threads | 2, recorded by the protocol |

The earlier CPU diagnostic enabled deterministic algorithms but did not recreate
every explicit GPU backend flag. It remains useful cross-environment evidence;
it must not be relabelled a fully environment-matched historical check. On CPU,
the omitted CUDA flags are not evidence that TF32 caused the observed mismatch.
The fixed-embedding arithmetic experiment separately identified Router arithmetic
sensitivity; the future GPU run is needed to test reproduction in its original
software/device context.

TF32 is a GPU arithmetic mode with different precision from strict float32.
Disabling it reproduces the original source setting; it is not a new model,
training method or accuracy improvement. The GPU model alone is insufficient to
establish numerical comparability.

## Prepared implementation

`tools/check_replayids_gpu_fidelity.py` is a separate verifier, not a modification
of the historical scorer or the immutable CPU report. It reuses the byte-bound
canonical scoring helper and invokes the original `_seed_process` implementation.

- Before scientific imports or input hashing, require a scheduled Linux compute
  step for the authorized account, one node/task, two CPUs and one visible GPU.
  Reject login nodes and CPU fallback. These guards supplement, not replace,
  independent scheduler/submission review.
- Verify the binding digest, verifier/helper source digests and every declared
  input. Resolve paths inside explicit roots and reject traversal/symlink escape.
- Stage a new clean copy of the 16 historical source files in scratch. Do not
  import old bytecode or additional mutable modules beside the original runtime.
- Compare the actually loaded Python/platform, torch, NumPy, CUDA, cuDNN, GPU and
  numerical controls with the recorded historical values. Verify the upstream
  FT-Transformer source hash using the historical adapter.
- Abort before probe inference on an environment mismatch. Report the mismatch,
  not a substitute CPU run or a tolerance-adjusted pass.
- Reconstruct the original encoder chunks of 512 and whole-probe Head/Router
  calls for all five seeds and four checkpoints. Keep `atol=1e-4`, `rtol=1e-5`
  and zero-argmax-change acceptance; preserve failures and partial reports.
- Record score and prediction hashes, mismatching sample hashes and margins,
  elapsed time and peak CUDA allocation/reservation. Do not fit a calibrator or
  estimate full-test performance on these probes.

The [input binding](../results/replayids-gpu-fidelity/GPU_FIDELITY_INPUTS.json)
covers **91 files / 427,738,339 bytes**: 16 original source files, five original
protocols, 60 checkpoint artifacts, one shared probe manifest, one data manifest
and eight official-test shards. All local files were checked against the protected
registry or its bound manifests before the binding was generated. Full test shards
are verified to recover exact probe rows; this is not a new full-test evaluation.
No traffic arrays, model weights, credentials or personal filesystem roots are
included in this public binding.

## Limits that remain explicit

`recorded_environment_matched` means only the recorded fields and recovered
source controls match. `historical_environment_identical_proven` remains false:
the old record did not pin the NVIDIA driver, BLAS build/instruction dispatch,
every binary hash or per-operation kernel selection. These cannot be invented
retrospectively. Even a future score-tolerance pass is not bit-exact equality.

Fourteen local contract tests were run: thirteen passed; the symlink-escape test
was skipped because this Windows host could not create the required symlink.
Tests cover allocation rejection, software/TF32 drift, fixed batching/tolerance,
path traversal, input mutation and failure-report preservation. They do not run
the GPU, simulate the real scheduler, measure utilization or prove full parity.
The existing canonical scoring tests separately exercise the encoder versus
whole-probe batching split.

## Operational implementation and release boundary

The separate coordinator implements resource sampling, conservative stopping,
online aggregate-only tracking and protected copying/checksums, including
partial-failure evidence. Its mocked/local tests do not establish GPU integration,
resource adequacy or live W&B delivery. Raw probe rows and sample identifiers
remain outside tracking; the verifier itself has no network client.

Remaining release gates are exact staged-file closure, reviewed log/output
paths, fresh live environment/limit checks, independent approval, and exact-action
user authorization. Training memory is context, not a measurement of this
inference-only workload. A bounded first profile must not be described as a
previously measured allocation or padded with dummy work to raise utilization.

No new GPU result or W&B run exists for this candidate. Stage B arithmetic,
full-test Stage C, calibration and score-specific SHAP/ETG validation remain
outstanding under the [staged protocol](REPLAYIDS_SCORE_VALIDATION_PROTOCOL.md).

## Refreshed official references

The former dated DICC URLs now return 404; the relocated official news pages
were checked on 7 September 2026. The [utilization policy](https://www.dicc.um.edu.my/news/resource-utilization-fairshare-policies)
retains the under-utilization cancellation rule. The [temporary GPU announcement](https://www.dicc.um.edu.my/news/technical-notice-temporary-gpu-allocation-adjustment-%F0%9F%93%A2)
still states two GPUs per job; its body contains an older effective date, so live
limits and newer directions remain necessary. A [7-9 September minimal-operations notice](https://www.dicc.um.edu.my/news/dicc-operating-notice-minimal-operations-from-7-9-september-2026)
warns of possible support/submission delays; it is not evidence of a cluster shutdown.
