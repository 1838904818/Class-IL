# Executable L1 objective-control pilot

This is a working experimental runner, not only a schema. It trains paired
binary residual low-rank heads on actual supplied NumPy embedding arrays,
measures consumed rows and optimizer steps, checkpoints Adam state, evaluates
held-out data, and audits its emitted receipts. **Only tiny synthetic CPU tests
have been executed. No real-data training, GPU profile, institutional submission,
or baseline reproduction is claimed.**

The estimand is deliberately narrow: plain binary cross-entropy versus the
existing conditional focal weighting, under one identical frozen Task-0 encoder,
head initialization, target assignment and sampling schedule. There is no router,
encoder update, NME/replay architecture comparison, full OFRA reproduction,
hyperparameter optimization or causal architectural claim.

## Run and audit

Dependencies are Python, NumPy and PyTorch. The process fixes CPU worker/BLAS
thread counts to one and does not create data-loader workers or services. Use a
new output directory. All paths below are placeholders, not a resource request.

```text
python -B l1_runner.py validate-input inputs/manifest.json
python -B l1_runner.py run inputs/manifest.json --output results/pilot --device cpu --max-units 1
python -B l1_runner.py run inputs/manifest.json --output results/pilot --device cpu --resume
python -B l1_runner.py audit results/pilot --manifest inputs/manifest.json
```

`--max-units 1` performs one actual paired head-training epoch and atomically
pauses. It is an instrumented prefix, not a completed study. Its size must be
chosen from the authorized profiling envelope; one epoch is not promised to be
short on an arbitrary dataset. Resume requires unchanged input bytes, source,
configuration, environment and online-tracking policy. Each unit is one paired
head epoch or one task evaluation; no test-selected checkpoint is possible.

CUDA is implemented but untested here. It requires a supported device and an
explicit deterministic `CUBLAS_WORKSPACE_CONFIG` before launch. No GPU request,
HPC script, remote command or allocation recommendation is supplied. Real
institutional runs remain subject to current rules, independent exact-hash review
and fresh user confirmation for each stage.

## Input and embedding stage

[INPUT_CONTRACT.md](INPUT_CONTRACT.md) defines strict, hash-bound manifests.
`materialize_embeddings.py` is an executable independent stage for actual OFRA
Task-0 monitoring metadata plus its NPZ state. It checks checkpoint index 0,
loads only encoder tensors and frozen normalization, streams raw float32 rows
through the frozen encoder, checks that tensor state did not change, and emits
a complete L1 input bundle and measured export profile.

When a newly selected common Task-0 cohort has no matching trained encoder,
`pretrain_task0.py` implements that stage. It reads only the completed P Task-0
`transform_fit` collection, fits float64 population mean/std (constant features
use scale one), trains one encoder plus a temporary shared multiclass CE
classifier using Adam, and exports the encoder and normalization. No calibration,
omitted, future-task or official-test arrays are opened. Configuration and
initialization are prospective pilot choices, not historical OFRA pretraining
parity, optimized settings or a preregistration.

```text
python -B pretrain_task0.py sampling/P_00 --manifest-sha256 TASK0_MANIFEST_SHA256 --config task0_pilot_config.json --output common/task0 --evidence-kind real --device cpu --max-steps 1
python -B pretrain_task0.py sampling/P_00 --manifest-sha256 TASK0_MANIFEST_SHA256 --config task0_pilot_config.json --output common/task0 --evidence-kind real --device cpu --resume
python -B pretrain_task0.py sampling/P_00 --manifest-sha256 TASK0_MANIFEST_SHA256 --config task0_pilot_config.json --output common/task0 --evidence-kind real --audit-only
```

`--max-steps` limits actual optimizer steps in that invocation. Before the first
update, identity validation and normalization still require the full selected
training cohort; this option is not a promise that an arbitrary input is cheap.
Checkpoints at the configured cadence, pause and final step include model,
normalization, Adam state and committed-journal boundaries. Resume determinism
uses an epoch-derived permutation and a step-derived dropout seed. Model and
optimizer precision, shape, options and parameter identity are checked before
restoration; no silent dtype conversion is accepted.

The outputs `checkpoint.json`, `inference-state.npz` and `pretrain-cost.json`
feed the O/P bridge directly. Metadata binds the actual common cohort,
normalization fitting IDs and encoder training IDs, plus measured profile and
state hashes. `PRETRAIN_PROFILE.json` contains actual productive steps,
multiclass targets, raw-row exposure, unique rows, tensor storage and process
memory; `COST_LEDGER.json` retains discarded and uncertain work separately.
The temporary multiclass classifier is retained in recovery checkpoints but
is not reused as an L1 binary head. No training metric selects a checkpoint.

```text
python -B materialize_embeddings.py raw/export-manifest.json --output inputs/materialized --device cpu --batch-size 512
```

MLP export is self-contained and tested on 18 synthetic rows. FT-Transformer
export additionally needs `--runtime-root` pointing to the exact hash-pinned
OFRA runtime recorded by the adapter. Its upstream dependency validates version
and source hash. The real FT path and historical numerical parity have **not**
been executed or certified here. No training is performed by this exporter.
Its source is equivalent to the encoder-only portion of the existing monitoring
loader, not a call that unnecessarily instantiates historical heads/routers.

The prospective sampling builder's `iter_partition(..., partition="fit")`
provides raw `x`, `labels`, `row_ids`, `group_ids` and `available_tasks` chunks.
Those selected increments must be materialized with the same frozen encoder,
while their calibration partition stays out of L1 fitting. Official held-out
test inputs must be separately bound. Never feed raw features into an embedding
field or claim a row-identity proxy establishes capture-independent evaluation.

## Measured evidence, not allocations

Each successful optimizer step and actually consumed batch is durably journalled
with original-row identifiers, binary targets and array indices. Paired exposure
digests and counters must be identical. The auditor reconstructs the deterministic
sampler from the bound input arrays, checks committed journals, recomputes
confusion-derived metrics, and verifies final state/checkpoint/result hashes.
Counters are never inferred from requested epochs or a replay capacity.

Storage accounting deduplicates Torch `untyped_storage` across tensor views.
It measures head parameters, gradients, Adam state and current-batch tensor
storage, plus shared input-array logical bytes and arrived replay-index bytes.
Sampled tensor storage is **not** an exhaustive activation peak. Actual OS
process-lifetime RSS high-water and current-RSS samples cover the process; CUDA
allocated/reserved peaks are separate when used. The loader may temporarily
load encoder tensors to verify their state hash; they are not resident during
head training. Encoder pretraining and embedding materialization are separate
cost scopes and are not silently charged as zero total method cost.

Both objective arms coexist and run sequentially in one process. Hence the
process memory peaks and timings describe a **paired harness**, not isolated
single-method deployment profiles or proof of equal hardware time. The initial
parameter capacity and actual exposure/update schedule are matched; focal's
additional arithmetic is measured rather than declared equal. No method-level
memory/latency advantage follows from these records.

## Failure, recovery and output safety

The output directory is exclusively created; an exclusive writer lock prevents
concurrent writes. Checkpoints are immutable files with a SHA-bound atomic
`LATEST.json` pointer. A resume restores the last complete paired epoch and its
Adam state. Per-attempt commit receipts reside **inside** the checkpoint, so a
crash after the checkpoint pointer commits cannot lose the attribution of work.
Failed/uncommitted work remains in `COST_LEDGER.json`. A crash between a durable
step-start and step-completion receipt is an explicit uncertain upper bound,
not an invented completed update. Hard-kill tails can remain unknown.

A hard kill can leave `WRITER.lock`. The runner never automatically deletes an
existing lock or resumes over it. Confirm that no writer remains, retain the
attempt logs, then have the account owner remove only that stale lock before
resuming. A failed embedding-export stage is not reused: its failure receipt
records elapsed cost, and a new explicitly chosen output is required.

All original array/artifact hashes and the manifest are rechecked before an L1
completion or bounded pause. The pretrainer and exporter recheck their selected
inputs and bound implementation files before completion. These are integrity
checks at execution boundaries, not a substitute for immutable input storage.
Observed pretraining costs include all recorded attempts; an unknown crash tail
is explicitly flagged as a lower bound, never silently described as an exact
total. Training elapsed observations end after final input validation; final
receipt serialization and scheduler-billed time require the external job profile.

`COMPLETE.json` is published last and binds `result.json`, the cost ledger and
the final checkpoint. `audit` returning `AUDIT_PASS` means local artifact,
sampler/accounting and confusion-metric consistency; it is not independent
scientific replication, data-governance approval or institutional permission.
Keep real journals, raw-row identities, data and checkpoints in controlled
storage. Publish only approved non-sensitive aggregates and source artifacts.

## Optional tracking

W&B is disabled and not imported in the default execution path. It requires both
the manifest's aggregate-governance flag and explicit `--wandb-project NAME
--allow-online`. No credentials are accepted as CLI arguments. The adapter
disables source/system/host capture and emits aggregate checkpoint metrics,
signed forgetting, per-class statistics and confusion tables only. It returns
run identity/URL with `client_finished_remote_verification_not_performed`:
client finish is not cloud-side verification. The adapter has only mocked tests;
no real online write was made.

## Local verification and remaining boundary

```text
python -B -m unittest discover -s tests -v
python -B synthetic_smoke.py --output synthetic_evidence
```

The smoke generator makes 12 fitting rows and 6 evaluation rows, then executes
the actual training and audit path. Small tests cover checkpoint/resume hash
equivalence, counted failed work, real MLP export, state/input tampering,
focal-threshold semantics, storage-view deduplication and offline defaults.

`pilot_config.json` contains prospective pilot defaults, not optimized values or
a preregistration. The remaining substantive work is to select and bind approved
real inputs, execute the shared Task-0 stage when needed, materialize actual
embeddings, execute and inspect measured resource prefixes, and then execute the
registered pilot. All three cores are implemented; no real-data run or resource
request is implied by these local tests.
Historical score-reconstruction/attribution-faithfulness gates remain separate.
