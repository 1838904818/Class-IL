# Bounded historical-score fidelity profile

Status: v5 validation-only parsing failed; the v6 repair is uploaded and
live validation passed. Following independent review and one fresh confirmation,
v6 was submitted as Job 453142. Its first queue/accounting state is PENDING
(Resources); submission is not execution or a numerical pass. See the
[account-status regression note](DICC_ACCOUNT_STATUS_VALIDATION.md).
No GPU pass, new model result, repair effect or live tracking run is reported here.
This is a self-contained numerical-fidelity gate. It authorizes no subsequent
experiment or automatic submission.

## Scientific scope

The verifier reconstructs the original frozen ReplayIDS probes at all 20
seed/checkpoint combinations (seeds 1, 2, 3, 4, 42; checkpoints 0--3).
It keeps the original 512-row encoder batching, whole-probe score calls,
deterministic settings, input binding, tolerances and zero changed predictions.
It does not train, select a model, fit calibration, compute new test accuracy,
or evaluate SHAP/ETG utility. A partial result cannot pass.

The 91-file input binding remains
`71e90f75c597268bfd941025edc8ee6bb91ca403c297287502f9f31bbea3b415`.
The coordinator does not modify the bound verifier or historical source files.
Matching the recorded environment fields cannot recover unrecorded historical
driver, BLAS, binary or kernel identities. A tolerance pass is not bit equality.

## Fixed decision rule

Score-fidelity PASS requires the exact code/input bindings to validate, all
recorded environment controls to match, and exactly one result for every one
of the 20 declared seed/checkpoint pairs. Every Head, Router and Joint score
cell must meet the existing `atol=1e-4, rtol=1e-5` comparison, with zero changed
predicted classes. The original scorer, seed set, evaluation support, batching
and tolerances must not be altered after inspecting this result.

Any missing, duplicate, nonfinite, mismatched or out-of-tolerance result fails
the gate. Environment mismatch, timeout, missing report and a partial run are
not passes. Preserve failure evidence and diagnose it; do not promote a scorer,
claim attribution validity or relax thresholds to rescue a pass. A score pass
only permits proposing a separately bound next experiment: it does not itself
establish bit equality, full-test performance, explanation faithfulness,
governance utility, superiority or novelty. The historical primary result table
remains unchanged. Any follow-up still needs independent review and new approval.

Operational completion is evaluated separately: intact protected hashes,
resource compliance and verified remote tracking are required in addition to
the scientific result. A client-side tracking receipt alone does not prove
remote delivery. Failed science must remain reportable even if file preservation
and tracking succeed; successful science does not erase a tracking failure.

## Execution envelope and honest resource uncertainty

The proposed first profile requests one A100, one node/task, two CPUs, 4 GiB host
RAM and ten minutes. This is a bounded measurement proposal, not a claim that
this inference workload has already been measured. It requires independent
resource review and fresh user approval. The earlier 13-GiB training allocation
does not justify this inference allocation. If review requires a different
envelope, all affected bindings, tests and approvals must be regenerated.

One foreground coordinator launches and waits for one verifier subprocess.
There is no fan-out, nested submission, installation, listener, training or
dummy workload. The verifier has an eight-minute deadline; the remaining two
minutes are reserved for stopping the child, saving evidence and tracking
finalization. A deadline is a cap, not a predicted completion time. A longer
follow-up requires diagnosis and separate approval; there is no automatic retry.

Every five seconds the coordinator reads utilization for the allocated GPU
only and CPU/RSS of its own process tree. A numeric CUDA ordinal may be remapped
inside a cgroup: telemetry uses a supplied UUID or matching Slurm global job/step
GPU identifiers, never assumes CUDA ordinal 0 is physical GPU 0. Missing or
ambiguous identifiers fail closed. See the official
[Slurm GRES guide](https://slurm.schedmd.com/gres.html) and
[step environment definitions](https://slurm.schedmd.com/srun.html).

The project guard stops a still-running verifier after any monitored CPU, GPU
or allocated-host-memory utilization remains below 10% continuously for 200
seconds, at 90% host-memory use, or after three consecutive telemetry/tracking
failures. It supplements rather than reproduces DICC's accounting decisions.
Summed RSS can double-count shared pages, and exited descendants can make CPU
tick deltas undercount activity. Short jobs or no samples provide limited/no
utilization evidence, not an invented high-efficiency pass. Compare with the
job's own accounting and `seff` after execution. No unrelated process or device
inventory is collected.

## Tracking and evidence

Tracking is online, aggregate-only, in the existing research project. Metadata,
machine information, source/git capture, requirements capture, automatic system
statistics and console upload are disabled. The allowlist contains checkpoint
progress, mismatch counts, score-error maxima, elapsed time and measured resource
usage. No raw rows, sample identifiers, model files or protected mismatch report
are uploaded. Initialization failure stops before scientific inference.

The SDK version is bound to 0.23.0 and checked from installed package metadata
before importing it. Environment-provided tracking keys are rejected; the
standard authentication file must be an owned regular mode-0600 file, checked
without reading its contents. The SDK receives a minimal environment with no
Slurm, username or hostname variables and a neutral argument list. Allocation
variables are passed only to the bound verifier. Early viewer collection is
disabled. A run name contains only a fixed prefix and operation-hash prefix;
exact config/history allowlists reject private names, paths, IDs and error text.
The SDK necessarily receives its local output/authentication directory, but
metadata/code/console collection is disabled and those paths are not metrics.

Core protected evidence comprises the available verifier report, operation
status, resource samples, tracking URL, and verifier stdout/stderr. Files are
copied to a new target only, checked by SHA-256 and capped at 16 MiB each. Missing
mandatory operation evidence, symlinks, a pre-existing destination or a copy
mismatch prevents the completion seal. Scientific failure reports are retained.

Core files are sealed **before** network finalization. A subsequent, separately
hashed tracking-finalization receipt binds the core manifest. A missing receipt
means finalization is unknown. A client finish return is not independent proof
that every metric reached the remote service; verify that service read-only.
Scientific score validity, protected-copy integrity, resource compliance and
tracking completion are separate outcomes. A good score report survives a later
tracking failure, but that failure still returns a nonzero operation exit code.

## Local validation and remaining release gates

The coordinator has 23 synthetic tests: 22 passed on Windows and one symlink
test was skipped because the host does not permit creating it. Tests include
complete versus duplicate/missing checkpoint coverage, tracking initialization
and finish failures, protected-copy hashes, own-process selection and remapped
GPU identifiers. SDK/process integration is mocked; no GPU or Slurm execution
is implied. Existing verifier and canonical-scoring tests remain separate.

The staged package is generated from the exact package manifest, without test
caches, bytecode, symlinks or extra directories. A shell-level allowlist rejects
extra operation files before Python starts. Scheduler logs use absolute paths
outside that immutable package. The candidate-specific submission guard is
validation-only by default, accepts only its absolute sbatch path, and checks
the approved hash again immediately before a separately authorized submission.

Before execution: verify the remote input closure and environment; review the
exact code/data/protocol/sbatch hashes, allocation and live limits; authorize
staging and output parents; obtain the independent submission verdict; then
obtain fresh exact-command user confirmation. Nothing in this document grants
remote write or submission authority. Other scientific criticism remains open
until its corresponding experiment passes or the associated claim is removed.
