# Executable accounting contract v1

The authoritative executable schema is `validator.py`. Unknown/missing keys are
rejected for executable records. The human-readable template is intentionally
incomplete and cannot enter result validation. The unit-test fixture generator
is a small complete example, labelled `synthetic` at protocol and report level.
There is no real-data passing example.

## Bound protocol

`status=BOUND`, dataset, unique seeds, disjoint class tasks, cumulative available
labels, all named `locks` SHA-256 fields, and initial/shared-Task-0 encoder hashes
per seed are mandatory. Locks are content bindings, not attestations that a
method is correct. The split registry is supplied and checked locally; other
locked artifacts are also hash-verified locally, but their substantive correctness
requires separate human/source review before real use. `method_specs_sha256` binds
each arm's loss, target assignment, sampling, trainable parameter policy, optimizer
schedule, checkpoint policy and initial head-state hashes per seed. These policy
strings must describe the actual implementation, not desired behaviour.
Shared pretraining must use the same trace and measured attributable cost for all
arms of a seed; the validator rejects different pretraining evidence under a
shared encoder label.

Each arm declares an ordered `phase_plan`. Every arm has `shared_pretrain` at
task 0 with multiclass training plus at least one phase covering every task.
Multiple phases per task permit separate family training and router fitting.
Each phase fixes `task`, `gradient`, `target_kind` (`binary`, `multiclass`, or
`none`), `device` (`cpu` or `cuda`), and numeric `limits` for every counter and
memory metric. No implied equality follows from limits. Limits are inclusive
upper bounds; successful training has at least one observed optimizer update.
Non-gradient phases require `target_kind=none` and a nonempty `na_reason`.

Each `contrast` specifies two known arms, an estimand, residual confounds and
exactly matched metrics. Matching is checked within each seed over the complete
run: counters sum across phases except unique raw rows, which use the union of
original ids; memory metrics take the maximum across phases.
No-gradient and gradient arms cannot claim equality of optimizer steps, gradient
raw presentations, or target counts. Their finite resource ceilings can still
be compared. A contrast with no equal axes is an explicitly bounded comparison,
not a matched one. Memory peaks do not add across sequential phases.

## Evidence bundle and trace

`accounting.json` binds the canonical JSON protocol hash and includes one record
per arm/seed/phase. Every artifact reference is a relative path plus SHA-256.
Absolute paths, traversal, symlink escape, duplicate JSON keys, nonfinite JSON
numbers, unknown keys, missing/duplicate records, bool-as-int and stale hashes
fail closed. Only local JSON is read; no pickle or code execution is supported.

The split registry is `{"rows":[{"id": SHA256, "class_id": int,
"partition":"train|calibration|test", "available_from_task":int}, ...]}`.
Identifiers denote stable ORIGINAL raw rows, not current buffer offsets. All
aliases and copied exemplars must resolve to the same id. For real data this
registry remains in controlled storage; publish aggregate counts/hashes only.
Duplicate/capture grouping and correctness of original identity are an external
dataset-audit responsibility, not something this validator can infer from ids.

Every record has a hash-bound trace `{"events":[...]}`. A `microbatch` event has
`rows` (ordered raw ids), `binary_targets` triples `[row_index, head_class, 0|1]`,
and `multiclass_targets` pairs `[row_index, class]`. Its targets must use available
classes, match true labels, and cover all rows. Binary target count may exceed
row count when multiple heads consume a single embedding: count every distinct
row/head decision. A row repeated at another position is another presentation.
A `step` event has only its kind and closes all pending microbatches; an orphan
step or an uncommitted trailing microbatch is rejected. Record successful actual
steps, not requested epochs. A `fit_read` event has rows but no gradient targets,
and accounts prototype/router/candidate reads in non-gradient phases.

All trace rows must be train-partition rows whose labels and row availability
are valid at that phase. Calibration and test reads belong in separate evaluation
or selection evidence, never this fitting trace. Old/new counts depend on class
introduction task, not whether the sampler calls a row replay. The validator
recomputes raw presentations, distinct raw ids, gradient/fit presentations,
old/new presentations, binary target decisions, multiclass targets, and actual
successful step count, then exactly compares the submitted `counters` object.

## Memory evidence

Each record binds a `memory` artifact with a phase-boundary checkpoint snapshot
and independent phase peaks. Snapshot `components` contain one item per unique
live storage: `storage_id`, `category`, `numel`, `itemsize`, `bytes`. Count owners
once; a shared storage cannot appear twice. Categories are encoder/head
parameters, centroid state, replay state, optimizer state, and other persistent
state. Every category must be declared; a truly absent category has an explicit
zero-size entry. Metadata, labels, counts, normalizer arrays, live teacher/common
copies, and buffers belong in their appropriate category. Zero is not N/A.

`checkpoint_retained_bytes` sums non-optimizer entries; `optimizer_bytes` is
separate. `parameter_bytes`, `centroid_bytes`, `replay_bytes`, and `other_bytes`
are disjoint components, never added again to their own total. This snapshot is
not the peak or serialized file size. Per-phase `host_peak_rss_bytes` is the
measured whole process-tree high-water mark; CUDA allocated/reserved peaks are
separate device measures, with reserved >= allocated. CPU device metrics are
explicit N/A (`null`, reason `cpu_only_phase`), not fabricated zero utilization.
Snapshots and peaks have named instrumentation methods and a declared scope.
A no-gradient phase has no optimizer updates, but can retain optimizer storage
from earlier operations; that resident state must be honestly included.
Residency is not an update.
An end snapshot with no optimizer may legitimately have zero optimizer bytes;
the actual phase peak still captures transient optimizer/activation/workspace
cost. Instrumentation must additionally archive maximum optimizer-state size
when it differs; this v1 snapshot field does not claim that maximum.

The validator cannot prove a trace was measured rather than invented or that a
component was not omitted from a hand-written inventory. Inspect instrumentation
source and emitted logs, reconcile hardware/profiler records and checkpoint
arrays, and bind that evidence before scientific use. It deliberately returns
`ACCOUNTING_CONSISTENT`, never `EXPERIMENT_APPROVED`.
