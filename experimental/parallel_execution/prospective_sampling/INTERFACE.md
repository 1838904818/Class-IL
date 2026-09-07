# Array and availability contract

## Input isolation

The `derive` command accepts one strict `available_train_increment` JSON, one
policy, a source root, an exclusive new output directory, and (after Task 0)
the previous completed increment. It never accepts a full source catalog,
future class/count fields or sealed-test contracts by schema. Unknown
keys, duplicate JSON keys, nonfinite numbers and nonconsecutive increments
fail closed. Available class IDs must be unique and may appear only once.
The normal class must first appear at Task 0 and never later.

Top-level keys are `schema_version=1`, `kind=available_train_increment`,
`dataset_id`, `namespace`, `increment`, `feature_dim`, `normal_class_id`,
`group_mode`, and `classes`. Each class has `class_id`, `name`,
`introduced_increment` equal to the current increment, `partition=train`, and
`shards`. A shard has `shard_id`, `features`, `row_ids`, `group_ids`, `labels`.
The last three may be null. Every nonnull array reference has exactly
`path`, lowercase `sha256`, and integer `rows`. Paths are relative to the
explicit source root; absolute/traversal/backslash/colon paths are rejected.

Feature arrays must be C-contiguous native float32 or float64 `(N,F)` matrices,
with finite values, expected row count and feature dimension, and one common
observed dtype across shards/increments. Object/pickle arrays, integer feature
matrices, wrong shapes, stale hashes and NaN/Inf fail. Optional label sidecars
must be integer `(N,)` with every value equal to the declared class. Where the
historical source supplies class-pure arrays without label sidecars, label
truth is the source-manifest contract, not an inferred or invented label.
The reader cannot discover whether deliberately mislabeled source bytes are
really training data; the pinned adapter's train references, source ownership
and external dataset governance establish that boundary, not a path keyword.

Hashes are verified before use and again after extraction. All input sidecars
receive independent shape/type/content and hash validation. The source root,
input paths, output ancestors and loaded artifact paths reject symlinks and
Windows reparse points. The design assumes an exclusively controlled user
workspace, not a hostile concurrent filesystem adversary; reviewed source
hashes and ownership remain necessary.

## Stable identity and groups

Provided `row_ids` and `group_ids` must be `(N,)` arrays with exact `S64`
lowercase SHA-256 strings. Without row IDs, the canonical relative source
path must equal `shard_id`, and a row ID is SHA-256 of the fixed identity
domain, dataset namespace, that path, and original source-row offset. Paths
must therefore retain their original source-object identity after transfer.
Rows copied into replay retain the same ID; buffer indices are not identities.
SQLite uniqueness rejects ID and original source-position collisions across
all completed increments. Actual lineage under renamed copies, semantic
duplicates or independently acquired identical flows requires source audit.

`group_mode=row_identity_proxy` writes the row ID as a singleton group proxy.
This is the explicit default for the available ReplayIDS source, which has no
capture-group sidecar. It does **not** establish capture independence or
deduplication. `provided_disjoint` requires every group sidecar and rejects a
group spanning fit/calibration anywhere in the prefix. It does not silently
change the fixed calibration quota to repair a group collision; such a source
needs a separately declared grouped redesign. Even a passed group-boundary
check leaves `capture_independent_established=false`, since the software
cannot authenticate the supplied group's scientific meaning.

## Selection and bounded memory

The rank function preserves the earlier reference's fixed domain
`ofra-prospective-sampling-reference-v1` and hashes only seed, split stage,
class ID and stable row ID. It uses no input manifest digest, test digest,
future count, policy hash, increment number or complete class registry in its
random domain. Provenance hashes are recorded separately from selection.

Calibration count is zero for a singleton/empty class or zero fraction;
otherwise it is `min(N-1, max(1, floor(N*numerator/denominator)))`. Lowest
calibration ranks define the calibration pool. Lowest independent fit ranks
then select normal fit rows up to the frozen cap; all attack fit rows remain.
Fitting, calibration and omitted IDs partition the source cohort exactly.
The common Task-0 transform pool uses the P normal selection and all Task-0
attack fit rows, independent of whether the materialized arm is O or P.

There is no Python all-row list, full-array load, permutation array or 10,000
row ceiling. SQLite stores cumulative identity/rank/selection state, with
disk temporary storage, an 8 MiB page-cache setting, disabled SQLite mmap and
one SQLite worker. Feature/sidecar checks and extraction use bounded chunks;
output arrays are written by memmap. The chunk size is an I/O implementation
parameter, not a measured DICC resource request. OS caches/mmap residency and
SQLite overhead still require a real peak-RSS/disk profile. Every new increment
copies the immutable previous ID ledger; this extra I/O/storage is disclosed.

## Output and loader

`load_increment(directory, expected_manifest_sha256=None)` verifies completion,
exact file closure, hashes and array shape/dtype bindings. A consumer should
pass the independently reviewed manifest SHA, not rely only on a mutable
directory's self-reported marker. `iter_partition(directory, partition='fit',
chunk_rows=8192, expected_manifest_sha256=None)` yields copied chunk dictionaries:

| Key | Shape/dtype | Meaning |
| --- | --- | --- |
| `x` | `(N,F)`, observed float32/64 | Raw feature rows; not standardized/embedded |
| `labels` | `(N,)`, int64 | Declared class label |
| `row_ids` | `(N,)`, S64 | Original stable source row identity |
| `group_ids` | `(N,)`, S64 | Provided group or explicitly labeled row proxy |
| `available_tasks` | `(N,)`, int64 | First labeled class-cohort availability |
| `source_index` | `(N,2)`, int64 | Source-shard slot and original row offset |

Each increment aggregates its newly released classes. Its `manifest.json`
contains `collections.fit`, `calibration`, `omitted`, and `transform_fit`, each
with total rows and refs (`path`, `sha256`, `rows`, `shape`, `dtype`) for the
arrays above. Omitted rows retain ID/label/provenance arrays but no feature copy.
Transform-fit arrays are materialized only at Task 0; later manifests carry
the same `common_transform_fit_sha256`. `source_shards` resolves source slots.
`classes` gives actual per-class counts; `historical_classes` accumulates them.
The SQLite ledger is a bound artifact, not an exposed service or server.

`binding` includes exact current input, policy, implementation and previous
manifest hashes. Increments reject changed dataset/namespace/dtype/policy/code,
old-class rewrites or invalid previous output. The official test is absent.

## Completion, failure and recovery

Output paths require exclusive creation and an existing safe parent. Files
are never written into historical directories. `STARTED.json` marks incomplete
work. The last write is `COMPLETE.json`, binding the manifest and every file.
Consumers must call the verifying loader, not test only for file existence.
Exceptions before completion leave `FAILED.json` and no usable increment.
Unlisted files, partial files, changed output arrays, changed prefixes and
bad completion markers are rejected by the loader.

`--resume` supports **verification and reuse of an already complete identical
increment only**. It revalidates source bytes and exact input/policy/code/prefix
bindings without rewriting output. It refuses incomplete output. Recovery is
at completed-increment boundaries: after diagnosing a failed increment, use
a new versioned destination and the last verified completed prefix. There is
no silent partial restart or automatic resubmission. A change to reviewed
bindings requires a new execution/review stage.

## Separate catalog/test stage

`adapt-replayids` is a metadata-only adapter for the exact existing source
schema. It emits a separate per-increment train contract and sealed test
contract. Its all-source hash remains in `ADAPTER_COMPLETE.json`, never in a
train selector input or RNG. Tests confirm that replacing future/test metadata
changes adapter provenance but not the early train contract or array outputs.

`verify-test` reads only `sealed_official_test` contracts and verifies every
full source test shard/hash/shape/finite feature value. It emits an independent
`OFFICIAL_TEST_VERIFIED` closure and never writes to source shards, resamples
test support, borrows calibration rows or calls the sampler. Evaluation still
requires the original class semantics/support and a separately frozen scorer.

## Executable cross-line raw export bridge

`prepare_l1_inputs.prepare_pair(p_prefix, o_prefix, test_directory, source_root,
checkpoint_metadata, checkpoint_state, config_path, output, evidence_kind=...,
pretrain_cost=None, chunk_rows=8192)` accepts complete consecutive P/O prefixes,
the separate official-test verification, and one shared frozen checkpoint. It
creates `P/export-input.json` and `O/export-input.json` in the strict raw schema
accepted by `fair_comparison/materialize_embeddings.export`. Every feature and
identity array is actually written with memmap; this is not a manifest-only
adapter. All increments' immutable `fit` arrays are concatenated in order;
neither calibration nor omitted rows are sampled or copied into training.

Raw train `row_ids`, `group_ids` and `available_tasks` are copied unchanged.
Official test is copied at full original support, with the same source-row ID
formula and explicit shard/offset lineage. SQLite verifies train/test row and
proxy-group disjointness. Test does not influence any training sampler or RNG.
The selected source has no test group sidecars, so this adapter explicitly
requires `row_identity_proxy`; it cannot establish capture independence.

The bridge requires float32 input, matching the L1 raw-export contract. It
rejects float64 rather than silently converting precision. Each raw split
contains `raw_features`, `labels`, `row_ids`, `available_tasks`, `group_ids`,
with the same shapes as the upstream fit partition except the renamed x key.
References contain only relative `path` and `sha256`. The offline aggregate
contains all increments for controlled execution; it is not an online API
that physically hides future files. L1 must still enforce task availability.

Checkpoint metadata must include `common_transform_fit_sha256` (the canonical
SHA-256 of the Task-0 `collections.transform_fit` object),
`normalization_fit_row_ids_sha256` and `encoder_training_row_ids_sha256` (both
the exact Task-0 transform-fit row-ID `.npy` file SHA). Dataset, Task-0 classes,
dimension, evidence kind and inference-state SHA must also match. These are
auditable provenance assertions, not proof that an arbitrary checkpoint was
honestly trained. Real training receipts/source closure provide that proof.

`verify_export_input(manifest_path)` verifies exact arm file closure and the
parent pair's final commitment before use. Each arm has a full hash manifest;
the pair's `COMPLETE.json` is written last. A first arm surviving a failed
second arm is not a completed pair and cannot be consumed through this verifier.
Unbound files, altered arrays/checkpoints, missing completion, source mutation
and existing output directories fail closed. No partial resume is offered.
