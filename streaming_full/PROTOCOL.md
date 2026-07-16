# Shard-backed OFRA full-validation protocol

This package is a separate validation path. It does not modify or import the
legacy OFRA method, multi-seed script, or dataset loaders. Training and
evaluation consume class-wise `.npy` shards through memory maps and bounded
batches.

## Manifest

The manifest uses schema version 1 and declares its semantic mode with
`problem_type`, `metric_profile`, and `task_semantics`. Supported pairs are:

- `intrusion_detection` with
  `nids_multiclass_with_binary_detection`; `normal_class_id` must be an
  explicit non-negative integer present in Task 0.
- `application_classification` with `generic_multiclass`;
  `normal_class_id` must be explicitly `null`.

The only supported task semantics are `class_incremental`. Legacy manifests
that omit both `problem_type` and `metric_profile` retain the original NIDS
interpretation. Partial declarations, mismatched profiles, or an inferred
normal class are rejected. The runner never guesses a normal class from its
name.

```json
{
  "schema_version": 1,
  "dataset": "dataset_name_and_version",
  "feature_dim": 78,
  "problem_type": "application_classification",
  "metric_profile": "generic_multiclass",
  "task_semantics": "class_incremental",
  "normal_class_id": null,
  "source": {
    "builder": "ofra-fullcache",
    "builder_version": "1.0",
    "feature_schema_sha256": "...",
    "builder_source_sha256": "...",
    "split_overlap_audit_sha256": "..."
  },
  "tasks": [[0, 1], [2, 3]],
  "classes": [
    {
      "id": 0,
      "name": "Normal",
      "train": [
        {"path": "normal_train_000.npy", "rows": 100000, "sha256": "..."}
      ],
      "test": [
        {"path": "normal_test_000.npy", "rows": 20000, "sha256": "..."}
      ]
    }
  ]
}
```

Class identifiers must be contiguous from zero, and the task list must contain
every class exactly once. For reportable runs, every shard needs its byte-level
SHA-256 and hash verification must remain enabled.

Problem type, metric profile, task semantics, and the nullable normal-class
identifier are part of the logical train/test fingerprints and protocol hash.
Changing only the semantic interpretation therefore creates a different
protocol and cannot reuse prior results.

For an `ofra-fullcache` build, sibling `manifest.json` and
`split_overlap_audit.json` files are mandatory. The loader verifies the
fullcache manifest's canonical self-hash, tool/dataset identity, its hash of
`streaming_manifest.json`, the builder-source canonical hash, and the actual
split-overlap sidecar hash. The protocol then records absolute paths and actual
hashes. No circular hash from the streaming manifest back to `manifest.json` is
required.

## Data and training semantics

- Normalization uses float64 streaming Chan/Welford population statistics from
  Task-0 training shards only. The statistics are frozen before pretraining.
- The encoder is pretrained on Task 0, then frozen.
- Each family head sees all positive rows on every epoch. Its negative set is a
  fresh, deterministic, without-replacement sample drawn from current-task
  classes and prior exemplars, capped by `negative_ratio` per positive.
- Each epoch records the exact positive and negative index-stream hashes,
  negative-source counts, optimizer-step and zero-negative-step counts, and
  label-wise loss, margin, and binary-confusion diagnostics. Across epochs the
  runner stores the exact negative exposure-multiplicity histogram. The
  histogram counter uses the smallest non-overflowing unsigned dtype (`uint8`
  for the formal 10-epoch protocol).
- Before training, `streaming_full.exposure_preflight` derives the same family
  counts from manifest metadata and the frozen `RunConfig`: positive rows,
  candidate-pool sources, selected negatives, count-exposure prior, optimizer
  steps, exact zero-negative steps, shortfall, and warnings. It neither opens
  test shards nor trains. Its deterministic self-hash is embedded in
  `protocol.json`; every completed result is checked against the planned
  counts and optimizer schedule.
- Training never constructs a global `permutation(N)`. A deterministic
  hierarchical sampler uses exact sequential-hypergeometric allocation for
  the negative subset, then shuffles shard order, block order within each
  shard, and rows within each bounded block. Every positive row occurs exactly
  once per epoch; negative rows are unique and reach the declared target.
- Exemplar candidates use an independent deterministic Algorithm-R reservoir.
  The default takes up to 5000 candidates and then applies farthest-first on
  those candidate embeddings. This is an auditable approximation, not exact
  farthest-first over the full class. Candidate and selected-array sizes and
  hashes are recorded.
- Cap and uncapped router arms share the identical encoder, heads, training
  trajectory, checkpoints, and matched cap centroids/counts/lambda.

`router_only_uncapped` and `joint_uncapped` mean that all training rows of a
class refine the router. They do not mean an unbounded in-memory dataset. The
router construction uses two bounded-memory passes. The first builds the
deterministic cap reservoir, performs full DP-Means followed by stable top-32,
then matched-refines the retained state on cap samples in original stream
order. The uncapped arm copies those final centroids, counts, and lambda. The
second shard pass refines only rows not selected into the cap reservoir. Thus a
class with no more than 3000 rows has byte-identical cap and uncapped states;
larger classes differ only through explicitly recorded complement rows. It is
not the legacy full-array DP-Means implementation.

## Evaluation views and prediction arms

The official test set is always the primary evaluation view. A repeatable
`--evaluation-view NAME=MANIFEST` argument can add a test-only Boolean mask
view, such as `duplicate_excluded`. The manifest is bound to the exact parent
streaming/fullcache manifests, logical train and official-test fingerprints,
semantic mode, every parent test shard, and every mask hash. It cannot contain
train shards,
escape its artifact directory, change class/task order, or leave a class with
zero retained support.

Build the exact-overlap view into a new directory (the builder refuses to
overwrite an existing artifact):

```powershell
python -m sensitivity_overlap `
  --manifest .artifacts/fullcache/dataset/streaming_manifest.json `
  --output-dir .artifacts/fullcache/dataset/evaluation_views/duplicate_excluded `
  --view-name duplicate_excluded `
  --work-directory .artifacts/overlap-work
```

Exact feature bytes and unique-overlap counts are indexed in disk-backed
SQLite; no global Python feature-key set is materialized. The resulting audit
is explicitly reconciled with the frozen `split_overlap_audit.json` counts and
hashes before the artifact directory is atomically published.

At every checkpoint the official batch is forwarded exactly once. The same
predictions are projected into official, retained, and excluded counts. For
every arm the runner verifies the integer identity
`CM_official = CM_retained + CM_excluded`. It also hashes model, router,
normalization, exemplar, and process RNG state before and after evaluation and
fails if evaluation mutates either state.

Each view reports exactly eight arms. The five primary arms are:

1. `head_only`
2. `router_only_cap3000`
3. `joint_cap3000`
4. `router_only_uncapped`
5. `joint_uncapped`

The three exposure-prior diagnostics are:

6. `head_only_exposure_prior_corrected`
7. `joint_cap3000_exposure_prior_corrected`
8. `joint_uncapped_exposure_prior_corrected`

For a family head with logits `(l0, l1)` and actual loss exposures `(P, N)`, the
diagnostic head score is
`sigmoid((l1-l0) - log(P/N))`, using reference prior 0.5. This is an inference-
only sensitivity diagnostic. It does not retrain a head, alter exemplars or
routers, or claim a calibrated posterior; in particular it does not correct
focal alpha/gamma weighting or negative-pool covariate shift.

The `cap3000` names identify the paper cells; the actual cap is stored in
`config.router_cap_samples` and should remain 3000 for paper runs. Each cell
saves its confusion matrix, accuracy, macro-F1, balanced accuracy, per-class
metrics, and task accuracies. NIDS profiles additionally save benign FPR and
binary attack-detection recall; generic profiles do not emit a
`binary_detection` object or binary-derived summary fields. The run summary
stores the complete task-accuracy matrix and computes forgetting separately
for each cell.

## Running

Generate (or read-only verify) the exposure preflight first:

```powershell
python -m streaming_full.exposure_preflight `
  --manifest .artifacts/fullcache/dataset/streaming_manifest.json `
  --config-json configs/run_config.json `
  --output .artifacts/exposure_preflight.json

python -m streaming_full.exposure_preflight `
  --manifest .artifacts/fullcache/dataset/streaming_manifest.json `
  --config-json configs/run_config.json `
  --output .artifacts/exposure_preflight.json `
  --verify-only
```

The non-verify command refuses to overwrite a different report; `--verify-only`
never writes.

Create a JSON object containing any non-default `RunConfig` fields, then run:

```powershell
python -m streaming_full `
  --manifest .artifacts/fullcache/dataset/streaming_manifest.json `
  --evaluation-view duplicate_excluded=.artifacts/evaluation_view_manifest.json `
  --config-json configs/run_config.json `
  --seeds 11 22 33 44 55 `
  --output-dir .artifacts/results/dataset
```

Outputs use result schema version 2: `protocol.json`, one
`result_seed_<seed>.json` per seed, and `summary.json`. The protocol records the
manifest and shard hashes, evaluation-view order and masks, source hashes,
environment, config, eight-arm registry, exposure formula, seeds, semantic
mode, and nullable normal-class identity. Per-seed files record timings,
normalization hashes, training and
exposure history, reservoir/router accounting, per-view checkpoint metrics,
view decomposition, and a deterministic result hash that excludes timing.

One non-blocking OS file lock, `.streaming_full.lock`, covers the complete
protocol/results/summary transaction for an output directory. A second writer
fails immediately. Long runs are resumable only under the identical protocol hash. Existing seed
files are reused only after dataset, seed, protocol, and deterministic-result
hash validation; an inconsistency stops the run. Protocol, seed results, and
summary files are written through `.tmp` plus atomic replacement. Structured
`OFRA_RUN_EVENT` lines report every seed start, end, validated skip, or failure
with elapsed seconds.

After all dataset directories contain the same five seeds, build the read-only
paired report with:

```powershell
python -m streaming_full.summarize `
  --input-root .artifacts/campaign-results `
  --output .artifacts/campaign-results/paired_inference.json `
  --expected-seeds 42 0 1 2 3 `
  --expected-datasets nsl-kdd unsw-nb15 cic-ids-2017 cic-ids-2018 cic-iot-2023 nf-ton-iot-v2
```

The aggregator requires result schema version 2, revalidates every source JSON,
and recomputes descriptive summaries. It reports all five paired values, delta
mean and sample SD, 95% Student-t CI, paired t-test, exact Wilcoxon signed-rank
enumeration, and separate preregistered Holm families: within-view primary
comparisons (`m=8`), within-view exposure-prior diagnostics (`m=3`),
retained-minus-official primary-arm sensitivity (`m=5`), and the corresponding
diagnostic-arm sensitivity (`m=3`). View sensitivity is a change in test
support on identical predictions, not a model-improvement estimate.
It explicitly records that with five non-zero pairs the smallest attainable
two-sided exact Wilcoxon p-value is 0.0625.

Run the synthetic-only smoke test with:

```powershell
python -m streaming_full.smoke_test
```

The smoke test creates temporary synthetic shards and deletes them on exit. It
does not open or train on any real dataset.
