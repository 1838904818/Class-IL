# L1 input and output contracts

All file references are `{"path":"bundle-relative-name", "sha256":"64 lower-case hex"}`.
The root is the input manifest's directory. Absolute paths, traversal, escaped
symlinks, stale bytes, duplicate JSON keys and nonfinite numbers are rejected.
Real inputs are never synthesized by a missing-field fallback. `synthetic_smoke`
is a programmatic complete schema example, labelled synthetic throughout.

## L1 input manifest

Top-level fields, with no extras:

- `schema_version`: integer 1.
- `dataset`: safe public identifier; `evidence_kind`: `real` or `synthetic`.
- `tasks`: ordered, disjoint lists of integer class ids; Task 0 has at least two.
- `group_mode`: `provided_disjoint` or `row_identity_proxy`.
- `splits`: `train` and `test`, each containing references for `embeddings`,
  `labels`, `row_ids`, `available_tasks`, and `group_ids`.
- `encoder_binding`: references `checkpoint`, `preprocessing`, `export_receipt`
  and `pretrain_cost_receipt`; actual `state_sha256`; `trained_tasks:[0]`;
  positive `embedding_dimension`.
- `config`: exactly the fields in `pilot_config.json`.
- `allow_aggregate_wandb`: boolean, default false in generated examples.

Each embedding file is a finite float32 N-by-D `.npy`; labels and availability
are int64 length N. Original-row ids and group ids are canonical SHA-256 strings
stored in fixed byte dtype `S64`. Row ids must be unique within each split and
disjoint across splits; group sets must also be disjoint. Every class has nonzero
train and test support. Availability equals its registered introduction task.
Repeated replay presentations retain the same original id. Label semantics are
class-incremental classification, not automatically benign/attack semantics.

The producer performs complete offline manifest/hash/shape/leakage validation,
including future-file metadata. This is not an online-data revelation emulator.
Training pools, thresholds, random seeds and replay selection use only arrived
classes; no later class population or test outcome controls an earlier update.
Replay indices are created only for arrived tasks. Sampling seeds do not use
future file, outcome or whole-manifest hashes.

The encoder `checkpoint` is a safe NPZ mapping original state-dictionary names
to numeric arrays, no pickle. The runner hashes actual tensors and checks
`state_sha256`; equality is not inferred from a seed or architecture label.
`preprocessing` is a hash-bound artifact describing/applied to the common export.

The export receipt has exactly `schema_version`, `status:COMPLETE`,
`evidence_kind`, `encoder_state_sha256`, `checkpoint_sha256`,
`preprocessing_sha256`, `split_array_sha256`, `export_environment` and
`cost_scope:separate_embedding_materialization_stage`. Split hash keys mirror
the split-array descriptors. When `export_environment.profile_sha256` exists,
the local `EXPORT_PROFILE.json` is also hash-verified. The supplied exporter
emits this binding with actual measurements.

The pretraining-cost receipt has exactly `status`, `evidence_kind`,
`encoder_state_sha256`, `cost_scope:shared_encoder_pretraining_separate_from_L1`
and `measurements`. Status `MEASURED` requires real `optimizer_steps`,
`raw_row_presentations`, and `elapsed_seconds`. When historical counters do not
exist, use `HISTORICAL_UNMEASURED_DISCLOSED` with `measurements:null`; this permits
the narrow shared-embedding L1 objective study but prohibits claims of fully
measured total method costs. No historical resource request substitutes for it.

New instrumented pretraining receipts additionally use
`measurement_scope:all_attempt_observed_compute_lower_bound_if_uncertain` and
boolean `unknown_tail` inside `measurements`. They include observed discarded
work, and never convert an interrupted, unobserved tail into an exact cost.
The detailed pretraining profile and attempt ledger remain separate artifacts.

## Raw embedding-export manifest

The exporter accepts the same identity/config/task fields, plus
`checkpoint_metadata`, `checkpoint_state`, `pretrain_cost_receipt` (a bound
reference or null), and `raw_splits`. It has no `encoder_binding` or `splits`.
Each raw split replaces `embeddings` with a float32 `raw_features` reference;
the remaining four label/identity/availability fields are unchanged.

Marked O/P producer inputs additionally require the immutable arm file closure,
`RAW_EXPORT_INPUT_COMPLETE` receipt and final `RAW_EXPORT_PAIR_COMPLETE`
transaction. Partial pairs, extra files, changed hashes and linked artifacts
fail before inference. Independent unmarked manifests remain supported. Source
hashes and producer completion are checked again before export completion.

Checkpoint metadata follows the real monitoring format: checkpoint 0,
`seen_classes`, `feature_dim`, `architecture`, `state_schema.encoder` mapping
parameter names to NPZ keys, `state_schema.normalization.mean/scale`, and
`inference_state_sha256`. Existing canonical metadata hashes are verified.
The exporter performs the same float64 `(raw-mean)/scale` followed by float32
conversion as the inspected frozen normalizer. It never refits statistics.
Normalization arrays must already be finite native float64 vectors; strings,
integers and other precision are rejected before any conversion. Encoder and
recovery tensors must match the registered model's exact dtype and shape.

The FT adapter pins three actual source files:

| Relative source | SHA-256 |
| --- | --- |
| `streaming_full/models.py` | `455c516b3da1fdc373263698ba5484050b4edde349b02ae2e29effae6e0d21c2` |
| `ofra_encoders/__init__.py` | `c66e35b2752cc4b8959ef97fc16ee0a4c26f45a3200c0e985940ad0ca70fd086` |
| `ofra_encoders/ft_transformer.py` | `3ed138ae19ac0874d5ea037b91c044a1b1355d65c27a9cec8f50ef095bc98fec` |

These are inspected implementation bindings, not a claim of historical GPU
numerical parity. The actual historical monitoring schema anchors are
`streaming_full/monitoring.py:256-296,387-421,614-639`; normalization is
`streaming_full/data.py:1428-1435`. The L1 head/loss anchors are
`streaming_full/models.py:30-47,150-165` and conditional application is
`streaming_full/validation.py:1251-1266,1394-1474`.

## Exact comparison and residual differences

Each new class receives a fresh output-space low-rank residual two-logit head:
`classifier(z + (alpha/rank) * z A^T B^T)`. A is Kaiming-uniform, B starts at
zero, and the classifier is PyTorch's linear initialization. A class-derived
seed constructs one initial state; both arms load byte-identical tensors.
Both use Adam with the same registered learning rate and weight decay.

For each head and epoch, all current-class positives are used once; negatives
are sampled without replacement from other current-task classes plus the same
bounded old-task exemplars. Negative count is `min(pool, ratio*positives)`.
Positive microbatch width is `max(1,batch_size//(ratio+1))`; proportional integer
allocation distributes the selected negatives over positive batches. The same
within-batch permutation and original-row stream are used in both arms.
The configured batch size must be at least `negative_ratio+1`, and is an actual
upper bound on the number of rows in any emitted training microbatch.

The conditional-focal arm applies
`alpha_y*(1-p_true)^gamma*CE` only when the current positive population is
strictly below `minority_threshold`; otherwise it is identical CE. This
exact threshold boundary is tested. After the last epoch the head freezes.
All seen positive head probabilities participate in a head-only argmax with
sorted-class deterministic tie breaking.

This pilot uses deterministic uniform exemplar selection, not the historical
OFRA farthest-first selection; task-level replay creation excludes current-task
aliases. Those choices are identical within the two-arm contrast but mean this
is not an exact historical OFRA rerun. A scoring/router change is not bundled
into the objective contrast.
The full embedding files remain available read-only on disk. The exemplar cap
limits old-row identities consumed by fitting, not total dataset storage or a
strict online-memory guarantee. Input-array logical bytes and process memory
are separately reported; this is not a memory-bounded streaming implementation.

## Output and recovery interface

The runner emits SHA-bound `INPUT_BINDING.json`, immutable `.pt` checkpoints,
atomic `LATEST.json`, per-attempt durable `.jsonl` journals, `COST_LEDGER.json`,
`result.json` and a final `COMPLETE.json`. Checkpoints contain only tensors and
primitive containers and are read with `torch.load(weights_only=True)`.
The checkpoint stores per-attempt commit receipts, optimizer recovery state,
initial/final model hashes, committed measurements and task evaluations.

Confusion matrices use sorted seen-class axes. Accuracy and Macro-F1 average
over that support; balanced accuracy is mean class recall. Signed forgetting
is the previous maximum recall minus current recall for previously evaluated
classes; negative improvement is retained. Task 0 forgetting is N/A, not zero.
Prediction hashes bind the emitted class-id stream. Evaluation never chooses
the checkpoint or a configuration. Reporting is per registered single seed;
the program does not turn repeated rows/classes into seed replication.
The pilot seed controls head initialization and sampling; the frozen encoder's
training identity is separately fixed by its checkpoint. Repeating head seeds
while reusing one encoder is not independent full-pipeline training replication.
