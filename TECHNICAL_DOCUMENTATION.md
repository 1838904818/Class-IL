# OFRA and ETG technical documentation

## Native-target and one-use implementation, 11 September 2026

The [implementation update](experimental/etg_exploratory_v1/IMPLEMENTATION.md)
adds fixed-context native SHAP extraction, same-forward logit capture, an
identity-exact native-anchored R1 calibrator and persistent one-use stage
orchestration. Synthetic tests cover route-batch preservation, actual SHAP
masker dtype/invariance behavior, rollback ordering and spent failed attempts.
The [lineage metadata](experimental/etg_exploratory_v1/LINEAGE_METADATA.json)
binds both checkpoints and 16 historical runtime files to archived evidence;
the latest general runtime must not be substituted. These are software and
metadata checks, not new experiment results. The full naive attribution plan
can entail millions of native batch calls; an independently reviewed resource
profile and verified real-bundle launcher are still required.

## Initial descriptive ETG pilot preparation, 11 September 2026

The [early-checkpoint protocol](experimental/etg_exploratory_v1/PROTOCOL.md)
fixes native D2 seed 1, checkpoint 000 to 001, and five selectors sharing one
bounded correction. The attribution increment is isolated against performance
alone; raw drift is the noise-subtraction ablation. The source-bound planner
verified 896 distinct role rows across four classes and two checkpoint file
bindings; only 64 old-class trigger rows are planned for paired attribution.
These are planned sample counts, not newly trained or evaluated populations.
Fourteen synthetic kernel tests passed. No real attribution, intervention or
governance utility result is reported. Capture independence remains absent;
acceptance would be empirical, not a risk certificate. Full native extraction,
prior-use accounting, sealed one-use execution, resource profiling and independent
review are required before a real run. The [novelty boundary](docs/ETG_NOVELTY_AND_FALSIFICATION_2026-09-11.md)
explicitly separates established SHAP, explanation-drift and bias-correction work
from the proposed explanation-informed selection mechanism.

## Input compatibility audit, 11 September 2026

The [read-only ETG input audit](results/etg-input-audit-20260911/README.md)
verified that equal calibration counts do not imply equal row identities:
26,770 of the newer P/O calibration rows belong to the native D2 fitting
selection. Do not transfer that pool or its embeddings to native D2 checkpoints
as an independent holdout. This does not revise the original P/O results.
The single Heartbleed calibration row and unestablished capture groups also
prevent the proposed full strict acceptance study. No new model or governance
experiment has run; the 10 September protocol's gates remain unchanged.

## Explanation-governance follow-up, 10 September 2026

The [minimal incremental-utility protocol](docs/ETG_MINIMAL_UTILITY_PROTOCOL_2026-09-10.md)
keeps OFRA fixed and compares the same bounded correction selected by performance
alone versus performance plus method-conditioned explanation change. This is a
prospective follow-up, not a new result or a change to the manuscript's RQ/RO.
The current ETG evidence remains an offline audit; real intervention utility,
independent capture grouping and the new attribution inputs are not established.
Architecture and fusion alternatives remain separate diagnostic studies.

## Current revision: v3.3.1, 7 September 2026

### Subsequent diagnostic evidence, 9 September 2026

The completed P training-seed-1 frozen-embedding L1 run now has a fixed
final-checkpoint competition diagnostic. Both loss arms reproduced their
archived prediction hashes. For Slowhttptest, 1,029 of 1,100 positives
crossed the binary 0.5 threshold but lost the multiclass argmax; the
corresponding binary false-positive rate was 65.38%. FTP-Patator showed
1,576 such losses among 1,588 positives and a binary FPR of 34.46%.
Thus observed competition does not establish useful binary discrimination
or demonstrate that thresholding would solve the problem.

See the [published diagnostic and provenance](results/p-seed1-final-competition-diagnostic/README.md).
This is one exploratory final checkpoint, not native routed OFRA, a P/O
performance comparison, an AUROC/AP evaluation or an ETG intervention.
The earlier preparation records below describe their respective dates.
The diagnostic does not update the manuscript's primary five-seed results.

### Complete P and O derivation update, 8 September 2026

All four increments of both data policies now pass real local derivation and
pair checks. P has 149476 fitting rows and O has 268697; both share 68313
calibration rows and the unchanged 227723-row official test. Attack selections,
calibration identities and common transform cohorts match. The 119221-row
difference consists of Task-0 Benign fitting candidates. This is completed data
preparation, not completed model training or a fair-exposure performance result.
See [paired derivation evidence](experimental/parallel_execution/prospective_sampling/real_pair_20260908/PAIR_RESULT.md).

### Initial Task 0 preparation record, 8 September 2026

The first real local prospective Task 0 derivation is complete. From 529439
training rows, it selected 5559 Benign and 5559 DoS GoldenEye fitting rows,
reserved 52943 calibration rows, and recorded 465378 omitted fitting candidates.
An independent source-index audit verified exhaustive, nonoverlapping coverage.
The common transform cohort has 11118 rows; no transform or encoder was fitted.
The measured local duration was 83.891 seconds with about 186.3 MiB observed
peak child working set. These are Windows measurements, not a DICC allocation
profile. See the [result and evidence](experimental/parallel_execution/prospective_sampling/real_input_preparation_20260908/TASK0_RESULT.md).
The subsequent update above completes data derivation; model training and
performance evaluation remain outstanding. Row-identity checks do not prove capture-group
independence. No new predictive benefit or algorithmic novelty is established.

### Infrastructure verification update, 8 September 2026

The [UUID-only telemetry repair](experimental/parallel_execution/operations/README.md)
uses a timeout-bounded identity helper. The [new local verification record](experimental/parallel_execution/UUID_REPAIR_VERIFICATION.json)
reports 192 executed tests passed and two platform skips across five software
tracks. These software tests do not establish governance benefit.
The subsequent [GPU probe verification](experimental/parallel_execution/operations/FIDELITY_VERIFIED_SCOPE.md)
completed as Job 454941: 20 seed/checkpoint pairs and 60 head/router/joint
score comparisons were numerically equal with zero prediction changes on the bound
probes. This is not a full-population test, new accuracy result, or proof of
historical driver/kernel identity. Manuscript integration remains pending.

Executable preparation: the [three-line execution package](experimental/parallel_execution/README.md)
now implements actual paired head optimization/accounting, immutable train-only
array derivation and its encoder/input bridge, bounded R1 fitting with independent
acceptance and sealed subsequent evaluation, and native raw-score/fusion controls.
The [verification record](experimental/parallel_execution/LOCAL_VERIFICATION.json)
concerns tiny synthetic CPU and mocked operational tests only; no real-data
training, SHAP extraction or intervention benefit is claimed by that local record.
The separately linked GPU probe verification is the subsequent empirical result.
The [candidate novelty analysis](experimental/parallel_execution/NOVELTY_AND_FALSIFICATION.md)
documents close prior art, conditional claims, simple controls and rejection
criteria. The [review-status delta](experimental/parallel_execution/REVIEW_STATUS_DELTA.md)
keeps empirical review requests open. The earlier
[reference-only package](experimental/parallel_research/README.md) is preserved
unchanged. Historical results and manuscript v3.3.1 are unchanged.

The [current manuscript](paper/revision_2026-09-07_v3_3_1/README.md) and
[technical audit supplement](docs/TECHNICAL_AUDIT_SUPPLEMENT_2026-09-07.md)
supersede the narrative status below. The submitted v3.2 is retained unchanged.
[Saved-evidence re-analysis](results/review-audit-20260907/README.md) reproduces
450 historical states/actions, 240 fixed-top-15 policy replays and performance-
stratified consensus. Primary ReplayIDS statistics use last-epoch Job 425539,
not the older guarded comparison. Provenance capsules bind recovered protected
protocol versions and all five probe manifests; they are redacted attestations.

ETG is an audit record, not a demonstrated operational governance mechanism.
The [experiment gates](docs/EXPERIMENT_GATES_2026-09-07.md) separate remaining
numerical/attribution fidelity, matched-baseline and actual repair-utility tests.
No new model run or validated algorithmic novelty is claimed by this revision.

Declaration correction: v3.3.1 acknowledges DICC computational resources and
removes unconfirmed assertions of no funding or competing interests. Those
author declarations remain pending confirmation before submission. Scientific
content is unchanged from v3.3; this is not a new experiment or a final paper.

Execution preparation: a [bounded fidelity/profile coordinator](docs/REPLAYIDS_FIDELITY_PROFILE.md)
now implements controlled execution, aggregate-only tracking and protected
failure evidence. All 91 remote input hashes were checked successfully; this
verifies input identity, not numerical reproduction. Local operational tests
use synthetic/mocked boundaries. GPU execution, measured resource suitability
and live tracking verification remain outstanding.

Staging correction: the v5 package matched all 15 remote file hashes, but its
validation-only guard rejected ANSI-coloured FULL account output. No job was
submitted. The [account-status regression note](docs/DICC_ACCOUNT_STATUS_VALIDATION.md)
records the v6 parser repair and 16 passing regressions. v6 has now been
uploaded: all 17 operation hashes and 91 input hashes match, and actual
validation-only checks pass. After independent review and fresh confirmation,
the exact candidate was submitted as Job 453142 at 06:48 UTC on 7 September.
The [submission record](results/replayids-gpu-fidelity/JOB_453142_SUBMISSION.json)
preserves its initial pending snapshot. The job subsequently FAILED 126:0
after 27 seconds: telemetry failed before any of the twenty checkpoint
comparisons completed. This is an infrastructure failure, not a scientific
score mismatch or an OOM diagnosis. The
[terminal evidence and local telemetry repair](experimental/parallel_execution/operations/README.md)
record protected hashes, memory headroom, explicit compute-environment
preservation, bounded diagnostics and preflight-before-verifier regression tests.
The proposed v3 coordinator is not uploaded, reviewed or submitted. Independent
W&B cloud finalization remains unverified. No numerical tolerance was relaxed.

## Previous manuscript and validation priorities (historical snapshot)

The previously submitted [v3.2 manuscript](paper/external_review_2026-09-07/README.md)
includes the completed five-seed prediction and Malaya attribution evidence,
the primary checkpoint correction and the CPU score-fidelity limitation.
The earlier snapshot below is retained as history, not the current completion register.
The [validation priorities](docs/VALIDATION_PRIORITIES_2026-09-07.md)
distinguish existing results from proposed score-fidelity, sensitivity,
faithfulness, fair-comparison and operational evaluations. The
[recovered five-seed EG grid](docs/RECOVERED_SENSITIVITY_2026-09-07.md)
contains 320 verified settings and alternative-metric coverage for 100 transitions.
Full three-method governance-state sensitivity and historical-environment GPU
parity remain unestablished. No new training, GPU verification or operational
ETG benefit is claimed by this documentation update.

The [revised research plan](docs/RESEARCH_PLAN_2026-09-07.md) makes independently
measurable intervention utility a condition for retaining ETG as a main contribution.
A [research-only score-decomposition reference](experimental/expansion_diagnostic/)
has synthetic control tests, not real-data validation or a novelty claim.

GPU validation preparation: [historical-environment verifier](docs/REPLAYIDS_GPU_FIDELITY_CANDIDATE.md)
now binds 91 inputs and restores the original deterministic/TF32 settings.
Local control tests and input hashes have been checked. A later coordinator
implements W&B/protected-output integration with synthetic tests; GPU execution,
resource profiling, real integration validation and submission approval remain pending.
This is preparation, not a new performance or fidelity result.

Canonical-path update: [all-checkpoint CPU fidelity](docs/REPLAYIDS_CANONICAL_FIDELITY.md)
used the original 512-row encoder batches and whole-probe Head/Router calls.
Five initial checkpoints passed; fifteen later checkpoints failed score tolerance.
Two argmax changes were the same seed-3 sample at two checkpoints. This is not a
matched historical GPU run or a revised performance result.

Validation preparation: the [staged score-validation protocol](docs/REPLAYIDS_SCORE_VALIDATION_PROTOCOL.md)
now binds all 20 primary checkpoint inputs. Canonical encoder batching is 512;
prior batch32/898 tests are stress diagnostics. Legacy reproduction and a
separately versioned direct64 candidate remain distinct, unpromoted stages.

Numerical isolation: [fixed-embedding Router diagnostics](docs/REPLAYIDS_ROUTER_ARITHMETIC.md)
confirm float32 Router batching sensitivity. A direct64 reference is more stable
in the tested cases but does not reproduce every saved CUDA prediction. No
arithmetic change or calibration candidate has been promoted.

Reconstruction update (7 September 2026): the [frozen-probe forward check](docs/REPLAYIDS_FORWARD_PARITY.md)
failed the fixed Router/Joint score tolerance in all five seeds on CPU. One
seed-4 probe prediction changed at batch 32. Calibration remains gated; original
protected benchmark results are unchanged.

Input readiness (7 September 2026): the [holdout and inference-state audit](docs/REPLAYIDS_SCORE_READINESS.md)
verifies 68,313 calibration rows and five final snapshots, not a fitted calibrator
or new result. Forward parity and all-checkpoint score reconstruction remain open.
The audit distinguishes Task-0 normalization from offline count-based sampling.

Checkpoint-policy clarification: the ReplayIDS figures in the balanced-replay
addendum below use the secondary guarded policy. The registered primary is
last epoch. See the [primary comparison and scoring diagnosis](docs/REPLAYIDS_PRIMARY_SCORING_DIAGNOSTIC.md)
for the corrected scope and the verified 85.51% primary accuracy. Historical
exports are retained; no test-based model promotion was performed.

Latest prediction addendum (7 September 2026):
[completed Balanced Replay50 paired replication](docs/BALANCED_REPLAY_FIVE_SEED_RESULTS.md).
Both datasets now include seeds 1, 2, 3, 4 and 42, with seeds 1-4 reported separately
because seed 42 selected the comparator. OFRA has higher final accuracy in every
pair; Malaya Macro-F1 gains are small on the new seeds, and ReplayIDS lower FPR
comes with lower attack recall. This addendum does not add SHAP/ETG evidence or
replace the registered prediction arm with a selected new rule.

## Archived four-seed snapshot (superseded)

> **Historical reference, not the current evidence summary.** The text below
> preserves an earlier four-seed release, including its model settings,
> preprocessing descriptions, result tables, and then-open work. Statements
> such as "current", "completed", and "not yet complete" below refer to that
> release, not to the present project. In particular, the 122/194-feature
> preprocessing descriptions must not be substituted for the later strict
> Task-0-only contracts, and the four-seed table is not the final five-seed table.

For the current completed-evidence narrative, use the
[v3.0 technical document and manuscript](paper/guarded_checkpoint_2026-09-03/).
The separate [v9 attribution/ETG register](reproducibility/ATTRIBUTION_ETG_V9_STATUS.md)
tracks the unfinished five-seed explanation campaign. Its partial outputs do
not supersede the completed single-seed pilot or establish a final aggregate.
The [repository overview](README.md) identifies the registered primary and
keeps the non-significant guarded-checkpoint comparison separate.

---

## 1. Research objective and evaluation boundary

The project studies class-incremental learning for network-traffic
classification. A class-incremental learner receives an ordered sequence of
tasks. Each task introduces classes that were not available in earlier tasks,
and evaluation covers all classes seen so far. The practical problem is to add
new traffic classes without retraining from scratch while limiting catastrophic
forgetting of earlier classes.

Four datasets are intrusion-detection benchmarks: NSL-KDD, UNSW-NB15,
CIC-IDS-2017, and CSE-CIC-IDS2018. MalayaNetwork_GT is an external
application-traffic dataset with application/service labels. It tests the same
class-incremental mechanism under non-IID capture conditions but is not treated
as an intrusion-detection benchmark.

Each dataset is processed, trained, and evaluated independently. Rows from
different datasets are never pooled into a shared training stream.

## 2. End-to-end architecture

```text
Raw dataset
    |
    v
Dataset-specific preprocessing contract
  - file and source-hash validation
  - label normalization and fixed class order
  - identifier removal and numerical conversion
  - dataset-specific train/test split
  - class/split shards and overlap audit
    |
    v
FT-Transformer encoder, 512-dimensional representation, 12 layers
    |
    +--------------------+----------------------+
    |                    |                      |
    v                    v                      v
Family-specific      DP-Means router       Exemplar memory
low-rank heads       class centroids       bounded replay
    |                    |
    +---------+----------+
              v
Five registered decision arms
  head-only | router cap | joint cap | router full | joint full
              |
              v
Checkpoint metrics and fixed-probe monitoring
              |
              v
SHAP expected-gradient explanation analysis
              |
              v
Offline ETG governance ledger
```

The classifier produces the prediction before SHAP and ETG are run. SHAP
explains the stored decision evidence. ETG records whether the explanation and
performance evidence satisfy a registered governance rule. ETG does not alter
the class predicted by the model in the published experiment.

## 3. Dataset-specific preprocessing contracts

`fullcache/specs.py` is the executable source of truth. The builder operates in
chunks, validates the required files and columns, rejects unexpected labels in
strict mode, converts features to finite numerical arrays, and writes
class-by-split shards. Every cache records input and output SHA-256 hashes,
feature order, class order, task schedule, row accounting, and a split-overlap
audit.

| Dataset | Input and split | Removed fields | Output and task schedule |
|---|---|---|---|
| NSL-KDD | `KDDTrain+.txt` and `KDDTest+.txt`; official split retained; categorical vocabularies for protocol, service, and flag are fitted on training data only | label and difficulty | 122 model features; classes Normal, DoS, Probe, R2L, U2R; tasks `(Normal, DoS) -> Probe -> R2L -> U2R` |
| UNSW-NB15 | official training and testing CSVs retained; categorical vocabularies for protocol, service, and state are fitted on training data only | id, binary label, attack category | 194 model features; ten classes; five two-class tasks beginning with Normal and Generic |
| CIC-IDS-2017 | eight labelled flow CSVs; deterministic 80/20 grouping by cleaned feature bytes | label, flow ID, source/destination IP, source port, timestamp, external IP | 78 features; eight grouped classes; four two-class tasks |
| CSE-CIC-IDS2018 | all ten official labelled traffic CSVs; deterministic 80/20 grouping by cleaned feature bytes | label, timestamp, flow ID, source IP/port, destination IP; destination port is retained | 78 features; seven grouped classes; four tasks |
| MalayaNetwork_GT | 31 derived-flow CSVs at frozen revision `384a59278f98490ee6e93aae017e748078d29b6a`; one frozen capture per class is held out | source/destination IP, source/destination port, timestamp | 77 numerical flow features; ten application classes; five two-class tasks |

For the feature-hash split, identical cleaned feature rows share the same split
assignment, preventing an exact cleaned duplicate from appearing in both train
and test. Malaya uses a capture-level split instead because rows from the same
capture are correlated; a row-level random split would leak capture-specific
structure into evaluation.

## 4. Model and continual-learning components

### 4.1 FT-Transformer encoder

The current formal configuration uses a 512-dimensional FT-Transformer with 12
layers, 16 attention heads, head dimension 32, and 0.1 attention/feed-forward
dropout. The encoder maps a numerical traffic row `x` to representation
`h(x)`. The large encoder is shared by all classes within one dataset run.

The initial task is trained for eight epochs. Each later task is trained for
ten epochs. An epoch is one pass over the training data selected for that
stage. More epochs provide more optimization steps but can also increase
overfitting or forgetting; eight and ten are protocol settings, not universal
optimal values.

### 4.2 Family-specific low-rank heads

Each seen class has a binary positive-versus-negative decision head. The head
uses a rank-8 low-rank adaptation with scaling parameter 16. For class `c`, the
head outputs probability `p(c,x)`. Family-specific heads allow a new class to
receive a new decision component without rebuilding a single fixed multiclass
output layer.

### 4.3 Bounded exemplar memory

The learner retains at most 50 exemplars per class from a candidate pool of at
most 5,000. Replay mixes selected older examples with the current task to
reduce catastrophic forgetting. This is a fixed study budget and does not
imply that 50 is generally optimal.

### 4.4 DP-Means router

The router maintains one or more centroids for each seen class in encoder space.
DP-Means can create an additional centroid when an embedding is sufficiently
far from the existing centroids, subject to a maximum of 32 centroids. The
creation threshold is derived from the registered 0.3 distance quantile.

For class `c`, the raw routing score is the negative distance from `h(x)` to
the nearest centroid. Scores are standardized across seen classes to obtain
`z(c,x)`. A higher value means the sample is more compatible with that class's
stored embedding geometry.

The capped router fits centroids from at most 3,000 selected samples per class;
the uncapped router uses all eligible samples. The cap controls router fitting
cost and prevents large classes from dominating centroid estimation. It does
not limit the number of test rows.

### 4.5 Registered decision arms

The experiment evaluates matched views of the same trained model:

- head-only: `s(c,x) = p(c,x)`;
- router cap: `s(c,x) = z_cap(c,x)`;
- joint cap: `s(c,x) = p(c,x) + 0.5 z_cap(c,x)`;
- router full: `s(c,x) = z_full(c,x)`;
- joint full: `s(c,x) = p(c,x) + 0.5 z_full(c,x)`.

The predicted class is `argmax_c s(c,x)`, meaning the class with the largest
score among all classes seen at that checkpoint. The joint weight 0.5, cap
3,000, quantile 0.3, and centroid limit 32 are registered project settings.

## 5. Training and evaluation protocol

The formal configuration uses focal loss with gamma 2.0 and alpha 0.75,
learning rate 0.001, batch size 384, evaluation batch size 512, and a
negative-to-positive sampling ratio of 4. Deterministic execution and shard
hash verification are enabled. Recovery checkpoints are validated by hash
before a run resumes.

At checkpoint `t`, the model is evaluated on every class seen through task
`t`. Reported metrics include:

- overall accuracy: fraction of correct predictions;
- Macro-F1: arithmetic mean of per-class F1, giving every class equal weight;
- balanced accuracy: arithmetic mean of per-class recall;
- average forgetting: mean decline from each earlier task's best historical
  accuracy to its current accuracy, reported in percentage points;
- benign false-positive rate and attack-detection recall for datasets with a
  valid benign/attack interpretation.

MalayaNetwork_GT does not report benign/attack metrics because its labels are
applications, not attack categories.

## 6. SHAP explanation analysis

SHAP is a family of additive feature-attribution methods. The current analyzer
uses `shap.GradientExplainer`, an expected-gradients approximation suitable for
the differentiable FT-Transformer. It does not use KernelSHAP.

For a fixed probe and checkpoint, the analyzer reconstructs the registered
`joint_cap3000` class margin and attributes that margin to input features. A
positive attribution pushes the explained margin upward; a negative value
pushes it downward. The largest absolute values identify the features with the
strongest contribution for that particular probe and decision. The commonly
shown top-15 list is therefore ranked by absolute SHAP magnitude, not selected
manually.

Explanation drift is measured across eligible class-by-adjacent-checkpoint
transitions. Feature-ranking overlap uses Jaccard similarity:

`J(A,B) = |A intersection B| / |A union B|`.

Here `A` and `B` are the selected top-feature sets at two adjacent checkpoints.
A silent explanation-drift event is a registered condition in which predictive
performance remains within its stability rule while the explanation changes
beyond the explanation threshold. The thresholds, including a 0.7 rule used
in the registered analysis, are study-defined governance settings and are not
claimed as universal standards.

## 7. ETG: Explanation Trust Graph

ETG is the project's offline explanation-governance layer. Its evidence unit is
a class-by-adjacent-checkpoint transition, not a packet, flow, sample, user, or
real-world incident. Each ledger row links the earlier checkpoint, later
checkpoint, performance evidence, explanation evidence, registered thresholds,
and resulting governance state.

The published strict ETG analysis can:

- certify an admission when the registered evidence is acceptable;
- refuse admission when evidence is insufficient or violates a rule;
- escalate a transition for further review;
- require strict recertification after drift;
- record whether that recertification passed or failed.

These actions are simulated governance outcomes. They are not evidence that a
human review, deployment block, or production remediation actually occurred.
The completed ledger consumes stored OFRA checkpoint evidence; it does not yet
demonstrate a closed-loop causal system in which ETG changes future OFRA routing
or training.

## 8. Current completed evidence

All values below are mean +/- sample standard deviation across seeds 1-4.

| Dataset | Arm | Accuracy | Macro-F1 | Balanced accuracy | Forgetting |
|---|---|---:|---:|---:|---:|
| MalayaNetwork_GT | Joint full | 56.14% +/- 3.00 | 21.04% +/- 3.85 | 22.89% +/- 3.92 | 3.23 +/- 0.88 pp |
| MalayaNetwork_GT | Joint cap 3,000 | 54.37% +/- 3.02 | 20.70% +/- 3.72 | 22.70% +/- 3.86 | 3.79 +/- 0.64 pp |
| NSL-KDD | Joint full | 68.51% +/- 2.87 | 38.32% +/- 2.97 | 40.44% +/- 2.83 | 2.60 +/- 1.15 pp |
| NSL-KDD | Joint cap 3,000 | 69.07% +/- 3.38 | 38.81% +/- 3.04 | 40.87% +/- 2.96 | 2.38 +/- 1.34 pp |

The low Malaya Macro-F1 and balanced accuracy show that its moderate overall
accuracy is driven by uneven class performance. The result must not be
summarized by overall accuracy alone. The high between-seed variability in the
NSL head-only arm also shows that conclusions must be based on paired,
multi-seed comparisons rather than one favorable run.

The completed Malaya seed-1 explanation analysis reported 12 silent-drift
events among 17 eligible transitions (70.59%). ETG recorded six certified
admissions, four refused admissions, four escalations, one strict
recertification, and two strict-recertification failures.

## 9. What is complete and what remains open

Completed in this release:

- large-model MalayaNetwork_GT seeds 1-4;
- large-model NSL-KDD seeds 1-4;
- Malaya seed-1 SHAP and ETG analysis from completed DICC Job 389896;
- CSE-CIC-IDS2018 A100 capacity evidence;
- executable preprocessing contracts for the five-dataset suite;
- deterministic result, source-binding, and publication hashes.

Not yet complete:

- the fifth registered seed;
- new FT-Transformer 512x12 formal results for CIC-IDS-2017, UNSW-NB15, and
  CSE-CIC-IDS2018;
- multi-seed SHAP and ETG estimates;
- inferential tests supporting superiority claims;
- a deployed feedback loop in which ETG decisions alter future OFRA routing or
  training.

The available four-seed values are descriptive intermediate evidence. They are
appropriate for progress reporting and reproducibility review, but they are not
presented as a complete final-paper result table.

## 10. Reproducibility and file map

- `fullcache/`: preprocessing and cache verification;
- `streaming_full/`: current training, routing, evaluation, monitoring, and
  recovery;
- `ofra_encoders/`: FT-Transformer integration;
- `formal_v2_explanation_etg/`: SHAP and ETG analyzer;
- `results/`: per-seed JSON, four-seed aggregate, ETG tables, and capacity
  profile;
- `reproducibility/`: exact runtime and analysis bindings;
- `SHA256SUMS.txt`: SHA-256 for every published file other than the manifest
  itself.

The authoritative quantitative records are the JSON and CSV artifacts under
`results/`. W&B is used to observe runs and compare logged metrics, but the
repository artifacts and their hashes remain the reproducibility source of
record.
