# OFRA and ETG technical documentation

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
reduce catastrophic forgetting. A controlled seed-42 diagnostic also tested
capacities 500 and 3,000 while holding the model, data, epochs, optimiser,
loss, router and seed fixed. The larger buffers improved final overall accuracy
and Benign false-positive rate but weakened class-balanced metrics. Capacity 50
therefore remains the reference for the next stage; the one-seed result does
not imply that 50 is universally optimal.

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

### 8.1 Strict five-seed prediction evidence

The table reports means across seeds 1, 2, 3, 4, and 42. Each dataset is an
independent experiment; model capacities are disclosed and are not pooled.

| Dataset | Model | Arm | Accuracy | Macro-F1 | Balanced accuracy | Forgetting |
|---|---|---|---:|---:|---:|---:|
| NSL-KDD | FT256x4 | Head only | 56.29% | 31.08% | 36.48% | 8.14 pp |
| NSL-KDD | FT256x4 | Joint cap 3,000 | 71.56% | 41.46% | 42.25% | 1.60 pp |
| UNSW-NB15 | FT256x4 | Head only | 68.78% | 21.16% | 23.52% | 2.68 pp |
| UNSW-NB15 | FT256x4 | Joint cap 3,000 | 61.95% | 23.50% | 28.45% | 8.97 pp |
| CIC-IDS-2017 | FT256x4 | Head only | 64.59% | 21.25% | 25.91% | 9.04 pp |
| CIC-IDS-2017 | FT256x4 | Joint cap 3,000 | 72.57% | 36.95% | 63.13% | 9.63 pp |
| CSE-CIC-IDS2018 | FT256x4 | Head only | 79.21% | 19.06% | 21.15% | 2.00 pp |
| CSE-CIC-IDS2018 | FT256x4 | Joint cap 3,000 | 50.16% | 33.55% | 52.15% | 19.07 pp |
| MalayaNetwork_GT | FT512x12 | Head only | 56.03% | 11.77% | 15.05% | 3.16 pp |
| MalayaNetwork_GT | FT512x12 | Joint cap 3,000 | 54.68% | 21.15% | 23.03% | 3.55 pp |

Routing is therefore dataset-dependent. It is favourable across all four
displayed metrics on NSL-KDD, improves class-balanced coverage at a cost on
UNSW-NB15, improves coverage without reducing forgetting on CIC-IDS-2017, and
produces an operationally severe false-positive and retention trade-off on
CSE-CIC-IDS2018. Malaya remains strongly imbalanced, so overall accuracy alone
is not an adequate summary.

### 8.2 Explanation and governance evidence

The completed Malaya seed-1 explanation analysis reported 12 silent-drift
events among 17 eligible transitions (70.59%). ETG recorded six certified
admissions, four refused admissions, four escalations, one strict
recertification, and two strict-recertification failures.

### 8.3 Seed-42 open-set and labelled-head pilot

The ReplayIDS expected-contract O1 pilot held FTP-Patator out until the final
labelled increment. Known-only calibration did not use the test set. Before the
label arrived, confidence-only, centroid-distance-only, conservative joint, and
empirical-joint gates all produced 0% unknown recall. The best AUROC was 0.5303,
the OSCR-style AUC was 0.4699, and the primary candidate buffer contained 509
known rows and no FTP-Patator rows. The current gate therefore failed as an
unknown-discovery mechanism.

After the label was supplied, the normal supervised OFRA update achieved
82.30% FTP-Patator recall. Old-class accuracy changed from 87.28% to 87.01%, a
reduction of 0.27 percentage points. The experiment did not create a head
automatically. It supports labelled adaptation only and remains a single-seed,
single-held-out-class diagnostic.

### 8.4 Seed-42 replay-capacity diagnostic

DICC Job 414908 compared exemplar capacities 50, 500 and 3,000 under the same
uncapped expected-contract data and `official/joint_cap3000` scoring view.
Final accuracy increased from 83.62% at replay 50 to 87.08% and 87.62% at
replay 500 and 3,000. Benign false-positive rate fell from 9.24% to 3.81% and
2.47%. However, balanced accuracy fell from 68.62% to 52.07% and 42.76%, while
attack recall fell from 61.06% to 58.31% and 55.46%. Replay 3,000 also reduced
Macro-F1 from 50.02% to 41.77%.

Neither larger buffer passed the registered minority-performance selection
rule, so replay 50 remains the reference. `joint_cap3000` is the separate
router sample cap; it is not the exemplar capacity being tuned. The diagnostic
uses seed 42 only and does not establish a five-seed ranking.

### 8.5 Paired five-seed checkpoint-selection diagnostic

DICC Job 425539 compared the last family-head epoch with the earliest epoch
that maximised binary Macro-F1 on a manifest-bound training-only calibration
split. Seeds `1, 2, 3, 4, 42` were paired on the same fixed data split, model,
optimizer, replay-50 budget, router, training budget and official
`joint_cap3000` evaluation arm.

| Metric | Last epoch | Training-only calibration | Paired delta |
|---|---:|---:|---:|
| Final accuracy | 85.51% +/- 6.18% | 87.52% +/- 4.78% | +2.02 pp |
| Final Macro-F1 | 54.40% +/- 5.07% | 56.34% +/- 4.06% | +1.95 pp |
| Average task accuracy | 79.78% +/- 6.86% | 77.88% +/- 9.57% | -1.91 pp |
| Forgetting | 3.52% +/- 2.60% | 5.18% +/- 5.43% | +1.66 pp |
| Attack recall | 85.73% +/- 14.98% | 83.76% +/- 16.72% | -1.97 pp |
| Benign FPR | 13.10% +/- 8.25% | 10.44% +/- 5.05% | -2.66 pp |

Every paired 95% confidence interval crossed zero. The Holm-adjusted p-value
for the two designated outcomes, Macro-F1 and forgetting, was 0.755161 for
each. The candidate therefore changes the trade-off but does not support a
superiority claim. Last-epoch selection remains the primary protocol.

Per-class evidence is similarly mixed. FTP-Patator F1 rises from 30.9% to
51.5% on average, while SSH-Patator recall falls from 82.3% to 69.0% and its F1
falls from 30.4% to 21.0%. Heartbleed has only two official-test rows and one
calibration row, so it uses the registered fallback and cannot support a strong
class-level conclusion.

## 9. What is complete and what remains open

Completed in this release:

- strict five-seed prediction evidence for all five independently processed
  datasets;
- Malaya seed-1 SHAP and ETG analysis from completed DICC Job 389896;
- three-method attribution-sensitivity pilot on the same routed score;
- CSE-CIC-IDS2018 strict 8+10-epoch five-seed campaign;
- seed-42 FTP-Patator open-set and post-label head diagnostic from DICC Job
  414686;
- seed-42 replay-capacity diagnostic from DICC Job 414908, with replay 50
  retained as the next-stage reference;
- paired five-seed checkpoint-selection diagnostic from DICC Job 425539, with
  last-epoch retention kept as the primary protocol;
- executable preprocessing contracts for the five-dataset suite;
- deterministic result, source-binding, and publication hashes.

Not yet complete:

- multi-seed SHAP and ETG estimates;
- rotated held-out-class and multi-seed open-set evaluation;
- a successful unknown gate or evidence supporting automatic head creation;
- multi-seed confirmation of any follow-up tuning configuration that improves
  both aggregate and minority-class operating metrics;
- inferential tests supporting superiority claims;
- a deployed feedback loop in which ETG decisions alter future OFRA routing or
  training.

The prediction table is a completed descriptive five-seed record. The open-set
result is diagnostic and negative for autonomous discovery; it must not be
promoted to a general open-world claim.

## 10. Reproducibility and file map

- `fullcache/`: preprocessing and cache verification;
- `streaming_full/`: current training, routing, evaluation, monitoring, and
  recovery;
- `ofra_encoders/`: FT-Transformer integration;
- `formal_v2_explanation_etg/`: SHAP and ETG analyzer;
- `results/`: per-seed JSON, five-seed aggregates, ETG tables, classifier
  comparisons, and the sanitised open-set pilot binding;
- `reproducibility/`: exact runtime and analysis bindings;
- `SHA256SUMS.txt`: SHA-256 for every published file other than the manifest
  itself.

The authoritative quantitative records are the JSON and CSV artifacts under
`results/`. W&B is used to observe runs and compare logged metrics, but the
repository artifacts and their hashes remain the reproducibility source of
record.
