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
Registered encoder arm
  - reported FT-Transformer: 512 dimensions, 12 layers
  - additive TabM comparison: 16 member embeddings averaged to 256 dimensions
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

### 4.2 Additive TabM encoder comparison

TabM is evaluated as an additional encoder arm rather than a replacement for
the registered OFRA method. The adapter produces 16 member embeddings with
shape `(batch, 16, 256)` and averages them across the member dimension to
obtain the single `(batch, 256)` representation required by the existing OFRA
interface. The family heads, exemplar budget, DP-Means routers, joint-score
weight, task schedule, and evaluation definitions remain unchanged.

This is a deliberately narrow integration. It tests whether a different
tabular representation improves the existing prediction pipeline without
introducing a new router or governance rule. It must be described as a
mean-embedding TabM adapter, not as a full member-wise redesign of OFRA.

### 4.3 Family-specific low-rank heads

Each seen class has a binary positive-versus-negative decision head. The head
uses a rank-8 low-rank adaptation with scaling parameter 16. For class `c`, the
head outputs probability `p(c,x)`. Family-specific heads allow a new class to
receive a new decision component without rebuilding a single fixed multiclass
output layer.

### 4.4 Bounded exemplar memory

The learner retains at most 50 exemplars per class from a candidate pool of at
most 5,000. Replay mixes selected older examples with the current task to
reduce catastrophic forgetting. This is a fixed study budget and does not
imply that 50 is generally optimal.

### 4.5 DP-Means router

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

### 4.6 Registered decision arms

The experiment evaluates matched views of the same trained model:

- head-only: `s(c,x) = p(c,x)`;
- router cap: `s(c,x) = z_cap(c,x)`;
- joint cap: `s(c,x) = p(c,x) + 0.5 z_cap(c,x)`;
- router full: `s(c,x) = z_full(c,x)`;
- joint full: `s(c,x) = p(c,x) + 0.5 z_full(c,x)`.

The predicted class is `argmax_c s(c,x)`, meaning the class with the largest
score among all classes seen at that checkpoint. The joint weight 0.5, cap
3,000, quantile 0.3, and centroid limit 32 are registered project settings.

### 4.7 CatBoost cumulative diagnostic

CatBoost is used only as a cumulative multiclass diagnostic. At each
checkpoint it trains a fresh classifier on all training rows belonging to the
classes seen so far. It therefore estimates classification headroom in the
fixed Malaya feature representation, but it does not use OFRA family heads,
bounded exemplar replay, DP-Means routing, or the cap-3,000 budget. Its native
feature attribution is not the registered routed-margin SHAP target and is not
entered into the ETG ledger. A CatBoost-OFRA hybrid would constitute a new
method and is outside the frozen protocol.

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

### 8.1 Additive classifier comparisons on MalayaNetwork_GT

The TabM mean-embedding adapter has completed the full OFRA prediction
pipeline for seeds `1, 2, 3, 4, 42`. The table reports the official test view;
all values are mean +/- sample standard deviation.

| TabM-OFRA arm | Accuracy | Macro-F1 | Balanced accuracy | Forgetting |
|---|---:|---:|---:|---:|
| Head only | 56.17% +/- 1.79 | 11.11% +/- 1.99 | 14.43% +/- 2.35 | 0.61 +/- 0.46 pp |
| Router only, cap 3,000 | 40.15% +/- 3.49 | 17.44% +/- 1.44 | 19.29% +/- 1.11 | 12.75 +/- 1.89 pp |
| Joint, cap 3,000 | 57.99% +/- 3.93 | 20.46% +/- 1.19 | 21.60% +/- 1.70 | 4.39 +/- 2.56 pp |
| Joint, uncapped | 57.93% +/- 3.84 | 20.34% +/- 1.31 | 21.48% +/- 1.87 | 4.33 +/- 2.39 pp |

The existing FT-Transformer 512x12 result contains only seeds `1-4`, so the
matched comparison uses those four common seeds. For `joint_cap3000`, TabM
raises accuracy from 54.37% to 59.59% (+5.23 percentage points) and Macro-F1
from 20.70% to 20.85% (+0.15 points), while balanced accuracy changes from
22.70% to 22.06% (-0.64 points) and forgetting is essentially unchanged
(3.79 versus 3.78 points). The gain is therefore mainly aggregate accuracy,
not broad class-balanced improvement.

The class audit explains this result. BitTorrent accounts for 5,537 of 10,370
test rows (53.39%), and its mean recall over the four common seeds rises by
10.54 points with TabM. Recall falls for Steam (-21.25 points), TeamViewer
(-11.57), Discord (-6.83), and Webex (-3.79). The five-seed TabM result cannot
be summarized as an across-the-board improvement.

Matched cumulative multiclass diagnostics over seeds `1, 2, 3, 4, 42` give
CatBoost 66.50% +/- 0.44 accuracy, 34.76% +/- 0.69 Macro-F1, 39.10% +/- 0.73
balanced accuracy, and 15.34 +/- 0.73 points of forgetting. The corresponding
cumulative TabM values are 65.01% +/- 0.96, 32.66% +/- 1.31, 34.41% +/- 1.51,
and 9.43 +/- 1.80 points. These cumulative results are classifier-capacity
diagnostics and are not matched OFRA comparisons.

## 9. What is complete and what remains open

Completed in this release:

- large-model MalayaNetwork_GT seeds 1-4;
- large-model NSL-KDD seeds 1-4;
- TabM mean-embedding OFRA prediction results on MalayaNetwork_GT for seeds
  `1, 2, 3, 4, 42`;
- matched cumulative CatBoost and TabM diagnostics on MalayaNetwork_GT for the
  same five seeds;
- Malaya seed-1 SHAP and ETG analysis from completed DICC Job 389896;
- CSE-CIC-IDS2018 A100 capacity evidence;
- executable preprocessing contracts for the five-dataset suite;
- deterministic result, source-binding, and publication hashes.

Not yet complete:

- the fifth registered FT-Transformer 512x12 seed for MalayaNetwork_GT and
  NSL-KDD;
- new FT-Transformer 512x12 formal results for CIC-IDS-2017, UNSW-NB15, and
  CSE-CIC-IDS2018;
- SHAP and ETG analysis for the TabM arm; the current TabM result covers the
  prediction pipeline only;
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
- `results/classifier_comparisons/`: bounded classifier-comparison summaries
  and their interpretation limits;
- `EXPERIMENT_SCOPE_FREEZE_20260813.md`: the frozen distinction between the
  primary method, additive comparisons, and changes requiring a new protocol;
- `reproducibility/`: exact runtime and analysis bindings;
- `SHA256SUMS.txt`: SHA-256 for every published file other than the manifest
  itself.

The authoritative quantitative records are the JSON and CSV artifacts under
`results/`. W&B is used to observe runs and compare logged metrics, but the
repository artifacts and their hashes remain the reproducibility source of
record.
