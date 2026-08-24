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

### 3.1 No-look-ahead preprocessing audit

The numerical normalization path passes a strict temporal audit. The
`FrozenTask0Stats` accumulator reads only Task-0 training shards, is frozen
before Task-0 pretraining, and rejects later updates. It uses float64 population
variance and never reads official-test rows.

The categorical schema is more limited. NSL-KDD and UNSW-NB15 fit one-hot
vocabularies on their complete official training partitions before the
class-incremental stream begins. Relative to a Task-0-only vocabulary:

- NSL-KDD contains six future-only columns (`service`: `aol`, `harvest`,
  `http_2784`, `http_8001`, `pm_dump`; `flag`: `RSTOS0`). They affect 115 of
  12,703 later-task rows (0.905%).
- UNSW-NB15 contains eight future-only columns (`proto`: `argus`, `bbn-rcc`,
  `crtp`, `egp`, `hmp`, `netblt`, `rdp`; `service`: `irc`). They affect 712 of
  79,341 later-task rows (0.897%).

This is bounded transductive schema information, not official-test leakage and
not future-label training. Nevertheless, the complete preprocessing pipeline
is not strictly no-look-ahead for those two datasets. CIC-IDS-2017,
CSE-CIC-IDS2018, and MalayaNetwork_GT use numerical model inputs and have no
data-derived categorical vocabulary. A strict repair requires rebuilding
NSL-KDD and UNSW-NB15 with a Task-0-only or externally fixed vocabulary and
rerunning the affected experiments.

## 4. Model and continual-learning components

### 4.1 FT-Transformer encoder

The Malaya formal configuration uses a 512-dimensional FT-Transformer with 12
layers, 16 attention heads, head dimension 32, and 0.1 attention/feed-forward
dropout. Its initial task is trained for eight epochs and each later task for
ten epochs. The current NSL-KDD, UNSW-NB15, and CIC-IDS-2017 formal rows use the
smaller FT256x4 protocol recorded in their bound artifacts. Model capacity is
therefore disclosed per dataset and is not silently pooled. In every case the
encoder maps a numerical traffic row `x` to representation `h(x)` and is shared
by all classes within that dataset run.

An epoch is one pass over the training data selected for that stage. More
epochs provide more optimization steps but can also increase overfitting or
forgetting; all epoch counts are protocol settings, not universal optima.

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

### 6.1 Attribution-method robustness

The registered explanation is Expected Gradients. A source-bound Malaya seed-1
pilot also evaluates two alternative rankings against the same frozen
`joint_cap3000` class margin, probes, checkpoints, top-15 rule, random-control
admission threshold, and ETG state machine:

- single-feature ablation replaces one input at a time with the frozen
  checkpoint mean and measures the resulting margin decrease;
- Gradient x Input multiplies each raw input by the local margin gradient.

Across the 30 common checkpoint-class rows, all three primary methods agree on
the admission decision for 40.0% and the ETG state for 43.3%. Across the 20
common adjacent class transitions, all three agree on the silent-drift
conclusion for 20.0%. Pairwise mean top-15 Jaccard is 0.567 for Expected
Gradients versus feature ablation, 0.380 for Expected Gradients versus Gradient
x Input, and 0.405 for feature ablation versus Gradient x Input. The associated
silent-drift counts are 12/17, 0/17, and 7/17, respectively. ETG outcomes are
therefore method-conditioned.

Integrated Gradients were also attempted with the mean Task-0 background and
16-point Gauss-Legendre quadrature. Its completeness check failed on this
piecewise routed score (mean row-level error 11.58; maximum 373.72), so it is
retained as a numerical diagnostic and excluded from primary agreement claims.
This failure is evidence that an attribution method suitable for a smooth
encoder is not automatically valid for the complete routed scorer.

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

### 8.1 Current five-seed prediction evidence

The current strict table uses seeds `1, 2, 3, 4, 42`. Each dataset was trained
and evaluated independently. NSL-KDD, UNSW-NB15, and CIC-IDS-2017 use the
FT-Transformer 256x4 protocol; MalayaNetwork_GT uses FT-Transformer 512x12.
The table is therefore a dataset-by-dataset audit, not a capacity-matched pooled
superiority test.

| Dataset | Model | Arm | Accuracy | Macro-F1 | Balanced accuracy | Forgetting |
|---|---|---|---:|---:|---:|---:|
| NSL-KDD | FT256x4 | Head only | 56.29% | 31.08% | 36.48% | 8.14 pp |
| NSL-KDD | FT256x4 | Joint cap 3,000 | 71.56% | 41.46% | 42.25% | 1.60 pp |
| UNSW-NB15 | FT256x4 | Head only | 68.78% | 21.16% | 23.52% | 2.68 pp |
| UNSW-NB15 | FT256x4 | Joint cap 3,000 | 61.95% | 23.50% | 28.45% | 8.97 pp |
| CIC-IDS-2017 | FT256x4 | Head only | 64.59% | 21.25% | 25.91% | 9.04 pp |
| CIC-IDS-2017 | FT256x4 | Joint cap 3,000 | 72.57% | 36.95% | 63.13% | 9.63 pp |
| MalayaNetwork_GT | FT512x12 | Head only | 56.03% | 11.77% | 15.05% | 3.16 pp |
| MalayaNetwork_GT | FT512x12 | Joint cap 3,000 | 54.68% | 21.15% | 23.03% | 3.55 pp |

The results are heterogeneous. Joint cap 3,000 improves all four displayed
metrics on NSL-KDD. On UNSW-NB15 it improves class-balanced metrics but lowers
overall accuracy and increases forgetting. On CIC-IDS-2017 it improves
accuracy and class-balanced metrics but does not reduce forgetting. On Malaya
it improves minority-class coverage while its accuracy and forgetting effects
remain uncertain. No universal-superiority claim is supported by this table.

For Malaya FT512x12, the mean +/- sample standard deviation is:

| Arm | Accuracy | Macro-F1 | Balanced accuracy | Forgetting |
|---|---:|---:|---:|---:|
| Head only | 56.03% +/- 0.45 | 11.77% +/- 0.81 | 15.05% +/- 0.55 | 3.16 +/- 4.39 pp |
| Joint cap 3,000 | 54.68% +/- 2.71 | 21.15% +/- 3.37 | 23.03% +/- 3.42 | 3.55 +/- 0.77 pp |

The paired joint-minus-head mean differences are -1.34 percentage points for
accuracy (95% CI -4.71 to +2.02), +9.37 points for Macro-F1 (95% CI +4.70 to
+14.04), +7.97 points for balanced accuracy (95% CI +3.72 to +12.23), and
+0.39 points for forgetting (95% CI -5.75 to +6.54; negative would favour the
joint arm). The result supports a class-balance improvement, not a stable
forgetting reduction. Pooled per-class results further show that joint cap
3,000 produces non-zero recall for every Malaya class, while head-only has zero
recall for six of ten classes; several minority recalls nevertheless remain
low.

### 8.2 Explanation drift and offline ETG evidence

The completed source-bound explanation analysis currently covers Malaya seed
1. The registered primary rule uses Expected Gradients, the actual
`joint_cap3000` class margin, top-15 features, a Jaccard drift threshold of 0.7,
and an allowed class-recall drop of 5 percentage points. It reports 12 silent
explanation-drift events among 17 eligible class-by-adjacent-checkpoint
transitions (70.59%). This is not a packet, flow, sample, or production-incident
rate.

ETG records six certified admissions, four refused admissions, four
escalations, one strict recertification, and two strict-recertification
failures. These are simulated offline governance outcomes over stored OFRA
evidence. They do not mean that a human review occurred or that ETG changed
future routing, replay, training, or predictions.

The sensitivity grid varies top-k over {10, 15, 20}, the Jaccard threshold over
{0.6, 0.7, 0.8}, and the allowed class-recall drop over {2, 5, 10} percentage
points. It confirms that the reported rate is threshold-sensitive. The third
dimension must not be called an overall-accuracy tolerance. Because the grid is
single-seed, it bounds the registered decision rule but does not identify a
universal operating threshold.

The attribution-method robustness pilot is also bounded to Malaya seed 1.
Expected Gradients, single-feature ablation, and Gradient x Input agree on all
admission decisions for 40.0% of the evaluated cases, on ETG state for 43.3%,
and on the silent-drift conclusion for 20.0%. Integrated Gradients failed the
recorded completeness diagnostic and is excluded from the primary comparison.
The governance conclusion is therefore method-sensitive and must be reported
with this limitation.

### 8.3 Additive classifier comparisons on MalayaNetwork_GT

The TabM mean-embedding adapter has completed the full OFRA prediction
pipeline for seeds `1, 2, 3, 4, 42`. The table reports the official test view;
all values are mean +/- sample standard deviation.

| TabM-OFRA arm | Accuracy | Macro-F1 | Balanced accuracy | Forgetting |
|---|---:|---:|---:|---:|
| Head only | 56.17% +/- 1.79 | 11.11% +/- 1.99 | 14.43% +/- 2.35 | 0.61 +/- 0.46 pp |
| Router only, cap 3,000 | 40.15% +/- 3.49 | 17.44% +/- 1.44 | 19.29% +/- 1.11 | 12.75 +/- 1.89 pp |
| Joint, cap 3,000 | 57.99% +/- 3.93 | 20.46% +/- 1.19 | 21.60% +/- 1.70 | 4.39 +/- 2.56 pp |
| Joint, uncapped | 57.93% +/- 3.84 | 20.34% +/- 1.31 | 21.48% +/- 1.87 | 4.33 +/- 2.39 pp |

The earlier four-common-seed FT-versus-TabM diagnostic remains an additive
classifier comparison rather than a primary OFRA claim. It suggested that the
TabM gain was mainly aggregate accuracy rather than broad class-balanced
improvement. The now-complete five-seed FT512x12 result is the authoritative
FT prediction record; the historical four-seed comparison remains useful only
for tracing how the capacity experiment was developed.

Matched cumulative multiclass diagnostics over seeds `1, 2, 3, 4, 42` give
CatBoost 66.50% +/- 0.44 accuracy, 34.76% +/- 0.69 Macro-F1, 39.10% +/- 0.73
balanced accuracy, and 15.34 +/- 0.73 points of forgetting. The corresponding
cumulative TabM values are 65.01% +/- 0.96, 32.66% +/- 1.31, 34.41% +/- 1.51,
and 9.43 +/- 1.80 points. These cumulative results are classifier-capacity
diagnostics and are not matched OFRA comparisons.

### 8.4 CIC-IDS-2018 protocol boundary

A separate FT256x4 campaign with one Task-0 epoch and one epoch per later task
completed seeds `1`, `2`, `3`, `4`, and `42` (DICC jobs 395350, 399060, 399246,
399313, and 399593). Because its model and schedule differ from the FT512x12
8/10-epoch campaign, it is reported separately.

| Arm | Accuracy | Macro-F1 | Balanced accuracy | Forgetting |
|---|---:|---:|---:|---:|
| Head only | 69.27% +/- 20.34 | 21.81% +/- 8.82 | 25.67% +/- 6.35 | 7.62 +/- 7.54 pp |
| Router only, cap 3,000 | 41.01% +/- 6.08 | 31.88% +/- 6.06 | 58.50% +/- 5.03 | 19.13 +/- 6.01 pp |
| Joint, cap 3,000 | 52.55% +/- 5.79 | 34.05% +/- 7.54 | 53.93% +/- 7.79 | 19.03 +/- 8.62 pp |
| Joint, uncapped | 53.52% +/- 8.36 | 34.33% +/- 8.05 | 53.94% +/- 8.08 | 18.09 +/- 11.08 pp |

Relative to head-only inference, joint cap-3,000 increases Macro-F1 by 12.24
points and balanced accuracy by 28.26 points, but reduces overall accuracy by
16.72 points and increases forgetting by 11.41 points. The cap has little mean
effect relative to joint uncapped. This is a class-balance trade-off rather
than a universal gain.

The older FT256x4 one-plus-one-epoch closure remains valid under its own
protocol. A stricter FT256x4 five-seed campaign with the current formal
schedule is still running on DICC and is excluded from the current
four-dataset table until all seeds, hashes, and protected outputs are complete.

## 9. What is complete and what remains open

Completed in this release:

- MalayaNetwork_GT FT512x12 seeds `1, 2, 3, 4, 42`;
- NSL-KDD, UNSW-NB15, and CIC-IDS-2017 FT256x4 five-seed formal results;
- TabM mean-embedding OFRA prediction results on MalayaNetwork_GT for seeds
  `1, 2, 3, 4, 42`;
- matched cumulative CatBoost and TabM diagnostics on MalayaNetwork_GT for the
  same five seeds;
- Malaya seed-1 SHAP and ETG analysis from completed DICC Job 389896;
- a source-bound three-method attribution robustness pilot, with the failed
  Integrated-Gradients completeness diagnostic retained separately;
- a preprocessing no-look-ahead code/data audit;
- the protocol-separated CSE-CIC-IDS2018 FT256x4 one-plus-one-epoch closure;
- CSE-CIC-IDS2018 A100 capacity evidence;
- executable preprocessing contracts for the five-dataset suite;
- deterministic result, source-binding, and publication hashes.

Not yet complete:

- the currently running stricter CSE-CIC-IDS2018 FT256x4 five-seed campaign;
- SHAP and ETG analysis for the TabM arm; the current TabM result covers the
  prediction pipeline only;
- multi-seed SHAP and ETG estimates;
- multi-seed attribution-method robustness estimates;
- a strict Task-0-only or externally fixed categorical vocabulary rebuild and
  rerun for NSL-KDD and UNSW-NB15;
- inferential tests supporting superiority claims;
- a deployed feedback loop in which ETG decisions alter future OFRA routing or
  training.

The current prediction table is a completed descriptive five-seed record for
four datasets. The fifth dataset, multi-seed explanation/ETG estimates, and the
strict categorical-vocabulary reruns remain open, so the project is not yet a
complete final-paper evidence package.

## 10. Reproducibility and file map

- `fullcache/`: preprocessing and cache verification;
- `streaming_full/`: current training, routing, evaluation, monitoring, and
  recovery;
- `ofra_encoders/`: FT-Transformer integration;
- `formal_v2_explanation_etg/`: SHAP and ETG analyzer;
- `results/`: per-seed JSON, current five-seed aggregate, paired and per-class
  tables, ETG records, sensitivity analysis, and capacity profile;
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
