# Explanation Stability and Trust Governance for Class-Incremental Intrusion Detection

**Technical report: OFRA and ETG**<br>
WU LIQIANG, Faculty of Computer Science and Information Technology, Universiti Malaya<br>
Supervisor: Prof. Nor Badrul Anuar<br>
Co-supervisor: Assoc. Prof. Dr. Aznul Asraf Sabri

## Abstract

Class-incremental network intrusion detection must learn new attack families
without a task identifier while retaining earlier decisions. Predictive accuracy
alone does not establish whether the features supporting an old decision remain
stable. This work evaluates silent explanation drift across consecutive
checkpoints and separates two experimental contributions. OFRA is an
oracle-free detector with a frozen MLP encoder, per-family low-rank adapters,
binary heads, bounded exemplars, and DP-Means centroid memory. ETG is a separate
detector-agnostic proof of concept that maintains per-family explanation-trust
states on a replay-trained shared MLP. Across 25 complete OFRA joint-score runs,
six of 298 accuracy-preserved class transitions meet the silent-drift criterion
(`2.0%`). Four datasets have protocol-matched five-seed paired forgetting
records, giving a geometric-mean reduction of `3.07x` at a `7.7-16.0`
percentage-point accuracy cost. The seed-42 ETG ledger certifies 34 of 39
families, refuses five, records ten escalation events, withholds two
explanation-drift notifications for uncertified families, and records three
mass-only re-certifications. The underlying NIDS alerts remain unchanged. The
ETG ledger has not yet been integrated with OFRA.

## 1. Problem statement, research questions, and objectives

An intrusion detector may preserve accuracy on an old attack while changing the
features that support its score. Such a change is operationally important when
analysts use explanations for triage, audit, or rule maintenance. The study asks:

1. How often does silent explanation drift occur under an oracle-free
   class-incremental protocol?
2. What retention, accuracy, and efficiency trade-offs does OFRA exhibit?
3. Can a detector-agnostic trust state machine execute bounded certification,
   refusal, escalation, and re-certification policies?

The corresponding objectives are to define a consecutive-checkpoint stability
protocol, evaluate OFRA across five IDS datasets, and test ETG as a separate
governance proof of concept.

## 2. Methodology

### 2.1 OFRA implementation

OFRA uses a two-layer MLP encoder (`d -> 128 -> 128`) with ReLU activations. It
is supervised-pretrained on Task 0 for eight epochs and then frozen. Each attack
family receives a rank-8 output-space adapter (`alpha=16`) and a `128 -> 2`
binary head. The per-family module contains 2,048 adapter parameters and 258
head parameters, or 2,306 trainable parameters.

Each new family is trained for ten epochs with Adam (`lr=1e-3`, batch 256), at
most four negatives per positive, and 50 farthest-first exemplars per old
family. Families with fewer than 1,000 positives use focal loss (`gamma=2`,
positive-class `alpha=0.75`); larger families use cross-entropy.

For family `f`, `p(f)` is the positive-class probability from its binary head.
DP-Means stores at most 32 centroids per family. Let `z(f)` be the within-sample
z-score of the negative nearest-centroid distance across all learned families.
Inference predicts the family with the largest joint score:

```text
s(f) = p(f) + 0.5 z(f)
```

DP-Means is therefore not a hard pre-selector, and the implementation does not
use a separate learned calibration model.

### 2.2 Explanation-stability protocol

For each class and seed, a fixed probe of at most 200 test samples is reused at
each checkpoint. SHAP GradientExplainer uses up to 100 Task-0 training samples
as a fixed background and 200 SHAP samples. The attribution vector is the mean
absolute SHAP value over the probe.

A class transition is accuracy-preserved when its accuracy change is greater
than `-0.05`. It is labelled silent drift when its top-15 attribution Jaccard is
below `0.70`. Cosine similarity and Kendall tau are reported alongside Jaccard.
The isolated-path explainer excludes routing. The decision-level protocol
explains the continuous target-family joint score rather than the discontinuous
argmax operation.

### 2.3 ETG proof-of-concept scope

ETG is evaluated separately on a replay-trained shared MLP. It is not an
already-completed OFRA governance layer. The complete state-machine ledger uses
seed 42, 50 replay samples per class, Adam at `1e-3`, batch 256, and six epochs
per task. Each class uses the first 100 test rows as a fixed probe.

Occlusion importance is the class-probability change when one standardised
feature is set to zero. Rationale mass is the joint probability drop after
zeroing the top-15 occlusion features. Admission uses the empirical 95th
percentile from 50 random 15-feature sets as its null. A family is certified
when its mass exceeds the null and otherwise becomes unexplainable. A certified
family escalates when accuracy is preserved but Jaccard against its frozen
admission rationale falls below `0.70`.

The current re-certification rule checks whether rationale mass again exceeds
the current null; it does not also require restored Jaccard stability.

### 2.4 Datasets and statistical scope

| Dataset | Used records | Effective families | Model input dimension | Split |
|---|---:|---:|---:|---|
| NSL-KDD | 148,517 | 5 | 122 | official train/test |
| UNSW-NB15 | 257,673 | 10 | 196 | official train/test |
| CIC-IDS-2017 | 754,376 of 2.83M | 7 | 78 | capped + fixed 80/20 |
| CIC-IDS-2018 | 300,928 | 7 | 78 | capped + fixed 80/20 |
| NF-ToN-IoT-v2 | 377,916 | 10 | 39 | capped + fixed 80/20 |

The formal OFRA explanation runs use seeds `{0,1,2,3,42}`. Data caps, splits,
and task order remain fixed; seeds affect stochastic training, sample selection,
and explanation probes. Paired t-tests are Holm-corrected within the four-test
forgetting family. Exact Wilcoxon tests at `n=5` do not reach `0.05`; this is a
statistical limitation of the five-run design.

## 3. Results

### 3.1 Silent explanation drift

The formal decision-level summary is computed from the 25 complete stored runs.

| Dataset | Events / accuracy-preserved transitions | Rate |
|---|---:|---:|
| NSL-KDD | 2 / 36 | 5.6% |
| UNSW-NB15 | 0 / 80 | 0.0% |
| CIC-IDS-2017 | 3 / 48 | 6.25% |
| CIC-IDS-2018 | 0 / 51 | 0.0% |
| NF-ToN-IoT-v2 | 1 / 83 | 1.2% |
| **Pooled** | **6 / 298** | **2.0%** |

NSL-KDD contributes `2/36` events and the pooled result is `6/298`. The isolated
family path has no silent-drift events (`0/299`). The joint-score
wrapper uses a slightly different standard-deviation convention from deployed
NumPy inference, so `6/298` is the stored decision-score protocol result rather
than a bit-exact deployed-function reproduction.

### 3.2 Forgetting and accuracy trade-off

| Dataset | Comparator | OFRA forgetting | Comparator forgetting | Reduction / Holm p | Accuracy cost |
|---|---|---:|---:|---:|---:|
| NSL-KDD | Replay-DPMeans | 0.0188 +/- 0.0111 | 0.1236 +/- 0.0359 | 6.59x / 0.0085 | -16.0 pp |
| UNSW-NB15 | iCaRL | 0.0999 +/- 0.0123 | 0.3525 +/- 0.0111 | 3.53x / 8.36e-6 | -7.7 pp |
| CIC-IDS-2017 | Replay-DPMeans | 0.0194 +/- 0.0069 | 0.0565 +/- 0.0156 | 2.91x / 0.0267 | -10.6 pp |
| CIC-IDS-2018 | iCaRL | 0.0902 +/- 0.0427 | 0.1181 +/- 0.0192 | 1.31x / 0.1019 (n.s.) | -7.8 pp |

The geometric-mean reduction across the four protocol-matched comparisons is
`3.07x`. The comparator set is the one stored in the four-dataset inferential
file; it is not a universal strongest-baseline claim over every archived method.

A legacy five-seed NF-ToN-IoT-v2 comparison produced `2.33x` versus iCaRL using
a Transformer encoder (`d=64`) with five masked-feature pretraining epochs. The
formal OFRA protocol uses a two-layer MLP (`d=128`) and eight epochs of Task-0
supervised pretraining. The legacy NF-ToN value is therefore excluded from the
formal table pending a protocol-matched rerun.

### 3.3 ETG governance outcomes

The complete seed-42 ledger records the following state-machine outcomes:

| Dataset | Certified | Refused | Explanation-drift notifications withheld | Escalation events | Re-certifications |
|---|---:|---:|---:|---:|---:|
| NSL-KDD | 5/5 | 0 | 0 | 2 | 1 |
| UNSW-NB15 | 9/10 | 1 | 1 | 1 | 0 |
| CIC-IDS-2017 | 6/7 | 1 | 0 | 2 | 1 |
| CIC-IDS-2018 | 7/7 | 0 | 0 | 4 | 1 |
| NF-ToN-IoT-v2 | 7/10 | 3 | 1 | 1 | 0 |
| **Total** | **34/39** | **5** | **2** | **10** | **3** |

These are transition counts on the separate shared MLP. They are not
ground-truth real drifts, false-alarm measurements, human-study outcomes, or ETG
performance on OFRA's six logged joint-score events.

A separate five-seed ETG experiment repeats admission only. Replacing the
occlusion ranking with input gradients changes 19 of 39 admission decisions,
showing that the attribution engine is load-bearing.

## 4. Interpretation

Freezing OFRA's encoder and old family modules removes silent-drift events along
the isolated path. Operational stability is not guaranteed because the set of
competing heads and router-score normalisation change as new families arrive.
The stored joint-score protocol therefore retains `2.0%` silent drift.

OFRA should be interpreted as a retention-stability operating point rather than
an overall accuracy winner. ETG establishes detector-agnostic state-machine
feasibility only. Connecting ETG to OFRA's exact routed score remains a planned
experiment.

## 5. Limitations and evidence boundaries

- The decision-level explainer covers a continuous score, not the discrete
  argmax, and is not bit-exact with deployed standardisation.
- Scaling statistics use the complete training partition before the stream. No
  test labels are used, but later training-task distributions are visible.
- Baseline replay budgets are 200 examples per class versus 50 for OFRA; the
  comparison is not memory-budget matched.
- The CIC-IDS-2017 cache contains an unused WebAttack label slot. Seven
  non-empty families are evaluated; the root cause of the slot has not yet been
  established.
- ETG's complete governance transitions use one seed, fixed class-level probes,
  and a mass-only re-certification rule. Analyst workload has not been measured.
- Three datasets use capped majority classes. Full-data Kaggle outputs and
  CIC-IoT-2023 are not included in the reported claims.
- Exact environment versions, device metadata, dataset checksums, latency, and
  computational cost are not included in the current experimental record.

## 6. Reproducibility map

- OFRA implementation and benchmarks: `src_v2/`
- Stability, faithfulness, and decision-score protocols: `src_v3/`
- ETG ledger and ablations: `exp_etg*.py`
- Raw result files: `results/*.json`
- Consolidated evidence record: `results/evidence_summary.json`
- Current paper: `paper_v3/paper.docx` and `paper_v3/paper.pdf`

Public datasets are downloaded separately and remain subject to their original
licences and distribution terms.
