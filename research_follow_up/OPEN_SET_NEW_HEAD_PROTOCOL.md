# OFRA unknown-attack rejection and new-head protocol

## Plain-language scenario

Today's OFRA must choose one of the classes it already knows. If a genuinely
new attack arrives, `argmax` still selects the largest existing score, even when
all scores are poor. This pilot inserts an **Unknown** gate before that forced
choice.

The system does not invent a semantic class name from a single flow. It first
rejects suspicious flows as Unknown, accumulates a candidate cluster, asks for
labels/analyst confirmation, and only then creates the next LoRA family head and
DP-Means centroids. The final labelled update uses OFRA's existing incremental
head-creation mechanism.

## End-to-end state machine

1. **Known-class scoring**: every flow receives the existing family-head
   probabilities `p`, DP-Means centroid distances `d`, and the current joint
   scores.
2. **Unknown gate**: a flow is rejected when both its maximum family-head
   confidence is unusually low and its nearest-centroid distance is unusually
   high relative to a known-only calibration set.
3. **Candidate buffer**: rejected flows are stored in a bounded research buffer
   and clustered. This is a candidate novel pattern, not yet a semantic class.
4. **Governance check**: require minimum support, persistence across time
   windows and analyst/ground-truth confirmation. Without confirmation, no new
   class head is created.
5. **Labelled increment**: confirmed samples form the next task. OFRA creates a
   new LoRA family head, selects replay exemplars, and builds new DP-Means
   centroids.
6. **Post-update audit**: measure new-class recall, old-class forgetting,
   Benign FPR, false-new-head rate and discovery delay.

## Pilot construction

Use ReplayIDS because the local expected-contract cache and prior OFRA result
already exist. For each pilot, move one attack class to a singleton final task:

- recommended first held-out classes: `FTP-Patator`, `DoS Slowhttptest`, and
  `DoS slowloris` in separate runs;
- do not hold out `DoS GoldenEye` in the first pilot because it shares Task 0
  with Benign; removing it would leave OFRA without the two-class Task-0
  pretraining contract;
- do not use `Heartbleed` as the main unknown class because it has only 11 rows
  total and cannot support a stable conclusion;
- keep Benign in Task 0;
- reserve 10% of each training class before training as calibration data;
- optionally apply the 50k/class fit-training cap, but never cap the official
  test set.

Immediately before the final task, evaluate the held-out class as Unknown. The
final task then supplies its labels, creates the real new head, and allows a
post-update accuracy/forgetting audit.

## Thresholds without test leakage

For known-only calibration rows:

- `p_max`: largest existing family-head probability;
- `d_min`: Euclidean distance to the nearest retained DP-Means centroid;
- `tau_p`: 5th percentile of known calibration `p_max`;
- `tau_d`: 95th percentile of known calibration `d_min`;
- conservative rule: reject if `p_max < tau_p AND d_min > tau_d`.

The held-out unknown class is not used to select these thresholds. Additional
head-only, distance-only and empirical-joint rules are reported as ablations,
not silently selected after seeing the best test result.

## Required controls

| Arm | Rule | Purpose |
|---|---|---|
| Closed-set | Always use current `argmax` | Shows how often unseen attacks are forced into old families |
| Confidence-only | `p_max < tau_p` | Tests the family heads alone |
| Distance-only | `d_min > tau_d` | Tests the DP-Means router alone |
| Conservative joint | both conditions true | Primary low-false-alarm gate |
| Empirical joint | calibrated combination of confidence and distance percentiles | Diagnostic continuous detector |
| Oracle boundary | held-out label is supplied at the declared task boundary | Upper bound after a legitimate labelled increment |

## Metrics

- known-class Accuracy, Macro-F1 and balanced accuracy before rejection;
- known false-unknown rate;
- unknown detection recall and precision;
- AUROC and AUPR for known-versus-unknown detection;
- open-set balanced accuracy and OSCR-style coverage/correctness curve;
- candidate-cluster support, persistence and evaluation-only purity;
- time/rows until a candidate is eligible for analyst review;
- new-class recall/F1 after the labelled increment;
- old-class average forgetting and Benign FPR after the new head is added;
- false new-head rate and bounded memory cost.

## Interpretation boundary

This experiment tests **open-world rejection plus labelled adaptation**. It does
not prove autonomous zero-day naming, does not allow an unlabeled sample to
create a semantic attack family, and is not adversarial robustness evidence.

SOUL provides a close precedent: it combines model confidence with similarity
to replay-buffer exemplars, accepts only high-confidence agreement, and still
uses analyst labels for uncertain novel data. Our pilot adapts that principle to
OFRA's multiclass family heads and DP-Means centroids.
