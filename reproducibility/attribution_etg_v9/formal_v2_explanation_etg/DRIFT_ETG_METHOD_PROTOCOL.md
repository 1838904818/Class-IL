# Explanation Drift and ETG Method Protocol

Status: pre-registered protocol for the new monitored MLP/FT-Transformer experiments.
This document separates established methods from study-specific operational rules. It must be read before interpreting any drift or ETG number.

## 1. What is sourced and what is proposed here?

| Component | Status in this study | Required wording |
|---|---|---|
| SHAP feature attribution | Existing method | Cite Lundberg and Lee (2017). |
| Jaccard set similarity | Existing similarity coefficient | Cite Jaccard; explain its use on top-k feature sets. |
| Kendall rank correlation and cosine similarity | Existing comparison metrics | Cite as established stability measures. |
| Prediction-preserving explanation-stability evaluation | Existing research direction | Cite explanation-sensitivity/stability work. |
| `top-15`, `Jaccard < 0.70`, and `Delta recall > -0.05` | Study-specific operational choices | State explicitly that these are pre-specified thresholds used in this study, not community standards. |
| Silent explanation-drift event and its denominator | Study-specific operationalisation | Say “we operationalise silent explanation drift as …”, not “silent drift is defined as …”. |
| Explanation-Trust Gate (ETG) name and state machine | Proposed proof of concept in this study | Say “we propose a prototype called ETG”; do not imply that ETG is an established external method. |
| Human-review escalation and explanation withholding | Study-specific governance policy | Describe as simulated policy outcomes, not observed human decisions or NIDS alert outcomes. |

A targeted literature and repository search did not identify an established method with the exact name “Explanation-Trust Gate”. This is not sufficient to claim worldwide novelty. Until a systematic literature review is completed, the defensible claim is that **this study proposes and evaluates a prototype called ETG**.

## 2. Explanation-stability quantities

For dataset `d`, seed `s`, class `c`, and adjacent checkpoints `t-1 -> t`, let `a[d,s,c,t]` be the class-level attribution vector computed on a fixed, hashed probe set and a fixed Task-0 background set. Let `S^k[d,s,c,t]` be the indices of its `k` largest absolute attribution values.

The top-k Jaccard overlap is

```text
J[d,s,c,t;k] = |S^k[d,s,c,t-1] intersect S^k[d,s,c,t]|
               -------------------------------------------------
               |S^k[d,s,c,t-1] union S^k[d,s,c,t]|
```

`J=1` means that the two checkpoints select the same top-k features; a smaller value means more feature-set churn. Continuous explanation instability is reported as `1-J`. Jaccard measures set overlap only; it does not measure attribution magnitude or feature order. Cosine similarity and Kendall tau are therefore reported alongside it.

For `k=15`, `J < 0.70` means that the two sets share at most 12 features, so at least three top-15 features have been replaced.

## 3. Study-specific silent explanation-drift rule

The previous code called the class-wise quantity “accuracy”. It is mathematically the class recall:

```text
Recall[d,s,c,t] = TP[d,s,c,t] / support[d,s,c,t]
DeltaRecall[d,s,c,t] = Recall[d,s,c,t] - Recall[d,s,c,t-1]
```

The primary event indicator is pre-specified as

```text
E[d,s,c,t] = 1(DeltaRecall[d,s,c,t] > -0.05
                 and J[d,s,c,t;15] < 0.70)
```

The reported rate is

```text
SilentExplanationDriftRate = sum(E)
  / sum(1(DeltaRecall > -0.05)).
```

The unit of analysis is a **class × adjacent-checkpoint transition**. It is not a percentage of packets, flows, test samples, attacks, or real-world concept-drift events. Every report and W&B card must show both the percentage and `events / eligible transitions`.

Preferred manuscript wording:

> We operationalise a silent explanation-drift event as an adjacent class-checkpoint transition in which class recall decreases by less than five percentage points while the Jaccard overlap between the two top-15 attribution sets is below 0.70. These thresholds are study-specific and are evaluated through sensitivity analysis; they are not presented as universal drift-detection standards.

## 4. Threshold sensitivity is mandatory

The primary cell is `k=15`, `J threshold=0.70`, and permitted recall drop `0.05`. The following grid must also be reported without retraining:

- `k in {5, 10, 15, 20}`;
- Jaccard threshold in `{0.50, 0.60, 0.70, 0.80}`;
- permitted recall drop in `{0.00, 0.02, 0.05, 0.10}`.

For every grid cell, record the event count, eligible-transition denominator, rate, dataset, model, seed set, score target, attribution method, probe size, and background size. A conclusion that disappears under nearby thresholds must be described as threshold-sensitive.

## 5. Exact score to explain in the monitored OFRA experiment

For family/class `f`, the formal joint score is

```text
r_f(x) = - min_mu ||e(x) - mu_f||_2
z_f(x) = (r_f(x) - mean_j r_j(x)) / (std_j r_j(x, ddof=0) + epsilon)
s_f(x) = p_f(x) + 0.5 z_f(x)
```

where `p_f` is the positive probability from the binary family head and the router uses the exact `cap3000` state. The new explanation target is the decision margin

```text
g_c(x) = s_c(x) - max_{f != c} s_f(x).
```

This is preferable to explaining `s_c` alone because the final prediction is determined by competition between families. The scorer must use population standard deviation (`ddof=0`) to match formal inference. For the current confirmation campaign, normalization and router values in the forward pass use the exact NumPy float32 inference formula; gradients pass through an algebraically equivalent Torch expression by value correction. The forward margin must match the formal inference margin within `1e-6`. The old archived wrapper used a different numerical path and is therefore not treated as a bit-exact reproduction of the formal deployed score.

For numerical stability at an exact encoder-to-centroid match, only the
differentiable router surrogate applies
`squared_distance = max(squared_distance, 1e-12)` before `sqrt`. The
straight-through correction still restores the exact NumPy float32 forward
score, so the floor does not alter the classifier output or family decision.
It defines the local gradient at this non-smooth boundary for attribution only.
Before any SHAP computation, every registered seed/checkpoint/class true-class
probe batch must produce finite margins and finite input gradients. Any
non-finite value fails closed; it is never replaced with zero or omitted.

## 6. Prediction and family-choice stability

On the same fixed probe identities, report quantities that do not depend on an explanation threshold:

```text
PredictionStability = mean_i 1(pred[t-1,i] == pred[t,i])
PredictionFlipRate = 1 - PredictionStability
JointFamilyChoiceChange = mean_i 1(joint_family[t-1,i] != joint_family[t,i])
```

If a router-only choice is reported, it must be computed from the router-only arm. A change in the final joint-score family must not be labelled “pure router drift”.

## 7. ETG is a proposed governance prototype

ETG is not a detector-accuracy metric. It is a per-family state machine governing whether a generated explanation is admitted for automated use.

Primary states:

```text
UNCERTIFIED -> CERTIFIED_STABLE or UNEXPLAINABLE
CERTIFIED_STABLE -> DRIFTED when the monitored drift rule fires
DRIFTED -> CERTIFIED_STABLE or UNEXPLAINABLE after re-certification
```

### 7.1 Admission

For a class `c`, let `q_c(x)` be the softmax probability obtained from all current joint scores. Let `R_c` be the top-15 explanation feature set and `mask(x,R)` replace those standardised features with the Task-0 reference value. Define rationale mass as

```text
m_c = mean_x [q_c(x) - q_c(mask(x,R_c))].
```

Construct 50 deterministic random 15-feature controls and let `n_c` be the 95th percentile of their deletion masses. ETG admits the explanation when `m_c > n_c`; otherwise its state becomes `UNEXPLAINABLE`. This admission test is a study-specific rule, not a published universal certification test.

### 7.2 Monitoring and escalation

If a certified family satisfies the pre-registered silent explanation-drift rule, ETG records a transition to `DRIFTED` and a simulated `human_review` action. This is an explanation-governance escalation. It is **not** a measured NIDS alert, true positive, false positive, or completed human review.

For an `UNEXPLAINABLE` family, the underlying classifier output remains available, but ETG withholds the untrusted explanation notification. W&B must call this **Explanation alert withheld**, not “NIDS alert suppressed”.

### 7.3 Re-certification

The archived ETG-v1 code re-certified when current rationale mass exceeded the current random null, even if Jaccard stability had not recovered. That is too weak for a primary claim.

The new primary ETG-v2 rule is pre-registered as

```text
re-certify iff current rationale mass > current random null
               and Jaccard(current, certified reference) >= 0.70.
```

The legacy mass-only rule may be reported as a sensitivity ablation, never as the primary re-certification result.

## 8. ETG outputs that are safe to report

- Certified families / eligible families;
- Refused or unexplainable families / eligible families;
- Explanation-governance escalations;
- Explanation alerts withheld because admission failed;
- Strict re-certifications and demotions;
- final state distribution;
- rationale mass, random-null threshold, and their margin;
- detector performance before and after any separately labelled repair experiment.

Without ground-truth drift-event labels, ETG results cannot establish drift-detection precision, recall, false-positive rate, or clinical/operational utility. Gate-only ETG does not update detector weights; repair experiments that update weights must be reported separately with their accuracy cost.

## 9. Legacy results and their boundary

The independently recomputed archived results are:

- joint-score silent explanation drift: `6 / 298 = 2.0134%`;
- isolated-family path: `0 / 299`;
- archived routed-score mean top-15 Jaccard: `0.91283`;
- seed-42 ETG PoC: `34/39` certified, `5/39` refused, 10 escalations, two explanation-governance notifications withheld, three mass-only re-certifications, and zero demotions.

These results use legacy seeds `{0,1,2,3,42}`, include NF-ToN rather than Malaya, and use an ETG shared-MLP experiment that is not connected to OFRA or FT-Transformer. They may appear only in a clearly labelled `legacy_poc` table/run. They must not be presented as new FT or formal-v3 results.

## 10. Required monitored-run provenance

Before a new FT or MLP result is allowed to publish drift/ETG metrics, it must include:

- hashed fixed probe and Task-0 background coordinates;
- per-checkpoint probe true label, predicted class, chosen joint family, head scores, router z-scores, and joint scores;
- checkpoint hashes for the encoder, heads, normalisation, and cap3000 router state;
- per-class attribution vectors or sufficient checkpoint state to reproduce them;
- exact model, dataset, seed, checkpoint, attribution method, score target, `k`, thresholds, and denominator;
- an ETG append-only ledger containing admission mass/null, Jaccard, recall change, state transition, and action;
- code, dependency, manifest, protocol, and output hashes.

If any required field is absent, W&B must display the metric as `N/A — not measured under this protocol`, not as zero.

## 11. W&B reader-facing labels

- `Overall Accuracy / 总体准确率 (higher is better)`
- `Average Forgetting / 平均遗忘 (lower is better)`
- `Top-15 Important-Feature Overlap / 重要特征重合率 (higher is more stable)`
- `Silent Explanation Drift / 静默解释漂移 (events / eligible class-transitions)`
- `Prediction Flip Rate / 固定探针预测翻转率 (lower is better)`
- `Joint Family-Choice Change / 联合得分家族变化率 (lower is better)`
- `ETG Certified / Refused / Escalated / Strictly Re-certified`
- `Attack Recall / 攻击检出召回率` and `Benign FPR / 正常流量误报率` for NIDS datasets only
- `Macro-F1`, `Balanced Accuracy`, parameter count, training time, throughput, and peak GPU memory

## 12. Related methods to cite

1. Lundberg, S. M., and Lee, S.-I. (2017). “A Unified Approach to Interpreting Model Predictions.” NeurIPS. https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
2. Yeh, C.-K., Hsieh, C.-Y., Suggala, A., Inouye, D. I., and Ravikumar, P. K. (2019). “On the (In)fidelity and Sensitivity of Explanations.” NeurIPS. https://proceedings.neurips.cc/paper/2019/hash/a7471fdc77b3435276507cc8f2dc2569-Abstract.html
3. Gan, Y. et al. (2022). “Is Your Explanation Stable? A Robustness Evaluation Framework for Feature Attribution.” https://arxiv.org/abs/2209.01782
4. Subramaniakuppusamy, K., and Gajjar, J. (2026). “Feature Attribution Stability Suite: How Stable Are Post-Hoc Attributions?” CVPRW. https://openaccess.thecvf.com/content/CVPR2026W/XAI4CV/papers/Subramaniakuppusamy_Feature_Attribution_Stability_Suite_How_Stable_Are_Post-Hoc_Attributions_CVPRW_2026_paper.pdf
5. Geifman, Y., and El-Yaniv, R. (2019). “SelectiveNet: A Deep Neural Network with an Integrated Reject Option.” ICML. https://proceedings.mlr.press/v97/geifman19a.html
6. Jaccard, P. (1912). “The Distribution of the Flora in the Alpine Zone.” New Phytologist, 11(2), 37–50. https://doi.org/10.1111/j.1469-8137.1912.tb05611.x

The citations support attribution, stability measurement, set overlap, and reject/abstention concepts. They do **not** transfer ownership of the study-specific thresholds or the ETG state machine to those sources.
