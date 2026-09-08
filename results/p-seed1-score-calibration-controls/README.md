# Exploratory final-checkpoint score calibration

9 September 2026. Verified output consistency from saved score matrices, parameters, predictions and labels. This is a frozen-representation independent-head L1 control, not native routed OFRA, and not a multi-seed or forgetting experiment.

Both CE and conditional-focal arms report the following classification results. Their original scores, head parameters and fitted calibration parameters differ; identical predictions do not establish loss equivalence.

| Control | Accuracy (%) | Macro-F1 (%) | Balanced accuracy (%) |
|---|---:|---:|---:|
| Original | 87.9349 | 37.6438 | 35.2265 |
| Probability-score temperature | 87.9349 | 37.6438 | 35.2265 |
| Probability-score offsets | 76.6159 | 10.8748 | 12.5148 |

The calibration partition contains 68,313 rows; the unchanged test partition contains 227,723 rows. Calibration uses unweighted mean multiclass NLL and no test labels. Scores are independent binary positive probabilities used as relative scores, not multiclass logits. Temperature uses T in [0.01,100]; class offsets fix class 0 to zero and bound the other offsets to [-10,10]. L-BFGS-B uses maxiter 500, ftol 1e-12, gtol 1e-8 and maxls 40. These are fixed implementation choices, not optimized scientific claims.

Positive global temperature preserves argmax. The offset control predicts Benign for 227,660 test rows and class 2 for 63. Although calibration NLL decreases, classification deteriorates. Offsets below -1 cannot beat a zero-offset Benign head when raw scores lie in [0,1]; all other attack-class offsets satisfy this condition. Class 6 reaches the lower bound. This negative control does not demonstrate that conventional calibration is generally inadequate or that a proposed new mechanism is superior.

Metrics in aggregate.json use fractions. Per-class recall is TP/support, not one-versus-rest accuracy. Precision uses predicted count. Macro-F1 averages the fixed class axis, zero denominator contributing zero; balanced accuracy averages positive-support recalls. Attack recall detects any non-Benign prediction among true attacks, even if subtype is wrong. Benign FPR uses true Benign rows. Final-only results do not measure forgetting.

The test set had previously been inspected; this is exploratory. Independent-capture validation, native architecture fairness, O-arm training and multi-seed confirmation remain separate requirements. ETG intervention benefit is not established here. Aggregate records contain provenance digests but no raw network rows, local paths, or presentation notes.

Background: [Guo et al., On Calibration of Modern Neural Networks, ICML 2017](https://proceedings.mlr.press/v70/guo17a.html). The probability-score controls here must not be conflated with standard multiclass-logit temperature scaling.
