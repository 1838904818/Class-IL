# Final-checkpoint calibration control

The [verified aggregate package](../results/p-seed1-score-calibration-controls/README.md) reports an exploratory P seed-1 independent-head calibration comparison. It is not native routed OFRA, not a multi-seed conclusion, and does not measure forgetting.

Original and probability-score temperature predictions have 87.9349% accuracy and 37.6438% Macro-F1. Class offsets reduce these to 76.6159% and 10.8748%. Lower calibration NLL therefore did not establish improved classification. Both loss arms share these final predictions but have different score matrices and fitted parameters.

The package includes per-class recall, precision, F1, support, predicted counts, confusion matrices, attack recall and Benign FPR, with explicit denominators and absent-class conventions. See its README for the fixed optimization protocol, data partition sizes, provenance digest and score-range limitation of the offset control.

These findings do not establish algorithmic novelty, eliminate the need for conventional calibration comparisons, or demonstrate operational ETG benefit.
