# Attribution-method robustness audit

## Scope

This source-bound pilot reuses the completed MalayaNetwork_GT seed-1 training
checkpoints and ETG evidence. It changes only the attribution method. All
primary methods explain the same `joint_cap3000` class margin on the same fixed
official-test probes and checkpoints, then apply the same top-15 ranking,
random-control admission threshold, recall tolerance, Jaccard threshold, and
ETG state machine.

The three primary methods are:

1. Expected Gradients, from the registered SHAP analysis;
2. single-feature ablation to the frozen checkpoint mean;
3. Gradient x Input on the routed class margin.

This is a single-seed method-dependence pilot, not a multi-seed estimate and
not a search for a uniquely correct explainer.

## Agreement results

| Pair | Mean top-15 Jaccard | Admission agreement | ETG-state agreement | Silent-drift event agreement |
|---|---:|---:|---:|---:|
| Expected Gradients vs feature ablation | 0.567 | 56.7% | 60.0% | 40.0% |
| Expected Gradients vs Gradient x Input | 0.380 | 63.3% | 60.0% | 35.0% |
| Feature ablation vs Gradient x Input | 0.405 | 60.0% | 56.7% | 65.0% |

Across all three methods, the admission decision agrees on 40.0% of the 30
checkpoint-class rows, the ETG state agrees on 43.3%, and the silent-drift
conclusion agrees on 20.0% of the 20 adjacent class transitions.

## Method-conditioned outcomes

| Method | Admitted rows | Silent-drift events / eligible transitions | Certified admissions | Refusals | Escalations |
|---|---:|---:|---:|---:|---:|
| Expected Gradients | 12 / 30 | 12 / 17 | 6 | 4 | 4 |
| Feature ablation | 13 / 30 | 0 / 17 | 5 | 5 | 0 |
| Gradient x Input | 3 / 30 | 7 / 17 | 1 | 9 | 0 |

The large change in drift and governance outcomes means that ETG conclusions
are method-conditioned. Expected Gradients remains the registered operational
explainer, but the current evidence does not justify treating it as ground
truth. A deployment-oriented extension should require method consensus,
uncertainty-aware admission, or independent calibration.

## Integrated-Gradients diagnostic

Integrated Gradients were attempted with the frozen Task-0 background mean and
16-point Gauss-Legendre quadrature. The completeness error remained excessive:
the mean of row-level mean absolute errors was 11.58 and the maximum absolute
error was 373.72. The complete routed scorer contains a piecewise routing path,
so a differentiable attribution method cannot be assumed numerically valid for
that full score. Integrated Gradients are retained in the machine-readable
artifact as a failed diagnostic and are excluded from the primary three-method
agreement statistics.

## Reproducibility

- Script: `formal_v2_explanation_etg/attribution_robustness.py`
- Machine-readable result:
  `results/malaya-network-gt/attribution-robustness-seed1/attribution_robustness.json`
- Mean-attribution archive:
  `results/malaya-network-gt/attribution-robustness-seed1/attribution_robustness_mean_attributions.npz`
- Result canonical SHA-256:
  `f55d5817445475340f095a96f07a92c29068a7460f150f73aa62ad633781f263`
