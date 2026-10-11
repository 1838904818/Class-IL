# Exploratory Exp 3 saturation diagnostic

This analysis is post hoc and does not establish a confirmatory detector or substantive innovation.
Thresholds are learned only from the frozen Exp 2 controls, and cross-cell results hold out one
dataset--learner cell at a time.

| Rule | Exp 2 negative FP | Exp 2 eligible positive detection | Exp 3 silent drift |
|---|---:|---:|---:|
| B2_repeat_mean | 0/104 (0.00%) | 34/34 (100.00%) | 692/1088 (63.60%) |
| NCD_current | 1/104 (0.96%) | 34/34 (100.00%) | 1076/1088 (98.90%) |
| noise95_absolute | 0/104 (0.00%) | 34/34 (100.00%) | 1076/1088 (98.90%) |
| noise95_relative_within | 0/104 (0.00%) | 34/34 (100.00%) | 1069/1088 (98.25%) |
| positive_anchor05_absolute | 0/104 (0.00%) | 29/34 (85.29%) | 690/1088 (63.42%) |
| positive_anchor05_relative_within | 0/104 (0.00%) | 29/34 (85.29%) | 693/1088 (63.69%) |

The minimum attainable permutation p-value occurs in 5351/5565 (96.15%) Exp 3 explainer-level rows.

## Interpretation

The negative-control 95th-percentile effect gate preserves all held-out eligible positives,
but it does not materially reduce Exp 3 saturation.  A stronger positive-anchored gate
reduces the Exp 3 rate, but it misses held-out eligible positives and therefore fails the
registered requirement of matching B2 sensitivity.  The current evidence does not support
promoting either gate as a validated replacement.

The fastest defensible next experiment is a fresh, frozen validation set containing subtle
interventions near the decision boundary and additional independent negative controls.  No
new full Exp 3 sweep is justified before that selective-validity gate is passed.
