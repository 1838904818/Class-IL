# Experiment Scope Freeze — 2026-08-13

## Primary method

The primary method remains the existing FT-Transformer OFRA prediction
pipeline followed by offline, post-hoc SHAP and ETG analysis. ETG does not
change later routing, training, or prediction decisions.

The following registered components remain unchanged:

- one frozen Task-0 encoder followed by per-class family heads;
- the existing negative-sampling, exemplar, and focal-loss protocol;
- DP-means cap-3000 and uncapped router arms;
- `joint = head probability + 0.5 * router population z-score`;
- the existing dataset-specific preprocessing contracts and class order;
- the registered routed-margin explanation target;
- the primary silent-drift cell: top-15, Jaccard below 0.70, and recall drop
  greater than -0.05;
- offline ETG admission, refusal, escalation, and strict recertification rules.

## Additive comparison arms

Two classifier comparisons are allowed without replacing the primary method:

1. `TabMMeanEncoder` is a minimally invasive OFRA backbone comparison. It
   averages TabM member embeddings to satisfy the existing single-embedding
   OFRA interface. It does not change family heads, routers, joint weighting,
   memory budgets, or ETG rules. It must be reported as a mean-embedding TabM
   adapter rather than a full member-wise TabM-OFRA redesign.
2. CatBoost is a cumulative multiclass diagnostic. Each checkpoint retrains
   on all seen-class training rows. It is not an OFRA arm, and native CatBoost
   SHAP is not the routed-margin SHAP protocol or an ETG result.

## Allowed work after the freeze

- run the fixed configurations on registered datasets and seeds;
- complete missing seeds, ablations already named in the protocol, and
  attribution-method robustness checks;
- reproduce results and verify hashes, checkpoints, metrics, and W&B records;
- fix implementation defects that do not change the registered mathematical
  method, with regression tests and an explicit result-equivalence check;
- improve documentation, tables, and figures without changing claims.

## Changes requiring a new protocol version

The following are outside this freeze and require a separately named method,
new configuration and code hashes, fresh pilot evidence, and a new review:

- a new router representation or routing algorithm;
- a different joint-score formula or router weight;
- new memory or exemplar rules;
- feedback from ETG into OFRA routing or training;
- CatBoost leaf embeddings, CatBoost family heads, or a CatBoost-OFRA hybrid;
- changing the primary SHAP target or ETG thresholds;
- selecting hyperparameters from the official test view and presenting the
  resulting run as confirmatory evidence.

This freeze keeps classifier comparisons additive and prevents exploratory
models from silently changing the paper's primary method.
