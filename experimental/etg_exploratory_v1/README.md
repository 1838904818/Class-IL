# Early-checkpoint ETG exploratory preparation

11 September 2026. **Metadata planner and synthetic decision-kernel checks only.
No completed intervention study, model inference or extraction of real SHAP.**

[Protocol](PROTOCOL.md) | [Candidate contribution and related work](../../docs/ETG_NOVELTY_AND_FALSIFICATION_2026-09-11.md)

This separately versioned descriptive pilot does not replace or relax
`parallel_execution/repair_controls`. It implements row-role planning,
method-noise diagnostics, five simple selectors and an empirical rollback
predicate. It is deliberately **not an end-to-end executable experiment**.
The selector's `attribution_validated` input is an internal caller precondition,
not an evidence-verification API; setting a Boolean never establishes fidelity.

Synthetic tests:

```powershell
python -B -m unittest discover -s experimental/etg_exploratory_v1 -p test_pilot.py -v
```

The metadata-only planner takes the existing research workspace root, verifies
native source/audit/checkpoint file bindings, reconstructs historical calibration
indices, and emits role digests to stdout. It does not open model/feature tensors,
create feature partitions, fit anything, contact HPC or write an experiment run.

```powershell
python -B experimental/etg_exploratory_v1/prepare_plan.py --workspace-root <research-workspace>
```

The planner's local input layout is a development locator convention, not a
publicly downloaded dataset bundle. Required files and hashes are recorded in
the plan and source. Independent execution review still requires full training
lineage, previously used-label accounting, exact score and perturbation fidelity,
a fixed explainer/action recipe, measured resources and a one-use driver.
