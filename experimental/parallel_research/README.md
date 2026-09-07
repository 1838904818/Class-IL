# Parallel research preparation

7 September 2026. **Local reference implementations and synthetic tests only.**
This package implements three independent preparation tracks. It is not a
new training result, approved cluster candidate, preregistration of historical
experiments, or proof of algorithmic novelty. Historical manuscripts, data,
checkpoints and published metrics are unchanged.

## Questions and deliverables

| Track | Research question | Local deliverable | Evidence still needed |
| --- | --- | --- | --- |
| [Fair comparison](fair_comparison/README.md) | Does any advantage survive controlled loss, exposure, capacity and memory accounting? | Source-grounded confound audit and fail-closed accounting checks | Instrumented, protocol-matched training and performance/resource measurements |
| [Prospective sampling](prospective_sampling/README.md) | Can early sampling decisions avoid future-class and test-metadata dependence? | Prefix-local selection reference, deterministic partitioning and invariance tests | Bound real-data derivation, integration with the loader, then matched training |
| [Repair controls](repair_controls/README.md) | Does an expansion-aware trigger improve a fixed repair budget beyond simple triggers? | Partition, evidence and budget checks plus a trigger reference | Faithful scores/attributions, implemented repair action and later held-out utility |

## Parallel work versus experimental dependencies

All three preparation tracks can run locally without the historical GPU
fidelity result. Fair-comparison instrumentation and prospective data
derivation also have no scientific dependency on historical attribution.
They must nevertheless obtain their own input bindings and execution review.

The ReplayIDS historical-score gate remains separate. Its success cannot
establish Malaya gradient faithfulness, architecture superiority or ETG
intervention utility. Conversely, a failed historical score reconstruction
does not invalidate unrelated synthetic partition/accounting tests.

Real-data use of the expansion diagnostic requires a declared, faithful score
function. Attribution-dependent triggers additionally require target-specific
attribution checks. Do not silently change legacy arithmetic to obtain a pass.

### Minimal staged experiment order

1. Complete the three local references and negative tests. Missing information
   must produce an explicit blocked/pending result, never fabricated measurements.
2. Derive a new versioned, increment-available sampling manifest in a separately
   approved CPU job. Preserve the original official test and source data.
   Compare the cap policies using the same prefix-local randomization; a direct
   old-versus-new comparison otherwise changes two factors at once.
3. Instrument a bounded matched-training pilot. A shared backbone and equal
   epochs or exemplars are not sufficient: report objective, optimization
   steps, row exposures, supervised targets and cumulative memory separately.
4. After scorer/attribution checks, run one fixed repair operation under the
   different triggers with isolated fitting, acceptance and later evaluation
   data. Only then consider expanding datasets/seeds or claiming utility.

Stages 2 and 3 may have independent branches; a training branch using newly
derived data must wait for that branch's actual manifest hashes. No automatic
job chain, array, upload, retry or submission is implemented by this package.
No resource request is justified by synthetic test duration.

## Review coverage and reporting

The machine-readable [dependency register](DEPENDENCY_GATES.json) maps these
tracks to the current review issue IDs. Preparing a validator does **not**
close an empirical reviewer request. In particular, matched capacity,
loss/exposure, no-look-ahead training and ETG utility remain to be tested.

Existing historical seeds 1, 2, 3, 4 and 42 and the screening role of seed 42
remain explicit. A future confirmatory study needs its own frozen selection,
split and multiplicity policy; collecting more seeds until significance is
not an acceptance rule. Rare-class support and uncertainty must be reported.
Malaya application labels must not be interpreted as attack/benign labels.

The proposed expansion-aware diagnostic is an order-specific decomposition,
not a causal attribution theorem. Explanation drift in class-incremental
learning is already studied by [CLEX](https://doi.org/10.1016/j.neucom.2024.127960).
Budgeted drift-to-action control is already studied by
[Drift2Act](https://arxiv.org/abs/2603.08578), a CAO Workshop paper at ICLR 2026.
The present package neither reproduces these methods nor establishes a new
priority claim. Its hypothesis is narrower: separating expansion mechanisms
may improve the allocation of a fixed repair budget. That hypothesis can fail.

## Local verification

Run each track's tests using its documented command. The release checker runs
only these small tests and checks scope/dependency metadata:

```text
python -B run_local_checks.py --output /your/local/check-report.json
```

The report distinguishes local tests from real-data integration and execution.
It must never be used as evidence of cluster approval, a trained model, a
useful repair, or a safe deployed system. No credentials or remote commands
are required. This package does not modify the prediction stream or the
historical ETG ledger.
