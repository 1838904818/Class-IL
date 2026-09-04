# Independent attribution/ETG publication checks

These checks supplement the immutable v9 runtime. They do not patch a running
job, recalculate attributions, select a model, change a threshold, or replace
the original checksum and aggregate validators.

## What is checked

- The training result supplies the checkpoint/class registry, rather than
  assuming that equal method row sets are necessarily complete. The frozen
  Malaya stream has 30 checkpoint/class rows and 20 adjacent-class transitions
  per method across five checkpoints.
- Feature IDs and names are checked against the bound 77-feature schema.
- Class recall and headline predictive metrics must match the bound
  `official/joint_cap3000` training result, not an alternative evaluation arm.
- Admission is recomputed from deletion mass and the shared random-control
  threshold. Drift eligibility, Jaccard, and event flags are independently
  recomputed from adjacent rows, retaining the registered strict inequalities.
- ETG states and actions are replayed, including one-checkpoint delayed
  re-certification and terminal `UNEXPLAINABLE` behavior.
- Boolean fields are actual booleans, feature sets are unique, and numeric
  quantities must be finite. Methods must share recall, probe/background
  counts, and the random null for each row.
- Drift agreement is reported separately over all transitions and over the
  shared eligible subset. Ineligible non-events can increase all-transition
  agreement; the two denominators must not be conflated.
- A rate with zero eligible transitions is `null` (undefined), never zero.
  This tool never emits a five-seed completion claim.

## Run locally

The semantic checker and its 18 synthetic tests use Python's standard library
only. The array checker adds 16 tests and requires NumPy (release QA: 2.4.6).
The aggregate checker adds 15 tests, using the byte-preserved v9 runtime for
a synthetic golden comparison. Together the directory contains 49 tests:

```text
python -B -m unittest discover -s reproducibility/attribution_etg_checks -p "test_*.py" -v
python -B reproducibility/attribution_etg_checks/validate_semantics.py --help
```

For each real seed, supply the robustness JSON, original training-result JSON,
and feature-schema JSON using `--artifact`, `--training`, and `--features`.
Each also requires an independently checked `--*-sha256` value. Obtain these
from the verified protected checksum chain, not by blindly trusting a hash
inside the same unverified file. Supply a new `--output` file; the checker
refuses to overwrite an existing report. It also checks that the robustness
artifact binds the supplied training-result and feature-schema hashes.

Keep the reports separate from original result artifacts. Each report carries
input hashes and its validator hash. Passing this checker is **not** a
completed five-seed experiment and not evidence that any explainer is correct.

## Validation assessment, 2026-09-04

**Share with caveats for code QA; final experimental interpretation remains
pending.** All 49 synthetic tests passed. Protected seed 1 and seed 2 artifacts
passed both the original canonical/summary validator and these additional
semantic checks. No seed-level outcome is promoted as a five-seed estimate.

The original inner validator checks self-consistent method scopes and stored
summaries, but it does not independently require the complete training-derived
row registry or recompute transition facts from checkpoint rows. A synthetic,
self-consistently rehashed joint omission can therefore pass that inner check.
This does not bypass a separately trusted protected-file checksum. The new
checks address that semantic gap; the verified real seed 1 and 2 files did
not exhibit it.

The original aggregator also tries to convert an undefined zero-denominator
rate to a float. A synthetic zero-eligibility fixture reproduces that error.
This checker preserves the undefined rate explicitly. It does not silently
rewrite the original aggregate or imply the running campaign has hit that
edge case. If it occurs, any later offline aggregation revision must be
versioned, tested, and disclosed.

Uncertainty must remain at the seed level: five repeated stochastic runs on
one fixed split are not five datasets, and the 20 within-seed transitions
are not independent seed replications. An all-seed descriptive mean and
interval require the five registered seed outcomes; undefined rates need
an explicit missingness policy before any such summary is reported.

## Remaining gates and limits

Run the original aggregate validator and both supplementary checkers for
**all five** seeds before final publication. Independently verify the
result/protocol hash chain, source history, aggregate statistics, and W&B
outputs. This checker does not
rerun numerical attribution or establish cross-device/batch equivalence.
It verifies ledger arithmetic, not real human review, security-alert utility,
or causal architectural benefit. No final charts or manuscript findings are
generated by this code-validation step.

## Stored attribution arrays and probe identities

`validate_arrays.py` first runs the semantic checks, then reads only the
fixed file list below from a local flat export. It verifies each protected
file against a checksum manifest whose SHA-256 must be supplied independently.
The original training result additionally binds the probe manifest; the
robustness artifact binds the feature schema, Expected-Gradients analysis,
and both attribution archives.

Required files inside `--seed-dir`:

```text
SHA256SUMS
result_seed_N.json
attribution_robustness.json
analysis.json
attributions.npz
attribution_robustness_mean_attributions.npz
probe_manifest.json
```

Pass the authorized feature schema separately with `--features`. The
`--protected-manifest-sha256` must come from the verified protected evidence
chain. `--output` must be a new file. Use `python -B
reproducibility/attribution_etg_checks/validate_arrays.py --help` for the CLI.
The program performs no remote access, model inference, or training.

Checks cover exact archive membership, duplicate entries, bounded archive
size, non-object arrays (`allow_pickle=False`), vector shapes, finite values,
nonnegative mean absolute attributions, and the bound
`abs(mean_signed) <= mean_abs` with dtype-dependent rounding tolerance.
An independent stable Python sort recomputes the top-15 indices, breaking
ties by feature index. Every Expected-Gradients sample-ID array must match
the ordered class-specific official-test manifest, and its rows must match
the Expected-Gradients JSON copied into the robustness result.

Protected seeds 1 and 2 each passed checks over 90 rankings, 180 stored mean
vectors, and 30 sample-ID arrays. Each uses the same 1,206 fixed test probes:
one class has 54 probes and the other nine have 128 each. Background sampling
has 256 Task-0 records. These are repeated checks on a fixed probe cohort,
**not independent experiments or 2,412 unique test probes**. The original
hash-bound validation records are in `validated/seed_1_arrays.json` and
`validated/seed_2_arrays.json`; they contain no outcome-performance table.

The array checks link stored means to published rankings and sample IDs to
the bound manifest. They do not recompute sample-level SHAP, establish that
the original raw feature rows are correct, prove derivative equivalence, or
replay the deletion/random-control experiment. Those distinctions remain
necessary even when every array check passes.

## Complete five-seed aggregate arithmetic

`validate_aggregate.py` independently recomputes the registered aggregate
from exactly seeds 1, 2, 3, 4, and 42. It does not call the runtime's
aggregation functions to calculate the expected values. Tests compare it
against the frozen runtime using clearly synthetic fixtures, then alter
means, confidence bounds, pooled counts, seed counts, and source bindings
to ensure those errors are rejected. The actual five-seed aggregate is
still pending; synthetic test success is not an experimental result.

The checker verifies:

- Seed-file byte hashes and canonical hashes against aggregate bindings;
- The fixed dataset, class/checkpoint registry, thresholds, score target,
  attribution scope, common data schema, and analysis-code binding;
- All pairwise and three-method agreement statistics from the stored rows;
- Per-seed and pooled ETG/drift counts;
- Equal-weight seed means, sample standard deviations, ranges, and the
  registered descriptive t intervals (`n=5`, `df=4`);
- The five predictive metrics for the `official/joint_cap3000` arm.

The t critical constant was also checked against `scipy.stats.t.ppf(0.975,4)`
to an absolute tolerance of `1e-12`. SciPy is not a runtime dependency of this
checker. As in the registered aggregate, displayed interval bounds are
clipped to `[0,1]`; these are descriptive seed-level summaries, not an
explainer-accuracy test, a causal comparison, or evidence of generalisation
across datasets. The seed-mean drift rate is not substituted with the ratio
of pooled events to pooled eligible transitions. A seed with an undefined
rate stops the five-seed summary and requires an explicit missingness policy;
it is never changed to zero or omitted to produce a four-seed answer.

CLI inputs are `--aggregate`, its independently verified `--aggregate-sha256`,
`--inputs`, and a new `--output` file. The input registry is a JSON list of
exactly five records, each containing `seed`, `path`, and `sha256`. Relative
paths resolve beside that registry. A partial registry is rejected before
reading the aggregate, and no success report is written. Paths and hashes
must be populated from the protected final evidence, not synthetic examples.

Passing arithmetic verification is not automatic publication approval.
All five semantic and array checks, training/protocol provenance, W&B
verification, and interpretation review remain required. This stage does
not treat within-seed transitions as independent seed replications, nor
equate Malaya application-classification evidence with NIDS attack metrics.
