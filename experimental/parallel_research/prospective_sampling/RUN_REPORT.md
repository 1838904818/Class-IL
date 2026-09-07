# Local verification report

Date: 2026-09-07. Scope: synthetic row identifiers only. Runtime: Python 3.11.9.

Executed from the workspace root:

```console
python -B -m unittest discover -s work/parallel_research_20260907/prospective_sampling -p "test_*.py" -v
```

Result: **33 tests passed**, zero failures and errors. The observed test-run
duration was 0.042 seconds; this is not a real-data performance measurement.
No additional dependencies or training framework were installed or imported.

Coverage includes:

- frozen Task-0 cap; fixed cap without data-dependent fallback;
- changed/added future classes and replaced future/test registry metadata do
  not change the selector's admissible prefix output;
- direct full-world/future-row and future/test argument rejection;
- deterministic seeds, input-order invariance, train digest recorded without
  entering the random domain, and same-new-RNG cap-only bridge invariants;
- disjoint/conservative fit, calibration and omitted sets; all attack fit
  rows retained; empty/singleton/two-row support handled explicitly;
- row-ID and class-ID collisions across current and historical cohorts;
- immutable historical selections, consecutive increments, strict roles and
  train-only inputs, invalid configuration, and a 10,000-ID reference limit.

The external registry replacement test intentionally never passes registry
metadata to the selector. Together with direct forbidden-argument rejection,
it tests the API boundary, not a production ingestion pipeline. No claim is
made that a real loader or release ledger enforces that boundary yet.

A targeted scan of the package found no absolute personal filesystem paths,
credential assignments, common access-token forms or private-key headers.
That scan supplements review; it is not a guarantee that every possible
secret representation is detectable. Public text is English and source
identifiers are workspace-relative. Nothing was committed or pushed.

## Evidence not produced

- No real training or official-test array read or independently rehashed.
- No production builder/loader, full dataset derivation, or matched model run.
- No GPU execution, live DICC check, upload, sbatch script, or job submission.
- No training resource profile or conclusion about accuracy, FPR or utility.
- No replacement of historical D2 data, model results, or manuscript claims.

`REFERENCE_BINDINGS.json` binds the five authored core files and records the
local test status. `SOURCE_AUDIT.md` binds the inspected historical sources.
Local tests do not satisfy the actual derived-data or HPC submission gates
specified in `PROTOCOL.md`.
