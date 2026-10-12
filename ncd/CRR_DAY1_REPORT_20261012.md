# CRR Section 12 day-1 report (2026-10-12)

## Decision

**CONTINUE to the preregistered experiments.** The bounded primary-source
search in `LITERATURE_DIFFERENCE_TABLE.md` found no prior class-incremental
method that both preserves feature-intervention/counterfactual dependence only
for old-vs-old class margins and deliberately leaves old-vs-new margins
unconstrained. This clears only the Section 12.4 veto. It does not establish
CRR effectiveness, substantive innovation, or publication readiness.

## Delivered implementation

- `crr_benchmarks.py`: hijack and necessity constructors plus all eight matched
  learners. The necessity constructor preserves the selected top-1 feature,
  shifts the second-ranked feature by the registered number of standard
  deviations, and records both pre/post dependence.
- `run_crr_experiment.py`: exact matrix runner with source-commit receipts.
- `select_crr_lambda.py`: seed 5/6-only lambda selection using a fixed rule.
- `aggregate_crr_section12.py`: exact five-seed paired test and fail-closed
  Section 12.7 report; the real-sequence gate remains pending until day 10.
- `hpc/run_crr_section12.sbatch`: archive/hash verification, remote import
  smoke, protected-result receipts, and separate per-dataset full jobs.
- `tests/test_crr_benchmarks.py`: four construction/restriction tests; 4/4
  passed locally.

## Exact-commit local smoke

Source commit: `f419b78a7c524735b25ab38ce1c261d2997cac68`. Configuration:
seed 1, one base epoch, one transition epoch, 512 rows/class, rho=0.9, k=2,
all eight methods, both datasets. All 32 expected rows completed. The table
below reports CRR only; it is an execution smoke, not a Section 12.7 result.

| Dataset | Benchmark | Macro-F1 | Primary dependence/confusion endpoint | Supporting endpoint | Cost / Replay |
|---|---:|---:|---:|---:|---:|
| NSL-KDD | hijack | 0.4394 | marker recall drop 0.0000 | decorrelated target recall 0.6934 | 2.01x |
| NSL-KDD | necessity | 0.5349 | c-d confusion 0.4131 | c/d recall 0.6563/0.3965 | 1.72x |
| UNSW-NB15 | hijack | 0.5497 | marker recall drop 0.0098 | decorrelated target recall 0.6035 | 1.78x |
| UNSW-NB15 | necessity | 0.3304 | c-d confusion 0.4766 | c/d recall 0.9551/0.0020 | 1.46x |

The one-epoch necessity smoke did not demonstrate the required plasticity:
the measured c-to-f2 dependence change was -0.00284 (NSL-KDD) and -0.0000571
(UNSW-NB15), and the UNSW synthetic-new-class recall was near zero. These are
explicit warnings, not conclusions; the registered multi-epoch calibration
and five-seed comparison decide the hypothesis.

## HPC state

- Job 485030: two-dataset, eight-method seed-1 GPU smoke; submitted and queued.
- Job 485032: NSL-KDD seed-5/6 lambda calibration followed by the registered
  8-method x 5-seed x 7-cell matrix; dependency `afterok:485030`.
- Job 485033: UNSW-NB15 matching calibration and full matrix; dependency
  `afterok:485030`.

All three jobs use the hash-verified archive for commit `f419b78`, including
both `ncd/` and the top-level `src/` package. Each has an exact-job read-only
heartbeat. The day-5 benchmark verdict remains pending terminal evidence from
485032 and 485033; the complete Section 12.7 verdict additionally remains
pending the registered real-sequence evidence scheduled for day 10.
