# Section 15 execution results (2026-10-13)

Live Slurm accounting and the scoped stderr logs were rechecked on 2026-10-13 (Asia/Kuala_Lumpur).
Registered source commit: `bfafe508e1bb2a6d2d4cb9c178e37123086ddb2f`.

| Job | Dataset / matrix | State | Exit | Elapsed | Batch MaxRSS |
|---|---|---|---|---|---|
| 485227 | NSL-KDD necessity | FAILED | 1:0 | 01:20:02 | 7600520K |
| 485228 | UNSW-NB15 necessity | FAILED | 1:0 | 02:35:31 | 7524228K |
| 485229 | NSL-KDD real sequence | PENDING, DependencyNeverSatisfied | not terminal | 00:00:00 | unavailable |
| 485230 | UNSW-NB15 real sequence | PENDING, DependencyNeverSatisfied | not terminal | 00:00:00 | unavailable |

Both necessity stderr logs identify the same collection failure:

```text
collect_crr_section15_lambdas.py, line 9
from ncd.crr_benchmarks import METHODS
ModuleNotFoundError: No module named 'ncd'
```

Evidence locations on DICC: operation-log directory
`operation-logs/20261012-crr-section15-v1`, with exact stderr basenames
`slurm-ncd-crr-s15-nsl-n-485227.err` and `slurm-ncd-crr-s15-unsw-n-485228.err`.
At the root of each protected necessity directory, only `JOB_EXIT.json` was found;
no root `COMPLETED.json`, `SHA256SUMS.txt`, or merged matrix artifact was available.

The registered Section 15 matrix has not completed. Its Section 12.7/15.3
scientific criteria cannot be adjudicated from these execution records.
Successful calibration subprocesses do not establish a completed five-seed
matrix. The previously recorded local final hijack gate remains failed.
No claim of CRR superiority, substantive innovation, or publication readiness
is supported by this run.

Next step: repair the collector import invocation and resume the registered
matrix with complete logs and source/config/seed provenance.

## Recovery submitted

The collector now inserts the repository root before importing `ncd`, including
under Python isolated mode. Its isolated command passed locally. The batch
script passed local and live account/association preflight and remote Bash
syntax checks. The cluster client has no `sbatch --test-only` support.

Recovery source: `bfde946a44ea7838f07399cf14c7570a40df7dca`.
The benchmark, runners, configuration and lambda-selection code are unchanged
from the original source. Saved calibration JSON and logs retain that original
provenance; the recovery validates their source, dataset and row count before
recollecting lambda choices. Completed matrix shards can be resumed with
`NCD_RESUME_ROOT`; incomplete cells rerun. Each job requests one A100, two CPUs,
16 GB RAM and a three-day maximum walltime, based on prior measured memory.

| Job | Matrix | Immediate verified state | Dependency |
|---|---|---|---|
| 485405 | NSL-KDD necessity | PENDING (Resources) | none |
| 485406 | UNSW-NB15 necessity | PENDING (Resources) | none |
| 485407 | NSL-KDD real sequence | PENDING (Dependency) | afterok:485405 |
| 485408 | UNSW-NB15 real sequence | PENDING (Dependency) | afterok:485406 |

Old jobs 485229 and 485230 were cancelled after confirming their failed
dependencies. New jobs have ten-minute monitors and a completion aggregation
automation. Scoped logs and protected outputs use run directory
`20261013-crr-section15-v2`. No new scientific results are available yet.

## Section 16 execution correction

The updated work order was pulled from `abe845d`. Jobs 485407/485408 were
cancelled and replaced by independent real-sequence jobs 485498 (NSL-KDD) and
485499 (UNSW-NB15), with no Slurm dependency. Both were immediately verified
pending. HPC source is `5ddc84ba840b8f3004447652eb6a89850f4369f6`;
lambda receipts retain source `bfde946a44ea7838f07399cf14c7570a40df7dca`.

Before each submission the same batch entry was executed locally in
`NCD_LOCAL_SMOKE=1` mode using the RTX 4060 Ti CUDA environment. Both datasets
completed all eight methods, seed 1, one epoch, at most 64 training/test rows
per class, and produced final `real_result.json`, CSV, PI-drift CSV and
`COMPLETED.json`. These smoke outputs are integrity checks, not scientific
comparison evidence. Smoke sampling exposed and fixed an immutable-data
assignment error before submission.

The shared matrix entry is `ncd/run_crr_real_matrix.py`. Formal local execution
uses the full registered configuration, NSL-KDD first then UNSW-NB15, with
source fixed at startup. The active local results are under
`ncd/results/crr/section16/local_v2`. Earlier partial cells are preserved under
`section16/local`. The first complete local or HPC dataset result will be used;
the other execution is a replication record. Ten-minute recovery monitors
now authorize prompt implementation repair with end-to-end smoke before retry.
