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
