# Paired extraction preparation components

Status: locally reviewed components; scheduled integration is not complete.

These components prepare the existing retrospective early-transition protocol:
64 old-class targets, two checkpoints, eight shared permutation seeds
(20270000 through 20270007), 32 Task-0 references and 78 features.
They do not produce an ETG efficacy result or authorize a new experiment.

- chunk_core.py: deterministic schedule, immutable identity checks, exclusive
  record writes, independently checked commit markers, and a complete
  1,024-record Cartesian gate. Partial, corrupt or unreviewed records block reuse.
- chunk_inputs.py: retains the original per-class 512-row context, including
  its actual tail length, and verifies copied rows against bound source data.
  Reference rows are identical across targets and checkpoints.
- chunk_monitor.py: bounded own-step Slurm accounting (at most three RPCs),
  optional own-step cgroup-v2 observations and interval-based utilization.
  The rolling window is time-weighted and clipped to exactly 60 seconds.
  Process RSS is not treated as allocation-level memory utilization.

CPU-only synthetic tests: 32 pass and one real symlink test is skipped on the
tested Windows system because symlink creation was not permitted. This includes
a full 1,024-record write/commit/read-back test. The skipped case remains
unverified on the target Linux allocation. No scientific dataset or model was
used in these tests.

Run from this directory with NumPy available:
`python -m unittest discover -p 'test_*.py' -v`

The input loader's real-data path additionally requires the separately bound
native profile and historical runtime modules. Synthetic tests inject a small
array loader; that is not evidence of real-data binding or historical parity.

The supervisor must enforce worker success before commit, exact scientific
and input hashes, protected archival, limited single-worker execution,
timeouts, monitoring failures and the reviewed resume list. These obligations
are not implemented by these component files alone. The isolated worker and
complete Slurm launcher remain outside this published component release.

Slurm field semantics and RPC caution:
[Official sstat reference](https://slurm.schedmd.com/sstat.html).
