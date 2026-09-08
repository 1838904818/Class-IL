# Local telemetry repair, not an approved job

## Update: GPU identity and bounded initialization

The subsequent Job 454585 failed after four seconds with exit 126 at telemetry
preflight. Its protected status reports `no_device` and query return code 6;
the verifier did not start and no online tracking run was generated. Three
protected core artifacts passed checksum verification after local retrieval.
This is not a scientific fidelity failure and not an OOM.

The former assumption that Slurm numeric GPU IDs could be used as NVML indices
is unsafe; see [Slurm GPU management](https://slurm.schedmd.com/gres.html).
The exact physical cause of the failed job remains unproven. Current local
source resolves the sole process-visible CUDA device UUID in a sequential
helper with a 10-second timeout, then queries that UUID only. No numeric
fallback, whole-node enumeration or visibility override is used. Native CUDA
return types are explicit. Identity errors stop before verifier launch and
produce protected, redacted failure records. The recorded measurement scope
now explicitly describes UUID-only selection.

The coordinator suite contains 37 synthetic tests: 36 passed and one platform
skip. These tests include mocked native API failures and protected timeout
paths; they do not establish successful CUDA execution on the cluster.
Independent review accepted the local candidate only. Remote checks and a new
explicit submission authorization are still required. No manuscript result
or novelty claim follows from this infrastructure repair.

The following account documents the earlier environment-preservation repair.
Its local-only statement describes that historical test stage, not the later
failed Job 454585.

The historical-fidelity prerequisite did not complete. Job 453142 ended on
7 September 2026 at 07:17:56 UTC after 27 seconds with Slurm FAILED 126:0.
The coordinator recorded TELEMETRY_FAILURE, three CalledProcessError events,
zero resource samples and zero of twenty checkpoint comparisons. This was
an infrastructure failure, not a completed numerical mismatch or a new
accuracy result. Accounting reported 3.85 GB peak memory out of 4 GB and
12.96% CPU efficiency. There is no measured GPU-utilization pass or OOM diagnosis.

The protected core-manifest SHA-256 is
`0d08a31ba62389387208d58234b91fb63571f6390d32c04312c075942b16283b`.
The protected GPU_FIDELITY.json SHA-256 is
`b97bee0c80a92872cfbec625299c267e897c9005b7c977d1a0ad03ae284df745`.
The recorded [tracking URL](https://wandb.ai/csnet/ofra-etg-leon-hpc/runs/uf7hiqzc)
is not independently cloud-verified: the client-finish receipt returned, but
the separate read-only API check failed authentication. No credential content
is included here. The terminal monitoring task was stopped.

## Repair implemented locally

The original coordinator preserves allocation environment for its verifier,
but removes most environment fields before starting the tracking client. Its
GPU query accidentally inherited that reduced environment. PATH was retained;
driver/library context such as LD_LIBRARY_PATH was not. This is a plausible
cause of the nonzero query, not a proven diagnosis of the actual device fault,
because the original error return code and output were not retained.

The revised coordinator:

- Explicitly passes the preserved compute environment to the query of the
  already allocated GPU. It never queries other users' GPUs or node capacity.
- Requires a successful bounded telemetry query before launching the verifier.
  A query failure therefore does not start a memory-heavy scientific child.
- Records the failed stage, return code, bounded output-prefix hashes and a
  fixed allowlist of diagnostic tags. It never logs raw output, environment,
  credentials or command lines.
- Keeps tracking in its minimal environment and preserves original scientific
  tolerances, checkpoint coverage, utilization guard and memory guard.
- Requires a new v3 operation contract with 8 GiB proposed memory, instead of
  silently accepting the previous 4 GiB contract. This provides headroom above
  the observed 3.85 GB peak, but remains a proposal for a measured profile,
  not proof that all twenty checkpoints fit or a request for an idle node.

Only mocked subprocess/SDK tests were run. There was no CUDA query, remote
write, upload, submission or resubmission. The old approved candidate remains
unchanged. This directory does not contain an approved operation binding or
sbatch file and cannot be used as submission approval. A fresh package must
bind the unchanged verifier and scientific inputs plus this changed code and
resource request; then pass current policy/live checks, independent review
and explicit one-use submission confirmation.

```text
python -B -m unittest discover -s . -p test_replayids_fidelity_profile.py -v
```

Preflight success establishes query availability only. Scientific fidelity
requires the subsequent actual GPU experiment and immutable verified results.
