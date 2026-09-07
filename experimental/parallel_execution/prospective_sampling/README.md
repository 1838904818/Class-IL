# Executable prospective train-array derivation

Status: implemented array builder/loader, verified on temporary synthetic
arrays only. No real arrays or real-data training were processed, and no HPC
or Git operation was performed for this package. This extends, but does not
modify or replace, the earlier synthetic-ID reference or historical D2 data.

The implementation does real `.npy` validation, disk-backed ID/rank selection,
chunked memmap extraction, provenance-array writing, immutable completion and
resume validation. It is not a schema-only planner. NumPy and the Python
standard library are the only runtime dependencies.

`prepare_l1_inputs.py` additionally implements the actual cross-line, offline
fit-only bridge to `fair_comparison/materialize_embeddings.py`. Its integration
tests run that frozen-encoder exporter and the L1 input loader on small arrays.
One uses an explicitly untrained MLP interface fixture. A second actually runs
the shared Task-0 pretrainer, O/P bridge, embedding export and L1 fitting/audit
on tiny synthetic arrays. Neither is real-data experimental evidence. The
bridge itself requires only NumPy/stdlib; cross-line inference and fitting
tests also require PyTorch. No local full-data load occurs.

## Fixed comparison and concrete input

`REPLAYIDS_INPUTS.json` binds the existing ReplayIDS source manifest and lists
all eight exact training shards and eight independent official-test shards.
The metadata file's hash was rechecked locally. The array hashes listed there
are historical bindings, **not freshly verified array results**. Their actual
verification is performed by the executable reader in the authorized stage.

The shipped `policy.P.json` uses sampling seed 42, deterministic 10% per-class
training calibration and a Task-0-frozen normal cap. `policy.O.json` is the
explicitly offline bridge: normal cap 124,780, from historical all-class fit
counts, and exactly the same new RNG/calibration procedure. It must never be
called a prospective independently chosen constant. The adapter preserves
the existing task order `[0,1], [2,3], [4,5], [6,7]` and class semantics.

Both arms use the **P fitting IDs at Task 0** as their common transform-fit
cohort. This cohort is a subset of both arms' fitting rows, includes no
calibration/test rows, and is frozen by identical x/label/ID file hashes.
O rejects a cap smaller than the P cap. This resolves the local transform-fit
identity choice without consulting future classes or test outcomes. The
reader writes raw features; the trainer must fit its numerical transform on
this common cohort and reuse the resulting transform/encoder state. It must
not silently refit the transform on O's larger normal pool.

The raw export bridge requires one checkpoint for both arms, bound to the
common Task-0 collection and exact normalization/encoder-training row-ID file
hashes. An old checkpoint lacking these bindings cannot silently be paired
with a newly fitted scaler. `fair_comparison/pretrain_task0.py` is the separate
common-cohort pretraining stage; actual checkpoint generation and measured
resource evidence remain controlled-execution dependencies.

Expected support is count arithmetic only: P cap 5,559 versus O cap 124,780,
P fitting rows 149,476 versus O 268,697, calibration rows 68,313 and unchanged
official-test support 227,723. Actual derived hashes, resource cost, accuracy
and FPR are unknown. Less normal fitting support can worsen detection/FPR.

## Run the small local tests

From this directory:

```console
python -B -m unittest discover -s . -p "test_*.py" -v
```

Tests create small temporary arrays under this directory and remove only their
own temporary outputs. One test uses 10,025 synthetic rows to verify that the
old 10,000-ID reference restriction is absent. It is not a real-data profile.
Real symlink creation can be skipped if the host lacks permission; the
Windows reparse-point rejection branch also has an independent simulated
stat test. Do not promote that simulation to a deployed-filesystem claim.

## Execution boundary

See `INTERFACE.md` for exact input/output fields, data isolation, failure and
resume semantics. See `EXECUTION.md` for the already selected adapter and
increment commands. Paths in those examples are symbolic locations inside an
authorized compute workspace, not user-specific paths or an authorization to
run preprocessing on a login node.

The next required work is **controlled real-source validation and derivation
profiling/execution**, not implementing the local production builder. Its
measured peak memory, disk use, latency, source hashes and completed-output
closure must be collected before training approval. Capture/group provenance,
data authorization and leakage beyond stable source row identity remain
external dataset conditions. They cannot be established by software tests.
