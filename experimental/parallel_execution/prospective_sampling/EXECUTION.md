# Selected real derivation stage: awaiting controlled execution

This file identifies the next executable work; it is not submission authority.
The current implementation needs no new sampling-method choice to run the
registered O/P bridge. Exact environment, source ownership/authorization,
resource measurement and institutional confirmation remain required.

Use an authorized compute workspace with the bound source tree at `source/`.
`source/streaming_manifest.json` must hash to
`99f6de7a6cdd09b91e9bc0e167304db4d4274e753adad5138ec7803f5338a15b`.
Preserve its relative `class_XX/train.npy` and `class_XX/test.npy` paths.
Create a protected new `derived/` parent for new outputs; do not reuse any
historical D2 path. The following commands have **not** been run on real data.

## 1. Isolate the existing metadata contract

```console
python -B sampling_builder.py adapt-replayids --source-manifest source/streaming_manifest.json --expected-sha256 99f6de7a6cdd09b91e9bc0e167304db4d4274e753adad5138ec7803f5338a15b --output contracts
```

This adapter reads JSON only. Review its per-increment train contracts and
separate official-test contract against `REPLAYIDS_INPUTS.json`. The full
catalog remains outside the sampler. The source has no capture-group file;
the explicit row-identity proxy mode must remain visible in downstream claims.

## 2. Profile and derive each train increment separately

First obtain authorized measured resource evidence for the Task-0 derivation
stage. Record runtime/environment hash, whole-process peak RSS, disk high-water
usage, wall time, read/write volume and CPU use; do not derive Slurm resource
requests from this package's small tests. Use an allocated CPU stage for large
preprocessing. Never run it on the login node.

For P, the exact sequential command forms are:

```console
python -B sampling_builder.py derive --input contracts/increment_00.json --source-root source --output derived/P_00 --policy policy.P.json
python -B sampling_builder.py derive --input contracts/increment_01.json --source-root source --output derived/P_01 --policy policy.P.json --previous derived/P_00
python -B sampling_builder.py derive --input contracts/increment_02.json --source-root source --output derived/P_02 --policy policy.P.json --previous derived/P_01
python -B sampling_builder.py derive --input contracts/increment_03.json --source-root source --output derived/P_03 --policy policy.P.json --previous derived/P_02
```

For the offline bridge O, use the same four input files, `policy.O.json`, and
distinct destinations `derived/O_00` through `derived/O_03` with the matching
O predecessor. There is no in-program loop that submits or launches stages.
Each real execution stage still needs its own reviewed immutable closure and
exact-command confirmation where required by the project rules.

After each completed increment, run the read-only output verifier:

```console
python -B sampling_builder.py check derived/P_00
```

Capture the exact input/output manifest, implementation, policy, array and
ledger SHA-256 closure. Verify O/P calibration IDs and attack-fit IDs match,
P normal fit IDs are nested within O, and both Task-0 common-transform
collections and `common_transform_fit_sha256` match. Only actual produced
hashes and resource records count as real derivation evidence.

## 3. Bind the complete official test separately

```console
python -B sampling_builder.py verify-test --input contracts/official_test.json --source-root source --output derived/official_test_verified
```

This stage verifies source bytes and full support without copying, balancing,
or modifying them. It cannot alter any sampler selection. Keep its binding
separate from the training RNG and preserve all 227,723 source test rows if
the real verification confirms the historical source contract.

## 4. Common-cohort checkpoint and downstream materialization

Pass the reviewed P/O manifests to `load_increment`/`iter_partition`; use the
common Task-0 transform-fit cohort and one bound frozen Task-0 encoder for
controlled embedding exports. The fair-comparison runner consumes separately
hash-bound embeddings, labels, original row/group IDs and availability values;
it must not label these raw x matrices as embeddings or silently fit on later
classes. This builder itself performs no transform fitting, embedding inference
or training. Run the separate `fair_comparison/pretrain_task0.py` controlled
stage against P Task 0's `transform_fit` collection; retain its common-cohort
metadata, frozen normalization/encoder state and measured pretraining receipt.
Do not substitute a historical checkpoint lacking those exact cohort bindings.
Real data/encoder numerical-fidelity checks remain dependencies.

After all four increments and the independent test binding complete, the
following raw bridge is executable (symbolic reviewed checkpoint paths):

```console
python -B prepare_l1_inputs.py --p-prefix derived/P_00 derived/P_01 derived/P_02 derived/P_03 --o-prefix derived/O_00 derived/O_01 derived/O_02 derived/O_03 --test-directory derived/official_test_verified --source-root source --checkpoint-metadata shared_task0/checkpoint.json --checkpoint-state shared_task0/inference-state.npz --pretrain-cost shared_task0/pretrain-cost.json --config ../fair_comparison/pilot_config.json --output derived/raw_export_pair --evidence-kind real
```

The two output files `derived/raw_export_pair/P/export-input.json` and
`derived/raw_export_pair/O/export-input.json` are directly accepted by
`fair_comparison/materialize_embeddings.export(manifest_path, output,
runtime_root=..., device_name=..., batch_size=...)`. Check the separate
exporter's input contract and pinned FT runtime requirements. This is an
explicitly offline aggregate execution container, not online data revelation.
Both arms share exactly one frozen state and full unchanged test support;
only immutable fit rows enter training. Every stage has its own measured cost.

If a stage fails, preserve the failure evidence and last completed increment.
Diagnose and create a new versioned candidate; do not overwrite the failed
directory or blindly resubmit. No local test output, metadata arithmetic, source
filename or configured byte ceiling is a real derived dataset, profile or
successful experiment. Live DICC constraints and fresh independent exact-hash
review/user confirmation are still mandatory before any submission.
