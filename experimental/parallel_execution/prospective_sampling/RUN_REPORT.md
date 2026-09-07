# Local implementation verification

Date: 2026-09-07. Scope: temporary synthetic arrays and small CPU inference.
No real dataset arrays were processed, no optimization/training was performed
by this suite, and no HPC, remote-write or Git action was used for this package.

Command, run from this package:

```console
python -B -m unittest discover -s . -p "test_*.py" -v
```

Result: **36 tests, 35 passed, one skipped**, 51.896 seconds. The skipped test
requires creating a real filesystem symlink, unavailable on this Windows host.
The separate simulated Windows reparse-point rejection regression passed.
This is not a deployed-filesystem security proof or a real resource profile.

Environment observed: Python 3.11.9, NumPy 2.4.6, PyTorch 2.11.0+cu128. The
cross-line test explicitly used CPU; it did not allocate a CUDA workload.

## What executed

- Real `.npy` memmap input validation and immutable per-increment writes;
  disk-backed row identity/rank selection; calibration/fit/omitted conservation.
- O/P shared calibration and attack fitting identities, nested normal fitting
  identities, identical P Task-0 transform-fit cohort and frozen cap.
- Input ordering, chunk size, seed repeatability and metadata prefix invariance.
  Replacing future-class/test metadata changes neither early contract nor
  early array output. One small synthetic case exceeds 10,000 rows.
- Invalid file/manifest hashes, shape/dtype/labels, row IDs, groups, paths,
  nonfinite numbers, duplicate JSON keys, changed input during reading/use,
  prefix rewrite, incomplete output, modified output and unsafe resume rejection.
- Actual cross-line integration: completed fit arrays -> `prepare_pair` raw
  P/O inputs -> `materialize_embeddings.export` -> `l1_runner.load_inputs`.
  It preserved row/group IDs and availability, excluded calibration/omitted
  rows, retained all eight synthetic test rows and used one identical encoder.
  Numerical output matched direct inference through the same tiny MLP.
- Missing common-cohort checkpoint bindings, mismatched evidence kind,
  incomplete class prefixes, invalid L1 configuration and unbound extra files
  fail closed. A forced second-arm failure cannot produce a usable pair.

The tiny checkpoint was an explicitly **untrained synthetic interface fixture**.
Its metadata does not establish any model pretraining result or accuracy claim.
The separate common-Task-0 pretrainer must generate an auditable real checkpoint
and training-cost receipt in the controlled execution stage.

## Source interpretation

Exact source and implementation hashes are in `SOURCE_BINDINGS.json`.
The new code preserves the original reference's fixed rank domain and logic;
it does not execute the synthetic reference or inherit its 10,000-row ceiling.
Its RNG input is at `sampling_builder.py:251`; source-row identity is at line
256. Whole-source binding is audit metadata and never a random-domain input.

The hash-bound historical D2 builder uses the full manifest SHA in its seed
(lines 37-39 and 82), prepares every class before selection (87-117), and takes
the maximum attack fit count over all prepared classes (119-125). These are
respectively metadata coupling and offline count lookahead, not proof of
test-label fitting or future-feature normalization. Legacy D2 is historical
context; **O versus P using the same new RNG** isolates cap availability.

## Remaining real-execution gates

The production builder, verifying loader and raw-export adapter are executable;
the remaining data work is controlled real input verification and derivation,
not writing another local builder. Actual source dtype/hash/feature fidelity,
source authorization and capture/group authenticity are unverified. No capture
independence is claimed in row-proxy mode. Float64 real input requires an
explicit reviewed precision contract because the L1 bridge accepts float32.

Real completed array/manifest closures, common-Task-0 training provenance,
encoder/export fidelity and measured CPU/RAM/disk/wall-time costs are still
required. Historical count arithmetic is not actual output support, a runtime
profile, accuracy evidence or proof that the smaller normal cap helps FPR.
Every HPC calculation requires current permitted resources, allocated compute,
fresh exact-hash review and the project's explicit final user confirmation.
