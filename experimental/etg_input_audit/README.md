# Read-only input compatibility auditor

This focused audit checks whether an already prepared P/O calibration pool can be reused with the archived native D2 data selection. It reads only JSON and small label/index/identity arrays; it reconstructs deterministic row indices, checks their historical hashes and emits a JSON report to stdout. It does not call a builder, train, infer, extract SHAP, create data partitions, mutate a ledger or contact DICC/W&B.

From this directory:

```text
python -B -m unittest test_audit_inputs.py -v
python -B audit_inputs.py --workspace-root PATH_TO_RECONSTRUCTION_WORKSPACE
```

The read-only reconstruction workspace must contain the source-bound records at the relative locators listed in `audit_inputs.py`: the original source manifest/data report, archived D2 manifest/sampling audit and builder source files, the published seed-1 baseline protocol, and P/O increment manifests with calibration labels, source indices and row/group IDs. Source records and feature data are not fabricated or downloaded by the script.

Native seed arithmetic and both index-list hashes must match before any overlap is reported. The script also verifies source-shard bindings, all eight classes, P/O metadata identity, and the prospective row-ID derivation. Inputs are capped at 32 MiB per file and one million indices per class; no feature tensor is opened. A mismatching dependency or index hash aborts the audit, not a fallback to approximate matching.

Six synthetic unit tests validate the arithmetic guards only. Neither these tests nor the input audit prove capture independence, explanation faithfulness, historical training execution or governance utility. The registered result is in [the evidence note](../../results/etg-input-audit-20260911/README.md). Runtime tested locally: Python 3.11.9, NumPy 2.4.6.

Only source code, tests, this note and the aggregate evidence are intended for publication. Do not add raw arrays, local absolute paths, checkpoint contents or credentials.
