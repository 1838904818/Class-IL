# ETG input compatibility and feasibility audit

11 September 2026. **Input audit completed; no model, attribution or intervention experiment executed.**

## Main finding

The newer P/O calibration pool must not be substituted for the native D2 calibration pool when reusing native D2 checkpoints. Both contain 68,313 rows, but their row-selection algorithms differ. Reconstructing the D2 selections and verifying all sixteen historical fit/calibration index digests shows that **26,770 P/O calibration rows belong to the native D2 fitting selection**. Only 6,813 are shared calibration rows; 34,730 others are omitted Benign rows in the native D2 selection.

This does not invalidate the original P/O calibration study: its own fitting/calibration row separation is a different, previously audited contract. The finding prevents a proposed cross-experiment substitution; it is not a newly discovered prediction result or proof that a historical run performed that substitution.

| Class | New P/O calibration rows | Also native D2 calibration | In native D2 fitting selection | Outside both D2 selections |
|---|---:|---:|---:|---:|
| Benign | 52,326 | 5,216 | 12,380 | 34,730 |
| DoS GoldenEye | 617 | 60 | 557 | 0 |
| DoS Hulk | 13,864 | 1,387 | 12,477 | 0 |
| DoS Slowhttptest | 329 | 35 | 294 | 0 |
| DoS slowloris | 347 | 42 | 305 | 0 |
| FTP-Patator | 476 | 40 | 436 | 0 |
| Heartbleed | 1 | 0 | 1 | 0 |
| SSH-Patator | 353 | 33 | 320 | 0 |
| Total | 68,313 | 6,813 | 26,770 | 34,730 |

The native sampling audit and manifest are bound to the archived D2 data selection used by the registered baseline protocol. This audit does not independently execute every checkpoint's training implementation or rehash the full feature population. It reconstructs **data-selection identities**, not training execution.

## Two further limitations

1. **Rare-class support.** Both pools reserve only one Heartbleed calibration row. The current strict runner needs a nonempty fitting subset and at least two independent acceptance groups for each seen class. Even the necessary lower bound of three distinct fitting/acceptance rows cannot be met for Heartbleed, before considering trigger or subsequent evaluation. More majority-class rows, resampling the same row or changing the training seed cannot fix this. This blocks the proposed all-eight-class final-stage acceptance study from this calibration pool; it does not by itself rule out an explicitly scoped earlier increment.
2. **Capture grouping.** All inspected P/O calibration group arrays are byte-identical to their row-ID arrays. The saved feature schema contains no Timestamp, Flow ID, Source IP or Destination IP. The bound source register has two CSV files; treating each whole source file as a group cannot furnish four disjoint nonempty roles. Neither this observation nor a row-ID hash establishes independence of finer sessions or captures. No claim is made that source files themselves are independent.

The four P/O metadata pairs were rehashed, their source-offset row IDs independently reconstructed, and every historical D2 fit/calibration index digest checked before overlap was calculated. The audit reads metadata arrays, not features or checkpoints. Details and ordinary file-byte hashes are in [audit.json](audit.json); the source is [the read-only auditor](../../experimental/etg_input_audit/).

## Execution consequence

Preserve the [10 September protocol](../../docs/ETG_MINIMAL_UTILITY_PROTOCOL_2026-09-10.md) unchanged and do not submit it with the incompatible P/O pool. For native D2 checkpoints, use their originally bound calibration identities and the corresponding native encoder/preprocessing, after checking lineage and prior use. P/O embeddings cannot silently replace native D2 embeddings.

Original D2 calibration identities address the pool-compatibility problem, **not** the missing independent groups, rare-class support or real repeated attribution inputs. The full strict study remains unready.

A separately versioned, descriptive offline pilot could be proposed on an earlier chronological increment, with native calibration, disjoint row identities, actual explanation extraction and explicit dependence limitations. Such a proposal would not inherit group-wise safety guarantees or constitute a confirmed governance benefit. It would need its own agreed protocol and independent execution review before running. No smaller class set, weakened threshold, new split or alternative statistical gate is authorized or implemented by this audit.
