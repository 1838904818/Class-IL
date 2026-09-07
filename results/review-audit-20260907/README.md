# Saved-evidence re-analysis, 7 September 2026

This package accompanies manuscript v3.3. It replays immutable records and does not train, run SHAP, perform new inference, repair a model or measure analyst benefit.

## Reproduction

Run from the repository root with Python and NumPy installed. The existing primary diagnostic also uses the repository's aggregation helper and its documented dependencies. All output paths below are new local analysis files, not replacements for historical results.

```sh
python tools/replayids_primary_diagnostic.py --repo . --output PRIMARY_RECHECK.json
python tools/audit_saved_evidence.py --evidence-root results/review-audit-20260907/source-records --comparison-root results/balanced-replay-five-seed --historical-analyzer reproducibility/attribution_etg_v9/formal_v2_explanation_etg/analyze.py --primary-diagnostic PRIMARY_RECHECK.json --output audit_recomputed
python tests/test_audit_saved_evidence.py
python experimental/expansion_diagnostic/test_expansion_diagnostic.py
```

Use the first command to verify source hashes and recompute confusion-derived primary metrics before the combined re-analysis. ReplayIDS paired tests use **Job 425539 last epoch**, not guarded Job 426307 from the older comparison exports. Malaya comparison exports are checked against their export registry. Both n=5 and n=4 (excluding screening seed 42) remain descriptive.

## Contents and limits

- `AUDIT_SUMMARY.json`: 450 historical ledger state/action matches; 240 fixed-top-15 policy replays; consensus across 150 shared checkpoint-class records; alternative saved EG similarities; 24 paired sign-flip/Holm comparisons.
- `VERSIONED_ETG_ROWS.json`: method, policy, implementation, result and probe-manifest bindings. Neutral display labels do not change original states or assert prediction reliability.
- `CONSENSUS_ROWS.json`: class-performance stratification. Rows share classes, checkpoints and training seeds; they are not independent replications.
- `PRIMARY_RECHECK.json`: regenerated last-epoch ReplayIDS diagnostic, including source input hashes. Its separate seven-metric correction is not the combined twelve-metric correction in `AUDIT_SUMMARY.json`.
- `ANALYSIS_BINDINGS.json`: current analysis-code hash and derived output hashes. The historical state-machine code is separately pinned by its immutable SHA-256.
- `source-records/`: compact historical analysis, robustness, training result and probe metadata for seeds 1, 2, 3, 4 and 42. No raw traffic, feature matrix, credentials or private working notes are included.

Each original protected analysis protocol was retrieved read-only and checked against the original checksum registry before export. `protocol_metadata.json` is a **redacted provenance attestation**, not the original byte-identical protocol. It binds the original protocol SHA, probe manifest SHA and recorded package versions. The full original protocol contains a private filesystem path and is not redistributed. This limits independent verification of the capsule itself; it must not be presented as a signed third-party attestation.

`SHA256SUMS` is the historical full registry: entries for files not included in this compact package are not claims that those files are present. Included historical files are verified against their matching entries. File hashes ensure identity, not the scientific validity of measurements. The public package manifest covers all distributed files.

The existing [five-seed EG overlap grid](../../docs/RECOVERED_SENSITIVITY_2026-09-07.md) contains 320 saved settings. That grid and the present fixed-top-15 ledger replays answer different questions. Changing top-k admission requires new masked score evaluations. A high agreement rate is not proof of correctness; 27 of 72 low-recall records have at least one historical certified state. ETG currently supports audit, not operational governance.

See the [technical supplement](../../docs/TECHNICAL_AUDIT_SUPPLEMENT_2026-09-07.md) and [remaining experiment gates](../../docs/EXPERIMENT_GATES_2026-09-07.md).
