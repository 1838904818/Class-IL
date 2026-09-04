# Attribution and ETG five-seed campaign: interim evidence register

Evidence checked: 2026-09-04 03:08 UTC.

## Status and scope

DICC Job **434747** is running the Malaya FT512x12 attribution/ETG v9
campaign for seeds **1, 2, 3, 4, 42**. Seeds 1 and 2 have protected outputs
whose file checksum manifests passed a fresh read-only verification. Seed 3
is in progress. This is **2/5 protected seed packages, not a completed
five-seed aggregate**.

The job performs post-hoc analysis of existing checkpoints; it does not
retrain a classifier. It compares Expected Gradients, Feature Ablation, and
Gradient x Input, followed by an offline, non-suppressing ETG ledger.
No online feedback, suppression of classifier decisions, or automatic
creation of attack classes or heads is being evaluated.

The explanation target is the CPU checkpoint reconstruction evaluated on
fixed true-class probe batches. Exact-forward wrapper/scorer agreement
within the same batch does **not** imply equivalence to archived GPU scores
or invariance to probe-batch partitioning. Cross-device prediction
differences are recorded separately; one such difference was recorded at
seed 1 checkpoint 4. Gradient stabilization is confined to the derivative
surrogate and is not evidence of mathematical equivalence of derivatives.

## Verified identifiers

SHA-256 values below are ordinary file-byte hashes, not canonical JSON
hashes. The seed checksum-manifest checks verify file integrity, not
scientific validity or statistical significance.

| Object | SHA-256 |
|---|---|
| Submitted v9 batch script | `edea800c779c674b8cf19e362103c9110ba7a2bf8da4393d49bf6b82716dac1c` |
| Seed 1 protected checksum manifest | `edc8bb9c4d2f7771f92446dad09de33fd7b52bc043019429c89b69e9533eedef` |
| Seed 1 robustness JSON | `8bf13cf4fe14063b56ac3adbf7d0e7c7797a721c9efea2f3813fbfe5f169b97b` |
| Seed 2 protected checksum manifest | `15c27991a372e1e1fd3681c37ba1d63b7feb996e4cec529ca9100134a73c07c6` |
| Seed 2 robustness JSON | `29708d1e8a6814d9efd208a62392cf2ac75e6e602306ee46710131ed29d46e73` |

## Outstanding completion gates

- Finish and validate the protected seed 3, 4, and 42 packages.
- Independently verify all five seed identities, inputs, code/configuration
  bindings, probe coverage, and aggregate checksum manifests.
- Verify method-pair agreement and ETG status sensitivity, drift
  denominators, threshold settings, missing/undefined cases, and uncertainty.
  The independent checker in `reproducibility/attribution_etg_checks/`
  passed protected seeds 1 and 2; it supplements rather than replaces the
  original artifact validator. All five seed packages must pass before a
  final aggregate is promoted.
- Confirm the completed aggregate W&B record and its allowlisted outputs.
- The versioned offline source snapshot is available at
  `reproducibility/attribution_etg_v9/README.md`: 46 offline unit tests and
  its isolated import inventory passed. Full-data reproduction and network
  publication are not covered by those tests. The repository's existing
  analysis entry point has not been overwritten. The snapshot also records
  the retrospectively verified recovery import missing from the original
  source manifest; this does not close its pre-submission provenance gap.
- Update the manuscript and technical documentation only after those checks;
  keep earlier single-seed pilots and this interim state distinct from the
  eventual five-seed result.

No aggregate effect, attribution-method independence, architecture
superiority, or publication readiness is established by this interim
register. This Malaya explanation campaign must not be conflated with
the separate ReplayIDS checkpoint-selection experiment.
