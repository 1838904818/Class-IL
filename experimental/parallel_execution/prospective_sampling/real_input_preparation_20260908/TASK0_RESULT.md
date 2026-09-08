# Prospective Task 0 derivation result

8 September 2026. Real local CPU derivation completed; no model training or GPU execution. The independent output check returned COMPLETE. Manifest SHA-256: b1700d9f3d1c4aaf08f7df541f863940bb69714e0296279e12e0f527a6f70a71.

| Class | Source training rows | Selected fit | Calibration | Omitted from fit |
|---|---:|---:|---:|---:|
| Benign | 523263 | 5559 | 52326 | 465378 |
| DoS GoldenEye | 6176 | 5559 | 617 | 0 |
| Total | 529439 | 11118 | 52943 | 465378 |

The prospective normal cap is 5559, derived from the current Task 0 attack fitting pool after calibration reservation. No later-class count or official-test performance controls this cap. Omitted records remain in the immutable selection ledger; original arrays are not deleted. Transform-fit selects the same 11118 identities, but no transform has yet been fitted.

An independent NumPy check confirmed that the combined fit/calibration/omitted source indices contain exactly 529439 unique pairs, cover each source row once, and have disjoint membership. Transform-fit and fit row IDs match. This is row identity validation, not proof of independent capture groups or absence of semantically duplicate flows.

The launcher recorded 83.891 seconds and 195301376 bytes observed peak child working set (about 186.3 MiB). These Windows measurements do not constitute a DICC resource profile or complete I/O, disk high-water or CPU-utilization accounting. The original profile status is CHILD_EXITED_OUTPUT_UNVERIFIED because it predates the separate successful output check; it is retained unchanged.

Scope remaining: P increments 1-3, the matched O derivation, independent full-test binding, common-cohort encoder training, paired objective experiments, and attribution/repair experiments. No new accuracy, forgetting, innovation or governance benefit is established.
