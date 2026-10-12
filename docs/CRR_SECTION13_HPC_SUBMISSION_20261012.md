# CRR Section 13 HPC submission receipt (2026-10-12)

The corrected Section 13 jobs were submitted under `leon12138` after live
`account-info` reported `FULL`, `cluster-info` listed `gpu-a100`, the current
association listed `long`, remote archive hashes passed, and the packaged
`ncd` and `src` imports passed. The pre-existing jobs 485030, 485032, and
485033 were left unchanged.

- Source commit: `679fecc3aded1ec4bd2827b3601ada4169e5a0c3`
- Archive: `crr-section13-679fecc.tar`
- Archive SHA-256: `4ee4f344a66252b9c35b2ff56a8e0264a49df1c067fe3931f2c9841d17fd3ad4`
- SBATCH SHA-256: `1fdd67fc63668e1cab7920ef129669f9fbb02c8316e3213af0782cc5e58dde83`

| Job | Scope | Dependency | Initial state | Resources |
|---:|---|---|---|---|
| 485084 | Replay Section 13.3 gate, both datasets | none | PENDING (Priority) | 1 A100, 2 CPU, 12 GB, 24 h |
| 485085 | corrected NSL-KDD calibration + 5 seeds | afterok:485084 | PENDING (Dependency) | 1 A100, 2 CPU, 12 GB, 24 h |
| 485086 | corrected UNSW-NB15 calibration + 5 seeds | afterok:485084 | PENDING (Dependency) | 1 A100, 2 CPU, 12 GB, 24 h |

The formal jobs run corrected Global-IR and gated RCL-tab; the RCL no-gate
version is an auxiliary diagnostic. Global-IR and RCL-tab each select lambda on
seeds 5 and 6 from `{0.25, 0.5, 1, 2}`. Formal row CSVs remain unavailable
until the jobs complete and therefore have not yet been committed.
