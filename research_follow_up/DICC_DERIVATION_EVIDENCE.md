# DICC A1/O1 derivation evidence

Evidence date: 2026-08-27 (Asia/Kuala_Lumpur)

## Scope

DICC Job `414596` performed CPU-only data derivation. It did not train OFRA,
select a model, measure accuracy or forgetting, run SHAP/ETG, or evaluate
unknown rejection. The artifacts below are immutable inputs for those later
experiments.

## Bound execution

- reviewed sbatch SHA-256:
  `35b10de69949fd0a691d1eddef97ae7950dd5911cae8286f380b64f98120c0a3`
- source manifest SHA-256:
  `99f6de7a6cdd09b91e9bc0e167304db4d4274e753adad5138ec7803f5338a15b`
- builder SHA-256:
  `e6bacab1659b1be09eee169f2b35be86639e82b7ccf5b031e7d7b3c323fad1c4`
- resources: one CPU, one GiB, zero GPUs, ten-minute limit
- outcome: `COMPLETED`, exit code `0:0`, elapsed time 12 seconds, empty stderr
- `seff` CPU efficiency: 33.33%

## Derived protocols

| Arm | Task sequence | Fit rows | Training-only calibration rows | Unchanged official test rows | Manifest SHA-256 | Audit SHA-256 |
|---|---|---:|---:|---:|---|---|
| A1 | `[[0,1],[2,3],[4,5],[6,7]]` | 119,137 | 68,313 | 227,723 | `13527d6c3e58fbd1026509647f0c528ec7158a512d57145a9301f6ca8d03e1bd` | `77f4af3340d0342a5956ceb94e892c768c709a7f5336a74aa2a6f1caedfb0562` |
| O1 | `[[0,1],[2,3],[4],[6,7],[5]]` | 119,137 | 68,313 | 227,723 | `7913f826ec8667f7556d8930a7d07ce80e72ad5dbac0c7d213e1b88f27424749` | `0f9f9780713f874cd97e51a062df15678ebbb71483944fc5e89740f2bb8f7efc` |

Both arms contain eight classes. Benign and DoS Hulk are the only fit-training
classes capped at 50,000 rows. No minority class is oversampled or fabricated.
FTP-Patator is class 5 and appears only in the final labelled O1 increment.

## Verified invariants

- calibration rows come only from the source training partition;
- fit and calibration indices are disjoint for every class;
- the official test partition is not sampled;
- each official test shard is copied byte-for-byte and verified against the
  source manifest SHA-256;
- the A1 and O1 fit/calibration files are identical; only their task order and
  the O1 held-out-class declaration differ;
- all copied protected metadata passed `sha256sum -c`;
- protected metadata checksum-list SHA-256:
  `7814fe85d0278448d161c286065d93fd809e6aba7680506daad389c99826a213`.

## Evidence boundary

This record establishes reproducible inputs, not model performance. A new
training script must bind the exact A1/O1 manifest and audit hashes above and
pass the independent DICC review and user-confirmation gates before submission.
