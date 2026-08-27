# Validation report for the OFRA follow-up directions

Validation date: 2026-08-27 (Asia/Kuala_Lumpur)

## Overall assessment

The hash-bound A1/O1 derivation, training prerequisites, and first O1 CPU
evaluation are complete. DICC Job `414686` produced a verified single-seed
open-set result. That result is negative for autonomous unknown discovery but
positive for supervised adaptation after a legitimate labelled increment.

## Data and execution bindings

| Item | SHA-256 |
|---|---|
| O1 derived manifest | `7913f826ec8667f7556d8930a7d07ce80e72ad5dbac0c7d213e1b88f27424749` |
| O1 sampling audit | `0f9f9780713f874cd97e51a062df15678ebbb71483944fc5e89740f2bb8f7efc` |
| Pre-update checkpoint manifest | `a20e0eaca311bd2d4d18d4d5e9068e74070d7dacee34a7afdefdbc9b52e74a73` |
| Post-update checkpoint manifest | `b9e14663ab11cf634554f95750a9c26c1ceddde444a1afc02bf85766c6a44a56` |
| Evaluator | `53c032086b25272be505ec2123cadd51d2c231748b9878a1f7ebc59fd1a0744f` |
| Result | `6c190af2dc4ee2662d13dbf1d1500b431f4f70de2d7a25e8598313d727368b82` |
| Protected checksum registry | `f48c36704d44546d24ee4db29938a27fd43b769d80fc78e2fce3078c6e294258` |

Job `414685` failed safely with an out-of-memory condition at inference batch
size 4,096. The reviewed retry changed the batch to 512. Job `414686` completed
in 46 minutes 21 seconds with 99.17% CPU efficiency and approximately 1.27 GiB
maximum memory usage.

## O1 outcome

- Held-out class: FTP-Patator, 1,588 test rows.
- Known classes: 226,135 test rows; pre-update accuracy 87.28%.
- Test used for threshold selection: no.
- Unknown recall: 0% for confidence-only, distance-only, conservative joint,
  and empirical-joint rules.
- Primary known false-unknown rate: 0.225%.
- Best AUROC: 0.5303; OSCR-style AUC: 0.4699.
- Primary candidate buffer: 509 known rows, zero held-out rows; not eligible for
  analyst review.
- Post-label new-class recall: 82.30%.
- Post-label old-class accuracy: 87.01%, a -0.27 percentage-point change.

The evaluator never creates a head automatically. The only new head is the
ordinary supervised OFRA head created after the declared labelled task
boundary.

## Remaining validation work

1. Rotate the held-out class to DoS Slowhttptest and DoS slowloris.
2. Redesign or learn the unknown score because the registered gate provides no
   separation for FTP-Patator.
3. Expand only a pre-registered selected gate to seeds 1, 2, 3, 4, and 42.
4. Continue to report the current result as a single-seed expected-contract
   reconstruction, not an unchanged ReplayIDS reproduction.
