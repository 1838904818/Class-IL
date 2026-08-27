# Consolidated execution protocol for the three OFRA follow-up directions

Date: 2026-08-27 (Asia/Kuala_Lumpur)

## Goal

Reduce user confirmations and scheduler overhead without merging methodologically
different results or wasting DICC resources.

The campaign uses three Slurm submissions rather than five:

1. **D1 combined CPU derivation**: derive A1 and O1 sequentially from the same
   hash-bound ReplayIDS source cache;
2. **T1 combined GPU training**: after D1 produces immutable remote hashes,
   train A1 and O1 sequentially in one A100 allocation, with separate output,
   recovery, W&B and protected-result directories;
3. **E1 O1 CPU evaluation**: evaluate unknown rejection and post-label new-head
   adaptation from the hash-bound O1 checkpoints.

Each submission still receives one exact-hash independent review and one user
confirmation. The three confirmations cannot be collapsed into one advance
approval because T1 depends on hashes that do not exist before D1 finishes, and
E1 depends on checkpoint hashes that do not exist before T1 finishes.

## Evidence isolation

Combining work into one Slurm allocation does not combine the evidence:

- A1 and O1 use different derived directories and manifests;
- A1 preserves tasks `[[0,1],[2,3],[4,5],[6,7]]`;
- O1 uses `[[0,1],[2,3],[4],[6,7],[5]]`;
- each arm has its own protocol, manifest, audit, result directory, W&B run,
  checkpoint set and SHA-256 inventory;
- failure in either arm makes the combined job incomplete; no partial campaign
  is reported as a complete comparison.

## Why E1 remains separate

The open-set evaluator is CPU-oriented. Running it after training inside the
A100 allocation would leave the GPU idle and risk violating DICC utilization
rules. It therefore remains a separate CPU job.

## Direction mapping

| Research direction | First evidence-producing stage | Later expansion |
|---|---|---|
| Training cap and imbalance protocol | A1 within D1/T1 | selected protocol to five seeds |
| Azizi/ReplayIDS-guided tuning | A1 baseline, then replay/anchoring/optimizer arms | architecture only after data and replay controls |
| Unseen attack and new head | O1 within D1/T1/E1 | rotate held-out Slowhttptest and slowloris |

## Current evidence boundary

- Literature analysis, protocols, builder and evaluator are ready.
- Local v3 A1/O1 derivations and tests are complete.
- No new A1/O1 model result exists yet.
- Existing paper numbers remain unchanged until immutable DICC outputs are
  collected and validated.
