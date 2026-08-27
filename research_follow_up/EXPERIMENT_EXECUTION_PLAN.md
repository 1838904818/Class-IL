# OFRA three-direction execution plan

## Current data readiness

The research protocols and deterministic builders are ready. DICC Job `414596`
completed the A1/O1 data derivation on 2026-08-27. This is data-preparation
evidence only: no new accuracy, forgetting or open-set result is claimed.

### Derived data candidates

| Candidate | Purpose | Fit-train rows | Calibration rows | Official test rows | Manifest SHA-256 | Audit SHA-256 |
|---|---|---:|---:|---:|---|---|
| `A1 seed-42 capped training` | Isolate the 50k/class training-cap effect while preserving the original task order | 119,137 | 68,313 | 227,723 | `13527d6c3e58fbd1026509647f0c528ec7158a512d57145a9301f6ca8d03e1bd` | `77f4af3340d0342a5956ceb94e892c768c709a7f5336a74aa2a6f1caedfb0562` |
| `O1 seed-42 FTP held out` | Hold FTP-Patator until a singleton final labelled task for reject-then-new-head evaluation | 119,137 | 68,313 | 227,723 | `7913f826ec8667f7556d8930a7d07ce80e72ad5dbac0c7d213e1b88f27424749` | `0f9f9780713f874cd97e51a062df15678ebbb71483944fc5e89740f2bb8f7efc` |

The 68,313 calibration rows are removed from fit-training. Benign and DoS Hulk
are the only fit-training classes capped at 50,000 in this ReplayIDS contract.
All test shards retain their original hashes. The builder creates each output
directory exclusively and validates every test shard against the source
manifest before and after materialisation. Local development artifacts may use
hard links, but the reviewed DICC derivation uses byte copies because BeeGFS
rejects cross-directory hard links. Each copied shard is verified against the
same manifest SHA-256. Unversioned and v2 local drafts remain superseded.

## Dependency order

For DICC execution, the compatible A1/O1 stages are consolidated as documented
in `CONSOLIDATED_EXECUTION_PROTOCOL.md`: one CPU derivation job, one sequential
GPU training job after the remote hashes exist, and one O1 CPU evaluation job.
This reduces confirmations without using arrays, fan-out or recursive
submission.

1. **A1 seed-42 capped-training pilot**
   - same FT256x4, epochs, focal loss, negative ratio, replay=50 and router cap;
   - use the original task order;
   - compare with the existing uncapped seed-42 result;
   - estimated A100 runtime: roughly 15–25 minutes, to be replaced by measured
     runtime after the first approved run.
2. **A2 replay pilot on the better data protocol**
   - compare exemplar capacities `50`, `500`, `3000`;
   - do not run this matrix until A1 identifies whether the capped protocol is
     beneficial;
   - estimated A100 runtime: roughly 3 times A1, because these are separate
     matched runs.
3. **A3 Benign-anchoring pilot**
   - only after A2 fixes the replay budget;
   - requires a CII-like data stream and must be labelled as a separate
     scenario, not mixed with the ordinary Class-IL result.
4. **O1 FTP-Patator open-set pilot**
   - use the pre-final monitoring checkpoint for unknown rejection;
   - use the final checkpoint after labelled FTP-Patator arrival for new-head
     adaptation and old-class forgetting;
   - run `evaluate_open_set.py` after immutable checkpoint artifacts exist.
5. **O2/O3 rotating held-out attacks**
   - repeat for DoS Slowhttptest and DoS slowloris;
   - the method is not supported by one held-out class alone.
6. **Selected-arm five-seed confirmation**
   - only the chosen A/O arms are expanded to `{1,2,3,4,42}`;
   - paired statistics and all NIDS metrics are mandatory.

## DICC boundary

Local hard links are storage-efficient development artifacts, not DICC training
evidence. The reviewed DICC job derived the same arrays from the hash-bound
source cache using verified byte copies. The resulting manifest and audit
hashes above are now the required input bindings for the training job.

The completed and remaining dependency stages are:

1. completed: independently review and run the lightweight CPU A1/O1 derivation
   as DICC Job `414596`;
2. completed: verify the two manifests, audits, copied test-shard invariants and
   protected metadata checksum list;
3. next: render the GPU training script against the exact hashes above and send
   that new script through the independent review gate;
4. after immutable O1 checkpoints exist, review and run the CPU-only open-set
   evaluator.

A GPU script drafted before stage 2 is only a template. It is not an exact,
hash-bound submission candidate and must not be submitted.

Every DICC candidate remains subject to:

- current live partition/QoS/account limits;
- one node and one task unless separately justified;
- measured CPU/GPU/memory requests and utilization monitoring;
- the independent review thread's exact `VERDICT: APPROVED`;
- one hash-bound user confirmation in the current conversation.

## Local validation status

As of 2026-08-27 (Asia/Kuala_Lumpur):

- `ruff` passes for the research package;
- all seven unit tests pass;
- both Python entry points compile;
- the formal `streaming_full.smoke_test` exits with code 0;
- repeated seed/protocol smoke runs reproduce the same deterministic result
  hash, and checkpoint resume accepts only the validated result;
- the smoke test is synthetic and does not constitute A1 or O1 evidence.

The complete local copy-mode A1/O1 derivation was measured at 2.663 seconds and
approximately 188.6 MiB peak working set per arm. DICC Job `414596` completed in
12 seconds with one allocated CPU, one GiB and no GPU. `seff` reported 33.33%
CPU efficiency; stderr was empty. All protected metadata checksums passed.
