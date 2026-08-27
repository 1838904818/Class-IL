# OFRA three-direction execution plan

## Current local readiness

The research protocols and deterministic builders are ready. No new result is
claimed and no DICC job has been submitted.

### Derived data candidates

| Candidate | Purpose | Fit-train rows | Calibration rows | Official test rows | Manifest SHA-256 | Audit SHA-256 |
|---|---|---:|---:|---:|---|---|
| `derived_replayids_cap50k_seed42_v3` | Isolate the 50k/class training-cap effect while preserving the original task order | 119,137 | 68,313 | 227,723 | `7e49feaa79f34caf041f41ef6bdcce2d59f277b428cfd2aa5a1d9600c129c26a` | `a8830d2979935f93ca7db89d9e17a2b91c245b9c138f45b46f6856eff4101524` |
| `derived_replayids_cap50k_unknown5_seed42_v3` | Hold FTP-Patator until a singleton final labelled task for reject-then-new-head evaluation | 119,137 | 68,313 | 227,723 | `7177bb6c7e08734aa078efdfad955c7c25f6e0b9f562b2e83601eccddbf44a03` | `4083daca10191ced3236bf56160c04cd76d29d66c7db5d20ed7c8b3fe6d2b1de` |

The 68,313 calibration rows are removed from fit-training. Benign and DoS Hulk
are the only fit-training classes capped at 50,000 in this ReplayIDS contract.
All test shards retain their original hashes. The v3 builder creates the output
directory exclusively, validates each test shard against the source manifest
before and after materialisation, and verifies hardlink device/inode identity.
The unversioned and v2 local drafts are superseded and must not be used for a
DICC candidate.

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

The local hard links are storage-efficient local artifacts, not uploadable
training evidence by themselves. DICC should derive the same arrays from the
already hash-bound source cache inside a scheduled job or a separately reviewed
preprocessing job. The resulting remote manifest and audit must be hashed before
the training job is reviewed.

This creates a mandatory two-stage dependency for the first DICC execution:

1. review and run one lightweight CPU batch job that derives A1 and O1 from the
   already bound remote source cache;
2. collect the remote `streaming_manifest.json`, `sampling_audit.json`, builder
   hash and test-shard invariants;
3. only then render the GPU training scripts with those exact remote hashes and
   send each script through the independent review gate.

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

The complete local A1 derivation was also measured at 1.217 seconds elapsed,
188.77 MiB peak working set and 525.83 MiB peak paged memory. This supports a
one-CPU, 1 GiB, 10-minute DICC derivation candidate while retaining margin for
networked scratch I/O. The remote job still requires `seff` validation.
