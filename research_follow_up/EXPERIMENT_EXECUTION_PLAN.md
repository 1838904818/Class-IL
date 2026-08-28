# OFRA three-direction execution plan

## Current data readiness

The research protocols and deterministic builders are ready. DICC Job `414596`
completed the A1/O1 data derivation on 2026-08-27. A1 training completed inside
Job `414606` and was retained only after exact result/checkpoint hash checks; the
outer job then failed before O1 because of a validator path error. Later
hash-bound training produced the O1 checkpoints, and CPU Job `414686` completed
the first open-set evaluation. All registered pre-label gates had 0%
FTP-Patator recall; the post-label supervised head achieved 82.30% recall.

The A1 fixed-cap seed-42 result is mixed. For joint-cap3000, relative to the
uncapped seed-42 reference, accuracy increased from 83.62% to 87.87% and
forgetting decreased from 10.70% to 3.52%. However, Macro-F1 fell from 50.02%
to 44.71% and balanced accuracy from 68.62% to 51.02%. The fixed 50,000-row cap
therefore does not pass the registered minority-performance decision rule.

Protocol D2 keeps every attack fit row and caps only Benign to the largest
attack fit pool. Reviewed DICC Job `414907` completed the seed-42 derivation in
11 seconds with empty stderr and all registered invariants passing. It produced
268,697 fit rows, 68,313 disjoint calibration rows and all 227,723 official test
rows. The protected metadata checksum-registry SHA-256 is
`8d164306e0e153ef661882133083d3520e07557c0fd839bc98eb8b1f684db8f7`.

### Derived data candidates

| Candidate | Purpose | Fit-train rows | Calibration rows | Official test rows | Manifest SHA-256 | Audit SHA-256 |
|---|---|---:|---:|---:|---|---|
| `A1 seed-42 capped training` | Isolate the 50k/class training-cap effect while preserving the original task order | 119,137 | 68,313 | 227,723 | `13527d6c3e58fbd1026509647f0c528ec7158a512d57145a9301f6ca8d03e1bd` | `77f4af3340d0342a5956ceb94e892c768c709a7f5336a74aa2a6f1caedfb0562` |
| `O1 seed-42 FTP held out` | Hold FTP-Patator until a singleton final labelled task for reject-then-new-head evaluation | 119,137 | 68,313 | 227,723 | `7913f826ec8667f7556d8930a7d07ce80e72ad5dbac0c7d213e1b88f27424749` | `0f9f9780713f874cd97e51a062df15678ebbb71483944fc5e89740f2bb8f7efc` |
| `D2 adaptive normal-only cap` | Reduce Benign dominance without discarding any attack fit rows | 268,697 | 68,313 | 227,723 | `6d6e467ed0687ff6eb2d171c9799c74ee16af3c5b65bea2e662fa569bc3d5e1b` | `697750d599f448ba1120ba60decac98abb607204eec1a431073673cc3b124b9b` |

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
2. **A2 replay pilot on the immutable uncapped protocol**
   - compare the completed exemplar capacity `50` reference with candidates
     `500` and `3000`;
   - keep the dataset, FT256x4 model, epochs, optimiser, focal loss and router
     fixed so only replay changes;
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
3. completed: hash-bound O1 checkpoint production;
4. completed: reviewed CPU-only open-set evaluation as DICC Job `414686`;
5. completed: reviewed D2 derivation as DICC Job `414907`;
6. completed: sequential replay-500/replay-3000 seed-42 screen as DICC Job
   `414908`; neither larger replay candidate passed the registered
   minority-performance selection rule;
7. next: retain replay 50, train the D2 adaptive normal-only data protocol,
   continue the controlled model-tuning stages, redesign the unknown gate,
   rotate held-out classes, and expand only frozen selected rules to multiple
   seeds.

A GPU script drafted before stage 2 is only a template. It is not an exact,
hash-bound submission candidate and must not be submitted.

Every DICC candidate remains subject to:

- current live partition/QoS/account limits;
- one node and one task unless separately justified;
- measured CPU/GPU/memory requests and utilization monitoring;
- the independent review thread's exact `VERDICT: APPROVED`;
- one hash-bound user confirmation in the current conversation.

## Local validation status

As of 2026-08-28 (Asia/Kuala_Lumpur):

- `ruff` passes for the research package;
- all 10 follow-up unit tests pass;
- both Python entry points compile;
- the formal `streaming_full.smoke_test` exits with code 0;
- repeated seed/protocol smoke runs reproduce the same deterministic result
  hash, and checkpoint resume accepts only the validated result;
- the smoke test is synthetic and does not constitute A1 or O1 evidence.

The complete local copy-mode A1/O1 derivation was measured at 2.663 seconds and
approximately 188.6 MiB peak working set per arm. DICC Job `414596` completed in
12 seconds with one allocated CPU, one GiB and no GPU. `seff` reported 33.33%
CPU efficiency; stderr was empty. All protected metadata checksums passed.
