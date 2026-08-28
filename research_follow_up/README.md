# OFRA follow-up research package

This directory separates three follow-up questions from the current formal OFRA
evidence. Nothing here changes or replaces the five-dataset, five-seed results.

1. `RELATED_DATA_PROTOCOLS.md` compares how closely related NIDS continual-
   learning studies construct, balance, cap and evaluate their data.
2. `AZIZI_TO_OFRA_TUNING_MATRIX.md` translates the ReplayIDS/Azizi evidence
   into a controlled OFRA tuning sequence.
3. `OPEN_SET_NEW_HEAD_PROTOCOL.md` defines an open-world pilot in which an
   unseen attack is rejected first and becomes a new OFRA head only after a
   labelled task boundary.
4. `build_train_protocol.py` derives a hash-audited training manifest with a
   deterministic per-class cap and a training-only calibration holdout while
   leaving the official test shards unchanged.
5. `evaluate_open_set.py` evaluates confidence, centroid-distance and combined
   unknown rejection from an OFRA monitoring checkpoint.
6. `CONSOLIDATED_EXECUTION_PROTOCOL.md` reduces the campaign to three reviewed
   Slurm submissions while keeping A1/O1 evidence and resource types separate.
7. `DICC_DERIVATION_EVIDENCE.md` records the completed, hash-bound A1/O1 data
   derivation that subsequent training candidates must use.
8. `../results/replayids-o1-open-set-seed42/` records the completed, sanitised
   seed-42 O1 evaluator result and immutable evidence hashes.
9. `DATA_PROTOCOL_V2.md` registers the adaptive normal-only training cap, and
   `build_train_protocol_v2.py` implements it without sampling attack or test
   rows.
10. `MODEL_TUNING_PROTOCOL.md` defines the controlled replay-capacity screen,
    its metrics and the order of later model-adjustment stages.

## Evidence boundary

- Existing formal results remain the primary paper evidence.
- Derived capped-data and open-set runs are new experiments and must use new
  output directories, new hashes and new run identifiers.
- A seed-42 pilot is diagnostic only. Publication claims require the agreed
  five seeds and the same frozen protocol.
- DICC Job `414596` completed the A1/O1 data derivation only. The later
  hash-bound training produced the required O1 checkpoints, and CPU Job
  `414686` completed the open-set evaluation.
- In Job `414686`, all registered gates had 0% FTP-Patator unknown recall. The
  post-label supervised update reached 82.30% new-class recall with a 0.27-
  percentage-point reduction in old-class accuracy. This is a negative result
  for autonomous discovery and a positive diagnostic for labelled adaptation.
- Every subsequent training or evaluation job still requires a fresh,
  independent hash-bound DICC review and one matching user confirmation.
- The adaptive data derivation completed as Job `414907`; it has not yet been
  used for model training.
- The replay-500/replay-3000 seed-42 diagnostic completed as Job `414908`.
  Larger replay improved final accuracy and Benign false-positive rate but
  weakened class-balanced performance, so replay 50 remains the reference for
  the next controlled stage.
