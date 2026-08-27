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

## Evidence boundary

- Existing formal results remain the primary paper evidence.
- Derived capped-data and open-set runs are new experiments and must use new
  output directories, new hashes and new run identifiers.
- A seed-42 pilot is diagnostic only. Publication claims require the agreed
  five seeds and the same frozen protocol.
- No DICC job is submitted by these files. A future job must pass the project's
  independent hash-bound DICC review and user confirmation gate.
