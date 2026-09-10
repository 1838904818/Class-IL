# EIAR diagnostic archive recovery

The ten small artifacts associated with Job 455775 were recovered on 10 September
2026 into a fresh protected archive. Original report and frozen-policy bytes
matched the existing result manifest. A separate SSH process checked all bound
files, the recovery receipt and both original checksum manifests. The original
failed archive was preserved.

The recovery used exclusive byte writes, file and directory synchronization and
readback verification. It did not retrain a model, run inference, change the
experiment's policy, alter W&B, or submit a new job. The cause of the earlier
archive discrepancy has not been established.

## Evidence boundary

`archive_status.json` records the original report/policy hashes, recovery-script
hash, recovery-receipt hash and independently checked status. This is an archive
maintenance record, not an updated manuscript benchmark or a new algorithm result.

A prior local audit independently recomputed aggregate and per-class metrics from
28 saved confusion matrices (four checkpoints and seven arms). The seed-1 EIAR
screen returned `DO_NOT_EXPAND_YET`; archive recovery does not change that outcome.
It is exploratory, not a five-seed or untouched-test confirmation. Independent
cloud tracking verification is still outstanding. Historical CPU-sampling and
memory-accounting discrepancies are not resolved by copying artifacts and must
not be used as efficiency evidence.
