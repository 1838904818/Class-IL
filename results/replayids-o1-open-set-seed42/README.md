# ReplayIDS O1 open-set and labelled-head pilot

This directory records the public, path-sanitised binding for DICC Job
`414686`. It is a **single-seed diagnostic pilot**, not a five-seed paper
result and not an unchanged reproduction of ReplayIDS. The stream is an
expected-contract reconstruction from the published implementation and cached
source data.

FTP-Patator was withheld until the final labelled increment. Thresholds were
calibrated on a training-only known-class holdout; the test set was not used to
select them. Before the label arrived, every registered unknown-rejection arm
reported 0% FTP-Patator recall. The primary conservative gate rejected 509
known rows and no FTP-Patator rows, so its candidate buffer was not eligible for
analyst review. The best detector AUROC was only 0.5303 and the OSCR-style AUC
was 0.4699.

After FTP-Patator was supplied as a legitimate labelled increment, OFRA created
the corresponding supervised head through its normal class-incremental update.
New-class recall was 82.30%; old-class accuracy changed from 87.28% to 87.01%,
a reduction of 0.27 percentage points.

The evidence therefore supports **labelled adaptation**, but it rejects the
current confidence-plus-centroid gate as an autonomous discovery mechanism.
No new head was created automatically by the open-set evaluator.

See `PUBLIC_BINDING.json` for metrics, W&B URL, checkpoint bindings, result
hashes, and the protected-registry hash.
