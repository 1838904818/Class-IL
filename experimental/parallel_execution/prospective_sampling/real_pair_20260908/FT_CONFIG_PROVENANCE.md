# Common encoder configuration provenance

Prepared configuration, not executed or approved for HPC submission.

The historical seed-1 protocol SHA-256 is f52b1a8c1a81b14d77eb5771349eddde2258edd781e40293d8d197ea9f0ba6b1. Its encoder architecture is FT256x4 with eight heads, head dimension 32, attention/feed-forward dropout 0.1 and one residual stream. Its pretraining uses eight epochs, batch size 384, Adam, learning rate 0.001 and zero weight decay. These values are retained in task0_ft256x4_seed1.json rather than selected from new test outcomes.

The new common cohort has 11118 rows. Eight complete epochs imply 88944 productive row presentations and 232 optimizer steps (29 batches per epoch, without gradient accumulation). A checkpoint interval of 29 steps is a new operational choice, not an assertion that historical checkpoint cadence was identical. An initial preparation arithmetic assertion failed and was corrected before any training execution.

The configuration retains the implementation-required pilot disclosure label. Historical hyperparameter reuse is not preregistration, historical runtime parity, or proof of optimality. The new cohort and temporary multiclass pretraining head mean this is new training, not recovery of an old model.

Only seed 1 is prepared for initial execution/resource assessment. A final five-seed comparison requires all paired seeds and a frozen protocol; no single-seed profile should be called the final result. A runtime/source closure, device-specific measurements, review and execution authorization are still required. P/O export must share the exact resulting encoder state and preprocessing arrays. This configuration does not resolve downstream exposure matching or instantiate the full OFRA router comparison.
