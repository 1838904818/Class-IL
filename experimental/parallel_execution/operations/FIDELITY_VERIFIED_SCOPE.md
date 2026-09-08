# ReplayIDS GPU score-fidelity verification

Assessment: share with explicit scope caveats. Job 454941 completed in 71
seconds, exit 0. This is an inference reconstruction check, not model training.

## Verified evidence

Protected GPU_FIDELITY.json SHA256:
`baf055d3ab9bb800a887ad5f5659be3ea222bf11b2fc2c18da626ba4c743e5e5`.
Core manifest SHA256:
`e39e86e6946661e55f0ca412cba28268b0f35eb2ff22c9bfce7ebd9fd5c2d6fd`.
All six core files and the separate tracking-finalization checksum passed.
Bound verifier requires atol=1e-4, rtol=1e-5 and encoder batch size 512.

The complete Cartesian product of seeds 1, 2, 3, 4, 42 and checkpoints 0–3
contains 20 unique records. Each compares head, router-z and joint scores:
60 score comparisons report exact numerical equality, zero maximum absolute error
and zero out-of-tolerance cells. Prediction mismatch counts are zero and saved
and reconstructed prediction hashes match in every record.

Per seed, probe row counts are 256, 512, 768 and 898. Their sum across seeds and
checkpoints is 12,170 probe evaluations, not 12,170 independent observations.
Reported verifier runtime is 45.681 seconds; allocated CUDA peak is 1,206,098,944
bytes and reserved peak is 2,046,820,352 bytes. These are not host memory or
whole-system training-cost estimates.

Tracking client finish returned exit 0. Recorded run:
https://wandb.ai/csnet/ofra-etg-leon-hpc/runs/c4079549
This local receipt is not an independent cloud API verification.

## Claims not established

- Full-population or all-dataset score identity.
- Historical driver/build/kernel identity: the report explicitly leaves this
  unproven despite agreement on recorded environment fields.
- Attribution faithfulness, SHAP method robustness or ETG intervention utility.
- New accuracy/forgetting results, improved performance, or algorithm novelty.
- Closure of all numerical or methodological comments outside this bound
  ReplayIDS probe/checkpoint reconstruction scope.

The original report field named `bit_exact` is computed with NumPy
`array_equal`. It establishes elementwise numerical equality, not bytewise
identity: signed zeros and equal values in different dtypes can compare equal.
The comparator checks score shapes and finiteness but does not explicitly
compare score dtypes or saved-versus-reconstructed score bytes. Accordingly,
this document does not claim identical floating-point bit patterns.

No raw-score arrays were recomputed locally in this audit. This review checks
the protected per-checkpoint outputs, hashes, coverage and bound verifier
requirements; subsequent manuscript integration must retain that distinction.
