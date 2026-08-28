# ReplayIDS replay-capacity diagnostic, seed 42

This directory records the public-safe summary of DICC Job `414908`. The job
changed only OFRA's per-class exemplar capacity from the completed replay-50
reference to 500 and 3,000. Dataset, FT-Transformer architecture, epochs,
optimiser, focal loss, router settings and seed were held fixed.

The registered primary view is `official/joint_cap3000`. Here, `cap3000` is the
router sample cap used to construct DP-Means centroids. It is not the replay
capacity. The two parameters can both equal 3,000 but control different parts
of the system.

## Result

| Replay capacity | Average task accuracy | Forgetting | Final accuracy | Macro-F1 | Balanced accuracy | Benign FPR | Attack recall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 76.49% | 10.70% | 83.62% | 50.02% | 68.62% | 9.24% | 61.06% |
| 500 | 51.86% | 9.26% | 87.08% | 49.18% | 52.07% | 3.81% | 58.31% |
| 3,000 | 48.39% | 9.43% | 87.62% | 41.77% | 42.76% | 2.47% | 55.46% |

Larger replay raised final overall accuracy and reduced Benign false-positive
rate and forgetting. It did not improve the registered primary
minority-sensitive metrics: both candidates reduced balanced accuracy, attack
recall and average task accuracy, and replay 3,000 materially reduced Macro-F1.
Replay 50 therefore remains the reference for the next controlled stage.

This is a single-seed diagnostic. It neither establishes statistical
superiority nor selects a publication configuration across five seeds.

## Execution and integrity

- DICC Job: `414908`
- Final state: `COMPLETED`, exit code `0:0`
- Runtime: 1 hour 4 minutes 55 seconds
- Allocation: one A100 GPU, two CPUs, 8 GiB memory
- Maximum measured memory: 7.82 GiB
- Replay-500 deterministic result SHA-256:
  `1af720f57da30fb9585a2459112bbd3737c2f477f4e22eba5cb76abe72dfb946`
- Replay-3,000 deterministic result SHA-256:
  `2305d75cb6638fe45452a2a04c5276819335471cc5a80c57785128b7bccc829b`
- Protected checksum-registry SHA-256:
  `a088ce4d77daea4ec597f9da0c39059283627bab6a8a70974447909cb9b171f3`

The compact machine-readable comparison is in `summary.json`. Raw result files
remain in protected experiment storage; this repository publishes the metrics
and hashes required to audit the reported conclusion without exposing local
storage paths.
