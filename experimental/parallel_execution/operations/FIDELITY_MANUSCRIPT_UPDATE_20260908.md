# Numerical reconstruction update

8 September 2026. Proposed insertion into manuscript v3.3.1 Section 4.10 and the response to W7, D17 and Q11. This text has not yet been incorporated into a rendered manuscript revision.

## Manuscript text

A subsequent A100 reconstruction reproduced the archived ReplayIDS probe scores across all five seeds (1, 2, 3, 4 and 42) and four checkpoints. The 20 checkpoint records contain 60 comparisons of head probabilities, router-normalised scores and joint scores. All comparisons showed exact numerical equality and zero prediction changes. The encoder batch size was fixed at the historical value of 512; head and router scoring operated on each complete checkpoint probe matrix. Probe counts of 256, 512, 768 and 898 per seed yield 12,170 repeated probe evaluations, not independent observations or a full-test evaluation. Job 454941 completed in 71 seconds. No training or calibration was performed, and this check does not add an accuracy estimate.

The result establishes reconstruction of these archived probe outputs under the recorded environment and batching contract. It does not establish numerical equivalence for arbitrary batching, perturbed inputs, all test rows or other datasets. The comparator uses elementwise numerical equality rather than bytewise identity. Historical driver, binary and kernel identity remains unproven. In particular, this ReplayIDS check does not validate the separate Malaya attribution experiment or the gradients used by an attribution wrapper. The earlier failed cross-environment diagnostic remains part of the numerical-portability evidence.

## Replacement response to W7 D17 Q11

Status: PARTLY CLOSED. The archived ReplayIDS probe reconstruction has now completed with exact numerical agreement and no prediction mismatches across 20 seed/checkpoint pairs. Protected result SHA-256: baf055d3ab9bb800a887ad5f5659be3ea222bf11b2fc2c18da626ba4c743e5e5. No historical scores or tolerances were changed. This addresses the bounded reconstruction failure, not the entire reproducibility or no-look-ahead request.

Remaining work includes prospective D2 sampling without future-class counts, capture/group provenance, and the requested attribution-target and perturbation checks. The successful ReplayIDS comparison cannot close the Malaya gradient-faithfulness question. Full historical environment identity and full-population reconstruction are not claimed.

## Replacement dependency wording for faithfulness

ReplayIDS archived-probe reconstruction is now available as a prerequisite for further work. Before interpreting new attribution results, verify the exact target, input grouping and numerical behaviour of that attribution path, including perturbed inputs. A class-batch CPU attribution path must not inherit the GPU whole-probe result without its own comparison. Gradient-independent attribution and deletion/insertion controls, repeated backgrounds and matched evaluation budgets remain required; ROAR/KAR would require retraining.

## Evidence

- Job 454941 protected GPU_FIDELITY.json, SHA-256 above.
- Core manifest SHA-256: e39e86e6946661e55f0ca412cba28268b0f35eb2ff22c9bfce7ebd9fd5c2d6fd.
- Checkpoint-level audit: experimental/parallel_execution/operations/FIDELITY_VERIFIED_SCOPE.md.
- Recorded tracking URL: https://wandb.ai/csnet/ofra-etg-leon-hpc/runs/c4079549 . Client completion receipt is available; cloud API verification is not asserted.

## Publication gate

Preserve prior manuscript files and their hashes. Apply these paragraphs to a new revision, check all related claims and response statuses, render and inspect every page, then publish the new revision and its checksums to both repositories. Do not label this source amendment a completed rendered manuscript.
