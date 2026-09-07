# ReplayIDS: frozen-probe reconstruction gate

7 September 2026. **Gate failed: do not use this CPU reconstruction for a new calibration or explanation claim.** This follows the [input-readiness audit](REPLAYIDS_SCORE_READINESS.md), which verified files and splits but did not prove forward parity. Neither result changes the original protected benchmark metrics.

## Fixed check

The original last-epoch checkpoint 003 for each seed (1, 2, 3, 4, 42) was evaluated on the same 898 checksum-bound official-test probes. All 16 original runtime source files matched their protocol hashes before import. The original inference-state files, saved score files, probe manifest, all eight local test shards and every selected feature row were verified. No fitting, training, threshold selection or full-test metric calculation was performed.

The comparison checks the stored positive-probability Head scores, population-standardized Router scores, Joint scores (`head + 0.5 * router_z`), and final argmax. Absolute tolerance `1e-4` and relative tolerance `1e-5` were fixed before the first comparison. Passing requires all score cells within tolerance and no prediction mismatch; zero argmax mismatches alone is insufficient. Bit-exactness is reported separately.

The training environment was PyTorch 2.6.0+cu118 / NumPy 2.2.6 on an A100. Reconstruction used PyTorch 2.11.0+cu128 / NumPy 2.4.6 **on CPU**, with two threads and batch size 32. The installed FT-Transformer 0.6.1 implementation was source-hash checked. This is a cross-environment check, not a same-environment reproduction.

## Findings

| Seed | Head max absolute error | Router max absolute error | Joint max absolute error | Argmax mismatches / 898 |
|---|---:|---:|---:|---:|
| 1 | 0.00000870 | 0.02832448 | 0.01416230 | 0 |
| 2 | 0.00001442 | 0.00868922 | 0.00434458 | 0 |
| 3 | 0.00002044 | 0.02502990 | 0.01251489 | 0 |
| 4 | 0.00000721 | 0.00644028 | 0.00322008 | 1 |
| 42 | 0.00002021 | 0.01198298 | 0.00599149 | 0 |

Head scores passed the numerical tolerance in every seed; Router and Joint failed in every seed. None of these matrices was bit-exact. One of 4,490 seed-probe decisions differed, but these are repeated evaluations of the same 898 probe rows, **not 4,490 independent test samples**. This fraction is not a SHAP drift rate, classification error estimate, or ETG admission metric.

A diagnostic repeat on seed 4 using a single batch of 898 rows reduced the Router maximum error to 0.00578213 and Joint maximum error to 0.00289178. It produced zero argmax mismatches but still failed the unchanged score tolerance. Changing batch size therefore affected this CPU reconstruction; it did not repair the parity gate.

## Interpretation and next requirements

- The protected state can be loaded, but high-fidelity cross-environment score reconstruction has not been demonstrated. Do not present the CPU path as an exact deployed scorer.
- The original Router uses float32 squared distances followed by per-row z-standardization. Cancellation in squared-distance arithmetic and amplification of small differences are plausible mechanisms, not isolated causes demonstrated by this check. CPU/GPU kernels, library versions and batching are confounded.
- Do not increase tolerance or select a batch size merely to claim success. The batch-898 run is a diagnostic, not a promoted configuration.
- Next isolate embedding differences from Router arithmetic, including fixed-embedding batch tests and an independently labelled high-precision diagnostic. Any change to the production distance calculation creates a new scoring candidate and requires separate validation.
- A matching-environment GPU parity job remains an option after local isolation, subject to the normal exact-hash review and submission approval. No such job was submitted here.
- Revised calibration, forgetting, attribution or ETG results remain unestablished. A final-checkpoint probe check cannot validate the earlier checkpoint trajectory or five-seed explanation robustness.

## Evidence and rerun

The [five-seed report](../results/replayids-forward-parity/FORWARD_PARITY_CHECKED.json) and [seed-4 batch diagnostic](../results/replayids-forward-parity/FORWARD_PARITY_BATCH898.json) preserve precise errors, input/source hashes, environments, tolerances and failed-gate flags. Raw traffic, model weights and credentials are excluded. The separate original runtime and authorized input copies are required:

```text
python tools/check_replayids_forward_parity.py --runtime ORIGINAL_RUNTIME --inputs INPUT_DIR --data DATA_DIR --protocols PROTOCOL_DIR --registry PROTECTED_REGISTRY --output parity.json --seeds 1 2 3 4 42 --batch-size 32
```

Tests of the published report verify integrity and the failed-gate interpretation, not independent neural inference. The measured inference/reload time was approximately 6-7 seconds per seed on the local CPU, excluding source/data verification; this is not an estimate of training time.
