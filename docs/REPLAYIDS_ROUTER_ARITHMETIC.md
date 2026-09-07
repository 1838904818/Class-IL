# ReplayIDS numerical isolation: fixed embeddings and crossed components

7 September 2026. This extends the [failed CPU forward-parity check](REPLAYIDS_FORWARD_PARITY.md). It diagnoses numerical sensitivity; it does not replace the registered scorer, improve a reported benchmark metric, or establish a completed calibration/SHAP/ETG experiment.

## Controlled comparisons

All five last-epoch final checkpoints were checked on the same 898 frozen probes. The original runtime files, protected checkpoints, saved probes, local test shards and selected feature rows were hash-verified. The diagnostic ran locally on CPU with two threads. No model was trained or fitted.

1. Compute one embedding matrix with encoder batch 32 and keep it fixed. Keep Head scores fixed too. Run the unchanged float32 Router with batches 1, 32, 128 and 898. Compare each with the 898-row Router call on the identical matrix.
2. On that same matrix, evaluate an explicitly separate reference: direct coordinate subtraction, float64 squared-distance accumulation and population z-standardization, followed by float32 output. Compare batches 32 and 898. This is not the production squared-norm expansion.
3. Recompute embeddings using a single 898-row encoder batch. Evaluate both Router formulas on these embeddings with identical whole-probe Router batching. Cross the Head scores from the two encoder batches with both Router outputs to isolate the observed prediction changes.

The official Router computes squared distance using `sum(x*x) + sum(c*c) - 2*x@c.T` in float32, clips negative roundoff to zero, then takes the nearest-centroid distance and per-row population z-score (`ddof=0`, epsilon `1e-8`). The direct64 reference instead sums `(x-c)**2` in float64. It changes arithmetic, not the stored model weights or centroids. None of its results is promoted as the registered primary.

## Measured changes

Maximum absolute Router-score differences:

| Seed | Original Router: changing only Router batch size | Original Router: encoder batch 32 vs 898 | Direct64: encoder batch 32 vs 898 |
|---|---:|---:|---:|
| 1 | 0.03695953 | 0.01115251 | 0.00001335 |
| 2 | 0.00950778 | 0.00539839 | 0.00000250 |
| 3 | 0.02503991 | 0.01090574 | 0.00001919 |
| 4 | 0.00832200 | 0.00357783 | 0.00001574 |
| 42 | 0.01292956 | 0.00491905 | 0.00000642 |

The first column proves Router arithmetic/batching sensitivity without changing the embedding matrix. Those fixed-embedding batch variations did not change argmax on these probes. The direct64 reference returned **bit-identical Router scores for batches 32 and 898** on each fixed embedding matrix. This bounded observation is not a proof of universal batch invariance.

Changing encoder batch size changed embedding components by at most approximately `1.91e-6` to `4.05e-6` per seed. The unchanged float32 Router produced much larger score differences; the direct64 differences remained within the previously fixed tolerance (`atol=1e-4`, `rtol=1e-5`) for these two CPU embedding matrices. Small embedding perturbations and cancellation-prone distance arithmetic are therefore a concrete sensitivity concern, not evidence that all cross-device errors have been isolated.

## What happened to the changed predictions?

With original Router arithmetic, seed 3 had zero differences from saved CUDA predictions using encoder batch 32, but one difference using encoder batch 898. Seed 4 showed the reverse: one difference with encoder batch 32 and zero with 898. Exchanging only the two Head score matrices did not change these counts. The crossed mismatch counts therefore depend on the Router branch in this tested configuration. Counts alone do not prove that every comparison changed exactly the same sample, or isolate all cross-device causes.

Direct64 gave the same mismatch counts across these two CPU embedding variants: zero for seed 3, but **one seed-4 prediction differed from saved CUDA** under each embedding batch. It is not an exact reconstruction fix for the old float32 scorer. We do not select arithmetic or batch size based on agreement with test labels, and do not interpret these mismatch counts as accuracy, forgetting, SHAP drift or ETG sensitivity.

## Research implications and next gates

- Keep the original protected scores and primary results immutable. Report the failed cross-environment parity gate; do not silently rewrite earlier results with direct64 outputs.
- For strict legacy reproduction, verify scores with the original GPU/software environment and canonical batching. This requires a separately reviewed job if local inspection is insufficient.
- For a numerically stable future scorer, freeze a versioned arithmetic contract and validate it independently. Direct64 is a candidate reference, not a demonstrated performance improvement or a new scientific contribution by itself.
- Before promotion, cover all checkpoints and seeds, compare full fixed-test metrics including per-class outcomes, signed forgetting, attack recall and Benign FPR, and assess runtime/memory cost. Reevaluate explanation/governance on the actual versioned score, not a mismatched wrapper.
- Only then fit a training-only calibration candidate with explicit rare-class support and no future-class tuning at earlier increments. No calibration was fitted in this work.

## Evidence and reproduction

[ROUTER_ARITHMETIC_ISOLATION.json](../results/replayids-router-arithmetic/ROUTER_ARITHMETIC_ISOLATION.json) records exact source/input hashes, per-seed embedding hashes, crossed comparisons, raw distance error and standard-deviation quantiles. Model weights, embeddings and raw traffic are not published in this package.

```text
python tools/isolate_replayids_router_arithmetic.py --runtime ORIGINAL_RUNTIME --inputs INPUT_DIR --data DATA_DIR --protocols PROTOCOL_DIR --registry PROTECTED_REGISTRY --output isolation.json
python tests/test_replayids_router_arithmetic.py
```

Synthetic tests check direct distances, near-equal large coordinates, input rejection and limited batch invariance. Published-report checks verify provenance and scope; they are not independent reruns of the neural experiment. The diagnostic used approximately 14 seconds per seed after setup, not including full input hashing, and is not a training-time estimate.
