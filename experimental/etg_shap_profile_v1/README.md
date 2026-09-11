# Single-target native SHAP resource profile

This is a new local-only execution candidate, not reuse of the component-profile approval. The previous component check completed with 84 native calls, 24 matching records and 17.766 seconds total; it does not establish SHAP runtime.

## Frozen scope

Use training seed 1 checkpoint 000, the first preselected Benign trigger row, its original 512-row context, and all 32 fixed Task-0 references. The loader also reconstructs the two already-bound contexts to verify lineage; only the first is explained. The worker does not invoke the separate fit, acceptance or evaluation selection/label APIs, fit an action, or compute test efficacy. Complete parent batches still contain companion features from other roles, and the original checkpoint loader reads its probe validation arrays. These accesses and class-stratified provenance are not blinded and do not establish untouched evaluation data.

Run SHAP 0.51.0 PermutationExplainer with seed 20260911, max_evals 157 and batch_size 1. Exact-equality Independent-masker invariance and lossless float32 conversion preserve the already-reviewed target definition. This is one explanation estimate with an antithetic permutation budget, not eight repeats and not the final noise-aware selector study. Exact invariants can change internal permutation counts. No reference or feature is removed to make the run faster.

The output is a signed 78-feature vector, native margin, base value, additive residual, actual native-call count, walltime and memory. Every perturbation replaces one target row while keeping its original companion rows. Reconstruction and repeat agreement do not establish explanation truth, historical A100 parity or a governance benefit. A zero vector remains a feasible extraction only, not zero harmful drift or an admissible noise-aware arm.

## Resource and execution controls

One supervised local worker; 2 CPU threads; PyTorch allocator limit 3 GiB; polled owned-process RSS limit 8 GiB; at most 5,200 native calls, 900 seconds in the call window and 960 seconds in the supervised process window. Admission retains at least 4,096 MiB free GPU memory and utilization at most 10 percent. Do not terminate or modify unrelated processes. These are limits, not measured SHAP runtime forecasts. No background fan-out, network logging or HPC operations.

The prospective call upper bound uses 32 references times a 157-mask budget plus direct pre/post calls. Actual skipping and wrapper calls must be measured; exhausting a budget is failure, not a partial successful attribution. The larger time limit relative to the 84-call component check accommodates a complete explanation and package initialization, with a parent watchdog.

Default mode validates hashes and metadata without loading project tensors. Execute mode requires an independently issued local-shap-profile-only receipt bound to candidate hash, run ID and create-only output directory. Review records are operational authorization, not digital signatures. Failed output directories are spent; any revision requires a new candidate and independent review.

Do not treat a successful profile as permission to run the 64-row, two-checkpoint, eight-repeat study. First assess its measured cost, complete the new candidate and review it separately. No classifier training, calibration fitting, ETG action or efficacy metric is authorized here.
