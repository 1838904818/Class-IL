# Native-context component profile: measured result

11 September 2026. **Completed local component/resource check; no intervention efficacy result.**

[Machine-readable evidence](profile_evidence.json) exports the original result and terminal JSON objects, recording their original byte hashes separately from export formatting. Original artifacts remain retained in the run directory. The experiment used native D2 ReplayIDS-aligned CIC-IDS-2017 checkpoints 000 and 001 from training seed 1, not pooled datasets.

| Measurement | Observed value |
|---|---:|
| Native forward calls | 84 |
| Native context rows processed, including repeats | 25,914 |
| Mask/checkpoint/target records | 24 |
| Exact repeat and adapter matches | 24 / 24 |
| Component profile window | 5.5 s |
| Complete supervised process window | 17.766 s |
| Peak polled owned-process RSS | 997,142,528 bytes |
| Peak PyTorch CUDA allocated | 1,208,628,224 bytes |
| Peak PyTorch CUDA reserved | 1,608,515,584 bytes |

The two target contexts retain 512 and 105 native companion rows. Six fixed perturbation conditions were checked per target and checkpoint. These are not 25,914 unique held-out samples. Resource counters cover this bounded workload; polling is not a hard OS memory guarantee and allocator counters exclude driver overhead.

The current RTX 4060 Ti / PyTorch 2.11.0+cu128 environment differs from the archived A100 environment. Exact same-environment repetition and adapter agreement do not establish equality with archived scores or full-population reproduction. No SHAP attribution vectors, calibration fits, ETG action decisions, accuracy gains or forgetting improvements were computed. No W&B write or HPC access occurred.

## Next evidence required

A separately reviewed complete-permutation resource pilot must measure real SHAP calls, vector validity and reconstruction before full extraction. A handful of manually specified masks cannot substitute for that test. Preserve native companion batches, the predeclared reference pool and role separation. The full pilot's 5,144,704-call planning arithmetic is not a measured runtime forecast; neither the 18-second process duration nor this small check licenses a full run. Subsequent efficacy testing must retain the frozen fair selectors and report negative or unavailable outcomes honestly.
