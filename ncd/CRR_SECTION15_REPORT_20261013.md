# CRR Section 15 results — 2026-10-13

## Decision

CRR does not meet all registered criteria. The local final hijack gate failed, so Section 15.3 replaces criterion 1. This is a completed bounded experiment, not evidence of substantive innovation or publication readiness.

## Provenance and scope

Necessity: DICC 485405/485406, source `bfde946a44ea7838f07399cf14c7570a40df7dca`, 80 rows per dataset. Real sequence: first-complete local RTX 4060 Ti results, NSL source `1560401b806371c117b080cbf5ca0bd4c30ab179`, UNSW source `e5817f70e84bb86e301580965fef3d633d30c6db`, 40 rows per dataset (computational files identical between these commits). Seeds are 1,2,3,4,42; lambda selection used only 5,6. Local real and HPC necessity use different devices: comparisons use within-run Replay-normalized training cost; absolute cross-device cost is not compared. HPC NSL 485498 is replication; HPC UNSW 485499 is still running and is not required for this adjudication.

All raw per-cell JSON/CSV and PI drift rows are under `results/crr/section15/registered`. Necessity protected manifests were checked; selected transferred merged results and local files are hashed in `ARTIFACT_HASHES.json`. No data files or checkpoints are published.

## Statistical rule

Exact one-sided paired sign-flip test over five seeds (32 assignments), alpha 0.05, following the existing Section 12 implementation; p-values are unadjusted. Effect-size and numerical thresholds must also pass. k=3 is primary; k=2 is separately reported and retained in the registered k>=2 requirement. Reported means are descriptive, not confidence intervals.

## Local final hijack gate

The pre-existing `HIJACK_VERDICT.json` reports `passed=false`, threshold 0.15, and `fallback_section_15_3_active=true`. No hijack-prevention claim is supported.

## NSL-KDD

| Method | Necessity BA k=3 | Necessity BA k=2 | Real Macro-F1 | Old-pair PI drift | Real training / Replay |
|---|---:|---:|---:|---:|---:|
| replay | 0.764129 | 0.735188 | 0.613780 | 0.228106 | 1.0000 |
| icarl | 0.659920 | 0.654973 | 0.621391 | 0.176713 | 1.0098 |
| replay_lwf | 0.762038 | 0.771649 | 0.521112 | 0.167376 | 1.0250 |
| kd_perturbed | 0.760094 | 0.764638 | 0.519848 | 0.159929 | 0.9989 |
| rcl_tab | 0.779276 | 0.738592 | 0.596142 | 0.181960 | 2.0208 |
| global_ir | 0.376434 | 0.408029 | 0.367246 | 0.153827 | 1.0966 |
| oldpair_kd | 0.715013 | 0.703579 | 0.558102 | 0.179687 | 1.1265 |
| crr | 0.641917 | 0.640442 | 0.542787 | 0.176468 | 1.2103 |

Criterion 1: FAIL; drift reduction 22.64%, paired p=0.03125, CRR/RCL-tab=0.9698.
Criterion 2: FAIL.
- k=3: FAIL; CRR minus Replay -12.221 percentage points; p above RCL-tab=0.96875, Global-IR=0.03125.
- k=2: FAIL; CRR minus Replay -9.475 percentage points; p above RCL-tab=1.00000, Global-IR=0.03125.

Criterion 3: FAIL; CRR minus Replay -7.099 points, minus RCL-tab -5.335 points.
Criterion 4: PASS; real training ratio 1.2103, necessity k2/k3 ratios 2.4485/2.4429.

Dataset overall: FAIL.

## UNSW-NB15

| Method | Necessity BA k=3 | Necessity BA k=2 | Real Macro-F1 | Old-pair PI drift | Real training / Replay |
|---|---:|---:|---:|---:|---:|
| replay | 0.981448 | 0.927550 | 0.368901 | 0.326907 | 1.0000 |
| icarl | 0.822834 | 0.750464 | 0.291742 | 0.289143 | 1.2641 |
| replay_lwf | 0.982211 | 0.957082 | 0.374398 | 0.279149 | 1.2775 |
| kd_perturbed | 0.982423 | 0.970786 | 0.377150 | 0.280168 | 1.3353 |
| rcl_tab | 0.981172 | 0.962339 | 0.378867 | 0.236730 | 5.2586 |
| global_ir | 0.450612 | 0.452848 | 0.286381 | 0.193885 | 1.6624 |
| oldpair_kd | 0.980038 | 0.926469 | 0.369025 | 0.286466 | 1.7565 |
| crr | 0.846844 | 0.817842 | 0.352569 | 0.253843 | 2.2646 |

Criterion 1: FAIL; drift reduction 22.35%, paired p=0.03125, CRR/RCL-tab=1.0723.
Criterion 2: FAIL.
- k=3: FAIL; CRR minus Replay -13.460 percentage points; p above RCL-tab=1.00000, Global-IR=0.03125.
- k=2: FAIL; CRR minus Replay -10.971 percentage points; p above RCL-tab=1.00000, Global-IR=0.03125.

Criterion 3: FAIL; CRR minus Replay -1.633 points, minus RCL-tab -2.630 points.
Criterion 4: PASS; real training ratio 2.2646, necessity k2/k3 ratios 2.5305/2.5214.

Dataset overall: FAIL.

## Interpretation and next decision

The registered evidence rejects the claim that CRR satisfies the combined stability, plasticity, predictive and cost criteria in these two settings. Retain the negative result and raw evidence. Further mechanism changes require a new explicit scientific plan; do not retune this registered comparison after inspecting the test results. The broader locked research questions remain open.

## Additional registered endpoints

Training ratios above exclude PI evaluation; its cost is separately reported below. Forgetting uses the runner definition: mean historical-best minus final recall over all classes, including final-task classes.

| Dataset | Method | Forgetting | Training seconds | PI evaluation seconds |
|---|---|---:|---:|---:|
| NSL-KDD | replay | 0.033387 | 14.538 | 21.146 |
| NSL-KDD | icarl | 0.036438 | 14.678 | 41.767 |
| NSL-KDD | replay_lwf | 0.017213 | 14.829 | 20.962 |
| NSL-KDD | kd_perturbed | 0.017743 | 14.478 | 21.266 |
| NSL-KDD | rcl_tab | 0.026781 | 29.287 | 19.996 |
| NSL-KDD | global_ir | 0.007456 | 15.867 | 21.232 |
| NSL-KDD | oldpair_kd | 0.031641 | 16.305 | 20.363 |
| NSL-KDD | crr | 0.029432 | 17.543 | 20.290 |
| UNSW-NB15 | replay | 0.159509 | 19.755 | 178.800 |
| UNSW-NB15 | icarl | 0.266305 | 24.854 | 341.958 |
| UNSW-NB15 | replay_lwf | 0.133869 | 25.145 | 177.960 |
| UNSW-NB15 | kd_perturbed | 0.128427 | 26.280 | 180.253 |
| UNSW-NB15 | rcl_tab | 0.131579 | 103.421 | 175.605 |
| UNSW-NB15 | global_ir | 0.077904 | 32.747 | 175.165 |
| UNSW-NB15 | oldpair_kd | 0.119168 | 34.579 | 179.186 |
| UNSW-NB15 | crr | 0.126125 | 44.620 | 178.069 |

## Final hijack attempt raw outcomes

| Dataset | Seed | rho | Target | Marker recall drop | Decorrelation recall drop |
|---|---:|---:|---|---:|---:|
| NSL-KDD | 1 | 0.9 | DoS | 0.022520 | 0.006434 |
| NSL-KDD | 1 | 1.0 | DoS | 0.037936 | 0.013941 |
| NSL-KDD | 2 | 0.9 | DoS | 0.023592 | 0.004290 |
| NSL-KDD | 2 | 1.0 | DoS | 0.065013 | 0.027882 |
| UNSW-NB15 | 1 | 0.9 | Generic | 0.005935 | 0.007154 |
| UNSW-NB15 | 1 | 1.0 | Generic | 0.234858 | 0.066875 |
| UNSW-NB15 | 2 | 0.9 | Generic | 0.008797 | 0.011075 |
| UNSW-NB15 | 2 | 1.0 | Generic | 0.289545 | 0.150178 |
