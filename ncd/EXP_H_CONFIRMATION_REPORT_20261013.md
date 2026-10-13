# Exp H confirmation extension — Work Order 19

Status at 19:06 on 2026-10-13 (Asia/Kuala_Lumpur): **S1 robustness complete; global cleanup and end-to-end smoke complete; formal S4 running on local RTX 4060 Ti, followed by S5; confirmation pending.** This is the single combined Section 19 report and will be updated with the two new settings. Delivery targets are 2026-10-14 for S1 and 2026-10-16 for S4/S5. No S4/S5 model, including smoke training, was run before the design below was written.

## 1. Scope and chronology

S1 was an exploratory discovery. Sections 19.2 and 19.3 were commissioned **after seeing S1**, so these are additional confirmation checks, not an analysis planned before the original experiment. The Section 17.6 intermediate verdict and the rejection of CRR are unchanged. Association is not a causal effect or an innovation claim. UNSW/Malaya cleanup reruns remain required but lower priority; neither blocks this extension. No oversampling is used.

## 2. S1 robustness — existing observations only

The frozen Section 17 S1 data contain 440 units from 8 methods × 5 seeds. Applying the unchanged pre-transition shifted-recall floor (<0.2) excludes 161, leaving 279 units and all five seed clusters. Average ranks of X and −Y are residualized on an intercept, rank(Z), additive method dummies and additive old-class dummies. This is not a method-by-class interaction or a causal model. Each of 2,000 bootstrap draws resamples five entire seed clusters with replacement and recomputes ranks and regression; the interval is the 2.5/97.5 percentile interval (random seed 1705). All draws are identified.

| S1 model | Partial Spearman | 95% seed-cluster CI | ≥0.2 and CI excludes zero |
|---|---:|---|---|
| Original control Z only | 0.434180 | [0.351888, 0.502028] | Yes |
| Required control Z + method + old class | **0.285364** | **[0.241491, 0.349309]** | **Yes** |

The S1 component of the Section 19.3 conjunction passes. That alone does **not** confirm harm; S4/S5 results are still required. With only five seed clusters, interval precision and external generalization remain limited.

### Per-method descriptive estimates

The requested per-method analysis controls Z. An additional old-class-adjusted column is shown to expose composition sensitivity, not to select a favorable variant. No per-method significance claim is made.

| Method | Eligible units | Control Z | Control Z + old class |
|---|---:|---:|---:|
| Replay | 37 | 0.471875 | 0.389191 |
| iCaRL | 33 | 0.200411 | 0.178122 |
| Replay + LwF | 37 | 0.588972 | 0.376966 |
| KD-perturbed | 37 | 0.384702 | 0.234324 |
| RCL-tab | 37 | 0.400584 | −0.059046 |
| Global-IR | 24 | 0.530143 | 0.578767 |
| Old-pair KD | 35 | 0.634135 | 0.489049 |
| CRR (unchanged rejected comparator) | 39 | 0.180762 | 0.045487 |

Positive counts: **8/8** controlling Z; **7/8** after the additional within-method class adjustment. Eligible class composition differs across methods.

### Leave one old class out

| Omitted old class | Remaining eligible units | Control Z | Control Z + method + class |
|---|---:|---:|---:|
| Normal | 159 | 0.516206 | **0.180254** |
| DoS | 185 | 0.431086 | 0.355898 |
| DDoS | 273 | 0.440557 | 0.287634 |
| Bruteforce | 228 | **0.265745** | 0.265207 |
| WebAttack | 271 | 0.452476 | 0.303008 |

Minimum: 0.265745 for Z-only, but **0.180254** for the fully adjusted analysis after removing Normal. Thus the signal stays positive in these deletions but does not uniformly retain the ≥0.2 threshold. Only five classes have eligible old-class transition units: Normal 120, DoS 94, DDoS 6, Bruteforce 51, WebAttack 8. Classes without such observations cannot supply a leave-one-class estimate. These deletions are descriptive, not additional decision gates.

## 3. S4/S5 design frozen before training

Frozen before new model execution on 2026-10-13. The exact machine-readable class/schema/cleanup/split specification in `ncd/exp_h_netflow.py` has SHA-256 **`7c0ca1ff207bc49db6131e90031dc3a2773a4638f2db0822b75766304144f96a`** (canonical sorted compact JSON). No class mapping, class order, endpoint, floor or threshold will be changed in response to model results. Data parsing may add only verified spelling aliases of the same declared labels, before training, and must fail on unknown labels rather than silently discard them.

Sources: the [UQ dataset authors' NetFlow-v2 description](https://staff.itee.uq.edu.au/marius/NIDS_datasets/), [NF-ToN-IoT-v2 official delivery](https://rdm.uq.edu.au/files/a4ad7080-ef9c-11ed-a964-b70596e96ad5), and [NF-CSE-CIC-IDS2018-v2 official delivery](https://rdm.uq.edu.au/files/ce5161d0-ef9c-11ed-827d-e762de186848). Cite Sarhan, Layeghy and Portmann, *Towards a Standard Feature Set for Network Intrusion Detection System Datasets*, 2022, DOI [10.1007/s11036-021-01843-0](https://doi.org/10.1007/s11036-021-01843-0). The official archive permits academic research with citation; it is not assumed to grant unrestricted commercial redistribution. Raw datasets and large weights stay local.

### Fixed common-family mapping

| Common family (fixed order) | NF-ToN-IoT-v2 | NF-CSE-CIC-IDS2018-v2 | Interpretation |
|---|---|---|---|
| Normal | Benign | Benign | Non-attack traffic |
| DoS | dos | DoS family, including the named DoS attack variants | Single-source denial-of-service family |
| DDoS | ddos | DDoS family, including named HOIC/LOIC variants | Distributed denial-of-service family |
| CredentialAttack | password | BruteForce family, including FTP/SSH brute force | Broad credential-attack family, **not identical attack implementations** |
| WebAttack | injection + xss | Web Attacks, SQL Injection, Brute Force -Web, Brute Force -XSS | Broad application-injection/web-attack family, **not identical subclass composition** |

Password in ToN includes brute-force or sniffing; consequently CredentialAttack is an explicitly chosen broader category, not an author-provided equivalence. WebAttack has heterogeneous subclass composition. No Backdoor↔Infiltration or Scanning↔Infiltration equivalence is assumed. ToN Backdoor, MITM, Ransomware and Scanning, and CSE Bot and Infiltration, are excluded from modeling but retained in the global conflict detection pass. No NF-UNSW training is added to S4/S5.

The raw schema is the same 43 NetFlow fields. Following the existing `src/data/nf_ton_iot_v2.py` identifier policy, both models omit source/destination IPv4 and source/destination L4 ports; **39 numeric features enter the models**, not 43. Label and Attack never enter X. Both sources must match the exact 43-column feature schema; there is no feature-alias alignment. Float32 representability/finite checks occur before grouping, signed zero is canonicalized, and exclusions are counted. No imputation from shifted data is permitted; the numeric scaler is fit on Task 0 source training only.

### Global cleanup and split (before caps or fitting)

1. Pool every raw file in each source, including supplied ToN train/test CSVs. The supplied split labels are not retained.
2. Compare complete canonical 39-feature float32 byte vectors across **both full sources**, excluding IPs, ports, labels and dataset identity from the key. Hash partitioning only accelerates grouping; full bytes determine equality.
3. For an identical feature group with more than one declared family label, remove the **entire** conflict group. Excluded attack types retain distinct labels in this check. Within-family subclass merging follows only the frozen table above.
4. Remove every remaining identical-feature group represented in **both** datasets from both sides. This symmetric exclusion avoids choosing one source as the winner and prevents direct exact overlap in either direction.
5. Retain one representative per remaining within-source feature group. Then retain only the five declared common families. Use deterministic feature-hash allocation (seed 42), approximately 80% training / 20% held out, separately counted per source/class.
6. Cap only training at 40,000 per class after cleanup and splitting, with deterministic selection. Keep **all** held-out observations, without class balancing, resampling or per-class caps. Each direction uses its own source holdout as ID and the other source's holdout as shifted test. Report actual sizes, duplicate/conflict removals and zero cross-view overlap checks.

S4 trains ToN and evaluates CSE shift; S5 reverses them. Class tasks are `[Normal, DoS] → [DDoS, CredentialAttack] → [WebAttack]`, retaining the two-classes-per-task CIC-style convention from Section 17. The final WebAttack class has no subsequent old-class transition and does not create a primary drift-harm unit. This yields six raw old-class transition units per cell, 240 per setting before the recall floor. The reciprocal directions share datasets and are **not independent external replications**.

Learners, frozen NSL-calibrated regularization, MLP training hyperparameters, PI instruments, seeds {1,2,3,4,42}, floor <0.2, seed-cluster bootstrap (2,000), and the Section 17.5 unit/method analyses are unchanged. Eight methods × five seeds × two settings = 80 formal cells. The same entry point must finish a one-seed, one-epoch, 64-rows/class end-to-end smoke before formal execution. Smoke values cannot support scientific claims. Checkpoint reuse is allowed only with matching configuration, source and prepared-data identities.

## 4. Confirmation rule — no current verdict

### Completed preparation and execution check

The two sources contain 35,834,204 raw rows (ToN 16,940,496; CSE 18,893,708). Preparation completed on 2026-10-13 in 361.52 seconds. It excluded 384 nonfinite/unrepresentable model-feature rows: ToN Benign 90 and MITM 41; CSE Benign 201 and DDoS-LOIC-UDP 52. No missing value was imputed from target data. The following accounting is over the remaining 35,833,820 rows:

| Cleanup quantity | Groups | Member rows |
|---|---:|---:|
| Conflicting mapped labels | 2,906 | 3,114,315 |
| Exact feature groups present in both sources | 3,655 | 5,953,429 |
| Union removed (the above categories overlap) | 5,595 | **6,399,536** |
| Additional within-source duplicate copies removed | — | **17,984,858** |
| Unique vectors retained before excluding non-common families | 11,449,426 | 11,449,426 |
| Non-common family vectors excluded after cleanup | 111,825 | 111,825 |
| Shared-family unique vectors available for splitting | **11,337,601** | **11,337,601** |

The union, duplicate copies and retained unique vectors sum exactly to the valid input count. Conflict and cross-source member rows must not be added separately. All eight saved arrays passed size/SHA-256 validation; raw input totals match both complete releases. Independent full-byte equality checks across the 2,600,150 saved rows found **zero within-view duplicate copies and zero overlap for all six pairs** of ToN/CSE train/holdout views. This is exact equality of the 39 canonical raw float32 features, not a claim about later standardization rounding. The full per-label and per-domain accounting is in `ncd/results/exp_h/section19/data_preparation.json` (SHA-256 `9157c7cb2cc4db2203245e8d7d3455f25d91964da332695c1d39e3cda1aa686e`), with the successful preparation log beside it.

| Domain | Training after the prespecified cap | Natural holdout (uncapped) | WebAttack train / holdout |
|---|---:|---:|---:|
| ToN | 200,000 | 488,083 | 40,000 / 222,879 |
| CSE | 132,021 | 1,780,046 | **217 / 49** |

CSE has only 266 distinct retained WebAttack vectors. This severe rarity and source/target composition difference are limitations, not reasons to oversample, change mapping or replace the holdout. The source cap removes training observations only; it never duplicates rare examples. WebAttack is the final task and is not a primary old-class transition endpoint, although its addition can affect earlier classes and it remains in prediction/retention summaries.

An initial preparation attempt stopped at a 256 MiB partition safety guard before model execution. Lossless per-chunk exact-key multiplicity compression fixed the high-frequency duplicate concentration without changing any cleanup rule; the original failure logs and source remain local. An independent expanded-versus-weighted check matched all removal counts and retained vectors in 205 cases. All 31 implementation tests passed. The same `run_exp_h_confirmation_local.ps1 -Smoke` entry then completed on the real prepared data: 16 cells, 96 transition rows, 80 retention rows and 48 checkpoints; exit 0 and empty stderr. Smoke uses at most 64 rows per class (only 49 CSE WebAttack holdout rows exist), one seed and one epoch; its analysis correctly remains **pending**, not scientific confirmation. Execution source: `562e65d0bb868e321bd2f97f571dfbe0e14e2b48`.

### Registered rule

**Harm confirmed** only if at least one of S4/S5 has primary partial Spearman ≥0.2 with its 95% CI excluding zero **and** S1 adjusted for method and class meets the same condition. Secondary PI and descriptive method correlations cannot replace a failed primary condition. If both S4 and S5 miss their condition, report **harm not confirmed**. A missing or unidentified required result remains pending, not a negative finding. If a new setting passes but S1 fails, the conjunction is not met and the discordance is reported explicitly; no threshold is changed. Causal harm or method effectiveness is not established by this rule.

| Component | Status | Current result |
|---|---|---|
| S1 method/class adjustment | Complete | Pass; 0.285364 [0.241491, 0.349309] |
| S4 ToN → CSE | Running | Formal execution started; no completed-setting estimate |
| S5 CSE → ToN | Next in the same local process | No formal results |
| Joint Section 19.3 decision | **Pending** | Cannot decide before new settings |

## 5. Reproducibility and limits

S1 input: `ncd/results/exp_h/section17/registered/exp_h_result.json`, SHA-256 `4b2724122bc5b3ddcfd1a2b6e02601b73c763c14a88c05d40ca3e9d6e41b0e7b`; experiment source `a8b90215a0ad369a11a5cbb07b2176e9d821f5f3`. The Section 19 work-order base is `539728e6a7161e840a27859ca0d8455615338d71`; the reanalysis separately records its script hash, command and log. Machine-readable estimates, all 2,000 draws, per-method/leave-one-class outputs and the run log are under `ncd/results/exp_h/section19/s1_robustness/`.

Reproduce S1: `python ncd/analyze_exp_h_robustness.py --input-dir ncd/results/exp_h/section17/registered --output-dir <new-output-directory> --bootstrap 2000 --bootstrap-seed 1705`. Seven numerical/contract unit tests passed. The original Z-only result and CI reproduce the frozen Section 17 values. An independent full-dummy/pseudoinverse implementation also recomputed both point estimates, all 2,000 bootstrap draws and all descriptive subsets; maximum absolute discrepancy was 6.66×10⁻¹⁶.

Formal S4/S5 started locally at 19:06 on 2026-10-13 using source `562e65d0bb868e321bd2f97f571dfbe0e14e2b48`: `python -I ncd/run_exp_h_confirmation.py --config ncd/configs/exp_h_section19.json --cleaned-data-root <prepared-cache-directory> --output-dir ncd/results/exp_h/section19/local_v1`. Log: `ncd/results/exp_h/section19/local_v1/run.log`; checkpoint/resume is enabled. The first S4 Replay seed-1 checkpoint was observed, with active GPU work and empty stderr. Partial checkpoints are not confirmation evidence. Source identity remains this execution commit despite later report-only commits. Ten-minute monitoring checks CPU/GPU activity and output/error state; PI has no per-feature log and log silence alone is not a failure. Failed implementation attempts retain their logs, and valid checkpoints are resumed rather than restarted.

The downloaded official CSE archive has SHA-256 `ad6307995d08ff8825d0193d5a1cfa61cf3dc4c6f223962ba3762a7775b5dc24` (612,059,380 bytes). The pre-training report version is preserved in commit `562e65d0bb868e321bd2f97f571dfbe0e14e2b48` (report SHA-256 `8b057f79d141b25752129d654e45e9e8a29c7edb60e05f189e27f2a8327ce5a4`); preparation records that version, not later status edits. Preparation source SHA-256 is `7c4a9fd22d433aaa65e0209083ca589648a442d0bd51b3d399b3a6ae69ea4af2`. Completed S4/S5 estimates will replace pending entries without changing the frozen design. S1 still inherits Section 18 limitations; this new cleanup does not retroactively clean historical results or discharge the separate UNSW/Malaya obligations.
