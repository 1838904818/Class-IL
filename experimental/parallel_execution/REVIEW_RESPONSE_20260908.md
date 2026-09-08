# Response to the complete automated review

8 September 2026. Response update to revision v3.3.1; the v3.3.2 manuscript amendment awaits rendered quality assurance. The 41 entries below cover every weakness, detailed comment and author question. Repeated requests share a response group. Editorial completion is not experimental completion. The review is automated feedback, not a journal decision.

## Principal changes

Governance has been removed from the title and headline contribution. ETG remains an auditable, method-conditioned ledger. Existing five-seed threshold grids, complementary stability metrics, top-15 policy sensitivity and performance-stratified consensus are now presented. The exact-forward/surrogate-backward limitation is explicit. No new training, attribution, calibration or intervention experiment was executed.

## How to read the status

CLOSED means the stated editorial or existing-evidence task is complete, not that algorithmic superiority is proven. PARTLY CLOSED means useful evidence was recovered but the full scientific request remains open. NEW EXPERIMENT REQUIRED means the plan is defined but no new result is claimed.

## Novelty

Comments W1. Status NEW EXPERIMENT REQUIRED. Manuscript 1.4, 2.3 and 5.5.

W1 — Algorithmic novelty is limited; composition does not isolate a causal advantage.

We no longer present the combination of established components as a demonstrated novel algorithm. CLEX already studies explanation drift in class-incremental learning, and Drift2Act already studies budgeted drift-to-action control. The candidate investigates state change, score-normalisation expansion, and new-rival competition separately. Its telescoping score decomposition is an order-dependent diagnostic, not a unique causal attribution. Synthetic checks establish arithmetic only.

Remaining work: Test whether the expansion-aware trigger improves later held-out risk/retention at a fixed repair budget against the same repair driven by random, periodic, error-only, disagreement-only and audit-only controls. Reject the mechanism if it provides no incremental benefit.

## Faithfulness

Comments W2, D1, D3, Q3. Status NEW EXPERIMENT REQUIRED. Manuscript 3.5, 4.2 and 6.

W2 — Piecewise routed scoring and failed IG raise attribution-faithfulness concerns.
D1 — Source-bound attribution is praised as avoiding surrogate-score pitfalls.
D3 — Use gradient-independent attribution or perturbation checks for the routed margin.
Q3 — Gradient-independent SHAP or ROAR/KAR faithfulness check?

We correct the favourable review premise: identical forward values do not eliminate a surrogate-backward limitation. The historical wrapper uses exact NumPy forward values with detached correction and a differentiable surrogate gradient. Finite gradients do not establish derivatives of the deployed implementation. The analysis concerns CPU true-class batches, not archived GPU score arrays. Piecewise differentiability alone does not prove that IG must fail. Existing deletion tests and Feature Ablation are useful but do not establish a full faithfulness guarantee.

Remaining work: ReplayIDS archived-probe reconstruction has completed, but does not validate the separate Malaya attribution path, gradients, arbitrary batching or perturbed inputs. Verify the attribution target and numerical path, then compare direct-score finite differences, gradient-independent attribution and deletion/insertion controls, with repeated backgrounds and fixed budgets. Distinguish a routed class margin from the softmax-probability deletion statistic; do not rename one as the other. ROAR/KAR requires retraining and is not already complete.

## Utility

Comments W3, D16. Status NEW EXPERIMENT REQUIRED. Manuscript Title, Abstract, 3.6, 4.5 and 5.3.

W3 — Offline ETG has no demonstrated operational or analyst benefit.
D16 — Show practical ETG triage actions, costs, and acceptable thresholds.

Governance is removed from the title and headline contribution. ETG is retained as an audit ledger and its historical state names are explicitly not safety certificates. In 72 checkpoint-class records with recall below 20%, 27 have at least one historical CERTIFIED_STABLE state. Low recall can therefore coexist with the ledger rule. Human-review action strings are simulated records, not evidence that an analyst acted. The old ledger and results remain immutable.

Remaining work: A separately versioned controller may nominate bounded recalibration or head repair, with independent acceptance data and matched label/compute budgets. No prediction suppression is permitted. If the trigger does not beat simple controls, retain only supplementary audit findings or remove ETG from the main method.

## Capacity

Comments W4, D5, Q1. Status NEW EXPERIMENT REQUIRED. Manuscript 3.1, 4.3 and 6.

W4 — FT512x12 versus FT256x4 creates a capacity confound.
D5 — Add capacity-matched sweeps on Malaya and a smaller dataset.
Q1 — Capacity-matched FT256x4 versus FT512x12 comparison?

Dataset-specific prediction audits and within-dataset score-arm diagnostics are distinguished from capacity-matched architecture evidence. FT512x12 Malaya cannot establish superiority over FT256x4 results from other datasets. Existing TabM and capacity diagnostics do not close this matched-design request.

Remaining work: Use identical data, splits, class order, optimiser, exposure and replay accounting for at least FT256x4/FT512x12 on Malaya and one intrusion dataset; report memory/compute as well as all predictive metrics.

## Baselines

Comments W5, D9, Q9. Status NEW EXPERIMENT REQUIRED. Manuscript 2.1, 3.10 and 4.9.

W5 — Replay comparator has different loss, memory and exposure; strong CIL baselines are missing.
D9 — Compare TRIL3 prototype pseudorehearsal under a matched protocol.
Q9 — Rehearsal-free/prototype baseline and memory/privacy trade-offs?

Balanced Replay50 is relabelled a same-data system comparison, not an architecture-only ablation. Equal exemplar counts omit centroids and model/optimizer memory. Binary focal heads and shared multiclass loss also differ. Cumulative CatBoost/TabM remain headroom diagnostics. TRIL3, PEARL and C-LoRA are discussed with their actual domains and adaptation requirements; no borrowed baseline is called a reproduced result.

Remaining work: Run a controlled baseline ladder matching loss, balanced sampling, updates/row exposure, total memory and backbone capacity. Add a pinned TRIL3 adaptation or rehearsal-free baseline under the same task stream. An adaptation must be labelled as such; low-rank routing is not automatically the published method.

## Stability

Comments W6, D6, Q6. Status NEW EXPERIMENT REQUIRED. Manuscript 4.1 and 4.2 plus audit supplement.

W6 — Top-k, threshold, fixed probes and one split limit explanation conclusions.
D6 — Add cross-split, alternative stability metrics, and faithfulness tests.
Q6 — Cross-split, background and counterfactual probe stability?

All five saved EG analyses contain 64 threshold settings each, now recovered and verified (320 seed-setting rows). All 100 saved transitions also contain cosine and Kendall tau-b. These quantify complementary aspects of the same rankings, not independent replications. Five training seeds share one split and fixed probes/backgrounds.

Remaining work: Use disjoint capture/data splits, multiple training-only backgrounds, repeated attribution seeds and meaningful perturbations. Changed top-k deletion admissions require new score evaluations. Do not treat a test-set threshold sweep as external calibration.

## Thresholds

Comments W9, Q2. Status PARTLY CLOSED EXISTING EVIDENCE. Manuscript 3.5, 3.6 and 4.1 plus audit supplement.

W9 — Jaccard 0.70 and the recall guard need rationale and sensitivity.
Q2 — Sensitivity across k, Jaccard and recall guards?

The guard is one-sided and strict: delta recall > -0.05, not a symmetric +/-5 pp band. Drift uses Jaccard <0.70. At k=15 and the 5 pp guard, EG mean seed rates across Jaccard thresholds 0.5/0.6/0.7/0.8 are 3.69/48.20/73.98/96.40%. We also replay the historical top-15 ledger at 16 Jaccard/guard settings for each of three methods and five seeds (240 policy replays), verifying all 450 primary states/actions. The thresholds are study-defined choices, not externally validated safety standards.

Remaining work: Full changed-k governance sensitivity needs deletion scores for each selected set. Operational threshold calibration and utility still require held-out experimental evidence.

## Fidelity

Comments W7, D17, Q11. Status PARTLY CLOSED; additional experiments required. Manuscript 3.10, 4.10 and 6; updated Section 4.10 awaits rendered publication.

W7 — D2 uses future-class counts and numerical reconstruction fails.
D17 — Pin environment/toolchain/container artifacts and demonstrate reproducibility.
Q11 — Container or GPU-matched rerun to reconcile numerical fidelity?

Task-0 numerical/vocabulary fitting is closed for the strict five-dataset table; D2 still uses later-class fitting counts for offline construction. These are different issues. The failed CPU/GPU comparison is retained. Pinning code or a container is necessary but not evidence of numerical parity. The exact source, score path, device, dtype, batch partition and toolchain all belong to the reproduction contract.

Completed since the previous response: Job 454941 reconstructed the archived ReplayIDS probes on A100 across five seeds and four checkpoints. All 60 head/router-z/joint comparisons were numerically equal, with zero prediction mismatches and no change to historical scores or tolerances. The 12,170 probe evaluations include repeated observations across checkpoints and seeds. The comparator checks numerical equality, not bytewise identity. Protected GPU_FIDELITY.json SHA-256 is baf055d3ab9bb800a887ad5f5659be3ea222bf11b2fc2c18da626ba4c743e5e5. Full historical driver/build/kernel identity remains unproven. This closes only the bound archived-probe reconstruction subtask.

Remaining work: Derive and evaluate a prospective sampling arm without future-class count access. Verify the numerical target for subsequent attribution and perturbed inputs; the successful whole-probe ReplayIDS run does not certify the Malaya class-batch attribution path. Complete capture/group provenance and archive missing environment evidence where recoverable. No full-population fidelity, calibration benefit or attribution faithfulness is claimed.

## Presentation

Comments W8, D15. Status CLOSED EDITORIAL. Manuscript Title, Abstract, 1.4 and Results signpost.

W8 — Dense interleaved caveats obscure the primary findings.
D15 — Stable accuracy and stable explanations must remain distinct.

The revised manuscript separates primary predictive comparisons, explanation/audit findings and exploratory diagnostics. Stable recall is not described as safety, class-balanced accuracy, or stable explanation. Governance is no longer a headline claim. Historical failed results and scientific limitations remain visible but repeated operational caveats are consolidated.

Remaining work: No new experiment is needed for this editorial correction. Scientific claims remain bounded by the remaining experimental gaps.

## Openset

Comments W10, D7, D13, Q8. Status NEW EXPERIMENT REQUIRED. Manuscript 2.4 and 4.6.

W10 — Open-set baseline comparisons are absent.
D7 — Open-set pilot needs standard curves, OpenAUC, and baselines.
D13 — Evaluate two-stage OpenMax/EVM-style rejection.
Q8 — OpenMax/EVM/energy filter for unknown discovery?

The single-seed FTP-Patator result already reports an OSCR-style AUC and a failed candidate buffer. It is not a complete standard OSCR curve/OpenAUC benchmark. VAEMax uses payload features, OpenMax and a second VAE stage; it is not a drop-in comparator for the current flow-feature model. No unlabelled semantic head is created.

Remaining work: Compare confidence, distance, energy and EVT/OpenMax-type rejection with matched features and training-only calibration; expand held-out classes/seeds, report OSCR curve definition, AUROC/AUPR/OpenAUC and accepted-known error. Introduce a semantic class/head only after labels arrive.

## Routing

Comments W11, D11, D12, Q7. Status NEW EXPERIMENT REQUIRED. Manuscript 2.4 and 5.5.

W11 — Task-ID/unknown alignment and geometry-consistent routing need comparison.
D11 — Compare task-specific BN with unknown-class alignment.
D12 — Investigate geometry-consistent/prototype-bank routing.
Q7 — Unknown-class/task-ID mechanisms to assist routing?

GCR evaluates industrial visual anomaly detection; task-specific BN plus unknown-logit task-ID prediction evaluates image CIL. Their mechanisms motivate routing controls but their published scores cannot be compared directly with NIDS. DP-Means remains the measured router; the same-state head/router/joint diagnostics do not constitute these external baselines.

Remaining work: Evaluate a shared-feature nearest-prototype bank and a separately adapted unknown/task-ID head under the same class stream. Record domain adaptations, old/new-class routing confusion and calibration support.

## Transfer

Comments W12, D14. Status CLOSED DISCUSSION TEST PENDING. Manuscript 3.1 and 6.

W12 — Cross-dataset generalisation and dataset-curation pitfalls need attention.
D14 — Discuss or test CIC17/CIC18 transfer and feature harmonisation.

Within-dataset five-seed results do not establish cross-dataset transfer. CIC17/CIC18 feature names alone do not guarantee common units, extractor versions, directionality or label definitions. Preserve training-only transforms, harmonise only justified common features and distinguish common known classes from unknown target classes. Malaya application labels cannot be recast as attack labels.

Remaining work: A transfer experiment remains a separate preregistered study. Capture holdouts, duplicate/leakage audits and feature/label harmonisation must precede CIC17-to-CIC18 comparisons; do not merge incompatible labels to obtain a larger accuracy number.

## Fusion

Comments D2, Q4. Status NEW EXPERIMENT REQUIRED. Manuscript 3.2 and 5.5.

D2 — Fixed p + 0.5z fusion should be calibrated on an appropriate holdout.
Q4 — Calibrated fusion under training-only cost constraints?

The 0.5 weight is a fixed project setting, not learned calibration or a published optimum. Existing head/router/joint results show trade-offs but cannot select a deployment coefficient. Class-set expansion changes score normalisation and the competing margin even when old parameters remain frozen.

Remaining work: Fit monotone/cost-sensitive fusion on increment-available fitting data, choose it on a separate acceptance holdout with predeclared missed-attack/FPR constraints, and evaluate only on locked later data. Include fixed 0/0.5/1 coefficients and an error-only trigger control.

## Statistics

Comments D4. Status CLOSED REANALYSIS. Manuscript 3.10 and supplementary paired statistics.

D4 — Use paired tests and multiplicity corrections consistently.

The overly broad statement about all paired tests is corrected. The n=5 minimum 0.0625 applies to the specified exact two-sided sign-flip or no-zero/tie signed-rank setup, not every paired test or a parametric t test. We recompute paired sign-flip comparisons and Holm corrections across 12 metrics from two datasets, separately for n=5 and the four non-screening seeds. The smallest adjusted p is 0.75. No confirmatory significance is claimed.

Remaining work: Additional independent splits/seeds can improve precision, but existing seeds should not be selectively extended until significant. Method/metric families must be locked before confirmatory evaluation.

## Resources

Comments D8, Q5. Status PARTLY CLOSED STATIC ACCOUNTING. Manuscript 3.2 and technical supplement.

D8 — Report training/inference cost and memory per added class.
Q5 — Runtime, head/centroid memory and inference latency?

The head has 2dr+2d+2 parameters: at r=8, d=512 gives 9,218 (36,872 float32 bytes); d=256 gives 4,610 (18,440 bytes). This excludes the shared encoder, optimizer, activations, buffers and router. Actual centroid storage is the sum of each centroid array nbytes; a fit cap of 3,000 rows is not 3,000 retained centroids. No unmeasured latency is supplied.

Remaining work: Profile data loading/preprocessing, training, pure inference and attribution separately on allocated hardware, with batch size, warmup, distributions and memory peaks. Existing scheduler elapsed time is not pure training time; an operation counter is not a latency benchmark.

## Provenance

Comments Q10. Status CLOSED REANALYSIS. Manuscript 4.5 and versioned ETG supplement.

Q10 — Per-row explainer/version metadata and performance-stratified consensus?

All 450 method-specific ledger rows are retrospectively enriched with explainer identity, implementation and policy hashes, training-result binding, target scope and display labels. Five original protocols were downloaded read-only from protected storage and checked against their saved SHA-256 registry entries. They verify Python 3.11.9, SHAP 0.51.0, NumPy 2.2.6, PyTorch 2.6.0+cu118 and all five probe-manifest hashes. Custom FA/GxI implementations are versioned by source hash. Public metadata capsules omit personal source paths and explicitly remain derived attestations. Original results are unchanged. Agreement is reported for 150 common records by descriptive recall bins with denominators.

Remaining work: The metadata and requested descriptive consensus analysis are complete. Consensus is not a calibrated acceptance rule or proof of utility; those remain separate experimental questions.

## Rank

Comments D10, Q12. Status NEW EXPERIMENT REQUIRED. Manuscript 2.1 and 5.5.

D10 — Compare dynamic versus fixed-rank low-rank allocation.
Q12 — Adaptive rank allocation for minority or drift-prone classes?

The current rank-8 head is fixed, not PEARL. Dynamic rank is existing prior art and cannot be claimed as a new contribution merely by implementation. With frozen old heads and encoder, new-class competition can reduce old recall without old-head parameter forgetting.

Remaining work: Only after a matched fixed-rank baseline, compare fixed ranks and adaptive allocation under an equal total parameter budget. Use current increment fitting evidence, not test-class performance, to assign rank. Report cumulative memory and per-class support.

## Experimental sequence

1. Retain the completed ReplayIDS archived-probe reconstruction; verify each subsequent attribution/perturbation path and capture missing provenance before interpreting its outputs.
2. Match baseline capacity, loss, exposure, total memory and prospective sampling; establish the architecture comparison.
3. Evaluate calibrated fusion and the expansion diagnostic with gradient-independent faithfulness and repeated backgrounds.
4. Evaluate a fixed-budget repair controller only against equal-budget simple triggers; retain governance only if it adds benefit.
5. Extend promising, preregistered configurations to independent splits and external baseline/open-set/rank/transfer tests. Report negative results. Do not select a headline method by repeatedly inspecting the same test set.
