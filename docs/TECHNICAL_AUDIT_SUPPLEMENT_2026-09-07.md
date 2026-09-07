# OFRA prediction and explanation audit supplement

7 September 2026. This supplement accompanies manuscript v3.3. It documents existing evidence and retrospective analyses, not newly trained models or a validated governance controller. The previously submitted v3.2 manuscript and original result files remain unchanged.

## Current system and scope

Each dataset has its own preprocessing contract, class stream, model and results. Records are not pooled across datasets. The strict prediction campaigns use Task-0-only numerical statistics and categorical vocabularies. Later unknown categorical values follow the registered all-zero field encoding. This does not make ReplayIDS D2 prospective: its separate normal-traffic cap uses the largest attack-class fitting count across the offline dataset, including future increments. A prospective cap comparison still needs training.

The shared encoder is trained at Task 0 and then frozen. Seen classes have independent output-space low-rank binary heads and DP-Means centroid banks. All applicable heads and router class scores contribute to the registered decision arms; routing is not an instruction to invoke only one LoRA module before scoring. The joint score combines a family-head score and a standardised router score with fixed weight 0.5, then selects the largest score. The router fit cap of 3,000 limits fitting examples per class; it is not a network width, replay-memory size or promise of 3,000 centroids.

An old class can lose recall when new rivals are introduced even if its encoder/head weights remain frozen. Consequently, output forgetting is not automatically evidence that old parameters were overwritten. A head-only/router-only/joint comparison on the same checkpoint isolates scoring choices but does not isolate loss, training exposure or capacity between separately trained systems.

## Explanation target and faithfulness boundary

The primary Malaya attribution campaign reconstructs checkpoints on CPU and evaluates fixed true-class probe batches. The registered target is a class margin against the strongest competing class. This is not necessarily the argmax class explanation for every probe. Attribution probes and the full-test class-recall denominator are distinct.

The implementation uses exact NumPy forward values with a detached correction and a differentiable backward surrogate. The same-batch forward fidelity gate passing is not proof that its gradients equal derivatives of the NumPy scorer. Finite gradients and SHAP library execution likewise do not establish faithfulness. The failed Integrated Gradients diagnostic is retained, while avoiding the incorrect assertion that every piecewise score necessarily violates IG assumptions. Direct finite-difference and gradient-independent checks remain necessary.

The rationale deletion statistic is another important distinction. It averages the decrease in the target class softmax probability when selected input features are replaced by their frozen means. Its target is not numerically identical to the attributed class margin. Feature replacement can create unrealistic correlated-feature combinations. The random-null deletion comparison is a limited perturbation diagnostic, not a causal intervention in a real network or a complete explanation-faithfulness guarantee.

## Recovered stability and policy sensitivity

Every seed has 64 saved EG drift settings: k in {5,10,15,20}, Jaccard threshold in {0.5,0.6,0.7,0.8}, and allowed recall drop in {0,0.02,0.05,0.10}. All 320 settings were recomputed from saved transitions. Eligibility is delta recall strictly greater than minus the allowed drop; drift is Jaccard strictly less than the threshold. At an allowed drop of zero, unchanged recall is not eligible under the original strict inequality. The guard is one-sided, not a symmetric performance-stability band.

At k=15 and allowed drop 0.05, EG mean seed drift rates are 3.69%, 48.20%, 73.98% and 96.40% for the four Jaccard thresholds. The registered primary has 61 events among 82 eligible transitions. Mean seed rate is 73.98%; the pooled ratio is 74.39%. Neither is a fraction of network flows. The denominator consists of repeated class transitions on one split, not 82 independent incidents.

All 100 saved transitions contain cosine similarity and Kendall tau-b. Means across seed means are 0.8525 and 0.7686. These values describe the broader saved importance vectors/rankings, whereas top-k Jaccard only measures selected membership. A high rank correlation and unstable membership around a top-k boundary can coexist. These are complementary diagnostics on the same data, not independent confirmations.

At fixed top-15, the original state-machine function was extracted from the hash-verified historical analyzer and replayed over all three methods, five seeds and sixteen Jaccard/recall-guard settings. All 450 primary method-specific states and actions exactly matched the stored records. The 240 replays reuse measured deletion masses and admissions. Changing k requires new deletion evaluations; it cannot be inferred from an overlap-only sweep. No thresholds were selected for deployment from these retrospective results.

## What ETG records and what it cannot certify

ETG remains the historical Explanation-Trust Governance ledger name. Governance is not a headline contribution of v3.3. The outputs record explanation-rule admission, drift flags, refusal and a one-checkpoint recertification process; human-review action strings are simulated, not evidence that a person reviewed the case. Prediction alerts are not suppressed.

Historical CERTIFIED_STABLE means the explanation rule passed, not that class recall, attack detection or safety passed. A class rejected at initial admission stays UNEXPLAINABLE under the historical terminal-state rule. The audit preserves these facts and supplies separate display labels EXPLANATION_RULE_PASSED, EXPLANATION_RULE_FAILED and EXPLANATION_CHANGE_FLAGGED. It does not rewrite the original states.

Across 150 common checkpoint-class records, the recall-bin results below are descriptive. Bin edges were fixed in the re-analysis before calculation but were not part of the original experiment. Repeated classes/checkpoints/seeds are dependent. Consensus can mean that every method is wrong or every rule fails.

| Recorded class recall | Records | Admission agreement | State agreement | Any historical certified state |
| --- | --- | --- | --- | --- |
| Below 20% | 72 | 34/72 (47.22%) | 42/72 (58.33%) | 27/72 |
| 20% to below 50% | 43 | 25/43 (58.14%) | 19/43 (44.19%) | 23/43 |
| 50% to below 80% | 15 | 6/15 (40.00%) | 5/15 (33.33%) | 10/15 |
| 80% to 100% | 20 | 10/20 (50.00%) | 15/20 (75.00%) | 5/20 |

The 450 enriched rows bind method identity, historical implementation hashes, policy hash, training-result hash, source artifact, target scope and probe/background-manifest hash. Five original protocols were downloaded read-only from protected storage and matched against their protected checksum entries. Their dependencies are Python 3.11.9, NumPy 2.2.6, PyTorch 2.6.0+cu118, SHAP 0.51.0 and SciPy 1.15.3. They also bind all five local probe manifests. Custom FA/GxI methods use implementation hashes as their version identifiers. Public metadata capsules omit personal source paths and are explicitly derived attestations, not byte-identical copies of the complete original protocols. Enrichment leaves historical results unchanged.

## Statistical and resource accounting

Balanced Replay50 comparisons are paired by training seed within each dataset. ReplayIDS uses the registered last-epoch Job 425539 primary, not the older guarded Job 426307 comparison exports. Its source hashes and confusion-derived metrics were rechecked before the combined audit. The re-analysis reports signed differences, pair counts, sample SD, exact two-sided sign-flip p values and Holm correction across twelve metrics spanning Malaya and ReplayIDS. The all-five analysis and the sensitivity analysis excluding screening seed 42 have separate correction families. The smallest adjusted p is 0.75. This is exploratory paired evidence; symmetric exchangeable differences are an assumption, and no significant or causal architectural advantage is asserted.

The minimum attainable p of 0.0625 with five nonzero pairs refers to the specified two-sided exact sign-flip or signed-rank setup, not every paired test. A parametric t test can yield a smaller value under its assumptions. Repeated thresholds, classes and test rows must not be substituted for independent training/split replications.

An output-space family head has two low-rank matrices of total size 2dr, a two-logit classifier with 2d weights and two biases, hence 2dr+2d+2 parameters. With rank 8, d=512 gives 9,218 parameters or 36,872 float32 bytes; d=256 gives 4,610 or 18,440 bytes. These are static head-only counts, not measurements of process memory. Centroid bytes must be counted from each stored array shape and dtype. Include replay rows, encoder parameters, optimizer states, activations, metadata and buffers in total memory.

No pure inference latency is inferred from Slurm elapsed time or parameter count. A valid profile separates preprocessing, loading, training, pure prediction, attribution and ledger processing; records batch size, warmup and hardware; and reports memory peaks and latency distributions. Container/code hashes do not prove GPU/CPU or batch-partition numerical equivalence.

## Closest prior work and remaining mechanism hypothesis

The following primary sources delimit what cannot be claimed as a new invention here. They motivate comparisons; their reported scores are not directly comparable across different domains, data and protocols.

| Prior work | Relevant mechanism | Consequence for this project |
| --- | --- | --- |
| CLEX — Cossu et al. 2024 | Explanation drift in class-incremental learning | Accuracy/explanation divergence is not our new observation |
| Drift2Act — Lamaakal et al. 2026 | Budgeted drift-to-action control with risk assessment | Merely attaching a repair action to drift is not new |
| TRIL3 — Garcia-Santaclara et al. | Prototype pseudorehearsal for tabular streams | Requires an actual protocol-matched baseline, not a citation alone |
| PEARL — Bhat et al. 2026 | Dynamic LoRA rank allocation in vision CL | Adaptive rank itself is established prior art |
| C-LoRA — Zhang et al. 2025 | Learnable routing of low-rank updates | Distinguish this paper from other similarly named CLoRA methods |
| GCR — Chae et al. 2026 | Shared frozen geometry for expert routing | Decision instability under expansion is already recognised |
| Xie et al. 2024 | Task-specific BN and unknown-logit task-ID selection | Domain adaptation to flow features must be explicit |
| VAEMax — Qiu et al. 2024 | Payload features, OpenMax and VAE secondary rejection | Not a drop-in flow-feature baseline |
| ScaIL and Weight Aligning | Correction of old/new classifier bias | New-class score/bias correction is not a novel idea alone |

The proposed diagnostic compares four score functions on identical old-class rows: A uses the old checkpoint and old normalisation/rival sets; B uses the new checkpoint with old sets; C uses the new checkpoint and expanded normalisation set but old rivals; D is the full current deployed function. The chosen path gives state effect B-A, normalisation effect C-B, and rival effect D-C. Their sum equals D-A. The ordering is a convention, not a unique causal allocation; B and C are diagnostic counterfactuals, not alternative headline predictions.

The unverified hypothesis is that this separation, combined with within-method attribution uncertainty and actual class-performance evidence, can spend a fixed repair budget more effectively than simple triggers. Legitimate expansion can still be harmful; a large rival effect is not a reason to ignore lost recall. The current reference implementation only has synthetic arithmetic tests. No real-data decomposition benefit, controller, novel algorithm or human-origin history is claimed.

The planned repair study holds fitting labels, acceptance labels, compute and repair mechanism constant across triggers. Candidate fitting and acceptance use disjoint increment-available partitions; official tests never select a repair. Compare random, periodic, prediction-error-only, disagreement-only, audit-only and the expansion-aware trigger. Unknown semantic labels require annotation before head creation. A controller enters the main contribution only if later held-out performance improves at the same budget; otherwise ETG remains supplementary audit evidence.

## Sources

CLEX: https://doi.org/10.1016/j.neucom.2024.127960

Drift2Act (CAO Workshop at ICLR 2026, not the ICLR main track): https://arxiv.org/abs/2603.08578

TRIL3: https://arxiv.org/abs/2407.09039 and https://doi.org/10.1016/j.engappai.2025.110908

PEARL (TMLR 2026): https://arxiv.org/abs/2505.11998

C-LoRA: https://arxiv.org/abs/2502.17920

GCR: https://arxiv.org/abs/2601.01856

Task-specific BN and OOD: https://arxiv.org/abs/2411.00430

VAEMax: https://arxiv.org/abs/2403.04193

ScaIL: https://arxiv.org/abs/2001.05755

Weight Aligning: https://arxiv.org/abs/1911.07053

Contextualised flow-based NIDS and evaluation pitfalls: https://arxiv.org/abs/2602.05594
