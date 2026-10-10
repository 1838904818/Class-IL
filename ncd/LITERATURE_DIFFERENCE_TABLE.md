# NCD closest-work difference table

Last primary-source check: 2026-10-10. Scope: explanation change, attribution
uncertainty, statistical inference, continual learning, and drift response. This
is a targeted 10-source comparison, not an exhaustive novelty opinion.

| Primary source | What it measures or controls | Handles attribution-estimation noise? | Continual learning / IDS? | Difference from the present RQ1 study |
|---|---|---|---|---|
| Cossu et al., **CLEX: Drifting explanations in continual learning**, Neurocomputing 2024, [DOI](https://doi.org/10.1016/j.neucom.2024.127960), [author-hosted paper](https://arpi.unipi.it/retrieve/a826458a-71e5-4f88-ba61-812379f59fd6/drifting.pdf) | Explanation drift of continual learners, including alignment to an offline model | No repeated-explainer null, permutation test, or FDR procedure was found in the paper | Class-incremental learning; not NIDS-specific | Establishes that explanation drift in CL is prior work and supplies the required B3 family. It does not validate whether a checkpoint-to-checkpoint attribution change exceeds estimator/probe variability. |
| Ebrahimi et al., **Remembering for the Right Reasons**, Applied AI Letters 2021, [DOI](https://doi.org/10.1002/ail2.44) | Adds explanation consistency to the continual-training objective to reduce forgetting | No; explanations are stored/regularized rather than given a sampling-based uncertainty test | Class-incremental vision; not IDS | A training intervention, not a post-hoc calibrated test of silent explanation drift. |
| Nugraha and Bauschert, **E2D2**, IFIP Networking 2026, [DOI](https://doi.org/10.23919/IFIPNetworking70592.2026.11578962), [reference implementation](https://github.com/DLTeamTUC/E2D2) | Window-level explanation fingerprints and Explanation Drift Intensity for unsupervised NIDS | The available paper metadata and implementation describe thresholded fingerprints, not checkpoint-label permutation with a repeated-explainer null | NIDS; not class-incremental learning | Closest domain-specific detector. It targets streaming data drift around an unsupervised anomaly score, whereas RQ1 targets silent dependency change across supervised Class-IL checkpoints. |
| Lamaakal et al., **Drift2Act**, 2026, [arXiv:2603.08578](https://arxiv.org/abs/2603.08578) | Drift-to-action control with label queries, costs, and online risk certificates | Not an attribution-estimation-noise method | General deployed streams; neither Class-IL nor IDS in its reported benchmarks | Prior art for non-suppressive drift response and safety governance; it does not validate attribution drift as a detector endpoint. |
| Covert and Lee, **Improving KernelSHAP**, AISTATS 2021, [PMLR](https://proceedings.mlr.press/v130/covert21a.html) | KernelSHAP convergence, bias/variance reduction, and uncertainty estimates | Yes, for Monte Carlo Shapley-value estimation | Neither CL nor IDS | Supports the premise that approximate attributions have estimator uncertainty, but does not test change between sequential model versions. |
| Hill et al., **Boundary-Aware Uncertainty for Feature Attribution Explainers**, AISTATS 2024, [PMLR](https://proceedings.mlr.press/v238/hill24a.html) | Instance-wise uncertainty combining decision-boundary complexity and explainer approximation uncertainty | Yes, via GPEC rather than the NCD resampling/permutation design | Neither CL nor IDS | Quantifies whether an individual explanation is uncertain; it is not a drift test and does not use silent class-level transitions. |
| Cifuentes et al., **The Distributional Uncertainty of the SHAP score**, 2024, [arXiv:2401.12731](https://arxiv.org/abs/2401.12731) | Ranges of SHAP scores over uncertainty in the entity-population distribution | Yes, distributional rather than repeated-estimator uncertainty | Neither CL nor IDS | Shows that the reference distribution itself changes SHAP values; it does not test sequential checkpoints or explainer consensus. |
| Hwang et al., **Explanation Multiplicity in SHAP**, 2026, [arXiv:2601.12654](https://arxiv.org/abs/2601.12654) | Repeated-run explanation variability, training/model-selection variability, rank churn, and randomized baselines | Yes; this is the closest noise/multiplicity prior art found | Neither Class-IL nor IDS | Material novelty risk: it already separates explanation-pipeline stochasticity from model multiplicity. The remaining distinction is the registered sequential-checkpoint test with class/transition FDR, cross-explainer consensus, silent-recall conditioning, and intervention-grounded validation. |
| Williamson and Feng, **Efficient nonparametric statistical inference on population feature importance using Shapley values**, ICML 2020, [PMLR](https://proceedings.mlr.press/v119/williamson20a.html) | Population Shapley variable importance with confidence intervals and hypothesis tests | Yes, for population feature-importance estimation | Neither CL nor IDS | Demonstrates formal inference for a global Shapley estimand, but not attribution-vector drift between versions of one continually learned model. |
| Adebayo et al., **Sanity Checks for Saliency Maps**, NeurIPS 2018, [paper](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html) | Model-parameter and data-randomization tests for saliency faithfulness | Not estimator-noise calibration; uses controlled randomization | Neither CL nor IDS | Establishes that an attribution method must respond to genuine model/data changes. It motivates the known-negative/known-positive intervention design in Exp 2 but is not a sequential drift test. |

## Difference finding

The exact conjunction registered here was **not found** in these ten primary
sources: a checkpoint-label permutation test over repeated class-level
attributions, with within-run FDR, cross-explainer consensus, silent-recall
conditioning, and intervention-derived positive/negative truth in task-free
Class-IL NIDS. However, that conjunction is not yet an established substantive
innovation. CLEX covers explanation drift across continual models, and Hwang et
al. already cover repeated-run SHAP multiplicity and randomized null baselines.
The defensible claim therefore depends on Exp 2 showing useful validity beyond
simple repeated averaging (B2), followed by Exp 3 measurement; the current Exp
2 evidence does not show NCD superiority over B2.

No source in this targeted set was found to perform the exact
**cross-checkpoint attribution permutation/bootstrap drift test** above. This
is a bounded search result, not proof that no such paper exists.
