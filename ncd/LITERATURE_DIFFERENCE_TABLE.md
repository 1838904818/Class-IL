# NCD closest-work difference table

Last primary-source check: 2026-10-12. Scope: explanation change, attribution
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

## IRD novelty veto check (Work Order Section 11.3, 2026-10-12)

Section 11.3 asks a narrower, mechanism-level question than the NCD review
above: has prior continual-learning work already trained a new checkpoint to
preserve an old checkpoint's response to feature/evidence interventions?  The
answer is **yes**.  The primary-source search therefore returns **STOP** for IRD
as currently formulated; implementation and HPC experiments were not started.

| Primary source | Distillation / intervention target | Continual learning? | Relation to the proposed IRD |
|---|---|---|---|
| Li and Hoiem, **Learning without Forgetting**, ECCV 2016, [paper](https://arxiv.org/abs/1606.09282) | Old-model output distribution on observed inputs | Yes | Establishes output-level checkpoint distillation, but does not preserve responses to intervened inputs. |
| Dhar et al., **Learning without Memorizing (LwM)**, CVPR 2019, [CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Dhar_Learning_Without_Memorizing_CVPR_2019_paper.html) | Output distillation plus attention-map distillation | Yes | Preserves an explanation/attention representation, not a clean-versus-intervened response contrast. |
| Ebrahimi et al., **Remembering for the Right Reasons**, 2021, [DOI](https://doi.org/10.1002/ail2.44), [OpenReview](https://openreview.net/forum?id=tHgJoMfy6nI) | Gradient/saliency explanations from an earlier model | Yes | Closest explanation-matching baseline, but the target is explainer output rather than a model-behaviour contrast under intervention. |
| Hu et al., **Distilling Causal Effect of Data in Class-Incremental Learning**, CVPR 2021, [CVF paper](https://openaccess.thecvf.com/content/CVPR2021/html/Hu_Distilling_Causal_Effect_of_Data_in_Class-Incremental_Learning_CVPR_2021_paper.html), [arXiv](https://arxiv.org/abs/2103.01737) | Causal/colliding effect of old and new data, designed to emulate replay without storing old data | Yes | A material naming and positioning risk, but its estimand is a data-stream colliding effect rather than the per-feature clean/intervened score difference proposed for IRD. |
| Bender et al., **Towards Fixing Clever-Hans Predictors with Counterfactual Knowledge Distillation**, ICCV Workshops 2023, [CVF](https://openaccess.thecvf.com/content/ICCV2023W/CVAMD/html/Bender_Towards_Fixing_Clever-Hans_Predictors_with_Counterfactual_Knowledge_Distillation_ICCVW_2023_paper.html) | Teacher feedback on counterfactual examples to remove confounder reliance | No | Shows that counterfactual KD can change shortcut reliance, but it is not a sequential checkpoint-retention objective. |
| Beyer et al., **Knowledge Distillation: A Good Teacher Is Patient and Consistent**, CVPR 2022, [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Beyer_Knowledge_Distillation_A_Good_Teacher_Is_Patient_and_Consistent_CVPR_2022_paper.html) | Teacher/student logits on identical augmented or mixup inputs | No | Directly motivates the required KD-perturbed ablation: it matches absolute outputs on perturbed inputs, not the perturbation-induced response difference. |
| Janetzky et al., **Continual Learning of Domain-Invariant Representations**, ICML 2026, [PMLR](https://proceedings.mlr.press/v306/janetzky26a.html) | Sequential invariance alignment combined with replay | Yes | Directly addresses shortcut-prone continual learning through invariant representations, but does not distil a per-sample intervention-response profile. |
| Bae, **Cross-corruption distillation for robust instruction following under text corruptions**, 2026, [DOI](https://doi.org/10.4218/etrij.2026-0159) | Teacher/student logit differences induced by identical input corruptions | No | Mechanistically close to IRD's difference-of-differences target, but outside continual learning. It narrows any defensible claim to the continual-retention setting rather than response-difference distillation itself. |
| Chen, Xiao, and Tang, **Hidden Forgetting in Continual Multimodal Learning: When Accuracy Survives but Grounding Fails**, arXiv:2607.02020, [abstract](https://arxiv.org/abs/2607.02020), [full text](https://arxiv.org/html/2607.02020) | The frozen previous checkpoint and current model are probed with counterfactual evidence-channel interventions. Clean/intervened output and answer-loss differences form reliance profiles; a Jensen--Shannon loss preserves the old profile during continual adaptation (their Eqs. 3--7). | **Yes** | **Novelty veto.** This already operationalizes the same core idea as IRD: preserve the previous model's dependency structure by distilling teacher/student behavioural changes under matched interventions. |

### Exact overlap and residual differences

The proposed IRD target is

\[
e^f_c(x,j)=s^f_c(x)-s^f_c(x^{(j)}),\qquad
\mathcal{L}_{\mathrm{IRD}}=\mathbb{E}(e^{\mathrm{new}}-e^{\mathrm{old}})^2.
\]

RCL defines counterfactual channel sensitivity from the difference between a
model's behaviour on the original input and on a matched channel-suppressed
input, normalizes those sensitivities into a reliance profile, and minimizes
the divergence between the previous and current checkpoints' profiles during
continual learning.  Thus both mechanisms:

1. freeze the previous checkpoint as teacher;
2. apply matched interventions to teacher and student inputs;
3. derive a dependency/reliance target from clean-versus-intervened behaviour;
4. regularize the continual update to preserve that target while learning new
   data.

IRD still differs in domain and implementation: tabular feature-wise marginal
replacement rather than multimodal channel suppression, class margin or
log-probability rather than token-distribution/answer-loss sensitivity,
replay-old-class samples rather than a replay-free current-stage objective,
and normalized squared effects rather than Jensen--Shannon divergence between
reliance profiles.  Those are potentially useful adaptations and may justify a
domain-specific study, but they do not support the current claim that
interventional reliance preservation itself is a new continual-learning
mechanism.  Under the explicit veto rule in Work Order Section 11.3, the
decision is therefore **STOP**.  No IRD implementation, hyperparameter search,
or 6-method HPC submission should proceed unless the project lead replaces or
materially narrows the proposed mechanism and its novelty claim.

This is a bounded primary-source search, not a legal or exhaustive novelty
opinion.  The stop decision follows the project's stricter preregistered rule.
