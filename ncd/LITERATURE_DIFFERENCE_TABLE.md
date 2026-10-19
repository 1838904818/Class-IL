# NCD closest-work difference table

## Section 26.8 核实补充（2026-10-19）

以下六篇均已正式发表；技术细节来自出版社全文或作者公开全文，发表信息另核对出版社/作者页面。范围包含多类 Class-IL、二元 domain-incremental 与持续新颖性检测，不能全部视为同一多类协议。“未单列”仅指所读全文的主要结果指标/表格，不是因全文不可得而猜测。

| 论文（正式出处；范围；来源定位） | 回放内存如何跨类分配 | 是否报告良性 FPR | 主要报告指标 |
|---|---|---|---|
| **CBRS-NIDS — On Handling Class Imbalance in Continual Learning based NIDS**（[AIMLSystems 2021](https://newslab.iith.ac.in/publications/on-handling-class-imbalance-in-continual-learning-based-network-intrusion-detection-systems/)；攻击类型顺序到达、二元输出）。[作者全文](https://newslab.iith.ac.in/publications/on-handling-class-imbalance-in-continual-learning-based-network-intrusion-detection-systems/download/)，§3.1.2.1、§4.2、§5，pp.3–6、Tables4–8。 | **两种平衡口径**：WACB 将全部攻击合并后与良性平衡；IACB 按原始类别平衡。不是唯一的全类别均分规则。 | 主结果未单列。 | Accuracy、Precision、Recall、F1；任务顺序/内存大小敏感性。 |
| **ECBRS/PAPA — Augmented Memory Replay-based Continual Learning Approaches for NIDS**（NeurIPS **2023**；二元 domain-incremental）。[官方全文](https://proceedings.neurips.cc/paper_files/paper/2023/file/3755a02b1035fbadd5f93a022170e46f-Paper-Conference.pdf)，§3.1、§4，pp.4、6–8、Tables3–4。 | **全局频数驱动**替换及少数类保护阈值，非固定 M/K；原始类别用于内存管理。 | 正文主结果未单列固定工作点 FPR；ROC-AUC 不等于 FPR。 | ROC-AUC、良性/攻击 PR-AUC、训练时间。 |
| **SPIDER — A semi-supervised continual learning-based NIDS**（[IEEE INFOCOM 2024](https://newslab.iith.ac.in/publications/spider-a-semi-supervised-continual-learning-based-network-intrusion-detection-system/)；二元 domain-incremental）。[作者全文](https://newslab.iith.ac.in/media/publications/pdfs/SPIDER-Suresh.pdf)，pp.2、4、6–8、Algorithm1、§V–VI。 | **其他**：主方法仅保留上一任务随机无标签样本，每任务更新；过去知识另由 GPM 子空间保持，非全历史逐类等额回放。 | 正文主结果未单列。 | ROC-AUC、良性/攻击 PR-AUC、时间、标签比例敏感性。 |
| **SSF — Continual Learning with Strategic Selection and Forgetting for NIDS**（IEEE INFOCOM **2025**；二元持续学习/漂移；[作者发表记录 p.2/P27](https://zhihanjiang.com/papers/CV_zhihanjiang_public.pdf)）。[作者全文](https://arxiv.org/html/2412.16264)，§III-B/C、Algorithms1–2、§IV、TablesI–II。 | **分布代表性/漂移驱动**：固定总缓冲，选择代表性新样本、移除不再代表分布的旧样本，非逐类等额。 | 作者全文主结果未单列。 | Accuracy、Precision、Recall、F1；标签预算与消融。 |
| **CND-IDS — Continual Novelty Detection for IDS**（[ACM/IEEE DAC 2025](https://ieeexplore.ieee.org/document/11132541/)；无监督二元新颖性检测）。[作者全文](https://arxiv.org/html/2502.14094)，§III-B/C、§IV、Figures3–5、TablesII–III。 | **无旧攻击样本回放**：旧模型潜表示正则；另保留洁净 Normal 参考集拟合 PCA，不等于完全不存数据。 | 作者全文主结果未单列。 | F1、平均经验性能、前向/后向迁移、PR-AUC、时间。 |
| **ACIL — Anomaly-aware class-incremental learning framework for NIDS**（[IJMLC 2026，17:104](https://link.springer.com/article/10.1007/s13042-025-02840-5)；二元及多类）。[作者上传全文](https://www.researchgate.net/publication/400834463_Anomaly-aware_class-incremental_learning_ACIL_framework_for_network_intrusion_detection_systems)，§4.1/4.2、§5.2.1、Table10/12–14，pp.10–11、15–19。 | **多策略框架**：reservoir、herding、CBRS 等，非单一配额；多数实验总缓冲 K=1000，另做大小消融。 | **部分/间接**：定义 FPR/TNR；主表未单列 FPR 数值，Table10 有 Normal 类准确率，不能记为完全不报告良性表现。 | Accuracy、F1、Forgetting、BWT、效率、逐类准确率。 |

**判定：原预设动机句不获支持，待负责人改写；可检验问题仍为配额与 FPR/检测率权衡。** 六篇不能支持“普遍各类均分”；CBRS-NIDS 已有良性/合并攻击二组平衡，不能将 Fixed-50 本身当作新机制。五篇未单列 FPR 只是本次定向样本观察，不是全领域比例或“无人关注误报”的结论；ACIL 的部分/间接报告必须保留。FARQ 的误差驱动配额及其增量价值仍需匹配对照检验，不由本表确立创新，亦不关闭 RQ1/RQ3。

核实边界与纠正：Two-Stage、AdvAs-IIDS、ExBCIL 全文继续标为 **不可得**，未重新搜索，也未猜测配额/FPR；不纳入这六篇可读样本。ACIL 正式发表为 **2026-02-16**，不是 DOI 中的 2025。原 `master_list` 所列 Few-Shot NIDS 的 [DOI 10.3390/fi16100383](https://doi.org/10.3390/fi16100383) 对应其他论文的更正，不能充当该 NIDS 引用，已从本表排除；本次未改动该清单。SSF/CND-IDS 的作者版本用于技术核实，正式发表身份另有上列来源，不将纯预印本计作已发表。

---

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

## CRR novelty veto check (Work Order Section 12.4, 2026-10-12)

**Decision: CONTINUE to the registered ground-truth experiments.**  In the
bounded primary-source search below, no method was found that jointly (i) runs
in class-incremental learning, (ii) preserves a teacher checkpoint's
feature-intervention or counterfactual effect specifically on **old-class vs
old-class margins**, and (iii) deliberately leaves every old-vs-new margin
unconstrained.  This is the exact Section 12.4 veto condition; it is not a
claim that CRR is already innovative or effective.  The latter still requires
all four Section 12.7 gates.

| Primary source | What is preserved / changed | Why it does not trigger the Section 12.4 veto |
|---|---|---|
| Chen, Xiao, and Tang, **Hidden Forgetting in Continual Multimodal Learning** (RCL), 2026, [full text](https://arxiv.org/html/2607.02020) | A frozen teacher and current model are compared under counterfactual evidence-channel suppression; the Jensen--Shannon loss preserves the resulting sample-level reliance profile (Eqs. 3--7). | RCL is a global sample/channel reliance constraint for multimodal task streams. The paper does not define old-class pairs, old-vs-new margins, or a class-pair-restricted variant that frees new-class contrasts. It is the required closest baseline, not a veto match. |
| Lesort, **Spurious Features in Continual Learning**, CoLLAs 2023, [PMLR](https://proceedings.mlr.press/v208/lesort23a.html) | Diagnoses local spurious features: features can look predictive in the partial data available at one stage but cease to be valid as the stream expands. | It motivates why old dependencies may need revision, but proposes neither a reliance-distillation loss nor an old-pair restriction. |
| Zhang, Chu, and Li, **Causally Sufficient and Necessary Feature Expansion for Class-Incremental Learning**, 2026, [full text](https://arxiv.org/html/2603.09145) | Uses intra-task and inter-task counterfactual features to regularize causal completeness and separability in an expansion model. | Counterfactuals train current-task representations against feature collision; the objective does not distil teacher/student feature-intervention effects on old-old class margins while exempting old-new margins. |
| Ashok, Joseph, and Balasubramanian, **Class-Incremental Learning with Cross-Space Clustering and Controlled Transfer**, ECCV 2022, [paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870104.pdf) | CSC preserves old feature-space class structure; CT estimates similarities between an incoming class and prior classes to promote positive forward transfer and reduce negative backward transfer. | It explicitly models class relationships and stability/plasticity, but transfers feature-space similarity rather than teacher/student intervention effects. CT primarily conditions old-new placement rather than preserving only old-old causal contrasts. |
| Kang, Park, and Han, **Class-Incremental Learning by Knowledge Distillation with Adaptive Feature Consolidation**, CVPR 2022, [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Kang_Class-Incremental_Learning_by_Knowledge_Distillation_With_Adaptive_Feature_Consolidation_CVPR_2022_paper.html) | Restricts changes to important representation maps while allowing less important maps to adapt. | Its unit of protection is a feature-map representation weighted by loss sensitivity, not a class-pair-specific interventional margin. |
| Douillard et al., **PODNet**, ECCV 2020, [paper](https://arxiv.org/abs/2004.13513) | Distils pooled intermediate outputs and uses a local-similarity classifier. | Representation distillation is not intervention-response distillation and is not restricted to old-old margin pairs. |
| Dong et al., **Few-Shot Class-Incremental Learning via Relation Knowledge Distillation**, AAAI 2021, [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/16213) | Maintains an exemplar relation graph and distils structural relations among exemplars/classes. | It is the closest pair/relationship-KD family, but its relation target is observational feature geometry, not the change in a class margin under a matched feature intervention. It therefore motivates OldPair-KD, the registered non-interventional ablation. |

### Residual novelty claim and fail-closed boundary

CRR's remaining technical distinction is narrow: for an old replay example of
class c, it preserves the teacher's intervention effect on margins m(c,k) only
for previously learned k, while applying no corresponding reliance constraint
to any margin involving a newly arriving class.  Ordinary output KD,
OldPair-KD, global intervention preservation, and RCL-style reliance profiles
each remove one of those elements and are therefore registered controls.

The search outcome authorizes implementation and testing only.  If the
two-benchmark experiment does not show both shortcut resistance and necessary
old-vs-new plasticity under Section 12.7, CRR must be rejected even though the
literature veto was not triggered.  This is a bounded search, not an exhaustive
or legal novelty opinion.
