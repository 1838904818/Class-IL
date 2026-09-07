# Candidate contributions and falsification plan

7 September 2026. **Prospective implementation, not validated novelty or a performance result.**

## What is already known

| Primary source | Relevant prior contribution | Consequence for this project |
|---|---|---|
| [StaR-MoE, Guo et al., 2026](https://arxiv.org/html/2605.17571v1) | Expansion-induced routing drift despite frozen historical experts; sensitivity-aware historical routing alignment and capacity regularization. | Neither identifying expansion drift nor keeping old experts frozen is our novelty. A comparison must distinguish its learned MoE routing alignment from independent-head/distance-score fusion. |
| [GCR, 2026](https://arxiv.org/html/2601.01856v1) | Geometry-consistent routing for task-agnostic continual anomaly detection. | Geometric routing stability already has close prior art; the domain and scorer differences must be explicit. |
| [BiC, Wu et al., 2019](https://arxiv.org/abs/1905.13260) | A small linear correction layer compensates old/new classifier bias, fitted with held-out validation. | A two-parameter score repair is a control/action primitive, not a new algorithm by itself. |
| [Weight Aligning, Zhao et al., 2020](https://arxiv.org/abs/1911.07053) | Weight-norm alignment addresses old/new classifier bias alongside distillation. | Calibration needs simple established alternatives, not only an unrepaired comparator. |
| [Drift2Act, 2026](https://arxiv.org/html/2603.08578v1) | Budgeted drift-to-action selection with online risk certificates, including recalibration and other interventions. The manuscript identifies an ICLR workshop, not the main conference. | A drift alert followed by a budgeted action is not a new governance concept. Our fixed-horizon group bounds are not its anytime guarantee. |
| [CLEX, Cossu et al., 2024, DOI 10.1016/j.neucom.2024.127960](https://pages.di.unipi.it/bacciu/publications/all/) | The authors' publication abstract describes explanation drift in class-incremental learning and the influence of replay and model updating. | Explanation drift after legitimate updates is not a newly discovered phenomenon. This entry is grounded in the author-hosted abstract, not a claimed full-text review. |

This is a targeted collision check, not an exhaustive novelty search or evidence of publication acceptance. These sources must be distinguished from experiments actually executed here.

## Candidate A: class-set-independent fusion scale

The implemented score is an independent binary-head probability plus a standardized negative nearest-centroid distance. Standardizing across all seen classes gives the old-pair margin

```text
p_i - p_j + lambda * (r_i - r_j) / (std_seen(r) + epsilon).
```

Appending a nonwinning class can change this margin by changing its denominator. The proposed control freezes the reference bank to training-only Task 0 and uses its per-input scale for every current class. Under unchanged preprocessing, encoder, old heads and reference routers, old-pair margins are invariant to class-axis expansion.

The narrow potential contribution is **useful, input-dependent but class-set-independent fusion scaling for this independent-head/centroid architecture**. It is not the general discovery of routing drift. The conditional invariance proof is insufficient: a global constant or unnormalized distance has the same invariance. The reference method must outperform these cheaper controls on held-out accuracy/retention/security utility at acceptable cost.

Required controls: head only; raw distance fusion; development-fixed scalar; all-seen normalization with the same floor formula; the historical std-plus-epsilon formula; fixed-reference scale. Separate genuine new-rival competition from old-pair reordering. Keep Task-0 order, scaling floor and lambda selection off official tests. Record degenerate reference scales and new-class recall; do not hide failure to learn new classes behind low forgetting.

**Rejection:** no advantage over the simple controls; unacceptable new-class/security regression; excessive Task-0 dependence; no meaningful incidence of the proposed mechanism; or historical score reconstruction fails. Rejecting this candidate is a legitimate outcome.

## Candidate B: repairability-aware intervention selection

The second candidate distinguishes three effects on an existing class's margin: model-state change, normalization-domain change and new-rival competition. This is an ordered counterfactual decomposition, not a causal attribution invariant to path order. Explanation change alone does not prove that the available action can fix an error.

The executable R1 action modifies only one existing binary head's logit difference by a positive affine map, leaving other heads and router scores frozen. A pointwise reachability upper bound first asks whether any allowed R1 parameters could beat that row's strongest rival. If not, this action cannot repair that row. Passing is only necessary: the same two parameters must work jointly across all rows, and fitting may still fail.

The potential contribution is **matching an intervention's limited capabilities to the mechanism of score change under explicit label and compute budgets**. Neither two-parameter calibration nor budgeted drift response is new. The central comparison is the SAME R1 fitter under random, periodic, error-only, disagreement-only, explanation/decomposition-aware and reachability-aware selectors. Include audit-only, zero-weight and no-attribution controls. Record unsuccessful and rejected interventions as costs.

The action is selected using trigger information, fitted on separate rows, accepted using separate held-out groups, then evaluated on a sealed later partition. Subsequent labels are unavailable until all decisions are immutable. Fixed-horizon simultaneous paired-group bounds require independent groups and are group-weighted; they are not arbitrary-reuse or anytime certificates. Insufficient group support must produce an inconclusive/rejected gate, not a relaxed statistical rule.

**Rejection:** no improvement over error-only or reachability-only controls after budgets and costs are matched; inconsistent support across seeds; harms to old/new class performance or false-positive behavior; or attribution extraction overhead exceeds its incremental benefit. If attribution does not help, remove it from the intervention selector. If R1 does not help, keep ETG only as an explicitly scoped audit instrument or remove it from the main contribution.

## The three work lines and their roles

| Line | What its executable establishes | What it cannot establish alone |
|---|---|---|
| Fair comparison | Identical initialization, sampled row/target streams and optimizer-step accounting for CE versus conditional focal heads; raw confusion and per-class retention. | Full OFRA versus shared-head architecture fairness; isolated per-method process memory; new architecture novelty. |
| Prospective data | Immutable increment-wise train-only sampling, Task-0-frozen policy, fit/calibration separation and conservation of row identities. | That balancing improves accuracy; causal independence of unknown capture groups; equivalent historical and prospective protocols. |
| Operational repair | Real R1 optimization, immutable selections, budget spending/rollback and later held-out evaluation. | Verified historical GPU fidelity, actual explanation extraction, live network governance or utility gains. |

Fairness and leakage prevention are necessary experimental controls, not claimed research innovations. They enable the two candidate hypotheses to be tested without confounding objective, data availability and action budget.

## Minimal staged evidence decision

1. Recover native scorer fidelity with the previously fixed numerical contract. Infrastructure failure is not a score mismatch and cannot be reported as a passed test.
2. Profile the real train-only sampler and bounded native checkpoint exporter. Verify raw affinities, feature/encoder/reference identities and all output hashes. Historical standardized z scores cannot be inverted into raw affinities.
3. Run a single development pilot of matched CE/focal and the six fusion controls. Report it as one seed, not a final finding. Do not expand an evidently unhelpful candidate merely to obtain more samples.
4. Execute the same-action intervention controls on disjoint, adequate groups with sealed later evaluation. If a method's group requirements cannot be met, report that limitation rather than silently substituting individual rows for independent groups.
5. Freeze the chosen protocol before confirmatory seeds and data sets. Publish all arms, failures, label/compute/storage costs, uncertainty and negative results. Refresh manuscript claims only after immutable results have been verified.

All numeric pilot settings in example fixtures are software-test choices, not optimized or scientifically preregistered constants. No result-driven novelty claim is authorized by this document.
