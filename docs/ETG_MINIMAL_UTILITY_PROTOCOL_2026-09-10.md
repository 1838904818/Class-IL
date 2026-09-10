# Explanation governance: minimal incremental-utility study

10 September 2026. **Prospective design and implementation audit; no new experimental result.**

## Question and scope

Does method-conditioned explanation change help decide when to attempt a bounded correction, beyond performance evidence alone, under the same label and intervention limits?

OFRA supplies the continually updated classifier; attribution describes its decision basis; ETG records evidence and the disposition of a proposed action. The historical ETG remains an offline, non-suppressing audit ledger. This study would test an additional offline action-selection function, not certify explanations as true, establish production safety, or replace the classifier architecture. Published RQ/RO wording and historical results are not changed by this protocol.

Keep the dataset, preprocessing, checkpoints, class order, LoRA heads, DP-Means router, fusion coefficient and scoring call boundaries fixed. Do not introduce EIAR, a new head, a new training loss or a larger encoder in this contrast.

## Three primary views, one common action

| Display name | Existing implementation arm | What changes |
|---|---|---|
| Audit only | `audit-only` | Record evidence; do not attempt a correction. |
| Performance trigger | `error-only` | Rank eligible old classes by the lower bound on their error increase. |
| Performance plus explanation | `expansion-w0` | Use the same harm ranking, but require explanation change to exceed method-internal noise. No decomposition weighting. |

The primary attribution contrast is **`expansion-w0` versus `error-only`**, not an expansion-weighted arm versus error-only. Both compute some shared score-path diagnostics in the current implementation, but neither uses a decomposition weight or reachability weight in its priority. This isolates the additional explanation eligibility gate. The historical code label `expansion-w0` is retained for reproducibility; it is not a new method name.

The existing runner requires all ten registered arms. Leave that executable contract intact: random, periodic, disagreement, weighted-decomposition and reachability controls remain supplementary controls. A three-row presentation does not authorize silently dropping registered arms or reducing multiplicity correction. A smaller executable would need a separately reviewed protocol version.

Every active arm gets the same single correction opportunity, fitting pool, acceptance pool, optimizer recipe and resource ceilings. The existing R1 operation changes only one old head's binary logit difference to `a * logit_difference + b`, with positive bounded `a`; it leaves the encoder, other heads and raw router scores unchanged. Identity is an allowed outcome. This is score calibration, not retraining or repairing the explanation itself.

Audit-only uses no correction budget. Gated arms may abstain. Report actual spending and unused capacity, not merely identical caps. Count attribution generation, failed fits, rejected proposals and rollback costs. An apparent benefit from spending less must be identified as a resource-saving result, not equal-spend accuracy improvement.

## Sequence and safeguards

1. **Check available evidence.** Bind one dataset and a development checkpoint transition by chronology and coverage, not by observed test gain. Keep Malaya application labels distinct from CIC-IDS-2017 attack labels. Seed/checkpoint selection is not finalized here.
2. **Verify the scoring target and attribution.** Use actual old/new native head logits and raw router affinities. Reproduce the relevant scorer on the exact partition and call boundaries. Check attribution against that target, including perturbations and repeated draws; numerical forward agreement alone is insufficient.
3. **Freeze the decision rule.** Use the same explainer at old/new checkpoints. Compare between-checkpoint change with within-checkpoint repeated-draw noise. Missing, degenerate or unverified explanations make that arm unavailable; they do not become zero drift. Do not select the method giving the preferred ETG decision. The primary method and all numerical settings remain unregistered until feasibility is established.
4. **Separate four data roles.** Trigger selects the class; fit estimates `(a,b)`; acceptance decides whether to keep it; subsequent evaluation measures consequences after all decisions lock. Decision labels must already be available at the increment. No official test rows may trigger, fit or accept an action. Do not recycle already trained-on rows as an independent acceptance holdout.
5. **Accept or roll back.** Require the registered target benefit and old/new-class non-harm bounds; include missed-attack/FPR constraints only where labels have those meanings. Inadequate independent groups or wide bounds mean abstention/inconclusive evidence, not a relaxed gate. Keep original scores and raw alerts alongside every proposed research output. No production alerts are suppressed and no human review is claimed.
6. **Evaluate once, then decide whether to expand.** A small development pilot can reject an unhelpful idea. It cannot establish efficacy across seeds or datasets. Reused, previously examined official tests must remain retrospective diagnostics, not fresh confirmation.

This trigger requires observed performance harm as well as explanation change. It therefore **does not act on silent explanation drift with no measured error increase**. Those cases remain audit-only. Nor does this pilot answer whether an old explanation can safely be shown to a person: that needs a separate, externally justified explanation-quality or user-task endpoint.

## Outcomes and rejection rules

The primary comparison is the paired difference in effective-policy old-class error between the two active primary arms on the locked subsequent partition. The effect unit is an independently supported group/study, not every correlated class-transition row. Predeclare the practical effect threshold and uncertainty family before accessing those outcomes. Check all-class and new-class harm alongside any apparent old-class benefit.

Report raw confusion matrices and per-class support, recall (true-class accuracy), precision and F1; overall accuracy, Macro-F1, balanced accuracy; selected targets, abstentions, accepted/rejected corrections, actual labels/steps/time/memory, and method-dependent decision agreement. For attack-labelled data also report attack recall and Benign FPR. A single before/after correction gives an old-class recall change, **not a complete forgetting trajectory**; longitudinal forgetting requires aligned checkpoint histories and an explicitly fixed definition.

Reject an explanation-governance utility claim if the explanation gate does not improve the same-action controls, harms protected classes, costs more than its incremental benefit, or relies on changing the explainer after inspecting results. Identical selected targets and effective outputs provide no evidence of added predictive utility. If insufficient support prevents acceptance, report **inconclusive**, not evidence that explanation governance is useless. Method-sensitive results must be presented as method-conditioned rather than universal certification.

## Readiness audit as of this revision

The [scoped readiness receipt](ETG_MINIMAL_UTILITY_READINESS_2026-09-10.json)
records source hashes, seven synthetic selector tests and remaining inputs.
It is not experiment approval or an empirical utility result.

| Component | Verified scope | Remaining requirement |
|---|---|---|
| Selector, R1 fit, rollback and ledger | Source implementation exists in [repair controls](../experimental/parallel_execution/repair_controls/README.md). | Real-data efficacy has not been established. |
| Primary contrast | The two existing priorities match; only the explanation gate differs. | Real repeated attribution inputs and their faithful-target evidence. |
| Historical Malaya explanation study | Three-method, five-seed aggregate exists in the [technical documentation](../TECHNICAL_DOCUMENTATION.md). | It is retrospective audit evidence, not this study's four-partition input bundle. Do not transfer its states across datasets. |
| Prospective ReplayIDS data | [P/O preparation](../experimental/parallel_execution/prospective_sampling/real_pair_20260908/PAIR_RESULT.md) has source-bound row identities. | Current source contract uses `row_identity_proxy`; capture independence is not established. |
| Real policy and execution | No new policy, dataset/transition binding or resource request is approved here. | Resolve lineage, grouping, target and holdout support; freeze numeric settings; obtain independent execution review. |

The immediate next gate is an input/provenance and group-support audit, **not another model-training job**. Do not manufacture capture IDs from row IDs, treat a metadata flag as proof, import software-demo thresholds, or use a new group ID to reuse held-out labels. Group-count feasibility must be checked against the actual simultaneous bound before paying for attribution extraction. If the existing dataset cannot support it, stop and report the limitation before proposing a changed statistical study.

## Contribution boundary and sources

The candidate contribution is narrowly the incremental value of a noise-aware, method-conditioned explanation gate for a fixed continual-classifier action. That is a hypothesis, not an established new algorithm or publication guarantee.

- Explanation drift in class-incremental learning is prior work: Cossu et al., *Drifting explanations in continual learning* (CLEX, 2024), [author publication and abstract](https://pages.di.unipi.it/bacciu/publications/all/), DOI 10.1016/j.neucom.2024.127960.
- Simple bias calibration is prior work: Wu et al., *Large Scale Incremental Learning* (BiC, 2019), [paper](https://arxiv.org/abs/1905.13260). The R1 action is not claimed to reproduce BiC's complete protocol.
- Budgeted drift-to-action governance is prior work: Lamaakal et al., *Drift-to-Action Controllers* (2026), [paper](https://arxiv.org/abs/2603.08578), CAO Workshop at ICLR 2026. The present fixed-horizon offline study does not inherit its online guarantees.

Sources checked 10 September 2026. The targeted comparison is not an exhaustive novelty search. The design must demonstrate added value beyond these established ideas before its contribution is strengthened.
