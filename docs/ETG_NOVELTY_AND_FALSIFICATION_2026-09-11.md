# Candidate contribution and falsification boundary

11 September 2026. **Research hypothesis, not a verified novelty claim.**

## One-sentence candidate

Test whether method-noise-adjusted explanation change adds actionable information
to performance-triggered correction of a fixed class-incremental classifier,
under matched intervention limits and explicit attribution-cost accounting.

The proposed contribution is the **decision mechanism and its controlled
evidence**, not assembling OFRA, SHAP and a ledger, renaming bias correction,
increasing a model, or detecting explanation drift for the first time.
No first-of-its-kind or Q1-readiness claim is justified by this preparation.

## Closest checked sources

| Source | Established contribution / overlap | What this candidate must add |
|---|---|---|
| Lundberg and Lee, SHAP (2017), [primary paper](https://arxiv.org/abs/1705.07874) | Feature attribution framework. | A tested use of attribution in a downstream decision; no claim of inventing SHAP. |
| Cossu et al., CLEX (2024), [university record](https://arpi.unipi.it/handle/11568/1245048), [paper](https://arpi.unipi.it/retrieve/a826458a-71e5-4f88-ba61-812379f59fd6/drifting.pdf) | Explanation change is already studied in class-incremental learning. | Measured incremental action-selection value, not another drift statistic alone. |
| Wu et al., BiC (2019), [primary paper](https://arxiv.org/abs/1905.13260) | Simple learned bias correction for incremental classification. | Keep calibration fixed across selectors; R1 alone cannot carry the novelty claim and is not a full BiC reproduction. |
| Lamaakal et al., Drift2Act (2026), [primary paper](https://arxiv.org/abs/2603.08578) | Budgeted drift response with risk certificates and multiple actions. CAO Workshop at ICLR 2026. | Isolate the incremental value of method-conditioned explanations in a fixed Class-IL action; no inherited online risk guarantees. |

Sources checked on 11 September 2026. The comparison of intended scope is our
inference from these papers, not an assertion that an exhaustive literature
search established absence of a similar rule. Simple noise subtraction is
not sufficient by itself to establish algorithmic novelty. The
[PermutationExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html)
supports the proposed model-agnostic implementation option, not its utility here.

## Evidence needed before strengthening the claim

1. Real native-scoring evidence: the explained score is the decision actually
   compared, including routing, class expansion and batch context.
2. Matched-action contrast: noise-aware versus error-only; neither changes the
   model or fitting recipe to win.
3. Mechanism ablation: raw-drift versus noise-aware; record whether suppressing
   method-internal variability actually improves action selection.
4. Low-information and cost controls: audit-only and random-harmful; compare
   actual spending and failed interventions, not only nominal budgets.
5. Locked held-out outcomes and later replication. An early single transition
   can reject or refine a candidate; it cannot prove general utility.

The [separate early-checkpoint protocol](../experimental/etg_exploratory_v1/PROTOCOL.md)
now defines the descriptive pilot permitted by the input-support audit.
Its row split cannot recover missing capture independence. The original strict
protocol and the current manuscript's result claims remain unchanged.

## Interpretation discipline

- If selectors yield the same correction and output, attribution has not shown
  incremental utility.
- If only calibration helps, credit calibration; do not credit ETG.
- If gating only saves fits, count all explanation costs before claiming economy.
- If the method fails or only works after test-driven tuning, retain the negative
  result and do not market a new algorithm.
- Even a useful action gate does not show that an explanation is correct, safe to
  reuse, or beneficial to human analysts. Those are different research endpoints.
