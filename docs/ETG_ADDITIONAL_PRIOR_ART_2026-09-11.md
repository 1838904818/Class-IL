# Additional prior-art boundary for explanation governance

Checked 11 September 2026. This supplements the frozen pilot materials; it does
not revise a bound candidate, change its thresholds, or report new experiments.

| Primary source | Relevant overlap | Consequence for this study |
|---|---|---|
| Haug et al., *Change Detection for Local Explainability in Evolving Data Streams*, CIKM 2022, [paper record](https://arxiv.org/abs/2209.02764) | CDLEEDS identifies obsolete local attributions after model/data changes and supports targeted explanation recalculation. | Detecting stale explanations and triggering their refresh are established ideas. An audit ledger or refresh instruction alone is not our novelty. |
| Masood et al., *Explanation audits reveal silent failures of machine learning models under distribution shift*, 24 August 2026, [journal article](https://link.springer.com/article/10.1007/s44163-026-02012-6) | Its controlled static-tabular study distinguishes pointwise warning signals from batch/group attribution audits. Confidence outperforms explanation volatility for pointwise error ranking, while audits reveal complementary feature-reliance information. Stable explanations can accompany incorrect shortcut reliance. | Explanation variation must not be equated with prediction risk, and stability must not certify correctness. A useful result must show downstream action value, not merely a visible attribution change. These published findings are not OFRA results. |

The scope comparisons above are our interpretation of the cited sources, not an
exhaustive priority search or proof that the proposed rule is absent elsewhere.

## Preserve the narrow falsifiable claim

The current candidate asks whether an empirical method-noise gate improves
selection of one fixed correction relative to a performance-only selector. The
noise subtraction, SHAP, calibration action and ledger cannot individually carry
a first-of-its-kind claim. A positive single-transition result remains preliminary.

Keep three distinctions explicit:

- Repeated attribution randomness is not the same as instability under input
  perturbations or change caused by expanding the competing class set.
- A selective action that improves held-out outcomes is different from a truthful
  explanation, analyst benefit, or a population-level safety certificate.
- A less active gate has not shown efficiency until explanation overhead and
  unsuccessful fits are included.

The frozen five-arm descriptive pilot is unchanged. If it is positive, a later
separately registered confirmation should also examine a low-cost confidence or
decision-margin selector with the same action and budget. Do not silently add,
tune or select that comparator after examining the current evaluation outcome.
OFRA joint scores are not multiclass probabilities: a confidence comparator
would need a stated score definition, not an undocumented softmax interpretation.

No source cited here demonstrates utility of the proposed OFRA/ETG intervention.
