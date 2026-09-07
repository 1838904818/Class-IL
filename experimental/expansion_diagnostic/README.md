# Expansion-aware score diagnosis: reference prototype

Status: research-only, synthetic tests only. No trained controller, real-data improvement, new SHAP method or algorithmic-priority claim is established.

For an old target class, the same ordered rows are evaluated under four margin definitions:

| Target | Checkpoint | Router normalization classes | Rival classes |
|---|---|---|---|
| A | Old | Old | Old |
| B | New | Old | Old |
| C | New | All seen | Old |
| D | New | All seen | All seen |

B-A measures change in supplied old-class scores; C-B measures the router-normalization-domain effect; D-C measures the enlarged rival set. These sum to D-A. The ordering is specified, not a unique causal decomposition. D-C cannot be positive for a fixed old-class target when the rival set grows; the other effects may have either sign.

Inputs are independently scored two-logit family-head positive probabilities and raw router affinities, aligned by class ID and probe row. Do not supply a multiclass softmax normalized across a changing class axis: that would confound the proposed components. The caller must verify the provenance and semantics of the supplied arrays. The reference validates shapes, class IDs, ordered row IDs, finiteness and parameters, but cannot prove provenance.

This float64 reference deliberately does not reproduce legacy float32 GPU arithmetic. It uses population standard deviation plus epsilon. Real-data use requires source-bound score reconstruction and a declared numerical-fidelity gate; the reference must not replace historical prediction results.

Only synthetic controls have been run: identity, pure expansion, rival monotonicity, class-axis permutation, state-only changes, invalid/misaligned inputs and degenerate normalization. Ten tests pass. Passing them is not evidence that the method is novel, useful or faithful on real traffic.

```text
python -m unittest discover -s experimental/expansion_diagnostic -p test_expansion_diagnostic.py
```

Requires NumPy. No model, checkpoint, network service or GPU is started.

A future utility test would compare the proposed diagnosis with ordinary drift triggers at the same repair/label budget and with the same repair action. Neither this reference nor the telescoping identity implements that controller. See [the research plan](../../docs/RESEARCH_PLAN_2026-09-07.md).
