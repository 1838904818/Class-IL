# Research plan: falsifiable contribution and evidence-gated intervention

Effective 7 September 2026. This plan prioritizes scientific contribution and fair evaluation over headline-accuracy optimization.

Develop and test a distinct, well-motivated class-incremental traffic method with reproducible evidence of its added value. First reconcile every external-review criticism with actual protected results and the submitted manuscript. Separate missing experiments from missing presentation. Improve the scientific contribution, not merely the headline accuracy, model size, number of runs or terminology. Publication readiness requires evidence; no journal outcome is guaranteed.

## Acceptance conditions

1. Each review issue has an exact evidence pointer, verified scope and one status: completed and presented; completed but missing from presentation; partly complete; absent; or proposed extension. Never infer completion from code or a draft script.
2. The prediction contribution survives matched-data, matched-capacity, loss/exposure and total-memory accounting. Report attack recall, FPR, Macro-F1, per-class support and retention together, including failures and seeds 1-4 separately from the screening seed.
3. The proposed mechanism is distinguished from established work through a documented novelty comparison and a falsifiable hypothesis. A new acronym, code implementation, combined standard modules, or optimization on previously examined test labels does not establish novelty.
4. A candidate algorithm must have a formal definition, executable reference, edge-case tests, ablations that remove the proposed mechanism, independent data/capture validation and a frozen evaluation protocol. Passing synthetic tests is not real-data evidence.
5. ETG is retained as a main contribution only if its decisions measurably improve an actual research/deployment-management action at the same labeling/compute budget. It must beat an alarm-only ledger and simple random, periodic, prediction-error-only and disagreement-only triggers, using the same repair mechanism and locked budget.
6. If ETG adds no independently measurable utility, remove governance from the title/abstract and main claims; preserve useful auditing and negative findings in supplementary material. A new controller must be versioned separately from the historical non-suppressing ETG ledger.
7. No raw alert suppression, arbitrary semantic head creation, test-label fitting, future-class calibration, or silent replacement of historical scores. Intervention effects are measured only on later held-out evaluation; data collection involving analysts requires applicable approval.

## Immediate findings

- Complete v9 seed packages contain 64 Expected-Gradients drift settings for EACH of seeds 1, 2, 3, 4 and 42, not only the public seed-1 grid. All 320 rows have now been recomputed against saved transitions and locally downloaded checksum registries; no remote re-verification or new attribution was performed.
- Saved transition rows include finite cosine similarity and Kendall tau-b for all 100 class transitions. Their coverage is now exported; this is not a new robustness experiment.
- Full three-method governance-state sensitivity remains a different question: changed top-k changes deletion mass and admissions, not just overlap counts.
- The v9 attribution scope explicitly describes the deterministic CPU/true-class-batch reconstruction, not numerical equivalence to archived GPU scores. Forward equality and backward derivative faithfulness are separate gates; a detached exact-forward/surrogate-gradient correction does not itself prove faithful gradients.

## Candidate mechanism for investigation

Working description: expansion-aware explanation diagnosis with evidence-gated repair. This is a hypothesis and reference prototype, not an established new algorithm or a verified novel contribution.

For an old class at adjacent checkpoints, compare FOUR registered score functions on exactly the same rows:

A. old checkpoint, old class set for normalization and rivals;
B. new checkpoint, old class set for normalization and rivals;
C. new checkpoint, full class set for normalization but old rival set;
D. new checkpoint, full class set for normalization and rivals, the deployed score.

The change from A to D separates along a chosen path into B minus A (state effect), C minus B (normalization-domain effect), and D minus C (new-rival effect). The sum telescopes exactly for scores. This ordering is a diagnostic convention, not a unique causal decomposition. The counterfactual B/C functions must never replace D in headline prediction evaluation.

The research hypothesis is that distinguishing these mechanisms and within-method attribution noise prevents unnecessary repairs triggered by legitimate class expansion, while directing a fixed intervention budget toward genuinely harmful old-class changes. A difference in explanations is not automatically damage; even a pure expansion effect can harm performance and must not be ignored.

The first implementation only computes the score decomposition and passes synthetic controls. Subsequent attribution must explain each registered function with consistent backgrounds/masks, rather than subtracting top-k sets or treating the telescoping identity as an attribution-faithfulness guarantee.

An intervention candidate may recalibrate fusion or update a bounded family head using increment-available training data. Candidate fitting and acceptance require separate held-out partitions or an explicitly justified sequential protocol; no repeated uncorrected testing until a candidate passes. Acceptance must satisfy predeclared missed-attack/FPR and retention constraints on sufficient support. Malaya has application labels and cannot use attack/FPR constraints.

## Closest known work and novelty risks

- CLEX already studies explanation drift in class-incremental learning and shows that similar predictive accuracy does not imply similar explanations. That general observation is not our novelty: Cossu et al., Neurocomputing 2024, https://doi.org/10.1016/j.neucom.2024.127960 .
- Drift2Act already describes budgeted drift-to-action control, delayed-label risk assessment and interventions. Adding an action after drift monitoring is not a new algorithm by itself: Lamaakal et al., 2026, https://arxiv.org/abs/2603.08578 .
- GCR already separates geometry-based routing from within-expert scoring in industrial image anomaly detection: https://arxiv.org/abs/2601.01856 .
- PEARL already studies adaptive LoRA ranks: https://arxiv.org/abs/2505.11998 . TRIL3 already studies prototype pseudorehearsal for tabular streams: https://doi.org/10.1016/j.engappai.2025.110908 .

The specific expansion/normalization/rival decomposition and its usefulness for budgeted repair need a deeper literature check and empirical disproof attempts. Algorithmic priority is not established by the reference implementation or by the telescoping identity.

## Execution boundary

Begin with local evidence recovery and the diagnostic reference. Keep the submitted v3.2 PDF immutable; prepare a later manuscript revision only after the evidence inventory is reconciled. Reference code and synthetic tests do not establish real-data utility. Any subsequent compute campaign must follow institutional policy and the separately reviewed execution protocol.

## Evidence update for v3.3

The [technical supplement](TECHNICAL_AUDIT_SUPPLEMENT_2026-09-07.md) now records the completed retrospective ledger, statistical and provenance audits. See [experiment gates](EXPERIMENT_GATES_2026-09-07.md) for remaining work. The current manuscript removes operational governance from its main contribution; a repair controller remains unvalidated.
