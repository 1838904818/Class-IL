# OFRA and ETG validation priorities

7 September 2026. This is a technical validation plan, not a new experimental result. The submitted v3.2 manuscript and all original benchmark outputs remain unchanged.

## Supported contribution and current evidence

The defensible contribution is a source-bound evaluation of class-incremental prediction, explanation changes and an offline, non-suppressing ETG ledger. Established components are attributed to their sources; the present experiments do not isolate an architecture-only causal gain or demonstrate operational safety.

- [Malaya and paired replay evidence](BALANCED_REPLAY_FIVE_SEED_RESULTS.md): OFRA improves overall accuracy and reduces forgetting, but the four new seeds show only a 0.23 pp Macro-F1 gain over Balanced Replay50.
- [ReplayIDS last-epoch primary](REPLAYIDS_PRIMARY_SCORING_DIAGNOSTIC.md): 85.51% accuracy and 54.40% Macro-F1 versus 70.65% and 36.88% for replay; benign FPR falls from 35.61% to 13.10%, while attack recall falls from 99.78% to 85.73%. Guarded checkpoint selection is secondary.
- [Malaya attribution audit](../results/malaya-attribution-etg-five-seed-v9/summary.json): five seeds on one split; all-method agreement averages 50% for admission, 54% for state and 35% for drift conclusions. Consensus does not establish correctness.
- [Recovered five-seed EG sensitivity](RECOVERED_SENSITIVITY_2026-09-07.md): 64 settings per seed, 320 verified rows and finite cosine/Kendall metrics for all 100 transitions. This corrects the earlier seed-1-only inventory; full three-method sequential governance replay remains separate.

## Sequenced work and completion criteria

| Priority | Work | Completion criterion | Current status |
|---|---|---|---|
| 1 | Historical GPU score fidelity | Match all 20 checkpoint score comparisons under the fixed tolerance, zero argmax differences, and a recorded environment; preserve failures | Local verifier prepared, execution not verified |
| 2 | Attribution and policy sensitivity | Hash-verified arrays; fixed k, threshold and recall-guard grid; full ETG state replay and denominators for all seeds and methods | Five-seed EG drift grid recovered; full three-method state grid open |
| 2 | Ledger provenance | Each derivative entry links method/version, target, probe/background, budget and policy hashes; originals preserved | Companion bindings and aggregate consensus available |
| 3 | Independent faithfulness | Bounded perturbation-based comparison and valid-feature tests after numerical validation; convergence and masking assumptions reported | Feature Ablation comparison exists; broader checks missing |
| 3 | Fusion calibration | One locked candidate fit only on increment-available calibration data, with predeclared costs/constraints and a fixed test protocol | No candidate improvement established |
| 4 | Matched training and capacity | Within-dataset paired design with explicit loss, exposure, total memory and parameter accounting | Current replay comparison only partly matched |
| 4 | One tabular continual baseline | Verified reference implementation and explicitly documented adaptation to the same data/task protocol | Reference identification completed, benchmark not run |
| 4 | Runtime and memory | Measured prediction latency/throughput, training cost and per-class memory; offline attribution cost reported separately | Not a complete deployment profile |
| Extension | Open-set, alternative split, adaptive rank | Separately registered protocol, no test-set selection or oracle inference inputs | No new execution claimed |
| Main-claim gate | Intervention utility | Better outcomes at equal labeling/compute budget than random, periodic, error-only and disagreement-only triggers using the same repair action; human studies require applicable review | Benefit unvalidated; demote or remove governance claim if utility fails |

The GPU verifier and its execution gates are described in [the prepared candidate](REPLAYIDS_GPU_FIDELITY_CANDIDATE.md). A local plan or successful unit test is not evidence of a submitted or successful GPU job. The [CPU audit](REPLAYIDS_CANONICAL_FIDELITY.md) remains a failed cross-environment gate, not a failure demonstrated in the original GPU environment.

## Protocol distinctions

1. A five-seed audit on one split measures training-seed variation, not cross-capture or cross-domain generalization. Repeated test rows and class transitions are not independent seed replicates.
2. Task-0-only scaling is distinct from prospective sampling. ReplayIDS D2's offline Benign cap uses future-class counts; a prospective alternative must use only information available at each increment.
3. The recovered five-seed EG grid uses recall change strictly greater than -0.05 at the primary setting, not a symmetric five-pp stability band. Preserve original operators and tie handling. Mean seed rate (73.98%) is distinct from pooled 61/82 (74.39%).
4. Changing top-k changes rationale mass and potentially ledger admissions; changing Jaccard thresholds can change recertification. Recounting drift events alone is not a new ETG ledger analysis.
5. An Integrated Gradients completeness failure warrants checking numerical convergence, target implementation and path behavior. Piecewise min/max alone does not prove every gradient attribution invalid. Perturbation agreement is also not a causal guarantee.
6. Exact two-sided sign-flip or Wilcoxon signed-rank tests with five nonzero pairs have minimum p=0.0625; the bound does not apply to a parametric paired t-test. Report effects, uncertainty, assumptions and all endpoints. Keep the screening seed 42 separate in sensitivity reporting.
7. New unlabeled inputs do not autonomously create semantic heads. Open-set rejection and subsequent labeled class updates are separate experimental stages.

## Related methods and attribution of novelty

The [revised research plan](RESEARCH_PLAN_2026-09-07.md) defines the ETG keep/remove gate and a research-only expansion-aware diagnostic. No new algorithmic priority or real-data intervention benefit is asserted.

CLEX already studies explanation drift in class-incremental learning: [Cossu et al., Neurocomputing 2024](https://doi.org/10.1016/j.neucom.2024.127960). Drift2Act already links drift monitoring to budgeted actions and risk assessment: [Lamaakal et al., 2026](https://arxiv.org/abs/2603.08578). Explanation drift itself, or adding an intervention after monitoring, is not sufficient novelty.

TRIL3 is a tabular pseudorehearsal method based on incremental prototypes and a neural decision forest. Its task-free stream differs from the present class-incremental protocol, so a comparison requires a documented adaptation: [Garcia-Santaclara et al., 2025](https://doi.org/10.1016/j.engappai.2025.110908).

PEARL studies dynamic LoRA rank allocation and rehearsal-free continual learning, evaluated on vision architectures; it is not an already matched tabular benchmark: [Bhat et al., TMLR 2026](https://arxiv.org/abs/2505.11998).

GCR studies geometry-consistent routing for industrial image anomaly detection, not a directly comparable NIDS experiment: [Chae et al., 2026](https://arxiv.org/abs/2601.01856). For attribution axioms, see [Sundararajan et al., ICML 2017](https://proceedings.mlr.press/v70/sundararajan17a.html).

The study-specific systems claim concerns the research question, source-bound score/attribution protocol and measured method-dependent governance outcomes. Component reuse alone is not evidence of algorithmic priority. Individual author contribution declarations require the authors' confirmation of their actual conceptual and experimental roles. No claim of unaided invention or confirmed publication readiness is made.
