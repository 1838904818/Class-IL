# DGC day 1: the across-class premise triggers the registered veto

Date: 2026-10-14 (Asia/Kuala_Lumpur). Scope: Work Order 20.2, existing results and literature only. **Stop before DGC implementation and Section 20.3 training.** This rejects the proposed route under its registered premise test, not every possible drift-informed mechanism. It informs RQ2/RO2; RQ1/RO1 and RQ3/RO3 remain open.

| Check | Status at stopping | Evidence |
|---|---|---|
| A: Joint headroom | Not verified; not a pass or a measured veto | Located historical Joint ID results, but no comparable Joint S1/KDDTest-21 shifted recall or retention. Joint seeds `{42,0,1,2,3}` differ from current `{1,2,3,4,42}`; historical files lack source/split bindings. The Joint-minus-RCL shifted headroom cannot be calculated. |
| B: across-class premise | **Veto** | Mean within-group Spearman **0.075000**; seed-cluster 95% CI **[-0.063393, 0.206306]**. Both mean ≤ 0.1 and CI containing zero trigger 20.2. |
| C: novelty | Incomplete; not a pass | Search stopped when B triggered the veto. No complete same-mechanism match had been verified, but coverage was insufficient to exclude one. |

## What was calculated

Input: [registered Section 17 results](results/exp_h/section17/registered/exp_h_result.json), S1 only. Within each `(method, seed, transition)`, correlate old-pair PI drift ranks with shifted recall **before minus after** ranks; equally average identifiable group correlations. This is not Section 19's pooled/class-adjusted statistic or direct validation of the proposed online JS signal.

The inherited pre-transition recall floor is 0.2: 279/440 class records remain. Of 120 groups, 112 are identifiable (271 records); eight singleton groups are null, not zero. Group sizes 1/2/3/4 occur in 8/68/41/3 groups; two-class correlations are necessarily ±1. Bootstrap resamples five whole seed clusters: 2,000 valid draws, seed 1705. Five clusters limit precision. No-floor sensitivity also fails: mean **0.042396**, CI **[-0.098472, 0.172439]**. Independent average-rank arithmetic reproduced both analyses to rounding.

## Evidence boundaries at stopping

A: historical Joint sources: `results/{CIC-IDS-2017,NSL-KDD}_results_5seed.json`. No new Joint evaluation, leakage audit, deduplication or training was performed.

C: [RCL](https://arxiv.org/html/2607.02020v1), §4.4/Appendix A.2, already has teacher-based **sample-level confidence/intervention-sensitivity gates**; no explicit class-level drift allocation does not mean no adaptive weighting. [Janetzky et al.](https://proceedings.mlr.press/v306/janetzky26a.html) was partly checked. [Rainbow Memory](https://arxiv.org/html/2103.17230), [EWC](https://arxiv.org/html/1612.00796) and [MAS](https://arxiv.org/html/1711.09601) were inspected; mechanism/variant coverage remains **incomplete**.

## Reproduction and decision

Analysis command: `python -I ncd/results/dgc/day1/analyze_s1_within_transition.py`; exit 0. Analysis base commit: `7f9b16d1055cd7e0de13c699dcd340e1516e595b`; experimental source remains `a8b90215a0ad369a11a5cbb07b2176e9d821f5f3`. Input SHA256: `4b2724122bc5b3ddcfd1a2b6e02601b73c763c14a88c05d40ca3e9d6e41b0e7b`.

[Analysis JSON](results/dgc/day1/s1_within_transition.json), [group values](results/dgc/day1/s1_within_transition_groups.csv), [class values](results/dgc/day1/s1_within_transition_classes.csv), [bootstrap draws](results/dgc/day1/s1_within_transition_bootstrap.csv), and [log](results/dgc/day1/s1_within_transition.log) retain the numeric evidence.

**Next step:** report the veto and await a new direction; no implementation, matrix, or day-5 efficacy decision. The work order lists eight methods but says seven in its delivery line; this remains unresolved and is not silently reduced. Historical CRR/Section 17–19 results are unchanged; UNSW/Malaya deduplicated reruns remain pending.
