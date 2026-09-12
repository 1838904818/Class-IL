# Decision-level contribution and falsification map

12 September 2026. Research design clarification, not a positive experimental result.
No threshold, arm, source binding or uploaded candidate is changed by this note.

## Candidate in one sentence

Test whether a method-noise-adjusted explanation gate selects a more useful
single old-head calibration action than performance evidence alone, while
holding the classifier, action, label roles and intervention limit fixed.

The relevant output is a decision (which class to calibrate, or no action),
not an attractive explanation plot. The intended human intellectual contribution
is the research question, the explicit noise-versus-change distinction, the
decision rule and the controlled falsification design. Assistance and authorship
must be described truthfully; these choices do not by themselves establish
priority, efficacy or publication readiness.

## Exact current rule

For each old class c, h_c is the change in empirical error on trigger rows.
Signed attribution vectors are L1-normalized. Half-L1 distance compares eight
paired old/new estimates for each row. The between-checkpoint distance is their
median. The noise reference is the larger within-checkpoint 90th percentile of
four disjoint repeat-pair distances. Let e_c be the median across class rows of
between-change minus noise.

Select the largest h_c among classes with h_c > 0 and e_c > 0.05, breaking ties
by class ID; otherwise abstain. Invalid attribution makes the comparison
unavailable rather than producing a zero or an affirmative no-action decision.
This is the already registered noise-aware rule, not a new estimator.

The same fitter then attempts one positive bounded affine correction a*z_c+b
to the selected old head's binary logit margin. Native router and other heads
stay unchanged. A separate empirical acceptance sample accepts or rolls back
the proposal. Only locked subsequent evaluation measures its consequences.
The fitter's numerical recipe still requires its own reviewed execution binding.

The eight repeats, quantile 0.9 and threshold 0.05 are project design choices,
not literature constants. This small empirical noise reference is not a
confidence interval, statistical certificate or correction for systematic
SHAP bias. Growing rival classes change the explained margin; that legitimate
competition is not automatically forgetting. Silent drift without observed
error harm remains audit-only.

## Where incremental value can actually appear

Every nonrandom active selector uses the same harm ranking; explanation only
filters eligibility. With two old classes and one attempt, the noise-aware
selector can retain the top harmful class, skip it for the other eligible class,
or abstain. If no class is harmful, it cannot initiate an intervention.

For successful deterministic fits, identical target, inputs and numerical
settings imply identical fitted parameters. Identical acceptance inputs and
rules then imply the same effective scoring function. Consequently the two
arms have identical predictions on any common evaluation population: improvement
over audit-only would credit the common calibration, not explanation selection.
This is a logical diagnostic, not a new learning theorem or experiment.

Same target alone is insufficient evidence: confirm semantic fit parameters,
native scorer binding and acceptance decision. Walltime fields can differ without
changing an action; a timeout or failed fit must not be mislabeled an algorithmic
advantage. Different actions are only an opportunity to demonstrate value, not
evidence that the explanation-selected action is better.

| Observed outcome | Permissible interpretation |
|---|---|
| Same effective correction and outputs | No observed predictive selection benefit; count extra attribution cost. |
| Different action, lower held-out old-class error and no protected-class harm | Preliminary incremental decision-value evidence; replication required. |
| Noise-aware outperforms raw-drift under the same action contract | Evidence supporting the noise-gate mechanism, conditional on this pilot. |
| Abstention avoids a harmful correction | Possible protective value; compare acceptance-only controls and total cost. |
| No accepted action, unavailable inputs or resource failure | Inconclusive utility evidence, not successful governance or proof of uselessness. |
| Better numbers only after changing thresholds or evaluated labels | Exploratory tuning, not confirmation of the frozen method. |

Report noise-aware minus error-only old-class balanced error, proposed and
effective outcomes, raw counts/per-class metrics, selections, failures and full
attribution overhead. The 1,024 estimates are repeated measurements for one
transition, not 1,024 independent interventions or training seeds. The pilot
has only two old-class candidates and cannot establish multi-transition or
five-seed generalization. Reused calibration-pool labels do not become fresh
confirmation merely because rows were assigned new roles.


## Why explanation change does not guarantee repairability

R1 changes the target score to sigmoid(a*z_c+b) + 0.5*r_c, where z_c
is its binary head-logit margin and r_c is the frozen native Router term.
All competing scores remain fixed. Even with unconstrained finite affine
parameters, the mathematical target score stays between 0.5*r_c and
1 + 0.5*r_c. In this real-valued equation, a strongest competing score above
the latter bound cannot be overtaken by this action. Registered finite bounds
can narrow the reachable range further. Implementation checks must calculate
the bound with the native float32 operations and exact argmax tie convention;
the real-valued argument alone is not a bit-level numerical proof.

Thus a large reliable explanation change is not a certificate that this
particular action can help. This bound is a diagnostic consequence of the
existing score equation, not a newly proposed algorithm or proof about the
observed data. No reachability gate is added to the frozen five-arm pilot.

## Stronger prior-art boundary

| Primary source | Established overlap | Remaining question here |
|---|---|---|
| Goldwasser and Hooker, UAI 2025, [Statistical Significance of Feature Importance Rankings](https://proceedings.mlr.press/v286/goldwasser25a.html) | Sampling uncertainty and reliable feature ranking, including StableSHAP. | Does an empirical noise gate add downstream action-selection value? We do not inherit their ranking guarantees. |
| Haug et al., CIKM 2022, [CDLEEDS](https://arxiv.org/abs/2209.02764) | Detect outdated local attributions and target their recalculation. | Calibrating a selected classifier head differs from refreshing explanations, but both need to be distinguished explicitly. |
| Lamaakal et al., CAO Workshop at ICLR 2026, [Drift2Act](https://arxiv.org/abs/2603.08578) | Budgeted drift response, risk certificates and corrective actions. | Test the marginal value of explanation noise adjustment, not claim the first drift-to-action system. |
| Li et al., Scientific Reports 2026, [computable audit framework](https://www.nature.com/articles/s41598-026-46652-1) | Risk-tiered explanations, an accountability ledger, bounded responses and holdout checks; controlled synthetic stress tests. | A ledger plus SHAP plus recalibration is already insufficient as a distinct contribution. |

Sources were checked on 12 September 2026. Scope distinctions are our
interpretation, not proof that no comparable method exists. The prior
[CLEX / BiC comparison](ETG_NOVELTY_AND_FALSIFICATION_2026-09-11.md) still applies:
neither explanation drift in Class-IL nor ordinary calibration is new.

## Execution boundary and next decision

The first 64-estimate native SHAP chunk is uploaded and submission-reviewed,
but not submitted. No paired explanation-selection or calibration outcome is
available. Finish and verify the registered extraction, freeze/review the common
R1 recipe, and run the five-arm locked comparison before strengthening claims.
Do not alter the approved extraction to seek a favorable hypothesis.

If the hypothesis fails, retain the result and identify whether the reason is
identical choices, an unavailable noise signal, ineffective calibration or
generalization failure. A simple separately registered decision-margin selector
may then serve as a lower-cost comparator. Do not silently replace the failed
method or call such a baseline a new algorithm.
