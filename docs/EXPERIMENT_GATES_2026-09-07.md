# Evidence gates after manuscript v3.3

7 September 2026. This is a prospective follow-up plan motivated by already inspected evidence, not preregistration of the historical study. Existing primary results and negative findings remain unchanged. No new experiment is authorised by this document.

## 1. Recover numerical and attribution validity first

Follow the [staged score protocol](REPLAYIDS_SCORE_VALIDATION_PROTOCOL.md). The next gate is a matched historical-environment GPU reconstruction over all five seeds and all twenty ReplayIDS checkpoints. The original monitor embeds in batches of 512, then computes Head and Router on the entire checkpoint probe matrix. Do not replace it with class-only or arbitrary 32-row calls. Preserve the registered tolerances; log bit-exactness separately, raw score hashes, margins and row-specific disagreements. Local canonical CPU checks passed five initial checkpoints and failed fifteen later ones; no historical GPU pass is claimed.

Then verify the actual Malaya explanation target separately: same-batch NumPy forward parity is not derivative faithfulness or parity to archived GPU scores. Compare direct-score finite differences away from ties, gradient-independent perturbations, deletion/insertion against paired random rankings, and repeated background/mask draws. Attribute the class margin and label the softmax-probability deletion statistic separately. Never attribute a silent change in numerical implementation to SHAP instability.

Pass criterion: the registered score is reproduced under its declared environment, and each attribution claim is supported by the corresponding faithful target check. Failed derivatives require changing or demoting the explainer, not renaming its output. This is the dependency gate for new mechanism tests.

## 2. Isolate architecture from training and resources

Run a bounded matched design on one fixed dataset before broad expansion. Keep the split, class order, available labels, encoder width/depth, initialisation seed, checkpoint rule and evaluation support fixed. Account separately for binary versus multiclass loss, focal weighting, negative sampling and old/new row exposure. Match or explicitly bound total bytes and optimisation steps, not just replay count or epochs.

Include shared-model bounded replay, frozen nearest-mean classification and the OFRA candidate. A tabular continual-learning comparator such as TRIL3 requires a faithful adaptation documented against its original protocol. PEARL, C-LoRA and geometry-based routing motivate component controls; a citation is not a reproduced baseline. Capacity studies change one registered factor at a time. Any adaptive-rank arm must be compared against matched fixed-rank, fixed-parameter and allocation controls.

Report all seeds, per-class supports and confusion counts, accuracy, Macro-F1, balanced accuracy, signed forgetting, attack recall and FPR where labels support binary attack semantics. Report train/load/prediction/attribution time and peak memory separately. Do not report attack recall for Malaya application labels. No test-selected winner is promoted.

## 3. Test the specific mechanism, not a new name

The [reference diagnostic](../experimental/expansion_diagnostic/) distinguishes state change, class-normalisation expansion and new-rival competition along a fixed four-score path. It currently passes synthetic arithmetic tests only. The decomposition is order-dependent and is not a proof of causality, attribution faithfulness, novelty or repair utility.

Freeze one candidate trigger after numerical gates and training-only development. It must use within-method uncertainty and performance evidence so that explanation change alone does not imply harmful forgetting. Fit and accept a repair on disjoint increment-available labelled partitions. Use the same bounded repair operation, labels and compute for all triggers: random, periodic, error-only, disagreement-only, audit-only and the candidate. An audit-only arm records flags without repair; its unused budget is disclosed rather than treated as a compute-matched intervention arm. Measure repair benefit and cost on subsequent held-out evaluation, including missed attacks, false positives, old-class retention and new-class learning. Do not suppress raw classification alerts.

An operational ETG contribution is admitted only if its trigger adds measured value beyond these controls at the same active intervention budget. Otherwise retain the historical ledger only as supplementary auditing evidence, or omit it from the main method. The next manuscript must not imply that a simulated human-review action was performed by a person.

## 4. Complete sensitivity, data and open-set controls

- Three-method top-15 state sensitivity already has 240 retrospective policy replays. Changed-k deletion admission requires new masked scoring; grid coverage alone does not answer that question. Separate threshold sensitivity, stochastic explainer variation and split/capture variation.
- D2's adaptive normal cap was chosen from offline fitting counts including later classes. A fully prospective cap must depend only on increment-available training data and requires a new training comparison. Strict Task-0 scaling does not erase this dataset-construction distinction.
- Cross-dataset CIC17/CIC18 transfer needs an explicit compatible feature/unit/label mapping, source-only transforms and a policy for target-only classes. Matching column names alone is insufficient. Keep duplicates and capture-related leakage out of train/test boundaries.
- Open-set controls compare confidence, distance, their combinations, and an adapted reference on the same withheld-class protocol. Fit thresholds only from available training/calibration data. Report OSCR, AUROC/AUPR, rejection sensitivity/FPR, buffer purity/support and post-label old/new-class outcomes. VAEMax uses payload features and is not a drop-in numerical comparator. Unlabelled clusters must not be assigned a semantic attack name or a new supervised head automatically.

## Stopping and reporting

This plan does not promise a publishable or first-of-its-kind algorithm. Real novelty requires a differentiated mechanism and evidence that survives the controls above. Report disconfirming results. Author intellectual decisions must be described truthfully, with research assistance disclosed as required.

No automatic multi-stage submission is authorised. Each executable candidate needs immutable code/data/protocol/environment bindings, measured resource requests, institutional compliance, independent review and fresh exact-command user confirmation. The current boundary is preparation for new experiments; v3.3 is a corrected research draft, not a claim that all scientific criticisms are resolved.
