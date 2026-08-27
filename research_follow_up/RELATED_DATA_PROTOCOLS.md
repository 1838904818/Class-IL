# Related NIDS continual-learning data protocols

## Short conclusion

The very high numbers in related work are not directly comparable with OFRA's
full-volume, multiclass, natural-test-distribution results. The largest protocol
differences are the label space, replay budget, Benign exposure, training caps,
row-wise versus duplicate/group-aware splitting, and whether the test set keeps
its natural imbalance.

The first fair OFRA follow-up is therefore **not** to balance the test set. It is
to cap only the training rows per class, keep every official test row, keep the
same task order and model, and report the same NIDS metrics. That isolates the
effect of training-volume imbalance without making the evaluation easier.

## Comparison table

| Study/protocol | Learning problem | Dataset construction | Imbalance handling | Test protocol | Seeds / caveats | What OFRA should borrow |
|---|---|---|---|---|---|---|
| Current formal OFRA | Multiclass Class-IL; each class appears at a labelled task boundary | Dataset-specific preprocessing; Task-0-train-only normalisation; full training shards | One-vs-rest heads; at most 4 negatives per positive; focal loss below 1,000 positives; 50 exemplars/class | Natural official test distribution; accuracy, Macro-F1, balanced accuracy, attack recall, benign FPR and forgetting | Five seeds `{1,2,3,4,42}` in the formal campaign | Keep as the unmodified reference arm |
| ReplayIDS primary (Ariffin/Azizi et al., 2026) | Eight-class CICIDS2017 CI/CII | Historical executable contract uses Tuesday+Wednesday counts, retained duplicates, invalid numeric values replaced by zero, row-stratified 60/20/20 split | ER-Stratified or ER-Balanced at 1%, 5%, 10% per class; CII reintroduces Benign every experience | Same row-stratified test split | Seed 42; paper prose and executable preprocessing differ; Heartbleed has 11 rows total | Test replay size and Benign anchoring, but preserve our stricter evidence labels |
| ReplayIDS cross-architecture | Eight-class CICIDS2017 CI/CII | All features continuous; separate pipeline | Majority training classes capped at 50,000; class-weighted CE capped at 20; replay buffer 10% | Full test retained | Seed 42, 5 epochs; trend comparison only | Use the **50k training cap + full test** as the first sampling pilot |
| Augmented Memory Replay (NeurIPS 2023) | Mainly task/domain-incremental NID across multiple datasets | Public NIDS benchmarks and a long-term AnoShift stream | ECBRS changes how the memory is populated under severe imbalance; PAPA reduces interference-search cost | Benchmark-specific natural streams | Broad dataset coverage; not the same multiclass OFRA protocol | Compare memory-population policy, not only raw-data oversampling |
| SOUL (2024) | Binary domain-incremental, semi-supervised open world | CIC17/18: identifiers and duplicates removed, NaN imputed with feature means, min-max scaling; UNSW removes six identifier/time fields then categorical encoding | Replay buffer + gradient projection memory; at most 20% labelled data | Day/task stream; PR-AUC and area-under-time | Seen/unseen tasks differ by dataset; binary labels remain `{benign, attack}` | Borrow confidence-plus-memory agreement for **unknown rejection**, not its binary headline numbers |
| SPIDER (INFOCOM 2024) | Semi-supervised continual NID | Dataset/task-specific stream | Projection memory and limited labels; no need to store a complete labelled history | Uses at most 20% labelled data | Different label supervision and model objective | Candidate comparison for low-label adaptation, not a direct OFRA baseline |
| Recent replay study (Applied Soft Computing, 2026) | Domain-incremental NID across MLP/LSTM/Transformer | Balanced and unbalanced regimes; sequential attack stages | Small replay memory; reported replay dominance across architectures | Cumulative seen-attack evaluation | Five random task orders | Supports testing replay before increasing encoder size |
| CLEAR (2026 preprint/article preview) | Hybrid known-class + zero-day/anomaly detection | Four NIDS datasets | Semantic replay plus supervised/unsupervised hybrid detection | Explicit zero-day/few-shot evaluation | Recent publisher preview; full protocol still needs independent reproduction | Supports a separate reject-then-adapt line rather than forcing every sample into a known class |

## ReplayIDS/Azizi facts that materially affect comparison

The primary ReplayIDS configuration uses a 32-dimensional categorical
embedding, six Transformer blocks, ten attention heads, dropout 0.1, AdamW at
`5e-4`, weight decay `1e-5`, batch 256, a plateau scheduler and early stopping.
Its most important experimental factor is replay: in its independent
cross-architecture benchmark, replay raises CI final accuracy to about
`0.948–0.966` across FT-Transformer, MLP and CNN, while the non-replay models
largely collapse. MLP+replay is slightly above FT+replay in CI, whereas
FT+replay is highest in CII. This is evidence that model capacity is conditional
on the replay/data protocol, not a monotonic source of accuracy.

The historical executable ReplayIDS contract also differs from the prose:
duplicates are retained and invalid numeric values are replaced with zero.
Therefore its published result must not be described as a deduplicated-data
result, and our prior seed-42 comparison remains an expected-contract
reconstruction rather than an unchanged reproduction.

## New OFRA training-cap protocol

The first derived protocol is fixed before looking at its result:

- input: the existing hash-bound ReplayIDS expected-contract manifest;
- training only: reserve 10% per class for threshold calibration, then uniformly
  sample without replacement up to 50,000 fit rows per class;
- test: keep every original test row and its original hash;
- normalisation: fit only on the derived Task-0 fit rows;
- task order, encoder, epochs, loss, negative ratio, router cap and prediction
  arms: unchanged from the seed-42 OFRA comparison;
- primary comparison: capped-train OFRA versus uncapped-train OFRA on the same
  unchanged test set;
- metrics: Accuracy, Macro-F1, balanced accuracy, average forgetting, attack
  recall, benign FPR, per-class recall/F1 and confusion matrix;
- pilot: seed 42; publication confirmation: `{1,2,3,4,42}`.

This protocol answers a narrow question: **does majority-class training volume
and imbalance explain part of OFRA's lower performance?** It does not claim that
50,000 is optimal, and it does not make the test distribution balanced.

## Primary sources

- Ariffin et al., ReplayIDS paper and implementation:
  <https://arxiv.org/abs/2608.04602> and
  <https://github.com/um-csnet/ReplayIDS>
- Amalapuram et al., Augmented Memory Replay, NeurIPS 2023:
  <https://papers.neurips.cc/paper_files/paper/2023/hash/3755a02b1035fbadd5f93a022170e46f-Abstract-Conference.html>
- Amalapuram et al., SOUL:
  <https://arxiv.org/abs/2412.00911>
- SPIDER implementation:
  <https://github.com/amalapuram/spider>
- Handling class imbalance in CL-based NIDS implementation:
  <https://github.com/amalapuram/handling_CI_in_CL-based-NIDS>
- Task-aware memory replay (TAMR):
  <https://doi.org/10.1016/j.comnet.2025.111712>
- Continual learning for adaptive IoT NIDS:
  <https://doi.org/10.1016/j.asoc.2026.116022>
- CLEAR zero-day/continual framework:
  <https://doi.org/10.1016/j.knosys.2026.116878>
