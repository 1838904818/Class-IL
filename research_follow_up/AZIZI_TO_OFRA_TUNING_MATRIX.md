# Azizi/ReplayIDS-guided OFRA tuning matrix

## Research question

Which part of the ReplayIDS result transfers to OFRA: the encoder, replay
budget, repeated Benign exposure, optimiser/training schedule, or the easier
data contract?

## Evidence-based answer before new runs

Replay is the dominant candidate. ReplayIDS' own cross-architecture benchmark
shows that MLP, CNN and FT-Transformer all become strong under replay, while
larger or more structured encoders do not rescue CI without replay. Therefore
OFRA should not jump directly from FT256x4 to a larger Transformer.

## Controlled sequence

Each stage changes one factor relative to the previous accepted reference. A
later stage is justified only if the earlier stage identifies headroom.

| Stage | Fixed factors | Changed factor | Arms | Decision rule |
|---|---|---|---|---|
| A0 reference | Current ReplayIDS expected-contract seed-42 configuration | None | current `exemplar_capacity=50`, uncapped training data | Existing immutable result only |
| A1 data-volume pilot | Model, epochs, loss, router, replay=50 | 50k/class cap on fit-train only; 10% train-only calibration holdout; full test unchanged | uncapped vs capped | Continue if Macro-F1 or balanced accuracy improves without unacceptable attack-recall loss |
| A2 replay-size pilot | Use the better A0/A1 data protocol; same model | exemplar capacity | `50`, `500`, `3000` | Select by Macro-F1/forgetting Pareto frontier and memory cost, not accuracy alone |
| A3 proportional replay diagnostic | Same as A2 | percentage-like replay with hard cap | `min(1% of class,3000)` and `min(5%,3000)` | Tests the ReplayIDS idea without allowing Benign memory to dominate |
| A4 Benign anchoring | Best replay size; same model | repeat fresh Benign rows at later task boundaries | CI versus CII-like Benign anchoring | Accept only if minority recall improves and Benign FPR does not become the sole source of the gain |
| A5 optimiser/schedule | Best data/replay arm; same FT256x4 | Adam versus AdamW, `5e-4`, `1e-3`, scheduler, early stopping | small predeclared matrix | Use validation Macro-F1 with a fixed patience; never tune on official test |
| A6 architecture | Best preceding protocol | backbone | MLP256-128, current FT256x4, Azizi-like TabTransformer/FT | Report parameters, time and memory; no claim that larger is better without five-seed evidence |

## Why not copy a 10% ReplayIDS buffer directly?

ReplayIDS defines memory as a percentage per class. On the OFRA/ReplayIDS
training split, 10% of Benign alone is over fifty thousand rows, versus OFRA's
current 50 rows per class. Copying 10% would simultaneously change memory,
training exposure, runtime and the operational premise of a small replay
buffer. The bounded sequence `50 → 500 → 3000`, plus percentage-with-cap
diagnostics, is interpretable and remains compatible with OFRA's memory claim.

## Early stopping contract

Early stopping is not “train fewer epochs until the test looks best.” It must:

1. reserve calibration/validation rows from training before fitting;
2. monitor a declared validation metric, preferably Macro-F1 under imbalance;
3. save the best checkpoint using a fixed patience and minimum improvement;
4. evaluate the untouched official test once after model selection;
5. record the maximum epoch, selected epoch and validation history.

The current formal runtime has a fixed epoch budget and no validation split.
Adding early stopping is therefore a new experimental arm, not a silent update
to the existing evidence.

## Minimum publication evidence

- five seeds `{1,2,3,4,42}` for the selected arm and the matched reference;
- paired per-seed statistics;
- Accuracy, Macro-F1, balanced accuracy, forgetting, attack recall, benign FPR;
- per-class results and confusion matrices;
- parameter count, replay rows/bytes, training time and peak GPU memory;
- immutable config/code/data/result hashes and W&B run links when available.
