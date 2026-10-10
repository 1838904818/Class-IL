# Exp 2 frozen four-cell result

## What ran

The frozen first Exp 2 matrix is complete for NSL-KDD and UNSW-NB15, Replay and iCaRL, and seeds 1/2/3/4/42. Each cell evaluates N1 same-checkpoint controls, N2 function-preserving controls, strong P1 shortcut interventions, three explainers, and the prespecified detector baselines. Replay also evaluates eligible P2 constructions. The frozen iCaRL P2 screens found zero eligible constructions, so iCaRL P2 is reported as unavailable rather than as a detector success or failure.

## Numbers

| Dataset | Method | NCD negative-control FP | B2 negative-control FP | NCD strong-P1 detection | B2 strong-P1 detection | NCD P2 detection | Point gate |
|---|---|---:|---:|---:|---:|---:|---|
| NSL-KDD | Replay | 0/34 | 0/34 | 8/8 | 8/8 | 5/5 | Pass |
| UNSW-NB15 | Replay | 0/30 | 0/30 | 5/5 | 5/5 | 4/4 | Pass |
| NSL-KDD | iCaRL | 0/20 | 0/20 | 9/9 | 9/9 | n/a | Pass |
| UNSW-NB15 | iCaRL | 1/20 (5.0%) | 0/20 | 4/4 | 4/4 | n/a | Pass at boundary |

Across the four cells, the descriptive—not prespecified pooled—total is 1/104 NCD false positives versus 0/104 for B2, 26/26 strong-P1 detections for both detectors, and 9/9 P2 detections for both detectors where P2 exists. N2 function-preservation checks have zero maximum score error in the iCaRL cells; the corresponding Replay checks are reported in their cell-level evidence.

Bootstrap removal reduces strong-P1 detection from 8/8 to 6/8 and P2 detection from 5/5 to 3/5 on NSL-KDD Replay; from 5/5 to 4/5 and 4/4 to 2/4 on UNSW Replay; from 9/9 to 4/9 on NSL-KDD iCaRL; and from 4/4 to 3/4 on UNSW iCaRL. However, removing explainer repeats preserves all eligible positive detections and removes the single UNSW iCaRL false positive. This ablation therefore does not uniquely validate the full NCD mechanism.

## Decision

The frozen point-estimate gate passes in all four cells, with UNSW iCaRL exactly at the 5% false-positive boundary. The available evidence does **not** establish NCD superiority: B2 ties every eligible positive result and has fewer negative-control false positives. The finite negative-control samples also do not establish a population false-positive rate below 5%; the cell-level Wilson upper bounds range from 10.2% to 23.6%.

The supported conclusion is narrower: repeated explanation measurement detects the constructed silent-drift interventions on these benchmark slices, while the simple repeat-mean B2 control is sufficient on all eligible cases observed here. This is not evidence that the current NCD combination is a substantive innovation, broadly general, or publication ready.

## Next

Do not claim NCD as the novel mechanism. Preserve B2 as the strongest simple control, add external threshold calibration before generalization claims, and either introduce a genuinely distinct mechanism with a decisive endpoint ablation or frame the contribution as a bounded measurement study of silent explanation drift.
