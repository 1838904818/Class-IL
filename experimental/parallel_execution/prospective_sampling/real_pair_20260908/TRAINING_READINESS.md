# Training readiness after paired data derivation

The real P/O data preparation is complete, but no approved scientific training configuration follows automatically from that result.

The existing task0_pilot_config.json uses MLP width 128, depth 2, one epoch, seed 1, batch size 256 and Adam-compatible settings. The downstream pilot_config.json uses seed 17 and ten epochs. These are explicitly disclosed pilot defaults, not the historical FT system or a confirmatory capacity-matched experiment. Their existence is not authorization to train them or evidence that the requested architecture comparison has been specified.

Do not use this lightweight template as a replacement for the intended FT comparison. A pipeline smoke test, if requested, must remain separately labelled and excluded from headline evidence.

Before scientific training, bind the intended encoder architecture and optimizer, shared Task-0 cohort, fixed checkpoint selection, seed schedule, frozen normalization, and training exposure. Both P and O must share the same newly trained encoder if the intended contrast is sampling after a common representation. If representation training is also varied, that is a separate factorial contrast and must not be attributed solely to sample selection.

Equal epochs on 149476 versus 268697 fitting rows do not imply equal optimizer updates or row presentations. Preserve both the practical fixed-epoch comparison and an explicitly exposure-matched comparison if both questions are studied. Record shared pretraining and export costs separately from head-training costs, and count replay rows, stored exemplars, centroids and all retained parameters in total memory.

The P/O dataset contrast can address preprocessing policy effects; it cannot alone establish OFRA novelty, ETG intervention utility, or superiority to a published continual-learning baseline. These require the separately specified matched comparisons and held-out evaluations. No model training was executed during this readiness review.
