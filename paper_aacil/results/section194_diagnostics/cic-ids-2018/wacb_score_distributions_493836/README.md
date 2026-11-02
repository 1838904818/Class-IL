# CSE2018 WACB final score distributions (job493836)

Share with caveats. Five final checkpoints were read without training or original result/threshold changes. Independent raw-array checks reproduce20 TEST confusion matrices,15 holdout FP counts, all saved prior-crossing scores,5 KS statistics,540 quantiles, all CDF knots, per-class recall and Normal absorption. All recomputed holdout FP deltas are retained in aggregate.json.

|Seed|1% BAcc %|1% TEST FPR %|Benign holdout/TEST KS|Holdout q99|TEST benign q99|
|---|---:|---:|---:|---:|---:|
|10|21.39756|0.9972000|0.00745911|4.8533083|4.8468788|
|11|24.72890|1.0466879|0.00944624|7.3839601|7.5598470|
|12|33.85526|0.9876373|0.00678361|2.3746459|2.3386706|
|13|58.33296|0.9844374|0.00905154|1.2606582|1.2553530|
|14|72.65342|1.0033395|0.01040315|1.6164257|1.6214957|

Benign KS ranges0.00678361–0.01040315 while measured TEST FPR remains approximately1%. BAcc ranges21.40–72.65%, mean42.19362% with sampleSD22.33438pp. These observations do not support explaining the large spread as a large benign FPR change or a demonstrated bisection solver failure. Existing stored calibration searches were monotone and budget-bracketed; independent current holdout FP counts agree. Per-class score quantiles and Normal absorption describe different saved-model decision geometry, not its feature-level cause or a proven memory/training mechanism.

Scores are ideal real-arithmetic prior-crossing tau values within each model. Original predictions still use float32 prior shift. Absolute score scale across independently trained seeds is not a common unit. Diagnostic forward batches4096 differ from original config2048: all required CMs/FP counts agree, but original per-row logits were not available for a bitexact comparison. The raw NPZ includes logits/scores/rowIDs/labels, not input features.

## Restore exact raw arrays

Each original seed NPZ is134–136MB. [GitHub documents a100MiB regular-file block and50MiB warning](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). Therefore raw bytes are split into45MiB parts without re-encoding. Each part and the full original SHA/size are in the scientific aggregate; shared row/labels are copied unchanged. Use the published restore tool with a fresh destination, for example:

    python ncd/hpc/restore-cse-wacb-score-arrays.py --archive <this-directory> --output <fresh-directory>

The tool checks every part and full SHA, then loads every array with allow_pickle=False and verifies shapes/dtypes. Original NPZ files are retained unchanged in protected local storage. No checkpoint/data features are redistributed.

## Costs and remaining scope

Successful job493836:87s wall,87.914s TotalCPU,step MaxRSS6823588KiB. The earlier cancelled attempt consumed1031s/2.513CPU seconds, and startup-only493824 consumed35s/9.427CPU seconds. Sequential HPC diagnostic workflow total1153s wall and99.854CPU seconds is reported separately from the successful run; do not call87s the whole cost or claim speedup. Old slowdown did not recur; its original cause remains unproved. Failed/startup streams are already published under ../wacb_startup_trace_493824, not counted as new model results.

New completed logs include import-time stderr, which is not empty and must not be described as stderr0. Per-cell effective_config/source commit/seed/command/checkpoint/data/holdout bindings and all arrays remain inspectable. Later export/restore/publication validation streams stay local, not inside this frozen archive. Original registered comparisons, manuscript/RNG footnote/current-source visual QA, RQ1/RQ3, substantive innovation and backup requirements remain separate and open.
