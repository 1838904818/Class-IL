# Saved-checkpoint common-score robustness (job493707)

60 cells /260 checkpoints; no training or mutation of original thresholds/results. Registered primary remains prior-shift; new score calibration is a posthoc robustness check. Normal holdout is permanent, prediction uses strict score > tau at1/2/5% targets. TEST FPR need not equal the holdout target. R3-A references remain as published, not newly re-evaluated with all baselines.

|Dataset|Method|Original BAcc %|Same-score BAcc %|Same-score TEST FPR %|paired-t Holm p|
|---|---|---:|---:|---:|---:|
|unsw-nb15|derpp|12.1669|15.8752|1.1312|8.4113e-06|
|unsw-nb15|icarl|12.8999|18.6923|0.8909|5.1772e-06|
|unsw-nb15|ecbrs-paper|11.7790|17.5483|1.1593|2.8255e-07|
|unsw-nb15|cer-bic-common|15.4362|21.5178|0.8979|3.7453e-05|
|cic-ids-2017|derpp|35.7505|33.2038|0.9579|0.00015457|
|cic-ids-2017|icarl|26.4742|26.3298|0.9823|1.8344e-06|
|cic-ids-2017|ecbrs-paper|19.1096|20.9889|0.9879|1.3831e-06|
|cic-ids-2017|cer-bic-common|35.8711|40.5658|0.9801|0.00015457|
|cic-ids-2018|derpp|14.3744|15.4147|1.0411|3.8826e-07|
|cic-ids-2018|icarl|28.4575|27.4671|0.9785|1.295e-05|
|cic-ids-2018|ecbrs-paper|14.3484|20.2154|1.0558|3.5809e-08|
|cic-ids-2018|cer-bic-common|34.5813|32.0157|0.9956|1.295e-05|

All comparisons use five paired seeds against as-published R3-A. Exact signflip is sensitivity only (minimum one-sided p1/32 cannot pass Holm4). Changes are not uniformly beneficial; no feature/mechanism causal attribution follows from changing the operating-point rule. Full2/5% results, attack recall, signed best-prior forgetting (including Normal), sample SD and individual seeds are in aggregate.json. Original-runtime CSE registered prior-rule4/4 paired-t/Holm PASS is reproduced; this is not overall research completion.

Costs: wall1686s, TotalCPU3213.365s; same-score step1445s, CPU2799.911s, MaxRSS7231468KiB. Original BiC14 validation occupied a separate sequential233s step. Per-cell elapsed excludes archive/data hashing and holdout prefetch. No hardware speedup claim. Complete job logs are sanitized; raw evidence hashes are retained locally. Each new JSON retains its pre-publication raw SHA; only publication metadata and path redaction are added.
