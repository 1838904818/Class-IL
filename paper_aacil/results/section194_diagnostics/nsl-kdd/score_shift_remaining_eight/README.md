# NSL remaining eight methods: final-checkpoint score diagnostics

Share with caveats:40 newly evaluated final checkpoints /160 original default+1/2/5% confusion matrices matched exactly on Linux (job493771). No model training or original result/threshold mutation. Raw NPZ bytes are preserved; JSON scientific values are unchanged with personal paths redacted and raw SHA receipts added. The prior R3-A/DER++ group is referenced by immutable paths/hashes, not copied here or counted as new evaluation.

## Ten-method descriptive comparison

Every method and seed at2% has194/9711=1.9977345279% TEST benign FPR. Selection uses TEST benign labels: this is posthoc, not registered S1. BAcc is a fraction in JSON and percent below. Sample SD is across five seeds.

|Method|2% BAcc mean ± sample SD (pp)|2% TEST FPR|5% BAcc mean|5% actual TEST FPR|
|---|---:|---:|---:|---:|
|R3-A|47.4398 ± 1.7596|1.9977345%|68.7949|4.6298013%|
|R3-L|47.4535 ± 3.9607|1.9977345%|67.1692|4.9943363%|
|ar|51.2739 ± 2.9270|1.9977345%|56.1496|4.9943363%|
|replay_bs|43.2456 ± 5.9554|1.9977345%|59.0132|4.9943363%|
|freq|43.2891 ± 3.6542|1.9977345%|57.3963|4.9943363%|
|wacb|43.7990 ± 2.6669|1.9977345%|64.6697|4.9943363%|
|derpp|42.6339 ± 5.4305|1.9977345%|62.9649|4.9943363%|
|icarl|34.5243 ± 0.8000|1.9977345%|39.5651|4.9943363%|
|ecbrs|37.6995 ± 2.5483|1.9977345%|53.4212|4.9943363%|
|bic|39.2961 ± 5.0811|1.9977345%|43.6592|4.9943363%|

Neural ar51.2739% exceeds R3-A47.4398% at the identical2% actual FPR; R3-L47.4535% is numerically close to R3-A. Do not present R3-A as a matched-FPR all-method winner. R3-A has a lower actual FPR at the5% budget owing to ties; the5% comparison is not strictly FPR-matched. Default and original holdout-target results remain unchanged, including weak/negative results.

The score rule remains the saved model's original gate margin or float32 prior-shift operation. Its numeric scale is not common across those rules. KS and within-model holdout/TEST quantile differences describe score transfer, not a causal feature mechanism. Raw arrays include logits/scores and rowIDs/labels, not input feature matrices.

Independent validation: SciPy KS40, directly interpolated quantiles960, NumPy prediction + plain-list CM80 and budget/metric checks; completed verifier log included. Aggregate directly rechecks all50 JSON/NPZ receipts, all TEST rowIDs/labels and descriptive means/sample SD. This does not rerun prior two-method models. No inferential posthoc p-values are promoted to registered evidence.

Cost: wall92s, TotalCPU26.740s, step MaxRSS1261872KiB. No hardware speedup claim. Full completed diagnostic/Slurm, software-sample, submission and static-check logs are redacted gzip with original byte/hash receipts. Export/publication streams created after this freeze stay local, not inside this archive. The effective config/source commit/seed/command and checkpoint/data bindings remain in each cell JSON.

Primary scientific scope and remaining RQ1/RQ3/innovation/manuscript source visual QA/backup requirements are unchanged. C493535 pending is not a scientific result.
