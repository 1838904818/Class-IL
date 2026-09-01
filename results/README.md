# Validated result index

## 2 September 2026: paired checkpoint-selection experiment

The current focused result package is
[`replayids-d2-checkpoint-selection-paired5/`](replayids-d2-checkpoint-selection-paired5/).
It contains the five paired seeds (`1, 2, 3, 4, 42`), aggregate and per-class
metrics, the registered no-look-ahead protocol, W&B source-run registry,
figures, and SHA-256 manifests. The conclusion is mixed and inconclusive;
last-epoch selection remains the primary protocol.

This directory contains the completed FT-Transformer 512x12 results available on 11 August 2026. Each dataset was trained and evaluated independently.

## Four-seed descriptive aggregate

The completed seed set is `1, 2, 3, 4`. These numbers must not be described as a five-seed result.

| Dataset | Scoring arm | Accuracy | Macro-F1 | Balanced accuracy | Forgetting |
|---|---|---:|---:|---:|---:|
| MalayaNetwork_GT | Head-only | 55.87% +/- 0.33 | 11.81% +/- 0.93 | 15.16% +/- 0.58 | 1.25 +/- 1.19 pp |
| MalayaNetwork_GT | Router cap 3,000 | 40.98% +/- 10.67 | 18.56% +/- 1.51 | 20.68% +/- 1.92 | 8.02 +/- 1.47 pp |
| MalayaNetwork_GT | Joint cap 3,000 | 54.37% +/- 3.02 | 20.70% +/- 3.72 | 22.70% +/- 3.86 | 3.79 +/- 0.64 pp |
| MalayaNetwork_GT | Router full | 46.48% +/- 2.53 | 19.66% +/- 2.62 | 21.61% +/- 2.74 | 7.90 +/- 1.33 pp |
| MalayaNetwork_GT | Joint full | 56.14% +/- 3.00 | 21.04% +/- 3.85 | 22.89% +/- 3.92 | 3.23 +/- 0.88 pp |
| NSL-KDD | Head-only | 61.94% +/- 19.92 | 34.76% +/- 10.29 | 38.12% +/- 6.94 | 5.44 +/- 9.48 pp |
| NSL-KDD | Router cap 3,000 | 67.48% +/- 3.93 | 45.80% +/- 4.99 | 50.83% +/- 6.26 | 9.93 +/- 5.18 pp |
| NSL-KDD | Joint cap 3,000 | 69.07% +/- 3.38 | 38.81% +/- 3.04 | 40.87% +/- 2.96 | 2.38 +/- 1.34 pp |
| NSL-KDD | Router full | 66.72% +/- 3.65 | 44.55% +/- 5.23 | 49.89% +/- 6.04 | 8.92 +/- 4.36 pp |
| NSL-KDD | Joint full | 68.51% +/- 2.87 | 38.32% +/- 2.97 | 40.44% +/- 2.83 | 2.60 +/- 1.15 pp |

Values are mean +/- sample standard deviation across the four completed seeds. The full per-seed records and deterministic hashes are in `aggregate_4seed.json`.

## Explanation-governance result

The completed MalayaNetwork_GT seed-1 analysis is under `malaya-network-gt/etg-seed1/`. Its primary silent explanation-drift rate is 12/17 class-by-adjacent-checkpoint transitions (70.59%). This is not a packet, flow, sample, or real-world incident rate. ETG actions are simulated governance outcomes, not completed human reviews.

## CSE-CIC-IDS2018 status

`cic-ids-2018/capacity_profile.json` is a non-reportable throughput and memory measurement used for resource planning. Formal CSE-CIC-IDS2018 model training was not complete at this release and no performance claim is made from the capacity profile.
