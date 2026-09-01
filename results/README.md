# Validated result index

This directory indexes validated evidence available on 1 September 2026. Each
dataset is trained and evaluated independently under its registered contract.

## Current paired five-seed checkpoint-selection result

`replayids-d2-checkpoint-selection-paired5/` contains the completed Job
`425539` comparison between last-epoch retention and training-only checkpoint
calibration for seeds `1, 2, 3, 4, 42`. Calibration improves final accuracy by
2.02 percentage points and Macro-F1 by 1.95 points, but reduces average task
accuracy by 1.91 points, increases forgetting by 1.66 points, and reduces
attack recall by 1.97 points. Every paired 95% confidence interval crosses
zero. The candidate is not promoted to the primary protocol.

## Historical four-seed descriptive aggregate

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

## ReplayIDS O1 open-set pilot

`replayids-o1-open-set-seed42/` contains a path-sanitised binding for completed
DICC Job `414686`. All registered pre-label unknown gates missed the held-out
FTP-Patator class (0% unknown recall). After its label was supplied, the normal
supervised OFRA update achieved 82.30% new-class recall while old-class accuracy
fell by 0.27 percentage points. This is single-seed diagnostic evidence and does
not establish autonomous new-head creation.

## Historical CSE-CIC-IDS2018 capacity snapshot

`cic-ids-2018/capacity_profile.json` is the earlier non-reportable throughput
and memory measurement used for resource planning. It predates the completed
strict five-seed campaign and must not be used as the current result.
