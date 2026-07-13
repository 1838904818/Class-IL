# Explanation Stability under Class-Incremental Intrusion Detection

This repository contains the research implementation and experimental record
for **OFRA** (Oracle-Free Routed Adapters) and **ETG** (Explanation-Trust Gate).
It includes stored experiment outputs and the current paper on explanation
stability in class-incremental network intrusion detection.

Current manuscript:

- [`paper_v3/paper.pdf`](paper_v3/paper.pdf)
- [`paper_v3/paper.docx`](paper_v3/paper.docx)

The consolidated numerical record is
[`results/evidence_summary.json`](results/evidence_summary.json). Full-data Kaggle
runs are ongoing and are not included in the claims below.

## Methods and evaluation scope

### OFRA

OFRA uses a frozen two-layer MLP encoder (`d -> 128 -> 128`), one rank-8
output-space adapter and binary head per family, bounded exemplars, and
DP-Means centroid memory. All learned family heads compete at inference through
the joint score `s(f) = p(f) + 0.5 z(f)`; DP-Means is not a hard family
pre-selector. Each family module has 2,306 trainable parameters.

### ETG

ETG is currently a separate detector-agnostic proof of concept on a
replay-trained shared MLP. It has not yet been integrated with OFRA. The complete
governance ledger uses seed 42; the five-seed ETG experiment repeats admission
only.

Historical implementation and result filenames may retain the method's earlier
identifier for traceability. The method reported in the current manuscript is
OFRA.

## Verified results

### Decision-level silent explanation drift

The formal joint-score result uses 25 complete runs: five datasets and seeds
`{0, 1, 2, 3, 42}`. A transition is counted when accuracy changes by more than
`-5` percentage points and top-15 attribution Jaccard is below `0.70`.

| Dataset | Events / preserved transitions | Rate |
|---|---:|---:|
| NSL-KDD | 2 / 36 | 5.6% |
| UNSW-NB15 | 0 / 80 | 0.0% |
| CIC-IDS-2017 | 3 / 48 | 6.25% |
| CIC-IDS-2018 | 0 / 51 | 0.0% |
| NF-ToN-IoT-v2 | 1 / 83 | 1.2% |
| **Pooled** | **6 / 298** | **2.0%** |

The isolated family path records `0/299` silent-drift events. The joint-score
wrapper is not yet a bit-exact reproduction of deployed NumPy standardisation,
so `6/298` is reported as the stored protocol result.

### Forgetting and accuracy trade-off

Four datasets have protocol-matched five-seed paired records. Their
geometric-mean forgetting reduction is `3.07x`; accuracy costs range from `7.7`
to `16.0` percentage points. CIC-IDS-2018 is not significant after Holm
correction (`p = 0.1019`).

| Dataset | Comparator | Reduction | Holm p | Accuracy cost |
|---|---|---:|---:|---:|
| NSL-KDD | Replay-DPMeans | 6.59x | 0.0085 | -16.0 pp |
| UNSW-NB15 | iCaRL | 3.53x | 8.36e-6 | -7.7 pp |
| CIC-IDS-2017 | Replay-DPMeans | 2.91x | 0.0267 | -10.6 pp |
| CIC-IDS-2018 | iCaRL | 1.31x | 0.1019 (n.s.) | -7.8 pp |

A legacy five-seed NF-ToN-IoT-v2 comparison yielded `2.33x` versus iCaRL under
a different Transformer encoder (`d=64`, five masked-feature pretraining
epochs). The formal OFRA protocol uses an MLP encoder (`d=128`) with eight
epochs of Task-0 supervised pretraining. The legacy NF-ToN value is therefore
excluded pending a protocol-matched rerun.

The five-seed OFRA arrays used for the paired analysis are retained in the
historically named `results/pilora_v05_*.json` files. Running
`python -m src_v2.paired_ttest` rebuilds the paired-test summaries.

### ETG proof of concept

The seed-42 ledger certifies 34 of 39 families, refuses five, records ten
escalation events, withholds two explanation-drift notifications for families
without an admission certificate, and records three mass-only
re-certifications. The underlying NIDS alerts remain unchanged. These are
state-machine transitions, not ground-truth drift labels or human-study
outcomes.

## Evidence boundaries

- CIC-IDS-2017, CIC-IDS-2018, and NF-ToN-IoT-v2 use capped majority classes;
  rare available samples are retained.
- Scaling statistics are fitted on the complete training partition before the
  task stream. No test labels are used, but later training-task distributions
  are visible to preprocessing.
- The CIC-IDS-2017 cache contains an unused WebAttack label slot; seven non-empty
  families are evaluated.
- Baseline replay budgets are not memory matched to OFRA.
- Full-data Kaggle outputs and CIC-IoT-2023 are outside the reported evidence.

## Repository layout

```text
src/                         shared-parameter Class-IL baselines
src_v2/                      OFRA implementation and benchmarks
src_v3/                      explanation-stability and faithfulness protocols
exp_etg*.py                  ETG ledger, ablation, multiseed, and repair studies
results/                     stored JSON outputs and consolidated summary
figs/                        experiment figures
paper_v3/                    current manuscript
tests/                       OFRA protocol and device regression checks
```

## Reproducing the stored studies

Key entry points are:

```bash
python -m src_v2.multi_seed_ofra
python -m src_v2.paired_ttest
python -m src_v3.decision_stability --help
python exp_etg.py
python exp_etg_ablation.py
python exp_etg_multiseed.py
python -m unittest discover -s tests -p "test_ofra_guardrails.py" -v
```

Datasets are downloaded separately and remain subject to their original
licences and distribution terms.
