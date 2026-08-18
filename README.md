# OFRA and ETG research code

This repository contains the current implementation and validated evidence for
the OFRA continual-learning study. Four intrusion-detection benchmarks and the
MalayaNetwork_GT external application-traffic dataset form the registered
five-dataset suite. Every dataset has its own preprocessing contract and is
trained and evaluated independently.

The current model uses an FT-Transformer encoder, family-specific low-rank
binary heads, DP-Means centroid routing, bounded exemplar memory, and five
matched scoring arms. The explanation stage uses SHAP expected gradients on
fixed checkpoint probes. ETG consumes explanation and performance evidence to
produce an offline governance ledger; it does not alter classifier predictions.

## Evidence available in this release

| Dataset or stage | Completed evidence | Status |
|---|---|---|
| MalayaNetwork_GT | FT-Transformer 512x12, seeds 1-4 | Four-seed descriptive result |
| NSL-KDD | FT-Transformer 512x12, seeds 1-4 | Four-seed descriptive result |
| Malaya explanation and ETG | Seed 1, completed DICC Job 389896 | Single-seed partial analysis |
| Malaya attribution robustness | Expected Gradients, feature ablation, and Gradient x Input on seed 1 | Source-bound single-seed pilot |
| CSE-CIC-IDS2018 | FT-Transformer 256x4, seeds 1, 2, 3, 4, and 42, plus A100 profile | Protocol-separated five-seed closure |
| CIC-IDS-2017 and UNSW-NB15 | Preprocessing and training implementation | No new 512x12 formal result in this snapshot |

The main FT-Transformer 512x12 aggregate uses `1, 2, 3, 4`; it is not a
five-seed result. The CIC-IDS-2018 closure uses `1, 2, 3, 4, 42`, but its
FT256x4 model and 1+1-epoch schedule are not pooled with the main result. The
fifth main-protocol seed and remaining large-model dataset runs are pending.
The detailed metrics, per-seed files, and hashes are in
[`results/README.md`](results/README.md) and
[`results/aggregate_4seed.json`](results/aggregate_4seed.json).

A consolidated description of the architecture, preprocessing contracts,
training protocol, SHAP analysis, ETG logic, results, and evidence boundaries
is available in [`TECHNICAL_DOCUMENTATION.md`](TECHNICAL_DOCUMENTATION.md).

## Four-seed headline results

Values below are mean +/- sample standard deviation across seeds 1-4.

| Dataset | Scoring arm | Accuracy | Macro-F1 | Forgetting |
|---|---|---:|---:|---:|
| MalayaNetwork_GT | Joint full | 56.14% +/- 3.00 | 21.04% +/- 3.85 | 3.23 +/- 0.88 pp |
| MalayaNetwork_GT | Joint cap 3,000 | 54.37% +/- 3.02 | 20.70% +/- 3.72 | 3.79 +/- 0.64 pp |
| NSL-KDD | Joint full | 68.51% +/- 2.87 | 38.32% +/- 2.97 | 2.60 +/- 1.15 pp |
| NSL-KDD | Joint cap 3,000 | 69.07% +/- 3.38 | 38.81% +/- 3.04 | 2.38 +/- 1.34 pp |

These results are descriptive. They do not establish statistical superiority,
and the low Malaya Macro-F1 indicates weak minority-class performance despite
moderate overall accuracy.

## Decision architecture

For an input `x`, the encoder produces an embedding `h(x)`. Each seen class
head returns a positive-class probability `p(c, x)`. The DP-Means router
returns the negative distance to the nearest class centroid; these values are
standardised across seen classes to obtain `z(c, x)`.

- Head-only: `s(c, x) = p(c, x)`
- Router-only: `s(c, x) = z(c, x)`
- Joint: `s(c, x) = p(c, x) + 0.5 z(c, x)`

The predicted class is the global `argmax` over all seen classes. Router cap
`3,000`, joint weight `0.5`, the maximum centroid count, and exemplar budgets
are study configurations rather than universal constants.

## Explanation and ETG result

The formal analyzer in `formal_v2_explanation_etg/analyze.py` uses
`shap.GradientExplainer`, an expected-gradients approximation. It does not use
KernelSHAP. The completed Malaya seed-1 analysis reported 12 silent explanation
drift events among 17 eligible class-by-adjacent-checkpoint transitions
(70.59%). This unit is not packets, flows, samples, or real-world incidents.

ETG produced six certified admissions, four refused admissions, four
escalations, one strict recertification, and two strict recertification
failures. These are simulated explanation-governance outcomes, not completed
human reviews. The public analysis and ledger are under
`results/malaya-network-gt/etg-seed1/`.

The attribution robustness pilot compares Expected Gradients with
single-feature ablation and Gradient x Input on the same 30 checkpoint-class
rows. Integrated Gradients were attempted but failed their completeness check
on the piecewise routed score; they are retained as a diagnostic and excluded
from the primary agreement claim. Exact agreement values and source hashes are
under `results/malaya-network-gt/attribution-robustness-seed1/`.

The no-look-ahead audit confirms Task-0-only numerical scaling. It also finds
bounded future-category schema exposure in NSL-KDD and UNSW-NB15, affecting
about 0.9% of later-task rows. See `audits/NO_LOOKAHEAD_AUDIT_2026-08-19.md`.

## Repository layout

```text
configs/                         model configurations
fullcache/                       dataset-specific cache builders
ofra_encoders/                   FT-Transformer integration
streaming_full/                  current training, recovery, and evaluation
formal_v2_explanation_etg/       expected-gradients and ETG analyzer
results/                         completed results and public-safe evidence
reproducibility/                 runtime source and binding hashes
tests/                           protocol and regression tests
src/ and src_v2/                 earlier baselines and compatibility code
```

Some compatibility modules retain former filenames so earlier experiment
records remain traceable. The method name used in the current study is OFRA.

## Data access

The data index and authorised mirrors are available from the public
[Kaggle dataset](https://www.kaggle.com/datasets/wuliqiang/leon-nids-classil).
Authoritative source links, licences, and preprocessing contracts are recorded
in [`DATASETS.md`](DATASETS.md). Raw datasets are not stored in Git.

## Environment

Create an isolated environment, install PyTorch for the target CUDA platform,
then install the recorded dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-repro.txt
python -m pip install -r requirements-fttransformer.lock
```

## Build a dataset cache

```bash
python -m fullcache \
  --data-root /path/to/datasets \
  --output-root /path/to/cache \
  --dataset malaya-network-gt
```

The cache builder records file hashes, feature schemas, split contracts, and
overlap audits. Dataset-specific details are in `DATASETS.md`,
`FULL_CACHE_README.md`, and `MALAYA_NETWORK_GT_PROTOCOL.md`.

## Run the current model

```bash
python -m streaming_full \
  --manifest /path/to/cache/malaya-network-gt/streaming_manifest.json \
  --output-dir /path/to/run-output \
  --seeds 1 \
  --config-json configs/ft_transformer_512x12_a100_formal.json \
  --device cuda:0
```

Do not disable shard-hash verification for reportable experiments.

## Validation

```bash
python -m unittest discover -s tests -p "test_*.py" -v

export OFRA_PROJECT_DIR="$(pwd)"
export PYTHONPATH="$OFRA_PROJECT_DIR/formal_v2_explanation_etg:$OFRA_PROJECT_DIR"
(cd formal_v2_explanation_etg && python -m unittest -v \
  test_analyze.py test_publish_wandb.py test_verify_submission_bindings.py)
```

`SHA256SUMS.txt` is generated from the published LF-normalised files. The
runtime manifests in `reproducibility/` bind the current code to the completed
training and analysis evidence.

## Manuscript and cited papers

The current v2.2 supervisor-response manuscript and the itemized response to
review findings 1--9 are under [`paper/`](paper/). A
complete linked library of every cited paper, together with a reusable BibTeX
file and redistribution notes, is under [`references/`](references/REFERENCES.md).
Publisher PDFs are not mirrored when redistribution permission has not been
established.
