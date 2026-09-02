# OFRA and ETG research code

This repository contains the current implementation and validated evidence for
the OFRA continual-learning study. Four intrusion-detection benchmarks and the
MalayaNetwork_GT external application-traffic dataset form the registered
five-dataset suite. Every dataset has its own preprocessing contract and is
trained and evaluated independently.

## Latest focused update: guarded checkpoint selection

The 3 September 2026 update adds a second paired five-seed checkpoint
experiment on the same ReplayIDS-aligned CIC-IDS-2017 D2 protocol. The new
candidate admits a non-final checkpoint only when training-only calibration
meets project-defined Macro-F1, positive-recall, and negative-FPR guards.
Implementation, tests, registered evidence, independent recomputation, W&B run
registry, and checksums are under
[`results/replayids-d2-checkpoint-recall-guard-paired5/`](results/replayids-d2-checkpoint-recall-guard-paired5/).
The aligned v3.0 manuscript and technical document are under
[`paper/guarded_checkpoint_2026-09-03/`](paper/guarded_checkpoint_2026-09-03/).

Relative to the immutable last-epoch baseline, the guarded rule changes final
accuracy by +1.59 percentage points, Macro-F1 by +1.53, forgetting by +0.04,
attack recall by -2.29, and benign FPR by -2.20. Every paired 95% confidence
interval includes zero. The guard largely removes the first selector's mean
forgetting penalty, but it does not establish superiority; last epoch remains
the registered primary protocol.

The current model uses an FT-Transformer encoder, family-specific low-rank
binary heads, DP-Means centroid routing, bounded exemplar memory, and five
matched scoring arms. The explanation stage uses SHAP expected gradients on
fixed checkpoint probes. ETG consumes explanation and performance evidence to
produce an offline governance ledger; it does not alter classifier predictions.

## Current evidence boundary

Five-seed prediction evidence is complete for the registered five-dataset
suite, with each dataset processed and evaluated independently. The guarded
checkpoint result is a paired five-seed mechanism ablation on one fixed D2 data
split. Source-bound SHAP and offline ETG evidence remains a Malaya seed-1
analysis, and protocol-matched external continual-learning baselines remain
open. No current result supports universal superiority or a deployed adaptive
OFRA-ETG control loop.

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
