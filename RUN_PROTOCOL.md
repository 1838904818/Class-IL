# Current deterministic run protocol

This protocol applies to the FT-Transformer 512x12 evidence published in this
repository. It supersedes earlier MLP pilot commands.

## Inputs

Build one cache at a time. Datasets are never concatenated.

```bash
python -m fullcache \
  --data-root /path/to/datasets \
  --output-root /path/to/cache \
  --dataset nsl-kdd
```

Replace `nsl-kdd` with another registered dataset identifier when building a
different cache. The generated `streaming_manifest.json` binds raw-file hashes,
processed-shard hashes, feature order, class order, task order, split contract,
and row accounting.

## Model run

```bash
python -m streaming_full \
  --manifest /path/to/cache/nsl-kdd/streaming_manifest.json \
  --output-dir /path/to/output/nsl-kdd/seed-1 \
  --seeds 1 \
  --config-json configs/ft_transformer_512x12_a100_formal.json \
  --device cuda:0
```

Repeat the command independently for each registered seed and dataset. The
completed public seed set is `1, 2, 3, 4`; it is not a five-seed result.

## Determinism and integrity

- deterministic PyTorch execution is enabled;
- processed-shard SHA-256 verification is mandatory;
- the result stores a semantic protocol hash and deterministic result hash;
- checkpoint recovery validates the recovery payload before resuming;
- W&B is an experiment-recording destination, not the source of record;
- the JSON result and its hashes remain the authoritative evidence.

The exact source binding used by the completed large-model runs is recorded in
`reproducibility/ft512x12_training_runtime.sha256`.

## Explanation and ETG

The completed ETG evidence is an offline analysis of MalayaNetwork_GT seed 1:

```bash
python formal_v2_explanation_etg/analyze.py --help
```

The analyzer validates its input bindings, reconstructs the registered
`joint_cap3000` decision, computes SHAP expected-gradient explanations on fixed
probes, evaluates explanation drift, and emits a simulated ETG governance
ledger. The analyzer does not retrain the model or change predictions. The
published analysis is bound to completed DICC Job `389896`.

## Evidence boundary

- MalayaNetwork_GT and NSL-KDD: four completed 512x12 seeds;
- MalayaNetwork_GT explanation/ETG: one completed seed;
- CSE-CIC-IDS2018: capacity profile only;
- CIC-IDS-2017 and UNSW-NB15: implementation present, but no new 512x12 formal
  result in this release.

Do not use capacity results as accuracy evidence and do not describe the
four-seed aggregate as a final five-seed statistical result.
