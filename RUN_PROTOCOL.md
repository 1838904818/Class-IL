# OFRA deterministic reproduction protocol

This directory is the frozen code snapshot for the 14 July 2026 local
reproducibility check. Run from this directory with the exact local dataset
tree supplied through `CLASS_IL_DATA_DIR`.

## Required process environment

```powershell
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:FULL_DATA = "1"
$env:MAX_PER_CLASS = "50000"
$env:CLASS_IL_DATA_DIR = "C:\path\to\datasets"
$env:CLASS_IL_RESULTS_DIR = "<output-directory>"
```

Example single-dataset command:

```powershell
python -X utf8 -u -m src_v2.multi_seed_ofra `
  --datasets "NSL-KDD" `
  --seeds "42" `
  --pretrain-epochs 8 `
  --epochs-per-task 10 `
  --exemplar-capacity 50 `
  --lora-rank 8 `
  --encoder-type mlp `
  --loss-fn focal `
  --n-layers 2 `
  --d-model 128 `
  --device cuda `
  --router-fit-max-samples 3000 `
  --out-suffix "nsl_repeat_a"
```

## Reproduction evidence stored in each JSON

- semantic protocol hash, excluding output name, verbosity, and elapsed time;
- SHA-256 manifest for all Python source files used by the run;
- SHA-256 for every raw dataset file read by the loader;
- dtype, shape, and SHA-256 for the processed train/test feature and label arrays;
- Python, package, CUDA, cuDNN, driver, GPU, and deterministic-kernel settings;
- a deterministic result SHA-256 covering metrics, accuracy matrices, router
  sample counts, protocol, and provenance while excluding runtime.

Two repeated runs pass only when their `run_fingerprint_sha256`, processed
input hashes, accuracy matrices, and `deterministic_result_sha256` are equal.

## Scope limitations

- This is a seed-42 reproducibility check, not a multi-seed final paper table.
- NSL-KDD and UNSW-NB15 use their complete fixed splits.
- CIC-IDS-2017 uses all local rows when `FULL_DATA=1`.
- CIC-IDS-2018 and NF-ToN-IoT-v2 use `MAX_PER_CLASS=50000`.
- CIC-IoT-2023 reads all 169 files, then retains Normal up to 100,000 and each
  attack family up to 50,000 under its current loader.
- The local CIC-IDS-2018 collection has nine traffic CSVs and is not the
  official ten-day complete mirror.
- The current CIC-IDS-2017 random row split has known exact train/test duplicate
  overlap. These runs are reproducibility pilots; the split must be revised
  before a final paper-grade multi-seed experiment.
