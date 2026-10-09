# Noise-Calibrated Drift experiments

This directory contains the current intervention-ground-truth experiment for explanation drift in task-free class-incremental NIDS.  It is a bounded Exp 2 pilot, not a publication-level validation.

## Environment and data

Use Python with PyTorch, NumPy, pandas, SciPy, and scikit-learn.  The datasets are not redistributed.  Place the authorized NSL-KDD files at:

```text
<data_root>/nsl-kdd/KDDTrain+.txt
<data_root>/nsl-kdd/KDDTest+.txt
```

Set `data_root` in `configs/exp2_nsl_replay_seed1.json` to the parent directory.  The preprocessing and class order are implemented in `data.py`.

## Reproduce the pilot

From the repository root:

```text
python -I -B ncd/groundtruth.py --config ncd/configs/exp2_nsl_replay_seed1.json
```

The compact auditable outputs and bounded interpretation are under `results/exp2/nsl-kdd/replay/seed1/`.  The full execution log and model checkpoints are retained in the private research repository and are intentionally not published here.
