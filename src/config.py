"""
Global configuration — hyperparameters, paths, seeds.
Import this anywhere with: from src.config import *
"""
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch

# Suppress noisy NaN warning from forgetting computation on first tasks
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
DEVICE = torch.device("cpu")

def seed_all(seed: int = SEED) -> None:
    """Seed Python random, numpy, and torch. Call at start of each run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
HIDDEN = 128
BATCH_SIZE = 256
EPOCHS_PER_TASK = 8
LR = 1e-3

# CL method-specific
REPLAY_BUFFER_PER_CLASS = 200
EWC_LAMBDA = 5000.0
LWF_T = 2.0           # distillation temperature
LWF_ALPHA = 1.0       # distillation weight

# DP-Means clustering (paper contribution)
DPMEANS_LAMBDA = None  # auto-set from feature distance if None
DPMEANS_MAX_ITER = 20

# iCaRL
ICARL_T = 2.0
ICARL_USE_NME = True   # nearest-mean-of-exemplars classifier at inference

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# In the repository, this file lives at src/config.py. In the Kaggle package,
# it lives at code/src/config.py while datasets remain beside code/.
CODE_ROOT = Path(__file__).resolve().parent.parent
ROOT = CODE_ROOT.parent if CODE_ROOT.name == "code" else CODE_ROOT
DATA_DIR = Path(os.environ.get("CLASS_IL_DATA_DIR", ROOT / "datasets")).expanduser().resolve()

if "CLASS_IL_RESULTS_DIR" in os.environ:
    RESULTS_DIR = Path(os.environ["CLASS_IL_RESULTS_DIR"]).expanduser().resolve()
elif Path("/kaggle/working").is_dir():
    RESULTS_DIR = Path("/kaggle/working/results")
else:
    RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

# Ensure result dirs exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Apply default seed at import time
# ---------------------------------------------------------------------------
seed_all(SEED)
