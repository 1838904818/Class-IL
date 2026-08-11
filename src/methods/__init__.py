"""Continual learning methods.

Each `run_*` function returns (acc_matrix, model).
The METHODS dict at the bottom is what main.py iterates over.
"""
from src.methods.joint import run_joint
from src.methods.naive import run_naive
from src.methods.ewc import run_ewc
from src.methods.lwf import run_lwf
from src.methods.replay import run_replay
from src.methods.replay_dpmeans import run_replay_dpmeans
from src.methods.replay_herding import run_replay_herding
from src.methods.icarl import run_icarl
from src.methods.ncm_frozen import run_ncm_frozen

# Master registry — methods can be enabled/disabled here without touching main.py
METHODS = {
    "Joint (upper bound)":   run_joint,
    "Naive (lower bound)":   run_naive,
    "EWC":                   run_ewc,
    "LwF":                   run_lwf,
    "Replay":                run_replay,
    "Replay-Herding":        run_replay_herding,
    "Replay-DPMeans":        run_replay_dpmeans,
    "iCaRL":                 run_icarl,
    "NCM-Frozen (2023)":     run_ncm_frozen,
}

__all__ = ["METHODS"]
