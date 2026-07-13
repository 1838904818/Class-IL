"""Forgetting-benchmark extras for the proposal:
  (#1) recent baseline NCM-Frozen on ALL 5 datasets x 5 seeds, to answer
       "is PILoRA's low forgetting just the frozen encoder?";
  (#2) the forgetting table on the new NF-ToN-IoT-v2 (key Phase-I baselines +
       PILoRA) x 5 seeds, for dataset symmetry with the other four.

Uses the cached .npz (fast, no CSV re-parse) + the same StandardScaler as the
canonical benchmark. Saves per (dataset, method, seed) to
results/forgetting_extra_{ds}_{method}_seed{N}.json -- never touches the
canonical {ds}_results.json. Run standalone (no args needed):

    python -X utf8 -u -m run_forgetting_extra            # not a module; see below
    python -X utf8 -u run_forgetting_extra.py
"""
import json
import time

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.config import seed_all, RESULTS_DIR
from src.data import DATASET_LOADERS
from src.methods.base import make_task_split
from src.methods.naive import run_naive
from src.methods.replay import run_replay
from src.methods.replay_dpmeans import run_replay_dpmeans
from src.methods.icarl import run_icarl
from src.methods.ncm_frozen import run_ncm_frozen
from src.metrics.accuracy import avg_accuracy, avg_forgetting
from src_v2.methods.pilora import run_pilora
from src_v3.build_cache import load_cached

SEEDS = [42, 0, 1, 2, 3]
ALL_DS = ["NSL-KDD", "UNSW-NB15", "CIC-IDS-2017", "CIC-IDS-2018", "NF-ToN-IoT-v2"]
# NF-ToN gets the full comparison set; the other 4 only need the new baseline
# (their canonical Phase-I + PILoRA numbers already exist).
NFTON_BASELINES = {
    "Naive": run_naive, "Replay": run_replay,
    "Replay-DPMeans": run_replay_dpmeans, "iCaRL": run_icarl,
    "NCM-Frozen": run_ncm_frozen,
}


def slug(s):
    return s.lower().replace("-", "_")


def forgetting(acc_matrix):
    T = acc_matrix.shape[0]
    af = []
    for j in range(T - 1):
        col = acc_matrix[:T - 1, j]
        if np.isnan(col).all():
            continue
        fv = acc_matrix[T - 1, j]
        if not np.isnan(fv):
            af.append(float(np.nanmax(col)) - fv)
    return float(np.mean(af)) if af else 0.0


def save(ds, method, seed, acc_matrix):
    out = RESULTS_DIR / f"forgetting_extra_{slug(ds)}_{slug(method)}_seed{seed}.json"
    json.dump({"dataset": ds, "method": method, "seed": seed,
               "avg_accuracy": avg_accuracy(acc_matrix),
               "avg_forgetting": avg_forgetting(acc_matrix),
               "forgetting_peakdef": forgetting(acc_matrix),
               "acc_matrix": acc_matrix.tolist()}, open(out, "w"), indent=2)
    print(f"  saved {out.name}: acc={avg_accuracy(acc_matrix):.4f} "
          f"forget={avg_forgetting(acc_matrix):.4f}")


def run_method(name, fn, X_tr, y_tr, X_te, y_te, tasks, in_dim, n_classes, seed):
    seed_all(seed)
    if name == "PILoRA":
        am, _ = run_pilora(X_tr, y_tr, X_te, y_te, tasks=tasks, in_dim=in_dim,
                           n_classes=n_classes, d_model=64, n_layers=2,
                           chunk_size=8, lora_rank=8, pretrain_epochs=5,
                           pretrain_samples=None, epochs_per_task=10,
                           exemplar_capacity=50, verbose=False)
    else:
        am, _ = fn(X_tr, y_tr, X_te, y_te, tasks, in_dim, n_classes)
    return am


def main():
    t_all = time.time()
    # cache (X,y) per dataset once
    for ds in ALL_DS:
        cpt = DATASET_LOADERS[ds][1]
        Xtr0, ytr, Xte0, yte, cls = load_cached(ds)
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr0).astype(np.float32)
        Xte = sc.transform(Xte0).astype(np.float32)
        tasks = make_task_split(len(cls), cpt)
        in_dim, n_classes = Xtr.shape[1], len(cls)
        # which methods for this dataset
        jobs = {"NCM-Frozen": run_ncm_frozen}
        if ds == "NF-ToN-IoT-v2":
            jobs = dict(NFTON_BASELINES)
            jobs["PILoRA"] = None  # handled specially
        print(f"\n=== {ds}  tasks={tasks}  methods={list(jobs)} ===")
        for method, fn in jobs.items():
            for seed in SEEDS:
                out = RESULTS_DIR / f"forgetting_extra_{slug(ds)}_{slug(method)}_seed{seed}.json"
                if out.exists():
                    print(f"  [skip exists] {out.name}")
                    continue
                t0 = time.time()
                try:
                    am = run_method(method, fn, Xtr, ytr, Xte, yte,
                                    tasks, in_dim, n_classes, seed)
                    save(ds, method, seed, am)
                    print(f"    {method} seed={seed} ({time.time()-t0:.0f}s)")
                except Exception as e:
                    print(f"    [ERROR] {ds} {method} seed={seed}: "
                          f"{type(e).__name__}: {e}")
    print(f"\nALL FORGETTING-EXTRA DONE ({time.time()-t_all:.0f}s)")


if __name__ == "__main__":
    main()
