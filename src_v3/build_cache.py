"""Build .npz caches of the standardized arrays for each dataset.

Re-parsing CIC-IDS-2017's 2.8M raw rows on every run is both slow and
memory-heavy (it can OOM/segfault once torch+shap are already resident).
Caching the post-loader (X_tr, y_tr, X_te, y_te) arrays once removes both
problems and is bit-identical to the live loader output.

Run standalone (low memory: only the loader is imported per dataset):
    python -X utf8 -u -m src_v3.build_cache
    python -X utf8 -u -m src_v3.build_cache --datasets CIC-IDS-2017
"""
import argparse
import gc

import numpy as np

from src.config import RESULTS_DIR

CACHE_DIR = RESULTS_DIR.parent / "datasets" / "_cache"

LOADERS = {
    "NSL-KDD": ("src.data.nslkdd", "load_nslkdd"),
    "UNSW-NB15": ("src.data.unsw_nb15", "load_unsw"),
    "CIC-IDS-2017": ("src.data.cic_ids_2017", "load_cicids_2017"),
    "CIC-IDS-2018": ("src.data.cic_ids_2018", "load_cicids_2018"),
    "NF-ToN-IoT-v2": ("src.data.nf_ton_iot_v2", "load_nf_ton_iot_v2"),
}


def cache_path(name):
    return CACHE_DIR / f"{name.lower().replace('-', '_')}.npz"


def build(name):
    import importlib
    mod_name, fn_name = LOADERS[name]
    fn = getattr(importlib.import_module(mod_name), fn_name)
    X_tr, y_tr, X_te, y_te, class_names = fn()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = cache_path(name)
    np.savez_compressed(
        out,
        X_tr=X_tr.astype(np.float32), y_tr=y_tr.astype(np.int64),
        X_te=X_te.astype(np.float32), y_te=y_te.astype(np.int64),
        class_names=np.array(class_names, dtype=object),
    )
    print(f"  cached {name}: train{X_tr.shape} test{X_te.shape} -> {out.name} "
          f"({out.stat().st_size/1e6:.1f} MB)")
    del X_tr, y_tr, X_te, y_te
    gc.collect()


def load_cached(name):
    """Return (X_tr, y_tr, X_te, y_te, class_names) from cache; build if absent."""
    p = cache_path(name)
    if not p.exists():
        build(name)
    d = np.load(p, allow_pickle=True)
    return (d["X_tr"], d["y_tr"], d["X_te"], d["y_te"],
            list(d["class_names"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets",
                    default="NSL-KDD,UNSW-NB15,CIC-IDS-2017,CIC-IDS-2018")
    args = ap.parse_args()
    for name in [s.strip() for s in args.datasets.split(",")]:
        print(f"Building cache: {name}")
        build(name)
    print("Done.")


if __name__ == "__main__":
    main()
