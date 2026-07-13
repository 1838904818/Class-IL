"""Router (decision-level) stability for PILoRA.

The isolated-path silent-drift metric (shap_stability.py) deliberately explains
the FROZEN per-family head and excludes the DPMeans router. That makes PILoRA's
0.0% an isolated-path / structural result: the explanation *conditional on
correct routing*. But in deployment a sample is first routed to a family, and
the router's centroids are updated every task — so an old-class sample can be
re-routed to a different family across increments, changing the decision (and
the explanation a deployer actually sees) even though every frozen adapter is
unchanged.

This script measures exactly that gap. For each old class c, on its fixed
explain set, we record the argmax family assigned by the FULL joint decision
(predict(return_routing=True)) at every snapshot, and report:

  router_stability(c, t-1 -> t) = fraction of class-c samples whose assigned
                                  family is UNCHANGED between snapshots
  reroute_rate                  = 1 - router_stability  (decision-level drift)

1.0 means routing is perfectly stable (decision-level == isolated-path);
< 1.0 quantifies how much decision-level explanation drift the router injects
on top of the structurally-zero adapter drift. This provides an empirical
deployment-level measurement of routing stability.

Uses the SAME fixed per-class explain sets and canonical PILoRA runner as
shap_stability.py, so the two metrics are directly comparable. No SHAP — only
routing forward passes — so it is fast.

Usage:
    python -X utf8 -u -m src_v3.router_stability --dataset NSL-KDD --seed 42
    python -X utf8 -u -m src_v3.router_stability --all
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.config import RESULTS_DIR, seed_all
from src.methods.base import make_task_split, subset_by_classes
from src_v3.build_cache import load_cached
from src_v3.shap_stability import DATASETS, run_pilora_snap


def routed_families(agent, X: np.ndarray):
    """Return the argmax family NAME assigned to each row of X by the full
    joint decision (head + DPMeans router) — the deployment routing."""
    _, routing = agent.predict(X, return_routing=True)
    return [r["family"] for r in routing]


def run_one(dataset: str, seed: int, n_explain: int, smoke: bool):
    t_start = time.time()
    seed_all(seed)
    cpt = DATASETS[dataset]
    X_tr, y_tr, X_te, y_te, class_names = load_cached(dataset)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr).astype(np.float32)
    X_te = sc.transform(X_te).astype(np.float32)
    n_classes = len(class_names)
    in_dim = X_tr.shape[1]
    tasks = make_task_split(n_classes, cpt)

    rng = np.random.default_rng(seed)
    explain_sets = {}
    for c in range(n_classes):
        idx = np.where(y_te == c)[0]
        take = min(n_explain, len(idx))
        if take == 0:
            continue
        explain_sets[c] = X_te[rng.choice(idx, size=take, replace=False)]

    seen_at = {i: sorted(set(sum(tasks[: i + 1], []))) for i in range(len(tasks))}
    pilora_pre = 2 if smoke else 8
    pilora_ep = 2 if smoke else 10

    store = {"route": {}}  # store["route"][snapshot][class] = [family names]

    def snapshot(i, agent):
        classes = seen_at[i]
        store["route"][i] = {}
        for c in classes:
            if c not in explain_sets:
                continue
            if agent.class_to_family.get(c) is None:
                continue
            store["route"][i][c] = routed_families(agent, explain_sets[c])
        seen = sorted(store["route"][i].keys())
        print(f"    [router] snapshot t={i}: recorded routing for classes {seen}")

    print(f"{dataset}: tasks={tasks} seed={seed} "
          f"{'[SMOKE]' if smoke else '[FULL]'}  (router-stability)")
    seed_all(seed)
    run_pilora_snap(X_tr, y_tr, tasks, in_dim, n_classes,
                    pilora_pre, pilora_ep, snapshot)

    # ---- transitions: fraction of class-c samples with unchanged family -----
    transitions = []
    all_stab = []
    for i in range(1, len(tasks)):
        prev, curr = store["route"].get(i - 1, {}), store["route"].get(i, {})
        per_class = {}
        for c in prev:
            if c not in curr:
                continue
            a, b = prev[c], curr[c]
            n = min(len(a), len(b))
            if n == 0:
                continue
            same = sum(1 for k in range(n) if a[k] == b[k]) / n
            per_class[str(c)] = {
                "router_stability": round(same, 4),
                "reroute_rate": round(1 - same, 4),
                "n": n,
            }
            all_stab.append(same)
        transitions.append({"from": i - 1, "to": i, "per_class": per_class})

    mean_stab = float(np.mean(all_stab)) if all_stab else float("nan")
    result = {
        "config": {"dataset": dataset, "seed": seed, "tasks": tasks,
                   "n_explain": n_explain, "smoke": smoke,
                   "metric": "fraction of old-class samples whose argmax family "
                             "(full joint decision) is unchanged t-1 -> t"},
        "mean_router_stability": mean_stab,
        "mean_reroute_rate": float(1 - mean_stab) if all_stab else float("nan"),
        "n_oldclass_transition_points": len(all_stab),
        "transitions": transitions,
    }
    ds_slug = dataset.lower().replace("-", "_")
    out = RESULTS_DIR / (f"router_stability_{ds_slug}_smoke.json" if smoke
                         else f"router_stability_{ds_slug}_seed{seed}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  -> mean router stability = {mean_stab:.4f} "
          f"(reroute {1-mean_stab:.4f}), n={len(all_stab)}  "
          f"[{time.time()-t_start:.0f}s]  saved {out.name}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="NSL-KDD", choices=list(DATASETS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", default=None, help="comma list, overrides --seed")
    ap.add_argument("--all", action="store_true",
                    help="all 5 datasets x seeds 0,1,2,3,42")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-explain", type=int, default=200)
    args = ap.parse_args()

    n_explain = 40 if args.smoke else args.n_explain
    if args.all:
        jobs = [(ds, s) for ds in DATASETS for s in (0, 1, 2, 3, 42)]
    else:
        seeds = ([int(x) for x in args.seeds.split(",")] if args.seeds
                 else [args.seed])
        jobs = [(args.dataset, s) for s in seeds]

    summary = {}
    for ds, s in jobs:
        r = run_one(ds, s, n_explain, args.smoke)
        summary.setdefault(ds, []).append(r["mean_router_stability"])

    print("\n" + "=" * 60)
    print(" ROUTER STABILITY SUMMARY (mean over old-class transition points)")
    print("=" * 60)
    for ds, vals in summary.items():
        vals = [v for v in vals if v == v]  # drop nan
        if vals:
            print(f"  {ds:<16} stability={np.mean(vals):.4f}±{np.std(vals):.4f} "
                  f"(reroute {1-np.mean(vals):.4f})  over {len(vals)} seeds")


if __name__ == "__main__":
    main()
