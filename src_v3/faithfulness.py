"""Faithfulness check for explanation-stability results (Phase III, item 2).

Question this answers: are the "stable" attributions also CORRECT ones?
A degenerate model could produce perfectly stable but meaningless attributions;
stability must therefore be guarded by a faithfulness check.

Metric (deletion-style, in the spirit of Petsiuk et al. 2018):
  For class c at snapshot t, with global attribution vector a_c (mean|SHAP|):
    drop_top  = mean score drop when the TOP-K features by a_c are replaced
                with the fixed Task-0 background mean
    drop_rand = mean score drop when K RANDOM features are replaced
                (averaged over N_RANDOM draws, fixed seed)
    faithfulness_gap = drop_top - drop_rand
  gap > 0  -> the attribution points at features the model genuinely relies on.

Protocol consistency with shap_stability.py:
  - same fixed per-class explanation sets (P2)
  - same fixed Task-0 background; its per-feature MEAN is the replacement value
  - same canonical training configs / runners (imported, not duplicated)

Outputs results/shap_faithfulness_{dataset}_seed{N}.json with per-method,
per-snapshot, per-class: drop_top / drop_rand / faithfulness_gap, plus the
attribution-stability numbers so joint (stability x faithfulness) analysis
needs only this one file.

Usage:
    python -X utf8 -u -m src_v3.faithfulness --dataset NSL-KDD --seed 42
    python -X utf8 -u -m src_v3.faithfulness --smoke --dataset NSL-KDD
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.config import RESULTS_DIR, seed_all
from src.methods.base import make_task_split, subset_by_classes
from src_v3.build_cache import load_cached
from src_v3.shap_stability import (
    DATASETS, TOP_K,
    MLPClassScore, PILoRAFamilyScore,
    attribution_vector, stability_metrics, per_class_accuracy,
    run_naive, run_replay, run_replay_dpmeans, run_pilora_snap,
)

N_RANDOM = 5  # random-K draws to average for drop_rand


def mean_score(score_model: torch.nn.Module, X: np.ndarray) -> float:
    score_model.eval()
    try:
        dev = next(score_model.parameters()).device
    except StopIteration:
        dev = torch.device("cpu")
    with torch.no_grad():
        s = score_model(torch.from_numpy(X.astype(np.float32)).to(dev))
    return float(s.mean())


def faithfulness_gap(score_model, X_explain: np.ndarray, attribution: np.ndarray,
                     replacement: np.ndarray, k: int, rng: np.random.Generator):
    """Deletion-style gap: drop(top-k by attribution) - mean drop(random-k)."""
    n_feat = X_explain.shape[1]
    s0 = mean_score(score_model, X_explain)

    top_idx = np.argsort(attribution)[-k:]
    X_top = X_explain.copy()
    X_top[:, top_idx] = replacement[top_idx]
    drop_top = s0 - mean_score(score_model, X_top)

    drops_rand = []
    for _ in range(N_RANDOM):
        rand_idx = rng.choice(n_feat, size=k, replace=False)
        X_rand = X_explain.copy()
        X_rand[:, rand_idx] = replacement[rand_idx]
        drops_rand.append(s0 - mean_score(score_model, X_rand))
    drop_rand = float(np.mean(drops_rand))

    return {
        "base_score": s0,
        "drop_top": float(drop_top),
        "drop_rand": drop_rand,
        "faithfulness_gap": float(drop_top - drop_rand),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dataset", default="NSL-KDD", choices=list(DATASETS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-explain", type=int, default=200)
    ap.add_argument("--n-background", type=int, default=100)
    ap.add_argument("--shap-nsamples", type=int, default=200)
    args = ap.parse_args()

    n_explain = 40 if args.smoke else args.n_explain
    n_background = 30 if args.smoke else args.n_background
    shap_ns = 50 if args.smoke else args.shap_nsamples
    base_epochs = 2 if args.smoke else 8
    pilora_pre = 2 if args.smoke else 8
    pilora_ep = 2 if args.smoke else 10

    t_start = time.time()
    seed_all(args.seed)
    cpt = DATASETS[args.dataset]
    X_tr, y_tr, X_te, y_te, class_names = load_cached(args.dataset)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr).astype(np.float32)
    X_te = sc.transform(X_te).astype(np.float32)
    n_classes = len(class_names)
    in_dim = X_tr.shape[1]
    tasks = make_task_split(n_classes, cpt)
    print(f"{args.dataset}: train={X_tr.shape} tasks={tasks} seed={args.seed}"
          f"  {'[SMOKE]' if args.smoke else '[FULL]'}  (faithfulness)")

    # fixed sets, identical protocol to shap_stability (P2/P3)
    rng = np.random.default_rng(args.seed)
    explain_sets = {}
    for c in range(n_classes):
        idx = np.where(y_te == c)[0]
        take = min(n_explain, len(idx))
        if take == 0:
            continue
        explain_sets[c] = X_te[rng.choice(idx, size=take, replace=False)]
    X_t0, _ = subset_by_classes(X_tr, y_tr, tasks[0])
    background = X_t0[rng.choice(len(X_t0), size=min(n_background, len(X_t0)),
                                 replace=False)]
    replacement = background.mean(axis=0)  # per-feature replacement values
    rng_faith = np.random.default_rng(args.seed + 1)  # independent random-K draws

    seen_at = {i: sorted(set(sum(tasks[: i + 1], []))) for i in range(len(tasks))}
    results = {"config": {
        "dataset": args.dataset, "seed": args.seed, "tasks": tasks,
        "n_explain": n_explain, "top_k": TOP_K, "n_random": N_RANDOM,
        "shap_nsamples": shap_ns, "smoke": args.smoke,
        "metric": "deletion gap: drop(top-k by mean|SHAP|) - mean drop(random-k), "
                  "replacement = Task-0 background mean",
    }, "methods": {}}

    def make_recorder(method_name, kind):
        store = {"attr": {}, "faith": {}, "acc": {}}

        def snapshot(i, model_or_agent):
            classes = seen_at[i]
            if kind == "mlp":
                model_or_agent.eval()
                _dev = next(model_or_agent.parameters()).device
                def predict_fn(X):
                    with torch.no_grad():
                        return model_or_agent(torch.from_numpy(
                            X.astype(np.float32)).to(_dev)).cpu().numpy().argmax(1)
            else:
                predict_fn = lambda X: model_or_agent.predict(X)
            store["acc"][i] = per_class_accuracy(predict_fn, X_te, y_te, classes)
            store["attr"][i], store["faith"][i] = {}, {}
            for c in classes:
                if c not in explain_sets:
                    continue
                if kind == "mlp":
                    sm = MLPClassScore(model_or_agent, c)
                else:
                    fam = model_or_agent.class_to_family.get(c)
                    if fam is None or fam not in model_or_agent.pool.heads:
                        continue
                    sm = PILoRAFamilyScore(model_or_agent, fam)
                a = attribution_vector(sm, explain_sets[c], background, shap_ns)
                store["attr"][i][c] = a
                store["faith"][i][c] = faithfulness_gap(
                    sm, explain_sets[c], a, replacement, TOP_K, rng_faith)
            gaps = {c: round(v["faithfulness_gap"], 3)
                    for c, v in store["faith"][i].items()}
            print(f"    [{method_name}] t={i} faith_gap={gaps}")
        return store, snapshot

    runs = {}
    print("\n[1/4] Naive")
    seed_all(args.seed)
    runs["Naive"], snap = make_recorder("Naive", "mlp")
    run_naive(X_tr, y_tr, tasks, in_dim, n_classes, base_epochs, snap)

    print("\n[2/4] Replay")
    seed_all(args.seed)
    runs["Replay"], snap = make_recorder("Replay", "mlp")
    run_replay(X_tr, y_tr, tasks, in_dim, n_classes, base_epochs, snap)

    print("\n[3/4] Replay-DPMeans")
    seed_all(args.seed)
    runs["Replay-DPMeans"], snap = make_recorder("Replay-DPMeans", "mlp")
    run_replay_dpmeans(X_tr, y_tr, tasks, in_dim, n_classes, base_epochs, snap)

    print("\n[4/4] PILoRA")
    seed_all(args.seed)
    runs["PILoRA"], snap = make_recorder("PILoRA", "pilora")
    run_pilora_snap(X_tr, y_tr, tasks, in_dim, n_classes,
                    pilora_pre, pilora_ep, snap)

    # serialize: faithfulness per snapshot + stability between snapshots
    for method, store in runs.items():
        snapshots = {}
        for i in sorted(store["faith"]):
            snapshots[str(i)] = {
                str(c): store["faith"][i][c] for c in store["faith"][i]}
        transitions = []
        for i in range(1, len(tasks)):
            per_class = {}
            for c in store["attr"].get(i - 1, {}):
                if c not in store["attr"].get(i, {}):
                    continue
                m = stability_metrics(store["attr"][i - 1][c], store["attr"][i][c])
                acc_prev = store["acc"][i - 1].get(c)
                acc_curr = store["acc"][i].get(c)
                m["acc_change"] = (None if acc_prev is None or acc_curr is None
                                   else round(acc_curr - acc_prev, 4))
                per_class[str(c)] = m
            transitions.append({"from": i - 1, "to": i, "per_class": per_class})
        results["methods"][method] = {
            "faithfulness_per_snapshot": snapshots,
            "transitions": transitions,
        }

    ds_slug = args.dataset.lower().replace("-", "_")
    out = RESULTS_DIR / (
        f"shap_faithfulness_{ds_slug}_smoke.json" if args.smoke
        else f"shap_faithfulness_{ds_slug}_seed{args.seed}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    # summary: per-method mean gap (positive => attributions are faithful)
    print("\n" + "=" * 72)
    print(f"{'Method':<16}{'mean faith_gap':<16}{'% snapshots gap>0':<20}")
    print("-" * 72)
    for method, r in results["methods"].items():
        gaps = [v["faithfulness_gap"]
                for snap_d in r["faithfulness_per_snapshot"].values()
                for v in snap_d.values()]
        if gaps:
            pos = 100 * sum(g > 0 for g in gaps) / len(gaps)
            print(f"{method:<16}{np.mean(gaps):<16.4f}{pos:<20.1f}")
    print(f"\nSaved: {out}   ({time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
