"""Decision-level explanation stability and oracle task-ID ablation.

This experiment extends the isolated-path silent-drift analysis in two ways:
  (1) extend the stability measurement to the FULL decision function — head +
      DPMeans router — so PILoRA's 0% is not tautological by construction;
  (2) run the oracle task-ID ablation in the same batch, to bound the
      Class-IL -> Task-IL gap.

(1) PILoRADecisionScore explains the JOINT score of class c's family:
        joint_c = head_weight * P(family_c | x) + router_weight * z(router_c)
    where the router z-normalisation runs over ALL current families. So even
    though class c's own head and centroids are frozen, joint_c (and its SHAP
    attribution) shifts as new families are added and change the normalisation
    and the inter-family competition — that shift IS the decision-level drift
    the isolated-path metric misses. Mirrors PILoRAAgent.predict (head_weight
    1.0, router_weight 0.5, softmax_prob calibration).

(2) Oracle ablation: per increment, for each seen class, compare
      actual-routing accuracy  (agent.predict, DPMeans routes the sample)
      oracle-routing accuracy   (route to the TRUE family, then its head)
    The gap is exactly the error the learned router imposes; if it is small,
    the setting is close to Task-IL (router ~ a good task-ID estimator).

Outputs results/decision_stability_{dataset}_seed{N}.json with, per method/
PILoRA: decision-level transitions (cosine/tau/Jaccard + acc_change) and the
per-snapshot oracle-vs-actual accuracy gap.

Usage:
    python -X utf8 -u -m src_v3.decision_stability --dataset NSL-KDD --seed 42
    python -X utf8 -u -m src_v3.decision_stability --smoke --dataset NSL-KDD
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.config import RESULTS_DIR, seed_all
from src.methods.base import make_task_split, subset_by_classes
from src_v3.build_cache import load_cached
from src_v3.shap_stability import (
    DATASETS, TOP_K,
    attribution_vector, stability_metrics, per_class_accuracy,
    run_pilora_snap,
)


class PILoRADecisionScore(nn.Module):
    """Joint head+router score for one family (the full decision-function path)."""

    def __init__(self, agent, target_family, head_weight=1.0, router_weight=0.5):
        super().__init__()
        self.agent = agent
        self.target = target_family
        self.hw, self.rw = head_weight, router_weight
        dev = next(agent.encoder.parameters()).device
        self.families = list(agent.pool.families)
        self.cents = {}
        for f in self.families:
            c = agent.router.centroids.get(f)
            if c is not None and len(c) > 0:
                self.cents[f] = torch.from_numpy(
                    np.asarray(c, dtype=np.float32)).to(dev)

    def forward(self, x):
        a = self.agent
        emb = a.encoder((x - a.feature_mean) / a.feature_std)  # (B, d)
        rs = []
        for f in self.families:
            c = self.cents.get(f)
            if c is None:
                rs.append(torch.full((emb.shape[0],), -1e9, device=emb.device))
            else:
                rs.append(-torch.cdist(emb, c).min(dim=1).values)  # (B,)
        rs = torch.stack(rs, dim=1)                                # (B, F)
        rz = (rs - rs.mean(dim=1, keepdim=True)) / (rs.std(dim=1, keepdim=True) + 1e-8)
        ti = self.families.index(self.target)
        head_prob = torch.softmax(a.pool.heads[self.target](emb), dim=1)[:, 1]
        return (self.hw * head_prob + self.rw * rz[:, ti]).unsqueeze(1)


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
    pilora_pre = 2 if args.smoke else 8
    pilora_ep = 2 if args.smoke else 10

    t0 = time.time()
    seed_all(args.seed)
    cpt = DATASETS[args.dataset]
    X_tr, y_tr, X_te, y_te, class_names = load_cached(args.dataset)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr).astype(np.float32)
    X_te = sc.transform(X_te).astype(np.float32)
    n_classes = len(class_names)
    in_dim = X_tr.shape[1]
    tasks = make_task_split(n_classes, cpt)
    print(f"{args.dataset}: tasks={tasks} seed={args.seed} "
          f"{'[SMOKE]' if args.smoke else '[FULL]'}  (decision-level + oracle)")

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
    seen_at = {i: sorted(set(sum(tasks[: i + 1], []))) for i in range(len(tasks))}

    store = {"attr": {}, "acc_actual": {}, "acc_oracle": {}}

    def oracle_acc(agent, c):
        """argmax==1 rate of class c's TRUE family head on class-c test rows."""
        fam = agent.class_to_family.get(c)
        if fam is None or fam not in agent.pool.heads:
            return None
        idx = np.where(y_te == c)[0]
        if len(idx) == 0:
            return None
        emb_np = agent.embed(X_te[idx])
        dev = next(agent.encoder.parameters()).device
        with torch.no_grad():
            emb = torch.from_numpy(emb_np.astype(np.float32)).to(dev)
            pred = agent.pool.heads[fam](emb).argmax(1).cpu().numpy()
        return float((pred == 1).mean())

    def snapshot(i, agent):
        classes = seen_at[i]
        store["attr"][i] = {}
        actual = per_class_accuracy(lambda X: agent.predict(X), X_te, y_te, classes)
        store["acc_actual"][i] = actual
        store["acc_oracle"][i] = {}
        for c in classes:
            if c not in explain_sets:
                continue
            fam = agent.class_to_family.get(c)
            if fam is None or fam not in agent.pool.heads:
                continue
            sm = PILoRADecisionScore(agent, fam)
            store["attr"][i][c] = attribution_vector(
                sm, explain_sets[c], background, shap_ns)
            store["acc_oracle"][i][c] = oracle_acc(agent, c)
        gaps = {c: round((store["acc_oracle"][i].get(c) or 0)
                         - actual.get(c, 0), 3)
                for c in store["attr"][i]}
        print(f"    [decision] t={i}  oracle-actual acc gap={gaps}")

    seed_all(args.seed)
    run_pilora_snap(X_tr, y_tr, tasks, in_dim, n_classes,
                    pilora_pre, pilora_ep, snapshot)

    # decision-level transitions (t-1 vs t) for PILoRA
    transitions = []
    for i in range(1, len(tasks)):
        per_class = {}
        for c in store["attr"].get(i - 1, {}):
            if c not in store["attr"].get(i, {}):
                continue
            m = stability_metrics(store["attr"][i - 1][c], store["attr"][i][c])
            a_prev = store["acc_actual"][i - 1].get(c)
            a_curr = store["acc_actual"][i].get(c)
            m["acc_change"] = (None if a_prev is None or a_curr is None
                               else round(a_curr - a_prev, 4))
            per_class[str(c)] = m
        transitions.append({"from": i - 1, "to": i, "per_class": per_class})

    # oracle vs actual accuracy per snapshot (the Class-IL -> Task-IL gap)
    oracle_summary = {}
    for i in store["acc_oracle"]:
        rows = {}
        for c in store["acc_oracle"][i]:
            o = store["acc_oracle"][i].get(c)
            a = store["acc_actual"][i].get(c)
            if o is not None and a is not None:
                rows[str(c)] = {"oracle": round(o, 4), "actual": round(a, 4),
                                "router_cost": round(o - a, 4)}
        oracle_summary[str(i)] = rows

    result = {
        "config": {"dataset": args.dataset, "seed": args.seed, "tasks": tasks,
                   "smoke": args.smoke, "top_k": TOP_K,
                   "score": "decision-level joint head+router (PILoRADecisionScore)",
                   "head_weight": 1.0, "router_weight": 0.5},
        "decision_level_transitions": transitions,
        "oracle_vs_actual": oracle_summary,
    }
    ds_slug = args.dataset.lower().replace("-", "_")
    out = RESULTS_DIR / (f"decision_stability_{ds_slug}_smoke.json" if args.smoke
                         else f"decision_stability_{ds_slug}_seed{args.seed}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    # console summary
    all_jac = [m["top15_jaccard"] for tr in transitions
               for m in tr["per_class"].values()]
    all_cost = [r["router_cost"] for s in oracle_summary.values()
                for r in s.values()]
    print("\n" + "=" * 64)
    print(f"  decision-level mean top15-Jaccard (PILoRA): "
          f"{np.mean(all_jac):.4f}" if all_jac else "  (no transitions)")
    print(f"  mean oracle-routing accuracy gain (router cost): "
          f"{np.mean(all_cost):+.4f}" if all_cost else "  (no oracle pts)")
    print(f"  Saved: {out}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
