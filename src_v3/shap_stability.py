"""SHAP explanation-stability pilot on NSL-KDD (Phase III primary contribution).

Measures per-class attribution stability ACROSS the Class-IL task sequence,
self-referenced increment-to-increment (t vs t-1) — no joint-training oracle.

Protocol decisions (these become the paper's protocol section):
  P1. Attribution method: shap.GradientExplainer (expected gradients) — one
      method used consistently for all models (MLP baselines AND the PILoRA
      differentiable head-path) so stability differences come from the MODEL,
      not the explainer.
  P2. Fixed explanation sets: for each class c, the SAME (seeded) sample of up
      to N_EXPLAIN test samples is explained at every snapshot. Model change is
      the only source of attribution change.
  P3. Fixed background: B_BACKGROUND samples drawn once from TASK-0 TRAINING
      data. Available from t=0 (no future-data leak) and never changes, so the
      reference distribution is constant across snapshots.
  P4. Per-class global attribution vector at snapshot t: mean(|SHAP|) over the
      class's explanation set -> one (n_features,) vector per class per t.
  P5. Stability metrics between consecutive snapshots (classes seen by t-1):
        - cosine similarity of attribution vectors
        - Kendall's tau over the full feature ranking
        - top-k Jaccard of the top-K feature sets (K=15)
  P6. Per-class test accuracy recorded at every snapshot -> enables the
      forgetting-vs-explanation-drift decomposition analysis.
  P7. PILoRA caveat (pilot): we explain the differentiable path
      normalize -> frozen encoder -> family LoRA head scalar for the family of
      the explained class. The DPMeans router contribution (non-differentiable
      centroid distances) is NOT inside the explained function; documented as a
      pilot limitation.

Methods compared: Naive (pathological-drift anchor), Replay 200/class (strong
rehearsal baseline), PILoRA-IDS canonical v0.5 config (hypothesis: parameter
isolation -> more stable old-class explanations).

Usage:
    python -X utf8 -u -m src_v3.shap_stability            # full pilot (~30-60 min)
    python -X utf8 -u -m src_v3.shap_stability --smoke    # wiring check (~3 min)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import shap
from scipy import stats as sstats
from sklearn.preprocessing import StandardScaler

from src.config import RESULTS_DIR, REPLAY_BUFFER_PER_CLASS, seed_all
from src.methods.base import build_model, make_task_split, subset_by_classes, train_one_task
from src.methods.replay_dpmeans import _dpmeans_select, DPMEANS_LAMBDA
from src_v2.methods.pilora import PILoRAAgent
from src_v3.build_cache import load_cached  # .npz cache: fast + low-memory load

TOP_K = 15

# GPU if available (RTX 4060 Ti); all canonical code infers device from the
# module, so CPU-only environments behave exactly as before.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# name -> classes_per_task. Arrays come from the cached loader (build_cache),
# which is bit-identical to the live loaders but avoids re-parsing 2.8M raw
# CIC-2017 rows (the source of the prior load-time OOM segfault).
DATASETS = {
    "NSL-KDD": 1,
    "UNSW-NB15": 2,
    "CIC-IDS-2017": 2,
    "CIC-IDS-2018": 2,
    "NF-ToN-IoT-v2": 2,
}


# ---------------------------------------------------------------------------
# Single-output wrappers (bullet-proof against shap multi-output API quirks)
# ---------------------------------------------------------------------------
class MLPClassScore(nn.Module):
    """Logit of one class from a Phase-I MLP."""

    def __init__(self, model: nn.Module, class_id: int):
        super().__init__()
        self.model = model
        self.class_id = class_id

    def forward(self, x):
        return self.model(x)[:, self.class_id:self.class_id + 1]


class PILoRAFamilyScore(nn.Module):
    """Differentiable scalar confidence of one family: norm -> encoder -> head.

    Mirrors PILoRAAgent.predict's head-path (head logits[:,1]-logits[:,0]);
    router distance term intentionally excluded (protocol P7).
    """

    def __init__(self, agent: PILoRAAgent, family: str):
        super().__init__()
        self.agent = agent
        self.family = family

    def forward(self, x):
        xn = (x - self.agent.feature_mean) / self.agent.feature_std
        emb = self.agent.encoder(xn)
        logits = self.agent.pool.heads[self.family](emb)  # (B, 2)
        return (logits[:, 1] - logits[:, 0]).unsqueeze(1)


# ---------------------------------------------------------------------------
# Attribution + metrics
# ---------------------------------------------------------------------------
def attribution_vector(score_model: nn.Module, X_explain: np.ndarray,
                       background: np.ndarray, nsamples: int) -> np.ndarray:
    """mean(|SHAP|) per feature for `score_model` over X_explain (protocol P4)."""
    score_model.eval()
    try:
        dev = next(score_model.parameters()).device
    except StopIteration:
        dev = torch.device("cpu")
    bg = torch.from_numpy(background.astype(np.float32)).to(dev)
    xe = torch.from_numpy(X_explain.astype(np.float32)).to(dev)
    explainer = shap.GradientExplainer(score_model, bg)
    sv = explainer.shap_values(xe, nsamples=nsamples)
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.asarray(sv)
    sv = sv.reshape(sv.shape[0], -1)  # (N, F) regardless of trailing output dim
    return np.abs(sv).mean(axis=0)


def stability_metrics(prev: np.ndarray, curr: np.ndarray) -> dict:
    """Protocol P5 metrics between consecutive attribution vectors."""
    na, nb = np.linalg.norm(prev), np.linalg.norm(curr)
    cosine = float(prev @ curr / (na * nb)) if na > 0 and nb > 0 else float("nan")
    tau, _ = sstats.kendalltau(prev, curr)
    top_prev = set(np.argsort(prev)[-TOP_K:])
    top_curr = set(np.argsort(curr)[-TOP_K:])
    jaccard = len(top_prev & top_curr) / len(top_prev | top_curr)
    return {"cosine": cosine, "kendall_tau": float(tau), f"top{TOP_K}_jaccard": jaccard}


def per_class_accuracy(predict_fn, X_te, y_te, classes):
    out = {}
    for c in classes:
        m = y_te == c
        if m.sum() == 0:
            continue
        out[int(c)] = float((predict_fn(X_te[m]) == c).mean())
    return out


# ---------------------------------------------------------------------------
# Method runners with per-task snapshot callback
# ---------------------------------------------------------------------------
def run_naive(X_tr, y_tr, tasks, in_dim, n_classes, epochs, snapshot):
    model = build_model(in_dim, n_classes).to(DEVICE)
    for i, task in enumerate(tasks):
        Xi, yi = subset_by_classes(X_tr, y_tr, task)
        train_one_task(model, Xi, yi, epochs=epochs)
        snapshot(i, model)


def run_replay(X_tr, y_tr, tasks, in_dim, n_classes, epochs, snapshot,
               buffer_per_class=REPLAY_BUFFER_PER_CLASS):
    model = build_model(in_dim, n_classes).to(DEVICE)
    buf_X, buf_y = [], []
    for i, task in enumerate(tasks):
        Xi, yi = subset_by_classes(X_tr, y_tr, task)
        Xc = np.vstack([Xi] + buf_X) if buf_X else Xi
        yc = np.concatenate([yi] + buf_y) if buf_y else yi
        train_one_task(model, Xc, yc, epochs=epochs)
        for c in task:
            m = yi == c
            if m.sum() == 0:
                continue
            idx = np.random.choice(np.where(m)[0],
                                   size=min(buffer_per_class, int(m.sum())),
                                   replace=False)
            buf_X.append(Xi[idx])
            buf_y.append(yi[idx])
        snapshot(i, model)


def run_replay_dpmeans(X_tr, y_tr, tasks, in_dim, n_classes, epochs, snapshot,
                       buffer_per_class=REPLAY_BUFFER_PER_CLASS):
    """Replay with DP-Means exemplar selection (Phase I contribution) + snapshots."""
    model = build_model(in_dim, n_classes).to(DEVICE)
    buf_X, buf_y = [], []
    for i, task in enumerate(tasks):
        Xi, yi = subset_by_classes(X_tr, y_tr, task)
        Xc = np.vstack([Xi] + buf_X) if buf_X else Xi
        yc = np.concatenate([yi] + buf_y) if buf_y else yi
        train_one_task(model, Xc, yc, epochs=epochs)
        model.eval()
        with torch.no_grad():
            feats_all = model.extract_features(
                torch.from_numpy(Xi.astype(np.float32)).to(DEVICE)).cpu().numpy()
        for c in task:
            m = yi == c
            if m.sum() == 0:
                continue
            sel_local = _dpmeans_select(feats_all[m], buffer_per_class,
                                        lam=DPMEANS_LAMBDA)
            sel_global = np.where(m)[0][sel_local]
            buf_X.append(Xi[sel_global])
            buf_y.append(yi[sel_global])
        snapshot(i, model)


def run_pilora_snap(X_tr, y_tr, tasks, in_dim, n_classes,
                    pretrain_epochs, epochs_per_task, snapshot):
    """Canonical v0.5 config (mlp encoder, d=128, n_layers=2, rank=8, focal)."""
    agent = PILoRAAgent(n_features=in_dim, d_model=128, n_layers=2,
                        lora_rank=8, exemplar_capacity=50, encoder_type="mlp")
    agent.fit_input_stats(X_tr)   # buffers assigned on CPU here...
    agent = agent.to(DEVICE)      # ...then everything moves together
    X_t0, y_t0 = subset_by_classes(X_tr, y_tr, tasks[0])
    agent.supervised_pretrain_encoder(X_t0, y_t0, n_classes=n_classes,
                                      epochs=pretrain_epochs, verbose=False)
    agent.freeze_encoder()
    for i, task in enumerate(tasks):
        Xi, yi = subset_by_classes(X_tr, y_tr, task)
        agent.train_task(Xi, yi, epochs=epochs_per_task, verbose=False)
        snapshot(i, agent)


# ---------------------------------------------------------------------------
# Main pilot
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny run to verify wiring")
    ap.add_argument("--dataset", default="NSL-KDD", choices=list(DATASETS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-explain", type=int, default=200)
    ap.add_argument("--n-background", type=int, default=100)
    ap.add_argument("--shap-nsamples", type=int, default=200)
    args = ap.parse_args()

    n_explain = 40 if args.smoke else args.n_explain
    n_background = 30 if args.smoke else args.n_background
    shap_ns = 50 if args.smoke else args.shap_nsamples
    base_epochs = 2 if args.smoke else 8          # Phase-I EPOCHS_PER_TASK
    pilora_pre = 2 if args.smoke else 8           # canonical pretrain
    pilora_ep = 2 if args.smoke else 10           # canonical epochs/task

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
    print(f"{args.dataset}: train={X_tr.shape} test={X_te.shape} tasks={tasks}"
          f"  seed={args.seed}  {'[SMOKE]' if args.smoke else '[FULL]'}")

    # ---- protocol P2/P3: fixed sets, drawn once with a dedicated RNG
    rng = np.random.default_rng(args.seed)
    explain_sets = {}
    for c in range(n_classes):
        idx = np.where(y_te == c)[0]
        take = min(n_explain, len(idx))
        if take == 0:
            print(f"  explain set class {c} ({class_names[c]}): EMPTY — skipped")
            continue
        explain_sets[c] = X_te[rng.choice(idx, size=take, replace=False)]
        print(f"  explain set class {c} ({class_names[c]}): {take} samples")
    X_t0, _ = subset_by_classes(X_tr, y_tr, tasks[0])
    background = X_t0[rng.choice(len(X_t0), size=min(n_background, len(X_t0)),
                                 replace=False)]
    print(f"  background: {len(background)} Task-0 train samples (fixed)")

    seen_at = {i: sorted(set(sum(tasks[: i + 1], []))) for i in range(len(tasks))}
    results = {"config": {
        "dataset": args.dataset, "seed": args.seed, "tasks": tasks,
        "n_explain": n_explain, "n_background": n_background,
        "shap_nsamples": shap_ns, "top_k": TOP_K, "smoke": args.smoke,
        "attribution": "shap.GradientExplainer mean|SHAP|",
        "background": "fixed Task-0 train sample",
        "pilora_note": "explains head-path only; router excluded (P7)",
    }, "methods": {}}

    def make_snapshot_recorder(method_name, kind):
        store = {"attr": {}, "acc": {}}

        def snapshot(i, model_or_agent):
            classes = seen_at[i]
            if kind == "mlp":
                model_or_agent.eval()
                def predict_fn(X):
                    with torch.no_grad():
                        return model_or_agent(torch.from_numpy(
                            X.astype(np.float32)).to(DEVICE)
                        ).cpu().numpy().argmax(1)
            else:
                predict_fn = lambda X: model_or_agent.predict(X)
            store["acc"][i] = per_class_accuracy(predict_fn, X_te, y_te, classes)
            store["attr"][i] = {}
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
                store["attr"][i][c] = attribution_vector(
                    sm, explain_sets[c], background, shap_ns)
            print(f"    [{method_name}] snapshot t={i}: "
                  f"acc={ {k: round(v, 3) for k, v in store['acc'][i].items()} }")
        return store, snapshot

    # ---- run the four methods (re-seeded each for comparability)
    runs = {}
    print("\n[1/4] Naive (drift anchor)")
    seed_all(args.seed)
    runs["Naive"], snap = make_snapshot_recorder("Naive", "mlp")
    run_naive(X_tr, y_tr, tasks, in_dim, n_classes, base_epochs, snap)

    print("\n[2/4] Replay 200/class")
    seed_all(args.seed)
    runs["Replay"], snap = make_snapshot_recorder("Replay", "mlp")
    run_replay(X_tr, y_tr, tasks, in_dim, n_classes, base_epochs, snap)

    print("\n[3/4] Replay-DPMeans (Phase I)")
    seed_all(args.seed)
    runs["Replay-DPMeans"], snap = make_snapshot_recorder("Replay-DPMeans", "mlp")
    run_replay_dpmeans(X_tr, y_tr, tasks, in_dim, n_classes, base_epochs, snap)

    print("\n[4/4] PILoRA-IDS (canonical v0.5)")
    seed_all(args.seed)
    runs["PILoRA"], snap = make_snapshot_recorder("PILoRA", "pilora")
    run_pilora_snap(X_tr, y_tr, tasks, in_dim, n_classes,
                    pilora_pre, pilora_ep, snap)

    # ---- stability metrics across consecutive snapshots (protocol P5)
    for method, store in runs.items():
        transitions = []
        for i in range(1, len(tasks)):
            prev_classes = set(store["attr"].get(i - 1, {}))
            per_class = {}
            for c in sorted(prev_classes):
                if c not in store["attr"].get(i, {}):
                    continue
                m = stability_metrics(store["attr"][i - 1][c], store["attr"][i][c])
                acc_prev = store["acc"][i - 1].get(c)
                acc_curr = store["acc"][i].get(c)
                m["acc_change"] = (None if acc_prev is None or acc_curr is None
                                   else round(acc_curr - acc_prev, 4))
                per_class[int(c)] = m
            transitions.append({"from": i - 1, "to": i, "per_class": per_class})
        results["methods"][method] = {
            "transitions": transitions,
            "per_class_acc": {str(t): {str(k): round(v, 4) for k, v in a.items()}
                              for t, a in store["acc"].items()},
        }

    ds_slug = args.dataset.lower().replace("-", "_")
    out = RESULTS_DIR / (
        f"shap_stability_{ds_slug}_smoke.json" if args.smoke
        else f"shap_stability_{ds_slug}_seed{args.seed}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    # ---- summary: mean old-class stability per method
    print("\n" + "=" * 80)
    print(f"{'Method':<10}{'mean cosine':<14}{'mean tau':<12}"
          f"{'mean top%d-Jacc' % TOP_K:<16}{'(old classes, all transitions)'}")
    print("-" * 80)
    for method, r in results["methods"].items():
        cos, tau, jac = [], [], []
        for tr in r["transitions"]:
            for c, m in tr["per_class"].items():
                cos.append(m["cosine"]); tau.append(m["kendall_tau"])
                jac.append(m[f"top{TOP_K}_jaccard"])
        if cos:
            print(f"{method:<10}{np.mean(cos):<14.4f}{np.mean(tau):<12.4f}"
                  f"{np.mean(jac):<16.4f}")
    print(f"\nSaved: {out}   ({time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
