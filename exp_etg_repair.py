# -*- coding: utf-8 -*-
"""ETG repair experiment and gate evaluation.

This experiment evaluates whether a training penalty can restore a DRIFTED
family's certified rationale and measures the associated accuracy cost.

For each family that is DRIFTED at the end of the stream (admitted at cert, but final occlusion
top-15 Jaccard vs its certified r_c < 0.70), we attempt a REPAIR: fine-tune the (shared) model on
the replay buffer + an occlusion-anchor that directly trains the model to RELY on the certified
features r_c (deleting an r_c feature should drop the class-c probability). Then we re-run ETG's
own gate: re-pass iff Jaccard(r_c, occ_now) >= 0.70 AND rationale-mass > random-null.
We report, per family and per repair strength lambda: did it re-certify, and the GLOBAL accuracy cost.
lambda=1 (gentle) vs lambda=8 (strong) traces the cost curve.
"""
import sys, copy, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from src.config import seed_all
from src.data import DATASET_LOADERS
from src.methods.base import make_task_split, subset_by_classes

DS = sys.argv[1] if len(sys.argv) > 1 else "NSL-KDD"
SEED, EPOCHS, K_BUF, H, N_PROBE, TOPK, R_CTRL = 42, 6, 50, 128, 100, 15, 50
REPAIR_STEPS, MARGIN, LAMBDAS = 150, 0.10, [0.0, 1.0, 8.0]   # lam=0 = CONTROL: extra training, NO anchor


class Net(nn.Module):
    def __init__(s, d, n, h=H):
        super().__init__(); s.enc = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU()); s.head = nn.Linear(h, n)
    def forward(s, x): return s.head(s.enc(x))


def occ_attr(model, X, c):
    with torch.no_grad():
        base = torch.softmax(model(X), 1)[:, c]; imp = np.zeros(X.shape[1], np.float32)
        for j in range(X.shape[1]):
            Xj = X.clone(); Xj[:, j] = 0.0
            imp[j] = float((base - torch.softmax(model(Xj), 1)[:, c]).mean())
    return imp


def top_k(v, k=TOPK): return set(np.argsort(v)[-k:].tolist())
def jac(a, b): return len(a & b) / len(a | b)


def del_gap(model, X, c, feats):
    feats = list(feats)
    if not feats: return float("nan")
    with torch.no_grad():
        full = torch.softmax(model(X), 1)[:, c]
        Xd = X.clone(); Xd[:, feats] = 0.0
        return float((full - torch.softmax(model(Xd), 1)[:, c]).mean())


def null_p95(model, X, c, seed):
    rng = np.random.RandomState(seed)
    return float(np.percentile([del_gap(model, X, c, rng.choice(X.shape[1], TOPK, replace=False)) for _ in range(R_CTRL)], 95))


def anchor_loss(model, X, c, rc):
    """differentiable: train the model to RELY on each certified feature j in rc
       (deleting j should drop class-c prob by >= MARGIN)."""
    base = torch.softmax(model(X), 1)[:, c]
    terms = []
    for j in rc:
        Xj = X.clone(); Xj[:, j] = 0.0
        drop = base - torch.softmax(model(Xj), 1)[:, c]
        terms.append(torch.relu(MARGIN - drop).mean())
    return torch.stack(terms).mean()


def global_acc(model, Xte, yte):
    with torch.no_grad():
        return float((model(torch.from_numpy(Xte.astype(np.float32))).argmax(1).cpu().numpy() == yte).mean())


loader, cpt = DATASET_LOADERS[DS]; seed_all(SEED)
Xtr, ytr, Xte, yte, classes = loader()
sc = StandardScaler(); Xtr = sc.fit_transform(Xtr).astype(np.float32); Xte = sc.transform(Xte).astype(np.float32)
d, ncl = Xtr.shape[1], len(classes); tasks = make_task_split(ncl, cpt)

# ---- train incremental stream, capture certified r_c + buffer ----
seed_all(SEED); model = Net(d, ncl); buf_X, buf_y, seen = [], [], []
cert_rc, cert_null, probe, admitted = {}, {}, {}, {}
for i, ti in enumerate(tasks):
    Xi, yi = subset_by_classes(Xtr, ytr, ti)
    Xp, yp = (np.vstack([Xi]+buf_X), np.concatenate([yi]+buf_y)) if buf_X else (Xi, yi)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    Xt, yt = torch.from_numpy(Xp.astype(np.float32)), torch.from_numpy(yp.astype(np.int64))
    model.train()
    for _ in range(EPOCHS):
        pm = torch.randperm(len(Xt))
        for s in range(0, len(Xt), 256):
            b = pm[s:s+256]; loss = F.cross_entropy(model(Xt[b]), yt[b]); opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    for c in ti: seen.append(c)
    for c in ti:
        mc = yi == c
        if mc.sum(): xs = Xi[mc]; sel = np.random.choice(len(xs), min(K_BUF, len(xs)), replace=False); buf_X.append(xs[sel]); buf_y.append(yi[mc][sel])
    for c in seen:
        if c in cert_rc or c not in set(np.unique(yte)): continue
        idx = np.where(yte == c)[0]
        if len(idx) == 0: continue
        probe[c] = torch.from_numpy(Xte[idx[:N_PROBE]].astype(np.float32))
        rc = top_k(occ_attr(model, probe[c], c)); nu = null_p95(model, probe[c], c, 7000 + int(c))
        cert_rc[c] = rc; cert_null[c] = nu
        admitted[c] = del_gap(model, probe[c], c, rc) > nu     # admitted at cert?

# ---- find final-DRIFTED admitted families ----
model.eval(); base_acc = global_acc(model, Xte, yte)
drifted = []
for c in cert_rc:
    if not admitted[c]: continue
    occ_now = top_k(occ_attr(model, probe[c], c))
    if jac(cert_rc[c], occ_now) < 0.70:
        drifted.append(c)
print(f"{DS}: admitted={sum(admitted.values())}  final-DRIFTED families={[classes[c] for c in drifted]}  base_acc={base_acc:.4f}")

# ---- repair each drifted family at each lambda ----
buf_all_X = np.vstack(buf_X).astype(np.float32); buf_all_y = np.concatenate(buf_y).astype(np.int64)
rows = []
for c in drifted:
    rc = cert_rc[c]; Xc = probe[c]
    bx_c = torch.from_numpy(np.vstack(buf_X)[np.concatenate(buf_y) == c].astype(np.float32))
    for lam in LAMBDAS:
        m2 = copy.deepcopy(model); m2.train()
        opt = torch.optim.Adam(m2.parameters(), 5e-4)
        BX, BY = torch.from_numpy(buf_all_X), torch.from_numpy(buf_all_y)
        for step in range(REPAIR_STEPS):
            b = torch.randperm(len(BX))[:256]
            ce = F.cross_entropy(m2(BX[b]), BY[b])
            loss = ce + lam * anchor_loss(m2, bx_c, c, rc)
            opt.zero_grad(); loss.backward(); opt.step()
        m2.eval()
        occ_r = top_k(occ_attr(m2, Xc, c)); j_r = jac(rc, occ_r)
        mass_r = del_gap(m2, Xc, c, rc); nu_r = null_p95(m2, Xc, c, 8000 + int(c))
        acc_r = global_acc(m2, Xte, yte)
        repassed = (j_r >= 0.70) and (mass_r > nu_r)
        rows.append((classes[c], lam, j_r, mass_r, nu_r, repassed, base_acc - acc_r))
        print(f"  {classes[c]:12s} lam={lam:>4}: Jaccard {jac(rc, top_k(occ_attr(model, Xc, c))):.2f}->{j_r:.2f}  "
              f"mass {mass_r:.3f} vs null {nu_r:.3f}  re-certified={repassed}  ACC COST={base_acc-acc_r:+.4f}")

import json
json.dump({"dataset": DS, "base_acc": round(base_acc, 4), "drifted": [classes[c] for c in drifted],
           "repairs": [{"family": nm, "lambda": lam, "jaccard_after": round(j, 3), "mass_after": round(m, 4),
                        "null_after": round(nu, 4), "re_certified": bool(rp), "acc_cost": round(ac, 4)}
                       for nm, lam, j, m, nu, rp, ac in rows]},
          open(f"results/etgrep_{DS.lower().replace('-','_')}.json", "w"), indent=2)
print(f"saved results/etgrep_{DS.lower().replace('-','_')}.json")
