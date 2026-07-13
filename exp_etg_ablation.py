# -*- coding: utf-8 -*-
"""ETG ablation: does the DELETION-FAITHFUL engine matter, or would any attribution do?

At each family's first-learning we run the ADMISSION test two ways:
  (A) OCCLUSION  rationale: top-15 features by deletion importance (ETG's real engine)
  (B) INPUT-GRAD rationale: top-15 features by |input-gradient|  (the cheaper, common choice)
Admission = del_gap(top-15 set) > random-15 p95 null  (SAME null for both).
If the two partitions DIFFER, the deletion-faithful engine is load-bearing (not interchangeable).
"""
import sys, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from src.config import seed_all
from src.data import DATASET_LOADERS
from src.methods.base import make_task_split, subset_by_classes

DS = sys.argv[1] if len(sys.argv) > 1 else "NSL-KDD"
SEED, EPOCHS, K_BUF, H, N_PROBE, TOPK, R_CTRL = 42, 6, 50, 128, 100, 15, 50


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


def ig_attr(model, X, c):
    x = X.clone().requires_grad_(True)
    g = torch.autograd.grad(model(x)[:, c].sum(), x)[0]
    return np.abs(g.mean(0).detach().cpu().numpy())          # |mean input-gradient| for ranking


def top_k(v, k=TOPK): return set(np.argsort(v)[-k:].tolist())


def del_gap(model, X, c, feats):
    feats = list(feats)
    if not feats: return float("nan")
    with torch.no_grad():
        full = torch.softmax(model(X), 1)[:, c]
        Xd = X.clone(); Xd[:, feats] = 0.0
        return float((full - torch.softmax(model(Xd), 1)[:, c]).mean())


loader, cpt = DATASET_LOADERS[DS]; seed_all(SEED)
Xtr, ytr, Xte, yte, classes = loader()
sc = StandardScaler(); Xtr = sc.fit_transform(Xtr).astype(np.float32); Xte = sc.transform(Xte).astype(np.float32)
d, ncl = Xtr.shape[1], len(classes); tasks = make_task_split(ncl, cpt); dev = "cpu"

seed_all(SEED); model = Net(d, ncl).to(dev); buf_X, buf_y, seen, done = [], [], [], set()
rows = []
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
        if c in done: continue
        idx = np.where(yte == c)[0]
        if len(idx) == 0: continue
        done.add(c)
        Xc = torch.from_numpy(Xte[idx[:N_PROBE]].astype(np.float32))
        rng = np.random.RandomState(7000 + int(c))
        null = float(np.percentile([del_gap(model, Xc, c, rng.choice(d, TOPK, replace=False)) for _ in range(R_CTRL)], 95))
        m_occ = del_gap(model, Xc, c, top_k(occ_attr(model, Xc, c)))
        m_ig = del_gap(model, Xc, c, top_k(ig_attr(model, Xc, c)))
        rows.append((classes[c], m_occ, m_occ > null, m_ig, m_ig > null, null))

occ_adm = sum(r[2] for r in rows); ig_adm = sum(r[4] for r in rows)
flips = sum(r[2] != r[4] for r in rows)
print(f"{DS}: families={len(rows)}  OCC admits={occ_adm}  INPUT-GRAD admits={ig_adm}  DECISION FLIPS={flips}")
for nm, mo, ao, mi, ai, nu in rows:
    flag = "  <-- FLIP" if ao != ai else ""
    print(f"  {nm:12s} occ_mass={mo:.3f} {'ADMIT' if ao else 'refuse':6s} | ig_mass={mi:.3f} {'ADMIT' if ai else 'refuse':6s} | null={nu:.3f}{flag}")
import json
json.dump({"dataset": DS, "n": len(rows), "occ_admits": occ_adm, "ig_admits": ig_adm, "flips": flips,
           "rows": [{"family": nm, "occ_mass": round(mo, 4), "occ_admit": bool(ao), "ig_mass": round(mi, 4), "ig_admit": bool(ai), "null": round(nu, 4)} for nm, mo, ao, mi, ai, nu in rows]},
          open(f"results/etgabl_{DS.lower().replace('-','_')}.json", "w"), indent=2)
print(f"saved results/etgabl_{DS.lower().replace('-','_')}.json")
