# -*- coding: utf-8 -*-
"""Adversarial audit of the ETG repair positive. Two attacks:
  (A) HIDDEN PER-FAMILY DAMAGE: global acc ~0 can hide the wall's usual victim. For each repaired
      family, measure EVERY family's accuracy under lam=0 (train-only) vs lam=8 (train+anchor);
      report the worst OTHER-family drop the anchor causes.
  (B) TEACHING-TO-THE-TEST: the anchor optimises deletion-reliance on r_c, the gate measures it.
      Independent checks on a HELD-OUT probe2 (different test rows from the gate's probe):
        - re-test the deletion gate on probe2 (rules out overfitting to the specific probe rows);
        - INSERTION faithfulness: keep ONLY r_c features (rest=0); does class-c prob recover?
          Insertion is a DIFFERENT operationalisation than the deletion-margin the anchor trained on.
"""
import sys, copy, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from src.config import seed_all
from src.data import DATASET_LOADERS
from src.methods.base import make_task_split, subset_by_classes

DS = sys.argv[1] if len(sys.argv) > 1 else "NSL-KDD"
SEED, EPOCHS, K_BUF, H, N_PROBE, TOPK, R_CTRL = 42, 6, 50, 128, 100, 15, 50
REPAIR_STEPS, MARGIN = 150, 0.10


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
    with torch.no_grad():
        full = torch.softmax(model(X), 1)[:, c]
        Xd = X.clone(); Xd[:, feats] = 0.0
        return float((full - torch.softmax(model(Xd), 1)[:, c]).mean())


def null_p95(model, X, c, seed):
    rng = np.random.RandomState(seed)
    return float(np.percentile([del_gap(model, X, c, rng.choice(X.shape[1], TOPK, replace=False)) for _ in range(R_CTRL)], 95))


def insertion(model, X, c, rc):
    with torch.no_grad():
        Xz = torch.zeros_like(X); Xz[:, list(rc)] = X[:, list(rc)]
        keep_only = float(torch.softmax(model(Xz), 1)[:, c].mean())
        full = float(torch.softmax(model(X), 1)[:, c].mean())
    return keep_only, full


def anchor_loss(model, X, c, rc):
    base = torch.softmax(model(X), 1)[:, c]; terms = []
    for j in rc:
        Xj = X.clone(); Xj[:, j] = 0.0
        terms.append(torch.relu(MARGIN - (base - torch.softmax(model(Xj), 1)[:, c])).mean())
    return torch.stack(terms).mean()


def perfam_acc(model, Xte, yte, ncl):
    with torch.no_grad():
        pred = model(torch.from_numpy(Xte.astype(np.float32))).argmax(1).cpu().numpy()
    return {c: float((pred[yte == c] == c).mean()) for c in range(ncl) if (yte == c).sum()}


def repair(model, buf_all_X, buf_all_y, bx_c, c, rc, lam):
    m2 = copy.deepcopy(model); m2.train(); opt = torch.optim.Adam(m2.parameters(), 5e-4)
    BX, BY = torch.from_numpy(buf_all_X), torch.from_numpy(buf_all_y)
    for _ in range(REPAIR_STEPS):
        b = torch.randperm(len(BX))[:256]
        loss = F.cross_entropy(m2(BX[b]), BY[b]) + (lam * anchor_loss(m2, bx_c, c, rc) if lam > 0 else 0.0)
        opt.zero_grad(); loss.backward(); opt.step()
    m2.eval(); return m2


loader, cpt = DATASET_LOADERS[DS]; seed_all(SEED)
Xtr, ytr, Xte, yte, classes = loader()
sc = StandardScaler(); Xtr = sc.fit_transform(Xtr).astype(np.float32); Xte = sc.transform(Xte).astype(np.float32)
d, ncl = Xtr.shape[1], len(classes); tasks = make_task_split(ncl, cpt)

seed_all(SEED); model = Net(d, ncl); buf_X, buf_y, seen = [], [], []
cert_rc, cert_null, probe, probe2, admitted = {}, {}, {}, {}, {}
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
        if c in cert_rc: continue
        idx = np.where(yte == c)[0]
        if len(idx) < 2: continue
        probe[c] = torch.from_numpy(Xte[idx[:N_PROBE]].astype(np.float32))
        probe2[c] = torch.from_numpy(Xte[idx[N_PROBE:2*N_PROBE]].astype(np.float32)) if len(idx) > N_PROBE else probe[c]
        rc = top_k(occ_attr(model, probe[c], c)); nu = null_p95(model, probe[c], c, 7000 + int(c))
        cert_rc[c] = rc; cert_null[c] = nu; admitted[c] = del_gap(model, probe[c], c, rc) > nu

model.eval()
drifted = [c for c in cert_rc if admitted[c] and jac(cert_rc[c], top_k(occ_attr(model, probe[c], c))) < 0.70]
buf_all_X = np.vstack(buf_X).astype(np.float32); buf_all_y = np.concatenate(buf_y).astype(np.int64)
base_perfam = perfam_acc(model, Xte, yte, ncl)
print(f"{DS}: drifted={[classes[c] for c in drifted]}")
rows = []
for c in drifted:
    rc = cert_rc[c]; bx_c = torch.from_numpy(np.vstack(buf_X)[np.concatenate(buf_y) == c].astype(np.float32))
    m0 = repair(model, buf_all_X, buf_all_y, bx_c, c, rc, 0.0)
    m8 = repair(model, buf_all_X, buf_all_y, bx_c, c, rc, 8.0)
    pf0, pf8 = perfam_acc(m0, Xte, yte, ncl), perfam_acc(m8, Xte, yte, ncl)
    # (A) worst OTHER-family drop caused by the anchor (lam8 vs lam0)
    drops = {classes[f]: pf0[f] - pf8[f] for f in pf8 if f != c}
    worst_fam = max(drops, key=drops.get); worst_drop = drops[worst_fam]
    # (B) held-out probe2 gate + insertion
    j_p2 = jac(rc, top_k(occ_attr(m8, probe2[c], c)))
    mass_p2 = del_gap(m8, probe2[c], c, rc); nu_p2 = null_p95(m8, probe2[c], c, 8500 + int(c))
    keep, full = insertion(m8, probe2[c], c, rc)
    repass_heldout = (j_p2 >= 0.70) and (mass_p2 > nu_p2)
    rows.append((classes[c], j_p2, repass_heldout, keep, full, worst_fam, worst_drop))
    print(f"  {classes[c]:12s} | held-out probe2: Jaccard={j_p2:.2f} mass={mass_p2:.3f}/null{nu_p2:.3f} repass={repass_heldout} | "
          f"insertion(keep r_c only)={keep:.2f} vs full {full:.2f} | worst victim: {worst_fam} drop {worst_drop:+.3f}")

import json
json.dump({"dataset": DS, "rows": [{"family": nm, "jaccard_heldout": round(j, 3), "repass_heldout": bool(rp),
           "insertion_keep_rc": round(k, 3), "insertion_full": round(f, 3), "worst_victim": wf, "worst_victim_drop": round(wd, 4)}
           for nm, j, rp, k, f, wf, wd in rows]},
          open(f"results/etgrepaudit_{DS.lower().replace('-','_')}.json", "w"), indent=2)
print(f"saved results/etgrepaudit_{DS.lower().replace('-','_')}.json")
