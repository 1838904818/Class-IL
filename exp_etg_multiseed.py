# -*- coding: utf-8 -*-
"""ETG admission robustness across seeds: is the 34/39 CERTIFIED vs UNEXPLAINABLE partition stable,
or a single-seed artifact? Run admission over 5 seeds; report admitted mean+-std and per-family rate."""
import sys, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from src.config import seed_all
from src.data import DATASET_LOADERS
from src.methods.base import make_task_split, subset_by_classes

DS = sys.argv[1] if len(sys.argv) > 1 else "NSL-KDD"
SEEDS = [42, 1, 2, 3, 4]
EPOCHS, K_BUF, H, N_PROBE, TOPK, R_CTRL = 6, 50, 128, 100, 15, 50


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


def del_gap(model, X, c, feats):
    feats = list(feats)
    if not feats: return float("nan")
    with torch.no_grad():
        full = torch.softmax(model(X), 1)[:, c]
        Xd = X.clone(); Xd[:, feats] = 0.0
        return float((full - torch.softmax(model(Xd), 1)[:, c]).mean())


loader, cpt = DATASET_LOADERS[DS]
seed_all(SEEDS[0]); Xtr0, ytr0, Xte0, yte0, classes = loader()
admit_counts, fam_admit = [], {classes[c]: 0 for c in range(len(classes))}
for SEED in SEEDS:
    seed_all(SEED)
    Xtr, ytr, Xte, yte, classes = loader()
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr).astype(np.float32); Xte = sc.transform(Xte).astype(np.float32)
    d, ncl = Xtr.shape[1], len(classes); tasks = make_task_split(ncl, cpt)
    seed_all(SEED); model = Net(d, ncl); buf_X, buf_y, seen, done = [], [], [], set(); adm = 0
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
            rng = np.random.RandomState(7000 + int(c) + SEED)
            null = float(np.percentile([del_gap(model, Xc, c, rng.choice(d, TOPK, replace=False)) for _ in range(R_CTRL)], 95))
            if del_gap(model, Xc, c, top_k(occ_attr(model, Xc, c))) > null:
                adm += 1; fam_admit[classes[c]] += 1
    admit_counts.append(adm)
    print(f"  seed {SEED}: admitted {adm}/{len(done)}")

n = len(fam_admit)
print(f"\n{DS}: admitted across {len(SEEDS)} seeds = {np.mean(admit_counts):.1f} +- {np.std(admit_counts):.1f}  (of {n})")
stable_adm = sum(v == len(SEEDS) for v in fam_admit.values())
stable_ref = sum(v == 0 for v in fam_admit.values())
print(f"  stably CERTIFIED (all {len(SEEDS)} seeds): {stable_adm}/{n}   stably UNEXPLAINABLE (0 seeds): {stable_ref}/{n}   borderline: {n-stable_adm-stable_ref}/{n}")
print(f"  per-family admit rate: { {k: f'{v}/{len(SEEDS)}' for k, v in fam_admit.items()} }")
import json
json.dump({"dataset": DS, "seeds": SEEDS, "admit_counts": admit_counts,
           "admit_mean": round(float(np.mean(admit_counts)), 2), "admit_std": round(float(np.std(admit_counts)), 2),
           "n_families": n, "stable_certified": stable_adm, "stable_unexplainable": stable_ref,
           "per_family_admit_rate": fam_admit},
          open(f"results/etgms_{DS.lower().replace('-','_')}.json", "w"), indent=2)
print(f"saved results/etgms_{DS.lower().replace('-','_')}.json")
