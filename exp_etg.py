# -*- coding: utf-8 -*-
"""ETG -- Explanation-Trust Gate (PoC). It does not update detector weights.

A per-attack-family trust state machine governing explanation use over the
class-incremental IDS stream. The admission test uses the rationale-mass
primitive.

  states (per family, persist across tasks):
    UNCERTIFIED -> {CERTIFIED_STABLE | UNEXPLAINABLE} -> DRIFTED -> {re-certify | UNEXPLAINABLE}
  ADMISSION  (first-learning): rationale_mass m_c > random-15 p95 null n_c  ->  CERTIFIED_STABLE (freeze r_c)
                               else                                          ->  UNEXPLAINABLE
  STABILITY  (later task, CERTIFIED_STABLE): acc preserved AND Jaccard(r_c, now)<0.70 -> DRIFTED  (GOVERNED)
             (later task, UNEXPLAINABLE):    same alarm condition would fire           -> SUPPRESSED (no drift event)
  RE-CERT    (DRIFTED revisited): m_c(now) > n_c(now) -> re-certify; else -> UNEXPLAINABLE
  ACTION     CERTIFIED_STABLE -> explanation accepted for automated use ;
             DRIFTED          -> force HUMAN review (rationale under re-certification) ;
             UNEXPLAINABLE    -> alert still RAISED, explanation SUPPRESSED, no drift alarm emitted.

Outputs comprise the per-family trust ledger and aggregate governance counts.
"""
import json, copy, sys, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from src.config import seed_all
from src.data import DATASET_LOADERS
from src.methods.base import make_task_split, subset_by_classes

DS = sys.argv[1] if len(sys.argv) > 1 else "NSL-KDD"
SEED, EPOCHS, K_BUF, H = 42, 6, 50, 128
N_PROBE, TOPK, ACC_THR, JAC_THR, R_CTRL = 100, 15, -0.05, 0.70, 50


class Net(nn.Module):
    def __init__(s, d, n, h=H):
        super().__init__(); s.enc = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU()); s.head = nn.Linear(h, n)
    def forward(s, x): return s.head(s.enc(x))


def occ_attr(model, X, c):
    with torch.no_grad():
        base = torch.softmax(model(X), 1)[:, c]
        imp = np.zeros(X.shape[1], dtype=np.float32)
        for j in range(X.shape[1]):
            Xj = X.clone(); Xj[:, j] = 0.0
            imp[j] = float((base - torch.softmax(model(Xj), 1)[:, c]).mean())
    return imp


def top_k(v, k=TOPK): return set(np.argsort(v)[-k:].tolist())
def jac(a, b, k=TOPK):
    ta, tb = top_k(a, k), top_k(b, k); return len(ta & tb) / len(ta | tb)


def del_gap(model, X, c, feats):
    feats = list(feats)
    if not feats: return float("nan")
    with torch.no_grad():
        full = torch.softmax(model(X), 1)[:, c]
        Xd = X.clone(); Xd[:, feats] = 0.0
        return float((full - torch.softmax(model(Xd), 1)[:, c]).mean())


def mass_and_null(model, X, c, seed):
    m = del_gap(model, X, c, top_k(occ_attr(model, X, c)))
    rng = np.random.RandomState(seed)
    n = float(np.percentile([del_gap(model, X, c, rng.choice(X.shape[1], TOPK, replace=False)) for _ in range(R_CTRL)], 95))
    return m, n


loader, cpt = DATASET_LOADERS[DS]
seed_all(SEED)
Xtr, ytr, Xte, yte, classes = loader()
sc = StandardScaler(); Xtr = sc.fit_transform(Xtr).astype(np.float32); Xte = sc.transform(Xte).astype(np.float32)
d, ncl = Xtr.shape[1], len(classes); tasks = make_task_split(ncl, cpt); dev = "cpu"
print(f"{DS}: d={d} ncl={ncl} tasks={tasks}")

seed_all(SEED)
model = Net(d, ncl).to(dev); buf_X, buf_y, seen = [], [], []
probe, cert, cert_acc = {}, {}, {}
state, m0, n0, ledger = {}, {}, {}, []          # state[c], admission mass/null, transition ledger
gov = supp = recert = demote = 0
for i, ti in enumerate(tasks):
    Xi, yi = subset_by_classes(Xtr, ytr, ti)
    Xp, yp = (np.vstack([Xi]+buf_X), np.concatenate([yi]+buf_y)) if buf_X else (Xi, yi)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    Xt, yt = torch.from_numpy(Xp.astype(np.float32)).to(dev), torch.from_numpy(yp.astype(np.int64)).to(dev)
    model.train()
    for _ in range(EPOCHS):
        pm = torch.randperm(len(Xt))
        for s in range(0, len(Xt), 256):
            b = pm[s:s+256]; loss = F.cross_entropy(model(Xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    for c in ti: seen.append(c)
    for c in ti:
        mc = yi == c
        if mc.sum(): xs = Xi[mc]; sel = np.random.choice(len(xs), min(K_BUF, len(xs)), replace=False); buf_X.append(xs[sel]); buf_y.append(yi[mc][sel])
    for c in seen:
        if c not in probe:
            idx = np.where(yte == c)[0]
            if len(idx) == 0: continue
            probe[c] = torch.from_numpy(Xte[idx[:N_PROBE]].astype(np.float32)).to(dev)
        occ = occ_attr(model, probe[c], c)
        with torch.no_grad(): acc_c = float((model(probe[c]).argmax(1).cpu().numpy() == c).mean())
        if c not in state:                                       # ADMISSION at first-learning
            m, n = mass_and_null(model, probe[c], c, 7000 + int(c))
            cert[c], cert_acc[c], m0[c], n0[c] = occ, acc_c, m, n
            state[c] = "CERTIFIED_STABLE" if m > n else "UNEXPLAINABLE"
            ledger.append({"family": classes[c], "task": i, "event": "ADMIT", "to": state[c],
                           "mass": round(m, 4), "null": round(n, 4)})
        else:
            dacc, j = acc_c - cert_acc[c], jac(cert[c], occ)
            alarm = (dacc > ACC_THR and j < JAC_THR)
            if state[c] == "CERTIFIED_STABLE" and alarm:          # GOVERNED transition
                state[c] = "DRIFTED"; gov += 1
                ledger.append({"family": classes[c], "task": i, "event": "DRIFT", "to": "DRIFTED",
                               "jaccard": round(j, 3), "action": "force_human_review"})
            elif state[c] == "UNEXPLAINABLE" and alarm:           # SUPPRESSED by the gate
                supp += 1
                ledger.append({"family": classes[c], "task": i, "event": "ALARM_SUPPRESSED",
                               "jaccard": round(j, 3), "reason": "no faithful rationale (admission refused)"})
            elif state[c] == "DRIFTED":                            # RE-CERT vs DEMOTE
                m, n = mass_and_null(model, probe[c], c, 9000 + int(c) + i)
                if m > n:
                    state[c] = "CERTIFIED_STABLE"; cert[c], cert_acc[c] = occ, acc_c; recert += 1
                    ledger.append({"family": classes[c], "task": i, "event": "RECERT", "to": "CERTIFIED_STABLE", "mass": round(m, 4), "null": round(n, 4)})
                else:
                    state[c] = "UNEXPLAINABLE"; demote += 1
                    ledger.append({"family": classes[c], "task": i, "event": "DEMOTE", "to": "UNEXPLAINABLE", "mass": round(m, 4), "null": round(n, 4)})

admitted = sum(v != "UNEXPLAINABLE" or m0[c] > n0[c] for c, v in state.items())  # admitted at entry
adm = sum(1 for c in state if m0[c] > n0[c]); rej = len(state) - adm
final = {classes[c]: state[c] for c in state}
print(f"\nADMISSION: {adm}/{len(state)} CERTIFIED  |  {rej}/{len(state)} UNEXPLAINABLE (refused)")
print(f"ALARM GOVERNANCE: {gov} governed (STABLE->DRIFTED, routed to human)  |  {supp} SUPPRESSED (fired on refused families)")
print(f"RE-CERT: {recert}   DEMOTE: {demote}")
print(f"final per-family state: {final}")
json.dump({"dataset": DS, "n_families": len(state), "admitted": adm, "refused": rej,
           "governed_alarms": gov, "suppressed_alarms": supp, "recert": recert, "demote": demote,
           "final_state": final, "admission_margin": {classes[c]: round(m0[c]-n0[c], 4) for c in state},
           "ledger": ledger},
          open(f"results/etg_{DS.lower().replace('-','_')}.json", "w"), indent=2)
print(f"saved results/etg_{DS.lower().replace('-','_')}.json")
