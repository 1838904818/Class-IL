"""Read-only independent CSE iCaRL checkpoint NME prediction reconstruction.

No training or original model/evaluation helpers. Discrepancies remain failures.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--data', type=Path, required=True)
p.add_argument('--seed', type=int, choices=range(11,15), required=True)
a = p.parse_args()
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
read = lambda path: json.loads(path.read_text(encoding='utf-8'))
root = a.root/f'seed{a.seed}'
r = read(root/'cell/result.json')
i = r['identity']
assert i['method']=='icarl' and i['dataset']=='CSE-CIC-IDS2018' and i['seed']==a.seed
defaults, calibrated = read(root/'default_stages.json'), read(root/'calibrated_stages.json')
xt = np.load(a.data/'test_x.npy', mmap_mode='r')
yt = np.load(a.data/'test_y.npy', mmap_mode='r')
x = np.load(a.data/'train_x.npy', mmap_mode='r')
y = np.load(a.data/'train_y.npy', mmap_mode='r')
n = len(i['class_order'])
normal = i['class_order'].index('Normal')
counts = np.bincount(y, minlength=n)
ids = np.flatnonzero(y==normal)
holdout = np.random.default_rng(a.seed*100003+normal*1009+17).permutation(ids)
holdout = holdout[:max(5,int(np.ceil(len(ids)*.1)))]
assert hashlib.sha256(holdout.tobytes()).hexdigest()==i['section26_holdout'][str(normal)]['validation_indices_sha256']
# HDD-friendly reads only: keep every original feature bit, row, and inference
# batch order. The cache is RAM-only and bounded to this Normal holdout.
cache_bytes=len(holdout)*x.shape[1]*x.dtype.itemsize
assert cache_bytes < 1024**3
print(json.dumps(dict(event='HOLDOUT_PREFETCH_START',seed=a.seed,rows=len(holdout),bytes=cache_bytes)),flush=True)
holdout_x=np.empty((len(holdout),x.shape[1]),dtype=x.dtype)
order=np.argsort(holdout)
for offset in range(0,len(order),8192):
    slots=order[offset:offset+8192]
    holdout_x[slots]=x[holdout[slots]]
print(json.dumps(dict(event='HOLDOUT_PREFETCH_COMPLETE',seed=a.seed,original_row_and_batch_order_preserved=True)),flush=True)
errors = []
previous_memory = {}
for cp, d, c in zip(r['checkpoints'], defaults, calibrated):
    path = root/'cell'/cp['checkpoint_file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==cp['checkpoint_sha256']
    s = torch.load(path, map_location='cpu', weights_only=True)
    assert s['identity']==i and s['task_index']==cp['task_index']
    seen, w = cp['seen_classes'], s['model_state']
    def features(values):
        v = F.relu(F.linear(values,w['features.0.weight'],w['features.0.bias']))
        return F.relu(F.linear(v,w['features.2.weight'],w['features.2.bias']))
    bx, by = torch.cat(s['buffer_x']), torch.cat(s['buffer_y'])
    batch = int(i['effective_config']['batch_size'])
    with torch.inference_mode():
        embedded = torch.cat([features(bx[j:j+batch]) for j in range(0,len(bx),batch)]).numpy()
    means = []
    for label in seen:
        mean = embedded[by.numpy()==label].mean(axis=0)
        mean /= np.linalg.norm(mean)+1e-12
        means.append(torch.from_numpy(mean.astype(np.float32)))
        memory = bx[by==label]
        if label in previous_memory:
            assert torch.equal(previous_memory[label], memory)
        previous_memory[label] = memory.clone()
    prototypes = F.normalize(torch.stack(means), dim=1)
    def scores(array, rows, size):
        pieces=[]
        with torch.inference_mode():
            for start in range(0,len(rows),size):
                v=torch.from_numpy(np.array(array[rows[start:start+size]],dtype=np.float32,copy=True))
                z=F.normalize(features(v),dim=1)
                pieces.append(-((z[:,None,:]-prototypes[None,:,:])**2).sum(2))
        return torch.cat(pieces)
    rows=np.flatnonzero(np.isin(yt,seen))
    def cm(pred):
        return np.bincount(yt[rows]*n+np.asarray(seen)[pred],minlength=n*n).reshape(n,n).tolist()
    default=scores(xt,rows,4096)
    if cm(default.argmax(1).numpy())!=d['metrics']['confusion_matrix']:
        errors.append(dict(task=cp['task_index'],view='default'))
    size=int(i['effective_config']['eval_batch_size'])
    logits=scores(xt,rows,size)
    normal_logits=scores(holdout_x,np.arange(len(holdout)),size)
    prior=torch.from_numpy(np.log(counts/counts.sum()).astype(np.float32)[seen])
    for alpha, point in c['calibration']['operating_points'].items():
        tau=point['total_tau']
        if cm((logits+tau*prior).argmax(1).numpy())!=c['operating_points'][alpha]['confusion_matrix']:
            errors.append(dict(task=cp['task_index'],view=alpha))
        fp=int(((normal_logits+tau*prior).argmax(1)!=seen.index(normal)).sum())
        if fp!=point['calibration_false_positives']:
            errors.append(dict(task=cp['task_index'],view=alpha,holdout_fp=fp,recorded_fp=point['calibration_false_positives']))
    print(json.dumps(dict(event='NME_CHECKPOINT_CHECKED',seed=a.seed,task=cp['task_index'],test_rows=len(rows),memory_rows=len(bx),errors_so_far=len(errors),accepted=False)),flush=True)
print(json.dumps(dict(event='NME_COMPLETE',seed=a.seed,checkpoints=4,test_cm_views=16,calibration_points=12,errors=errors,accepted=False,remaining='Full data/source, current TRAIN memory membership, training schedule and cost checks')),flush=True)
raise SystemExit(1 if errors else 0)
