"""Read-only independent NSL DER++ checkpoint forward and CM verification.

Uses tensor affine/ReLU primitives directly, not the training/evaluation helpers.
This does not establish unique memory row provenance or whole-job acceptance.
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
a = p.parse_args()
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
read = lambda path: json.loads(path.read_text(encoding='utf-8'))
xtrain = np.load(a.data/'train_x.npy', mmap_mode='r')
ytrain = np.load(a.data/'train_y.npy', mmap_mode='r')
xtest = np.load(a.data/'test_x.npy', mmap_mode='r')
ytest = np.load(a.data/'test_y.npy', mmap_mode='r')
count = 0
calibration_discrepancies = []
test_discrepancies = []
for seed in range(10, 15):
    root = a.root/f'seed{seed}'
    result = read(root/'cell/result.json')
    identity = result['identity']
    assert identity['dataset'] == 'NSL-KDD' and identity['method'] == 'derpp'
    assert identity['seed'] == seed and result['completed']
    defaults = read(root/'default_stages.json')
    calibrated = read(root/'calibrated_stages.json')
    n = len(identity['class_order'])
    normal = identity['class_order'].index('Normal')
    counts = np.bincount(ytrain, minlength=n)
    allowed = {}
    pools = {}
    for c in range(n):
        class_ids = np.flatnonzero(ytrain == c)
        perm = np.random.default_rng(seed*100003+c*1009+17).permutation(class_ids)
        pools[c] = np.sort(perm[max(5, int(np.ceil(len(class_ids)*.1))):])
        features = np.ascontiguousarray(xtrain[pools[c]])
        row_dtype = np.dtype((np.void, features.shape[1]*features.dtype.itemsize))
        allowed[c] = features.view(row_dtype).ravel().copy()
    ids = np.flatnonzero(ytrain == normal)
    holdout = np.random.default_rng(seed*100003+normal*1009+17).permutation(ids)
    holdout = holdout[:max(5, int(np.ceil(len(ids)*.1)))]
    assert hashlib.sha256(holdout.tobytes()).hexdigest() == identity['section26_holdout'][str(normal)]['validation_indices_sha256']
    for cp, default, calibration in zip(result['checkpoints'], defaults, calibrated):
        path = root/'cell'/cp['checkpoint_file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == cp['checkpoint_sha256']
        state = torch.load(path, map_location='cpu', weights_only=True)
        assert state['identity'] == identity and state['seen_classes'] == cp['seen_classes']
        seen = cp['seen_classes']
        task = identity['tasks'][cp['task_index']]
        new_rows = sum(len(pools[c]) for c in task)
        assert state['new_training_rows'] == state['train_rows'] == new_rows
        assert state['train_class_counts'] == {str(c): len(pools[c]) if c in task else 0 for c in seen}
        schedule = state['training_schedule']
        steps = max(8*((new_rows+255)//256), 2000)
        assert schedule['rows_per_epoch'] == new_rows
        assert schedule['optimizer_steps'] == schedule['target_steps'] == steps
        memory = state['derpp_memory']
        assert memory['per_class'] == 200 and memory['n_classes'] == n
        memory_rows = 0
        for c in seen:
            used = min(200, int(memory['observations'][c]))
            features = np.ascontiguousarray(memory['x'][c, :used].numpy())
            row_dtype = np.dtype((np.void, features.shape[1]*features.dtype.itemsize))
            assert np.isin(features.view(row_dtype).ravel(), allowed[c]).all(), (seed, cp['task_index'], c, 'memory outside allowed TRAIN features')
            memory_rows += used
        assert memory_rows == state['replay_examples_after']
        w = state['model_state']
        assert set(w) == {'features.0.weight', 'features.0.bias', 'features.2.weight', 'features.2.bias', 'head.weight', 'head.bias'}
        def forward(x, rows, batch):
            pieces = []
            with torch.inference_mode():
                for offset in range(0, len(rows), batch):
                    v = torch.from_numpy(np.array(x[rows[offset:offset+batch]], dtype=np.float32, copy=True))
                    v = F.relu(F.linear(v, w['features.0.weight'], w['features.0.bias']))
                    v = F.relu(F.linear(v, w['features.2.weight'], w['features.2.bias']))
                    pieces.append(F.linear(v, w['head.weight'], w['head.bias'])[:, seen])
            out = torch.cat(pieces)
            assert torch.isfinite(out).all()
            return out
        rows = np.flatnonzero(np.isin(ytest, seen))
        def cm(pred):
            return np.bincount(ytest[rows]*n+np.asarray(seen)[pred], minlength=n*n).reshape(n,n).tolist()
        default_logits = forward(xtest, rows, 4096)
        actual_default = cm(default_logits.argmax(1).numpy())
        if actual_default != default['metrics']['confusion_matrix']:
            test_discrepancies.append(dict(seed=seed, task=cp['task_index'], view='default', recomputed=actual_default, recorded=default['metrics']['confusion_matrix']))
        batch = int(identity['effective_config']['eval_batch_size'])
        logits = forward(xtest, rows, batch)
        normal_logits = forward(xtrain, holdout, batch)
        prior = torch.from_numpy(np.log(counts/counts.sum()).astype(np.float32)[seen])
        for alpha, point in calibration['calibration']['operating_points'].items():
            tau = point['total_tau']
            pred = (logits+tau*prior).argmax(1).numpy()
            actual = cm(pred)
            expected = calibration['operating_points'][alpha]['confusion_matrix']
            if actual != expected:
                shifted = logits+tau*prior
                top = shifted.topk(2, dim=1).values
                margin = top[:, 0]-top[:, 1]
                near = np.flatnonzero((margin <= 2e-5).numpy())
                near_records = [dict(test_row=int(rows[j]), truth=int(ytest[rows[j]]), prediction=int(seen[pred[j]]), margin=float(margin[j]), logits=logits[j].tolist(), shifted=shifted[j].tolist()) for j in near]
                test_discrepancies.append(dict(seed=seed, task=cp['task_index'], view=alpha, recomputed=actual, recorded=expected, near_ties=near_records))
            fp = int(((normal_logits+tau*prior).argmax(1) != seen.index(normal)).sum())
            if fp != point['calibration_false_positives']:
                shifted = normal_logits+tau*prior
                attacks = [j for j, c in enumerate(seen) if c != normal]
                gap = shifted[:, attacks].max(1).values-shifted[:, seen.index(normal)]
                order = torch.argsort(gap, descending=True)
                rank = int(point['calibration_false_positives'])
                border = [dict(rank=j, train_row=int(holdout[int(order[j])]), attack_minus_normal=float(gap[order[j]])) for j in range(max(0,rank-1), min(len(order),rank+2))]
                calibration_discrepancies.append(dict(seed=seed, task=cp['task_index'], alpha=alpha, recorded_fp=point['calibration_false_positives'], recomputed_fp=fp, boundary=border))
        count += 1
        print(json.dumps(dict(event='CHECKPOINT_PREDICTIONS_CHECKED', seed=seed, task=cp['task_index'], test_rows=len(rows), views=4, optimizer_steps=steps, allowed_train_feature_memory_rows=memory_rows, unique_origin_row_provenance=False, accepted=False)), flush=True)
print(json.dumps(dict(event='PREDICTIONS_COMPLETE', checkpoints=count, test_cm_views=count*4, calibration_points=count*3, calibration_discrepancies=calibration_discrepancies, test_discrepancies=test_discrepancies, accepted=False, remaining='Resolve prediction/calibration discrepancies if any; training schedule, memory membership/provenance, full source and resource/cost acceptance')), flush=True)
raise SystemExit(1 if calibration_discrepancies or test_discrepancies else 0)
