"""Read-only HPC iCaRL identity, schedule and allowed-TRAIN feature checks.

Membership is not unique originating-row provenance when duplicate rows exist.
Prediction verification is separately recorded by DICC job493605.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--data',type=Path,required=True)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--entry',type=Path,required=True)
a=p.parse_args()
torch.set_num_threads(2)
read=lambda path:json.loads(path.read_text(encoding='utf-8'))
digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
x=np.load(a.data/'train_x.npy',mmap_mode='r')
y=np.load(a.data/'train_y.npy')
metadata=read(a.data/'data_quality.json')
canonical=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
pending={}
excluded={}
total_training=0.
total_selection=0.
for seed in range(11,15):
    root=a.root/f'seed{seed}'
    r=read(root/'cell/result.json')
    i=r['identity']
    assert i['seed']==seed and i['method']=='icarl' and r['completed']
    assert i['portable_entry_sha256']==digest(a.entry)
    assert i['data_metadata_sha256']==canonical(metadata)
    assert i['lambda_file_sha256']==digest(a.source/i['lambda_file']) and i['regularization']==2.
    cfg=i['effective_config']
    assert cfg['epochs_per_task']==8 and cfg['min_steps_per_task']==2000
    assert cfg['learning_rate']==.001 and cfg['batch_size']==256 and cfg['buffer_per_class']==200
    assert cfg['hidden']==128 and cfg['hidden_layers']==2 and cfg['temperature']==2.
    assert not cfg['smoke'] and not cfg['section26'] and cfg['section178_literature']
    assert cfg['input_mapping_lifetime']=='scoped-npy-row-gather-v1; exact row values; original herding arithmetic'
    sizes={}
    mask=np.zeros(len(y),dtype=bool)
    for c in range(len(i['class_order'])):
        rows=np.flatnonzero(y==c)
        perm=np.random.default_rng(seed*100003+c*1009+17).permutation(rows)
        k=max(5,int(np.ceil(len(rows)*.1)))
        holdout=perm[:k]
        mask[holdout]=True
        sizes[c]=len(rows)-k
        receipt=i['section26_holdout'][str(c)]
        assert receipt['train_count']==sizes[c] and receipt['validation_count']==k
        assert receipt['validation_indices_sha256']==hashlib.sha256(holdout.tobytes()).hexdigest()
    excluded[seed]=mask
    previous={}
    train_seconds=selection_seconds=0.
    for cp in r['checkpoints']:
        path=root/'cell'/cp['checkpoint_file']
        assert digest(path)==cp['checkpoint_sha256'] and read(path.with_suffix('.json'))==cp
        s=torch.load(path,map_location='cpu',weights_only=True)
        assert s['identity']==i and s['task_index']==cp['task_index']
        seen=cp['seen_classes']
        assert seen==[c for task in i['tasks'][:cp['task_index']+1] for c in task]
        current=i['tasks'][cp['task_index']]
        new=sum(sizes[c] for c in current)
        old=sum(len(values) for values in previous.values())
        assert s['new_training_rows']==new and s['train_rows']==new+old
        assert s['replay_examples_before']==old
        memory={}
        for features,labels in zip(s['buffer_x'],s['buffer_y']):
            c=int(labels[0])
            assert (labels==c).all() and len(features)==min(200,sizes[c])
            assert torch.isfinite(features).all()
            memory[c]=features
            if c in previous:
                assert torch.equal(features,previous[c])
        assert sorted(memory)==seen
        assert s['replay_examples_after']==sum(len(v) for v in memory.values())
        assert s['train_class_counts']=={str(c):sizes[c] if c in current else len(previous[c]) for c in seen}
        schedule=s['training_schedule']
        assert schedule['rows_per_epoch']==new+old
        steps=max(8*((new+old+255)//256),2000)
        assert schedule['optimizer_steps']==schedule['target_steps']==steps
        assert s['training_seconds']==cp['training_seconds'] and s['buffer_selection_seconds']==cp['buffer_selection_seconds']
        assert s['training_seconds']>=0 and s['buffer_selection_seconds']>=0
        train_seconds+=s['training_seconds']
        selection_seconds+=s['buffer_selection_seconds']
        previous=memory
    for c,features in previous.items():
        values=np.ascontiguousarray(features.numpy())
        dtype=np.dtype((np.void,values.shape[1]*values.dtype.itemsize))
        pending[seed,c]=[values.view(dtype).ravel().copy(),np.zeros(len(values),dtype=bool)]
    total_training+=train_seconds
    total_selection+=selection_seconds
    print(json.dumps(dict(event='ICARL_STATES_CHECKED',seed=seed,checkpoints=4,training_seconds=train_seconds,selection_seconds=selection_seconds,accepted=False)),flush=True)
dtype=np.dtype((np.void,x.shape[1]*x.dtype.itemsize))
for offset in range(0,len(y),50000):
    stop=min(len(y),offset+50000)
    values=np.array(x[offset:stop],copy=True).view(dtype).ravel()
    labels=y[offset:stop]
    for (seed,c),(wanted,found) in pending.items():
        if found.all():continue
        eligible=(labels==c)&~excluded[seed][offset:stop]
        if eligible.any():found|=np.isin(wanted,values[eligible])
    if offset%500000==0:
        print(json.dumps(dict(event='MEMORY_SCAN',rows=stop,found=sum(int(found.sum()) for _,found in pending.values()),total=sum(len(found) for _,found in pending.values()))),flush=True)
    if all(found.all() for _,found in pending.values()):break
assert all(found.all() for _,found in pending.values()), 'Memory feature missing from allowed TRAIN pool'
print(json.dumps(dict(event='ICARL_STATES_COMPLETE',seeds=4,checkpoints=16,feature_memberships=sum(len(found) for _,found in pending.values()),unique_origin_row_provenance=False,training_seconds=total_training,selection_seconds=total_selection,accepted=False,remaining='Full frozen dependency/data binding and archival/cost attribution acceptance')),flush=True)
