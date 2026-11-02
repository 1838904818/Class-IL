"""Read-only BiC HPC state/row-role/progress and imported-payload checks.

Does not train or recompute predictions; that remains a separate verification.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--directory',type=Path,required=True)
p.add_argument('--original',type=Path,required=True)
p.add_argument('--data',type=Path,required=True)
p.add_argument('--entry',type=Path,required=True)
p.add_argument('--reference-result',type=Path,required=True)
p.add_argument('--resume-snapshot',type=Path,help='Immutable epoch6 snapshot for seed14 only')
a=p.parse_args()
torch.set_num_threads(2)
read=lambda path:json.loads(path.read_text(encoding='utf-8'))
digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
def equal(x,y):
    if isinstance(x,torch.Tensor):assert isinstance(y,torch.Tensor) and x.dtype==y.dtype and torch.equal(x,y)
    elif isinstance(x,dict):
        assert x.keys()==y.keys()
        for key in x:equal(x[key],y[key])
    elif isinstance(x,(list,tuple)):
        assert type(x)==type(y) and len(x)==len(y)
        for v,w in zip(x,y):equal(v,w)
    else:assert x==y,(x,y)
r=read(a.directory/'result.json')
i=r['identity']
assert r['completed'] and r['scientific_result'] and not i['smoke']
assert i['dataset']=='CSE-CIC-IDS2018' and i['seed'] in (13,14) and len(i['source_sha256'])==403
assert i['method']=='cer-bic-flow-cnn-common' and i['portable_entry_sha256']==digest(a.entry)
reference=read(a.reference_result)['identity']
assert i['data_binding']==reference['data_binding'] and i['data_metadata_sha256']==reference['data_metadata_sha256']
original=read(a.original/'identity.json')
assert original['source_sha256']==i['source_sha256']
assert {k:v for k,v in original['effective_config'].items() if k!='problem_scout_data_root'}=={k:v for k,v in i['effective_config'].items() if k!='problem_scout_data_root'}
cfg=i['effective_config']
assert cfg['schedule']=='cer_common_adam_8epochs_min2000' and cfg['optimizer']=='Adam'
assert cfg['epochs_per_task']==8 and cfg['min_steps_per_task']==2000 and cfg['learning_rate']==.001
assert cfg['stage2_epochs']==200 and cfg['batch_size']==256
resume=None
if i['seed']==14:
    assert a.resume_snapshot and digest(a.resume_snapshot)=='bd2e9b692c9fd7f91203e52670412fe51f585efceb968a3aaaedad09591d3c36'
    resume=torch.load(a.resume_snapshot,map_location='cpu',weights_only=True)
    assert resume['identity']==i and resume['next_epoch']==6 and resume['steps']==236286
    assert len(resume['history'])==6 and not resume['done']
    assert not any(any(term in k.lower() for term in ('rng','random','generator')) for k in resume)
y=np.load(a.data/'train_y.npy')
yt=np.load(a.data/'test_y.npy')
train,valid={},{}
for c in range(len(i['class_order'])):
    rows=np.flatnonzero(y==c)
    perm=np.random.default_rng(i['seed']*100003+c*1009+17).permutation(rows)
    k=max(5,math.ceil(len(rows)*.1))
    valid[c],train[c]=perm[:k],np.sort(perm[k:])
    assert hashlib.sha256(valid[c].tobytes()).hexdigest()==i['holdout'][str(c)]['validation_indices_sha256']
held=np.concatenate(list(valid.values()))
prior=None
imported_seconds=new_seconds=0.
imports=0
for t,task in enumerate(i['tasks']):
    cp=r['checkpoints'][t]
    path=a.directory/cp['checkpoint_file']
    assert digest(path)==cp['checkpoint_sha256'] and read(path.with_suffix('.json'))==cp
    s=torch.load(path,map_location='cpu',weights_only=True)
    assert s['identity']==i and s['task_index']==t
    seen=[c for group in i['tasks'][:t+1] for c in group]
    assert s['seen_classes']==cp['seen_classes']==seen
    for folder,name in (('checkpoints',path.name),('progress',f'backbone_{t:03d}.pt')):
        target=a.directory/folder/name
        payload=torch.load(target,map_location='cpu',weights_only=True)
        assert payload['identity']==i
        if i['seed']==13 and t<3:
            source=a.original/folder/name
            old=torch.load(source,map_location='cpu',weights_only=True)
            assert old['identity']==original
            proof=payload['portable_import']
            assert proof['source_sha256']==digest(source) and proof['original_identity']==original
            assert proof['source_file']==f'{folder}/{name}' and proof['payload_unchanged']
            equal({k:v for k,v in payload.items() if k not in ('identity','portable_import')},{k:v for k,v in old.items() if k!='identity'})
            imports+=1
        else:assert 'portable_import' not in payload
    replay={c:v.numpy() for c,v in s['replay'].items()}
    correction={c:v.numpy() for c,v in s['correction'].items()}
    retained=np.concatenate(list(replay.values())+list(correction.values()))
    assert len(retained)==len(np.unique(retained))==cp['memory_slots']==cp['unique_row_ids']
    assert len(retained)<=200*len(seen) and not np.intersect1d(retained,held).size
    assert sorted(replay)==sorted(correction)==seen
    for c in seen:
        assert len(replay[c])<=180 and len(correction[c])<=20
        ids=np.concatenate((replay[c],correction[c]))
        assert np.isin(ids,train[c]).all() and (y[ids]==c).all()
        if prior is not None and c in prior['replay']:
            assert torch.equal(s['replay'][c],prior['replay'][c])
            assert torch.equal(s['correction'][c],prior['correction'][c])
    current,stopping={},{}
    for c in task:
        perm=np.random.default_rng((i['seed']*10000+t)*1009+c).permutation(train[c])
        nb=min(20,max(1,len(perm)//3))
        ns=min(200,max(1,math.ceil(.1*len(perm))))
        bias,stop,fit=perm[:nb],perm[nb:nb+ns],perm[nb+ns:]
        current[c],stopping[c]=np.sort(fit),np.sort(stop)
        assert np.array_equal(correction[c],np.sort(bias))
        assert np.array_equal(s['current_rows'][c].numpy(),current[c])
        assert np.array_equal(s['stopping_rows'][c].numpy(),stopping[c])
        assert np.isin(replay[c],current[c]).all()
    fit=s['backbone_fit_rows'].numpy()
    expected=np.concatenate(list(current.values())+([] if prior is None else [v.numpy() for v in prior['replay'].values()]))
    assert np.array_equal(fit,expected)
    stops=np.concatenate(list(stopping.values()))
    biases=s['bias_fit_rows'].numpy()
    assert not np.intersect1d(fit,np.concatenate((stops,biases,held))).size
    assert not np.intersect1d(stops,np.concatenate((biases,held))).size
    assert np.isin(biases,np.concatenate(list(correction.values()))).all()
    assert len(set(np.bincount(y[biases],minlength=len(i['class_order']))[seen]))==1
    batches=math.ceil(len(fit)/256)
    target=max(8*batches,2000)
    schedule=s['training_schedule']
    assert schedule['optimizer_steps']==target and schedule['epochs']==math.ceil(target/batches)
    assert len(schedule['history'])==schedule['epochs']
    for epoch,h in enumerate(schedule['history'],1):
        assert h['epoch']==epoch and math.isfinite(h['validation_loss']) and h['lr']==.001
        assert h['optimizer_steps']==min(batches,target-(epoch-1)*batches)
    if resume is not None and t==0:
        assert schedule['history'][:6]==resume['history']
        assert sum(h['optimizer_steps'] for h in schedule['history'][:6])==resume['steps']
    epochs=200 if t else 0
    assert s['bias_schedule']['epochs']==epochs
    assert s['bias_schedule']['optimizer_steps']==epochs*math.ceil(len(biases)/256)
    progress=torch.load(a.directory/f'progress/backbone_{t:03d}.pt',map_location='cpu',weights_only=True)
    assert progress['done'] and progress['history']==schedule['history']
    for name,value in s['model_state'].items():
        if name not in (f'scales.{t}',f'offsets.{t}'):assert torch.equal(value,progress['best'][name]),name
    if prior is not None:
        for j in range(t):
            for key in ('scales','offsets'):assert torch.equal(s['model_state'][f'{key}.{j}'],prior['model_state'][f'{key}.{j}'])
    for view in [r['default_stages'][t]['metrics']]+list(r['calibrated_stages'][t]['operating_points'].values()):
        cm=np.asarray(view['confusion_matrix'],dtype=np.int64)
        supports=cm.sum(1)
        expected_test=np.bincount(yt,minlength=len(i['class_order']))
        assert np.array_equal(supports[seen],expected_test[seen])
        assert int(cm.sum())==int(np.isin(yt,seen).sum())
    seconds=s['training_seconds']
    assert seconds==cp['training_seconds'] and seconds>=0
    if i['seed']==13 and t<3:imported_seconds+=seconds
    else:new_seconds+=seconds
    prior=s
    print(json.dumps(dict(event='BIC_STATE_CHECKED',task=t,true_row_ids=len(retained),imported=i['seed']==13 and t<3,training_seconds=seconds,predictions_recomputed=False)),flush=True)
assert imports==(6 if i['seed']==13 else 0)
assert abs(imported_seconds+new_seconds-r['training_seconds_total'])<1e-8
summary=read(a.directory/'summary.json')
assert summary['completed'] and summary['scientific_result'] and not summary['smoke']
print(json.dumps(dict(event='BIC_STATES_COMPLETE',seed=i['seed'],checkpoints=4,exact_imported_payloads=imports,imported_training_seconds=imported_seconds,new_training_seconds=new_seconds if resume is None else None,recorded_training_seconds_total=imported_seconds+new_seconds,resume_preexisting_training_seconds=resume['seconds'] if resume is not None else None,recorded_post_resume_boundary_seconds=new_seconds-resume['seconds'] if resume is not None else None,new_attempt_summary_elapsed_seconds=summary['elapsed_seconds'],resume_completed_epochs_preserved=6 if resume is not None else None,rng_restored=False if resume is not None else None,predictions_recomputed=False,accepted=False)),flush=True)
