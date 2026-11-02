"""Independently check CSE iCaRL archive mathematics and publish only its new scope."""
import argparse
import gzip
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import re
import shutil
import subprocess
import numpy as np
import torch

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--base',type=Path,required=True)
p.add_argument('--workspace',type=Path,required=True)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--publish',action='store_true')
a=p.parse_args()
B=a.base.resolve(strict=True)
W=a.workspace.resolve(strict=True)
S=a.source.resolve(strict=True)
read=lambda path:json.loads(path.read_text(encoding='utf-8'))
sha=lambda raw:hashlib.sha256(raw).hexdigest()
REL='paper_aacil/results/section178_literature/cic-ids-2018'
names=[f'icarl-scoped-npy_seed{s}' for s in range(10,15)]+['icarl-scoped-npy_overview']
roots={10:B/'runs/section178/cic-ids-2018/icarl-scoped-npy/seed10'}
roots.update({s:B/f'validation/hpc-20261102T1345/a/icarl/seed{s}' for s in range(11,15)})
i=read(roots[11]/'cell/result.json')['identity']
home=i['command'][0].split('/classil-20261102/')[0]
scratch=i['effective_config']['problem_scout_data_root'].split('/classil-a-v2-20261102/')[0]
replacements=[(str(W).replace('\\','/'),'@@WORKSPACE@@'),(str(B).replace('\\','/'),'@@EXPERIMENT_ROOT@@'),(home,'@@HPC_HOME@@'),(scratch,'@@HPC_SCRATCH@@'),(os.environ.get('USERPROFILE','').replace('\\','/'),'@@LOCAL_HOME@@')]
def redact(value):
    for old,new in replacements:
        if old:
            for spelling in (old,old.replace('/','\\\\'),old.replace('/','\\')):value=value.replace(spelling,new)
    value=re.sub(r'(?<![A-Za-z])[CE]:[/\\]','@@LOCAL_DRIVE@@/',value)
    return value.replace(Path(home).name,'@@HPC_USER@@')
def clean_obj(value):return json.loads(redact(json.dumps(value)))
def close(x,y):
    if x is None or y is None:assert x is y
    else:assert np.isclose(x,y,atol=1e-12,rtol=0),(x,y)
def calculate(result,view):
    stages=[]
    recalls=[]
    for cp,d,c in zip(result['checkpoints'],result['default_stages'],result['calibrated_stages']):
        point=d['metrics'] if view=='default' else c['operating_points'][view]
        cm=np.asarray(point['confusion_matrix'],dtype=np.int64)
        support,predicted=cm.sum(1),cm.sum(0)
        seen=np.asarray(cp['seen_classes'])
        diagonal=cm.diagonal()
        rc={int(k):float(diagonal[k]/support[k]) for k in seen}
        recalls.append(rc)
        stages.append(dict(bacc=float(np.mean(diagonal[seen]/support[seen])),accuracy=float(diagonal.sum()/support.sum()),
            macro_f1=float(np.mean(2*diagonal[seen]/(support[seen]+predicted[seen]))),normal_fpr=float(1-diagonal[0]/support[0]),
            attack_recall=float(1-cm[1:,0].sum()/support[1:].sum())))
    perclass=[]
    for stage,task in enumerate(result['identity']['tasks']):
        for c in task:
            prior=[r[c] for r in recalls[stage:-1]]
            perclass.append(dict(class_id=c,name=result['identity']['class_order'][c],first=recalls[stage][c],peak=max(r[c] for r in recalls[stage:]),best_prior=max(prior) if prior else None,final=recalls[-1][c],forgetting=max(prior)-recalls[-1][c] if prior else None))
    out=dict(stages[-1],incremental_bacc=float(np.mean([s['bacc'] for s in stages])),incremental_accuracy=float(np.mean([s['accuracy'] for s in stages])),forgetting=float(np.mean([v['forgetting'] for v in perclass if v['forgetting'] is not None])),stages=stages,per_class=perclass)
    return out
metrics={}
def decode(raw):
    for enc in ('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig','gb18030'):
        try:return raw.decode(enc)
        except UnicodeError:pass
    raise ValueError('Undecodable original log')
def source_start(name):
    rows=[json.loads(line) for line in decode((B/'logs'/name).read_bytes()).splitlines() if line.startswith('{')]
    return next(e for e in rows if e.get('event')=='START')['source_sha256']
sources={10:source_start('icarl-cse-scoped-seed10.monitor.stdout.log'),11:source_start('cer-bic-common-cse-scoped-seed12.monitor.stdout.log')}
for path,expected in sources[11].items():
    assert sha((W/'work/section151_publish'/path).read_bytes())==expected,path
for seed,root in roots.items():
    directory=S/f'icarl-scoped-npy_seed{seed}'
    original=read(root/'cell/result.json')
    original['default_stages']=read(root/'default_stages.json')
    original['calibrated_stages']=read(root/'calibrated_stages.json')
    assert read(directory/'result.json')==clean_obj(original)
    assert read(directory/'summary.json')==clean_obj(read(root/'summary.json'))
    state=read(directory/'source_state.json')
    assert state['original_result_sha256']==sha((root/'cell/result.json').read_bytes())
    assert state['frozen_source_sha256']==sources[10 if seed==10 else 11]
    records=read(directory/'checkpoints.json')
    assert len(records)==4
    memories=read(directory/'memory_features.json')['stages']
    for cp,record,memory in zip(original['checkpoints'],records,memories):
        path=root/'cell'/cp['checkpoint_file']
        assert record['sha256']==sha(path.read_bytes())==cp['checkpoint_sha256']
        assert record['metadata']==clean_obj(cp)
        payload=torch.load(path,map_location='cpu',weights_only=True)
        assert payload['identity']==original['identity']
        assert memory['task']==cp['task_index'] and not memory['unique_origin_row_ids_verified']
        for features,labels,item in zip(payload['buffer_x'],payload['buffer_y'],memory['classes']):
            values=np.ascontiguousarray(features.numpy())
            keys=values.view(np.dtype((np.void,values.shape[1]*values.dtype.itemsize))).ravel()
            assert item['class_id']==int(labels[0]) and item['slots']==len(values)
            assert item['unique_class_feature_values']==len(np.unique(keys))
            assert item['feature_sha256']==sha(values.tobytes()) and item['label_sha256']==sha(labels.numpy().tobytes())
    metrics[seed]={}
    archived=read(directory/'metrics.json')
    for view in ('default','0.01','0.02','0.05'):
        actual=calculate(original,view)
        metrics[seed][view]=actual
        for key,value in actual.items():
            if key=='stages':
                for x,y in zip(value,archived[view][key]):
                    for field in x:close(x[field],y[field])
            elif key=='per_class':
                for x,y in zip(value,archived[view][key]):
                    for field in ('first','peak','best_prior','final','forgetting'):close(x[field],y[field])
            else:close(value,archived[view][key])
    cost=read(directory/'costs.json')
    close(cost['training_seconds'],sum(c['training_seconds'] for c in original['checkpoints']))
    close(cost['buffer_selection_seconds'],sum(c['buffer_selection_seconds'] for c in original['checkpoints']))
aggregate=read(S/'overview/aggregate.json')
perclass=read(S/'overview/per_class_aggregate.json')
for view in aggregate:
    for key in aggregate[view]:
        values=[metrics[s][view][key] for s in roots]
        close(np.mean(values),aggregate[view][key]['mean'])
        close(np.std(values,ddof=1),aggregate[view][key]['sample_sd'])
    for j,item in enumerate(perclass[view]):
        for key in ('first','peak','best_prior','final','forgetting'):
            values=[metrics[s][view]['per_class'][j][key] for s in roots]
            if values[0] is None:assert item[key] is None
            else:
                close(np.mean(values),item[key]['mean'])
                close(np.std(values,ddof=1),item[key]['sample_sd'])
files={}
for record in read(S/'overview/log_artifacts.json'):
    name=record['file'].removeprefix('logs/').removesuffix('.gz')
    archived=gzip.decompress((S/'overview'/record['file']).read_bytes()).decode('utf-8')
    candidates=[B/'logs'/name,B/'validation/hpc-20261102T1345/a'/name,
                B/'validation/hpc-493605-20261102'/name,
                B/'validation/hpc-493605-20261102/protected-prediction-validation-20261102'/name,
                B/'hpc-cse-job-a-20261102/failure-logs'/name]
    matching=[path for path in candidates if path.is_file() and sha(path.read_bytes())==record['original_sha256']]
    if not matching:
        # These two own-export streams were copied before that same exporter
        # printed its completion. They are empty point-in-time snapshots, not
        # scientific training/verification evidence. The final logs stay local.
        assert name in ('cse-icarl-completed-export-20261102.retry3.stdout.log','cse-icarl-completed-export-20261102.retry3.stderr.log')
        assert record['original_sha256']==sha(b'') and archived==''
    else:assert archived==redact(decode(matching[0].read_bytes())),name
for name in names:
    source=S/('overview' if name.endswith('_overview') else name)
    for path in source.rglob('*'):
        if not path.is_file():continue
        raw=path.read_bytes()
        decoded=(gzip.decompress(raw) if path.suffix=='.gz' else raw).decode('utf-8')
        assert not re.search(r'(?<![A-Za-z])[A-Za-z]:[/\\]|/home/user/[A-Za-z0-9]|/scr/user/[A-Za-z0-9]|gh'+'p_[A-Za-z0-9]{20}|-----BE'+'GIN [A-Z ]*PRI'+'VATE KEY-----',decoded),path.name
        files[f'{REL}/{name}/{path.relative_to(source).as_posix()}']=raw
assert len(files)==177
print(json.dumps(dict(event='CSE_ICARL_ARCHIVE_VALIDATED',artifacts=len(files),independent_stage_views=80,independent_class_views=140,scope='Five existing completed cells, independent NumPy math; no training')),flush=True)
technical={}
for name in ('check-hpc-cse-icarl-states.py','export-cse-icarl-completed.py','publish-cse-icarl-completed.py'):
    raw=(B/'admin'/name).read_bytes()
    assert not re.search(r'(?<![A-Za-z])[A-Za-z]:[/\\]|/home/user/[A-Za-z0-9]|/scr/user/[A-Za-z0-9]|gh'+'p_[A-Za-z0-9]{20}|-----BE'+'GIN [A-Z ]*PRI'+'VATE KEY-----',raw.decode('utf-8')),name
    technical['ncd/hpc/'+name]=raw
files.update(technical)
def git(repo,*args):
    r=subprocess.run(['git','-C',str(repo),*args],text=True,encoding='utf-8',stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if r.returncode:
        print(r.stdout,flush=True)
        raise RuntimeError('git failed: '+str(args[0]))
    return r.stdout.strip()
if a.publish:
    repos=[(W/'work/section165_paper_publish','https://github.com/1838904818/Class-IL-in-IDS.git',''),(W/'work/github_supervisor_sync_20261010','https://github.com/1838904818/Class-IL.git','main')]
    scopes=[f'{REL}/{name}' for name in names]+list(technical)
    with (W/'work/section151_local/publication.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            heads=[]
            for repo,url,branch in repos:
                assert git(repo,'remote','get-url','origin')==url and git(repo,'branch','--show-current')==branch
                assert not git(repo,'diff','--name-only') and not git(repo,'diff','--cached','--name-only')
                git(repo,'fetch','origin')
                git(repo,'merge','--ff-only','origin/main')
                for scope in scopes:
                    assert not git(repo,'ls-tree','--name-only','origin/main','--',scope)
                    assert not (repo/scope).exists()
                for name in names:shutil.copytree(S/('overview' if name.endswith('_overview') else name),repo/REL/name)
                for path,raw in technical.items():
                    (repo/path).parent.mkdir(parents=True,exist_ok=True)
                    (repo/path).write_bytes(raw)
                git(repo,'add','--sparse','--force','--',*scopes)
                assert sorted(git(repo,'diff','--cached','--name-only').splitlines())==sorted(files)
                git(repo,'diff','--cached','--check')
                git(repo,'commit','-m','493533 RUNNING 5/6 raw; 493534 COMPLETED 60/60; 493535 PENDING: publish five verified CSE iCaRL cells',
                    '-m','Archive20 checkpoints and60 calibration views; seed10 local source400, seeds11-14 HPC493533 source403 plus explicit portable entry. Independent Linux493605 predictions exactly reproduced. Preserve weak results, Windows numerical portability limits, non-unique feature provenance and shared-allocation cost limits. No training reruns. This does not close overall S1, RQ1/RQ3, innovation or manuscript QA.')
                git(repo,'push','origin','HEAD:main')
                head=git(repo,'rev-parse','HEAD')
                assert git(repo,'ls-remote','origin','refs/heads/main').split()[0]==head
                for path in files:assert git(repo,'hash-object',str(repo/path))==git(repo,'rev-parse',head+':'+path)
                heads.append(head)
                print(json.dumps(dict(event='PUBLISHED_CSE_ICARL',repository=url,head=head,artifacts=len(files))),flush=True)
            for path in files:assert git(repos[0][0],'rev-parse',heads[0]+':'+path)==git(repos[1][0],'rev-parse',heads[1]+':'+path)
            print('DUAL_REMOTE_HEAD_ALL177_ARTIFACT_AND3_SOURCE_BLOBS_CROSS_REPO_VERIFIED',flush=True)
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
