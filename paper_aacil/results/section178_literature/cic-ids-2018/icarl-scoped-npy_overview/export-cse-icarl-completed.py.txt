"""Archive five completed CSE scoped iCaRL cells without training or overwrites."""
import argparse
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import torch

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--base',type=Path,required=True)
p.add_argument('--workspace',type=Path,required=True)
p.add_argument('--destination',type=Path)
a=p.parse_args()
B=a.base.resolve(strict=True)
W=a.workspace.resolve(strict=True)
DEST=(a.destination or B/'admin/cse-icarl-completed-publication').resolve()
assert DEST.is_relative_to(B/'admin')
assert not DEST.exists(), 'Never overwrite an export'
read=lambda path:json.loads(path.read_text(encoding='utf-8'))
sha=lambda raw:hashlib.sha256(raw).hexdigest()
roots={10:B/'runs/section178/cic-ids-2018/icarl-scoped-npy/seed10'}
roots.update({s:B/f'validation/hpc-20261102T1345/a/icarl/seed{s}' for s in range(11,15)})
first=read(roots[11]/'cell/result.json')['identity']
remote_home=first['command'][0].split('/classil-20261102/')[0]
remote_scratch=first['effective_config']['problem_scout_data_root'].split('/classil-a-v2-20261102/')[0]
replacements=[(str(W).replace('\\','/'),'@@WORKSPACE@@'),(str(B).replace('\\','/'),'@@EXPERIMENT_ROOT@@'),
              (remote_home,'@@HPC_HOME@@'),(remote_scratch,'@@HPC_SCRATCH@@'),
              (os.environ.get('USERPROFILE','').replace('\\','/'),'@@LOCAL_HOME@@')]
def redact(text):
    for old,token in replacements:
        if old:
            for spelling in (old,old.replace('/','\\\\'),old.replace('/','\\')):
                text=text.replace(spelling,token)
    text=re.sub(r'(?<![A-Za-z])[CE]:[/\\]', '@@LOCAL_DRIVE@@/', text)
    text=text.replace(Path(remote_home).name,'@@HPC_USER@@')
    assert not re.search(r'/home/user/[A-Za-z0-9]|/scr/user/[A-Za-z0-9]|(?<![A-Za-z])[A-Za-z]:[/\\]|gh'+'p_[A-Za-z0-9]{20}|-----BE'+'GIN [A-Z ]*PRI'+'VATE KEY-----',text)
    return text
def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(redact(json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2))+'\n',encoding='utf-8',newline='\n')
def text(raw):
    for enc in ('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig','gb18030'):
        try:return raw.decode(enc)
        except UnicodeError:pass
    raise ValueError('Undecodable log')
# Scan before making an immutable destination, including prior failed-export
# tracebacks. Source-code regex examples are not themselves private keys.
for pattern in ('icarl-cse*.log','cse-icarl*.log','scoped-npy*.log','hpc-cse-icarl*.log','hpc-cse-nme*.log','hpc-frozen-source*.log','cse-job-a*.log','prediction-validation-submit*.log'):
    for path in (B/'logs').glob(pattern):
        redact(text(path.read_bytes()))
for pattern in ('*cse*icarl*.py','*scoped-npy*.py'):
    for path in (B/'admin').glob(pattern):
        try:
            redact(path.read_text(encoding='utf-8'))
        except AssertionError:
            raise ValueError('Source redaction requires inspection: '+path.name) from None
# Reuse a previously inspected pure-Python CM implementation, not its exporter.
tree=ast.parse((B/'admin/export-cse-ecbrs-paper-completed.py').read_text(encoding='utf-8'))
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='metrics')
namespace={'statistics':statistics}
exec(compile(ast.Module(body=[node],type_ignores=[]),'<pure CM math>','exec'),namespace)
metrics=namespace['metrics']
def events(path):
    return [json.loads(line) for line in text(path.read_bytes()).splitlines() if line.startswith('{')]
local_events=events(B/'logs/icarl-cse-scoped-seed10.monitor.stdout.log')
start=next(e for e in local_events if e.get('event')=='START')
terminal=next(e for e in local_events if e.get('event')=='TERMINAL')
assert terminal['exit_code']==0 and terminal['source_unchanged'] and any(e.get('event')=='CELL_VALIDATED' for e in local_events)
assert len(start['source_sha256'])==400
source403=next(e for e in events(B/'logs/cer-bic-common-cse-scoped-seed12.monitor.stdout.log') if e.get('event')=='START')
assert len(source403['source_sha256'])==403
assert all(source403['source_sha256'][k]==v for k,v in start['source_sha256'].items())
frozen_check=events(B/'logs/hpc-frozen-source-check-20261102.stdout.log')[-1]
assert frozen_check['files']==403 and not frozen_check['mismatches']
states=events(B/'logs/hpc-cse-icarl-states-20261102.stdout.log')[-1]
assert states['event']=='ICARL_STATES_COMPLETE' and states['feature_memberships']==5600
assert not (B/'logs/hpc-cse-icarl-states-20261102.stderr.log').stat().st_size
validation=B/'validation/hpc-493605-20261102/protected-prediction-validation-20261102'
for seed in range(11,15):
    proof=events(validation/f'cse-icarl-{seed}.stdout.log')[-1]
    assert proof['event']=='NME_COMPLETE' and proof['checkpoints']==4 and not proof['errors']
    assert not (validation/f'cse-icarl-{seed}.stderr.log').stat().st_size
payloads={}
all_metrics={}
costs=[]
reference=read(roots[10]/'cell/result.json')['identity']
for seed,root in roots.items():
    result=read(root/'cell/result.json')
    summary=read(root/'summary.json')
    identity=result['identity']
    assert result['completed'] and summary['completed'] and summary['scientific_result'] and not summary['smoke']
    assert identity['seed']==seed and identity['method']=='icarl' and identity['dataset']=='CSE-CIC-IDS2018'
    assert identity['data_binding']==reference['data_binding'] and identity['data_metadata_sha256']==reference['data_metadata_sha256']
    differences={k for k in identity['effective_config'] if identity['effective_config'][k]!=reference['effective_config'].get(k)}
    assert differences<= {'problem_scout_data_root'}
    result['default_stages']=read(root/'default_stages.json')
    result['calibrated_stages']=read(root/'calibrated_stages.json')
    assert len(result['checkpoints'])==len(result['default_stages'])==len(result['calibrated_stages'])==4
    records=[]
    memories=[]
    for cp in result['checkpoints']:
        path=root/'cell'/cp['checkpoint_file']
        raw=path.read_bytes()
        assert sha(raw)==cp['checkpoint_sha256'] and read(path.with_suffix('.json'))==cp
        state=torch.load(path,map_location='cpu',weights_only=True)
        assert state['identity']==identity
        records.append(dict(file=cp['checkpoint_file'],bytes=len(raw),sha256=sha(raw),metadata=cp))
        classes=[]
        for features,labels in zip(state['buffer_x'],state['buffer_y']):
            import numpy as np
            values=np.ascontiguousarray(features.numpy())
            c=int(labels[0])
            keys=values.view(np.dtype((np.void,values.shape[1]*values.dtype.itemsize))).ravel()
            classes.append(dict(class_id=c,slots=len(values),unique_class_feature_values=len(np.unique(keys)),feature_sha256=sha(values.tobytes()),label_sha256=sha(labels.numpy().tobytes())))
        memories.append(dict(task=cp['task_index'],classes=classes,unique_origin_row_ids_verified=False))
    per_view={view:metrics(result,view) for view in ('default','0.01','0.02','0.05')}
    all_metrics[str(seed)]=per_view
    cost=dict(seed=seed,training_seconds=sum(c['training_seconds'] for c in result['checkpoints']),
              buffer_selection_seconds=sum(c['buffer_selection_seconds'] for c in result['checkpoints']),summary_elapsed_seconds=summary['elapsed_seconds'],
              execution='local' if seed==10 else 'DICC493533 shared concurrent allocation',
              monitor_wall_seconds=terminal['elapsed_seconds'] if seed==10 else None,
              sampled_host_peak_bytes=terminal['peak_rss_bytes'] if seed==10 else None,
              host_peak_scope='owned child sampled peak' if seed==10 else 'not separately measured; allocation peak is not per-cell peak')
    costs.append(cost)
    payloads[seed]=(result,summary,records,memories,cost)
DEST.mkdir()
for seed,(result,summary,records,memories,cost) in payloads.items():
    out=DEST/f'icarl-scoped-npy_seed{seed}'
    put(out/'result.json',result)
    put(out/'summary.json',summary)
    put(out/'checkpoints.json',records)
    put(out/'memory_features.json',dict(stages=memories,limitation='Allowed TRAIN feature membership does not identify unique source rows when duplicate features exist.'))
    put(out/'costs.json',cost)
    put(out/'metrics.json',all_metrics[str(seed)])
    put(out/'source_state.json',dict(frozen_source_files=400 if seed==10 else 403,
        frozen_source_sha256=start['source_sha256'] if seed==10 else source403['source_sha256'],
        portable_entry_sha256=None if seed==10 else result['identity']['portable_entry_sha256'],
        original_result_sha256=sha((roots[seed]/'cell/result.json').read_bytes()),
        archive_redacts_personal_paths=True,checkpoint_binary_retained_outside_git=True))
overview=DEST/'overview'
put(overview/'per_seed_metrics.json',all_metrics)
aggregate={}
perclass={}
for view in ('default','0.01','0.02','0.05'):
    aggregate[view]={key:dict(mean=statistics.mean(all_metrics[str(s)][view][key] for s in roots),sample_sd=statistics.stdev(all_metrics[str(s)][view][key] for s in roots)) for key in ('bacc','accuracy','macro_f1','normal_fpr','attack_recall','incremental_bacc','incremental_accuracy','forgetting')}
    perclass[view]=[]
    for j in range(7):
        item=dict(class_id=j,name=reference['class_order'][j])
        for key in ('first','peak','best_prior','final','forgetting'):
            values=[all_metrics[str(s)][view]['per_class'][j][key] for s in roots]
            item[key]=None if values[0] is None else dict(mean=statistics.mean(values),sample_sd=statistics.stdev(values))
        perclass[view].append(item)
put(overview/'aggregate.json',aggregate)
put(overview/'per_class_aggregate.json',perclass)
put(overview/'costs.json',dict(per_seed=costs,note='Training, selection and elapsed overlap; concurrent seed elapsed sums are not job wall. CPU hardware differs for seed10 versus11–14.'))
put(overview/'scope.json',dict(dataset='CSE-CIC-IDS2018',method='herding + softmax-KL + NME scoped iCaRL adaptation',seeds=list(roots),registered_scope='Existing CSE S1 baseline block, not NSL extension',
    limitations=['Not exact original sigmoid-BCE reproduction','Feature membership is not unique source-row provenance','Windows float32 boundary predictions differ; original Linux independent verification exactly matches','Shared allocation host peak is not per-cell peak','RQ1/RQ3, substantive innovation and overall S1 remain open'],
    frozen_patch_reference='paper_aacil/results/section178_literature/cic-ids-2018/ecbrs-paper_seed10/execution_source.patch',patch_base_git_commit=start['git_commit'],new_attempt_job=493533,prediction_verification_job=493605))
log_paths=set()
for pattern in ('icarl-cse*.log','cse-icarl*.log','scoped-npy*.log','hpc-cse-icarl*.log','hpc-cse-nme*.log','hpc-frozen-source*.log','cse-job-a*.log','prediction-validation-submit*.log'):
    log_paths.update((B/'logs').glob(pattern))
log_paths.update((B/'validation/hpc-20261102T1345/a').glob('icarl-*.log'))
log_paths.update((B/'validation/hpc-493605-20261102').rglob('*.log'))
log_paths.update((B/'validation/hpc-493605-20261102').glob('slurm-*'))
log_paths.update((B/'hpc-cse-job-a-20261102/failure-logs').glob('493528*'))
logs=[]
for path in sorted(log_paths):
    raw=path.read_bytes()
    name=path.name
    target=overview/'logs'/(name+'.gz')
    assert not target.exists(),name
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(gzip.compress(redact(text(raw)).encode('utf-8'),mtime=0))
    logs.append(dict(file='logs/'+name+'.gz',original_sha256=sha(raw),redacted=True))
put(overview/'log_artifacts.json',logs)
scripts=list((B/'admin').glob('*cse*icarl*.py'))+list((B/'admin').glob('*scoped-npy*.py'))
scripts += [B/'admin/check-hpc-cse-nme-predictions.py',B/'admin/verify-hpc-predictions.py',B/'admin/verify-hpc-predictions.sbatch',Path(__file__)]
for path in sorted(set(scripts)):
    (overview/(path.name+'.txt')).write_text(redact(path.read_text(encoding='utf-8')).rstrip()+'\n',encoding='utf-8',newline='\n')
for name in ('run_cse_bic_common.py','run_cse_bic_scoped.py','cse_bic_stage1_reuse.py'):
    path=W/'work/section151_publish/ncd'/name
    assert sha(path.read_bytes())==source403['source_sha256']['ncd/'+name]
    (overview/(name+'.txt')).write_text(redact(path.read_text(encoding='utf-8')),encoding='utf-8',newline='\n')
print(json.dumps(dict(event='EXPORTED_CSE_ICARL',seeds=5,checkpoints=20,calibrated_views=60,artifacts=sum(p.is_file() for p in DEST.rglob('*')),aggregate=aggregate)),flush=True)
