"""Freeze the completed five-seed CSE BiC group; reuse existing scientific checks.

No training, model inference, old queue guards, or existing export replacement.
"""
import argparse,ast,copy,gzip,hashlib,json,re,statistics
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();B=a.base;W=a.workspace
assert not a.output.exists(),'Preserve every earlier complete or partial export'
torch.set_num_threads(2)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def decode(raw):
    for enc in (('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig'),'gb18030'):
        try:return raw.decode(enc)
        except UnicodeError:pass
    raise ValueError('Undecodable evidence log')
def redact(text):
    # Longest known roots first; do not change scientific numeric values.
    for root,token in ((W.as_posix(),'@@WORKSPACE@@'),(B.as_posix(),'@@EXPERIMENT_ROOT@@'),((Path.home()/'miniconda3/envs/research').as_posix(),'@@PYTHON_ENV@@'),(Path.home().as_posix(),'@@LOCAL_HOME@@')):
        for spelling in (root,root.replace('/','\\\\'),root.replace('/','\\')):text=text.replace(spelling,token)
    text=re.sub(r'@@HPC_HOME@@/"\\\s]+','@@HPC_HOME@@',text)
    text=re.sub(r'@@HPC_SCRATCH@@/"\\\s]+','@@HPC_SCRATCH@@',text)
    text=re.sub(r'(?i)(?<![A-Za-z])[CE]:[/\\]+','@@LOCAL_DRIVE@@/',text)
    assert not re.search(r'@@HPC_HOME@@@@HPC_SCRATCH@@/\\]|ghp_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----',text)
    return text
def events(p):return [json.loads(v) for v in decode(p.read_bytes()).splitlines() if v.startswith('{')]
def pure_functions(path,names,env):
    tree=ast.parse(path.read_text(encoding='utf-8'));nodes=[v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name in names]
    assert {v.name for v in nodes}==set(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env);return env
first=pure_functions(B/'admin/export-cer-bic-common-cic17-completed.py',('independent','compare'),{'np':np})
second=pure_functions(B/'admin/export-cse-ecbrs-paper-completed.py',('metrics',),{'statistics':statistics})
roots={10:B/'runs/section178/cic-ids-2018/cer-bic-common/seed10',11:B/'runs/section178/cic-ids-2018/cer-bic-common-scoped/seed11',12:B/'runs/section178/cic-ids-2018/cer-bic-common-scoped/seed12',13:B/'validation/hpc-20261102T1345/a/bic/seed13',14:B/'validation/hpc-493652-bic14-20261102/seed14'}
proof13=events(B/'logs/hpc-cse-bic13-states-20261102.stdout.log')[-1];proof14=events(B/'logs/hpc-cse-bic14-states-20261102.stdout.log')[-1]
assert proof13['exact_imported_payloads']==6 and proof14['resume_completed_epochs_preserved']==6 and proof14['rng_restored'] is False
for seed in (13,14):assert not (B/f'logs/hpc-cse-bic{seed}-states-20261102.stderr.log').stat().st_size
for seed,pred in ((13,B/'validation/hpc-493612-20261102/protected-bic13-prediction-validation-20261102/check.stdout.log'),(14,B/'validation/hpc-493652-bic14-20261102/prediction/bic14-original.stdout.log')):
    if seed==13 and not pred.exists():
        matches=list((B/'validation/hpc-493612-20261102').rglob('*.stdout.log'));assert len(matches)==1;pred=matches[0]
    proof=events(pred)[-1];assert proof['event']=='BIC_PREDICTIONS_COMPLETE' and proof['checkpoints']==4 and not proof['failures']
    assert not Path(str(pred).replace('.stdout.log','.stderr.log')).stat().st_size
views=('default','0.01','0.02','0.05');rows=[];payload={};reference=None
for seed,root in roots.items():
    r=read(root/'result.json');i=r['identity'];summary=read(root/'summary.json')
    assert r['completed'] and summary['completed'] and not i['smoke'] and i['seed']==seed
    assert i['method']=='cer-bic-flow-cnn-common' and i['dataset']=='CSE-CIC-IDS2018'
    assert i['effective_config']['schedule']=='cer_common_adam_8epochs_min2000' and len(r['checkpoints'])==4
    if reference is None:reference=i
    for key in ('class_order','tasks','data_binding','data_metadata_sha256','original_train_counts'):assert i[key]==reference[key],key
    assert {k for k in i['effective_config'] if i['effective_config'][k]!=reference['effective_config'].get(k)}<={'problem_scout_data_root'}
    terminal=None
    if seed<=12:
        stem='cer-bic-common-cse-' if seed==10 else 'cer-bic-common-cse-scoped-'
        log=events(B/f'logs/{stem}seed{seed}.monitor.stdout.log');terminal=next(v for v in log if v.get('event')=='TERMINAL')
        assert terminal['exit_code']==0 and terminal['source_unchanged'] and any(v.get('event')=='CELL_VALIDATED' for v in log)
        validation=events(B/f'logs/{stem}seed{seed}.validation.stdout.log');assert any(v.get('validated') and v.get('checkpoints')==4 and v.get('calibrations')==12 for v in validation)
    records=[];memories=[]
    for cp in r['checkpoints']:
        path=root/cp['checkpoint_file'];assert sha(path)==cp['checkpoint_sha256'] and read(path.with_suffix('.json'))==cp
        state=torch.load(path,map_location='cpu',weights_only=True);assert state['identity']==i
        roles={key:{str(c):ids.tolist() for c,ids in state[key].items()} for key in ('replay','correction')}
        ids=[v for classes in roles.values() for values in classes.values() for v in values]
        assert len(ids)==len(set(ids))==cp['unique_row_ids']==cp['memory_slots']
        progress=root/f"progress/backbone_{cp['task_index']:03d}.pt"
        records.append(dict(file=cp['checkpoint_file'],bytes=path.stat().st_size,sha256=sha(path),metadata=cp,progress_sha256=sha(progress)))
        memories.append(dict(task=cp['task_index'],retained_original_train_row_ids=roles,unique_row_ids=len(ids)))
    metrics={v:first['independent'](r,v) for v in views}
    for v in views:
        other=second['metrics'](r,v);other.pop('incremental_accuracy');first['compare'](metrics[v],other)
    total=sum(cp['training_seconds'] for cp in r['checkpoints']);assert abs(total-r['training_seconds_total'])<1e-8
    imported=(i.get('stage1_reuse') or {}).get('original_training_seconds',0.)
    if seed==13:imported=sum(cp['training_seconds'] for cp in r['checkpoints'][:3])
    preserved=13903.58767033182 if seed==14 else 0.
    costs=dict(seed=seed,recorded_training_seconds_total=total,prior_imported_training_seconds=imported,resume_preexisting_epoch_seconds=preserved,
        newly_recorded_training_seconds=total-imported-preserved,backbone_seconds=sum(cp['training_schedule']['seconds'] for cp in r['checkpoints']),
        bias_seconds=sum(cp['bias_schedule']['seconds'] for cp in r['checkpoints']),selection_seconds=sum(cp['selection_seconds'] for cp in r['checkpoints']),
        summary_elapsed_seconds=summary['elapsed_seconds'],local_sampled_monitor=terminal,
        linux_new_attempt_job=493533 if seed==13 else 493652 if seed==14 else None,
        host_peak_scope='Local owned-process sampled RSS' if seed<=12 else 'Seed13 shared allocation has no per-cell peak; seed14 continuation step peak6119196K excludes original shared attempt',
        overlap_warning='Training, selection, summary and monitor measurements overlap. Old interrupted partial work and concurrent job wall are not summed as serial per-cell training; mixed hardware is not a speedup comparison.')
    assert abs(costs['backbone_seconds']+costs['bias_seconds']-total)<1e-8
    publish_result=copy.deepcopy(r)
    if seed==14:
        publish_result['original_checkpoint_identity']=copy.deepcopy(i)
        publish_result['identity']['resume']=dict(from_epoch=6,rng_restored=False,original_job=493533,resumed_job=493652)
    row=dict(seed=seed,views=metrics,costs=costs);rows.append(row)
    payload[seed]=(publish_result,summary,records,memories,row,i)
aggregate={v:{key:dict(mean=statistics.mean(row['views'][v][key] for row in rows),sample_sd=statistics.stdev(row['views'][v][key] for row in rows)) for key in ('bacc','accuracy','macro_f1','normal_fpr','attack_recall','incremental_bacc','forgetting')} for v in views}
perclass={v:[dict(class_id=c,name=reference['class_order'][c],**{key:None if rows[0]['views'][v]['per_class'][c][key] is None else dict(mean=statistics.mean(row['views'][v]['per_class'][c][key] for row in rows),sample_sd=statistics.stdev(row['views'][v]['per_class'][c][key] for row in rows)) for key in ('first','peak','best_prior','final','forgetting')}) for c in range(7)] for v in views}
logpaths=set()
for pattern in ('cer-bic-common-cse-*.log','cse-bic-*.log','bic14-*.log','bic13-prediction*.log','hpc-cse-bic*.log','hpc-bic14-*.log','cse-job-a*.log','hpc-frozen-source*.log'):
    logpaths.update(p for p in (B/'logs').glob(pattern) if p.is_file() and not p.name.startswith('cse-bic-completed-export-'))
for folder in (B/'validation/hpc-493652-bic14-20261102/logs',B/'validation/hpc-493652-bic14-20261102/prediction',B/'validation/hpc-493612-20261102',B/'validation/hpc-493533-rightsize-20261102/protected-a493533-rightsize-20261102'):
    logpaths.update(p for p in folder.rglob('*') if p.is_file() and (p.suffix in ('.log','.out','.err')))
sources=[B/'admin'/n for n in ('export-cse-bic-completed.py','check-hpc-cse-bic-states.py','check-hpc-cse-bic-predictions.py','deliver-cse-bic14-raw.py','resume-cse-bic14-rightsized.sbatch','check-cer-bic-common.py','export-cer-bic-common-cic17-completed.py')]
# Pre-scan all text before creating a never-overwritten export root.
for path in logpaths:redact(decode(path.read_bytes()))
for path in sources:redact(path.read_text(encoding='utf-8'))
a.output.mkdir()
def put(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(redact(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))+'\n',encoding='utf-8',newline='\n')
for seed,(r,s,records,memories,row,i) in payload.items():
    dest=a.output/f'cer-bic-common-completed_seed{seed}'
    for name,value in (('result.json',r),('summary.json',s),('checkpoints.json',records),('retained_row_ids.json',memories),('metrics.json',row['views']),('costs.json',row['costs']),('source_state.json',dict(original_result_sha256=sha(roots[seed]/'result.json'),bound_source_sha256=i['source_sha256'],git_commit=i['git_commit'],portable_entry_sha256=i.get('portable_entry_sha256'),checkpoint_binaries_retained_outside_git=True))):put(dest/name,value)
overview=a.output/'overview';put(overview/'aggregate.json',aggregate);put(overview/'per_class_aggregate.json',perclass);put(overview/'per_seed_metrics.json',rows)
reuse=payload[11][5]['stage1_reuse'];failed=B/'runs/section178/cic-ids-2018/cer-bic-common/seed11'
assert sha(failed/'identity.json')==reuse['identity_sha256'] and sha(failed/'progress/backbone_000.pt')==reuse['progress_sha256']
put(overview/'failed_attempt_and_resume_evidence.json',dict(seed11_failed_identity=read(failed/'identity.json'),seed11_stage1_reuse= reuse,
    seed13_completed_task_import_proof=proof13,seed14_epoch6_resume_proof=proof14,
    seed14_original_job=493533,seed14_completion_job=493652,seed14_continuation_accounting=dict(wall_seconds=8138,total_cpu_seconds=6830,step_peak_rss_kib=6119196,scope='Continuation attempt only; not the original concurrent allocation or uninterrupted whole-cell peak'),
    note='Failed/interrupted attempts remain immutable outside Git. These reused records are not new training or repeated acceptance scans.'))
put(overview/'scope.json',dict(dataset='CSE-CIC-IDS2018',method='common-budget flow CNN BiC; not native packet CER',seeds=list(roots),checkpoints=20,calibrations=60,
    source_scope='seed10 local401; seeds11/12 local403; seeds13/14 Linux403 plus explicit portable entry. Source bindings remain distinct.',
    runtime_scope='Original Windows evaluation for10–12; Linux evaluation for13–14. Seed13 imports Windows task0–2 tensors without reoptimization. Uniform same-score evaluation493707 is separate, not original-prior runtime harmonization.',
    evidence_scope='Reuse completed original role/source/prediction checks for10–13; new14 role/schedule/resume check and original Linux prediction exact. No re-training or old verification rerun.',
    limitations=['Global RNG states absent at epoch6 resume; rng_restored=false','Recorded costs overlap; mixed hardware is not a speedup benchmark','Original failed attempts and imported training are retained','Overall S1, RQ1/RQ3, substantive innovation, manuscript QA and backup reconciliation are not closed'],
    frozen_source_reference='paper_aacil/results/section178_literature/cic-ids-2018/ecbrs-paper_seed10/execution_source.patch'))
put(overview/'consumer_aliases.json',dict(canonical_completed_scope='cer-bic-common-completed_seed10..14',raw_consumer_scope='cer-bic-common_seed10..14',same_five_cells=True,additional_experiments_or_replicates=False))
logrecords=[]
for path in sorted(logpaths):
    rel=path.relative_to(B).as_posix();name=rel.replace('/','__')+'.gz';raw=path.read_bytes();dest=overview/'logs'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(gzip.compress(redact(decode(raw)).encode(),mtime=0));logrecords.append(dict(source_relative=rel,artifact='logs/'+name,original_sha256=hashlib.sha256(raw).hexdigest(),original_bytes=len(raw)))
put(overview/'log_artifacts.json',logrecords)
for path in sources:
    dest=overview/'source'/path.name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(redact(path.read_text(encoding='utf-8')),encoding='utf-8',newline='\n')
for path in a.output.rglob('*'):
    if path.is_file():redact(decode(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()))
print(json.dumps(dict(event='CSE_BIC_COMPLETE_EXPORT',artifacts=sum(p.is_file() for p in a.output.rglob('*')),checkpoints=20,calibrations=60,aggregate=aggregate)),flush=True)
