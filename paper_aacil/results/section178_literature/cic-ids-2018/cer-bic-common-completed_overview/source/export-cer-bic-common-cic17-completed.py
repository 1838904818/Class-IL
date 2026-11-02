"""Archive five completed common-budget flow BiC cells without retraining."""
import difflib
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import numpy as np
import torch

BASE = Path('@@EXPERIMENT_ROOT@@')
REPO = Path('@@WORKSPACE@@/work/section151_publish')
DEST = BASE/'admin/cer-bic-common-cic17-publication'
VIEWS = ('default','0.01','0.02','0.05')
KEYS = ('bacc','accuracy','macro_f1','normal_fpr','attack_recall','incremental_bacc','forgetting')
SCHEDULE = 'cer_common_adam_8epochs_min2000'
spec = importlib.util.spec_from_file_location('established_helpers', BASE/'admin/export-cic17-ecbrs-paper-completed.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def events(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.startswith('{')]

def independent(result, view):
    """NumPy recomputation directly from CMs; does not call the export metric helper."""
    recalls = np.full((4,len(result['identity']['class_order'])),np.nan)
    stages = []
    for t, cp in enumerate(result['checkpoints']):
        point = (result['default_stages'][t]['metrics'] if view=='default'
                 else result['calibrated_stages'][t]['operating_points'][view])
        cm = np.asarray(point['confusion_matrix'],dtype=np.float64)
        support, predicted, diag = cm.sum(axis=1), cm.sum(axis=0), cm.diagonal()
        seen = np.asarray(cp['seen_classes'])
        assert np.all(support[seen]>0) and np.all(cm>=0)
        recalls[t,seen] = diag[seen]/support[seen]
        stages.append(dict(bacc=float(np.mean(recalls[t,seen])),accuracy=float(diag.sum()/cm.sum()),
            macro_f1=float(np.mean(2*diag[seen]/(support+predicted)[seen])),
            normal_fpr=float(cm[0,1:].sum()/support[0]),
            attack_recall=float(cm[1:,1:].sum()/support[1:].sum())))
    classes = []
    for first, task in enumerate(result['identity']['tasks']):
        for c in task:
            prior = float(np.max(recalls[first:-1,c])) if first<3 else None
            classes.append(dict(class_id=c,name=result['identity']['class_order'][c],
                first=float(recalls[first,c]),peak=float(np.max(recalls[first:,c])),
                best_prior=prior,final=float(recalls[-1,c]),
                forgetting=prior-float(recalls[-1,c]) if prior is not None else None))
    return dict(**stages[-1],incremental_bacc=float(np.mean([s['bacc'] for s in stages])),
        forgetting=float(np.mean([c['forgetting'] for c in classes if c['forgetting'] is not None])),
        stages=stages,per_class=classes)

def compare(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: compare(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b): compare(x,y)
    elif isinstance(a,(int,float)):
        assert abs(a-b)<1e-12,(a,b)
    else: assert a==b,(a,b)

def main():
    assert not DEST.exists(), 'Never overwrite a completed or partial export'
    os.environ.update(TEMP=str(BASE/'tmp'),TMP=str(BASE/'tmp'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    torch.set_num_threads(2)
    queue = BASE/'logs/cer-bic-common-cic17-queue-11-14.stdout.log'
    assert 'ALL_COMPLETE' in queue.read_text(encoding='utf-8')
    assert not (BASE/'logs/cer-bic-common-cic17-queue-11-14.stderr.log').stat().st_size
    smoke = h.read(BASE/'runs/section178/cic-ids-2017/cer-bic-common-smoke-seed10/result.json')
    assert smoke['completed'] and not smoke['scientific_result'] and smoke['identity']['smoke']
    bound = smoke['identity']['source_sha256']
    assert len(bound)==398 and all(digest(REPO/p)==v for p,v in bound.items())
    formal_identity = h.read(BASE/'runs/section178/cic-ids-2017/cer-bic-common/seed10/result.json')['identity']
    validated, rows = [], []
    for seed in range(10,15):
        src = BASE/f'runs/section178/cic-ids-2017/cer-bic-common/seed{seed}'
        result, summary = h.read(src/'result.json'),h.read(src/'summary.json')
        ident = result['identity']
        assert result['completed'] and result['scientific_result'] and not ident['smoke']
        assert ident['seed']==seed and ident['dataset']=='CIC-IDS-2017'
        assert ident['effective_config']['schedule']==SCHEDULE
        assert ident['source_sha256']==bound and summary['method']=='cer-bic-flow-cnn-common'
        assert summary['completed'] and summary['scientific_result'] and not summary['smoke']
        assert len(result['checkpoints'])==len(result['default_stages'])==len(result['calibrated_stages'])==4
        for key in ('tasks','class_order','data_metadata_sha256','git_commit'):
            assert ident[key]==smoke['identity'][key]
        # Smoke arrays are deliberately subsampled; full input equality is across formal seeds.
        for key in ('data_binding','effective_config','original_train_counts'):
            assert ident[key]==formal_identity[key]
        monitor = events(BASE/f'logs/cer-bic-common-cic17-seed{seed}.monitor.stdout.log')
        start = next(e for e in monitor if e.get('event')=='START')
        terminal = next(e for e in monitor if e.get('event')=='TERMINAL')
        assert start['source_sha256']==bound
        assert terminal['exit_code']==0 and terminal['reason'] is None and terminal['source_unchanged']
        assert terminal['wall_seconds']<3300 and terminal['sampled_peak_rss_bytes']<10*1024**3
        assert any(e.get('event')=='CELL_VALIDATED' for e in monitor)
        for kind in ('','monitor.','validation.'):
            assert not (BASE/f'logs/cer-bic-common-cic17-seed{seed}.{kind}stderr.log').stat().st_size
        # A new read-only check preserves both original validation logs and this invocation.
        prefix = BASE/f'logs/cer-bic-common-cic17-seed{seed}.export-validation'
        with Path(str(prefix)+'.stdout.log').open('xb') as out, Path(str(prefix)+'.stderr.log').open('xb') as err:
            checked = subprocess.run([sys.executable,'-I','-B','-u',str(BASE/'admin/check-cer-bic-common.py'),str(src)],stdout=out,stderr=err)
        assert checked.returncode==0, 'Preserve failed checker logs; do not retrain'
        verification = Path(str(prefix)+'.stdout.log').read_text(encoding='utf-8')
        check = json.loads(verification)
        assert check['validated'] and check['checkpoints']==4 and check['calibrations']==12
        retention = []
        for cp in result['checkpoints']:
            path = src/cp['checkpoint_file']
            assert digest(path)==cp['checkpoint_sha256']
            state = torch.load(path,map_location='cpu',weights_only=False)
            roles = {key:{str(c):v.tolist() for c,v in state[key].items()} for key in ('replay','correction')}
            flat = [r for role in roles.values() for ids in role.values() for r in ids]
            assert len(flat)==len(set(flat))==cp['unique_row_ids']==cp['memory_slots']
            progress = src/f"progress/backbone_{cp['task_index']:03d}.pt"
            retention.append(dict(task_index=cp['task_index'],checkpoint_file=cp['checkpoint_file'],
                checkpoint_sha256=cp['checkpoint_sha256'],checkpoint_bytes=path.stat().st_size,
                backbone_progress_file=progress.relative_to(src).as_posix(),backbone_progress_sha256=digest(progress),
                retained_row_ids=roles,memory_slots=len(flat),unique_row_ids=len(set(flat)),
                row_id_semantics='Zero-based original registered TRAIN-array row index; not feature deduplication.'))
        views = {v:independent(result,v) for v in VIEWS}
        for v in VIEWS: compare(views[v],h.metrics(result,v))
        cps=result['checkpoints']
        costs=dict(backbone_training_seconds=sum(c['training_schedule']['seconds'] for c in cps),
            bias_training_seconds=sum(c['bias_schedule']['seconds'] for c in cps),
            training_seconds_including_bias=sum(c['training_seconds'] for c in cps),
            herding_selection_seconds=sum(c['selection_seconds'] for c in cps),
            calibrated_evaluation_seconds=sum(c['evaluation_seconds'] for c in result['calibrated_stages']),
            calibration_seconds=sum(c['calibration_seconds'] for c in result['calibrated_stages']),
            summary_elapsed_seconds=summary['elapsed_seconds'],monitor_wall_seconds=terminal['wall_seconds'],
            sampled_peak_rss_bytes=terminal['sampled_peak_rss_bytes'])
        assert abs(costs['backbone_training_seconds']+costs['bias_training_seconds']-costs['training_seconds_including_bias'])<1e-8
        assert abs(costs['training_seconds_including_bias']-result['training_seconds_total'])<1e-8
        row=dict(seed=seed,views=views,costs=costs,memory_by_stage=[{k:r[k] for k in ('task_index','memory_slots','unique_row_ids')} for r in retention])
        rows.append(row)
        validated.append((seed,src,result,summary,start,terminal,verification,retention,row))
        print('VERIFIED_SEED',seed,flush=True)
    base=smoke['identity']['git_commit']
    patch=subprocess.check_output(['git','-C',str(REPO),'diff',base,'--',*bound],text=True,encoding='utf-8')
    for path in bound:
        if subprocess.run(['git','-C',str(REPO),'cat-file','-e',base+':'+path],capture_output=True).returncode:
            content=(REPO/path).read_text(encoding='utf-8')
            patch+=f'diff --git a/{path} b/{path}\nnew file mode 100644\n'
            patch+=''.join(difflib.unified_diff([],content.splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+path))
    reverse=subprocess.run(['git','-C',str(REPO),'apply','--check','--reverse','-'],input=patch.encode(),capture_output=True)
    assert reverse.returncode==0,reverse.stderr
    public=h.redact(patch)
    restored=public
    for old,token in h.REPLACEMENTS: restored=restored.replace(token,old)
    assert restored==patch
    aggregate={v:{k:dict(mean=float(np.mean([r['views'][v][k] for r in rows])),
        sample_sd=float(np.std([r['views'][v][k] for r in rows],ddof=1))) for k in KEYS} for v in VIEWS}
    for v in VIEWS:
        for k in KEYS:
            vals=[h.metrics(x[2],v)[k] for x in validated]
            compare(aggregate[v][k],dict(mean=statistics.mean(vals),sample_sd=statistics.stdev(vals)))
    logs=[f'cer-bic-common-cic17-queue-11-14.{s}.log' for s in ('stdout','stderr')]
    logs += [f'cer-bic-common-cic17-smoke.monitor.{s}.log' for s in ('stdout','stderr')]
    logs += [f'cer-bic-common-tests.{s}.log' for s in ('stdout','stderr')]
    logs += [f'cer-bic-common-cic17-seed10.smoke.{k}{s}.log' for k in ('','validation.') for s in ('stdout','stderr')]
    # Preserve the native CIC budget-failure evidence that triggered the override.
    logs += [f'cer-bic-cic17-seed10.{k}{s}.log' for k in ('','monitor.','smoke.','smoke.validation.') for s in ('stdout','stderr')]
    logs += [f'cer-bic-cic17-smoke.monitor.{s}.log' for s in ('stdout','stderr')]
    admin=('check-cer-bic-common.py','monitor-cer-bic-common-cic17.py',
           'monitor-cer-bic-common-cic17-followup.py','queue-cer-bic-common-cic17-11-14.py')
    payload={name+'.txt':h.redact((BASE/'admin'/name).read_text(encoding='utf-8')) for name in admin}
    payload['test_cer_bic_common.py.txt']=h.redact((REPO/'tests/test_cer_bic_common.py').read_text(encoding='utf-8'))
    for name in logs: h.redact((BASE/'logs'/name).read_text(encoding='utf-8'))
    for x in validated: h.redact(json.dumps(x[2]))
    assert all(digest(REPO/p)==v for p,v in bound.items())
    DEST.mkdir()
    scope=('BiC with the Section192 family-wide common stage-1 override on registered flow-vector single-channel Conv1D inputs, '
        'not native packet 10x4 reproduction. Architecture and stage-2 values are explicit engineering defaults. '
        'Registered multi-stage tasks and 200 unique TRAIN rows per seen class including Normal replace the paper protocol. '
        'Replay and correction rows are persistent without old-class refresh. All five CIC17 cells were newly run under this separate common identity as explicitly authorized by Section192; native results remain untouched historical sensitivity evidence, excluded from this comparison family. Stage1 uses Adam lr0.001 batch256 max(8 full epochs,2000 steps), final iterate, no early stopping or decay. Separate BiC stage2 uses the unchanged explicit engineering default of 200 SGD epochs. '
        'S1 four-method Holm family, RQ1, RQ3 and substantive innovation gate remain incomplete.')
    for seed,src,result,summary,start,terminal,verification,retention,row in validated:
        out=DEST/f'cer-bic-common_seed{seed}'
        out.mkdir()
        names=[f'cer-bic-common-cic17-seed{seed}.{k}{s}.log' for k in ('','monitor.','validation.','export-validation.') for s in ('stdout','stderr')]
        result.update(execution_summary=summary,command=start['command'],source_state=dict(base_commit=base,
            source_sha256=bound,dirty_patch='execution_source.patch'),resource_verification=terminal,
            publication_scope=scope,cost_accounting=row['costs'],log_artifacts=h.log_export(out,names))
        h.write_json(out/'result.json',result)
        h.write_json(out/'independent_metrics.json',row)
        h.write_json(out/'retained_row_ids_and_checkpoint_hashes.json',retention)
        (out/'execution_source.patch').write_text(public,encoding='utf-8',newline='\n')
        (out/'verification.txt').write_text(h.redact(verification),encoding='utf-8',newline='\n')
    overview=DEST/'overview'
    overview.mkdir()
    h.write_json(overview/'aggregate.json',aggregate)
    native_dir=BASE/'runs/section178/cic-ids-2017/cer-bic/seed10'
    native_terminal=next(x for x in events(BASE/'logs/cer-bic-cic17-seed10.monitor.stdout.log') if x.get('event')=='TERMINAL')
    assert native_terminal['exit_code']!=0 and native_terminal['source_unchanged']
    assert not (native_dir/'result.json').exists()
    h.write_json(overview/'native_budget_failure.json',dict(scientific_result=False,terminal=native_terminal,
        identity_sha256=digest(native_dir/'identity.json'),progress_sha256=digest(native_dir/'progress/backbone_000.pt'),
        interpretation='Time-budget termination before first complete task, not algorithm performance failure; partial binaries remain preserved; no resume or overwrite.'))
    h.write_json(overview/'per_seed_metrics.json',rows)
    h.write_json(overview/'costs_and_scope.json',dict(scope=scope,seeds=list(range(10,15)),checkpoints=20,calibrations=60,
        cost_totals={k:sum(r['costs'][k] for r in rows) for k in rows[0]['costs'] if k!='sampled_peak_rss_bytes'},
        sampled_peak_rss_bytes=max(r['costs']['sampled_peak_rss_bytes'] for r in rows),
        cost_notes='Training includes stage-1 and bias, excludes herding; calibration/evaluation and summary elapsed are overlapping measurements and must not be added together. RSS is sampled, not exact peak. Original checkpoint and progress binaries remain preserved on the experiment drive, identified by SHA256; retained real row IDs are exported.',
        metric_notes='Final and incremental balanced accuracy; macro-F1 over seen classes; signed best-prior-minus-final recall forgetting over classes introduced before the final task. Mean and sample SD (ddof=1) over five seeds. Each view uses its own stage CMs; no cross-view forgetting.'))
    h.write_json(overview/'log_artifacts.json',h.log_export(overview,logs))
    for name,content in payload.items(): (overview/name).write_text(content,encoding='utf-8',newline='\n')
    print(json.dumps(dict(output=str(DEST),artifacts=sum(p.is_file() for p in DEST.rglob('*')),checkpoints=20,calibrations=60,aggregate=aggregate)),flush=True)

if __name__=='__main__': main()
