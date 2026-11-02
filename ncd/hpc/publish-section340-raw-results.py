"""Deliver existing completed results at analysis-consumer paths; no model evaluation or training."""
import argparse
import hashlib
import json
import msvcrt
from pathlib import Path
import re
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--base', type=Path, required=True)
p.add_argument('--workspace', type=Path, required=True)
p.add_argument('--publish', action='store_true')
a = p.parse_args()
B, W = a.base.resolve(strict=True), a.workspace.resolve(strict=True)
read = lambda x: json.loads(x.read_text(encoding='utf-8-sig'))
digest = lambda x: hashlib.sha256(x).hexdigest()
snapshot = B/'validation/hpc-20261102T1345/b/protected-b-v2'
linux = dict(platform='Linux x86_64, glibc2.34', python='3.11.9', torch='2.6.0+cu118', sklearn='1.6.1')
overlay = dict(linux, sklearn='1.7.2')
windows = dict(platform='Windows-10-10.0.22631-SP0', python='3.10.20', torch='2.6.0+cu124', sklearn='1.7.2')
files = {}
records = []

def redact(text):
    # Preserve all numerical result fields while replacing machine-specific roots.
    for root, token in ((str(W), '@@WORKSPACE@@'), (str(B), '@@EXPERIMENT_ROOT@@')):
        for spelling in (root, root.replace('\\', '/'), root.replace('\\', '\\\\')):
            text = text.replace(spelling, token)
    text = re.sub(r'/home/user/[^/"\\\s]+', '@@HPC_HOME@@', text)
    text = re.sub(r'/scr/user/[^/"\\\s]+', '@@HPC_SCRATCH@@', text)
    text = re.sub(r'(?i)[A-Z]:[/\\]+Users[/\\]+[^/\\"\s]+', '@@LOCAL_HOME@@', text)
    text = re.sub(r'(?i)[CE]:[/\\]+', '@@LOCAL_DRIVE@@/', text)
    return text

def stages(r, root):
    for field in ('default_stages', 'calibrated_stages'):
        path = root/(field+'.json')
        if field not in r and path.exists():
            r[field] = read(path)
    for j, cp in enumerate(r['checkpoints']):
        default = cp.get('default_decision') or r['default_stages'][j]
        default = default.get('metrics', default)
        cal = cp if 'operating_points' in cp else r['calibrated_stages'][j]
        assert 'confusion_matrix' in default
        assert 'confusion_matrix' in cal['operating_points']['0.01']
        assert cp['seen_classes']
    # Original calibration objects contain taus/order statistics and row-ID hashes.
    text = json.dumps(r)
    assert ('tau' in text or 'threshold' in text) and ('indices_sha256' in text or 'row_ids_sha256' in text)

def add(root, target, runtime, job, status='pending independent scientific acceptance', note=''):
    source = root/'result.json'
    if not source.exists():
        source = root/'cell/result.json'
    r = read(source)
    assert r['completed'] and read(root/'summary.json')['completed']
    stages(r, root)
    identity = r['identity']
    if 'class_order' not in r:r['class_order']=identity['class_order']
    if 'tasks' not in r:r['tasks']=identity['tasks']
    r['runtime'] = runtime
    r['delivery'] = dict(status=status, job_id=job, source_result_sha256=digest(source.read_bytes()),
        source_result=source.relative_to(B).as_posix(), original_result_fields_preserved=True,
        no_training_or_model_changes=True, note=note,
        runtime_version_evidence='Same installed runtime inspected after execution; pinned job environment and entry overlay preserved. Not an original per-cell host telemetry sample.')
    # Copy calibration evidence verbatim rather than recomputing thresholds.
    evidence=[]
    for j, cp in enumerate(r['checkpoints']):
        cal = cp if 'operating_points' in cp else r['calibrated_stages'][j]
        evidence.append(dict(task=j, calibration=cal.get('calibration', cp.get('calibration'))))
    r['calibration_evidence'] = evidence
    raw=(redact(json.dumps(r, ensure_ascii=False, indent=2, allow_nan=False))+'\n').encode()
    assert not re.search(rb'/home/user/|/scr/user/|[CE]:[/\\]|ghp_[A-Za-z0-9]{20}|-----BEGIN [A-Z ]*PRIVATE KEY', raw)
    files[target] = raw
    records.append((target, job, runtime['platform'], status))

lit='paper_aacil/results/section178_literature/cic-ids-2018'
freq='paper_aacil/results/section165_confirmation/cse-cic-ids2018'
nsl='paper_aacil/results/section194_nsl'
for s in range(10,15):
    root=B/('runs/section178/cic-ids-2018/icarl-scoped-npy/seed10' if s==10 else f'validation/hpc-20261102T1345/a/icarl/seed{s}')
    add(root, f'{lit}/icarl_seed{s}/result.json', windows if s==10 else linux, 'local-20261029' if s==10 else '493533',
        'scientific checks passed; analysis runtime harmonization pending',
        'Compatibility delivery of the already-published icarl-scoped-npy cell; not a new experiment. NME/softmax-KL adaptation; feature membership does not prove unique originating row IDs.')
for s in range(10,14):
    root=B/(f'runs/section178/cic-ids-2018/'+('cer-bic-common' if s==10 else 'cer-bic-common-scoped')+f'/seed{s}') if s<13 else B/'validation/hpc-20261102T1345/a/bic/seed13'
    add(root, f'{lit}/cer-bic-common_seed{s}/result.json', windows if s<13 else linux,
        'local' if s<13 else '493533',
        'previous local scientific acceptance; runtime harmonization pending' if s<13 else 'raw complete; independent checks retained; final archive acceptance pending',
        'Common Adam 8-epoch/min2000 schedule, NOT native CER schedule. Seed13 imports three completed Windows tasks exactly, trains only task3 on Linux, and evaluates all stages on Linux.' if s==13 else 'Common-budget flow-CNN BiC, not native packet CER reproduction.')
for m in ('freq','wacb'):
    for s in range(10,15):
        add(snapshot/f'cic-ids-2018/{m}/seed{s}', f'{freq}/{m}_seed{s}/result.json', overlay, '493534',
            note='Unregistered extension; raw completion is not independent scientific acceptance.')
for m in ('R3-A','R3-L','ar','replay_bs','freq','wacb','derpp','icarl','ecbrs','bic'):
    for s in range(10,15):
        sub=f'confirmation/nsl-kdd/{m}/seed{s}' if m.startswith('R3-') else f'nsl-kdd/{m}/seed{s}'
        add(snapshot/sub, f'{nsl}/{m}_seed{s}/result.json', overlay, '493534',
            note='NSL-KDD unregistered extension; ecbrs=paper-memory adaptation, bic=common-budget flow CNN; ar=neural AR, not R3-A stateless.')
assert len(records)==69

common='''Acceptance is pending for raw deliveries; completion is not independent scientific acceptance or a passed S1 test. No training, weights, thresholds, confusion matrices, or completed stages were changed for this delivery. Original result hashes and source-relative locations are embedded. Per-stage default and 1/2/5% operating-point matrices and original calibration evidence are preserved. Versions are from the installed pinned environments, not invented per-cell telemetry. Small JSON delivery does not replace protected checkpoints/logs or their later archival checks.

NSL-KDD and Freq/WACB additions are unregistered extensions. Do not pool them into the original registered family or claim overall research completion. RQ1, RQ3, substantive innovation, manuscript QA, and backup reconciliation remain open.
'''
boundaries='''Windows/Linux provenance: CSE iCaRL10 and BiC10-12 were computed locally on Windows. iCaRL11-14 and every B cell were computed on Linux. BiC13 imported the exact completed Windows task0-2 weights/progress, trained task3 on Linux, and evaluated all four stages on Linux. BiC14 is still training and is deliberately absent.

No completed seed was trained twice. Windows diagnostic reevaluation of Linux-trained checkpoints is not a second training result. Linux verification493605 reproduced all CSE iCaRL11-14 and NSL DER++10-14 TEST matrices and holdout counts exactly. Windows reevaluation differed at float32 decision boundaries: NSL DER++11 final1% has one additional Normal-to-Probe error, giving Windows-minus-Linux final BAcc = -100/(9711*5) = -0.002059520131809288 percentage points; the other four DER++ final1% matrices matched. For CSE iCaRL11, three calibrated stage matrices differed and default matrices matched; its exact final1% BAcc delta has not yet been quantified in the retained diagnostic output and must not be guessed. The original Linux matrices, not Windows diagnostic values, are delivered for Linux-trained cells.

S1 must not silently mix runtimes. Where both evaluation versions exist, use Linux evaluation for every seed of that baseline, retaining the Windows diagnostic provenance. A Linux-harmonized evaluation of locally completed seed10 (and BiC10-12) is pending, without retraining those seeds. These compatibility files alone do not assert a runtime-harmonized S1 comparison.
'''
files[f'{lit}/README.md']=('# CSE completed-cell compatibility delivery\n\n'+common+'\n'+boundaries+'\nFive iCaRL cells already have a separate, independently checked archive at icarl-scoped-npy_seed10-14 and icarl-scoped-npy_overview. Those180 published payloads are unchanged. Here the consumer names icarl and cer-bic-common match s178_check.py. BiC is common-budget, not renamed native CER.\n').encode()
files[f'{freq}/README.md']=('# CSE Freq/WACB raw completion\n\n'+common+'\nAll ten cells: Linux job493534, sklearn1.7.2 overlay, torch2.6.0+cu118. CSE per_class.py routes these methods to section165_confirmation/cse-cic-ids2018/<method>_seedN/result.json; this preserves that existing consumer route.\n\n'+boundaries).encode()
files[f'{nsl}/README.md']=('# NSL-KDD raw CPU extension\n\n'+common+'\nAll50 cells: Linux job493534, sklearn1.7.2 overlay, torch2.6.0+cu118. Names R3-A/R3-L, ar (neural), replay_bs, freq, wacb, derpp, icarl, ecbrs (paper-memory), bic (common-budget).\n\n'+boundaries).encode()

def git(repo,*args):
    r=subprocess.run(['git','-C',str(repo),*args],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if r.returncode:
        print(r.stdout.decode('utf-8',errors='replace'),flush=True)
        raise RuntimeError('git '+args[0]+' failed')
    return r.stdout.decode('utf-8').strip()

print(json.dumps(dict(event='RAW_DELIVERY_READY',result_json=len(records),payloads=len(files),accepted_as_new_science=False)),flush=True)
if a.publish:
    repos=[(W/'work/section165_paper_publish','https://github.com/1838904818/Class-IL-in-IDS.git',''),
           (W/'work/github_supervisor_sync_20261010','https://github.com/1838904818/Class-IL.git','main')]
    with (W/'work/section151_local/publication.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            heads=[]
            for repo,url,branch in repos:
                assert git(repo,'remote','get-url','origin')==url
                assert git(repo,'branch','--show-current')==branch
                assert not git(repo,'diff','--name-only') and not git(repo,'diff','--cached','--name-only')
                git(repo,'fetch','origin');git(repo,'merge','--ff-only','origin/main')
                for name,raw in files.items():
                    assert not git(repo,'ls-tree','--name-only','HEAD','--',name),name
                    dest=repo/name
                    assert not dest.exists(),name
                    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
                git(repo,'add','--sparse','--',*files)
                assert sorted(git(repo,'diff','--cached','--name-only').splitlines())==sorted(files)
                git(repo,'diff','--cached','--check')
                message='pushed 69 result.json: CSE icarl 10-14, cer-bic-common 10-13, Freq/WACB 10-14; NSL 10 methods x 5; A 5/6 RUNNING, C PENDING(Resources)'
                git(repo,'commit','-m',message,'-m','Deliver completed raw records at analysis-consumer paths per section340. Acceptance/runtime harmonization pending where stated; no training reruns. B493534 COMPLETED60/60. Preserve default and calibrated matrices, original calibration evidence, runtime provenance, original result SHA256 and Windows boundary limitations. BiC14 unfinished and absent.')
                git(repo,'push','origin','HEAD:main')
                head=git(repo,'rev-parse','HEAD');assert git(repo,'ls-remote','origin','refs/heads/main').split()[0]==head
                for name,raw in files.items():
                    expected=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
                    assert git(repo,'rev-parse',head+':'+name)==expected,name
                heads.append(head)
                print(json.dumps(dict(event='RAW_RESULTS_PUBLISHED',repository=url,head=head,results=69,payloads=len(files))),flush=True)
            for name in files:
                assert git(repos[0][0],'rev-parse',heads[0]+':'+name)==git(repos[1][0],'rev-parse',heads[1]+':'+name)
            print('DUAL_REMOTE_ALL72_BLOBS_VERIFIED',flush=True)
        finally:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
