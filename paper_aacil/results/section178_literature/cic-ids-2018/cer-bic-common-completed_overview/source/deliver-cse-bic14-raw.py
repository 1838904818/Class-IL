"""Deliver one completed BiC14 raw result; never retrain or replace old cells.

The source snapshot is read-only. Independent scientific acceptance is separate.
"""
import argparse,copy,hashlib,json,msvcrt,re,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--directory',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--publish',action='store_true')
p.add_argument('--a-state',choices=('RUNNING','COMPLETED'));p.add_argument('--c-state',choices=('PENDING','RUNNING','COMPLETED','FAILED'));p.add_argument('--c-cells',type=int,default=0);a=p.parse_args()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(raw):return hashlib.sha256(raw).hexdigest()
source=a.directory/'result.json';r=read(source);summary=read(a.directory/'summary.json');i=r['identity']
assert r['completed'] and summary['completed'] and i['seed']==14 and not i['smoke']
assert i['dataset']=='CSE-CIC-IDS2018' and i['method']=='cer-bic-flow-cnn-common'
assert len(r['checkpoints'])==len(r['default_stages'])==len(r['calibrated_stages'])==4
assert i['portable_entry_sha256']=='15ce00b645525a9eaed22ff1045ac95cf99ab967216e1ace508b8da848f0ef3c'
for task,(cp,d,c) in enumerate(zip(r['checkpoints'],r['default_stages'],r['calibrated_stages'])):
    assert cp['task_index']==task
    assert cp['seen_classes']==[v for group in i['tasks'][:task+1] for v in group]
    reference=d['metrics']['confusion_matrix'];supports=list(map(sum,reference))
    assert sum(supports)>0
    for alpha in ('0.01','0.02','0.05'):
        cm=c['operating_points'][alpha]['confusion_matrix']
        assert list(map(sum,cm))==supports and all(isinstance(v,int) and v>=0 for row in cm for v in row)
        point=c['calibration']['operating_points'][alpha]
        assert point['calibration_false_positives']<=point['maximum_false_positives']
r=copy.deepcopy(r)
r['original_checkpoint_identity']=copy.deepcopy(i)
r['original_checkpoint_identity_sha256']=sha(json.dumps(i,sort_keys=True,separators=(',',':')).encode())
r['original_checkpoint_identity_sha256_scope']='Canonical original unsanitized identity; personal paths in this public copy are redacted.'
r['identity']['resume']=dict(from_epoch=6,rng_restored=False,original_job=493533,resumed_job=493652)
r['class_order']=i['class_order'];r['tasks']=i['tasks']
r['runtime']=dict(python='3.11.9',torch='2.6.0+cu118',numpy='2.2.6',platform='Linux',evidence='Pinned installed environment inspection, not per-cell automatic telemetry')
r['delivery']=dict(status='raw complete; independent scientific acceptance and full archive pending',original_job=493533,completion_job=493652,
    source_result_sha256=sha(source.read_bytes()),source_result='BiC14 completed snapshot/result.json',original_scientific_numbers_preserved=True,
    checkpoint_identity_preserved_separately=True,no_training_or_threshold_change=True,
    cost_scope='Recorded training includes preserved epoch1-6 plus completed continuation. New-attempt elapsed is not whole-cell training; original allocation wall includes concurrent cells. Interrupted partial epoch cost and both attempt streams remain separate.',
    limitations='Common-budget flow CNN, not native CER reproduction. RNG state was not restored. Raw delivery does not close scientific acceptance, S1, RQ1/RQ3, innovation, manuscript QA or backup reconciliation.')
text=json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)
# Redact all personal absolute paths without touching scientific numeric fields.
text=re.sub(r'@@HPC_HOME@@/"\\\s]+','@@HPC_HOME@@',text)
text=re.sub(r'@@HPC_SCRATCH@@/"\\\s]+','@@HPC_SCRATCH@@',text)
text=re.sub(r'(?i)[CE]:[/\\]+Users[/\\]+[^/"\s]+','@@LOCAL_HOME@@',text)
text=re.sub(r'(?i)(?<![A-Za-z])[CE]:[/\\]+','@@LOCAL_DRIVE@@/',text)
raw=(text+'\n').encode()
assert not re.search(rb'@@HPC_HOME@@@@HPC_SCRATCH@@/\\]|ghp_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----',raw)
target='paper_aacil/results/section178_literature/cic-ids-2018/cer-bic-common_seed14/result.json'
source_target='ncd/hpc/deliver-cse-bic14-raw.py'
readme='paper_aacil/results/section178_literature/cic-ids-2018/README.md'
print(json.dumps(dict(event='BIC14_RAW_READY',completed=True,source_result_sha256=sha(source.read_bytes()),accepted=False)),flush=True)
if not a.publish:raise SystemExit(0)
assert a.a_state and a.c_state and 0<=a.c_cells<=10,'Use freshly verified states/counts when publishing'
W=a.workspace;heads=[]
with (W/'work/section151_local/publication.lock').open('r+b') as lock:
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:
        def git(repo,*args):
            z=subprocess.run(['git','-C',str(repo),*args],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            if z.stderr:print(z.stderr.decode(errors='replace'),end='',flush=True)
            assert z.returncode==0,args
            return z.stdout.decode().strip()
        repos=[(W/'work/section165_paper_publish','https://github.com/1838904818/Class-IL-in-IDS.git',''),(W/'work/github_supervisor_sync_20261010','https://github.com/1838904818/Class-IL.git','main')]
        for repo,url,branch in repos:
            assert git(repo,'remote','get-url','origin')==url and git(repo,'branch','--show-current')==branch
            assert not git(repo,'diff','--name-only') and not git(repo,'diff','--cached','--name-only')
            git(repo,'fetch','origin');git(repo,'merge','--ff-only','origin/main')
            assert not git(repo,'ls-tree','--name-only','HEAD','--',target)
            assert not git(repo,'ls-tree','--name-only','HEAD','--',source_target)
            old_readme=git(repo,'show','HEAD:'+readme)
            updated_readme=old_readme.replace('BiC14 is still training and is deliberately absent.','BiC14 raw completion is now delivered separately; full scientific acceptance remains pending.')
            updated_readme+='\n\nBiC14 raw completion delivery: cer-bic-common_seed14/result.json now preserves all original four-stage default and1/2/5% matrices. Original493533 and continuation493652 provenance, rng_restored=false and the original checkpoint identity are retained separately. This raw delivery is not independent scientific acceptance, completed five-seed archival acceptance, or S1 closure.\n'
            payloads={target:raw,source_target:Path(__file__).read_bytes(),readme:updated_readme.encode()}
            for name,payload in payloads.items():
                assert not re.search(rb'@@HPC_HOME@@@@HPC_SCRATCH@@/\\]|ghp_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----',payload),name
                dest=repo/name
                if name!=readme:assert not dest.exists()
                dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
            git(repo,'add','--sparse','--',*payloads);assert set(git(repo,'diff','--cached','--name-only').splitlines())==set(payloads)
            git(repo,'diff','--cached','--check')
            git(repo,'commit','-m',f'Publish BiC14 raw:493652 {a.a_state} A6/6,B493534 COMPLETED60/60,C493535 {a.c_state}{a.c_cells}/10','-m','Deliver the missing common-budget cell without changing original scientific numbers. Preserve RNG-restoration limitation and original checkpoint identity separately. Full independent acceptance, scientific archive and same-score493707 status require their own evidence; no S1 closure claimed.')
            git(repo,'push','origin','HEAD:main');head=git(repo,'rev-parse','HEAD')
            assert git(repo,'ls-remote','origin','refs/heads/main').split()[0]==head
            for name,payload in payloads.items():assert git(repo,'rev-parse',head+':'+name)==hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest(),name
            heads.append(head);print(json.dumps(dict(event='BIC14_RAW_PUBLISHED',repository=url,head=head)),flush=True)
        for name in payloads:assert git(repos[0][0],'rev-parse',heads[0]+':'+name)==git(repos[1][0],'rev-parse',heads[1]+':'+name)
        print('BIC14_RAW_DUAL_BLOB_AND_REMOTE_HEADS_VERIFIED',flush=True)
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
