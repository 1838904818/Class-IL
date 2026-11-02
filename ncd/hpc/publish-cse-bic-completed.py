"""Dual-publish one frozen five-seed BiC archive with independent blob proofs."""
import argparse,gzip,hashlib,json,msvcrt,os,re,shutil,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--publish',action='store_true');p.add_argument('--evaluation-state',choices=('RUNNING','COMPLETED','FAILED'),required=True);p.add_argument('--c-state',choices=('PENDING','RUNNING','COMPLETED','FAILED'),required=True);p.add_argument('--c-cells',type=int,default=0);a=p.parse_args();B=a.base;W=a.workspace
files={};prefix='paper_aacil/results/section178_literature/cic-ids-2018'
for path in a.archive.rglob('*'):
    if path.is_file():
        rel=path.relative_to(a.archive).as_posix();target=rel.replace('overview/','cer-bic-common-completed_overview/',1) if rel.startswith('overview/') else rel
        files[prefix+'/'+target]=path.read_bytes()
assert len(files)==186
for name in ('export-cse-bic-completed.py','verify-cse-bic-completed-export.py','publish-cse-bic-completed.py'):files['ncd/hpc/'+name]=(B/'admin'/name).read_bytes()
for name,raw in files.items():
    value=gzip.decompress(raw) if name.endswith('.gz') else raw
    assert not re.search(rb'/home/user/[A-Za-z0-9]|/scr/user/[A-Za-z0-9]|(?<![A-Za-z])[CE]:[/\\]|ghp_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----',value),name
print(json.dumps(dict(event='CSE_BIC_PUBLICATION_VALIDATED',artifacts=186,sources=3,scientific_group='same five cells, not five new experiments')),flush=True)
if not a.publish:raise SystemExit(0)
stderr=(B/'logs/cse-bic-completed-publication-20261102.git.stderr.log').open('ab')
def git(repo,*args):
    r=subprocess.run(['git','-c','core.longpaths=true','-C',str(repo),*args],stdout=subprocess.PIPE,stderr=stderr);assert r.returncode==0,args
    return r.stdout.decode().strip()
private=W/'work/section165_paper_publish';public=W/'work/github_supervisor_sync_20261010';readme=prefix+'/README.md'
with (W/'work/section151_local/publication.lock').open('r+b') as lock:
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:
        heads=[]
        for index,(repo,url,branch) in enumerate(((private,'https://github.com/1838904818/Class-IL-in-IDS.git',''),(public,'https://github.com/1838904818/Class-IL.git','main'))):
            assert git(repo,'remote','get-url','origin')==url and git(repo,'branch','--show-current')==branch
            assert not git(repo,'diff','--name-only') and not git(repo,'diff','--cached','--name-only')
            git(repo,'fetch','origin');git(repo,'merge','--ff-only','origin/main')
            if index==0:
                text=git(repo,'show','HEAD:'+readme)+'\n\nCompleted original-runtime BiC scientific archive: canonical cer-bic-common-completed_seed10 through seed14 and cer-bic-common-completed_overview now contain20 checkpoint receipts,60 calibrations, all four views, class-wise signed forgetting, five-seed mean/sample SD, unique retained TRAIN row IDs, costs, relevant failures and validation/source evidence. The earlier cer-bic-common_seedN raw files are consumer aliases of these SAME five cells, not extra experiments. Original Windows10–12 and Linux13–14 evaluation identities remain explicit; this is not original-prior runtime harmonization or overall S1/research closure.\n\nFive-seed default BAcc58.7230% (sampleSD13.1818pp) and FPR58.3855% must be read together. At the1% holdout target, BAcc34.5813% (sampleSD8.0443pp), measured TEST FPR0.9948% and attack recall23.9263% show weak low-FPR recognition, not broad superiority. Seed14 retains rng_restored=false and split-attempt cost accounting. Uniform same-score493707 evaluation is separate.\n'
                files[readme]=text.encode()
            for name,raw in files.items():
                dest=repo/name
                if os.name=='nt':dest=Path('\\\\?\\'+str(dest.resolve()))
                if name!=readme:
                    assert not git(repo,'ls-tree','--name-only','HEAD','--',name),name
                    if dest.exists():
                        assert dest.read_bytes()==raw,('Partial publication bytes changed',name)
                        continue
                dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
            names=list(files)
            for offset in range(0,len(names),20):git(repo,'add','--sparse','-f','--',*names[offset:offset+20])
            assert set(git(repo,'diff','--cached','--name-only').splitlines())==set(files)
            git(repo,'diff','--cached','--check')
            git(repo,'commit','-m',f'Archive five CSE BiC seeds:493652 COMPLETED A6/6,B493534 COMPLETED60/60,C493535 {a.c_state}{a.c_cells}/10','-m',f'Original20 checkpoint/60 calibration evidence, independent CM mathematics, per-class forgetting and costs; preserve Windows/Linux scope, seed11/13 imports, seed14 epoch6 RNG limitation and interrupted-attempt accounting. Same-score493707 {a.evaluation_state} is separate; no overall S1, RQ1/RQ3 or innovation closure claimed.')
            git(repo,'push','origin','HEAD:main');head=git(repo,'rev-parse','HEAD');assert git(repo,'ls-remote','origin','refs/heads/main').split()[0]==head
            for name,raw in files.items():assert git(repo,'rev-parse',head+':'+name)==hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),name
            heads.append(head);print(json.dumps(dict(event='CSE_BIC_GROUP_PUBLISHED',repository=url,head=head,payloads=len(files))),flush=True)
        for name in files:assert git(private,'rev-parse',heads[0]+':'+name)==git(public,'rev-parse',heads[1]+':'+name)
        print('CSE_BIC_ALL190_PAYLOADS_TWO_REMOTE_HEADS_AND_CROSSREPO_VERIFIED',flush=True)
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
