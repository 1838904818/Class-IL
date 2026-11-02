"""Publish completed read-only robustness results to both research destinations."""
import argparse,gzip,hashlib,json,msvcrt,os,re,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--publish',action='store_true');a=p.parse_args();B=a.base;W=a.workspace
files={p.relative_to(a.archive).as_posix():p.read_bytes() for p in a.archive.rglob('*') if p.is_file()}
assert sum(n.endswith('/same_score_calibration.json') for n in files)==60
for name in ('export-section344-completed.py','publish-section344-completed.py'):files['ncd/hpc/'+name]=(B/'admin'/name).read_bytes()
prefix='paper_aacil/results/section178_literature/same-score-robustness-493707'
for stem in ('section344-real-aggregate-20261102','section344-completed-export-20261102','section344-reference-independent-20261102'):
    for stream in ('stdout','stderr'):
        path=B/'logs'/f'{stem}.{stream}.log';files[prefix+'/logs/'+path.name+'.gz']=gzip.compress(path.read_bytes(),mtime=0)
def scan(name,raw):
    value=gzip.decompress(raw) if name.endswith('.gz') else raw
    assert not re.search(rb'/home/user/[A-Za-z0-9]|/scr/user/[A-Za-z0-9]|(?<![A-Za-z])[CE]:[/\\]|ghp_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----',value),name
for n,v in files.items():scan(n,v)
print(json.dumps(dict(event='SECTION344_PUBLICATION_VALIDATED',cells=60,new_payloads=len(files))),flush=True)
if not a.publish:raise SystemExit(0)
stderr=(B/'logs/section344-completed-publication-20261102.git.stderr.log').open('ab')
def git(repo,*args):
    r=subprocess.run(['git','-c','core.longpaths=true','-C',str(repo),*args],stdout=subprocess.PIPE,stderr=stderr);assert r.returncode==0,args
    return r.stdout.decode().strip()
private=W/'work/section165_paper_publish';public=W/'work/github_supervisor_sync_20261010'
readme='paper_aacil/results/section178_literature/cic-ids-2018/README.md';who='docs/WORK_ORDER_2026-10-09.md'
with (W/'work/section151_local/publication.lock').open('r+b') as lock:
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:
        heads=[];common=None
        for index,(repo,url,branch) in enumerate(((private,'https://github.com/1838904818/Class-IL-in-IDS.git',''),(public,'https://github.com/1838904818/Class-IL.git','main'))):
            assert git(repo,'remote','get-url','origin')==url and git(repo,'branch','--show-current')==branch
            assert not git(repo,'diff','--name-only') and not git(repo,'diff','--cached','--name-only')
            git(repo,'fetch','origin');git(repo,'merge','--ff-only','origin/main')
            if index==0:
                files[readme]=(git(repo,'show','HEAD:'+readme)+'\n\nSection344 completed read-only job493707:60 cells /260 checkpoints, all failures empty; independent list-CM, Student-t CDF, bitmask signflip, Holm and as-published R3-A paired differences verified. New same_score_calibration.json files are alongside unchanged original results. Full three-dataset comparisons,1/2/5% targets, class-wise signed forgetting, measured FPR, costs and sanitized logs: ../same-score-robustness-493707/README.md. Registered CSE original-prior4/4 paired-t/Holm PASS from section347 is reproduced. Same-score1% comparisons also retain4/4 rejection within each dataset, but baseline changes are not uniformly positive and R3-A references remain as published. Exact signflip cannot clear Holm4 with five pairs. No matched-actual-TEST-FPR, mechanism-causal, historical-runtime homogenization or overall research closure claim.\n').encode()
                attrs=git(repo,'show','HEAD:.gitattributes')+'\n\n# Exact completed common-score evidence bytes\n'+'\n'.join('/'+n+' -text' for n in sorted(files))+'\n';files['.gitattributes']=attrs.encode();common=dict(files)
            own=dict(common)
            if index==0:
                own[who]=(git(repo,'show','HEAD:'+who)+'\n\n### Sections347/348 execution result: job493707 COMPLETED,60/60\n\nJob493652 COMPLETED0:0; original A cells6/6 raw complete. BiC14 raw6249ff7 and full five-seed scientific archive2b034a5 were dual-published with RNG restoration false and original-runtime limits. Job493707 COMPLETED0:0 (wall28:06, TotalCPU53:33.365),60 same-score cells/260 checkpoints, no failures; original BiC14 prediction4cp/16CM/24brackets exact. All60 results and comparison statistics delivered with independent CM/statistical checks. C493535 remains PENDING(Resources),0/10, no ETA or idle-node probing. Registered CSE prior-shift4/4 PASS from347 is reproduced; new common-score robustness retains4/4 paired-t/Holm rejection within each dataset. Original primary remains unchanged. R3-A references are as published, not newly runtime-harmonized. RQ1/RQ3/innovation/manuscript source visual QA/backup remain open.\n').encode()
            for n,raw in own.items():
                if n!=who:scan(n,raw)
                dest=repo/n
                if os.name=='nt':dest=Path('\\\\?\\'+str(dest.resolve()))
                if n not in (readme,who,'.gitattributes'):
                    assert not git(repo,'ls-tree','--name-only','HEAD','--',n),n
                    if dest.exists():assert dest.read_bytes()==raw,n
                dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
            names=list(own)
            for offset in range(0,len(names),20):git(repo,'add','--sparse','-f','--',*names[offset:offset+20])
            assert set(git(repo,'diff','--cached','--name-only').splitlines())==set(own)
            git(repo,'-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check')
            git(repo,'commit','-m','493707 COMPLETED60/60 same-score cells;493535 PENDING0/10','-m','Publish260 saved-checkpoint robustness evaluations and original-rule comparisons, independent CM mathematics, paired t/Holm and exact signflip, effective configurations, runtime limits and completed logs. No training or mutation of original results; registered primary unchanged.')
            git(repo,'push','origin','HEAD:main');head=git(repo,'rev-parse','HEAD');assert git(repo,'ls-remote','origin','refs/heads/main').split()[0]==head
            for n,raw in own.items():assert git(repo,'rev-parse',head+':'+n)==hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),n
            heads.append(head);print(json.dumps(dict(event='SECTION344_PUBLISHED',repository=url,head=head,common_payloads=len(common))),flush=True)
        for n in common:assert git(private,'rev-parse',heads[0]+':'+n)==git(public,'rev-parse',heads[1]+':'+n)
        print('SECTION344_ALL_COMMON_RAW_BLOBS_AND_TWO_REMOTE_HEADS_VERIFIED',flush=True)
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
