"""Dual-publish the frozen remaining-method NSL scientific diagnostic archive."""
import argparse,gzip,hashlib,json,msvcrt,os,re,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--c-state',choices=('PENDING','RUNNING','COMPLETED','FAILED'),required=True);p.add_argument('--c-cells',type=int,required=True);a=p.parse_args();B=a.base;W=a.workspace
assert 0<=a.c_cells<=10
prefix='paper_aacil/results/section194_diagnostics/nsl-kdd/score_shift_remaining_eight'
files={prefix+'/'+v.relative_to(a.archive).as_posix():v.read_bytes() for v in a.archive.rglob('*') if v.is_file()};assert len(files)==94
for n in ('verify-nsl-remaining-score-points.py','export-nsl-remaining-completed.py','publish-nsl-remaining-completed.py'):files['ncd/hpc/'+n]=(B/'admin'/n).read_bytes()
for n,v in files.items():
    if n.endswith('.npz'):continue
    text=gzip.decompress(v) if n.endswith('.gz') else v
    assert not re.search(rb'/home/user/[A-Za-z0-9]|/scr/user/[A-Za-z0-9]|(?<![A-Za-z])[CE]:[/\\]|ghp_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY',text),n
err=(B/'logs/nsl-remaining-completed-publication-20261102.git.stderr.log').open('ab')
def git(repo,*args):
    r=subprocess.run(['git','-c','core.longpaths=true','-C',str(repo),*args],stdout=subprocess.PIPE,stderr=err);assert r.returncode==0,args
    return r.stdout.decode().strip()
private=W/'work/section165_paper_publish';public=W/'work/github_supervisor_sync_20261010';wo='docs/WORK_ORDER_2026-10-09.md'
with (W/'work/section151_local/publication.lock').open('r+b') as lock:
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:
        heads=[];common=None
        for index,(repo,url,branch) in enumerate(((private,'https://github.com/1838904818/Class-IL-in-IDS.git',''),(public,'https://github.com/1838904818/Class-IL.git','main'))):
            assert git(repo,'remote','get-url','origin')==url and git(repo,'branch','--show-current')==branch
            assert not git(repo,'diff','--name-only') and not git(repo,'diff','--cached','--name-only')
            git(repo,'fetch','origin');git(repo,'merge','--ff-only','origin/main')
            if index==0:
                files['.gitattributes']=(git(repo,'show','HEAD:.gitattributes')+'\n\n# Exact completed NSL remaining-method diagnostic bytes\n'+'\n'.join('/'+n+' -text' for n in sorted(files))+'\n').encode();common=dict(files)
            own=dict(common)
            if index==0:own[wo]=(git(repo,'show','HEAD:'+wo)+f'\n\n### Section342 remaining NSL diagnostics completed and delivered\n\n493771 COMPLETED0:0,40/40 final cells,160 original default+1/2/5% CM exact, failures empty, stderr0. Wall92s/TotalCPU26.740s/stepMaxRSS1261872KiB. No training or original result/threshold mutation. Remaining eight-method94-artifact archive and actual checker/export/publisher sources delivered; original493670 R3-A/DER++ group is only referenced, never rerun or republished as new work. Independent SciPyKS40/direct960quantiles/plain-list80posthocCM plus numeric JSON/NPZ-byte/holdoutID and ten-method mean/sampleSD checks agree. At identical194/9711 TEST benign false positives, neural ar mean BAcc51.2739199% exceeds R3-A47.4398322%; R3-L47.4534697% is numerically close. No R3-A matched-FPR all-method clean win. At5% R3-A actual4.6298% differs from others4.9943%, so not strictly matched. All posthoc points use TEST benign labels and replace no registered primary/S1. C493535 {a.c_state},{a.c_cells}/10; no idle-node probing or migration. WACB CSE scoreCDF/KS, manuscript RNG footnote/current-source sync/visual QA, RQ1/RQ3, substantive innovation and backup remain pending.\n').encode()
            for n,v in own.items():
                if n not in (wo,'.gitattributes'):assert not git(repo,'ls-tree','--name-only','HEAD','--',n),n
                dest=repo/n
                if os.name=='nt':dest=Path('\\\\?\\'+str(dest.resolve()))
                if n not in (wo,'.gitattributes') and dest.exists():assert dest.read_bytes()==v,('Partial differs',n)
                dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(v)
            names=list(own)
            for offset in range(0,len(names),20):git(repo,'add','--sparse','-f','--',*names[offset:offset+20])
            assert set(git(repo,'diff','--cached','--name-only').splitlines())==set(own)
            git(repo,'-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check')
            git(repo,'commit','-m',f'493771 COMPLETED40/40 NSL diagnostic cells;493535 {a.c_state}{a.c_cells}/10','-m','Archive remaining eight methods with exact original CM reconciliation, independent score/posthoc checks and ten-method descriptive comparison. Neural ar exceeds R3-A at identical2% TEST FPR;5% actual rates differ. Preserve TEST-informed scope and unchanged registered primary.')
            git(repo,'push','origin','HEAD:main');head=git(repo,'rev-parse','HEAD');assert git(repo,'ls-remote','origin','refs/heads/main').split()[0]==head
            for n,v in own.items():assert git(repo,'rev-parse',head+':'+n)==hashlib.sha1(b'blob '+str(len(v)).encode()+b'\0'+v).hexdigest(),n
            heads.append(head);print(json.dumps(dict(repository=url,head=head,common_payloads=len(common))),flush=True)
        for n in common:assert git(private,'rev-parse',heads[0]+':'+n)==git(public,'rev-parse',heads[1]+':'+n)
        print('NSL_REMAINING_ALL98_COMMON_BLOBS_TWO_REMOTE_HEADS_AND_CROSSREPO_VERIFIED',flush=True)
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
