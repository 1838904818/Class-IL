"""Publish exact-byte WACB diagnostic parts and independently validated science."""
import argparse,gzip,hashlib,json,msvcrt,os,re,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True);a=p.parse_args();B=a.base;W=a.workspace
archive=B/'admin/cse-wacb-score-publication-v1'
prefix='paper_aacil/results/section194_diagnostics/cic-ids-2018/wacb_score_distributions_493836'
files={prefix+'/'+v.relative_to(archive).as_posix():v.read_bytes() for v in archive.rglob('*') if v.is_file()};assert len(files)==37
for n in ('verify-cse-wacb-score-distributions.py','restore-cse-wacb-score-arrays.py','export-cse-wacb-score-distributions.py','publish-cse-wacb-score-distributions.py'):files['ncd/hpc/'+n]=(B/'admin'/n).read_bytes()
for n,v in files.items():
    assert len(v)<=45*1024**2,n
    if '.npz' in n:continue
    data=gzip.decompress(v) if n.endswith('.gz') else v
    assert not re.search(rb'/home/user/[A-Za-z0-9]|/scr/user/[A-Za-z0-9]|(?<![A-Za-z])[CE]:[/\\]|ghp_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY',data),n
err=(B/'logs/cse-wacb-score-publication-20261102.git.stderr.log').open('ab')
def git(repo,*args):
    r=subprocess.run(['git','-c','core.longpaths=true','-C',str(repo),*args],stdout=subprocess.PIPE,stderr=err);assert r.returncode==0,args
    return r.stdout.decode().strip()
private=W/'work/section165_paper_publish';public=W/'work/github_supervisor_sync_20261010';wo='docs/WORK_ORDER_2026-10-09.md'
readme='paper_aacil/results/section178_literature/cic-ids-2018/README.md'
note='''

### WACB final score distributions completed (493836)

Five final saved models,20 original TEST CM and15 holdout FP counts independently agree exactly. Raw logits/scores/rowIDs/labels, source bindings, CDF/KS/quantiles/per-class recall and Normal absorption are delivered under ../../section194_diagnostics/cic-ids-2018/wacb_score_distributions_493836. Benign KS0.00678361–0.01040315 and TEST FPR approximately1% coexist with BAcc21.40–72.65%; this is descriptive evidence, not a proven feature/training cause or bisection failure. Diagnostic batch4096 differs from original config2048; CM/FP exactness does not establish original logits bitexactness. No training or old threshold/result mutation.

Original large NPZ bytes are delivered as45MiB parts with full/part SHA and actual restore code; shared row-label NPZ is unchanged. All six NPZ were reassembled and every array loaded. Job493836 COMPLETED0:0,87s wall/87.914s CPU/6823588KiB stepRSS; expected import-time stderr is nonempty. Earlier owner-cancelled493796 and startup-only493824 costs remain separately visible, not erased or counted as model results. Old slowdown was not reproduced; cause unproved. Manuscript RNG footnote/current-source visual QA, RQ1/RQ3, innovation and backup gates remain open.
'''
with (W/'work/section151_local/publication.lock').open('r+b') as lock:
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:
        heads=[];common=None
        for index,(repo,url,branch) in enumerate(((private,'https://github.com/1838904818/Class-IL-in-IDS.git',''),(public,'https://github.com/1838904818/Class-IL.git','main'))):
            assert git(repo,'remote','get-url','origin')==url and git(repo,'branch','--show-current')==branch
            assert not git(repo,'diff','--name-only') and not git(repo,'diff','--cached','--name-only')
            git(repo,'fetch','origin');git(repo,'merge','--ff-only','origin/main')
            if index==0:
                files[readme]=(git(repo,'show','HEAD:'+readme)+note).encode()
                files['.gitattributes']=(git(repo,'show','HEAD:.gitattributes')+'\n\n# Exact WACB score diagnostic bytes and raw-array parts\n'+'\n'.join('/'+n+' -text' for n in sorted(files))+'\n').encode();common=dict(files)
                assert len(common)==43
            own=dict(common)
            if index==0:own[wo]=(git(repo,'show','HEAD:'+wo)+note+'\n493836 completed5/5 and independently verified; scientific archive delivered with45MiB exact-byte parts. Only493535 remains PENDING(Priority),0/10 at publication; no ETA, migration or idle probing. No completed old scientific group was rerun or republished.\n').encode()
            for n,v in own.items():
                if n not in (wo,readme,'.gitattributes'):assert not git(repo,'ls-tree','--name-only','HEAD','--',n),n
                dest=repo/n
                if os.name=='nt':dest=Path('\\\\?\\'+str(dest.resolve()))
                if n not in (wo,readme,'.gitattributes') and dest.exists():assert dest.read_bytes()==v,('Partial differs',n)
                dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(v)
            names=list(own)
            for offset in range(0,len(names),15):git(repo,'add','--sparse','-f','--',*names[offset:offset+15])
            assert set(git(repo,'diff','--cached','--name-only').splitlines())==set(own)
            git(repo,'-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check')
            git(repo,'commit','-m','493836 COMPLETED5/5 WACB score diagnostics;493535 PENDING0/10','-m','Deliver independently checked score distributions with exact-byte45MiB array parts and restore tool. Preserve batch/runtime limits, failed-attempt costs and descriptive rather than causal conclusions.')
            print(json.dumps(dict(event='COMMITTED_PUSH_START',repository=url)),flush=True)
            git(repo,'push','origin','HEAD:main');head=git(repo,'rev-parse','HEAD');assert git(repo,'ls-remote','origin','refs/heads/main').split()[0]==head
            for n,v in own.items():assert git(repo,'rev-parse',head+':'+n)==hashlib.sha1(b'blob '+str(len(v)).encode()+b'\0'+v).hexdigest(),n
            heads.append(head);print(json.dumps(dict(repository=url,head=head,common_payloads=len(common))),flush=True)
        for n in common:assert git(private,'rev-parse',heads[0]+':'+n)==git(public,'rev-parse',heads[1]+':'+n)
        print('WACB_ALL43_COMMON_RAW_BLOBS_TWO_REMOTE_HEADS_CROSSREPO_VERIFIED',flush=True)
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
