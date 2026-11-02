"""Copy only saved section178 result/checkpoint inputs for read-only evaluation."""
import argparse,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);p.add_argument('--seed',type=int);a=p.parse_args()
assert not a.destination.exists(),'Input snapshot is immutable; use a new destination'
B=a.base;W=a.workspace;count=0;missing=[]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for z in iter(lambda:f.read(8*1024**2),b''):h.update(z)
    return h.hexdigest()
for ds in ('unsw-nb15','cic-ids-2017','cic-ids-2018'):
    for method in ('derpp','icarl','ecbrs-paper','cer-bic-common'):
        for seed in ([a.seed] if a.seed is not None else range(10,15)):
            root=B/f'runs/section178/{ds}/{method}/seed{seed}'
            if ds=='unsw-nb15' and method=='derpp' and seed==10:root=W/'work/section178_local/confirmation/unsw-nb15/derpp/seed10'
            if ds=='cic-ids-2018' and method=='icarl':root=B/('runs/section178/cic-ids-2018/icarl-scoped-npy/seed10' if seed==10 else f'validation/hpc-20261102T1345/a/icarl/seed{seed}')
            if ds=='cic-ids-2018' and method=='ecbrs-paper' and seed==13:root=B/'runs/section178/cic-ids-2018/ecbrs-paper/seed13-attempt1-after-reboot'
            if ds=='cic-ids-2018' and method=='cer-bic-common':root=B/('runs/section178/cic-ids-2018/cer-bic-common/seed10' if seed==10 else f'runs/section178/cic-ids-2018/cer-bic-common-scoped/seed{seed}' if seed<13 else f'validation/hpc-20261102T1345/a/bic/seed{seed}')
            rp=root/('cell/result.json' if method in ('derpp','icarl') else 'result.json')
            if not rp.exists():missing.append([ds,method,seed]);continue
            r=json.loads(rp.read_text(encoding='utf-8'));assert r['completed'] and r['identity']['seed']==seed
            dest=a.destination/ds/method/f'seed{seed}';dest.mkdir(parents=True)
            shutil.copyfile(rp,dest/'result.json');assert sha(rp)==sha(dest/'result.json')
            if 'calibrated_stages' not in r:
                src=root/'calibrated_stages.json';shutil.copyfile(src,dest/'calibrated_stages.json');assert sha(src)==sha(dest/'calibrated_stages.json')
            for cp in r['checkpoints']:
                src=rp.parent/cp['checkpoint_file'];target=dest/cp['checkpoint_file'];assert sha(src)==cp['checkpoint_sha256']
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target);assert sha(target)==cp['checkpoint_sha256']
            count+=1
            print(json.dumps(dict(event='SAVED_INPUT_STAGED',dataset=ds,method=method,seed=seed,checkpoints=len(r['checkpoints']))),flush=True)
print(json.dumps(dict(event='SAVED_INPUT_STAGING_COMPLETE',cells=count,missing=missing,training=False)),flush=True)
