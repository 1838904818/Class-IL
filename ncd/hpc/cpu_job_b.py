"""Bounded Job B: CSE Freq/WACB and all ten NSL CPU methods, seeds10-14."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

METHODS=('R3-A','R3-L','ar','replay_bs','freq','wacb','derpp','icarl','ecbrs','bic')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    root=Path(os.environ['NCD_OUTPUT_ROOT']);code=Path(__file__).resolve().parent
    data=Path(os.environ['NCD_DATA_ROOT']); logs=root/'logs';logs.mkdir(parents=True,exist_ok=True)
    cells=[('NSL-KDD',m,s) for m in METHODS for s in ((10,) if a.smoke else range(10,15))]
    cells += [('CSE-CIC-IDS2018',m,s) for m in ('freq','wacb') for s in ((10,) if a.smoke else range(10,15))]
    active=[];stopping=False;failures=[]
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if not a.smoke:assert int(os.environ['SLURM_CPUS_PER_TASK'])>=64
    try:
        while cells or active:
            cse_active=sum(x[3]=='CSE-CIC-IDS2018' for x in active)
            while cells and len(active)<(1 if a.smoke else 16) and not stopping:
                eligible=next((k for k,c in enumerate(cells) if c[0]!='CSE-CIC-IDS2018' or cse_active<2),None)
                if eligible is None:break
                dataset,method,seed=cells.pop(eligible)
                slug='nsl-kdd' if dataset=='NSL-KDD' else 'cic-ids-2018'
                target=root/slug/method/f'seed{seed}'
                command=[sys.executable,'-I','-B','-u']
                env=os.environ.copy()
                threads=4 if method not in ('ecbrs','bic','R3-A','R3-L') else 2
                for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]=str(threads)
                prefix=logs/f'{slug}-{method}-{seed}'
                attempt=0
                while Path(str(prefix)+f'.attempt{attempt}.stdout.log').exists():attempt+=1
                prefix=Path(str(prefix)+f'.attempt{attempt}')
                if method in ('R3-A','R3-L'):
                    seed=1 if a.smoke else seed
                    command += [str(code/'stateless_cpu_portable.py'),'--method',method,'--seed',str(seed),
                                '--output-root',str(root),'--threads','2']
                    target=root/('smoke' if a.smoke else 'confirmation')/dataset.lower()/method/f'seed{seed}'
                elif method=='bic':
                    env['NCD_OUTPUT_ROOT']=str(root/slug)
                    target=root/slug/'bic'/(f'smoke-seed{seed}' if a.smoke else f'seed{seed}')
                    command += [str(code/'nsl_bic_portable.py'),'--seed',str(seed),
                                '--output-dir',str(target),'--log-prefix',str(root/slug/'logs'/prefix.name)]
                elif method=='ecbrs':
                    command += [str(code/'ecbrs_cpu_portable.py'),'--seed',str(seed),
                                '--output-dir',str(target),'--log-prefix',str(prefix)]
                else:
                    command += [str(code/'fixed_cpu_portable.py'),'--method',method,'--seed',str(seed),
                                '--output-dir',str(target),'--threads',str(threads)]
                command += ['--dataset',dataset,'--data-root',str(data),'--resume']
                if a.smoke:command += ['--smoke']
                summary=target/'summary.json'
                if summary.exists():
                    value=json.loads(summary.read_text());assert value['completed']
                    print(f'COMPLETE_PRESERVED {dataset} {method} {seed}',flush=True);continue
                out=open(str(prefix)+'.stdout.log','xb');err=open(str(prefix)+'.stderr.log','xb')
                child=subprocess.Popen(command,stdout=out,stderr=err,env=env)
                active.append((child,out,err,dataset,method,seed,summary,target))
                if dataset=='CSE-CIC-IDS2018':cse_active+=1
                print(json.dumps(dict(event='CELL_STARTED',dataset=dataset,method=method,seed=seed,pid=child.pid,command=command)),flush=True)
            for item in active[:]:
                child,out,err,dataset,method,seed,summary,target=item;rc=child.poll()
                if rc is None:continue
                out.close();err.close();active.remove(item)
                print(json.dumps(dict(event='CELL_EXIT',dataset=dataset,method=method,seed=seed,exit_code=rc)),flush=True)
                if rc != 0:
                    failures.append((dataset,method,seed,rc))
                    print(json.dumps(dict(event='CELL_FAILED',dataset=dataset,method=method,seed=seed,exit_code=rc,traceback_log=err.name)),flush=True)
                    continue
                value=json.loads(summary.read_text());assert value['completed'] and value['scientific_result']==(not a.smoke)
                if not a.smoke:
                    protected=Path(os.environ['NCD_PROTECTED_ROOT'])/target.relative_to(root)
                    for path in target.rglob('*.json'):
                        dest=protected/path.relative_to(target);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
                print(f'CELL_COMPLETE {dataset} {method} {seed}',flush=True)
            if stopping:raise RuntimeError('Allocation stopping: retain checkpoints, no automatic submission')
            time.sleep(.25 if a.smoke else 5)
        if failures:raise RuntimeError(f'Cells failed after all other work was preserved: {failures}')
        print('SMOKE_COMPLETE' if a.smoke else 'JOB_B_ALL_COMPLETE',flush=True)
    finally:
        for child,*_ in active:
            if child.poll() is None:child.terminate()
        for child,out,err,*_ in active:
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            out.close();err.close()


if __name__=='__main__':main()
