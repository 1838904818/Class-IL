"""Two allocated V100S GPUs, three bounded ER cells per GPU, no submissions."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    root=Path(os.environ['NCD_OUTPUT_ROOT']);code=Path(__file__).resolve().parent
    logs=root/'logs';logs.mkdir(parents=True,exist_ok=True)
    gpus=os.environ.get('CUDA_VISIBLE_DEVICES','0,1').split(',')
    if not a.smoke:
        assert os.environ.get('SLURM_JOB_PARTITION') in ('gpu-v100s','gpu-titan')
        assert len(gpus)==2 and int(os.environ['SLURM_CPUS_PER_TASK'])>=12
    cells=[(d,s) for d in ('CSE-CIC-IDS2018','NSL-KDD') for s in ((10,) if a.smoke else range(10,15))]
    active=[];stopping=False;sampled=0.;failures=[]
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        while cells or active:
            while cells and len(active)<(1 if a.smoke else 6) and not stopping:
                dataset,seed=cells.pop(0)
                lane=min(range(len(gpus)),key=lambda k:sum(x[3]==k for x in active))
                target=root/dataset.lower()/f'seed{seed}'
                summary=target/'summary.json'
                if summary.exists():
                    value=json.loads(summary.read_text());assert value['completed']
                    print(f'COMPLETE_PRESERVED {dataset} {seed}',flush=True);continue
                attempt=0;prefix=logs/f'{dataset}-seed{seed}'
                while Path(str(prefix)+f'.attempt{attempt}.stdout.log').exists():attempt+=1
                prefix=Path(str(prefix)+f'.attempt{attempt}')
                out=open(str(prefix)+'.stdout.log','xb');err=open(str(prefix)+'.stderr.log','xb')
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpus[lane]
                env['NCD_SECTION81_PARENT_PID']=str(os.getpid())
                command=[sys.executable,'-I','-B','-u',str(code/'tabt_cpu_gpu_portable.py'),
                    '--dataset',dataset,'--seed',str(seed),'--confirmation','--memory-setting','fixed200',
                    '--dependency-root',os.environ['NCD_TABT_DEPENDENCY_ROOT'],
                    '--data-root',os.environ['NCD_DATA_ROOT'],'--output-dir',str(target),'--resume']
                if a.smoke:command+=['--smoke']
                child=subprocess.Popen(command,stdout=out,stderr=err,env=env)
                active.append((child,out,err,lane,dataset,seed,summary,target))
                print(json.dumps(dict(event='CELL_STARTED',dataset=dataset,seed=seed,gpu=gpus[lane],pid=child.pid,command=command)),flush=True)
            for item in active[:]:
                child,out,err,lane,dataset,seed,summary,target=item;rc=child.poll()
                if rc is None:continue
                active.remove(item);out.close();err.close()
                print(json.dumps(dict(event='CELL_EXIT',dataset=dataset,seed=seed,exit_code=rc)),flush=True)
                if rc != 0:
                    failures.append((dataset,seed,rc))
                    print(json.dumps(dict(event='CELL_FAILED',dataset=dataset,seed=seed,exit_code=rc,traceback_log=err.name)),flush=True)
                    continue
                value=json.loads(summary.read_text());assert value['completed'] and value['scientific_result']==(not a.smoke)
                if not a.smoke:
                    protected=Path(os.environ['NCD_PROTECTED_ROOT'])/target.relative_to(root)
                    for path in target.rglob('*.json'):
                        dest=protected/path.relative_to(target);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
                print(f'CELL_COMPLETE {dataset} {seed}',flush=True)
            if not a.smoke and time.monotonic()-sampled>60:
                subprocess.run(['nvidia-smi','--id='+','.join(gpus),
                    '--query-gpu=uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],check=True)
                sampled=time.monotonic()
            if stopping:raise RuntimeError('Allocation stopping: preserve task checkpoints; no automatic submission')
            time.sleep(.25 if a.smoke else 5)
        if failures:raise RuntimeError(f'Cells failed after all other work was preserved: {failures}')
        print('SMOKE_COMPLETE' if a.smoke else 'JOB_C_ALL_COMPLETE',flush=True)
    finally:
        for child,*_ in active:
            if child.poll() is None:child.terminate()
        for child,out,err,*_ in active:
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            out.close();err.close()


if __name__=='__main__':main()
