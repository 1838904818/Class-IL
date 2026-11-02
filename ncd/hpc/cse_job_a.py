"""CSE-only bounded six-cell queue; completed BiC11/12 are verification-only."""
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    code=Path(__file__).resolve().parent
    output=Path(os.environ['NCD_OUTPUT_ROOT'])
    cache=Path(os.environ['NCD_CACHE_ROOT'])
    data=Path(os.environ['NCD_DATA_ROOT'])
    for name in ('NCD_FROZEN_ROOT','NCD_DATA_ROOT','NCD_CACHE_ROOT'):
        assert Path(os.environ[name]).is_dir(),name
    if not a.smoke:
        assert os.environ.get('SLURM_JOB_PARTITION') == 'cpu-epyc-genoa'
        assert int(os.environ['SLURM_CPUS_PER_TASK']) >= 12
        for seed in (11,12):
            root=cache/f'accepted/bic-seed{seed}'
            result=json.loads((root/'result.json').read_text())
            summary=json.loads((root/'summary.json').read_text())
            assert result['completed'] and summary['completed'] and summary['scientific_result']
            assert result['identity']['seed']==seed and len(result['checkpoints'])==4
            assert 'CELL_VALIDATED' in (root/'monitor.stdout.log').read_text()
            assert 'data_source_row_roles_memory_and_recomputed_predictions' in (root/'validation.stdout.log').read_text()
            print(f'PROTECTED_COMPLETE_VERIFIED BiC seed{seed}; no training',flush=True)
        assert (cache/'imports/bic-seed13/identity.json').is_file()
    cells=[('bic',10),('icarl',10)] if a.smoke else [('bic',13),('bic',14)]+[('icarl',s) for s in range(11,15)]
    output.mkdir(parents=True,exist_ok=True)
    logs=output/'logs'; logs.mkdir(exist_ok=True)
    active=[]; stopping=False; failures=[]
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    start=time.monotonic()
    try:
        while cells or active:
            while cells and len(active)<(1 if a.smoke else 6) and not stopping:
                method,seed=cells.pop(0)
                target=output/method/(f'smoke-seed{seed}' if a.smoke else f'seed{seed}')
                summary=target/'summary.json'
                if summary.exists():
                    value=json.loads(summary.read_text()); assert value['completed']
                    print(f'COMPLETE_PRESERVED {method} seed{seed}',flush=True); continue
                attempt=0
                while (logs/f'{method}-{seed}-attempt{attempt}.stdout.log').exists(): attempt+=1
                prefix=logs/f'{method}-{seed}-attempt{attempt}'
                command=[sys.executable,'-I','-B','-u',str(code/f'cse_{method}_portable.py'),
                    '--dataset','CSE-CIC-IDS2018','--seed',str(seed),
                    '--data-root',str(data),'--output-dir',str(target),'--resume']
                if method=='bic': command+=['--log-prefix',str(prefix)]
                else: command+=['--method','icarl','--threads','2']
                if a.smoke: command+=['--smoke']
                out=open(str(prefix)+'.stdout.log','xb'); err=open(str(prefix)+'.stderr.log','xb')
                child=subprocess.Popen(command,stdout=out,stderr=err)
                active.append((child,out,err,method,seed,summary))
                print(json.dumps(dict(event='CELL_STARTED',method=method,seed=seed,pid=child.pid,command=command)),flush=True)
            for item in active[:]:
                child,out,err,method,seed,summary=item
                rc=child.poll()
                if rc is None: continue
                out.close();err.close();active.remove(item)
                print(json.dumps(dict(event='CELL_EXIT',method=method,seed=seed,exit_code=rc)),flush=True)
                if rc != 0:
                    failures.append((method,seed,rc))
                    print(json.dumps(dict(event='CELL_FAILED',method=method,seed=seed,exit_code=rc,traceback_log=err.name)),flush=True)
                    continue
                value=json.loads(summary.read_text());assert value['completed']
                if not a.smoke:
                    protected=Path(os.environ['NCD_PROTECTED_ROOT'])/method/f'seed{seed}'
                    for path in target.rglob('*.json'):
                        dest=protected/path.relative_to(target)
                        dest.parent.mkdir(parents=True,exist_ok=True)
                        shutil.copyfile(path,dest)
                print(f'CELL_COMPLETE {method} seed{seed}',flush=True)
            if stopping:
                raise RuntimeError('Allocation stopping; preserve task/epoch checkpoints, no automatic resubmission')
            time.sleep(.25 if a.smoke else 5)
        if failures:raise RuntimeError(f'Cells failed after all other work was preserved: {failures}')
        print('SMOKE_COMPLETE' if a.smoke else 'JOB_A_ALL_COMPLETE',flush=True)
    finally:
        for child,out,err,*_ in active:
            if child.poll() is None: child.terminate()
        for child,out,err,*_ in active:
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            out.close();err.close()
        print(f'QUEUE_ELAPSED_SECONDS={time.monotonic()-start:.3f}',flush=True)


if __name__=='__main__':main()
