"""One local B2 lane, publishing each completed cell before advancing."""
from __future__ import annotations
import argparse
import json
import gzip
import os
from pathlib import Path
import shutil
import re
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--method',choices=('ar','er'),required=True)
    p.add_argument('--after-pid',type=int,required=True)
    p.add_argument('--after-cell',type=Path,required=True)
    a=p.parse_args()
    base=a.workspace.resolve(); work=base/'work'
    source=work/'github_private_current'; public=work/'github_supervisor_sync_20261010'
    private=work/'section151_publish'; outputs=work/'section151_local'
    import psutil
    import msvcrt

    def git(repo,*args):
        return subprocess.check_output(['git','-C',str(repo),*args],text=True,stderr=subprocess.STDOUT).strip()

    def publish(cell):
        result=json.loads((cell/'result.json').read_text())
        key=a.method+f"_s{result['scenario']}_seed{result['seed']}"
        relative='paper_aacil/results/section151_b2/'+key
        with (outputs/'publication.lock').open('a+b') as lock:
            if lock.tell()==0:
                lock.write(b'0');lock.flush()
            while True:
                try:
                    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);break
                except OSError:
                    time.sleep(2)
            try:
                if not (private/relative).exists():
                    subprocess.run([sys.executable,'-I','-B',str(source/'ncd/export_section151.py'),
                        '--cell',str(cell),'--destination',str(private/'paper_aacil/results/section151_b2'),
                        '--redact-root',str(base)],check=True)
                if not (public/relative).exists():
                    shutil.copytree(private/relative,public/relative)
                for f in (private/relative).iterdir():
                    text=(gzip.decompress(f.read_bytes()).decode('utf-8') if f.suffix=='.gz' else f.read_text())
                    if re.search(r'[A-Za-z]:[/\\]Users[/\\]|github_pat_|ghp_[A-Za-z0-9]+|-----BEGIN .*PRIVATE KEY|sk-[A-Za-z0-9]{20}',text):
                        raise RuntimeError('Sensitive pattern in publication artifact')
                for repo,remote in ((private,'https://github.com/1838904818/Class-IL-in-IDS.git'),
                                    (public,'https://github.com/1838904818/Class-IL.git')):
                    if git(repo,'remote','get-url','origin')!=remote:
                        raise RuntimeError('Publication remote identity changed')
                    git(repo,'add','--',relative)
                    staged=git(repo,'diff','--cached','--name-only').splitlines()
                    if any(not name.startswith(relative+'/') for name in staged):
                        raise RuntimeError('Unrelated staged changes; preserve them')
                    if staged:
                        git(repo,'diff','--cached','--check')
                        git(repo,'commit','-m',f'Report B2 {a.method.upper()} scenario {result["scenario"]} seed {result["seed"]}')
                    git(repo,'fetch','origin')
                    git(repo,'rebase','origin/main')
                    git(repo,'push','origin','HEAD:main')
                    head=git(repo,'rev-parse','HEAD')
                    remotehead=git(repo,'ls-remote','origin','refs/heads/main').split()[0]
                    if head!=remotehead:
                        raise RuntimeError('Remote HEAD verification failed')
                for f in ('result.json','stdout.log.gz','stderr.log.gz'):
                    if git(private,'rev-parse',f'HEAD:{relative}/{f}')!=git(public,'rev-parse',f'HEAD:{relative}/{f}'):
                        raise RuntimeError('Dual publication blob mismatch')
                print('PUBLISHED '+key,flush=True)
            finally:
                lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)

    if psutil.pid_exists(a.after_pid):
        proc=psutil.Process(a.after_pid)
        if not any('run_section151_'+a.method+'.py' in x for x in proc.cmdline()):
            raise RuntimeError('Initial PID is not the expected B2 lane')
        proc.wait()
    initial=a.after_cell.resolve()
    if not (initial/'result.json').exists():
        raise RuntimeError('Initial cell failed; inspect its retained logs before retry')
    publish(initial)
    for seed in (42,1,2,3,4):
        for scenario in (1,2):
            relative=f'paper_aacil/results/section151_b2/{a.method}_s{scenario}_seed{seed}/result.json'
            if (private/relative).exists() and (public/relative).exists():
                continue
            cell=outputs/f'{a.method}_scenario{scenario}_seed{seed}_queue'
            if cell.exists():
                raise RuntimeError('Existing attempt must be diagnosed, not overwritten: '+str(cell))
            cmd=[sys.executable,'-I','-B','-u',str(source/f'ncd/run_section151_{a.method}.py'),
                 '--upstream',str(work/'section81_replayids_upstream'),
                 '--data',str(work/'replayids_ofra_seed42_build/dataset'),
                 '--output',str(cell),'--scenario',str(scenario),'--seed',str(seed)]
            if a.method=='er':
                cmd+=['--dependencies',str(work/'section81_dependencies_v1'),
                      '--runtime',str(work/'section151_runtime')]
            print('START '+cell.name,flush=True)
            with Path(str(cell)+'.stdout.log').open('wb') as out, Path(str(cell)+'.stderr.log').open('wb') as err:
                subprocess.run(cmd,cwd=base,stdout=out,stderr=err,check=True)
            publish(cell)
    print('LANE_COMPLETE '+a.method+'; paired-oracle reconciliation and Malaya remain',flush=True)


if __name__=='__main__':
    main()
