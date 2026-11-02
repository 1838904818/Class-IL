"""Sequential read-only forward verification in the original Linux runtime.

No model fitting, optimizer, result replacement, installation, or submissions.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import numpy as np
import torch

assert os.environ.get('SLURM_JOB_PARTITION') == 'cpu-epyc-genoa'
assert int(os.environ['SLURM_CPUS_PER_TASK']) == 4
code=Path(os.environ['NCD_VALIDATION_CODE']).resolve(strict=True)
out=Path(os.environ['NCD_OUTPUT_ROOT'])
protected=Path(os.environ['NCD_PROTECTED_ROOT'])
out.mkdir(parents=True,exist_ok=False)
protected.mkdir(parents=True,exist_ok=False)
print(json.dumps(dict(event='VERIFICATION_RUNTIME',job=os.environ['SLURM_JOB_ID'],python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),training=False)),flush=True)
print(torch.__config__.show(),flush=True)
for filename in ('check-hpc-cse-nme-predictions.py','check-hpc-derpp-predictions.py',Path(__file__).name):
    print(json.dumps(dict(event='VERIFIER_SOURCE',file=filename,sha256=hashlib.sha256((code/filename).read_bytes()).hexdigest())),flush=True)
scratch=Path('/scr/user')/os.environ['USER']
commands=[]
for seed in range(11,15):
    commands.append((f'cse-icarl-{seed}',[sys.executable,'-I','-B','-u',str(code/'check-hpc-cse-nme-predictions.py'),'--root',str(scratch/'classil-a-v2-20261102/cells/icarl'),'--data',str(scratch/'classil-a-v2-20261102/data/cic-ids-2018'),'--seed',str(seed)]))
commands.append(('nsl-derpp',[sys.executable,'-I','-B','-u',str(code/'check-hpc-derpp-predictions.py'),'--root',str(scratch/'classil-b-v2-20261102/cells/nsl-kdd/derpp'),'--data',str(scratch/'classil-b-v2-20261102/data/nsl-kdd')]))
failures=[]
for name,command in commands:
    print(json.dumps(dict(event='VERIFY_START',name=name,command=command)),flush=True)
    stdout=out/f'{name}.stdout.log'
    stderr=out/f'{name}.stderr.log'
    with stdout.open('xb') as o,stderr.open('xb') as e:
        rc=subprocess.run(command,stdout=o,stderr=e,check=False).returncode
    for path in (stdout,stderr):
        shutil.copyfile(path,protected/path.name)
    print(json.dumps(dict(event='VERIFY_EXIT',name=name,exit_code=rc,accepted=False)),flush=True)
    if rc or stderr.stat().st_size:
        failures.append(name)
print(json.dumps(dict(event='VERIFY_COMPLETE',checks=len(commands),failures=failures,accepted=False,scope='Independent checkpoint predictions only; full scientific acceptance remains separate')),flush=True)
raise SystemExit(1 if failures else 0)
