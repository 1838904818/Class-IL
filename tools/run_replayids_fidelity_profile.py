"""One-allocation fidelity/profile coordinator; never submits or trains.

Only the bound verifier may be launched. Online tracking accepts numeric aggregate
fields only; protected output copying also preserves scientific failure reports.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import time

FILES = ('GPU_FIDELITY.json', 'OPERATION_STATUS.json', 'RESOURCE_SAMPLES.json',
         'WANDB_RUN.json', 'verifier.stdout.log', 'verifier.stderr.log')
EXIT_TRACKING = 82
WANDB_VERSION='0.23.0'
HISTORY_FIELDS={'Progress/completed_checkpoints','Progress/required_checkpoints',
    'Fidelity/passing_checkpoints','Fidelity/prediction_mismatches',
    'Fidelity/score_cells_outside_tolerance','Fidelity/max_absolute_score_error',
    'Time/verifier_seconds','Resource/gpu_percent','Resource/cpu_allocation_percent',
    'Resource/process_tree_rss_mib','Time/operation_seconds'}

def history(values):
    if not set(values)<=HISTORY_FIELDS:raise ValueError('Unapproved tracking field')
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values.values()):
        raise ValueError('Tracking permits nonnegative finite numeric values only')
    return dict(values)

def tracking_identity(operation_sha):
    if not re.fullmatch('[0-9a-f]{64}',operation_sha):raise ValueError('Invalid operation hash')
    return ('replayids-fidelity-'+operation_sha[:12],{
        'purpose':'frozen-probe fidelity and first resource profile','required_checkpoints':20,
        'encoder_batch_size':512,'atol':1e-4,'rtol':1e-5,'operation_sha256':operation_sha})

def validate_netrc(metadata,uid):
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode)!=0o600 or metadata.st_uid!=uid:
        raise ValueError('Standard authentication file must be a user-owned regular mode-0600 file')

def prepare_tracking_environment(env,home,uid,expected_version):
    """Stat authentication metadata only; never read or log credential contents."""
    if any(k in env for k in ('WANDB_API_KEY','WANDB_API_KEY_FILE','WANDB_ACCESS_TOKEN')):
        raise ValueError('Environment-provided tracking credentials are prohibited')
    validate_netrc((home/'.netrc').lstat(),uid)
    if expected_version!=WANDB_VERSION or importlib.metadata.version('wandb')!=expected_version:
        raise ValueError('Tracking SDK version differs from the reviewed version')
    allowed={'PATH','HOME','LANG','LC_ALL','LC_CTYPE','TZ','TMPDIR','TMP','TEMP',
        'SSL_CERT_FILE','SSL_CERT_DIR','REQUESTS_CA_BUNDLE','CURL_CA_BUNDLE'}
    clean={k:v for k,v in env.items() if k in allowed};clean['HOME']=str(home)
    clean.update(WANDB_MODE='online',WANDB_CONSOLE='off',WANDB_DISABLE_CODE='true',
        WANDB_DISABLE_GIT='true',WANDB_ERROR_REPORTING='false',WANDB_SENTRY_DSN='',
        WANDB_SILENT='true',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    return clean

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    os.replace(tmp,path)

def inside(root,path):
    base=Path(root).resolve();p=Path(path).resolve()
    if p==base or not p.is_relative_to(base):raise ValueError('Output outside declared own root')
    return p

def aggregate(report):
    rows=report.get('checkpoints',[])
    if len(rows)>20:raise ValueError('Unexpected checkpoint coverage')
    fields={'Progress/completed_checkpoints':len(rows),'Progress/required_checkpoints':20,
        'Fidelity/passing_checkpoints':sum(bool(r.get('passed')) for r in rows),
        'Fidelity/prediction_mismatches':sum(r['prediction_mismatch_count'] for r in rows),
        'Fidelity/score_cells_outside_tolerance':sum(v['outside_tolerance_cells'] for r in rows for v in r['scores'].values()),
        'Fidelity/max_absolute_score_error':max((v['max_abs_error'] for r in rows for v in r['scores'].values()),default=0.),
        'Time/verifier_seconds':report.get('elapsed_seconds',0.)}
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in fields.values()):
        raise ValueError('Invalid aggregate metric')
    return fields

def gpu_query(device):
    if not re.fullmatch(r'(?:[0-9]+|GPU-[A-Za-z0-9-]+)',device):
        raise ValueError('One explicit allocated GPU identifier required; MIG telemetry unsupported')
    return ['nvidia-smi','--id='+device,'--query-gpu=utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits']

def allocated_gpu(env):
    """Use a UUID or Slurm's global step ID, never a remapped CUDA ordinal."""
    visible=env.get('CUDA_VISIBLE_DEVICES','')
    gpu_query(visible)
    if visible.startswith('GPU-'):return visible
    step=env.get('SLURM_STEP_GPUS','');job=env.get('SLURM_JOB_GPUS','')
    gpu_query(step)
    if job!=step:raise ValueError('One matching global job/step GPU ID required')
    return step

def scientific_pass(report,code,binding):
    rows=report.get('checkpoints',[])
    identities=[(r.get('seed'),r.get('checkpoint')) for r in rows]
    expected={(s,c) for s in (1,2,3,4,42) for c in range(4)}
    score_names={'head_scores','router_z_scores','joint_scores'}
    aggregate(report)
    return bool(code==0 and report.get('binding_sha256')==binding and
        report.get('schema')=='replayids-gpu-fidelity-v1' and report.get('status')=='PASS' and
        report.get('coverage_complete') is True and report.get('all_passed') is True and
        len(rows)==20 and set(identities)==expected and all(
            r.get('passed') is True and r['prediction_mismatch_count']==0 and
            set(r['scores'])==score_names and all(v['outside_tolerance_cells']==0 for v in r['scores'].values())
            for r in rows))

def parse_gpu(text):
    lines=text.strip().splitlines()
    if len(lines)!=1:raise ValueError('Telemetry must describe exactly one allocated GPU')
    vals=[float(v.strip()) for v in lines[0].split(',')]
    if len(vals)!=3 or not all(math.isfinite(v) for v in vals):raise ValueError('Invalid GPU telemetry')
    util,used,total=vals
    if not 0<=util<=100 or not 0<=used<=total or total<=0:raise ValueError('GPU telemetry range')
    return {'gpu_percent':util,'gpu_memory_used_mib':used,'gpu_memory_total_mib':total}

class UtilizationGuard:
    """Conservative continuous-low-resource window, not a DICC scheduler oracle."""
    def __init__(self,seconds=200):self.seconds=seconds;self.low_since={}
    def observe(self,t,sample):
        for k in ('gpu_percent','cpu_allocation_percent','host_memory_percent'):
            value=sample[k]
            if not math.isfinite(value) or not 0<=value:raise ValueError('Invalid utilization')
            if value<10:self.low_since.setdefault(k,t)
            else:self.low_since.pop(k,None)
        return sorted(k for k,start in self.low_since.items() if t-start>=self.seconds)

def protect(source,target):
    """Copy only allowlisted small products into a new directory; no overwrite."""
    source=Path(source).resolve();target=Path(target)
    target.mkdir(parents=False,exist_ok=False,mode=0o700)
    hashes={}
    for name in FILES:
        src=source/name
        if not src.exists():continue
        if src.is_symlink() or not src.is_file() or src.stat().st_size>16*1024*1024:
            raise ValueError('Unexpected protected artifact')
        dst=target/name;shutil.copyfile(src,dst);dst.chmod(0o600)
        if digest(src)!=digest(dst):raise ValueError('Protected copy hash mismatch')
        hashes[name]=digest(dst)
    for required in ('OPERATION_STATUS.json','RESOURCE_SAMPLES.json','WANDB_RUN.json'):
        if required not in hashes:raise ValueError('Missing operation evidence')
    (target/'PROTECTED_RESULTS_SHA256SUMS.txt').write_text(
        ''.join(f'{h}  {name}\n' for name,h in sorted(hashes.items())),encoding='utf-8',newline='\n')
    write(target/'PROTECTED_COMPLETE.json',{'files':hashes,'manifest_sha256':digest(target/'PROTECTED_RESULTS_SHA256SUMS.txt'),
        'scope':'copy completeness, not a scientific pass','verifier_report_present':'GPU_FIDELITY.json' in hashes})
    return hashes

def process_sample(root_pid,proc=Path('/proc'),ticks=None,page_bytes=None):
    """Read only this process tree, never enumerate unrelated /proc processes."""
    ticks=ticks or os.sysconf('SC_CLK_TCK');page_bytes=page_bytes or os.sysconf('SC_PAGE_SIZE')
    pending=[(root_pid,None)];seen=set();cpu=rss=0
    while pending:
        pid,parent=pending.pop()
        if pid in seen:continue
        seen.add(pid);base=proc/str(pid)
        try:
            fields=(base/'stat').read_text().rsplit(')',1)[1].split()
            if parent is not None and int(fields[1])!=parent:continue
            cpu+=(int(fields[11])+int(fields[12]))/ticks;rss+=int(fields[21])*page_bytes
            for thread in (base/'task').iterdir():
                pending.extend((int(c),pid) for c in (thread/'children').read_text().split())
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return cpu,rss

def verify_operation(plan,folder):
    if plan.get('schema')!='replayids-fidelity-profile-v2' or plan.get('walltime_seconds')!=600:
        raise ValueError('Unexpected operation contract')
    if plan.get('verifier_deadline_seconds')!=480 or plan.get('allocated_memory_mib')!=4096:
        raise ValueError('Resource/deadline contract changed')
    if plan.get('wandb_version')!=WANDB_VERSION:raise ValueError('Tracking SDK version not bound')
    if plan.get('wandb')!={'entity':'csnet','project':'ofra-etg-leon-hpc','group':'replayids-score-fidelity-profile-v2'}:
        raise ValueError('Unexpected tracking destination')
    required={'run_replayids_fidelity_profile.py','check_replayids_gpu_fidelity.py',
              'check_replayids_canonical_fidelity.py','GPU_FIDELITY_INPUTS.json'}
    if set(plan.get('executables',{}))!=required:raise ValueError('Executable closure incomplete')
    for name,h in plan['executables'].items():
        if digest(folder/name)!=h:raise ValueError('Operation executable hash mismatch')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operation',type=Path,required=True);p.add_argument('--operation-sha256',required=True)
    a=p.parse_args()
    if platform.system()!='Linux':raise ValueError('An allocated Linux step is required')
    import pwd,socket
    from check_replayids_gpu_fidelity import require_allocation
    user=pwd.getpwuid(os.getuid()).pw_name
    require_allocation(os.environ,socket.gethostname(),user)
    compute_env=dict(os.environ)
    if digest(a.operation)!=a.operation_sha256:raise ValueError('Operation binding mismatch')
    folder=a.operation.resolve().parent;plan=load(a.operation);verify_operation(plan,folder)
    # No environment installation, credential reading, or scientific imports here.
    gpu_cmd=gpu_query(allocated_gpu(os.environ))
    scratch=inside(Path('/scr/user')/user,plan['scratch_parent'])
    protected=inside(Path.home(),plan['protected_parent'])
    if not scratch.is_dir() or not protected.is_dir():raise ValueError('Reviewed output parents must exist')
    suffix='job-'+os.environ['SLURM_JOB_ID']
    run_dir=scratch/suffix;target=protected/suffix
    if target.exists():raise ValueError('Protected job target already exists')
    run_dir.mkdir(mode=0o700,exist_ok=False)
    status={'schema':'fidelity-profile-operation-v2','job_id':os.environ['SLURM_JOB_ID'],
        'operation_sha256':a.operation_sha256,'status':'PREPARING','verifier_exit_code':None,
        'scientific_pass':False,'new_model_training':False,'new_test_accuracy':False}
    tracking={'status':'NOT_STARTED','url':None};samples=[];run=None;child=None;code=1
    stop={'signal':None};signal.signal(signal.SIGTERM,lambda s,f:stop.update(signal=s))
    signal.signal(signal.SIGINT,lambda s,f:stop.update(signal=s))
    start=time.monotonic();guard=UtilizationGuard(600/3)
    write(run_dir/'OPERATION_STATUS.json',status)
    try:
        clean=prepare_tracking_environment(compute_env,Path.home(),os.getuid(),plan['wandb_version'])
        # The coordinator/SDK sees only a minimal environment. The bound verifier
        # receives the original allocation environment explicitly, never via SDK.
        os.environ.clear();os.environ.update(clean);sys.argv=['fidelity-profile']
        tracking_name,tracking_config=tracking_identity(a.operation_sha256)
        import wandb
        run=wandb.init(**plan['wandb'],name=tracking_name,mode='online',dir=str(run_dir),
            config=tracking_config,
            settings=wandb.Settings(init_timeout=20,x_disable_meta=True,x_disable_stats=True,
                x_disable_machine_info=True,x_disable_viewer=True,x_save_requirements=False,disable_code=True,
                disable_git=True,disable_job_creation=True,save_code=False,console='off'))
        tracking={'status':'ONLINE','url':run.url};write(run_dir/'WANDB_RUN.json',tracking)
        command=[sys.executable,'-B',str(folder/'check_replayids_gpu_fidelity.py'),
            '--binding',str(folder/'GPU_FIDELITY_INPUTS.json'),'--binding-sha256',plan['executables']['GPU_FIDELITY_INPUTS.json'],
            '--runtime',plan['runtime_root'],'--protected',plan['protected_input_root'],'--data',plan['data_root'],
            '--output',str(run_dir/'verification'),'--canonical-helper',str(folder/'check_replayids_canonical_fidelity.py')]
        root=os.getpid();previous_cpu,_=process_sample(root);previous_t=time.monotonic();failures=0
        with (run_dir/'verifier.stdout.log').open('wb') as out,(run_dir/'verifier.stderr.log').open('wb') as err:
            child=subprocess.Popen(command,stdout=out,stderr=err,env=compute_env)
            while child.poll() is None:
                now=time.monotonic();elapsed=now-start
                if stop['signal'] or elapsed>=480:
                    status['status']='SIGNALLED' if stop['signal'] else 'DEADLINE';code=124;break
                try:
                    raw=subprocess.run(gpu_cmd,capture_output=True,text=True,check=True,timeout=5)
                    sample=parse_gpu(raw.stdout);cpu,rss=process_sample(root)
                    sample.update(elapsed_seconds=elapsed,process_tree_rss_bytes=rss,
                        cpu_allocation_percent=max(0.,(cpu-previous_cpu)/(now-previous_t)/2*100),
                        host_memory_percent=rss/(4096*1024**2)*100)
                    previous_cpu=cpu;previous_t=now;samples.append(sample)
                    low=guard.observe(elapsed,sample)
                    if low or sample['host_memory_percent']>=90:
                        status['status']='LOW_UTILIZATION' if low else 'HOST_MEMORY_GUARD';status['resource_flags']=low;code=125;break
                    rpt=run_dir/'verification/GPU_FIDELITY.json'
                    metrics=aggregate(load(rpt)) if rpt.exists() else {'Progress/completed_checkpoints':0}
                    metrics.update({'Resource/gpu_percent':sample['gpu_percent'],'Resource/cpu_allocation_percent':sample['cpu_allocation_percent'],
                        'Resource/process_tree_rss_mib':rss/1024**2,'Time/operation_seconds':elapsed})
                    run.log(history(metrics));failures=0
                except Exception as exc:
                    failures+=1;status['telemetry_error_type']=type(exc).__name__
                    if failures>=3:status['status']='TELEMETRY_FAILURE';code=126;break
                time.sleep(5)
            else:code=child.returncode;status['status']='VERIFIER_FINISHED'
    except Exception as exc:
        status['error_type']=type(exc).__name__;status['status']='OPERATION_ERROR'
        if run is None:tracking={'status':'INIT_FAILED','url':None};code=EXIT_TRACKING
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
        if child is not None:status['verifier_exit_code']=child.returncode
        rpt=run_dir/'verification/GPU_FIDELITY.json'
        if rpt.exists():
            shutil.copyfile(rpt,run_dir/'GPU_FIDELITY.json')
            try:
                report=load(rpt)
                status['scientific_pass']=scientific_pass(report,code,plan['executables']['GPU_FIDELITY_INPUTS.json'])
                if code==0 and not status['scientific_pass']:code=2
            except Exception as exc:
                status['final_report_error_type']=type(exc).__name__;status['scientific_pass']=False;code=1
            else:
                if run is not None:
                    try:run.log(history(aggregate(report)))
                    except Exception as exc:
                        status['final_tracking_error_type']=type(exc).__name__;code=EXIT_TRACKING
        elif code==0:
            status['status']='MISSING_VERIFIER_REPORT';code=2
        status.update(pre_tracking_finish_exit_code=code,elapsed_seconds=time.monotonic()-start,
            tracking_finish_confirmed=False,tracking_receipt='TRACKING_FINALIZATION.json, if subsequently created')
        write(run_dir/'WANDB_RUN.json',tracking);write(run_dir/'RESOURCE_SAMPLES.json',{
            'samples':samples,'sample_seconds':5,'host_memory_scope':'summed RSS of this process and its descendants; shared pages may be double-counted',
            'gpu_scope':'only the allocated UUID or Slurm global step GPU ID; never a remapped CUDA ordinal',
            'cpu_scope':'own process-tree tick deltas; exited children can cause conservative undercounting',
            'measurement_scope':'first workload profile; not a prior measured resource justification'})
        write(run_dir/'OPERATION_STATUS.json',status)
        # Seal core evidence BEFORE potentially blocking network finalization.
        # A missing separate receipt means cloud finalization is unconfirmed.
        hashes=protect(run_dir,target)
        receipt={'status':'NOT_STARTED','exit_code':code,'core_manifest_sha256':digest(target/'PROTECTED_RESULTS_SHA256SUMS.txt')}
        if run is not None:
            try:
                run.finish(exit_code=code);receipt['status']='CLIENT_FINISH_RETURNED'
            except Exception as exc:
                receipt.update(status='CLIENT_FINISH_FAILED',error_type=type(exc).__name__);code=EXIT_TRACKING
        receipt['exit_code']=code
        write(target/'TRACKING_FINALIZATION.json',receipt)
        (target/'TRACKING_FINALIZATION_SHA256SUMS.txt').write_text(
            digest(target/'TRACKING_FINALIZATION.json')+'  TRACKING_FINALIZATION.json\n',encoding='utf-8',newline='\n')
        print(json.dumps({'status':status['status'],'scientific_pass':status['scientific_pass'],
            'protected_files':len(hashes),'wandb_url':tracking.get('url')}),flush=True)
    return code

if __name__=='__main__':raise SystemExit(main())
