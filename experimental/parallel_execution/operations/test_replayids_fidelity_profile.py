"""Synthetic coordinator tests: no CUDA execution, credentials or remote writes."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile
import types
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).resolve().parent/'run_replayids_fidelity_profile.py'
if not SOURCE.exists():SOURCE=Path(__file__).resolve().parents[1]/'tools/run_replayids_fidelity_profile.py'
spec=importlib.util.spec_from_file_location('fidelity_profile',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def valid_report():
    rows=[]
    for seed in (1,2,3,4,42):
        for cp in range(4):
            rows.append({'seed':seed,'checkpoint':cp,'passed':True,'prediction_mismatch_count':0,
                'scores':{k:{'outside_tolerance_cells':0,'max_abs_error':0.} for k in
                    ('head_scores','router_z_scores','joint_scores')},'sample_id_sha256':'PRIVATE_TEST_ID'})
    return {'schema':'replayids-gpu-fidelity-v1','binding_sha256':'a'*64,'status':'PASS',
        'coverage_complete':True,'all_passed':True,'checkpoints':rows,'elapsed_seconds':5.}

class Metrics(unittest.TestCase):
    def test_allowlist(self):
        fields=m.aggregate(valid_report())
        self.assertEqual(fields['Progress/completed_checkpoints'],20)
        self.assertNotIn('PRIVATE_TEST_ID',json.dumps(fields))
        self.assertTrue(all(type(v) in (int,float) for v in fields.values()))
    def test_nonfinite_fails(self):
        r=valid_report();r['elapsed_seconds']=float('nan')
        with self.assertRaises(ValueError):m.aggregate(r)
    def test_complete_pass(self):self.assertTrue(m.scientific_pass(valid_report(),0,'a'*64))
    def test_duplicate_or_missing_fails(self):
        r=valid_report();r['checkpoints'][-1]=copy.deepcopy(r['checkpoints'][0])
        self.assertFalse(m.scientific_pass(r,0,'a'*64))
        r['checkpoints'].pop();self.assertFalse(m.scientific_pass(r,0,'a'*64))
    def test_false_pass_flags_and_binding(self):
        self.assertFalse(m.scientific_pass(valid_report(),1,'a'*64))
        self.assertFalse(m.scientific_pass(valid_report(),0,'b'*64))
        r=valid_report();r['checkpoints'][0]['scores']['joint_scores']['outside_tolerance_cells']=1
        self.assertFalse(m.scientific_pass(r,0,'a'*64))

class Telemetry(unittest.TestCase):
    def test_gpu_mapping(self):
        env={'CUDA_VISIBLE_DEVICES':'0','SLURM_STEP_GPUS':'5','SLURM_JOB_GPUS':'5'}
        self.assertEqual(m.allocated_gpu(env),'5')
        self.assertIn('--id=5',m.gpu_query(m.allocated_gpu(env)))
        self.assertEqual(m.allocated_gpu({'CUDA_VISIBLE_DEVICES':'GPU-a-b'}),'GPU-a-b')
    def test_invalid_gpu_mapping(self):
        for v in ('','0,1','MIG-abc','0; whoami','-1'):
            with self.subTest(v=v),self.assertRaises(ValueError):m.gpu_query(v)
        for env in ({'CUDA_VISIBLE_DEVICES':'0'},
                    {'CUDA_VISIBLE_DEVICES':'0','SLURM_STEP_GPUS':'5','SLURM_JOB_GPUS':'4'}):
            with self.assertRaises(ValueError):m.allocated_gpu(env)
    def test_gpu_parse(self):
        self.assertEqual(m.parse_gpu(' 25, 800, 40000\n')['gpu_percent'],25)
        for v in ('N/A, 1, 40','1,2,40\n1,2,40','nan,2,40','101,2,40','1,50,40'):
            with self.subTest(v=v),self.assertRaises(ValueError):m.parse_gpu(v)
    def test_low_usage_and_recovery(self):
        g=m.UtilizationGuard();s={'gpu_percent':9.,'cpu_allocation_percent':11.,'host_memory_percent':15.}
        self.assertEqual(g.observe(0,s),[]);self.assertEqual(g.observe(199,s),[])
        self.assertEqual(g.observe(200,s),['gpu_percent'])
        self.assertEqual(g.observe(201,{**s,'gpu_percent':10}),[])
        self.assertEqual(g.observe(202,s),[]);self.assertEqual(g.observe(401,s),[])
        self.assertEqual(g.observe(402,s),['gpu_percent'])
    def test_own_process_tree_only(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def proc(pid,parent,utime,stime,rss,children):
                base=root/str(pid);(base/'task'/str(pid)).mkdir(parents=True)
                fields=['0']*22;fields[0]='S';fields[1]=str(parent)
                fields[11]=str(utime);fields[12]=str(stime);fields[21]=str(rss)
                (base/'stat').write_text(str(pid)+' (own process) '+' '.join(fields))
                (base/'task'/str(pid)/'children').write_text(children)
            proc(100,1,100,50,10,'101 102 999')
            proc(101,100,20,30,5,'')
            proc(102,999,9000,9000,9000,'') # reparented/reused PID is excluded
            (root/'888').mkdir();(root/'888/stat').write_text('unrelated unreadable-format process')
            self.assertEqual(m.process_sample(100,root,100,4096),(2.,15*4096))

class Protected(unittest.TestCase):
    def setup_source(self,root):
        src=root/'scratch';src.mkdir()
        for n in ('OPERATION_STATUS.json','RESOURCE_SAMPLES.json','WANDB_RUN.json'):m.write(src/n,{'status':'FAILED'})
        return src
    def test_failure_artifacts_are_protected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=self.setup_source(root);target=root/'protected'
            hashes=m.protect(src,target);self.assertEqual(len(hashes),3)
            seal=m.load(target/'PROTECTED_COMPLETE.json')
            self.assertFalse(seal['verifier_report_present'])
            self.assertEqual(seal['manifest_sha256'],m.digest(target/'PROTECTED_RESULTS_SHA256SUMS.txt'))
            for n,h in hashes.items():self.assertEqual(m.digest(target/n),h)
            with self.assertRaises(FileExistsError):m.protect(src,target)
    def test_missing_does_not_seal(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'scratch';src.mkdir();target=root/'protected'
            with self.assertRaises(ValueError):m.protect(src,target)
            self.assertFalse((target/'PROTECTED_COMPLETE.json').exists())
    def test_oversize_does_not_seal(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=self.setup_source(root);target=root/'protected'
            with (src/'verifier.stdout.log').open('wb') as f:f.truncate(16*1024*1024+1)
            with self.assertRaises(ValueError):m.protect(src,target)
            self.assertFalse((target/'PROTECTED_COMPLETE.json').exists())
    def test_symlink_does_not_seal(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=self.setup_source(root);target=root/'protected'
            try:(src/'GPU_FIDELITY.json').symlink_to(src/'OPERATION_STATUS.json')
            except OSError:self.skipTest('Host does not permit creating test symlinks')
            with self.assertRaises(ValueError):m.protect(src,target)
    def test_path_confinement(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self.assertEqual(m.inside(root,root/'child'),root/'child')
            for target in (root,root.parent/'outside'):
                with self.assertRaises(ValueError):m.inside(root,target)

class Binding(unittest.TestCase):
    def test_exact_executable_closure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);names=('run_replayids_fidelity_profile.py','check_replayids_gpu_fidelity.py',
                'check_replayids_canonical_fidelity.py','GPU_FIDELITY_INPUTS.json')
            for n in names:(root/n).write_text('synthetic fixture\n')
            plan={'schema':'replayids-fidelity-profile-v3','walltime_seconds':600,'verifier_deadline_seconds':480,
                'allocated_memory_mib':8192,'wandb_version':'0.23.0','wandb':{'entity':'csnet','project':'ofra-etg-leon-hpc',
                    'group':'replayids-score-fidelity-profile-v3'},'executables':{n:m.digest(root/n) for n in names}}
            m.verify_operation(plan,root)
            bad=copy.deepcopy(plan);bad['wandb']['project']='someone-else'
            with self.assertRaises(ValueError):m.verify_operation(bad,root)
            bad=copy.deepcopy(plan);bad['executables'].pop(names[0])
            with self.assertRaises(ValueError):m.verify_operation(bad,root)
            (root/names[0]).write_text('changed')
            with self.assertRaises(ValueError):m.verify_operation(plan,root)

class CoordinatorFlow(unittest.TestCase):
    """Mocked SDK/process boundary; this is not a GPU or Slurm integration test."""
    def exercise(self,init_fails=False,finish_fails=False,telemetry_fails=False,running=False):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);scratch=root/'scratch';protected=root/'protected'
            scratch.mkdir();protected.mkdir();folder=SOURCE.parent
            names=('run_replayids_fidelity_profile.py','check_replayids_gpu_fidelity.py',
                'check_replayids_canonical_fidelity.py','GPU_FIDELITY_INPUTS.json')
            op=root/'OPERATION.json'
            plan={'schema':'replayids-fidelity-profile-v3','walltime_seconds':600,'verifier_deadline_seconds':480,
                'allocated_memory_mib':8192,'wandb_version':'0.23.0','wandb':{'entity':'csnet','project':'ofra-etg-leon-hpc',
                    'group':'replayids-score-fidelity-profile-v3'},'executables':{n:'a'*64 for n in names},
                'scratch_parent':str(scratch),'protected_parent':str(protected),
                'runtime_root':'synthetic-runtime','protected_input_root':'synthetic-input','data_root':'synthetic-data'}
            m.write(op,plan)
            class Run:
                url='https://wandb.invalid/synthetic-run'
                def log(self,metrics):pass
                def finish(self,exit_code):
                    if finish_fails:raise RuntimeError('synthetic finish failure')
            def init(**kwargs):
                self.assertTrue(kwargs['settings']['x_disable_meta'])
                self.assertTrue(kwargs['settings']['x_disable_viewer'])
                self.assertEqual(kwargs['mode'],'online')
                self.assertNotIn('job-123',kwargs['name'])
                if init_fails:raise RuntimeError('synthetic init failure')
                return Run()
            class Child:
                returncode=0
                polls=0
                def poll(self):
                    self.polls+=1
                    return None if running and self.polls==1 else 0
            telemetry_calls=[]
            def telemetry(command,env):
                telemetry_calls.append(dict(env))
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'GPU-a-b')
                if telemetry_fails:raise RuntimeError('synthetic telemetry error')
                return {'gpu_percent':25.,'gpu_memory_used_mib':10.,'gpu_memory_total_mib':100.}
            def popen(command,**kwargs):
                self.assertFalse(telemetry_fails, 'preflight failure must not launch verifier')
                out=Path(command[command.index('--output')+1]);out.mkdir()
                m.write(out/'GPU_FIDELITY.json',valid_report());return Child()
            sdk=types.SimpleNamespace(Settings=lambda **kw:kw,init=init)
            pwd=types.SimpleNamespace(getpwuid=lambda uid:types.SimpleNamespace(pw_name='test-owner'))
            core=types.SimpleNamespace(require_allocation=lambda *args:None)
            env={'SLURM_JOB_ID':'123','CUDA_VISIBLE_DEVICES':'GPU-a-b'}
            with patch.dict('sys.modules',{'wandb':sdk,'pwd':pwd,'check_replayids_gpu_fidelity':core}),\
                 patch.object(m.platform,'system',return_value='Linux'),patch.object(m.os,'getuid',return_value=1,create=True),\
                 patch.object(m,'inside',side_effect=lambda base,p:Path(p)),patch.object(m,'verify_operation'),\
                 patch.object(m,'prepare_tracking_environment',return_value={}),\
                 patch.object(m,'read_allocated_gpu',side_effect=telemetry),\
                 patch.object(m,'process_sample',return_value=(0.,5*1024**3 if running else 0)),\
                 patch.object(m.time,'sleep'),patch.object(m.signal,'signal'),\
                 patch.object(m.subprocess,'Popen',side_effect=popen),patch.dict(m.os.environ,env),\
                 patch.object(m.sys,'argv',['profile','--operation',str(op),'--operation-sha256',m.digest(op)]):
                code=m.main()
            target=protected/'job-123';status=m.load(target/'OPERATION_STATUS.json')
            receipt=m.load(target/'TRACKING_FINALIZATION.json')
            seal=m.load(target/'PROTECTED_COMPLETE.json')
            for n,h in seal['files'].items():self.assertEqual(m.digest(target/n),h)
            if running:self.assertGreaterEqual(len(telemetry_calls),2)
            return code,status,receipt
    def test_success_seals_evidence(self):
        code,status,receipt=self.exercise()
        self.assertEqual(code,0);self.assertTrue(status['scientific_pass'])
        self.assertEqual(receipt['status'],'CLIENT_FINISH_RETURNED')
    def test_tracking_init_fails_closed(self):
        code,status,receipt=self.exercise(init_fails=True)
        self.assertEqual(code,82);self.assertFalse(status['scientific_pass'])
        self.assertEqual(receipt['status'],'NOT_STARTED')
    def test_finish_failure_preserves_independent_evidence(self):
        code,status,receipt=self.exercise(finish_fails=True)
        self.assertEqual(code,82);self.assertTrue(status['scientific_pass'])
        self.assertEqual(receipt['status'],'CLIENT_FINISH_FAILED')

    def test_gpu_preflight_failure_does_not_launch_verifier(self):
        code,status,receipt=self.exercise(telemetry_fails=True)
        self.assertEqual(code,126)
        self.assertFalse(status['scientific_pass'])
        self.assertEqual(status['status'],'TELEMETRY_PREFLIGHT_FAILURE')
        self.assertIsNone(status['verifier_exit_code'])
        self.assertEqual(receipt['status'],'NOT_STARTED')

    def test_running_child_query_survives_sdk_environment_minimization(self):
        code,status,receipt=self.exercise(running=True)
        self.assertEqual(code,0)
        self.assertTrue(status['scientific_pass'])
        self.assertNotEqual(status['status'],'HOST_MEMORY_GUARD')

class TrackingPrivacy(unittest.TestCase):
    def test_history_rejects_private_fields_and_text(self):
        for key in ('SLURM_JOB_ID','hostname','username','sample_id_sha256','private_path','exception'):
            with self.subTest(key=key),self.assertRaises(ValueError):m.history({key:'sensitive sentinel'})
        for value in ('/home/private','synthetic-owner','SLURM_JOB_ID=123',float('nan'),-1):
            with self.subTest(value=value),self.assertRaises(ValueError):m.history({'Time/operation_seconds':value})
    def test_identity_is_hash_only_and_config_exact(self):
        name,config=m.tracking_identity('a'*64)
        self.assertEqual(name,'replayids-fidelity-'+'a'*12)
        self.assertEqual(set(config),{'purpose','required_checkpoints','encoder_batch_size','atol','rtol','operation_sha256'})
        for value in ('job-123','/home/private','synthetic-owner','a'*64+'extra'):
            with self.assertRaises(ValueError):m.tracking_identity(value)
    def test_auth_metadata_only(self):
        m.validate_netrc(types.SimpleNamespace(st_mode=stat.S_IFREG|0o600,st_uid=1000),1000)
        for mode,uid in ((stat.S_IFLNK|0o600,1000),(stat.S_IFREG|0o644,1000),(stat.S_IFREG|0o600,0)):
            with self.assertRaises(ValueError):m.validate_netrc(types.SimpleNamespace(st_mode=mode,st_uid=uid),1000)
    def test_environment_minimization_and_sdk_pin(self):
        home=Path('/synthetic/home');metadata=types.SimpleNamespace(st_mode=stat.S_IFREG|0o600,st_uid=1000)
        env={'HOME':str(home),'PATH':'/usr/bin','SLURM_JOB_ID':'123','HOSTNAME':'gpu-private',
             'USER':'private-user','CUSTOM_PRIVATE_PATH':'/secret/path','WANDB_NAME':'private-name'}
        with patch.object(Path,'lstat',return_value=metadata),patch.object(m.importlib.metadata,'version',return_value='0.23.0'):
            clean=m.prepare_tracking_environment(env,home,1000,'0.23.0')
            self.assertFalse(any(k.startswith('SLURM_') for k in clean))
            self.assertFalse({'HOSTNAME','USER','CUSTOM_PRIVATE_PATH','WANDB_NAME'}&clean.keys())
            for key in ('WANDB_API_KEY','WANDB_API_KEY_FILE','WANDB_ACCESS_TOKEN'):
                with self.assertRaises(ValueError):m.prepare_tracking_environment({**env,key:'unused sentinel'},home,1000,'0.23.0')
        with patch.object(Path,'lstat',return_value=metadata),patch.object(m.importlib.metadata,'version',return_value='0.24.0'):
            with self.assertRaises(ValueError):m.prepare_tracking_environment(env,home,1000,'0.23.0')

class TelemetryRegression(unittest.TestCase):
    def test_original_compute_environment_is_explicit_and_not_mutated(self):
        env={'PATH':'synthetic-bin','LD_LIBRARY_PATH':'synthetic-driver-library',
             'CUDA_VISIBLE_DEVICES':'GPU-a-b','SLURM_STEP_GPUS':'GPU-a-b'}
        captured={}
        def call(command,**kwargs):
            captured.update(kwargs)
            self.assertEqual(command[1],'--id=GPU-a-b')
            return types.SimpleNamespace(stdout='25, 800, 40000\n')
        with patch.object(m.subprocess,'run',side_effect=call),patch.dict(m.os.environ,{},clear=True):
            sample=m.read_allocated_gpu(m.gpu_query('GPU-a-b'),env)
        self.assertEqual(captured['env'],env)
        self.assertIsNot(captured['env'],env)
        self.assertEqual(captured['timeout'],5)
        self.assertEqual(sample['gpu_percent'],25)
    def test_redacted_nonzero_diagnostic(self):
        error=m.subprocess.CalledProcessError(9,['private-command'],output='private-host',
            stderr='Failed to initialize NVML: Driver/library version mismatch; secret-sentinel')
        result=m.bounded_error(error,'allocated_gpu_query')
        encoded=json.dumps(result)
        self.assertEqual(result['return_code'],9)
        self.assertIn('driver_library_mismatch',result['stderr_known_error_tags'])
        for private in ('private-command','private-host','secret-sentinel'):
            self.assertNotIn(private,encoded)
    def test_timeout_and_truncated_diagnostic(self):
        error=m.subprocess.TimeoutExpired('private-command',5,output=b'x'*9000,stderr=b'private-sentinel')
        result=m.bounded_error(error,'allocated_gpu_query')
        self.assertTrue(result['stdout_hash_truncated'])
        self.assertEqual(result['stdout_bytes'],9000)
        self.assertNotIn('private-sentinel',json.dumps(result))
    def test_old_memory_contract_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as d:
            old={'schema':'replayids-fidelity-profile-v2','walltime_seconds':600,
                 'verifier_deadline_seconds':480,'allocated_memory_mib':4096}
            with self.assertRaises(ValueError):m.verify_operation(old,Path(d))

if __name__=='__main__':unittest.main()
