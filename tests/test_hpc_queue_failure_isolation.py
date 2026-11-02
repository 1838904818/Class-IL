"""Software-only smoke fixtures: a failed child must not abort later cells."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


class FailureIsolation(unittest.TestCase):
    def exercise(self,name,expected):
        parent=Path(os.environ['NCD_TEST_TMP_ROOT']).resolve(strict=True)
        with tempfile.TemporaryDirectory(prefix='queue-fixture-',dir=parent) as folder:
            base=Path(folder).resolve()
            self.assertTrue(base.is_relative_to(parent))
            for part in ('data','cache','frozen'):(base/part).mkdir()
            spec=importlib.util.spec_from_file_location('queue_fixture_'+name,ROOT/'ncd/hpc'/name)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            commands=[]
            class Child:
                def __init__(self,command,**kwargs):
                    commands.append(command)
                    self.pid=100000+len(commands)
                    self.rc=1 if len(commands)==1 else 0
                    if self.rc:
                        kwargs['stderr'].write(b'TEST_FIXTURE_ONLY: simulated child traceback\n')
                    else:
                        if '--output-dir' in command:
                            target=Path(command[command.index('--output-dir')+1])
                        else:
                            target=Path(command[command.index('--output-root')+1])/'smoke'/command[command.index('--dataset')+1].lower()/command[command.index('--method')+1]/('seed'+command[command.index('--seed')+1])
                        target.mkdir(parents=True,exist_ok=True)
                        (target/'summary.json').write_text(json.dumps(dict(completed=True,
                            scientific_result=False,test_fixture_only=True)),encoding='utf-8')
                def poll(self):return self.rc
                def wait(self,timeout=None):return self.rc
                def terminate(self):raise AssertionError('Completed fixture child must not be terminated')
                def kill(self):raise AssertionError('Completed fixture child must not be killed')
            env=dict(NCD_OUTPUT_ROOT=str(base/'output'),NCD_DATA_ROOT=str(base/'data'),
                NCD_CACHE_ROOT=str(base/'cache'),NCD_FROZEN_ROOT=str(base/'frozen'),
                NCD_TABT_DEPENDENCY_ROOT=str(base/'frozen'))
            with patch.dict(os.environ,env),patch('sys.argv',[name,'--smoke']),\
                    patch.object(module.subprocess,'Popen',Child),\
                    patch.object(module.time,'sleep',lambda _:None),\
                    patch.object(module.signal,'signal',lambda *_:None):
                with self.assertRaisesRegex(RuntimeError,'Cells failed after all other work was preserved'):
                    module.main()
            self.assertEqual(len(commands),expected)
            self.assertEqual(len(list((base/'output').rglob('summary.json'))),expected-1)

    def test_job_a(self):self.exercise('cse_job_a.py',2)
    def test_job_b(self):self.exercise('cpu_job_b.py',12)
    def test_job_c(self):self.exercise('tabt_job_c.py',2)


if __name__=='__main__':unittest.main()
