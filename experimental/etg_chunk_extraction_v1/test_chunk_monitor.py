import unittest
from types import SimpleNamespace
from pathlib import Path
from chunk_monitor import *


class MonitorTests(unittest.TestCase):
    def test_sstat_units(self):
        r=parse_sstat('123.0|4156264K|4156264K|00:04:07.520|1\n','123.0')
        self.assertEqual(r['max_rss_bytes'],4156264*1024)
        self.assertEqual(r['cpu_seconds'],247.52)
        self.assertEqual(memory_bytes('1.5G'),1610612736)
    def test_missing_and_foreign_rejected(self):
        for line in ('','124.0|1G|1G|00:00:01|1','123.0||1G|00:00:01|1',
                     '123.0|1G|1G|00:00:01|2','123.0|1G|1G|00:61:01|1'):
            with self.assertRaises(ValueError):parse_sstat(line,'123.0')
    def test_rpc_hard_budget(self):
        seen=[]
        def run(args,**kw):seen.append(args);return SimpleNamespace(stdout='123.0|1G|1G|00:00:01|1')
        s=LimitedSstat('123','0',run)
        for event in ('initial','midpoint','terminal'):s.sample(event)
        with self.assertRaises(ValueError):s.sample('initial_retry')
        self.assertEqual(len(seen),3)
        self.assertTrue(all('--jobs=123.0' in cmd for cmd in seen))
    def test_failed_rpc_is_spent(self):
        def run(*a,**k):raise RuntimeError('fake failure')
        s=LimitedSstat('123','0',run)
        with self.assertRaises(RuntimeError):s.sample('initial')
        self.assertEqual(s.calls,1)
    def test_interval_not_lifetime_cpu(self):
        self.assertEqual(interval_cpu(100,101,10),5)
        with self.assertRaises(ValueError):interval_cpu(100,99,10)
    def test_own_cgroup_only(self):
        p=own_cgroup_v2('0::/slurm/uid_5/job_123/step_0','123','0',Path('/unused'))
        self.assertEqual(p.name,'step_0')
        for row in ('0::/slurm/job_124/step_0','0::/slurm/job_123/step_1','0::/slurm/job_123/../step_0'):
            with self.assertRaises(ValueError):own_cgroup_v2(row,'123','0',Path('/unused'))
    def test_low_use_waits_for_window_and_grace(self):
        w=Window()
        for elapsed in range(0,540,15):self.assertIsNone(w.update(elapsed,5,5))
        self.assertEqual(w.update(540,5,5),'LOW_UTILIZATION')
    def test_single_gpu_dip_not_failure(self):
        w=Window()
        for elapsed in range(0,900,15):self.assertIsNone(w.update(elapsed,50,0 if elapsed==450 else 80))
    def test_missing_telemetry_is_not_zero(self):
        w=Window();w.update(0,50,80)
        self.assertEqual(w.update(45,50,80),'MONITORING_INSUFFICIENT')
    def test_jittered_window_clips_first_interval(self):
        w=Window()
        for t,cpu in ((240,50),(255,30),(275,5),(290,5),(305,5)):
            self.assertIsNone(w.update(t,cpu,80))
        self.assertEqual(w.low_since,305)

if __name__=='__main__':unittest.main()
