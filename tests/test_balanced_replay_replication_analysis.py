"""Regression tests for the published paired aggregation tool."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_balanced_replay_replication import SEEDS, METRICS, analyze, exact_p, holm, summary

class AnalysisTests(unittest.TestCase):
    def test_exact_minima_and_zeros(self):
        for ranked in (True, False):
            self.assertEqual(exact_p([1]*5, ranked), .0625)
            self.assertEqual(exact_p([1]*4, ranked), .125)
            self.assertEqual(exact_p([0]*5, ranked), 1)
            self.assertEqual(exact_p([1,-1,1,-1], ranked), 1)

    def test_holm_and_sample_sd(self):
        self.assertEqual(holm({'a':.01,'b':.04,'c':.03}), {'a':.03,'c':.06,'b':.06})
        self.assertAlmostEqual(summary([1,2,3,4])['sample_sd'], 1.2909944487358056)

    def write_pairs(self, root):
        paths=[]
        for s in SEEDS:
            b={k:(-.02 if k=='average_forgetting' else .4) for k in METRICS[:5]}
            audit=dict(status='verified',seed=s,dataset='malaya-network-gt',result_file_sha256='a'*64,
                       methods={'balanced_replay50':b})
            ap=root/f'seed{s}_audit.json'; ap.write_text(json.dumps(audit))
            pair=dict(status='verified_single_seed_pair',seed=s,dataset='malaya-network-gt',
                baseline_result_sha256='a'*64,baseline_audit_sha256=hashlib.sha256(ap.read_bytes()).hexdigest(),
                metrics={k:dict(balanced_replay50=v,ofra_joint_cap3000=v+.1,ofra_minus_replay_pp=10.) for k,v in b.items()},
                checkpoints=5,final_test_rows=5,per_class=[dict(class_id=0,class_name='test',support=5)])
            pp=root/f'seed{s}_comparison.json';pp.write_text(json.dumps(pair));paths.append(pp)
        return paths

    def test_pairing_sensitivity_and_signed_forgetting(self):
        with tempfile.TemporaryDirectory() as temp:
            paths=self.write_pairs(Path(temp))
            result=analyze(paths)
            self.assertEqual(json.dumps(result), json.dumps(analyze(list(reversed(paths)))))
            cohorts=result['cohorts']
            self.assertEqual(cohorts['new_seeds_1_to_4_sensitivity']['seeds'],[1,2,3,4])
            m=cohorts['all_five_descriptive']['metrics']['average_forgetting']
            self.assertEqual(m['balanced_replay50']['mean'],-2)
            self.assertEqual(m['ofra_losses'],5)

    def test_missing_duplicate_and_tampered_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            paths=self.write_pairs(Path(temp))
            with self.assertRaises(ValueError): analyze(paths[:-1])
            with self.assertRaises(ValueError): analyze(paths+[paths[0]])
            p=paths[0].with_name('seed1_audit.json');p.write_text(p.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'hash mismatch'): analyze(paths)

    def test_missing_metric_and_nonfinite_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            paths=self.write_pairs(Path(temp)); original=json.loads(paths[0].read_text())
            row=copy.deepcopy(original);row['metrics'].pop('average_task_accuracy')
            paths[0].write_text(json.dumps(row))
            with self.assertRaises(ValueError): analyze(paths)
            row=copy.deepcopy(original);row['metrics']['final_macro_f1']['ofra_joint_cap3000']=float('nan')
            paths[0].write_text(json.dumps(row))
            with self.assertRaises(ValueError): analyze(paths)

    def test_manifest_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); paths=self.write_pairs(root)
            (root/'EXPORT_SHA256.json').write_text(json.dumps({paths[0].name:'0'*64}))
            with self.assertRaisesRegex(ValueError,'manifest mismatch'): analyze(paths)

if __name__=='__main__': unittest.main()
