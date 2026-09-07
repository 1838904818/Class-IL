import unittest
import tempfile, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from audit_saved_evidence import signflip_p, holm, jaccard, source_ledger, primary_pairs, SEEDS

class AuditMathTests(unittest.TestCase):
    def test_five_pairs_minimum(self): self.assertEqual(signflip_p([1,2,3,4,5]), .0625)
    def test_four_pairs_minimum(self): self.assertEqual(signflip_p([1,2,3,4]), .125)
    def test_zero_differences(self): self.assertEqual(signflip_p([0,0,0]),1.)
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError): signflip_p([1,float('nan')])
    def test_holm_reordering(self): self.assertEqual(holm([.04,.01,.03]),[.06,.03,.06])
    def test_holm_capped(self): self.assertEqual(holm([.9,.8]),[1.,1.])
    def test_overlap(self): self.assertEqual(jaccard([1,2],[2,3]),1/3)
    def test_historical_implementation_hash_required(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'modified.py';p.write_text('raise RuntimeError("must never execute")')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):source_ledger(p,.7)
    def fixture(self):
        return {'checkpoint_policies':{'primary':'Job 425539 last epoch, not superseded'},
            'inference_primary':'official/joint_cap3000','cohorts':{'all_five_descriptive':{
                'seeds':SEEDS,'versus_replay':{'primary':{str(k):{'ofra_minus_replay_pp':{'values':[1,2,3,4,5]}} for k in range(7)}}}}}
    def load_fixture(self,x):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'primary.json';p.write_text(json.dumps(x));return primary_pairs(p)
    def test_primary_pair_extraction(self):
        rows=self.load_fixture(self.fixture())
        self.assertEqual(rows[-1]['seed'],42)
        self.assertEqual(rows[-1]['metrics']['0']['ofra_minus_replay_pp'],5)
    def test_guarded_policy_not_primary(self):
        x=self.fixture();x['checkpoint_policies']['primary']='Job 426307 guarded checkpoint'
        with self.assertRaisesRegex(ValueError,'checkpoint policy'):self.load_fixture(x)
    def test_changed_seed_order_rejected(self):
        x=self.fixture();x['cohorts']['all_five_descriptive']['seeds']=[42,1,2,3,4]
        with self.assertRaisesRegex(ValueError,'seed order'):self.load_fixture(x)

if __name__=='__main__': unittest.main()
