"""Canonical batching unit tests and published diagnostic integrity."""
import hashlib,json
from pathlib import Path
import sys,unittest
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from check_replayids_canonical_fidelity import canonical_score,compare

class Tests(unittest.TestCase):
    def test_encoder_chunked_heads_router_whole(self):
        import torch
        calls={'encoder':[],'heads':[],'router':[]}
        class Encoder(torch.nn.Module):
            def forward(self,x):calls['encoder'].append(len(x));return x
        class Head(torch.nn.Module):
            def forward(self,x):calls['heads'].append(len(x));return x[:,:2]
        class Router:
            def scores(self,x,seen,variant):
                calls['router'].append(len(x));return np.zeros((len(x),len(seen)),dtype=np.float32)
        model=SimpleNamespace(mean=np.zeros(2),scale=np.ones(2),encoder=Encoder(),heads={0:Head(),1:Head()},router=Router(),metadata={'seen_classes':[0,1]},device=torch.device('cpu'))
        result=canonical_score(model,np.ones((5,2),dtype=np.float32),2)
        self.assertEqual(calls,{'encoder':[2,2,1],'heads':[5,5],'router':[5]})
        np.testing.assert_array_equal(result['predicted_class_id'],np.zeros(5))
    def test_compare_detects_score_change_without_argmax_change(self):
        original={'class_axis':np.array([0,1]),'predicted_class_id':np.array([0])}
        for k in ['head_scores','router_z_scores','joint_scores']:original[k]=np.array([[2.,1.]])
        changed={k:v.copy() for k,v in original.items()};changed['router_z_scores'][0,0]+=.1
        errors,ids=compare(changed,original)
        self.assertEqual(len(ids),0);self.assertEqual(errors['router_z_scores']['outside_tolerance_cells'],1)
    def test_compare_rejects_nonfinite(self):
        d={'class_axis':np.array([0,1]),'head_scores':np.array([[np.nan,0.]])}
        with self.assertRaises(ValueError):compare(d,d)
    def test_compare_rejects_class_axis(self):
        with self.assertRaises(ValueError):compare({'class_axis':np.array([0,1])},{'class_axis':np.array([1,0])})
    def test_report_integrity(self):
        d=json.loads((ROOT/'results/replayids-canonical-fidelity/CANONICAL_CPU_FIDELITY.json').read_text())
        source=(ROOT/'tools/check_replayids_canonical_fidelity.py').read_bytes().replace(b'\r\n',b'\n')
        self.assertEqual(hashlib.sha256(source).hexdigest(),d['source_lf_sha256'])
        self.assertTrue(d['coverage_complete']);self.assertEqual(len(d['checkpoints']),20)
        self.assertFalse(d['environment_matched']);self.assertFalse(d['all_passed'])
        self.assertFalse(d['calibrator_fitted']);self.assertFalse(d['new_test_performance_computed'])
        for r in d['checkpoints']:
            self.assertEqual(r['encoder_batch_size'],512)
            self.assertEqual(r['prediction_mismatch_count'],len(r['mismatches']))
            expected=r['prediction_mismatch_count']==0 and not any(v['outside_tolerance_cells'] for v in r['scores'].values())
            self.assertEqual(r['passed'],expected)

if __name__=='__main__':unittest.main()
