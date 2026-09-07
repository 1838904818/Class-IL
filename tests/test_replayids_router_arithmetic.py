"""Synthetic numerical tests plus integrity of the frozen-probe diagnostic."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from isolate_replayids_router_arithmetic import stable_distances,stable_router,difference

class Tests(unittest.TestCase):
    def test_exact_self_and_separated_distance(self):
        x=np.asarray([[0,0],[3,4]],dtype=np.float32)
        np.testing.assert_array_equal(stable_distances(x,x[:1]),[0,-5])
    def test_large_nearly_equal_coordinates(self):
        x=np.asarray([[1000.01,1000.01]],dtype=np.float32)
        c=np.asarray([[1000.,1000.]],dtype=np.float32)
        expected=-np.linalg.norm(x.astype(np.float64)-c.astype(np.float64),axis=1)
        np.testing.assert_array_equal(stable_distances(x,c),expected)
        self.assertLess(float(expected[0]),0)
    def test_invalid_input(self):
        for x,c in [(np.zeros((2,2)),np.zeros((0,2))),(np.zeros((2,3)),np.zeros((1,2))),(np.array([[np.nan,0]]),np.zeros((1,2)))]:
            with self.assertRaises(ValueError):stable_distances(x,c)
    def test_batch_invariance_synthetic(self):
        rng=np.random.default_rng(14)
        x=rng.normal(size=(13,7)).astype(np.float32)
        states={i:SimpleNamespace(centroids=rng.normal(size=(20,7)).astype(np.float32)) for i in range(3)}
        full=stable_router(x,states,[0,1,2])
        split=np.vstack([stable_router(x[i:i+2],states,[0,1,2]) for i in range(0,len(x),2)])
        np.testing.assert_array_equal(full,split)
    def test_difference_metric(self):
        d=difference(np.asarray([1.,2.]),np.asarray([1.,3.]))
        self.assertEqual(d['max_abs_error'],1)
        self.assertEqual(d['changed_cells'],1)
    def test_published_source_and_scope(self):
        r=json.loads((ROOT/'results/replayids-router-arithmetic/ROUTER_ARITHMETIC_ISOLATION.json').read_text())
        source=(ROOT/'tools/isolate_replayids_router_arithmetic.py').read_bytes().replace(b'\r\n',b'\n')
        self.assertEqual(hashlib.sha256(source).hexdigest(),r['source_lf_sha256'])
        self.assertEqual([x['seed'] for x in r['seed_results']],[1,2,3,4,42])
        self.assertFalse(r['new_candidate_promoted'])
        self.assertFalse(r['calibration_fitted'])
        for row in r['seed_results']:
            self.assertEqual(row['direct64_router_batch32_vs_full']['changed_cells'],0)
            self.assertGreater(max(x['router_vs_same_embedding_full_batch']['max_abs_error'] for x in row['batch_checks']),0)
            self.assertEqual(row['direct64_router_embedding32_vs_full']['outside_fixed_tolerance'],0)

if __name__=='__main__':unittest.main()
