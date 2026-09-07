"""Synthetic split tests and checks of published readiness metadata."""
import hashlib
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from audit_replayids_score_readiness import indices,index_hash,audit

class Tests(unittest.TestCase):
    def test_published_source_hash(self):
        d=json.loads((ROOT/'results/replayids-score-readiness/READINESS.json').read_text())
        code=(ROOT/'tools/audit_replayids_score_readiness.py').read_bytes().replace(b'\r\n',b'\n')
        self.assertEqual(hashlib.sha256(code).hexdigest(),d['audit_source_lf_sha256'])
    def test_tampered_input_fails_closed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'streaming_manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Pinned input hash mismatch'):
                audit(p,p/'registry',p/'protocols')
    def test_disjoint_reproducible_splits(self):
        a=indices(42,'f'*64,0,100,.1,30);b=indices(42,'f'*64,0,100,.1,30)
        self.assertEqual(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1]);np.testing.assert_array_equal(a[2],b[2])
        self.assertEqual((len(a[1]),len(a[2])),(10,30))
        self.assertEqual(np.intersect1d(a[1],a[2]).size,0)
    def test_single_row_and_rare_class(self):
        _,cal,fit=indices(42,'a'*64,6,1,.1,999)
        self.assertEqual((len(cal),len(fit)),(0,1))
        _,cal,fit=indices(42,'a'*64,6,7,.1,999)
        self.assertEqual((len(cal),len(fit)),(1,6))
    def test_invalid_parameters_rejected(self):
        for n,frac,cap in [(0,.1,10),(10,1.,10),(10,-.1,10),(10,.1,0)]:
            with self.assertRaises(ValueError):indices(42,'a'*64,0,n,frac,cap)
    def test_published_index_hashes_recompute(self):
        d=json.loads((ROOT/'results/replayids-score-readiness/READINESS.json').read_text())
        source='99f6de7a6cdd09b91e9bc0e167304db4d4274e753adad5138ec7803f5338a15b'
        for row in d['classes']:
            cap=124780 if row['class_id']==0 else row['source_train_rows']
            _,cal,fit=indices(42,source,row['class_id'],row['source_train_rows'],.1,cap)
            self.assertEqual(index_hash(cal),row['calibration_indices_sha256'])
            self.assertEqual(index_hash(fit),row['fit_indices_sha256'])
        self.assertFalse(d['readiness']['forward_reconstruction_parity_verified'])
        self.assertFalse(d['readiness']['new_calibrator_fitted'])

if __name__=='__main__':unittest.main()
