"""Stored-score integrity checks, not a forward-inference test."""
import hashlib,json
from pathlib import Path
import sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from audit_replayids_checkpoint_coverage import validate_scores

class Tests(unittest.TestCase):
    def setup_case(self):
        records=[{'class_id':i,'shard_ordinal':0,'local_row':i,'sample_id_sha256':str(i)*64} for i in range(2)]
        head=np.array([[.8,.2],[.3,.7]],dtype=np.float32)
        router=np.array([[1,-1],[-1,1]],dtype=np.float32)
        joint=head+np.float32(.5)*router
        scores={'head_scores':head,'router_z_scores':router,'joint_scores':joint,'class_axis':np.array([0,1]),
          'true_class_id':np.array([0,1]),'shard_ordinal':np.array([0,0]),'local_row':np.array([0,1]),
          'sample_id_sha256':np.asarray([r['sample_id_sha256'].encode() for r in records],dtype='S64'),
          'predicted_class_id':joint.argmax(axis=1)}
        return scores,{'seen_classes':[0,1],'score_rows':2},records
    def test_valid(self):self.assertEqual(validate_scores(*self.setup_case()),2)
    def test_corrupt_joint(self):
        s,c,r=self.setup_case();s['joint_scores'][0,0]+=.1
        with self.assertRaisesRegex(ValueError,'joint'):validate_scores(s,c,r)
    def test_swapped_identity(self):
        s,c,r=self.setup_case();s['sample_id_sha256']=s['sample_id_sha256'][::-1]
        with self.assertRaisesRegex(ValueError,'identities'):validate_scores(s,c,r)
    def test_corrupt_class_axis(self):
        s,c,r=self.setup_case();s['class_axis']=np.array([1,0])
        with self.assertRaisesRegex(ValueError,'axis'):validate_scores(s,c,r)
    def test_wrong_argmax(self):
        s,c,r=self.setup_case();s['predicted_class_id'][0]=1
        with self.assertRaisesRegex(ValueError,'argmax'):validate_scores(s,c,r)
    def test_published_coverage_and_source(self):
        d=json.loads((ROOT/'results/replayids-checkpoint-coverage/CHECKPOINT_COVERAGE.json').read_text())
        self.assertEqual({(r['seed'],r['checkpoint']) for r in d['rows']},{(s,c) for s in [1,2,3,4,42] for c in range(4)})
        self.assertEqual(d['artifact_count'],60)
        self.assertFalse(d['new_forward_inference_performed']);self.assertFalse(d['new_forgetting_computed'])
        code=(ROOT/'tools/audit_replayids_checkpoint_coverage.py').read_bytes().replace(b'\r\n',b'\n')
        self.assertEqual(hashlib.sha256(code).hexdigest(),d['source_lf_sha256'])
        for r in d['rows']:
            self.assertEqual(r['probe_rows'],[256,512,768,898][r['checkpoint']])

if __name__=='__main__':unittest.main()
