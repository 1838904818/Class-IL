"""Publication integrity tests; these do not independently rerun neural inference."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'results/replayids-forward-parity'

class Tests(unittest.TestCase):
    def setUp(self):
        self.report=json.loads((RESULT/'FORWARD_PARITY_CHECKED.json').read_text())
        self.batch=json.loads((RESULT/'FORWARD_PARITY_BATCH898.json').read_text())
    def test_source_binding(self):
        source=(ROOT/'tools/check_replayids_forward_parity.py').read_bytes().replace(b'\r\n',b'\n')
        for r in (self.report,self.batch):
            self.assertEqual(hashlib.sha256(source).hexdigest(),r['audit_source_lf_sha256'])
    def test_protected_state_continuity(self):
        readiness=json.loads((ROOT/'results/replayids-score-readiness/READINESS.json').read_text())
        states={r['seed']:r for r in readiness['states']}
        for row in self.report['seeds']:
            for key in ('inference_state_sha256','checkpoint_manifest_sha256'):
                self.assertEqual(row[key],states[row['seed']][key])
    def test_failed_gate_not_promoted(self):
        self.assertEqual([s['seed'] for s in self.report['seeds']],[1,2,3,4,42])
        self.assertFalse(self.report['all_passed'])
        for row in self.report['seeds']:
            self.assertTrue(row['scores']['head_scores']['allclose'])
            self.assertFalse(row['scores']['router_z_scores']['allclose'])
            self.assertFalse(row['scores']['joint_scores']['allclose'])
            self.assertFalse(row['passed'])
            self.assertEqual(row['prediction_mismatches'],len(row['prediction_mismatch_records']))
    def test_batch_diagnostic_remains_failure(self):
        self.assertEqual(self.batch['batch_size'],898)
        self.assertEqual(len(self.batch['seeds']),1)
        self.assertEqual(self.batch['seeds'][0]['seed'],4)
        self.assertEqual(self.batch['seeds'][0]['prediction_mismatches'],0)
        self.assertFalse(self.batch['all_passed'])
        for key in ('atol','rtol','probe_manifest_sha256'):
            self.assertEqual(self.batch[key],self.report[key])
    def test_sample_scope_and_no_fitting(self):
        self.assertEqual(self.report['probe_row_count'],898)
        self.assertEqual(sum(s['prediction_mismatches'] for s in self.report['seeds']),1)
        self.assertFalse(self.report['calibrator_fitted'])
        self.assertFalse(self.report['new_test_metrics_computed'])
        self.assertEqual(self.report['environment']['device'],'cpu')

if __name__=='__main__':unittest.main()
