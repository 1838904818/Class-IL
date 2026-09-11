"""Synthetic CPU tests only; never load project tensors or request CUDA."""
import sys
import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'etg_exploratory_v1')]
from native_target import CallBudget, FixedContextTarget
from shap_core import extract_one
import local_shap


class ShapProfileTests(unittest.TestCase):
    def target(self, max_calls=1000, zero=False):
        def score(raw):
            p = np.zeros((len(raw), 2), dtype=np.float32)
            p[:, 0] = .5 if zero else np.float32(.5) + raw[:, 0]*np.float32(.1)
            p[:, 1] = np.float32(.5)
            z = np.zeros_like(p)
            return dict(class_axis=np.array([0, 1]), head_scores=p,
                        router_z_scores=z, joint_scores=p.copy(),
                        predicted_class_id=p.argmax(1))
        parent = np.array([[1, 2, 3], [2, 1, 0]], dtype=np.float32)
        return FixedContextTarget(score, parent, 0, [0, 1], 0, CallBudget(max_calls, 60))

    def test_complete_real_shap(self):
        t = self.target()
        result = extract_one(t, np.zeros((2, 3), dtype=np.float32), 20260911)
        self.assertEqual(len(result['attributions']), 3)
        self.assertLess(abs(result['additive_residual']), 1e-6)
        self.assertGreater(result['native_calls'], 2)
        self.assertEqual(result['repeat_count'], 1)
        self.assertFalse(result['statistical_certificate'])
        self.assertAlmostEqual(result['attributions'][0], .1, places=6)

    def test_deterministic_seed(self):
        refs = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.float32)
        a = extract_one(self.target(), refs, 7)
        b = extract_one(self.target(), refs, 7)
        self.assertEqual(a['attributions'], b['attributions'])
        self.assertEqual(a['native_calls'], b['native_calls'])

    def test_exhausted_budget(self):
        with self.assertRaises(ValueError):
            extract_one(self.target(1), np.zeros((2, 3), dtype=np.float32), 7)

    def test_bad_reference_dtype(self):
        with self.assertRaises(ValueError):
            extract_one(self.target(), np.zeros((2, 3), dtype=np.float64), 7)

    def test_invalid_seed(self):
        with self.assertRaises(ValueError):
            extract_one(self.target(), np.zeros((2, 3), dtype=np.float32), -1)

    def test_zero_is_not_efficacy(self):
        r = extract_one(self.target(zero=True), np.zeros((2, 3), dtype=np.float32), 7)
        self.assertEqual(r['l1_norm'], 0)
        self.assertTrue(r['zero_vector_means_feasibility_only'])

    def test_wrong_candidate_hash(self):
        with self.assertRaises(ValueError):
            local_shap.candidate(__file__, '0'*64)

    def test_journal_is_line_delimited_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'progress.jsonl'
            rows = [{'stage': 'SHAP_START'}, {'stage': 'NATIVE_PROGRESS', 'native_calls': 100},
                    {'stage': 'SHAP_COMPLETE'}]
            with path.open('x', encoding='utf-8') as f:
                for row in rows:
                    local_shap.journal_record(f, row)
            self.assertEqual([json.loads(line) for line in path.read_text().splitlines()], rows)

    def test_journal_rejects_nonfinite_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'progress.jsonl'
            with path.open('x', encoding='utf-8') as f:
                with self.assertRaises(ValueError):
                    local_shap.journal_record(f, {'value': float('nan')})
            self.assertEqual(path.stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()
