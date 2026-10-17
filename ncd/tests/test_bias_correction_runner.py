import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from dataclasses import dataclass

import numpy as np
from ncd.run_bias_correction import (
    ROOT, validate_request, training_holdout, reserve_calibration,
    balanced_calibration, old_memory_arrays, cell_lock, verify_formal_selection,
    digest, source_commit, configuration_input_hashes, SOURCE_PATHS,
)


@dataclass
class ToyData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    class_order: list


class RunnerTests(unittest.TestCase):
    def test_formal_requires_frozen_selection(self):
        with self.assertRaisesRegex(ValueError, 'require'):
            verify_formal_selection(None, {}, 'NSL-KDD', 'replay', 'wa', 0, ROOT / 'missing')

    def test_formal_selection_recomputes_development_mean_and_winner(self):
        config = ROOT / 'ncd/configs/bias_correction_section23.json'
        spec = json.loads(config.read_text())
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            groups = [{'dataset': d, 'base': b, 'correction': c}
                      for d in spec['datasets'] for b in ('replay', spec['best_methods'][d])
                      for c in spec['candidates']]
            chosen = next(g for g in groups if (g['dataset'], g['base'], g['correction'])
                          == ('NSL-KDD', 'replay', 'bic'))
            chosen.update(selected_candidate=1, candidates=[])
            for candidate, values in enumerate(((.2, .4), (.6, .8))):
                proofs = []
                for seed, value in zip((5, 6), values):
                    result = directory / f'dev_{candidate}_{seed}.json'
                    result.write_text(json.dumps({'completed': True, 'identity': {
                        'smoke': False, 'phase': 'development', 'seed': seed,
                        'candidate': candidate, 'dataset': 'NSL-KDD', 'base_method': 'replay',
                        'correction': 'bic', 'evaluation_role': 'training_holdout_not_official_ID_test'},
                        'checkpoints': [{'id': {'macro_f1': value}}]}), encoding='utf-8')
                    proofs.append({'seed': seed, 'result': result.name, 'result_sha256': digest(result)})
                chosen['candidates'].append({'candidate': candidate,
                    'mean_validation_macro_f1': sum(values) / 2, 'development_results': proofs})
            selection = {'selection_seeds': [5, 6], 'selection_metric': spec['selection_metric'],
                         'provenance': {'config_sha256': digest(config),
                                        'git_commit': source_commit(ROOT),
                                        'config_input_hashes': configuration_input_hashes(spec),
                                        'source_hashes': {p: digest(ROOT / p) for p in SOURCE_PATHS}},
                         'selections': groups}
            path = directory / 'selection.json'
            path.write_text(json.dumps(selection), encoding='utf-8')
            proof = verify_formal_selection(path, spec, 'NSL-KDD', 'replay', 'bic', 1, config)
            self.assertEqual(proof['selected_candidate'], 1)
            with self.assertRaisesRegex(ValueError, 'differs'):
                verify_formal_selection(path, spec, 'NSL-KDD', 'replay', 'bic', 0, config)
            inputs = selection['provenance']['config_input_hashes']
            key = next(iter(inputs))
            original = inputs[key]
            inputs[key] = '0' * 64
            path.write_text(json.dumps(selection), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'inputs changed'):
                verify_formal_selection(path, spec, 'NSL-KDD', 'replay', 'bic', 1, config)
            inputs[key] = original
            chosen['candidates'][1]['mean_validation_macro_f1'] = .99
            path.write_text(json.dumps(selection), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'mean differs'):
                verify_formal_selection(path, spec, 'NSL-KDD', 'replay', 'bic', 1, config)

    def test_seed_roles_and_budget_freeze(self):
        spec = json.loads((ROOT / 'ncd/configs/bias_correction_section23.json').read_text())
        validate_request(spec, 'NSL-KDD', 'replay', 'wa', 0, 5, 'development')
        validate_request(spec, 'NSL-KDD', 'replay_lwf', 'wa', 0, 42, 'formal')
        for seed, phase in [(1, 'development'), (5, 'formal'), (42, 'development')]:
            with self.assertRaises(ValueError):
                validate_request(spec, 'NSL-KDD', 'replay', 'wa', 0, seed, phase)
        spec['batch_size'] = 384
        with self.assertRaises(ValueError):
            validate_request(spec, 'NSL-KDD', 'replay', 'wa', 0, 5, 'development')

    def test_development_never_uses_official_test(self):
        x = np.arange(120, dtype=np.float32).reshape(60, 2)
        y = np.repeat(np.arange(3), 20)
        data = ToyData(x, y, np.full((30, 2), -999, np.float32), np.zeros(30, np.int64), ['a', 'b', 'c'])
        out, split = training_holdout(data, 5, .1)
        self.assertEqual(len(out.y_train), 54)
        self.assertEqual(len(out.y_test), 6)
        self.assertTrue(split['disjoint_indices'])
        self.assertFalse(np.intersect1d(out.x_train[:, 0], out.x_test[:, 0]).size)
        self.assertTrue(np.all(out.x_test >= 0))
        self.assertEqual(len(set(out.x_train[:, 0]) | set(out.x_test[:, 0])), 60)
        with self.assertRaises(ValueError):
            training_holdout(data, 42, .1)

    def test_persistent_calibration_never_enters_training(self):
        x = np.arange(460, dtype=np.float32).reshape(230, 2)
        y = np.r_[np.zeros(200, np.int64), np.ones(30, np.int64)]
        tx, ty, hx, hy = reserve_calibration(x, y, 10000, 20, .1)
        self.assertEqual([len(v) for v in hy], [20, 3])
        self.assertEqual(len(ty), 207)
        self.assertFalse(np.intersect1d(tx[:, 0], np.concatenate(hx)[:, 0]).size)
        vx, vy = balanced_calibration(hx, hy, 12)
        self.assertEqual(np.bincount(vy).tolist(), [3, 3])
        self.assertTrue(set(vx[:, 0]).issubset(set(np.concatenate(hx)[:, 0])))

    def test_derpp_snapshot_filters_new_classes(self):
        import torch
        mem = SimpleNamespace(per_class=2, observations=[3, 1, 2],
                              x=torch.arange(12).reshape(3, 2, 2).float())
        x, y = old_memory_arrays(mem, [], [], (0, 1), 2)
        self.assertEqual(y.tolist(), [0, 0, 1])
        self.assertEqual(x.shape, (3, 2))

    def test_cell_lock_releases(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / '.execution.lock'
            with cell_lock(path):
                with self.assertRaises((OSError, RuntimeError)):
                    with cell_lock(path):
                        pass
            with cell_lock(path):
                pass


if __name__ == '__main__':
    unittest.main()
