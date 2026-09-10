"""Synthetic temporary arrays and mocked processes only; no research tensors."""
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
import weakref
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'etg_exploratory_v1'))
import bundle
import local_profile as launcher
import profile_core
from pilot_core import digest
from native_target import array_sha


class Fixture:
    def __init__(self, root):
        self.source, self.native = root / 'source', root / 'native'
        self.source.mkdir(); self.native.mkdir(); self.mapping = {}
        for cid in (0, 1):
            values = (np.arange(900*78).reshape(900, 78)/10000 + cid).astype(np.float32)
            cal = np.arange(617, dtype=np.int64)
            source_path = self.source / f'class{cid}.npy'; np.save(source_path, values)
            native_path = self.native / f'cal{cid}.npy'; np.save(native_path, values[cal])
            ordinals = [139, 575]
            self.mapping[cid] = {'source': {'path': source_path.name, 'sha256': launcher.file_sha(source_path)},
                'source_rows': 900, 'calibration': cal, 'background': list(range(700, 716)),
                'calibration_file': {'path': native_path.name, 'sha256': launcher.file_sha(native_path)},
                'roles': {'trigger': {'offsets': ordinals, 'ordinals': ordinals,
                    'row_ids': [digest([cid, i]) for i in ordinals]}}}
        self.targets = bundle.selected_targets(self.mapping)

    def inputs(self):
        return bundle.ProfileInputs(self.mapping, self.source, self.native, self.targets)


class Model:
    def __init__(self, cp):
        self.axis = np.arange(2 if cp == 0 else 4, dtype=np.int64)
        self.metadata = {'seen_classes': self.axis.tolist()}
        self.calls = []

    def score(self, raw):
        self.calls.append(len(raw))
        p = np.full((len(raw), len(self.axis)), .2, dtype=np.float32)
        p[:, 0] = .4 + .001*raw[:, 0] + .001*raw[:, 0].mean()
        z = np.zeros_like(p)
        return {'class_axis': self.axis.copy(), 'head_scores': p, 'router_z_scores': z,
            'joint_scores': p.copy(), 'predicted_class_id': self.axis[p.argmax(1)]}


def captured(model, raw):
    return dict(model.score(raw), binary_logits=np.zeros((len(raw), len(model.axis), 2), np.float32))


class BundleTests(unittest.TestCase):
    def test_reconstruction_keeps_full_and_partial_contexts(self):
        with tempfile.TemporaryDirectory() as d:
            f = Fixture(Path(d)); inputs = f.inputs()
            self.assertEqual([len(inputs.contexts[c]) for c in (0, 1)], [512, 105])
            self.assertEqual([t['row_in_parent'] for t in inputs.targets], [139, 63])
            self.assertEqual(inputs.references.shape, (32, 78))
            self.assertFalse(hasattr(inputs, 'evaluation'))
            inputs.recheck_files()

    def test_file_tampering_blocks_before_mapping(self):
        with tempfile.TemporaryDirectory() as d:
            f = Fixture(Path(d)); (f.native/'cal0.npy').write_bytes(b'bad-fixture')
            with self.assertRaisesRegex(ValueError, 'digest'):
                f.inputs()

    def test_native_rows_differ_from_bound_source_even_with_new_hash(self):
        with tempfile.TemporaryDirectory() as d:
            f = Fixture(Path(d)); path = f.native/'cal0.npy'
            values = np.load(path); values[0, 0] += 1; np.save(path, values)
            f.mapping[0]['calibration_file']['sha256'] = launcher.file_sha(path)
            with self.assertRaisesRegex(ValueError, 'reconstructed'):
                f.inputs()

    def test_same_size_alternate_target_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            f = Fixture(Path(d)); f.targets[0]['calibration_ordinal'] += 1
            with self.assertRaisesRegex(ValueError, 'target binding'):
                f.inputs()

    def test_pickle_or_wrong_dtype_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path = root/'bad.npy'; np.save(path, np.zeros((1, 78), np.float64))
            with self.assertRaisesRegex(ValueError, 'array contract'):
                bundle.open_features(root, {'path': path.name, 'sha256': launcher.file_sha(path)}, 1)

    def test_readonly_inputs_and_mutation_after_loading(self):
        with tempfile.TemporaryDirectory() as d:
            f = Fixture(Path(d)); inputs = f.inputs()
            p = f.source/'class0.npy'
            with p.open('ab') as handle:
                handle.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                inputs.recheck_files()


class ProfileTests(unittest.TestCase):
    def test_first_model_is_not_retained_when_next_is_allocated(self):
        with tempfile.TemporaryDirectory() as d:
            inputs = Fixture(Path(d)).inputs()
            def generated():
                m = Model(0); ref = weakref.ref(m)
                yield 0, m
                del m
                self.assertIsNone(ref())
                yield 1, Model(1)
            with patch.object(profile_core, 'capture_native', captured):
                profile_core.profile(generated(), inputs, {'max_native_calls': 96, 'native_seconds': 30})

    def test_wrong_wrapper_value_emits_first_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            inputs = Fixture(Path(d)).inputs()
            with patch.object(profile_core, 'capture_native', captured), \
                 patch.object(profile_core.FixedContextTarget, '__call__', return_value=np.array([0.])):
                with self.assertRaises(profile_core.Mismatch) as found:
                    profile_core.profile([(0, Model(0))], inputs, {'max_native_calls': 96, 'native_seconds': 30})
                self.assertEqual(found.exception.diagnostic['check'], 'wrapper-native-margin')

    def test_masked_native_input_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            inputs = Fixture(Path(d)).inputs(); model = Model(0); original = model.score
            def bad(x):
                x[0, 0] += 1
                return original(x)
            model.score = bad
            with patch.object(profile_core, 'capture_native', captured):
                with self.assertRaisesRegex(ValueError, 'changed profile input'):
                    profile_core.profile([(0, model)], inputs, {'max_native_calls': 96, 'native_seconds': 30})

    def test_exact_diagnostic_call_count_and_no_scientific_claim(self):
        with tempfile.TemporaryDirectory() as d:
            inputs = Fixture(Path(d)).inputs(); models = [Model(0), Model(1)]; events = []
            with patch.object(profile_core, 'capture_native', captured):
                r = profile_core.profile(enumerate(models), inputs,
                    {'max_native_calls': 96, 'native_seconds': 30}, on_record=events.append)
            self.assertEqual(r['native_calls'], 84); self.assertEqual(len(events), 24)
            self.assertEqual(models[0].calls, [512]*21+[105]*21)
            self.assertEqual(models[1].calls, models[0].calls)
            self.assertFalse(r['scientific_result_generated']); self.assertFalse(r['shap_attributions_computed'])
            self.assertFalse(r['archived_score_parity_verified'])

    def test_bad_repeat_stops_before_completion(self):
        with tempfile.TemporaryDirectory() as d:
            inputs = Fixture(Path(d)).inputs(); m = Model(0); original = m.score
            def flaky(raw):
                r = original(raw)
                if len(m.calls) == 2:
                    r['head_scores'][0, 0] += np.float32(.01)
                    r['joint_scores'] = r['head_scores'].copy()
                return r
            m.score = flaky
            with self.assertRaises(profile_core.Mismatch) as found:
                profile_core.profile([(0, m)], inputs, {'max_native_calls': 96, 'native_seconds': 30})
            self.assertEqual(found.exception.diagnostic['check'], 'unmasked-repeat')
            self.assertIn('first_index', found.exception.diagnostic)

    def test_budget_exhaustion_not_infinite_retry(self):
        with tempfile.TemporaryDirectory() as d:
            inputs = Fixture(Path(d)).inputs(); m = Model(0)
            with patch.object(profile_core, 'capture_native', captured):
                with self.assertRaisesRegex(ValueError, 'budget'):
                    profile_core.profile([(0, m)], inputs, {'max_native_calls': 3, 'native_seconds': 30})
            self.assertEqual(len(m.calls), 3)

    def test_wrong_checkpoint_order_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'order'):
            profile_core.profile([(1, Model(1))], None, {'max_native_calls': 96, 'native_seconds': 30})

    def test_incomplete_checkpoint_sequence_is_not_success(self):
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            profile_core.profile([], None, {'max_native_calls': 96, 'native_seconds': 30})

    def test_masks_change_only_selected_features(self):
        row = np.arange(78, dtype=np.float32); refs = np.full((32, 78), -1, np.float32)
        cases = dict(profile_core.masks(row, refs))
        np.testing.assert_array_equal(cases['first-feature'][1:], row[1:])
        np.testing.assert_array_equal(cases['last-feature'][:-1], row[:-1])
        np.testing.assert_array_equal(cases['alternating-features'][1::2], row[1::2])
        np.testing.assert_array_equal(cases['unmasked'], row)


class LauncherTests(unittest.TestCase):
    def test_console_helper_name_in_another_directory_does_not_get_exemption(self):
        from types import SimpleNamespace
        child = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=20),
            ppid=lambda: 10, exe=lambda: str(Path.cwd()/'conhost.exe'))
        tracked = SimpleNamespace(pid=10, memory_info=lambda: SimpleNamespace(rss=100),
            children=lambda recursive: [child])
        with patch.object(launcher, 'console_helper_path', return_value=Path('/trusted/system/conhost.exe').resolve()):
            rss, unexpected = launcher.owned_usage(tracked)
        self.assertEqual(rss, 120); self.assertTrue(unexpected)

    def supervised_fixture(self, root, code, seconds=3, rss=8*1024**3):
        out = root/'run'; real_popen = subprocess.Popen
        c = {'output': str(out), 'run_id': 'fixture', 'profile_seed': 20260911,
             'limits': {'process_seconds': seconds, 'rss_bytes': rss}}
        def spawn(*args, **kwargs):
            return real_popen([sys.executable, '-I', '-B', '-c', code], **kwargs)
        with patch.object(launcher, 'gpu_available', return_value={}), \
             patch.object(launcher, 'candidate', return_value=c), \
             patch.object(launcher.subprocess, 'Popen', side_effect=spawn), patch('builtins.print'):
            status = launcher.execute(c, root/'candidate.json', 'a'*64, 'b'*64)
        return status, json.loads((out/'PROFILE_TERMINAL.json').read_text())

    def test_supervisor_timeout_stops_only_its_synthetic_child(self):
        with tempfile.TemporaryDirectory() as d:
            code = 'import time; time.sleep(20)'
            status, terminal = self.supervised_fixture(Path(d), code, seconds=.3)
            self.assertEqual(status, 1); self.assertEqual(terminal['reason'], 'WALL_BUDGET_EXCEEDED')
            self.assertFalse(terminal['scientific_result_generated'])

    def test_supervisor_rss_limit_stops_synthetic_child(self):
        with tempfile.TemporaryDirectory() as d:
            status, terminal = self.supervised_fixture(Path(d), 'import time; time.sleep(20)', rss=1)
            self.assertEqual(status, 1); self.assertEqual(terminal['reason'], 'RSS_BUDGET_EXCEEDED')

    def test_zero_exit_without_complete_artifact_is_failed(self):
        with tempfile.TemporaryDirectory() as d:
            status, terminal = self.supervised_fixture(Path(d), 'pass')
            self.assertEqual(status, 1); self.assertEqual(terminal['status'], 'FAILED')
            self.assertNotIn('PROFILE_RESULT.json', terminal['artifacts'])

    def test_candidate_bad_bytes_block_before_any_dependency_or_tensor_access(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'candidate.json'; launcher.fresh_json(path, {})
            with patch.object(launcher, 'version') as dependencies:
                with self.assertRaisesRegex(ValueError, 'candidate hash'):
                    launcher.candidate(path, 'a'*64)
                dependencies.assert_not_called()

    def test_execute_without_approval_does_not_launch_process(self):
        with patch.object(launcher, 'candidate', return_value={}), \
             patch.object(launcher, 'execute') as execute, \
             patch.object(sys, 'argv', ['local_profile', '--candidate', 'fixture.json', '--sha256', 'a'*64, '--execute']):
            with self.assertRaisesRegex(ValueError, 'receipt required'):
                launcher.main()
            execute.assert_not_called()

    def test_validation_default_does_not_call_worker_or_execute(self):
        with patch.object(launcher, 'candidate', return_value={}), \
             patch.object(launcher, 'execute') as execute, patch.object(launcher, 'worker') as worker, \
             patch.object(sys, 'argv', ['local_profile', '--candidate', 'fixture.json', '--sha256', 'a'*64]), \
             patch('builtins.print'):
            self.assertEqual(launcher.main(), 0)
            execute.assert_not_called(); worker.assert_not_called()

    def test_create_only_receipt_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'result.json'; launcher.fresh_json(path, {'status': 'fixture'})
            with self.assertRaises(FileExistsError):
                launcher.fresh_json(path, {'status': 'replacement'})

    def test_approval_rejects_missing_or_unbound_verdict(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'review.json'
            value = {'verdict': 'BLOCKED', 'scope': 'local-component-profile-only',
                'candidate_sha256': 'a'*64, 'run_id': 'fixture', 'output': 'fixture', 'reviewer_reference': 'test'}
            launcher.fresh_json(path, value)
            with self.assertRaisesRegex(ValueError, 'independent review'):
                launcher.approval(path, launcher.file_sha(path), 'a'*64, {'run_id': 'fixture', 'output': 'fixture'})

    def test_wrong_approval_bytes_are_rejected_first(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'review.json'; launcher.fresh_json(path, {})
            with self.assertRaisesRegex(ValueError, 'receipt hash'):
                launcher.approval(path, '0'*64, 'a'*64, {})

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, 'relative'):
                launcher.regular(d, '../elsewhere')

    def test_gpu_busy_is_a_stop_without_output_creation(self):
        c = {'limits': {'required_gpu_free_mib': 4096, 'max_start_gpu_utilization_percent': 10}}
        from types import SimpleNamespace
        with patch.object(launcher.subprocess, 'run', return_value=SimpleNamespace(stdout='5000, 68\n')):
            with self.assertRaisesRegex(ValueError, 'GPU occupied'):
                launcher.gpu_available(c)

    def test_gpu_free_memory_is_checked_even_when_idle(self):
        c = {'limits': {'required_gpu_free_mib': 4096, 'max_start_gpu_utilization_percent': 10}}
        from types import SimpleNamespace
        with patch.object(launcher.subprocess, 'run', return_value=SimpleNamespace(stdout='1000, 0\n')):
            with self.assertRaisesRegex(ValueError, 'insufficient'):
                launcher.gpu_available(c)

    def test_worker_has_to_be_spawned_by_claiming_parent(self):
        with tempfile.TemporaryDirectory() as d:
            launcher.fresh_json(Path(d)/'CLAIM.json', {'candidate_sha256': 'a'*64, 'parent_pid': -1})
            with self.assertRaisesRegex(ValueError, 'supervised'):
                launcher.worker({'output': d}, 'a'*64)
            self.assertFalse((Path(d)/'WORKER_SPENT.json').exists())


if __name__ == '__main__':
    unittest.main()
