"""Fail-closed input and environment contract tests; no remote execution."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
MODULE = (HERE if (HERE / 'check_replayids_gpu_fidelity.py').exists() else HERE.parent / 'tools') / 'check_replayids_gpu_fidelity.py'
spec = importlib.util.spec_from_file_location('gpu_fidelity', MODULE)
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)


class Tests(unittest.TestCase):
    def env(self):
        return {'SLURM_JOB_ID': '123', 'SLURM_STEP_ID': '0', 'SLURM_JOB_NUM_NODES': '1',
                'SLURM_NTASKS': '1', 'SLURM_CPUS_PER_TASK': '2', 'CUDA_VISIBLE_DEVICES': '0'}

    def historical(self):
        return {'python': '3.11.9', 'platform': 'Linux', 'numpy': '2.2.6', 'torch': '2.6.0+cu118',
                'cuda_version': '11.8', 'gpu_name': 'NVIDIA A100-SXM4-80GB', 'cudnn_version': 90100}

    def plan(self):
        return {'schema': 'replayids-gpu-fidelity-inputs-v1', 'seeds': [1,2,3,4,42],
                'checkpoints': [0,1,2,3], 'atol': 1e-4, 'rtol': 1e-5, 'encoder_batch_size': 512,
                'registry_sha256': gpu.REGISTRY_SHA, 'scorer': 'official/joint_cap3000',
                'files': [{'root': root, 'path': path, 'sha256': 'a'*64}
                          for root, path in sorted(gpu.required_inputs())]}

    def test_accepts_single_scheduled_step(self):
        r = gpu.require_allocation(self.env(), 'gpu06', 'leon12138')
        self.assertEqual(r['SLURM_JOB_ID'], '123')

    def test_rejects_login_even_with_slurm_variables(self):
        with self.assertRaises(ValueError): gpu.require_allocation(self.env(), 'login01.example', 'leon12138')

    def test_rejects_other_user(self):
        with self.assertRaises(ValueError): gpu.require_allocation(self.env(), 'gpu06', 'another-user')

    def test_rejects_missing_or_excess_allocation(self):
        for k, v in [('SLURM_JOB_ID', ''), ('SLURM_STEP_ID', 'batch'), ('SLURM_JOB_NUM_NODES', '2'),
                     ('SLURM_NTASKS', '2'), ('SLURM_CPUS_PER_TASK', '8'), ('CUDA_VISIBLE_DEVICES', '0,1'),
                     ('CUDA_VISIBLE_DEVICES', ''), ('CUDA_VISIBLE_DEVICES', '-1')]:
            with self.subTest(k=k, v=v), self.assertRaises(ValueError):
                env = self.env(); env[k] = v
                gpu.require_allocation(env, 'gpu06', 'leon12138')

    def test_recorded_match_is_not_complete_historical_identity(self):
        h = self.historical(); r = gpu.environment_comparison({**h, **gpu.SETTINGS}, h)
        self.assertTrue(r['recorded_environment_matched'])
        self.assertFalse(r['historical_environment_identical_proven'])
        self.assertEqual(len(r['unrecorded_historical_fields']), 4)

    def test_same_gpu_does_not_hide_software_or_settings_drift(self):
        h = self.historical()
        for field, value in [('torch', '2.11.0'), ('numpy', '2.4.6'), ('matmul_allow_tf32', True),
                             ('cudnn_allow_tf32', True), ('cudnn_benchmark', True),
                             ('float32_matmul_precision', 'high'), ('CUBLAS_WORKSPACE_CONFIG', ':16:8')]:
            actual = {**h, **gpu.SETTINGS, field: value}
            with self.subTest(field=field):
                self.assertFalse(gpu.environment_comparison(actual, h)['recorded_environment_matched'])

    def test_missing_settings_fail_closed(self):
        h = self.historical()
        self.assertFalse(gpu.environment_comparison(h, h)['recorded_environment_matched'])

    def test_relative_path_guard(self):
        with tempfile.TemporaryDirectory() as d:
            for path in ('../x', '/x', 'C:/x', 'x\\y', ''):
                with self.subTest(path=path), self.assertRaises(ValueError): gpu.safe_path(d, path)
            self.assertEqual(gpu.safe_path(d, 'a/b'), Path(d).resolve()/'a/b')

    def test_symlink_escape(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as other:
            try: (Path(d)/'link').symlink_to(other, target_is_directory=True)
            except OSError: self.skipTest('Symlink creation unavailable on this host')
            with self.assertRaises(ValueError): gpu.safe_path(d, 'link/x')

    def test_gate_cannot_change_batch_tolerance_or_coverage(self):
        for key, value in [('seeds', [42]), ('checkpoints', [3]), ('atol', 1e-2), ('rtol', .1),
                           ('encoder_batch_size', 898), ('scorer', 'direct64'), ('registry_sha256', 'b'*64)]:
            plan = self.plan(); plan[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): gpu.validate_plan(plan)

    def test_duplicate_or_bad_bindings_rejected(self):
        plan = self.plan(); plan['files'].append(copy.deepcopy(plan['files'][0]))
        with self.assertRaises(ValueError): gpu.validate_plan(plan)
        for field, value in [('root', 'other'), ('path', '../x'), ('sha256', 'abcd')]:
            plan = self.plan(); plan['files'][0][field] = value
            with self.assertRaises(ValueError): gpu.validate_plan(plan)

    def test_byte_hash_not_filename_acceptance(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d)/'x.py'; f.write_bytes(b'exact')
            plan = {'files': [{'root': 'runtime', 'path': 'x.py', 'sha256': hashlib.sha256(b'exact').hexdigest()}]}
            gpu.verify_inputs(plan, {'runtime': Path(d)})
            f.write_bytes(b'changed')
            with self.assertRaises(ValueError): gpu.verify_inputs(plan, {'runtime': Path(d)})

    def test_missing_checkpoint_or_extra_input_rejected(self):
        self.assertEqual(len(gpu.required_inputs()), 91)
        gpu.validate_plan(self.plan())
        plan = self.plan(); plan['files'].pop()
        with self.assertRaises(ValueError): gpu.validate_plan(plan)
        plan = self.plan(); plan['files'].append({'root': 'runtime', 'path': 'extra.py', 'sha256': 'a'*64})
        with self.assertRaises(ValueError): gpu.validate_plan(plan)

    def test_partial_failure_report_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            r = {'status': 'ERROR', 'coverage_complete': False, 'all_passed': False, 'checkpoints': []}
            gpu.write_report(Path(d), r)
            self.assertEqual(json.loads((Path(d)/'GPU_FIDELITY.json').read_text()), r)
            self.assertFalse((Path(d)/'GPU_FIDELITY.json.tmp').exists())


if __name__ == '__main__': unittest.main()
