"""Allocation-only, hash-bound reconstruction of the historical frozen probes.

This verifier does not submit jobs, train, calibrate, or change the deployed scorer.
An operational launcher with reviewed resource and publication controls is required.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import socket
import sys
import time

SEEDS = [1, 2, 3, 4, 42]
REGISTRY_SHA = 'de43ff1d51f2993741d5a18807f129f1f80db8a97360c0289312d1944a63c58c'
SETTINGS = {
    'deterministic_algorithms': True, 'cudnn_deterministic': True,
    'cudnn_benchmark': False, 'matmul_allow_tf32': False,
    'cudnn_allow_tf32': False, 'float32_matmul_precision': 'highest',
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'torch_num_threads': 2,
    'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def safe_path(root, relative):
    p = PurePosixPath(relative)
    if not relative or p.is_absolute() or '..' in p.parts or '\\' in relative or ':' in relative:
        raise ValueError('Unsafe bound relative path')
    base = Path(root).resolve()
    result = (base / relative).resolve()
    if not result.is_relative_to(base):
        raise ValueError('Input escapes its declared root')
    return result


def require_allocation(env, hostname, actual_user):
    if actual_user != 'leon12138' or hostname.split('.')[0].lower().startswith('login'):
        raise ValueError('Not the authorized compute-node context')
    required = {'SLURM_JOB_NUM_NODES': '1', 'SLURM_NTASKS': '1', 'SLURM_CPUS_PER_TASK': '2'}
    if any(env.get(k) != v for k, v in required.items()):
        raise ValueError('Expected one node, one task and two allocated CPUs')
    if not re.fullmatch(r'[0-9]+', env.get('SLURM_JOB_ID', '')):
        raise ValueError('Missing Slurm job identity')
    if not re.fullmatch(r'[0-9]+', env.get('SLURM_STEP_ID', '')):
        raise ValueError('Run through a scheduled srun step, not a login shell')
    visible = env.get('CUDA_VISIBLE_DEVICES', '')
    if not visible or ',' in visible or visible in ('-1', 'NoDevFiles'):
        raise ValueError('Exactly one allocated visible GPU is required')
    return {k: env[k] for k in (*required, 'SLURM_JOB_ID', 'SLURM_STEP_ID', 'CUDA_VISIBLE_DEVICES')}


def environment_comparison(actual, historical):
    # These historical fields were measured. Settings are additionally recovered
    # from the hash-bound original _seed_process, not inferred library defaults.
    expected = {k: historical[k] for k in (
        'python', 'platform', 'numpy', 'torch', 'cuda_version', 'gpu_name', 'cudnn_version')}
    expected.update(SETTINGS)
    checks = {k: {'expected': v, 'actual': actual.get(k), 'passed': actual.get(k) == v}
              for k, v in expected.items()}
    return {
        'recorded_environment_matched': all(v['passed'] for v in checks.values()),
        'checks': checks,
        'historical_environment_identical_proven': False,
        'unrecorded_historical_fields': [
            'NVIDIA driver build', 'NumPy BLAS library/build and instruction dispatch',
            'loaded binary hashes', 'per-operation selected numerical kernels',
        ],
    }


def required_inputs():
    runtime = {('runtime', 'streaming_full/' + n + '.py') for n in (
        '__init__', '__main__', 'capacity_profile', 'data', 'exposure_preflight',
        'models', 'monitoring', 'recovery', 'routers', 'runner', 'smoke_test',
        'summarize', 'validation', 'wandb_tracking')}
    runtime.update({('runtime', 'ofra_encoders/' + n + '.py') for n in ('__init__', 'ft_transformer')})
    protected = {('protected', f'last_epoch/seed_{seed}/protocol.json') for seed in SEEDS}
    protected.add(('protected', 'last_epoch/seed_1/monitoring_seed_1/probe_manifest.json'))
    for seed in SEEDS:
        for cp in range(4):
            prefix = f'last_epoch/seed_{seed}/monitoring_seed_{seed}/checkpoint_{cp:03d}/'
            protected.update({('protected', prefix + n) for n in (
                'checkpoint_manifest.json', 'inference_state.npz', 'probe_scores.npz')})
    data = {('data', f'class_{c:02d}/test_000.npy') for c in range(8)}
    data.add(('data', 'streaming_manifest.json'))
    return runtime | protected | data


def validate_plan(plan):
    if plan.get('schema') != 'replayids-gpu-fidelity-inputs-v1':
        raise ValueError('Unexpected binding schema')
    if plan.get('seeds') != SEEDS or plan.get('checkpoints') != [0, 1, 2, 3]:
        raise ValueError('Coverage cannot be narrowed')
    if plan.get('atol') != 1e-4 or plan.get('rtol') != 1e-5 or plan.get('encoder_batch_size') != 512:
        raise ValueError('Registered tolerance or batching changed')
    if plan.get('registry_sha256') != REGISTRY_SHA:
        raise ValueError('Unrecognized protected registry')
    if plan.get('scorer') != 'official/joint_cap3000':
        raise ValueError('This is not a new scoring-rule experiment')
    if len(plan.get('files', [])) != len({(r['root'], r['path']) for r in plan.get('files', [])}):
        raise ValueError('Duplicate binding')
    if {(r['root'], r['path']) for r in plan['files']} != required_inputs():
        raise ValueError('Exact 91-file coverage is required before imports/inference')
    for r in plan['files']:
        if r['root'] not in ('runtime', 'protected', 'data') or not re.fullmatch('[0-9a-f]{64}', r['sha256']):
            raise ValueError('Invalid input binding')
        safe_path(Path.cwd(), r['path'])
    return plan


def verify_inputs(plan, roots):
    index = {}
    for r in plan['files']:
        path = safe_path(roots[r['root']], r['path'])
        if digest(path) != r['sha256']:
            raise ValueError('Bound input mismatch: ' + r['root'] + '/' + r['path'])
        index[r['root'], r['path']] = path
    return index


def write_report(directory, report):
    target = directory / 'GPU_FIDELITY.json'
    tmp = directory / 'GPU_FIDELITY.json.tmp'
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    os.replace(tmp, target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('binding', 'runtime', 'protected', 'data', 'output', 'canonical_helper'):
        p.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    p.add_argument('--binding-sha256', required=True)
    a = p.parse_args()
    # No scientific imports, input hashing or output creation on a login node.
    if platform.system() != 'Linux':
        raise ValueError('This entry point requires an authorized Linux allocation')
    import pwd
    allocation = require_allocation(os.environ, socket.gethostname(), pwd.getpwuid(os.getuid()).pw_name)
    if digest(a.binding) != a.binding_sha256:
        raise ValueError('Binding digest mismatch')
    plan = validate_plan(load(a.binding))
    self_hash = hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    helper_hash = hashlib.sha256(a.canonical_helper.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    if self_hash != plan['verifier_lf_sha256'] or helper_hash != plan['canonical_helper_lf_sha256']:
        raise ValueError('Executable digest mismatch')
    output = a.output.resolve()
    scratch = Path('/scr/user') / pwd.getpwuid(os.getuid()).pw_name
    if not output.is_relative_to(scratch.resolve()):
        raise ValueError('Active output must be in the authorized scratch tree')
    output.mkdir(parents=False, exist_ok=False)
    report = {'schema': 'replayids-gpu-fidelity-v1', 'binding_sha256': a.binding_sha256,
              'source_lf_sha256': self_hash, 'allocation': allocation, 'checkpoints': [],
              'coverage_complete': False, 'all_passed': False, 'status': 'VERIFYING_INPUTS',
              'calibrator_fitted': False, 'new_test_performance_computed': False,
              'historical_environment_identical_proven': False}
    started = time.monotonic()
    write_report(output, report)
    try:
        roots = {'runtime': a.runtime, 'protected': a.protected, 'data': a.data}
        index = verify_inputs(plan, roots)
        protocol = load(index['protected', 'last_epoch/seed_1/protocol.json'])
        # Import only a clean copy of the original source files, never historical
        # bytecode or additional mutable source present beside the original tree.
        runtime = output / 'verified_runtime'
        for r in protocol['source']['files']:
            src = index['runtime', r['path']]
            if digest(src) != r['sha256']:
                raise ValueError('Original protocol source mismatch')
            dst = safe_path(runtime, r['path'])
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            if digest(dst) != r['sha256']:
                raise ValueError('Staged runtime mismatch')
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(runtime))
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            os.environ[name] = '2'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        import numpy as np
        import torch
        torch.set_num_threads(2)
        # Invoke the exact hash-bound historical setup, including all TF32 flags.
        from streaming_full.validation import _seed_process
        from streaming_full.monitoring import load_checkpoint
        from streaming_full.data import array_sha256, canonical_sha256
        from ofra_encoders import verify_ft_transformer_dependency
        _seed_process(1, True)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError('One CUDA device is required; CPU fallback is forbidden')
        actual = {
            'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__,
            'torch': torch.__version__, 'cuda_version': torch.version.cuda,
            'gpu_name': torch.cuda.get_device_name(0), 'cudnn_version': torch.backends.cudnn.version(),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'torch_num_threads': torch.get_num_threads(),
            **{k: os.environ[k] for k in ('CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
        }
        report['environment'] = actual
        report['environment_comparison'] = environment_comparison(actual, protocol['environment'])
        report['upstream'] = verify_ft_transformer_dependency()
        if not report['environment_comparison']['recorded_environment_matched']:
            report['status'] = 'ENVIRONMENT_MISMATCH'
            write_report(output, report)
            return 3
        spec = importlib.util.spec_from_file_location('bound_canonical', a.canonical_helper)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        probe_path = index['protected', 'last_epoch/seed_1/monitoring_seed_1/probe_manifest.json']
        probe = load(probe_path)
        if canonical_sha256({k: v for k, v in probe.items() if k != 'canonical_sha256'}) != probe['canonical_sha256']:
            raise ValueError('Probe canonical mismatch')
        mf = index['data', 'streaming_manifest.json']
        if digest(mf) != probe['streaming_manifest_sha256']:
            raise ValueError('Probe/data mismatch')
        shards = {}
        for c in load(mf)['classes']:
            for j, r in enumerate(c['test']):
                path = index['data', r['path']]
                if digest(path) != r['sha256']:
                    raise ValueError('Data manifest mismatch')
                shards[c['id'], j] = np.load(path, mmap_mode='r', allow_pickle=False)
        records = probe['official_test']['samples']
        rows = []
        for r in records:
            x = np.asarray(shards[r['class_id'], r['shard_ordinal']][r['local_row']:r['local_row']+1], dtype=np.float32)
            if array_sha256(x) != r['feature_row_sha256']:
                raise ValueError('Probe feature mismatch')
            rows.append(x)
        raw = np.vstack(rows)
        torch.cuda.reset_peak_memory_stats()
        for seed in SEEDS:
            conf = load(index['protected', f'last_epoch/seed_{seed}/protocol.json'])
            if conf['source'] != protocol['source'] or conf['environment'] != protocol['environment']:
                raise ValueError('Inconsistent seed source or environment')
            if conf['config']['eval_batch_size'] != 512 or conf['config']['family_checkpoint_selection'] != 'last':
                raise ValueError('Original batching or checkpoint policy changed')
            for cp in range(4):
                prefix = f'last_epoch/seed_{seed}/monitoring_seed_{seed}/checkpoint_{cp:03d}/'
                meta = index['protected', prefix + 'checkpoint_manifest.json']
                metadata = load(meta)
                if metadata['inference_state_file'] != 'inference_state.npz':
                    raise ValueError('Unexpected state file reference')
                # Refuse indirect unbound state accesses before model construction.
                state = index['protected', prefix + 'inference_state.npz']
                if state != safe_path(meta.parent, metadata['inference_state_file']):
                    raise ValueError('State path mismatch')
                _seed_process(seed, True)
                model = load_checkpoint(meta, device='cuda:0')
                seen = sorted(c for task in conf['tasks'][:cp+1] for c in task)
                if metadata['seed'] != seed or metadata['checkpoint'] != cp or metadata['seen_classes'] != seen:
                    raise ValueError('Checkpoint identity mismatch')
                if metadata['probe_manifest_file_sha256'] != digest(probe_path):
                    raise ValueError('Checkpoint probe mismatch')
                selected = [i for i, r in enumerate(records) if r['class_id'] in seen]
                ids = np.array([records[i]['sample_id_sha256'].encode('ascii') for i in selected], dtype='S64')
                with np.load(index['protected', prefix + 'probe_scores.npz'], allow_pickle=False) as saved:
                    if not np.array_equal(ids, saved['sample_id_sha256']):
                        raise ValueError('Frozen probe order mismatch')
                    torch.cuda.synchronize()
                    tick = time.monotonic()
                    computed = helper.canonical_score(model, raw[selected], 512)
                    torch.cuda.synchronize()
                    elapsed = time.monotonic() - tick
                    errors, changed = helper.compare(computed, saved, plan['atol'], plan['rtol'])
                    mismatches = []
                    for i in changed:
                        old = np.sort(saved['joint_scores'][i]); new = np.sort(computed['joint_scores'][i])
                        mismatches.append({'sample_id_sha256': ids[i].decode(),
                            'saved_class': int(saved['predicted_class_id'][i]),
                            'reconstructed_class': int(computed['predicted_class_id'][i]),
                            'saved_top2_margin': float(old[-1]-old[-2]),
                            'reconstructed_top2_margin': float(new[-1]-new[-2])})
                    result = {'seed': seed, 'checkpoint': cp, 'probe_rows': len(selected),
                              'scores': errors, 'mismatches': mismatches, 'elapsed_seconds': elapsed,
                              'prediction_mismatch_count': len(changed),
                              'score_sha256': {k: array_sha256(computed[k]) for k in ('head_scores', 'router_z_scores', 'joint_scores')},
                              'saved_prediction_sha256': array_sha256(saved['predicted_class_id']),
                              'reconstructed_prediction_sha256': array_sha256(computed['predicted_class_id']),
                              'passed': not len(changed) and not any(v['outside_tolerance_cells'] for v in errors.values())}
                del model
                report['checkpoints'].append(result)
                report['status'] = 'RUNNING'
                report['elapsed_seconds'] = time.monotonic() - started
                report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
                report['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved()
                write_report(output, report)
                print(json.dumps({k: result[k] for k in ('seed', 'checkpoint', 'passed', 'elapsed_seconds', 'prediction_mismatch_count')}), flush=True)
        report['coverage_complete'] = len(report['checkpoints']) == 20
        report['all_passed'] = report['coverage_complete'] and all(r['passed'] for r in report['checkpoints'])
        report['status'] = 'PASS' if report['all_passed'] else 'FIDELITY_FAIL'
        write_report(output, report)
        return 0 if report['all_passed'] else 2
    except Exception as exc:
        report['status'] = 'ERROR'
        report['error_type'] = type(exc).__name__
        write_report(output, report)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
