"""Local-only, hash-bound one-shot component profile. Default is validation only.

The review receipt is an external authorization record, not a digital signature.
Never generate APPROVED here; operator must obtain independent review first.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from importlib.metadata import version


def check(ok, message):
    if not ok:
        raise ValueError(message)


def file_sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def regular(root, relative):
    root = Path(root).absolute(); rel = Path(relative)
    check(not rel.is_absolute() and not rel.drive and '..' not in rel.parts and
          all(':' not in x for x in rel.parts), 'relative binding required')
    path = root / rel
    for p in (path, *path.parents):
        check(not p.is_symlink() and (not p.exists() or
              not getattr(p.lstat(), 'st_file_attributes', 0) & 0x400), 'linked binding forbidden')
    check(path.is_file() and path.resolve().is_relative_to(root.resolve()), 'missing contained file')
    return path


def fresh_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def read_json(path):
    check(Path(path).is_file() and Path(path).stat().st_size <= 2 * 1024**2, 'bounded JSON required')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def candidate(path, expected):
    check(len(expected) == 64 and file_sha(path) == expected, 'candidate hash mismatch')
    c = read_json(path)
    check(c['schema'] == 'etg-local-component-profile-v1' and
          c['scope'] == 'LOCAL_COMPONENT_PROFILE_NOT_ETG_STUDY', 'candidate scope')
    check(c['scientific_result_authorized'] is False and c['hpc_authorized'] is False, 'scope escalation')
    check(socket.gethostname() == c['host'] and not socket.gethostname().lower().startswith('login')
          and not os.environ.get('SLURM_JOB_ID'), 'local host only; not an HPC launcher')
    check(platform.python_version() == c['python_version'] and
          Path(sys.executable).resolve() == Path(c['python_executable']).resolve(), 'interpreter changed')
    for name, expected_version in c['packages'].items():
        check(version(name) == expected_version, 'dependency version changed: ' + name)
    check(c['limits'] == {'max_native_calls': 96, 'native_seconds': 120,
        'process_seconds': 180, 'rss_bytes': 8 * 1024**3, 'gpu_allocator_bytes': 3 * 1024**3,
        'required_gpu_free_mib': 4096, 'max_start_gpu_utilization_percent': 10, 'cpu_threads': 2},
        'profile limits changed; require a new protocol')
    check(c['device'] == 'cuda:0' and c['checkpoints'] == [0, 1], 'fixed diagnostic scope')
    keys = [(b['root'], b['path']) for b in c['bindings']]
    check(len(keys) == len(set(keys)) and c['bindings'], 'duplicate/empty bindings')
    required = [('repository', 'experimental/etg_native_profile_v1/' + n)
                for n in ('local_profile.py', 'bundle.py', 'profile_core.py')]
    required += [('repository', 'experimental/etg_exploratory_v1/' + n)
                 for n in ('native_loader.py', 'native_target.py', 'pilot_core.py', 'prepare_plan.py',
                           'LINEAGE_METADATA.json', 'PREPARED_INPUTS.json', 'policy.json')]
    required += [('repository', 'experimental/etg_input_audit/audit_inputs.py')]
    required += [(b['root'], b['path']) for b in c['metadata'].values()]
    required += [('native', f'seed_1/checkpoint_{cp:03d}/{n}') for cp in (0, 1)
                 for n in ('checkpoint_manifest.json', 'inference_state.npz', 'probe_scores.npz')]
    check(set(required) <= set(keys), 'incomplete import/input closure')
    check(c['profile_seed'] == 20260911, 'fixed diagnostic seed changed')
    for b in c['bindings']:
        f = regular(c['roots'][b['root']], b['path'])
        check(f.stat().st_size == b['bytes'] and file_sha(f) == b['sha256'], 'bound input changed: ' + b['path'])
    script = Path(__file__).resolve()
    check(script == regular(c['roots']['repository'], c['launcher']).resolve(), 'wrong launcher location')
    check(('repository', c['launcher']) in keys, 'launcher missing from binding closure')
    check(c['output'] and Path(c['output']).is_absolute(), 'absolute output required')
    parent = Path(c['output']).parent
    check(parent.is_dir() and Path(c['output']).name == 'run-' + c['run_id'], 'output binding')
    for p in (parent, *parent.parents):
        check(not p.is_symlink() and not getattr(p.lstat(), 'st_file_attributes', 0) & 0x400, 'linked output parent')
    return c


def approval(path, expected, candidate_sha, c):
    check(file_sha(path) == expected, 'review receipt hash mismatch')
    a = read_json(path)
    check(a.get('verdict') == 'APPROVED' and a.get('scope') == 'local-component-profile-only'
          and a.get('candidate_sha256') == candidate_sha and a.get('run_id') == c['run_id']
          and a.get('output') == c['output'] and a.get('reviewer_reference'), 'missing exact independent review')
    return a


def gpu_available(c):
    r = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=10)
    lines = r.stdout.strip().splitlines(); check(len(lines) == 1, 'one GPU record required')
    free, utilization = [int(x.strip()) for x in lines[0].split(',')]
    check(free >= c['limits']['required_gpu_free_mib'] and
          utilization <= c['limits']['max_start_gpu_utilization_percent'],
          'GPU occupied or insufficient free memory; no process started')
    return {'free_mib': free, 'utilization_percent': utilization}


def console_helper_path():
    if os.name != 'nt':
        return None
    import ctypes
    buffer = ctypes.create_unicode_buffer(32768)
    length = ctypes.windll.kernel32.GetSystemDirectoryW(buffer, len(buffer))
    check(0 < length < len(buffer), 'cannot locate Windows system directory')
    return (Path(buffer.value) / 'conhost.exe').resolve()


def owned_usage(tracked):
    """Include a verified direct Windows console helper, reject worker fan-out."""
    import psutil
    rss = tracked.memory_info().rss
    allowed = console_helper_path()
    for child in tracked.children(recursive=True):
        try:
            rss += child.memory_info().rss
            if allowed is None or child.ppid() != tracked.pid or Path(child.exe()).resolve() != allowed:
                return rss, True
        except psutil.NoSuchProcess:
            continue
    return rss, False


def configure_paths(c):
    repo = Path(c['roots']['repository'])
    sys.path[:0] = [str(repo / 'experimental/etg_native_profile_v1'),
                    str(repo / 'experimental/etg_exploratory_v1')]


def worker(c, candidate_sha):
    out = Path(c['output']); claim = read_json(out / 'CLAIM.json')
    check(claim['candidate_sha256'] == candidate_sha and claim['parent_pid'] == os.getppid(),
          'worker must be supervised by the claiming parent')
    fresh_json(out / 'WORKER_SPENT.json', {'pid': os.getpid(), 'candidate_sha256': candidate_sha})
    configure_paths(c)
    import gc
    import numpy as np
    import torch
    from bundle import identities, ProfileInputs, selected_targets
    from native_loader import load_native
    from native_target import array_sha
    from profile_core import profile, Mismatch
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(c['profile_seed']); np.random.seed(c['profile_seed'])
    check(torch.cuda.is_available(), 'CUDA unavailable; no silent CPU fallback')
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., c['limits']['gpu_allocator_bytes']/total), 0)
    torch.cuda.reset_peak_memory_stats(0)
    def meta(key):
        b = c['metadata'][key]
        return read_json(regular(c['roots'][b['root']], b['path']))
    mapping = identities(meta('policy'), meta('prepared'), meta('audit'), meta('source'))
    inputs = ProfileInputs(mapping, c['roots']['source'], c['roots']['native'], c['targets'])
    def models():
        for cp in c['checkpoints']:
            cp_root = Path(c['roots']['native']) / 'seed_1' / f'checkpoint_{cp:03d}'
            model = load_native(c['roots']['historical_runtime'], cp_root, cp, 'cuda')
            yield cp, model
            del model
            gc.collect(); torch.cuda.empty_cache()
    journal = out / 'progress.jsonl'
    with journal.open('x', encoding='utf-8') as f:
        def record(value):
            f.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n'); f.flush(); os.fsync(f.fileno())
        try:
            result = profile(models(), inputs, c['limits'], torch.cuda.synchronize, record)
        except Mismatch as error:
            fresh_json(out / 'FIRST_MISMATCH.json', error.diagnostic)
            raise
    inputs.recheck_files()
    result['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(0)
    result['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved(0)
    result['candidate_sha256'] = candidate_sha
    result['references_sha256'] = array_sha(inputs.references)
    result['current_environment'] = {'packages': c['packages'], 'gpu': torch.cuda.get_device_name(0)}
    result['historical_environment_match'] = False
    result['profile_seed'] = c['profile_seed']
    fresh_json(out / 'PROFILE_RESULT.json', result)


def execute(c, candidate_path, candidate_sha, receipt_sha):
    import psutil
    gpu = gpu_available(c)  # Before spending the output or loading CUDA.
    out = Path(c['output']); out.mkdir()  # Existing or partial runs never reused.
    fresh_json(out / 'CLAIM.json', {'candidate_sha256': candidate_sha, 'review_sha256': receipt_sha,
        'run_id': c['run_id'], 'parent_pid': os.getpid(), 'start_gpu': gpu})
    env = os.environ.copy()
    for name in ('PYTHONPATH', 'PYTHONSTARTUP', 'PYTHONHOME'):
        env.pop(name, None)
    env.update(PYTHONHASHSEED=str(c['profile_seed']), CUBLAS_WORKSPACE_CONFIG=':4096:8',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        WANDB_MODE='disabled', CUDA_VISIBLE_DEVICES='0')
    command = [sys.executable, '-I', '-B', str(Path(__file__).resolve()), '--candidate',
               str(Path(candidate_path).resolve()), '--sha256', candidate_sha, '--worker']
    process = None; peak = 0; start = time.monotonic(); reason = None
    def stop_owned_tree():
        if process is None or process.poll() is not None:
            return
        try:
            descendants = psutil.Process(process.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            descendants = []
        for child in reversed(descendants):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        process.kill(); process.wait(timeout=10)
        psutil.wait_procs(descendants, timeout=5)
    try:
        with (out / 'stdout.txt').open('xb') as stdout, (out / 'stderr.txt').open('xb') as stderr:
            process = subprocess.Popen(command, cwd=out, env=env, stdout=stdout, stderr=stderr,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            tracked = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    rss, unexpected_child = owned_usage(tracked); peak = max(peak, rss)
                except psutil.NoSuchProcess:
                    break
                if time.monotonic()-start > c['limits']['process_seconds']:
                    reason = 'WALL_BUDGET_EXCEEDED'; break
                if rss > c['limits']['rss_bytes']:
                    reason = 'RSS_BUDGET_EXCEEDED'; break
                if unexpected_child:
                    reason = 'UNEXPECTED_CHILD_PROCESS'; break
                time.sleep(.2)
            if reason:
                stop_owned_tree()
            process.wait(timeout=10)
            if process.returncode != 0 and reason is None:
                reason = 'WORKER_FAILED'
        result_path = out / 'PROFILE_RESULT.json'
        if reason is None:
            result = read_json(result_path)
            check(result['status'] == 'COMPONENT_PROFILE_COMPLETE' and result['native_calls'] == 84
                  and len(result['records']) == 24 and result['candidate_sha256'] == candidate_sha,
                  'incomplete worker result')
            check(all(result[k] is False for k in ('archived_score_parity_verified',
                'full_population_parity_verified', 'shap_attributions_computed',
                'etg_actions_fitted', 'efficacy_metrics_computed', 'scientific_result_generated')),
                'unsupported worker claim')
            candidate(candidate_path, candidate_sha)  # Recheck bound bytes after execution.
    except BaseException as error:
        reason = reason or type(error).__name__
        if process is not None and process.poll() is None:
            stop_owned_tree()
    artifacts = {p.name: file_sha(p) for p in out.iterdir() if p.is_file()}
    terminal = {'status': 'COMPONENT_PROFILE_COMPLETE' if reason is None else 'FAILED',
        'reason': reason, 'candidate_sha256': candidate_sha, 'peak_polled_rss_bytes': peak,
        'rss_includes_owned_console_helper': True,
        'elapsed_process_window_seconds': time.monotonic()-start, 'artifacts': artifacts,
        'scientific_result_generated': False, 'hpc_accessed': False, 'wandb_written': False,
        'profile_success_does_not_authorize_full_extraction': True}
    fresh_json(out / 'PROFILE_TERMINAL.json', terminal)
    print(json.dumps(terminal, sort_keys=True))
    return 0 if reason is None else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True); p.add_argument('--sha256', required=True)
    p.add_argument('--execute', action='store_true'); p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--approval', type=Path); p.add_argument('--approval-sha256')
    args = p.parse_args()
    check(not (args.execute and args.worker), 'exclusive mode')
    c = candidate(args.candidate, args.sha256)
    if args.worker:
        worker(c, args.sha256); return 0
    if not args.execute:
        print(json.dumps({'status': 'BINDINGS_VALIDATED_NO_EXECUTION', 'candidate_sha256': args.sha256,
            'real_data_loaded': False, 'review_approved': False, 'gpu_availability_checked': False}))
        return 0
    check(args.approval and args.approval_sha256, 'independent review receipt required')
    approval(args.approval, args.approval_sha256, args.sha256, c)
    return execute(c, args.candidate, args.sha256, args.approval_sha256)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({'status': 'BLOCKED', 'error_type': type(error).__name__, 'reason': str(error)[:300]}))
        raise SystemExit(2)
