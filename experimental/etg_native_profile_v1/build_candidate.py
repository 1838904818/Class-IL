"""Read-only local candidate builder. Emits private machine paths to stdout.

Hashes files and reconstructs metadata indices, but never loads feature/model
tensors or creates experiment outputs. Archive stdout privately, not publicly.
"""
import argparse
import hashlib
from importlib.metadata import distribution, version
import json
from pathlib import Path
import platform
import socket
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'etg_exploratory_v1')]
from bundle import identities, selected_targets
from native_loader import safe_file, sha
from pilot_core import require


def build(workspace, output_parent):
    ws = Path(workspace).resolve(); repository = HERE.parents[1]
    native = ws / 'work/replayids_score_readiness_20260907'
    source = ws / 'work/replayids_ofra_seed42_build/ofra_streaming_cache_v2'
    historical = ws / 'work/hpc_transfer_20260730/upload_ready/20260901-replayids-d2-checkpoint-selection-paired5-v3/runtime'
    prefix = 'experimental/etg_exploratory_v1/'
    read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
    policy, prepared, lineage = [read(repository / (prefix + name)) for name in
                                ('policy.json', 'PREPARED_INPUTS.json', 'LINEAGE_METADATA.json')]
    require(sha(repository / (prefix+'policy.json')) == prepared['policy_sha256'], 'policy binding')
    require(sha(native/'sampling_audit.json') == policy['sampling_audit_sha256'] and
            sha(native/'streaming_manifest.json') == policy['native_manifest_sha256'] and
            sha(source/'streaming_manifest.json') == policy['source_manifest_sha256'], 'native metadata binding')
    audit = read(native/'sampling_audit.json'); original = read(source/'streaming_manifest.json')
    mapping = identities(policy, prepared, audit, original)
    dependency = Path(distribution('tab-transformer-pytorch').locate_file('')).resolve()
    roots = {'repository': str(repository), 'native': str(native), 'source': str(source),
             'historical_runtime': str(historical), 'dependency': str(dependency)}
    bindings = []
    def bind(root, path, expected=None):
        f = safe_file(roots[root], path); actual = sha(f)
        require(expected is None or expected == actual, 'unexpected bound file: '+path)
        bindings.append({'root': root, 'path': path, 'sha256': actual, 'bytes': f.stat().st_size})
    allow = read(repository / (prefix+'PUBLIC_FILES.json'))['files']
    for path in allow:
        bind('repository', path)
    bind('repository', 'experimental/etg_input_audit/audit_inputs.py', prepared['auditor_sha256'])
    for name in ('bundle.py', 'profile_core.py', 'local_profile.py', 'build_candidate.py', 'test_profile.py'):
        bind('repository', 'experimental/etg_native_profile_v1/' + name)
    for ref in lineage['source_files']:
        bind('historical_runtime', ref['path'], ref['sha256'])
    bind('dependency', 'tab_transformer_pytorch/ft_transformer.py',
         'db62c6e258467bb2d85b738fe1839f0b4279ec92f0bdbb83400ddd42fadd4d42')
    metadata = {'policy': {'root': 'repository', 'path': prefix+'policy.json'},
        'prepared': {'root': 'repository', 'path': prefix+'PREPARED_INPUTS.json'},
        'audit': {'root': 'native', 'path': 'sampling_audit.json'},
        'source': {'root': 'source', 'path': 'streaming_manifest.json'}}
    bind('native', 'sampling_audit.json', policy['sampling_audit_sha256'])
    bind('native', 'streaming_manifest.json', policy['native_manifest_sha256'])
    bind('source', 'streaming_manifest.json', policy['source_manifest_sha256'])
    for cp in prepared['checkpoints']:
        base = f"seed_1/checkpoint_{cp['checkpoint']:03d}/"
        for name, expected in cp['sha256'].items():
            bind('native', base+name, expected)
        manifest = read(native / (base+'checkpoint_manifest.json'))
        require(manifest['probe_scores_file'] == 'probe_scores.npz', 'checkpoint dependency path')
        bind('native', base+'probe_scores.npz', manifest['probe_scores_sha256'])
    for cid in (0, 1):
        for root, ref in [('source', mapping[cid]['source']), ('native', mapping[cid]['calibration_file'])]:
            bind(root, ref['path'], ref['sha256'])
    require(len({(b['root'], b['path']) for b in bindings}) == len(bindings), 'duplicate binding')
    run_id = 'etg-native-components-20260911-v1'
    parent = Path(output_parent).resolve(); require(parent.is_dir(), 'existing output parent required')
    return {'schema': 'etg-local-component-profile-v1',
        'scope': 'LOCAL_COMPONENT_PROFILE_NOT_ETG_STUDY', 'run_id': run_id,
        'scientific_result_authorized': False, 'hpc_authorized': False,
        'host': socket.gethostname(), 'python_executable': str(Path(sys.executable).resolve()),
        'python_version': platform.python_version(), 'packages': {n: version(n) for n in
            ('numpy', 'torch', 'shap', 'tab-transformer-pytorch', 'einops', 'hyper-connections', 'psutil')},
        'roots': roots, 'bindings': bindings, 'metadata': metadata, 'targets': selected_targets(mapping),
        'launcher': 'experimental/etg_native_profile_v1/local_profile.py', 'profile_seed': 20260911,
        'checkpoints': [0, 1], 'device': 'cuda:0', 'output': str(parent/('run-'+run_id)),
        'limits': {'max_native_calls': 96, 'native_seconds': 120, 'process_seconds': 180,
                   'rss_bytes': 8*1024**3, 'gpu_allocator_bytes': 3*1024**3,
                   'required_gpu_free_mib': 4096, 'max_start_gpu_utilization_percent': 10, 'cpu_threads': 2},
        'historical_environment_match': False,
        'prior_use': 'Retrospective native calibration; companion features include other registered roles; no outcome-guided selection or efficacy metrics.',
        'resource_limits_are': 'Local abort ceilings for a short feasibility profile, not measured HPC requests.',
        'next_gate': 'Independent review, exact receipt binding, idle GPU and explicit local execution authorization.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--output-parent', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.workspace_root, a.output_parent), indent=2, allow_nan=False))
