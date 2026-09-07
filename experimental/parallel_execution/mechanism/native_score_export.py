"""Bound OFRA checkpoint-to-raw-score bridge; not historical GPU certification."""
from pathlib import Path
import argparse
import hashlib
import os
import shutil
import socket
import sys
import time

# The sibling exporter owns the shared encoder and canonical file helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fair_comparison'))
import l1_runner as r
from materialize_embeddings import build_bound_encoder
import numpy as np
import torch
from anchor_fusion import no_links


def safe_ref(root, ref):
    path = no_links(root / ref['path'])
    r.require(not path.is_symlink() and not any(p.is_symlink() for p in path.parents), 'linked input')
    return r.relative_file(root, ref)


def array_hash(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def load_bound(root, binding, runtime_root, device):
    r.exact(binding, {'metadata', 'state'}, 'checkpoint binding')
    metadata = r.read_json(safe_ref(root, binding['metadata']))
    r.integer(metadata['checkpoint'])
    state_file = safe_ref(root, binding['state'])
    r.require(metadata['inference_state_sha256'] == binding['state']['sha256'], 'state/metadata mismatch')
    if 'canonical_sha256' in metadata:
        r.require(r.object_hash({k: v for k, v in metadata.items() if k != 'canonical_sha256'}) == metadata['canonical_sha256'], 'canonical checkpoint mismatch')
    classes = metadata['seen_classes']
    r.require(isinstance(classes, list) and len(classes) >= 2 and len(set(classes)) == len(classes), 'class axis')
    for c in classes:
        r.integer(c)
    encoder = build_bound_encoder(metadata, runtime_root).to(device)
    heads, centers = {}, {}
    schema, arch = metadata['state_schema'], metadata['architecture']
    with np.load(state_file, allow_pickle=False) as archive:
        def tensors(mapping, expected):
            r.exact(mapping, set(expected), 'state tensor keys')
            values = {}
            for name, key in mapping.items():
                a = np.array(archive[key], copy=True)
                template = expected[name].detach().cpu().numpy()
                r.require(a.dtype == template.dtype and a.shape == template.shape and np.isfinite(a).all(),
                          'state dtype/shape/nonfinite mismatch; implicit conversion forbidden')
                values[name] = torch.from_numpy(a).to(device)
            return values
        encoder.load_state_dict(tensors(schema['encoder'], encoder.state_dict()), strict=True)
        mean = np.array(archive[schema['normalization']['mean']])
        scale = np.array(archive[schema['normalization']['scale']])
        r.require(mean.dtype == scale.dtype == np.float64 and mean.shape == scale.shape == (metadata['feature_dim'],) and np.isfinite(mean).all()
                  and np.isfinite(scale).all() and (scale > 0).all(), 'invalid normalization')
        for c in classes:
            head = r.FamilyHead(arch['d_model'], arch['lora_rank'], arch['lora_alpha']).to(device)
            head.load_state_dict(tensors(schema['heads'][str(c)], head.state_dict()), strict=True)
            head.eval()
            heads[c] = head
            centroid = np.array(archive[schema['cap3000_router'][str(c)]['centroids']])
            r.require(centroid.dtype == np.float32 and centroid.ndim == 2 and 0 < len(centroid) and centroid.shape[1] == arch['d_model']
                      and np.isfinite(centroid).all(), 'invalid router bank')
            centers[c] = centroid
    encoder.eval()
    for model in (encoder, *heads.values()):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    identity = {'encoder': r.tensor_state_hash(encoder.state_dict()),
                'mean': array_hash(mean), 'scale': array_hash(scale),
                'heads': {str(c): r.tensor_state_hash(h.state_dict()) for c, h in heads.items()},
                'centroids': {str(c): array_hash(v) for c, v in centers.items()}}
    return {'metadata': metadata, 'encoder': encoder, 'heads': heads, 'centers': centers,
            'mean': mean, 'scale': scale, 'identity': identity, 'device': device}


@torch.no_grad()
def block_scores(model, raw):
    """Same algebra as DualRouter._distances; no reconstruction from z scores."""
    values = np.asarray(raw, dtype=np.float64)
    normalized = ((values - model['mean']) / model['scale']).astype(np.float32)
    r.require(np.isfinite(normalized).all(), 'nonfinite normalized input')
    embedding = model['encoder'](torch.from_numpy(normalized).to(model['device']))
    e = embedding.cpu().numpy().astype(np.float32)
    r.require(np.isfinite(e).all(), 'nonfinite embedding')
    classes = model['metadata']['seen_classes']
    p = np.empty((len(e), len(classes)), np.float32)
    margin = np.empty(p.shape, np.float64)
    raw_router = np.empty(p.shape, np.float32)
    for j, c in enumerate(classes):
        logits = model['heads'][c](embedding)
        p[:, j] = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        logits64 = logits.cpu().numpy().astype(np.float64)
        margin[:, j] = logits64[:, 1] - logits64[:, 0]
        centroid = model['centers'][c]
        d2 = (np.einsum('ij,ij->i', e, e)[:, None]
              + np.einsum('ij,ij->i', centroid, centroid)[None, :] - 2.0 * e @ centroid.T)
        np.maximum(d2, 0.0, out=d2)
        raw_router[:, j] = -np.sqrt(d2.min(axis=1), dtype=np.float32)
    r.require(np.isfinite(p).all() and np.isfinite(margin).all() and np.isfinite(raw_router).all(), 'nonfinite exported score')
    return p, margin, raw_router


def _run(manifest, output, *, runtime_root=None, device_name='cpu', mapped):
    # This is also callable locally for tiny tests; never calculate on a login node.
    r.require(not socket.gethostname().split('.')[0].lower().startswith('login'), 'compute on login node forbidden')
    if 'dicc.um.edu.my' in socket.getfqdn().lower() or os.environ.get('SLURM_JOB_ID'):
        r.require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID'), 'scheduled compute step required')
    manifest = no_links(manifest).resolve()
    root, initial_hash = manifest.parent, r.sha(manifest)
    spec = r.read_json(manifest)
    r.exact(spec, {'schema', 'evidence_kind', 'batch_rows', 'checkpoints', 'arrays'}, 'score export manifest')
    r.require(spec['schema'] == 'native-ofra-score-pair-v1' and spec['evidence_kind'] in ('synthetic', 'real'), 'export schema')
    r.integer(spec['batch_rows'], 1)
    r.require(spec['batch_rows'] <= 4096, 'bounded inference batch required')
    r.exact(spec['checkpoints'], {'reference', 'old', 'new'}, 'checkpoint set')
    r.exact(spec['arrays'], {'raw_features', 'row_ids', 'group_ids'}, 'input array set')
    arrays = {}
    for k, desc in spec['arrays'].items():
        arrays[k] = np.load(safe_ref(root, desc), mmap_mode='r', allow_pickle=False)
        mapped.append(arrays[k])
    x = arrays['raw_features']
    r.require(x.ndim == 2 and x.dtype in (np.float32, np.float64) and len(x) > 0, 'raw feature array')
    for key in ('row_ids', 'group_ids'):
        a = arrays[key]
        r.require(a.ndim == 1 and len(a) == len(x) and a.dtype.kind in 'SU' and all(a.tolist()), 'row/group identity')
    r.require(len(set(arrays['row_ids'].tolist())) == len(x), 'duplicate rows')
    device = torch.device(device_name)
    r.require(device.type in ('cpu', 'cuda') and (device.type != 'cuda' or torch.cuda.is_available()), 'device unavailable')
    torch.use_deterministic_algorithms(True)
    if device.type == 'cuda':
        r.require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') in (':4096:8', ':16:8'), 'deterministic CUBLAS required')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    output = no_links(output)
    r.require(not output.exists() and not output.is_symlink() and not any(p.is_symlink() for p in output.parents), 'output overwrite/link')
    r.require(output.resolve() != root and root not in output.resolve().parents, 'output must be outside inputs')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {'schema': spec['schema'], 'evidence_kind': spec['evidence_kind'], 'manifest_sha256': initial_hash,
              'status': 'STARTED', 'rows': len(x), 'batch_rows': spec['batch_rows'], 'environment': r.environment(device),
              'historical_gpu_fidelity_verified': False, 'training_lineage_verified': False,
              'labels_read': False, 'attributions_computed': False, 'checkpoints': {}, 'arrays': {}}
    try:
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        for role in ('reference', 'old', 'new'):
            model = load_bound(root, spec['checkpoints'][role], runtime_root, device)
            meta = model['metadata']
            r.require(meta['feature_dim'] == x.shape[1], 'feature contract width mismatch')
            if role == 'reference':
                r.require(meta['checkpoint'] == 0, 'reference must be checkpoint 0')
            report['checkpoints'][role] = {'checkpoint': meta['checkpoint'], 'classes': [str(c) for c in meta['seen_classes']],
                                          'identity': model['identity'], 'dataset': meta['dataset'], 'seed': meta['seed']}
            if role != 'reference':
                classes = meta['seen_classes']
                dest = {k: np.lib.format.open_memmap(output / f'{role}_{k}.npy', mode='w+', shape=(len(x), len(classes)),
                                                     dtype=np.float64 if k == 'head_logits' else np.float32)
                        for k in ('head', 'head_logits', 'raw')}
                mapped.extend(dest.values())
                for begin in range(0, len(x), spec['batch_rows']):
                    scores = block_scores(model, x[begin:begin + spec['batch_rows']])
                    for key, value in zip(('head', 'head_logits', 'raw'), scores):
                        dest[key][begin:begin + len(value)] = value
                for value in dest.values():
                    value.flush()
                    value._mmap.close()
                dest.clear()
                del value
            del model
        ref, old, new = (report['checkpoints'][k] for k in ('reference', 'old', 'new'))
        r.require(ref['dataset'] == old['dataset'] == new['dataset'] and ref['seed'] == old['seed'] == new['seed'], 'dataset/seed changed')
        r.require(old['checkpoint'] < new['checkpoint'] and set(ref['classes']) <= set(old['classes']) < set(new['classes']), 'not a valid expansion')
        same_encoder = all(row['identity'][k] == ref['identity'][k] for row in (old, new) for k in ('encoder', 'mean', 'scale'))
        same_reference = all(row['identity']['centroids'][c] == ref['identity']['centroids'][c]
                             for row in (old, new) for c in ref['classes'])
        report['fixed_reference_state_eligible'] = same_encoder and same_reference
        report['serialized_task0_reference_verified'] = True
        for role in ('old', 'new'):
            shutil.copyfile(safe_ref(root, spec['arrays']['row_ids']), output / f'{role}_row_ids.npy')
        shutil.copyfile(safe_ref(root, spec['arrays']['group_ids']), output / 'group_ids.npy')
        for path in sorted(output.glob('*.npy')):
            report['arrays'][path.stem] = {'path': path.name, 'sha256': r.sha(path)}
        for binding in spec['checkpoints'].values():
            for desc in binding.values():
                safe_ref(root, desc)
        for desc in spec['arrays'].values():
            safe_ref(root, desc)
        r.require(r.sha(manifest) == initial_hash, 'manifest changed during extraction')
        report.update(status='COMPLETE', elapsed_seconds=time.perf_counter() - started,
                      process_memory=r.process_memory(),
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None)
        r.atomic_json(output / 'EXPORT_RECEIPT.json', report)
        r.atomic_json(output / 'COMPLETE.json', {'receipt_sha256': r.sha(output / 'EXPORT_RECEIPT.json'),
                                               'arrays': report['arrays'], 'status': 'COMPLETE'})
        return report
    except BaseException as exc:
        report.update(status='FAILED', error_type=type(exc).__name__, elapsed_seconds=time.perf_counter() - started)
        r.atomic_json(output / 'FAILED.json', report)
        raise


def run(manifest, output, *, runtime_root=None, device_name='cpu'):
    mapped = []
    try:
        return _run(manifest, output, runtime_root=runtime_root, device_name=device_name, mapped=mapped)
    finally:
        for value in mapped:
            mapping = getattr(value, '_mmap', None)
            if mapping is not None and not mapping.closed:
                mapping.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--runtime-root', type=Path)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    a = p.parse_args()
    run(a.manifest, a.output, runtime_root=a.runtime_root, device_name=a.device)


if __name__ == '__main__':
    main()
