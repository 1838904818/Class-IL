"""Native D2 metadata planning and read-only, profile-only tensor materialization."""
import json
from pathlib import Path
import numpy as np
from native_loader import safe_file, sha
from pilot_core import digest, partition_indices, require
from prepare_plan import load_auditor


def read_bound(root, reference):
    path = safe_file(root, reference['path'])
    require(sha(path) == reference['sha256'], 'metadata hash mismatch')
    require(path.stat().st_size <= 32 * 1024**2, 'oversized metadata')
    return json.loads(path.read_text(encoding='utf-8'))


def identities(policy, saved, audit, source):
    """Reconstruct offsets only. No feature or checkpoint tensor access."""
    require(policy['schema'] == saved['schema'] == 'etg-early-descriptive-v1', 'pilot schema')
    require(policy['classes'] == [0, 1, 2, 3] and policy['old_classes'] == [0, 1], 'class scope')
    require(source['feature_dim'] == 78, 'feature width')
    require(audit['source_manifest']['sha256'] == policy['source_manifest_sha256'], 'source lineage')
    require(audit['configuration']['seed'] == 42, 'data split seed')
    auditor = load_auditor()
    records = {r['id']: r for r in audit['classes']}
    source_classes = {r['id']: r for r in source['classes']}
    saved_rows = {r['class_id']: r for r in saved['rows']}
    result = {}
    for cid in policy['classes']:
        rec = records[cid]
        shards = source_classes[cid]['train']
        require(len(shards) == 1 and shards[0]['rows'] == rec['source_train_rows'], 'source shard count')
        cal, fit = auditor.reconstruct_indices(rec, policy['source_manifest_sha256'], 42)
        roles = partition_indices(cal, fit, cid, policy['split_salt'], policy['role_counts_per_class'])
        items = {}
        for role, offsets in roles.items():
            ordinal = np.searchsorted(cal, offsets)
            expected = saved_rows[cid]['roles'][role]
            require(digest(offsets) == expected['source_offsets_sha256'] and
                    digest(ordinal.tolist()) == expected['calibration_ordinals_sha256'] and
                    digest([[int(i)//512, int(i)%512] for i in ordinal]) ==
                    expected['native_class_batch512_context_sha256'], 'planned role changed')
            items[role] = {'offsets': offsets, 'ordinals': ordinal.tolist(), 'row_ids': [
                digest(['native-d2-row-v1', policy['source_manifest_sha256'], shards[0]['sha256'], cid, i])
                for i in offsets]}
        background = []
        if cid in policy['old_classes']:
            background = sorted(sorted(fit.tolist(), key=lambda i:
                (digest([policy['split_salt'], 'background', cid, i]), i))[:policy['background_per_old_class']])
            require(digest(background) == saved_rows[cid]['background']['source_offsets_sha256'], 'background changed')
        result[cid] = {'calibration': cal, 'fitting': fit, 'roles': items, 'background': background,
                       'source': shards[0], 'calibration_file': rec['calibration'],
                       'source_rows': rec['source_train_rows']}
    ids = [r for c in result.values() for role in c['roles'].values() for r in role['row_ids']]
    require(len(ids) == 896 and len(set(ids)) == 896, 'duplicate/missing role identities')
    return result


def selected_targets(mapping):
    # Fixed chronology/position, not predictions: first Benign, last GoldenEye trigger.
    output = []
    for cid, index in [(0, 0), (1, -1)]:
        role = mapping[cid]['roles']['trigger']
        ordinal = role['ordinals'][index]
        start = (ordinal // 512) * 512
        output.append({'class_id': cid, 'source_offset': role['offsets'][index],
            'row_id': role['row_ids'][index], 'calibration_ordinal': ordinal,
            'parent_start': start, 'parent_rows': min(512, len(mapping[cid]['calibration']) - start),
            'row_in_parent': ordinal - start, 'native_shard_ordinal': 0})
    return output


def open_features(root, reference, rows):
    path = safe_file(root, reference['path'])
    require(sha(path) == reference['sha256'], 'feature digest mismatch before mapping')
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    require(values.dtype == np.float32 and values.shape == (rows, 78) and
            not values.flags.writeable and values.flags.c_contiguous, 'native feature array contract')
    return values


class ProfileInputs:
    """Only two trigger contexts plus Task-0 references; no action/evaluation API."""
    def __init__(self, mapping, source_root, native_root, planned_targets):
        self.mapping = mapping
        self.source_root, self.native_root = Path(source_root), Path(native_root)
        require(selected_targets(mapping) == planned_targets, 'profile target binding changed')
        self.targets = planned_targets
        self.contexts = {}
        refs = []
        for target in planned_targets:
            cid = target['class_id']; rec = mapping[cid]
            source = open_features(source_root, rec['source'], rec['source_rows'])
            cal = open_features(native_root, rec['calibration_file'], len(rec['calibration']))
            start, n = target['parent_start'], target['parent_rows']
            parent = np.array(cal[start:start+n], copy=True, order='C')
            offsets = rec['calibration'][start:start+n]
            require(np.array_equal(parent, source[offsets]) and np.isfinite(parent).all(),
                    'native context differs from reconstructed source rows')
            reference = np.array(source[rec['background']], copy=True, order='C')
            require(reference.shape == (16, 78) and np.isfinite(reference).all(), 'reference rows')
            self.contexts[cid] = parent; refs.append(reference)
            # Do not keep large memmaps or copy complete source/calibration populations.
            del source, cal
        self.references = np.concatenate(refs, axis=0)
        require(self.references.shape == (32, 78), 'shared Task-0 background')

    def recheck_files(self):
        for cid in (0, 1):
            rec = self.mapping[cid]
            for root, ref in [(self.source_root, rec['source']), (self.native_root, rec['calibration_file'])]:
                require(sha(safe_file(root, ref['path'])) == ref['sha256'], 'input changed during profile')
