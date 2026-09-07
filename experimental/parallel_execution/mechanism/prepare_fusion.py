"""Prepare an immutable fixed-policy input from verified native export files."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import anchor_fusion as a


def prepare(export_dir, policy_file, output):
    export_dir, policy_file = a.no_links(export_dir).resolve(), a.no_links(policy_file).resolve()
    receipt = a.strict_json(export_dir / 'EXPORT_RECEIPT.json')
    complete = a.strict_json(export_dir / 'COMPLETE.json')
    receipt_hash = a.sha(export_dir / 'EXPORT_RECEIPT.json')
    a.require(complete['status'] == receipt['status'] == 'COMPLETE' and complete['receipt_sha256'] == receipt_hash, 'incomplete native score export')
    a.require(complete['arrays'] == receipt['arrays'], 'export array receipt mismatch')
    a.require(receipt['evidence_kind'] in ('synthetic', 'real'), 'explicit native evidence kind required')
    a.require(receipt['fixed_reference_state_eligible'] is True, 'Task-0 reference state changed')
    policy_hash = a.sha(policy_file)
    policy = a.strict_json(policy_file)
    a.require(set(policy) == {'schema', 'weight', 'floor', 'fixed_scale', 'tolerance', 'block_rows', 'selection_origin'}, 'fusion policy closure')
    a.require(policy['schema'] == 'fixed-fusion-controls-v1' and policy['selection_origin'] == 'development_only_frozen_before_evaluation', 'no test-selected parameters')
    # Exercise all numerical settings before copying output arrays.
    a.check_transition([[.2,.8]], [[0.,-1.]], [[.2,.8,0.]], [[0.,-1.,-10.]],
                       ['a','b'], ['a','b','c'], ['a','b'], **{k: policy[k] for k in ('weight', 'floor', 'fixed_scale', 'tolerance')})
    a.require(type(policy['block_rows']) is int and 0 < policy['block_rows'] <= 4096, 'bounded block size')
    output = a.no_links(output)
    a.require(not output.exists() and not output.is_symlink() and not any(p.is_symlink() for p in output.parents), 'refuse output overwrite/link')
    a.require(export_dir not in output.resolve().parents, 'immutable export must not be modified')
    arrays = {}
    for key in ('old_head', 'old_raw', 'new_head', 'new_raw', 'old_row_ids', 'new_row_ids'):
        desc = receipt['arrays'][key]
        value = a.bound_array(export_dir, desc)
        value._mmap.close()
        arrays[key] = {'path': key + '.npy', 'sha256': desc['sha256']}
    output.mkdir(parents=True, exist_ok=False)
    for key in arrays:
        shutil.copyfile(export_dir / receipt['arrays'][key]['path'], output / arrays[key]['path'])
        a.require(a.sha(output / arrays[key]['path']) == arrays[key]['sha256'], 'copy hash mismatch')
    reference, old, new = (receipt['checkpoints'][k] for k in ('reference', 'old', 'new'))
    def object_hash(obj):
        return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    manifest = {'schema': 'expansion-fusion-pilot-v1', 'status': 'BOUND', 'evidence_kind': receipt['evidence_kind'],
                'independent_binary_head_probabilities': True, 'reference_origin': 'task0_train_only_frozen_router_bank',
                'reference_classes': reference['classes'], 'old_classes': old['classes'], 'new_classes': new['classes'],
                'reference_state_sha256': object_hash(reference['identity']['centroids']),
                'encoder_state_sha256': reference['identity']['encoder'],
                'feature_contract_sha256': object_hash({k: reference['identity'][k] for k in ('mean', 'scale')}),
                'score_export_receipt_sha256': receipt_hash, 'arrays': arrays,
                **{k: policy[k] for k in ('weight', 'floor', 'fixed_scale', 'tolerance', 'block_rows')}}
    a.require(a.sha(export_dir / 'EXPORT_RECEIPT.json') == receipt_hash and a.sha(policy_file) == policy_hash, 'source changed')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    (output / 'PREPARATION.json').write_text(json.dumps({'status': 'INPUTS_PREPARED', 'policy_sha256': policy_hash,
        'native_export_receipt_sha256': receipt_hash, 'manifest_sha256': a.sha(output / 'manifest.json'),
        'evidence_kind': receipt['evidence_kind'], 'serialized_state_verified': True,
        'training_lineage_verified': False, 'historical_gpu_fidelity_verified': False}) + '\n', encoding='utf-8')
    return output / 'manifest.json'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    v = p.parse_args()
    prepare(v.export, v.policy, v.output)
