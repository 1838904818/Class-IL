"""Pair a checksummed replay result audit with an exact same-seed OFRA result."""
import argparse
import hashlib
import json
from pathlib import Path

METRICS = ('average_task_accuracy', 'final_overall_accuracy', 'final_macro_f1',
           'final_balanced_accuracy', 'average_forgetting',
           'final_attack_detection_recall', 'final_benign_false_positive_rate')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def compare(audit_path, ofra_path, expected_ofra_hash):
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if audit.get('status') != 'verified':
        raise ValueError('Baseline audit must be verified')
    if digest(ofra_path) != expected_ofra_hash:
        raise ValueError('OFRA hash mismatch')
    raw_path = audit_path.parent / 'baseline_result.json'
    if digest(raw_path) != audit['result_file_sha256']:
        raise ValueError('Baseline result hash mismatch')
    for filename, expected in audit['protected_checksums'].items():
        if Path(filename).name != filename or digest(audit_path.parent / filename) != expected:
            raise ValueError('Protected file binding mismatch')
    raw = json.loads(raw_path.read_text(encoding='utf-8'))
    ofra = json.loads(ofra_path.read_text(encoding='utf-8'))
    if ofra['seed'] != audit['seed'] or ofra['dataset'] != audit['dataset']:
        raise ValueError('Dataset or seed mismatch')
    if raw['protocol']['seed'] != audit['seed'] or raw['protocol']['dataset'] != audit['dataset']:
        raise ValueError('Baseline protocol identity mismatch')
    baseline = audit['methods']['balanced_replay50']
    scores = ofra['summary']['views']['official']['joint_cap3000']
    rows = ofra['checkpoints'][-1]['views']['official']['arms']['joint_cap3000']['per_class']
    b_rows = baseline['per_class']
    identity = lambda seq: [(r['class_id'], r['class_name'], r['support']) for r in seq]
    if identity(rows) != identity(b_rows):
        raise ValueError('Final class order, label or support mismatch')
    if len(raw['results']['balanced_replay50']['checkpoints']) != len(ofra['checkpoints']):
        raise ValueError('Checkpoint count mismatch')
    for key in METRICS:
        if baseline.get(key) != raw['results']['balanced_replay50']['summary'].get(key):
            raise ValueError('Audit summary differs from bound raw result')
    metrics = {key: {'balanced_replay50': baseline.get(key), 'ofra_joint_cap3000': scores.get(key),
                    'ofra_minus_replay_pp': 100 * (scores[key] - baseline[key])}
               for key in METRICS if scores.get(key) is not None and baseline.get(key) is not None}
    return dict(status='verified_single_seed_pair', dataset=audit['dataset'], seed=audit['seed'],
                ofra_result_sha256=digest(ofra_path), baseline_result_sha256=digest(raw_path),
                baseline_audit_sha256=digest(audit_path), metrics=metrics,
                checkpoints=len(ofra['checkpoints']), final_test_rows=sum(r['support'] for r in rows),
                per_class=[dict(class_id=r['class_id'], class_name=r['class_name'], support=r['support'],
                                ofra={k:r[k] for k in ('precision','recall','f1')},
                                balanced_replay50={k:b[k] for k in ('precision','recall','f1')}) for r,b in zip(rows,b_rows)],
                wandb_url=baseline['wandb_url'], wandb_cloud_independently_verified=False,
                limitations=['Single-seed evidence, not a five-seed or publication-final conclusion.',
                             'Seed 42 selected the comparator; report four new seeds separately as sensitivity.',
                             'Equal epochs are not equal compute: replay doubles later-task row exposure.',
                             'OFRA retains router centroids in addition to exemplars; total memory differs.',
                             'Matching labels and supports is not a row-identity proof; immutable data contracts bind the split.',
                             'W&B URL is recorded in the protected result; independent cloud access remains unverified.'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('audit', type=Path)
    parser.add_argument('ofra', type=Path)
    parser.add_argument('--ofra-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = compare(args.audit, args.ofra, args.ofra_sha256)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k in ('dataset','seed','metrics')}, indent=2))
