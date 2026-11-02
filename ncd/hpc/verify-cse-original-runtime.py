"""Independent saved-CM arithmetic/reference checks; no model execution.

This verifies derived comparisons, not independent per-logit reconstruction.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--results', type=Path, required=True)
p.add_argument('--inputs', type=Path, required=True)
p.add_argument('--source', type=Path, required=True)
a = p.parse_args()
read = lambda path: json.loads(path.read_text(encoding='utf-8'))
summary = read(a.results / 'summary.json')
assert not summary['software_only'] and not summary['failures']
expected = [('icarl', 10), ('cer-bic-common', 10), ('cer-bic-common', 11), ('cer-bic-common', 12)]
assert [(v['method'], v['seed']) for v in summary['completed']] == expected
source_sha = hashlib.sha256(a.source.read_bytes()).hexdigest()
rows, changes, fp_changes = [], [], []
for method, seed in expected:
    r = read(a.results / f'{method}-seed{seed}.json')
    root = a.inputs / 'cic-ids-2018' / method / f'seed{seed}'
    original = read(root / 'result.json')
    assert r['source_result_sha256'] == hashlib.sha256((root / 'result.json').read_bytes()).hexdigest()
    assert r['source_code_sha256'] == source_sha
    assert r['forward_source_sha256'] == '3fe63c8f404d5217a6c63108e7169132bab7b75751a03146d17038ecc856ca2a'
    assert r['source_identity'] == original['identity']
    assert r['runtime']['platform'].startswith('Linux') and not r['software_only']
    assert len(r['stages']) == len(original['checkpoints']) == 4
    defaults = original.get('default_stages') or read(root / 'default_stages.json')
    calibrated = original.get('calibrated_stages') or read(root / 'calibrated_stages.json')
    for t, stage in enumerate(r['stages']):
        cp = original['checkpoints'][t]
        assert stage['task_index'] == t and stage['checkpoint_sha256'] == cp['checkpoint_sha256']
        assert set(stage['views']) == {'default', '0.01', '0.02', '0.05'}
        for view, entry in stage['views'].items():
            reference = defaults[t]['metrics'] if view == 'default' else calibrated[t]['operating_points'][view]
            old = reference['confusion_matrix']
            new = entry['metrics']['confusion_matrix']
            assert old == entry['reference_confusion_matrix']
            n = len(old)
            assert len(new) == n and all(len(z) == n for z in new)
            assert all(type(v) is int and v >= 0 for z in new for v in z)
            assert [sum(z) for z in new] == [sum(z) for z in old]
            delta = [[new[j][k] - old[j][k] for k in range(n)] for j in range(n)]
            assert delta == entry['confusion_matrix_delta']
            if any(v for z in delta for v in z):
                changes.append(dict(method=method, seed=seed, task=t, view=view, delta=delta))
            recalls = [Fraction(new[c][c], sum(new[c])) for c in cp['seen_classes']]
            assert [float(Fraction(new[c][c], sum(new[c]))) if sum(new[c]) else 0.0 for c in range(n)] == entry['metrics']['per_class_recall']
            old_bacc = sum((Fraction(old[c][c], sum(old[c])) for c in cp['seen_classes']), Fraction()) / len(recalls)
            new_bacc = sum(recalls, Fraction()) / len(recalls)
            if view != 'default':
                point = calibrated[t]['calibration']['operating_points'][view]
                assert entry['original_total_tau'] == point['total_tau']
                assert entry['reference_holdout_false_positives'] == point['calibration_false_positives']
                assert type(entry['holdout_false_positives']) is int and entry['holdout_false_positives'] >= 0
                if entry['holdout_false_positives'] != point['calibration_false_positives']:
                    fp_changes.append(dict(method=method, seed=seed, task=t, view=view,
                                           delta=entry['holdout_false_positives'] - point['calibration_false_positives']))
            if t == 3 and view == '0.01':
                rows.append(dict(method=method, seed=seed, windows_bacc_percent=float(100 * old_bacc),
                                 linux_bacc_percent=float(100 * new_bacc),
                                 delta_pp=float(100 * (new_bacc - old_bacc)),
                                 exact_delta_fraction=str(100 * (new_bacc - old_bacc))))
print(json.dumps(dict(final_one_percent=rows, cm_views_checked=64, holdout_points_checked=48,
                      changed_confusion_matrices=changes, changed_holdout_points=fp_changes,
                      scope='Exact integer CM/reference/tau/delta and rational BAcc arithmetic; no independent model or raw-logit rerun.'), indent=2))
