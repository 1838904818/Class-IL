"""Fixed-policy, score-level confusion evaluation after a completed mechanism run."""
from pathlib import Path
import argparse
import json
import numpy as np
import anchor_fusion as a


MODES = ('all_seen', 'all_seen_legacy_formula', 'reference', 'fixed_scalar', 'head_only', 'raw_router')


def metrics(cm, classes):
    support, predicted = cm.sum(1), cm.sum(0)
    a.require((support > 0).all(), 'all registered classes require evaluation support')
    tp = cm.diagonal()
    recall = tp / support
    precision = np.divide(tp, predicted, out=np.zeros(len(classes)), where=predicted > 0)
    f1 = np.divide(2 * recall * precision, recall + precision, out=np.zeros(len(classes)), where=(recall + precision) > 0)
    return {'class_axis': classes, 'confusion_counts': cm.tolist(), 'accuracy': float(tp.sum() / support.sum()),
            'macro_f1': float(f1.mean()), 'balanced_accuracy': float(recall.mean()),
            'per_class': {c: {'support': int(support[i]), 'recall': float(recall[i]),
                              'precision': float(precision[i]), 'f1': float(f1[i])} for i, c in enumerate(classes)}}


def evaluate(manifest, completed_pilot, labels_file, labels_sha256, output, *, evidence_kind):
    a.require(evidence_kind in ('synthetic', 'prospective-score-evaluation'), 'explicit evidence kind')
    manifest, completed_pilot = a.no_links(manifest).resolve(), a.no_links(completed_pilot).resolve()
    receipt = a.strict_json(completed_pilot / 'COMPLETE.json')
    policy_sha = a.sha(manifest)
    a.require(receipt['manifest_sha256'] == policy_sha and receipt['result_sha256'] == a.sha(completed_pilot / 'RESULT.json'), 'pilot receipt mismatch')
    pilot = a.strict_json(completed_pilot / 'RESULT.json')
    a.require(pilot['status'] == 'SCORE_PILOT_COMPLETE', 'pilot incomplete')
    p = a.strict_json(manifest)
    a.require(pilot['evidence_kind'] == p['evidence_kind'] and evidence_kind ==
              ('synthetic' if p['evidence_kind'] == 'synthetic' else 'prospective-score-evaluation'),
              'evaluation cannot relabel the bound evidence kind')
    output = a.no_links(output)
    a.require(not output.exists() and not output.is_symlink() and not any(x.is_symlink() for x in output.parents), 'output exists or linked')
    a.require(manifest.parent not in output.resolve().parents and completed_pilot not in output.resolve().parents,
              'separate output required')
    arrays = {}
    labels = None
    label_ids = None
    try:
        arrays = {n: a.bound_array(manifest.parent, d) for n, d in p['arrays'].items()}
        labels_file = a.no_links(labels_file).resolve()
        a.require(a.sha(labels_file) == labels_sha256, 'label bundle hash mismatch')
        label_spec = a.strict_json(labels_file)
        a.require(set(label_spec) == {'schema', 'labels', 'row_ids', 'score_export_receipt_sha256'}, 'label bundle closure')
        a.require(label_spec['schema'] == 'fusion-evaluation-labels-v1' and
                  label_spec['score_export_receipt_sha256'] == p['score_export_receipt_sha256'], 'label/source export mismatch')
        labels = a.bound_array(labels_file.parent, label_spec['labels'])
        label_ids = a.bound_array(labels_file.parent, label_spec['row_ids'])
        n = len(arrays['old_head'])
        a.require(labels.ndim == 1 and len(labels) == n and labels.dtype.kind == 'U', 'Unicode class-ID label vector required')
        a.require(np.array_equal(label_ids, arrays['old_row_ids']) and np.array_equal(label_ids, arrays['new_row_ids']),
                  'label-to-score row identity/order mismatch')
        old, new = list(a.axis(p['old_classes'])), list(a.axis(p['new_classes']))
        a.require(set(labels.tolist()) == set(new), 'unknown or missing class support')
        cms = {mode: {'old': np.zeros((len(old), len(old)), np.int64), 'new': np.zeros((len(new), len(new)), np.int64)} for mode in MODES}
        for begin in range(0, n, p['block_rows']):
            end = min(n, begin + p['block_rows'])
            y = labels[begin:end]
            for mode in MODES:
                for stage, classes in (('old', old), ('new', new)):
                    h, raw = (arrays[stage + '_' + key][begin:end] for key in ('head', 'raw'))
                    s = a.fuse(h, raw, classes, p['reference_classes'], weight=p['weight'], floor=p['floor'],
                               mode=mode, fixed_scale=p['fixed_scale'])
                    predictions = a.winner(s, classes)
                    eligible = np.isin(y, classes)
                    lookup = {c: i for i, c in enumerate(classes)}
                    truth = np.array([lookup[c] for c in y[eligible]], np.int64)
                    pred = np.array([lookup[c] for c in predictions[eligible]], np.int64)
                    np.add.at(cms[mode][stage], (truth, pred), 1)
        report = {'status': 'COMPLETE', 'evidence_kind': evidence_kind, 'policy_sha256': policy_sha,
                  'labels_sha256': labels_sha256, 'source_provenance_verified': False,
                  'historical_gpu_parity_verified': False, 'training_performed': False,
                  'full_stream_forgetting_computed': False, 'parameter_selection_performed': False, 'controls': {}}
        for mode, counts in cms.items():
            before, after = metrics(counts['old'], old), metrics(counts['new'], new)
            report['controls'][mode] = {'old_checkpoint_old_classes_only': before, 'new_checkpoint_all_classes': after,
                'old_class_adjacent_recall_drop': {c: before['per_class'][c]['recall'] - after['per_class'][c]['recall'] for c in old},
                'note': 'Signed adjacent-checkpoint recall drop, not maximum-past full-stream forgetting.'}
        a.require(a.sha(labels_file) == labels_sha256 and a.sha(manifest) == policy_sha, 'input changed during evaluation')
        for key in ('labels', 'row_ids'):
            a.require(a.sha(labels_file.parent / label_spec[key]['path']) == label_spec[key]['sha256'], 'label bundle changed')
        for desc in p['arrays'].values():
            a.require(a.sha(manifest.parent / desc['path']) == desc['sha256'], 'score input changed')
        output.mkdir(parents=True, exist_ok=False)
        (output / 'EVALUATION.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        (output / 'COMPLETE.json').write_text(json.dumps({'sha256': a.sha(output / 'EVALUATION.json')}) + '\n', encoding='utf-8')
        return report
    finally:
        for value in [*arrays.values(), labels, label_ids]:
            mapping = getattr(value, '_mmap', None)
            if mapping is not None and not mapping.closed:
                mapping.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--labels-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--evidence-kind', choices=('synthetic', 'prospective-score-evaluation'), required=True)
    v = p.parse_args()
    evaluate(v.manifest, v.pilot, v.labels, v.labels_sha256, v.output, evidence_kind=v.evidence_kind)
