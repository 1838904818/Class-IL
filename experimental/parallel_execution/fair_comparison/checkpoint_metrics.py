"""Aggregate-only checkpoint payloads; no network, credentials or SDK imports.

Fractions stay in [0,1]; signed recall forgetting stays in [-1,1].
Per-class recall is not relabelled as overall accuracy. The caller must bind
and review the destination and persist delivery state before using a transport.
"""
import math
import re


def checkpoint_payload(checkpoint, *, arm, checkpoint_sha256, history):
    if arm not in ('ce', 'conditional_focal'):
        raise ValueError('Unknown experimental arm')
    if not isinstance(checkpoint_sha256, str) or not re.fullmatch('[a-f0-9]{64}', checkpoint_sha256):
        raise ValueError('Committed checkpoint digest required')
    task = checkpoint['task']
    if type(task) is not int or task < 0:
        raise ValueError('Invalid task')
    if not isinstance(history, list) or len(history) != task:
        raise ValueError('Complete committed checkpoint history required')
    previous = {}
    for i, old in enumerate(history):
        if old['task'] != i:
            raise ValueError('Noncontiguous history')
        # Validate every supplied historical metric too. The caller must verify
        # the checkpoint file bindings before constructing this aggregate view.
        checkpoint_payload(old, arm=arm, checkpoint_sha256=checkpoint_sha256, history=history[:i])
        for row in old['per_class']:
            previous[row['class_id']] = max(previous.get(row['class_id'], -1), row['recall'])
    axis = checkpoint['class_axis']
    if not isinstance(axis, list) or not axis or any(type(c) is not int or c < 0 for c in axis) or len(set(axis)) != len(axis):
        raise ValueError('Invalid class axis')
    cm = checkpoint['confusion']
    if not isinstance(cm, list) or len(cm) != len(axis) or any(
        not isinstance(row, list) or len(row) != len(axis) or
        any(type(v) is not int or v < 0 for v in row) for row in cm):
        raise ValueError('Invalid confusion counts')
    support = [sum(row) for row in cm]
    if any(n == 0 for n in support):
        raise ValueError('Official-test protocol requires support for every seen class')
    if not set(previous).issubset(axis):
        raise ValueError('Previously seen class disappeared')
    total = sum(support)
    if not total:
        raise ValueError('Empty evaluation')
    rows = checkpoint['per_class']
    if len(rows) != len(axis) or [row['class_id'] for row in rows] != axis:
        raise ValueError('Per-class axis mismatch')
    metrics = {'task': task}
    def finite(value, low=0, high=1):
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError('Invalid metric')
        return value
    for name in ('accuracy', 'macro_f1', 'balanced_accuracy'):
        metrics[f'{arm}/{name}'] = finite(checkpoint[name])
    if not math.isclose(checkpoint['accuracy'], sum(cm[i][i] for i in range(len(axis))) / total, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError('Accuracy does not match confusion matrix')
    per_class = []
    for i, row in enumerate(rows):
        if type(row['support']) is not int or row['support'] != support[i]:
            raise ValueError('Support does not match confusion matrix')
        values = [finite(row[k]) for k in ('precision', 'recall', 'f1')]
        predicted = sum(r[i] for r in cm)
        precision = cm[i][i] / predicted if predicted else 0.0
        recall = cm[i][i] / support[i] if support[i] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in zip(values, (precision, recall, f1))):
            raise ValueError('Per-class metric does not match confusion matrix')
        per_class.append([task, axis[i], support[i], *values])
    for name, column in (('macro_f1', 5), ('balanced_accuracy', 4)):
        if not math.isclose(checkpoint[name], sum(row[column] for row in per_class) / len(axis), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError('Aggregate metric mismatch')
    expected = [(row['class_id'], previous[row['class_id']] - row['recall'])
                for row in rows if row['class_id'] in previous]
    supplied = checkpoint['signed_forgetting']
    if not isinstance(supplied, list) or len(supplied) != len(expected):
        raise ValueError('Invalid per-class forgetting')
    for row, (c, value) in zip(supplied, expected):
        if type(row['class_id']) is not int or row['class_id'] != c or not math.isclose(
            finite(row['signed_recall_forgetting'], -1, 1), value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError('Forgetting differs from historical best recall')
    forgetting = checkpoint['mean_signed_forgetting']
    if not expected:
        if forgetting is not None:
            raise ValueError('Initial checkpoint has no forgetting')
    else:
        value = sum(v for _, v in expected) / len(expected)
        if not math.isclose(finite(forgetting, -1, 1), value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError('Mean forgetting differs from historical best recall')
        metrics[f'{arm}/mean_signed_recall_forgetting'] = value
    return {'event_id': f'{checkpoint_sha256}:{arm}:{task}', 'metrics': metrics,
            'per_class': {'columns': ['task', 'class_id', 'support', 'precision', 'recall', 'f1'], 'data': per_class},
            'confusion': {'columns': ['task', 'true_class_id', 'predicted_class_id', 'count'],
                          'data': [[task, c, d, cm[i][j]] for i, c in enumerate(axis) for j, d in enumerate(axis)]}}
