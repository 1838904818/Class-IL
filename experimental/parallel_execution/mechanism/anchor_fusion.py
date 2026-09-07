"""Fixed-reference fusion: a research scorer, never a historical-score replacement."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import platform
import stat
import time
import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def no_links(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        require(not part.is_symlink(), 'symlink path forbidden')
        if part.exists():
            require(not (getattr(part.lstat(), 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 1024)),
                    'Windows reparse path forbidden')
    return path


def strict_json(path):
    path = no_links(path)
    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, 'duplicate JSON key')
            out[k] = v
        return out
    def number(s):
        v = float(s)
        require(math.isfinite(v), 'nonfinite JSON number')
        return v
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs,
                      parse_float=number, parse_constant=number)


def axis(ids):
    require(isinstance(ids, (list, tuple)) and len(ids) >= 2, 'two or more classes required')
    require(all(type(c) is str and c for c in ids), 'class IDs must be nonempty strings')
    require(len(set(ids)) == len(ids), 'duplicate class ID')
    return tuple(ids)


def matrix(a, ids, probability=False):
    original = np.asarray(a)
    require(original.dtype.kind in 'iuf', 'numeric score arrays required; implicit string conversion forbidden')
    v = np.asarray(a, dtype=np.float64)
    require(v.ndim == 2 and v.shape[1] == len(ids) and np.isfinite(v).all(), 'invalid score matrix')
    if probability:
        require(((v >= 0) & (v <= 1)).all(), 'independent positive probabilities required')
    return v


def fuse(head, raw, classes, reference_classes, *, weight, floor, mode='reference', fixed_scale=None):
    """Row-wise reference only; no statistics across query rows or future classes.

    Reference banks must have frozen scores under the same feature/encoder state.
    Floor is an explicit prospective scale choice, not an empirically safe default.
    """
    ids = axis(classes)
    refs = axis(reference_classes)
    require(set(refs) <= set(ids), 'reference class is unavailable')
    require(type(weight) in (float, int) and math.isfinite(weight) and weight >= 0, 'invalid weight')
    require(type(floor) in (float, int) and math.isfinite(floor) and floor > 0, 'invalid scale floor')
    h, r = matrix(head, ids, True), matrix(raw, ids)
    require(h.shape == r.shape, 'head/router shape mismatch')
    require(mode in ('reference', 'all_seen', 'all_seen_legacy_formula', 'fixed_scalar', 'head_only', 'raw_router'), 'unknown control')
    cols = sorted(range(len(ids)), key=lambda j: ids[j])
    refcols = [ids.index(c) for c in sorted(refs)] if mode == 'reference' else cols
    center = np.mean(r[:, refcols], axis=1, keepdims=True)
    scale = np.maximum(np.std(r[:, refcols], axis=1, keepdims=True, ddof=0), floor)
    if mode == 'all_seen_legacy_formula':
        scale = np.std(r[:, refcols], axis=1, keepdims=True, ddof=0) + floor
    require(np.isfinite(center).all() and np.isfinite(scale).all(), 'nonfinite normalization statistics')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        if mode == 'fixed_scalar':
            require(type(fixed_scale) in (float, int) and math.isfinite(fixed_scale) and fixed_scale > 0,
                    'fixed-scalar control needs a bound development-only scale')
            z = r / fixed_scale
        elif mode == 'raw_router':
            z = r
        elif mode == 'head_only':
            z = np.zeros_like(r)
        else:
            z = (r - center) / scale
        out = h + weight * z
    require(np.isfinite(out).all(), 'nonfinite fusion')
    return out


def winner(scores, classes):
    ids = axis(classes)
    v = matrix(scores, ids)
    order = np.array(sorted(range(len(ids)), key=lambda j: ids[j]))
    return np.asarray(ids)[order[np.argmax(v[:, order], axis=1)]]


def check_transition(old_head, old_raw, new_head, new_raw, old_classes, new_classes,
                     reference_classes, *, weight, floor, fixed_scale, tolerance):
    old_ids, new_ids = axis(old_classes), axis(new_classes)
    require(set(old_ids) < set(new_ids), 'must be a genuine class expansion')
    require(type(tolerance) in (int, float) and math.isfinite(tolerance) and tolerance >= 0, 'invalid tolerance')
    oh, ort = matrix(old_head, old_ids, True), matrix(old_raw, old_ids)
    nh, nrt = matrix(new_head, new_ids, True), matrix(new_raw, new_ids)
    require(oh.shape == ort.shape and nh.shape == nrt.shape and len(oh) == len(nh), 'unaligned rows')
    oldcols = [new_ids.index(c) for c in old_ids]
    unchanged = bool(np.array_equal(oh, nh[:, oldcols]) and np.array_equal(ort, nrt[:, oldcols]))
    result = {'rows': len(oh), 'frozen_old_input_scores': unchanged, 'controls': {}}
    for mode in ('all_seen', 'all_seen_legacy_formula', 'reference', 'fixed_scalar', 'head_only', 'raw_router'):
        a = fuse(oh, ort, old_ids, reference_classes, weight=weight, floor=floor,
                 mode=mode, fixed_scale=fixed_scale)
        b = fuse(nh, nrt, new_ids, reference_classes, weight=weight, floor=floor,
                 mode=mode, fixed_scale=fixed_scale)
        before = winner(a, old_ids)
        restricted = winner(b[:, oldcols], old_ids)
        full = winner(b, new_ids)
        max_margin_change = 0.0
        for j in range(len(old_ids)):
            for k in range(j):
                max_margin_change = max(max_margin_change, float(np.max(np.abs(
                    (a[:, j] - a[:, k]) - (b[:, oldcols[j]] - b[:, oldcols[k]])), initial=0)))
        row = {'old_restricted_winner_changes': int(np.count_nonzero(before != restricted)),
               'full_winner_changes': int(np.count_nonzero(before != full)),
               'new_class_winners': int(np.count_nonzero(~np.isin(full, old_ids))),
               'max_old_pair_margin_change': max_margin_change}
        if mode == 'reference':
            row['conditional_invariance_checked'] = unchanged
            row['conditional_invariance_pass'] = (max_margin_change <= tolerance) if unchanged else None
        result['controls'][mode] = row
    return result


def bound_array(root, item):
    require(set(item) == {'path', 'sha256'}, 'array binding fields')
    require(type(item['path']) is str and item['path'] and ':' not in item['path'] and '\\' not in item['path'], 'portable relative path required')
    root = no_links(root)
    relative = Path(item['path'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'array path escape')
    path = no_links(root / relative)
    require(path.is_file() and not path.is_symlink(), 'missing or linked array')
    require(root.resolve() in path.resolve().parents, 'array outside manifest root')
    current = path
    while current != root:
        require(not current.is_symlink(), 'linked parent')
        current = current.parent
    require(sha(path) == item['sha256'], 'array SHA mismatch')
    a = np.load(path, mmap_mode='r', allow_pickle=False)
    require(a.dtype.kind in 'iufUS', 'unsupported array type')
    return a


def execute(manifest, output):
    manifest = no_links(manifest).resolve()
    manifest_hash = sha(manifest)
    p = strict_json(manifest)
    require(p['schema'] == 'expansion-fusion-pilot-v1', 'unknown manifest')
    require(p['independent_binary_head_probabilities'] is True, 'incorrect head score semantics')
    require(p['reference_origin'] == 'task0_train_only_frozen_router_bank', 'reference provenance')
    require(p['status'] == 'BOUND', 'unbound manifest')
    require(p['evidence_kind'] in ('synthetic', 'real'), 'explicit score evidence kind required')
    for key in ('reference_state_sha256', 'encoder_state_sha256', 'feature_contract_sha256', 'score_export_receipt_sha256'):
        require(type(p[key]) is str and len(p[key]) == 64 and all(c in '0123456789abcdef' for c in p[key]), 'missing digest')
    require(type(p['block_rows']) is int and 0 < p['block_rows'] <= 4096, 'unbounded block')
    root = manifest.parent
    names = ('old_head', 'old_raw', 'new_head', 'new_raw', 'old_row_ids', 'new_row_ids')
    require(set(p['arrays']) == set(names), 'array closure')
    arrays = {n: bound_array(root, p['arrays'][n]) for n in names}
    n = len(arrays['old_head'])
    require(n > 0 and all(len(a) == n for a in arrays.values()), 'row length mismatch')
    require(arrays['old_row_ids'].ndim == arrays['new_row_ids'].ndim == 1, 'row ID dimensions')
    require(arrays['old_row_ids'].dtype.kind in 'US' and arrays['new_row_ids'].dtype.kind in 'US', 'row ID type')
    require(np.array_equal(arrays['old_row_ids'], arrays['new_row_ids']), 'row alignment mismatch')
    ids = arrays['old_row_ids'].tolist()
    require(all(ids) and len(set(ids)) == len(ids), 'duplicate or empty rows')
    output = no_links(output)
    require(not output.exists() and not output.is_symlink(), 'refuse output overwrite')
    require(output.resolve() != root and root not in output.resolve().parents, 'output must be outside input root')
    require(not any(parent.is_symlink() for parent in output.parents), 'linked output parent')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {'schema': p['schema'], 'manifest_sha256': manifest_hash, 'status': 'RUNNING',
              'evidence_kind': p['evidence_kind'],
              'scientific_novelty_proven': False, 'historical_fidelity_pass': False,
              'classification_performance_measured': False, 'input_provenance_verified_beyond_supplied_files': False,
              'role': 'score-level mechanism pilot, not training or deployment', 'rows': 0,
              'numpy': np.__version__, 'python': platform.python_version(), 'controls': {}}
    try:
        frozen = True
        for start in range(0, n, p['block_rows']):
            stop = min(start + p['block_rows'], n)
            row = check_transition(*(arrays[k][start:stop] for k in names[:4]),
                p['old_classes'], p['new_classes'], p['reference_classes'],
                weight=p['weight'], floor=p['floor'], fixed_scale=p['fixed_scale'], tolerance=p['tolerance'])
            frozen = frozen and row['frozen_old_input_scores']
            report['rows'] += row['rows']
            for mode, values in row['controls'].items():
                out = report['controls'].setdefault(mode, {})
                for key in ('old_restricted_winner_changes', 'full_winner_changes', 'new_class_winners'):
                    out[key] = out.get(key, 0) + values[key]
                key = 'max_old_pair_margin_change'
                out[key] = max(out.get(key, 0.0), values[key])
        report['frozen_old_input_scores'] = frozen
        delta = report['controls']['reference']['max_old_pair_margin_change']
        report['conditional_invariance_pass'] = delta <= p['tolerance'] if frozen else None
        for name in names:
            require(sha(root / p['arrays'][name]['path']) == p['arrays'][name]['sha256'], 'input changed during scoring')
        require(sha(manifest) == manifest_hash, 'manifest changed during scoring')
        report['status'] = 'SCORE_PILOT_COMPLETE'
    except Exception as exc:
        report['status'] = 'FAILED'
        report['error_type'] = type(exc).__name__
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter() - started
        with (output / 'RESULT.json').open('x', encoding='utf-8', newline='\n') as f:
            json.dump(report, f, indent=2, allow_nan=False)
            f.write('\n')
        if report['status'] == 'SCORE_PILOT_COMPLETE':
            with (output / 'COMPLETE.json').open('x', encoding='utf-8', newline='\n') as f:
                json.dump({'result_sha256': sha(output / 'RESULT.json'), 'manifest_sha256': manifest_hash}, f)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(execute(a.manifest, a.output), indent=2))


if __name__ == '__main__':
    main()
