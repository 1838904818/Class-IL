"""Section 19 NetFlow-v2 preparation: frozen mapping, global clean, then split.

The 43-column release schema includes four flow identifiers. As in the
repository's reference loader, models and exact equality use the other 39
numeric columns. Preparation is CPU-only, streaming and separate from training.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
import zipfile

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ncd.data import PreparedData, make_tasks

IDENTIFIERS = ['IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT']
FEATURES = IDENTIFIERS + [
    'PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'IN_PKTS', 'OUT_BYTES', 'OUT_PKTS',
    'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS',
    'FLOW_DURATION_MILLISECONDS', 'DURATION_IN', 'DURATION_OUT', 'MIN_TTL',
    'MAX_TTL', 'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT', 'MIN_IP_PKT_LEN',
    'MAX_IP_PKT_LEN', 'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES',
    'RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_IN_PKTS',
    'RETRANSMITTED_OUT_BYTES', 'RETRANSMITTED_OUT_PKTS',
    'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT',
    'NUM_PKTS_UP_TO_128_BYTES', 'NUM_PKTS_128_TO_256_BYTES',
    'NUM_PKTS_256_TO_512_BYTES', 'NUM_PKTS_512_TO_1024_BYTES',
    'NUM_PKTS_1024_TO_1514_BYTES', 'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT',
    'ICMP_TYPE', 'ICMP_IPV4_TYPE', 'DNS_QUERY_ID', 'DNS_QUERY_TYPE',
    'DNS_TTL_ANSWER', 'FTP_COMMAND_RET_CODE',
]
MODEL_FEATURES = FEATURES[4:]
CLASSES = ['Normal', 'DoS', 'DDoS', 'CredentialAttack', 'WebAttack']
DOMAINS = ('ton', 'cic')


def _token(label):
    return re.sub(r'[^a-z0-9]', '', str(label).lower())


TON_LABELS = {
    'benign': 'Normal', 'dos': 'DoS', 'ddos': 'DDoS',
    'password': 'CredentialAttack', 'injection': 'WebAttack',
    'xss': 'WebAttack', 'backdoor': 'excluded:ton:Backdoor',
    'scanning': 'excluded:ton:Scanning', 'mitm': 'excluded:ton:MITM',
    'ransomware': 'excluded:ton:Ransomware',
}
CIC_LABELS = {
    'benign': 'Normal', 'dos': 'DoS', 'ddos': 'DDoS',
    'bruteforce': 'CredentialAttack', 'ftpbruteforce': 'CredentialAttack',
    'sshbruteforce': 'CredentialAttack',
    'dosattackshulk': 'DoS', 'dosattacksgoldeneye': 'DoS',
    'dosattacksslowhttptest': 'DoS', 'dosattacksslowloris': 'DoS',
    'ddosattackhoic': 'DDoS', 'ddosattackloicudp': 'DDoS',
    'ddosattacksloichttp': 'DDoS',
    'webattacks': 'WebAttack', 'webattack': 'WebAttack',
    'bruteforceweb': 'WebAttack', 'bruteforcexss': 'WebAttack',
    'sqlinjection': 'WebAttack',
    'bot': 'excluded:cic:Bot', 'infiltration': 'excluded:cic:Infiltration',
    'infilteration': 'excluded:cic:Infiltration',
}
FROZEN_SPEC = {
    'version': 'section19-netflow-v1',
    'source_schema_features': FEATURES,
    'model_and_equality_features': MODEL_FEATURES,
    'excluded_identifiers': IDENTIFIERS,
    'class_order': CLASSES,
    'tasks': [[0, 1], [2, 3], [4]],
    'normalized_label_mapping': {'ton': TON_LABELS, 'cic': CIC_LABELS},
    'label_normalization': 'lowercase, remove non-alphanumeric characters',
    'equality': 'complete little-endian finite float32 model vector; negative zero equals zero',
    'conflict_rule': 'drop entire feature group with multiple mapped labels, including excluded labels',
    'cross_source_rule': 'drop entire feature group present in both domains, including same-label groups',
    'within_source_rule': 'retain one vector per remaining feature group',
    'split_rule': 'feature-hash group split seed42: SplitMix64(FNV1a32words XOR42) modulo5 ==0 is test',
    'test_fraction': 0.2,
    'train_cap_per_class': 40000,
    'train_sampling': 'smallest independent feature-hash priorities seed42001 after dedup and split',
    'test_cap': None,
    'oversampling': False,
    'preprocessing': 'StandardScaler fit only on source Task0 training rows; no target fitting',
    'semantic_limitations': [
        'CredentialAttack is a broad parent: ToN password includes brute force or sniffing, not identical to CSE brute force.',
        'WebAttack is a broad parent: attack implementations and subtype mixtures differ across domains.',
        'Backdoor is not mapped to Infiltration; Scanning has no shared target category.',
        'WebAttack is the final task and has no old-class drift unit.',
    ],
}


def spec_sha256():
    payload = json.dumps(FROZEN_SPEC, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_features(values):
    """Canonical model representation; reject invalid rows rather than impute."""
    x64 = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(x64).all(axis=1) & (np.abs(x64) <= np.finfo(np.float32).max).all(axis=1)
    x = np.array(x64[valid], dtype='<f4', order='C', copy=True)
    x[x == 0] = 0.0
    return x, valid


def hash_rows(x):
    """Routing/split hash only; duplicate membership always checks all bytes."""
    words = np.ascontiguousarray(x, dtype='<f4').view('<u4').reshape(len(x), x.shape[1])
    h = np.full(len(x), np.uint64(14695981039346656037), dtype=np.uint64)
    with np.errstate(over='ignore'):
        for column in words.T:
            h = (h ^ column.astype(np.uint64)) * np.uint64(1099511628211)
        h = (h ^ np.uint64(x.shape[1])) * np.uint64(1099511628211)
    return h


def mix64(h):
    with np.errstate(over='ignore'):
        h = (np.asarray(h, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15))
        h = (h ^ (h >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return h ^ (h >> np.uint64(31))


def clean_bucket(x, labels, domains, weights=None):
    """Exact equality, conflict removal, and symmetric cross-domain exclusion."""
    x = np.ascontiguousarray(x, dtype='<f4')
    key = x.view(np.dtype((np.void, x.shape[1] * 4))).ravel()
    _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    if weights is None:
        weights = np.ones(len(x), dtype=np.uint64)
    weights = np.asarray(weights, dtype=np.uint64)
    sizes = np.zeros(len(first), dtype=np.uint64)
    np.add.at(sizes, inverse, weights)
    lo = np.full(len(first), 255, dtype=np.uint8)
    hi = np.zeros(len(first), dtype=np.uint8)
    dlo = np.full(len(first), 255, dtype=np.uint8)
    dhi = np.zeros(len(first), dtype=np.uint8)
    np.minimum.at(lo, inverse, labels)
    np.maximum.at(hi, inverse, labels)
    np.minimum.at(dlo, inverse, domains)
    np.maximum.at(dhi, inverse, domains)
    conflict = lo != hi
    cross = dlo != dhi
    keep_group = ~(conflict | cross)
    kept = first[keep_group]
    stat = {
        'input_rows': int(weights.sum()), 'feature_groups': len(first),
        'conflict_groups': int(conflict.sum()), 'conflict_member_rows': int(sizes[conflict].sum()),
        'cross_source_groups': int(cross.sum()), 'cross_source_member_rows': int(sizes[cross].sum()),
        'conflict_or_cross_groups': int((conflict | cross).sum()),
        'conflict_or_cross_member_rows': int(sizes[conflict | cross].sum()),
        'within_source_excess_removed': int((sizes[keep_group] - 1).sum()),
        'retained_unique_all_labels': len(kept),
    }
    return x[kept], labels[kept], domains[kept], stat


def _smallest(x, priorities, cap):
    if len(x) <= cap:
        return x, priorities
    threshold = np.partition(priorities, cap - 1)[cap - 1]
    candidate = np.flatnonzero(priorities <= threshold)
    # A full-byte secondary order makes even hash-priority collisions stable.
    keys = np.ascontiguousarray(x[candidate]).view(np.dtype((np.void, x.shape[1] * 4))).ravel()
    order = np.lexsort((keys, priorities[candidate]))[:cap]
    chosen = candidate[order]
    return x[chosen], priorities[chosen]


@contextmanager
def csv_stream(path):
    """Read the data member directly from the official multi-member ZIP."""
    path = Path(path)
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as archive:
            members = [n for n in archive.namelist()
                       if Path(n).name == 'NF-CSE-CIC-IDS2018-v2.csv']
            if len(members) != 1:
                raise ValueError('ZIP must contain exactly one official NF-CSE-CIC-IDS2018-v2.csv')
            with archive.open(members[0]) as stream:
                yield stream
    else:
        with path.open('rb') as stream:
            yield stream


def _counts(y):
    return {name: int(np.sum(y == i)) for i, name in enumerate(CLASSES)}


def _validate_header(path):
    with csv_stream(path) as stream:
        columns = list(pd.read_csv(stream, nrows=0).columns)
    expected = set(FEATURES) | {'Label', 'Attack'}
    if len(columns) != len(expected) or set(columns) != expected:
        raise ValueError(f'Unexpected NetFlow-v2 schema in {Path(path).name}: {columns}')


def prepare_cache(inputs, output_dir, frozen_report, *, chunksize=50000, buckets=512):
    """Run only after the mapping hash is in the parent-owned frozen report."""
    report = Path(frozen_report)
    digest = spec_sha256()
    if digest not in report.read_text(encoding='utf-8'):
        raise ValueError('Frozen report must contain the exact mapping/specification SHA256 before preparation')
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Refusing to overwrite a populated preparation directory')
    out.mkdir(parents=True, exist_ok=True)
    if not 16 <= buckets <= 4096 or chunksize <= 0:
        raise ValueError('Unsafe partition or chunk count')
    files = {d: [Path(p).resolve() for p in inputs[d]] for d in DOMAINS}
    if not all(files.values()):
        raise ValueError('Both domains require input CSV files')
    for paths in files.values():
        for path in paths:
            _validate_header(path)
    # Conservative CSV-to-binary estimate, plus holdouts and operational margin.
    source_bytes = 0
    for paths in files.values():
        for path in paths:
            if path.suffix.lower() == '.zip':
                with zipfile.ZipFile(path) as archive:
                    source_bytes += sum(i.file_size for i in archive.infolist()
                                        if Path(i.filename).name == 'NF-CSE-CIC-IDS2018-v2.csv')
            else:
                source_bytes += path.stat().st_size
    required = int(source_bytes * 1.3 + 2e9)
    if shutil.disk_usage(out).free < required:
        raise OSError(f'Insufficient free disk for bounded preparation: need {required} bytes')
    start = time.time()
    temp = out / 'partitions'
    temp.mkdir()
    # Keep the shared class integer labels exactly 0..4.
    labels_all = CLASSES + sorted((set(TON_LABELS.values()) | set(CIC_LABELS.values())) - set(CLASSES))
    label_to_id = {v: i for i, v in enumerate(labels_all)}
    dtype = np.dtype([('x', '<f4', (39,)), ('label', 'u1'), ('domain', 'u1'), ('count', '<u8')])
    inputs_receipt, raw_counts, invalid_counts = [], {}, {}
    total_rows = 0
    for domain_id, domain in enumerate(DOMAINS):
        mapping = FROZEN_SPEC['normalized_label_mapping'][domain]
        raw_counts[domain], invalid_counts[domain] = Counter(), Counter()
        for path in files[domain]:
            print(f'Input {domain}/{path.name}', flush=True)
            inputs_receipt.append({'domain': domain, 'file': path.name, 'bytes': path.stat().st_size,
                                   'sha256': file_sha256(path)})
            with csv_stream(path) as stream:
                reader = pd.read_csv(stream, usecols=MODEL_FEATURES + ['Attack', 'Label'],
                                     chunksize=chunksize, low_memory=False)
                for frame in reader:
                    _partition_frame(frame, path, domain, domain_id, mapping, raw_counts,
                                     invalid_counts, label_to_id, dtype, temp, buckets)
                    total_rows += len(frame)
                    if total_rows % 1000000 < chunksize:
                        print(f'partitioned {total_rows:,} raw rows', flush=True)
    return _finish_cache(out, temp, report, digest, start, inputs_receipt, raw_counts,
                         invalid_counts, labels_all, buckets, chunksize, dtype)


def _partition_frame(frame, path, domain, domain_id, mapping, raw_counts,
                     invalid_counts, label_to_id, dtype, temp, buckets):
    raw = frame['Attack'].astype(str).str.strip()
    raw_counts[domain].update(raw.value_counts().to_dict())
    normalized = raw.map(_token)
    unknown = set(normalized) - set(mapping)
    if unknown:
        raise ValueError(f'Unfrozen labels in {path.name}: {sorted(unknown)}')
    mapped = normalized.map(mapping)
    # Labels are never features. Check the released binary labels agree.
    binary = pd.to_numeric(frame['Label'], errors='coerce').to_numpy()
    if not np.array_equal(binary, (mapped != 'Normal').to_numpy(dtype=np.int64)):
        raise ValueError(f'Binary/multiclass label disagreement in {path.name}')
    numeric = frame[MODEL_FEATURES].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)
    x, valid = canonical_features(numeric)
    invalid_counts[domain].update(raw[~valid].value_counts().to_dict())
    label = mapped[valid].map(label_to_id).to_numpy(dtype=np.uint8)
    record = np.empty(len(x), dtype=dtype)
    record['x'], record['label'], record['domain'] = x, label, domain_id
    record['count'] = 1
    # Lossless per-chunk run counting prevents massive repeated keys from
    # overflowing one partition. No feature/label/source group is discarded;
    # downstream conflict/member accounting uses these multiplicities.
    key = record.view(np.dtype((np.void, dtype.itemsize))).ravel()
    _, first, multiplicity = np.unique(key, return_index=True, return_counts=True)
    record = record[first]
    record['count'] = multiplicity
    routes = (mix64(hash_rows(record['x'])) % np.uint64(buckets)).astype(np.int64)
    for b in np.unique(routes):
        with (temp / f'{b:04d}.bin').open('ab') as stream:
            record[routes == b].tofile(stream)


def _finish_cache(out, temp, report, digest, start, inputs_receipt, raw_counts,
                  invalid_counts, labels_all, buckets, chunksize, dtype):
    total_stat = Counter()
    domain_stat = {d: {'clean_counts': Counter(), 'train_before_cap': Counter(), 'test_uncapped': Counter()} for d in DOMAINS}
    train = {(d, c): (np.empty((0, 39), '<f4'), np.empty(0, np.uint64)) for d in DOMAINS for c in range(5)}
    test_spools = {d: out / f'{d}_test.bin' for d in DOMAINS}
    test_dtype = np.dtype([('x', '<f4', (39,)), ('y', 'u1')])
    for b in range(buckets):
        path = temp / f'{b:04d}.bin'
        if not path.exists():
            continue
        if path.stat().st_size > 256 * 1024 * 1024:
            raise MemoryError(f'Partition {b} exceeds 256MiB safety bound; inspect key-frequency concentration')
        record = np.fromfile(path, dtype=dtype)
        x, y, domain_ids, stats = clean_bucket(record['x'], record['label'], record['domain'], record['count'])
        total_stat.update(stats)
        shared = y < len(CLASSES)
        total_stat['excluded_unique_after_clean'] += int((~shared).sum())
        x, y, domain_ids = x[shared], y[shared], domain_ids[shared]
        h = hash_rows(x)
        is_test = mix64(h ^ np.uint64(42)) % np.uint64(5) == 0
        priorities = mix64(h ^ np.uint64(42001))
        for domain_id, domain in enumerate(DOMAINS):
            domain_mask = domain_ids == domain_id
            for c, name in enumerate(CLASSES):
                belongs = domain_mask & (y == c)
                domain_stat[domain]['clean_counts'][name] += int(belongs.sum())
                tr = belongs & ~is_test
                te = belongs & is_test
                domain_stat[domain]['train_before_cap'][name] += int(tr.sum())
                domain_stat[domain]['test_uncapped'][name] += int(te.sum())
                old_x, old_p = train[(domain, c)]
                train[(domain, c)] = _smallest(np.concatenate([old_x, x[tr]]),
                                              np.concatenate([old_p, priorities[tr]]), 40000)
            te = domain_mask & is_test
            if te.any():
                result = np.empty(int(te.sum()), dtype=test_dtype)
                result['x'], result['y'] = x[te], y[te]
                with test_spools[domain].open('ab') as stream:
                    result.tofile(stream)
        # Only this call's newly created partition is removed; no raw input touched.
        path.unlink()
        if b % 32 == 0:
            print(f'cleaned partition {b}/{buckets}; retained {total_stat["retained_unique_all_labels"]:,}', flush=True)
    cache_files = []
    for domain in DOMAINS:
        folder = out / domain
        folder.mkdir()
        xt = np.concatenate([train[(domain, c)][0] for c in range(5)])
        yt = np.concatenate([np.full(len(train[(domain, c)][0]), c, np.int64) for c in range(5)])
        test_count = sum(domain_stat[domain]['test_uncapped'].values())
        if any(domain_stat[domain]['test_uncapped'][n] == 0 or not len(train[(domain, c)][0])
               for c, n in enumerate(CLASSES)):
            raise ValueError(f'A frozen class has an empty train/test partition in {domain}; do not modify mapping after results')
        np.save(folder / 'train_x.npy', xt, allow_pickle=False)
        np.save(folder / 'train_y.npy', yt, allow_pickle=False)
        xe = np.lib.format.open_memmap(folder / 'test_x.npy', mode='w+', dtype='<f4', shape=(test_count, 39))
        ye = np.lib.format.open_memmap(folder / 'test_y.npy', mode='w+', dtype='<i8', shape=(test_count,))
        offset = 0
        with test_spools[domain].open('rb') as stream:
            while len(part := np.fromfile(stream, dtype=test_dtype, count=chunksize)):
                xe[offset:offset + len(part)] = part['x']
                ye[offset:offset + len(part)] = part['y']
                offset += len(part)
        xe.flush()
        ye.flush()
        del xe, ye
        test_spools[domain].unlink()
        domain_stat[domain]['train_after_cap'] = _counts(yt)
        for path in sorted(folder.glob('*.npy')):
            cache_files.append({'file': path.relative_to(out).as_posix(), 'bytes': path.stat().st_size,
                                'sha256': file_sha256(path)})
    receipt = {
        'status': 'complete', 'frozen_mapping_sha256': digest, 'frozen_spec': FROZEN_SPEC,
        'freeze_report': report.name, 'freeze_report_sha256_at_preparation': file_sha256(report),
        'preparation_script_sha256': file_sha256(__file__),
        'inputs': inputs_receipt, 'raw_label_counts': raw_counts,
        'invalid_feature_rows_by_raw_label': invalid_counts,
        'global_cleaning': dict(total_stat), 'domains': domain_stat,
        'cache_files': cache_files, 'elapsed_seconds': time.time() - start,
        'limitations': FROZEN_SPEC['semantic_limitations'] + [
            'Equality is exact in cached float32 representation, not the original decimal strings.',
            'Hash partition approximates 80/20 per class; all holdout rows retained without balancing.',
            'Whole-group exclusions use both domains labels before training as prespecified data hygiene, not target parameter fitting.',
        ],
    }
    (out / 'preparation.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'complete', 'domains': domain_stat, 'global_cleaning': dict(total_stat)}), flush=True)
    return receipt


def prepare_setting(setting, config):
    if setting not in ('S4', 'S5'):
        raise ValueError('Only frozen settings S4 and S5 are supported')
    configured = config.get('cleaned_data_root', config.get('netflow_data_root'))
    if not configured:
        raise ValueError('cleaned_data_root is required')
    root = (ROOT / configured).resolve()
    receipt_path = root / 'preparation.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    expected = config.get('frozen_mapping_sha256')
    if expected != spec_sha256() or receipt.get('frozen_mapping_sha256') != expected:
        raise ValueError('Mapping hash differs from the frozen code/config/preparation')
    if receipt.get('status') != 'complete' or receipt.get('frozen_spec') != FROZEN_SPEC:
        raise ValueError('Preparation is incomplete or its specification changed')
    required = {f'{domain}/{split}_{key}.npy' for domain in DOMAINS
                for split in ('train', 'test') for key in ('x', 'y')}
    declared = [entry['file'] for entry in receipt['cache_files']]
    if len(declared) != 8 or len(set(declared)) != 8 or set(declared) != required:
        raise ValueError('Prepared cache receipt must declare exactly the eight required unique array paths')
    for entry in receipt['cache_files']:
        path = (root / entry['file']).resolve()
        if (root not in path.parents or path.stat().st_size != entry['bytes']
                or file_sha256(path) != entry['sha256']):
            raise ValueError('Prepared array path/hash mismatch')
    source, target = ('ton', 'cic') if setting == 'S4' else ('cic', 'ton')
    load = lambda d, split, key: np.load(root / d / f'{split}_{key}.npy', allow_pickle=False)
    xt, yt = load(source, 'train', 'x'), load(source, 'train', 'y')
    xe, ye = load(source, 'test', 'x'), load(source, 'test', 'y')
    xs, ys = load(target, 'test', 'x'), load(target, 'test', 'y')
    for features, labels in ((xt, yt), (xe, ye), (xs, ys)):
        if (features.ndim != 2 or features.shape[1] != 39 or labels.shape != (len(features),)
                or not np.issubdtype(labels.dtype, np.integer)
                or not np.isfinite(features).all() or not np.isin(labels, np.arange(5)).all()):
            raise ValueError('Invalid prepared feature or label shape/type/value')
    tasks = make_tasks(len(CLASSES), 2)
    scaler = StandardScaler().fit(xt[np.isin(yt, tasks[0])])
    transformed = [np.asarray(scaler.transform(x), dtype=np.float32) for x in (xt, xe, xs)]
    if not all(np.isfinite(x).all() for x in transformed):
        raise ValueError('Nonfinite transformed NetFlow data')
    data = PreparedData(transformed[0], yt, transformed[1], ye, list(CLASSES), tasks,
                        list(MODEL_FEATURES), [receipt_path.name])
    metadata = {
        'setting': setting, 'name': f'NF-{source} to NF-{target} v2 shared families',
        'dataset': 'NF-ToN-IoT-v2' if source == 'ton' else 'NF-CSE-CIC-IDS2018-v2',
        'problem_type': 'intrusion_detection', 'source_domain': source, 'target_domain': target,
        'class_order': CLASSES, 'tasks': tasks,
        'train_counts': _counts(yt), 'id_counts': _counts(ye), 'shift_counts': _counts(ys),
        'frozen_mapping_sha256': expected, 'preparation_sha256': file_sha256(receipt_path),
        'input_files': receipt['inputs'], 'cache_files': receipt['cache_files'],
        'global_cleaning': receipt['global_cleaning'], 'domain_cleaning': receipt['domains'],
        'preprocessing': FROZEN_SPEC['preprocessing'], 'caveats': receipt['limitations'],
        'feature_alignment': [
            {'setting': setting, 'training_feature': n, 'shift_feature': n,
             'action': 'identity', 'notes': 'Shared NetFlow-v2 numeric schema; no alias alignment'}
            for n in MODEL_FEATURES
        ] + [
            {'setting': setting, 'training_feature': n, 'shift_feature': n,
             'action': 'drop', 'notes': 'Network identifier excluded from both model and equality key'}
            for n in IDENTIFIERS
        ],
    }
    return data, transformed[2], ys, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--print-spec', action='store_true')
    parser.add_argument('--ton-csv', nargs='+')
    parser.add_argument('--cic-csv', nargs='+')
    parser.add_argument('--output-dir')
    parser.add_argument('--frozen-report')
    parser.add_argument('--chunksize', type=int, default=50000)
    parser.add_argument('--buckets', type=int, default=512)
    args = parser.parse_args()
    if args.print_spec:
        print(json.dumps({'frozen_mapping_sha256': spec_sha256(), 'frozen_spec': FROZEN_SPEC}, indent=2))
        return
    if not all((args.ton_csv, args.cic_csv, args.output_dir, args.frozen_report)):
        parser.error('Both input domains, output directory and frozen report are required')
    prepare_cache({'ton': args.ton_csv, 'cic': args.cic_csv}, args.output_dir, args.frozen_report,
                  chunksize=args.chunksize, buckets=args.buckets)


if __name__ == '__main__':
    main()
