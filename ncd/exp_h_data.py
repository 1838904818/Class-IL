"""Fixed data views for work-order 17; no fitting on shifted observations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ncd.data import (
    PreparedData, NSL_COLUMNS, CACHE_DATASETS, _load_raw, _sample_shards,
    _cap_for, make_tasks,
)

ROOT = Path(__file__).resolve().parents[1]


def _path(value):
    return (ROOT / value).resolve()


def _transform(train, y_train, test, shifted, categorical, tasks):
    numeric = [c for c in train.columns if c not in categorical]
    frames = [f.copy() for f in (train, test, shifted)]
    for frame in frames:
        frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan)
    transformer = ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')),
                              ('scaler', StandardScaler())]), numeric),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False,
                                     dtype=np.float32), categorical),
    ], remainder='drop', verbose_feature_names_out=True)
    transformer.fit(frames[0].loc[np.isin(y_train, tasks[0])])
    arrays = [np.asarray(transformer.transform(f), dtype=np.float32) for f in frames]
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError('Nonfinite transformed Exp H data')
    return (*arrays, list(transformer.get_feature_names_out()))


def _counts(y, classes):
    return {name: int(np.sum(y == c)) for c, name in enumerate(classes)}


def prepare_nsl(config):
    root = _path(config['data_root'])
    train, yt, test, ye, classes, categorical, files = _load_raw('NSL-KDD', root)
    raw_test = pd.read_csv(root / 'nsl-kdd/KDDTest+.txt', header=None, names=NSL_COLUMNS)
    difficulty = raw_test['difficulty'].to_numpy()
    if not np.isin(difficulty, np.arange(22)).all():
        raise ValueError('Invalid official NSL difficulty field')
    # UNB defines KDDTest-21 as KDDTest+ excluding difficulty==21.
    hard = difficulty < 21
    if len(test) != 22544 or int(hard.sum()) != 11850:
        raise ValueError('NSL official Test+/Test-21 row counts differ')
    tasks = make_tasks(len(classes), 1)
    xt, xe, xs, names = _transform(train, yt, test, test.loc[hard], categorical, tasks)
    data = PreparedData(xt, yt, xe, ye, classes, tasks, names, files)
    metadata = {
        'setting': 'S3', 'name': 'NSL-KDD to KDDTest-21',
        'dataset': 'NSL-KDD', 'problem_type': 'intrusion_detection',
        'source': 'https://www.unb.ca/cic/datasets/nsl.html',
        'shift_construction': 'Official definition: KDDTest+ rows with difficulty < 21',
        'train_counts': _counts(yt, classes), 'id_counts': _counts(ye, classes),
        'shift_counts': _counts(ye[hard], classes),
        'preprocessing': 'Exactly Section15: fit numeric/categorical transform on Task0 training only',
        'caveats': ['KDDTest-21 is a nested harder subset of KDDTest+, not an independent domain. '
                    'Primary Z uses the original full KDDTest+ to preserve Section15 comparability.'],
        'feature_alignment': [{'setting': 'S3', 'training_feature': n,
                               'shift_feature': n, 'action': 'identity',
                               'notes': 'Same Task0-fitted transform'} for n in train.columns],
    }
    return data, xs, ye[hard].copy(), metadata


# CICFlowMeter long names (2017) -> abbreviated names (2018), exact and auditable.
CIC_ALIASES = {
    'Destination Port': 'Dst Port', 'Total Fwd Packets': 'Tot Fwd Pkts',
    'Total Backward Packets': 'Tot Bwd Pkts',
    'Total Length of Fwd Packets': 'TotLen Fwd Pkts',
    'Total Length of Bwd Packets': 'TotLen Bwd Pkts',
    'Flow Bytes/s': 'Flow Byts/s', 'Flow Packets/s': 'Flow Pkts/s',
    'Fwd IAT Total': 'Fwd IAT Tot', 'Bwd IAT Total': 'Bwd IAT Tot',
    'Fwd Header Length': 'Fwd Header Len', 'Bwd Header Length': 'Bwd Header Len',
    'Fwd Packets/s': 'Fwd Pkts/s', 'Bwd Packets/s': 'Bwd Pkts/s',
    'Min Packet Length': 'Pkt Len Min', 'Max Packet Length': 'Pkt Len Max',
    'Packet Length Mean': 'Pkt Len Mean', 'Packet Length Std': 'Pkt Len Std',
    'Packet Length Variance': 'Pkt Len Var', 'Average Packet Size': 'Pkt Size Avg',
    'Avg Fwd Segment Size': 'Fwd Seg Size Avg', 'Avg Bwd Segment Size': 'Bwd Seg Size Avg',
    'Fwd Avg Bytes/Bulk': 'Fwd Byts/b Avg', 'Fwd Avg Packets/Bulk': 'Fwd Pkts/b Avg',
    'Fwd Avg Bulk Rate': 'Fwd Blk Rate Avg', 'Bwd Avg Bytes/Bulk': 'Bwd Byts/b Avg',
    'Bwd Avg Packets/Bulk': 'Bwd Pkts/b Avg', 'Bwd Avg Bulk Rate': 'Bwd Blk Rate Avg',
    'Subflow Fwd Packets': 'Subflow Fwd Pkts', 'Subflow Fwd Bytes': 'Subflow Fwd Byts',
    'Subflow Bwd Packets': 'Subflow Bwd Pkts', 'Subflow Bwd Bytes': 'Subflow Bwd Byts',
    'Init_Win_bytes_forward': 'Init Fwd Win Byts',
    'Init_Win_bytes_backward': 'Init Bwd Win Byts',
    'act_data_pkt_fwd': 'Fwd Act Data Pkts', 'min_seg_size_forward': 'Fwd Seg Size Min',
}
for _direction in ('Fwd', 'Bwd'):
    for _stat in ('Max', 'Min', 'Mean', 'Std'):
        CIC_ALIASES[f'{_direction} Packet Length {_stat}'] = f'{_direction} Pkt Len {_stat}'
for _flag in ('FIN', 'SYN', 'RST', 'PSH', 'ACK', 'URG', 'ECE'):
    CIC_ALIASES[f'{_flag} Flag Count'] = f'{_flag} Flag Cnt'


def prepare_cic(config):
    root = _path(config['cic_cache_root'])
    source = root / 'cic-ids-2017'
    target = root / 'cic-ids-2018'
    schemas = [json.loads((p / 'feature_schema.json').read_text(encoding='utf-8')) for p in (source, target)]
    old_names, new_names = [s['feature_columns'] for s in schemas]
    kept = [n for n in old_names if n != 'Fwd Header Length.1']
    mapped = [CIC_ALIASES.get(n, n) for n in kept]
    if len(kept) != 77 or len(set(mapped)) != 77 or set(mapped) != set(new_names) - {'Protocol'}:
        raise ValueError('CIC feature alignment is incomplete or ambiguous')
    source_indices = [old_names.index(n) for n in kept]
    target_indices = [new_names.index(n) for n in mapped]
    classes = list(CACHE_DATASETS['CIC-IDS-2017']['class_order'])
    arrays, labels = {}, {}
    for split in ('train', 'test', 'shift'):
        xs, ys = [], []
        specification = CACHE_DATASETS['CIC-IDS-2017' if split != 'shift' else 'CSE-CIC-IDS2018']
        for c, name in enumerate(classes):
            if split == 'shift' and name not in schemas[1]['class_order']:
                continue
            directory = (target / 'test' if split == 'shift' else source / split) / name
            configured_cap = specification['train_cap' if split == 'train' else 'test_cap']
            values = _sample_shards(sorted(directory.glob('part-*.npy')),
                                   _cap_for(configured_cap, name),
                                   42000 + (0 if split == 'train' else 1000) + c)
            xs.append(values[:, target_indices if split == 'shift' else source_indices])
            ys.append(np.full(len(values), c, dtype=np.int64))
        arrays[split], labels[split] = np.concatenate(xs), np.concatenate(ys)
    tasks = make_tasks(len(classes), 2)
    xt, xe, xs, names = _transform(
        pd.DataFrame(arrays['train'], columns=kept), labels['train'],
        pd.DataFrame(arrays['test'], columns=kept), pd.DataFrame(arrays['shift'], columns=kept),
        [], tasks,
    )
    data = PreparedData(xt, labels['train'], xe, labels['test'], classes, tasks, names,
                        [str(source / 'feature_schema.json'), str(target / 'feature_schema.json')])
    metadata = {
        'setting': 'S1', 'name': 'CIC-IDS-2017 to CSE-CIC-IDS2018 shared families',
        'dataset': 'CIC-IDS-2017', 'problem_type': 'intrusion_detection',
        'train_counts': _counts(labels['train'], classes),
        'id_counts': _counts(labels['test'], classes), 'shift_counts': _counts(labels['shift'], classes),
        'preprocessing': 'Raw-valued frozen caches; only source Task0 trains imputer/scaler; 77 named aligned features',
        'sampling': 'Existing NCD deterministic source caps (Normal train160000/test40000); target test10000/class maximum',
        'caveats': ['PortScan has no target family and is excluded from shift units; retained as a training class.',
                    'CICFlowMeter versions and capture-generation differences remain potential confounds.'],
        'feature_alignment': [dict(setting='S1', training_feature=a, shift_feature=b,
                                    action='aligned', notes='CICFlowMeter semantic name')
                              for a, b in zip(kept, mapped)] + [
            dict(setting='S1', training_feature='Fwd Header Length.1', shift_feature='',
                 action='drop', notes='Duplicate source header feature, absent in target'),
            dict(setting='S1', training_feature='', shift_feature='Protocol',
                 action='drop', notes='Target-only feature'),
        ],
    }
    return data, xs, labels['shift'], metadata


def prepare_setting(setting, config):
    if setting == 'S3':
        return prepare_nsl(config)
    if setting == 'S1':
        return prepare_cic(config)
    if setting == 'S2':
        from ncd.exp_h_malaya import prepare_malaya
        return prepare_malaya(config)
    raise ValueError(f'Unknown Exp H setting: {setting}')
