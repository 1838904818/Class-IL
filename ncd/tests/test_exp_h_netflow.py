"""Synthetic preparation tests only; these do not run a research model."""
import inspect
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd

from ncd import exp_h_netflow as nf


def test_schema_and_frozen_contract():
    assert len(nf.FEATURES) == len(set(nf.FEATURES)) == 43
    assert len(nf.MODEL_FEATURES) == 39
    assert not set(nf.IDENTIFIERS).intersection(nf.MODEL_FEATURES)
    assert nf.FROZEN_SPEC['test_cap'] is None
    assert nf.FROZEN_SPEC['tasks'] == [[0, 1], [2, 3], [4]]
    assert nf.TON_LABELS['password'] == 'CredentialAttack'
    assert nf.TON_LABELS['backdoor'] != nf.CIC_LABELS['infiltration']


def test_canonical_numeric_limits_and_signed_zero():
    x, valid = nf.canonical_features([[0., -0.], [np.nan, 1.], [np.inf, 1.], [1e40, 1.], [2., 3.]])
    assert valid.tolist() == [True, False, False, False, True]
    assert x.dtype == np.float32
    assert not np.signbit(x[0]).any()


def test_conflict_members_and_cross_domain_removal_are_not_pairs():
    x = np.asarray([[1], [1], [1], [2], [2], [3], [3], [4]], '<f4')
    labels = np.asarray([0, 0, 1, 0, 0, 0, 0, 2], 'u1')
    domains = np.asarray([0, 0, 0, 0, 1, 0, 0, 1], 'u1')
    kept, y, d, stat = nf.clean_bucket(x, labels, domains)
    assert set(kept.ravel()) == {3, 4}
    assert stat['conflict_member_rows'] == 3
    assert stat['cross_source_member_rows'] == 2
    assert stat['within_source_excess_removed'] == 1
    assert stat['input_rows'] == stat['conflict_or_cross_member_rows'] + stat['within_source_excess_removed'] + len(kept)


def test_excluded_label_can_conflict_with_shared_class():
    x = np.asarray([[1., 2.], [1., 2.], [3., 4.]], '<f4')
    kept, _, _, stat = nf.clean_bucket(x, np.asarray([0, 5, 0], 'u1'), np.zeros(3, 'u1'))
    assert kept.tolist() == [[3., 4.]]
    assert stat['conflict_member_rows'] == 2


def test_cross_source_conflicts_count_union_once():
    x = np.asarray([[1.], [1.], [1.]], '<f4')
    kept, _, _, stat = nf.clean_bucket(x, np.asarray([0, 1, 1], 'u1'), np.asarray([0, 1, 1], 'u1'))
    assert len(kept) == 0
    assert stat['conflict_member_rows'] == stat['cross_source_member_rows'] == 3
    assert stat['conflict_or_cross_member_rows'] == 3


def test_lossless_chunk_counts_preserve_all_member_statistics():
    x = np.asarray([[1.], [1.], [2.], [2.], [3.]], '<f4')
    labels = np.asarray([0, 0, 0, 1, 0], 'u1')
    domain = np.zeros(5, 'u1')
    weights = np.asarray([2000000, 3000000, 2, 3, 1], np.uint64)
    kept, _, _, stat = nf.clean_bucket(x, labels, domain, weights)
    assert set(kept.ravel()) == {1, 3}
    assert stat['input_rows'] == 5000006
    assert stat['conflict_member_rows'] == 5
    assert stat['within_source_excess_removed'] == 4999999


def test_bucket_equality_compares_complete_features():
    x = np.asarray([[1., 2.], [1., 3.], [1., 2.]], '<f4')
    kept, _, _, _ = nf.clean_bucket(x, np.zeros(3, 'u1'), np.zeros(3, 'u1'))
    assert len(kept) == 2


def test_deterministic_cap_with_hash_priority_ties():
    x = np.asarray([[9.], [4.], [7.], [2.]], '<f4')
    p = np.asarray([1, 1, 0, 2], np.uint64)
    a, pa = nf._smallest(x, p, 2)
    b, pb = nf._smallest(x[::-1], p[::-1], 2)
    assert np.array_equal(a, b)
    assert np.array_equal(pa, pb)


def test_empty_hash_rows_and_canonical_zero_hash():
    assert nf.hash_rows(np.empty((0, 39), '<f4')).shape == (0,)
    x, _ = nf.canonical_features([[0., -0.], [-0., 0.]])
    assert nf.hash_rows(x)[0] == nf.hash_rows(x)[1]


def test_preparation_refuses_missing_freeze(tmp_path):
    report = tmp_path / 'report.md'
    report.write_text('not frozen', encoding='utf-8')
    with unittest.TestCase().assertRaisesRegex(ValueError, 'Frozen report'):
        nf.prepare_cache({'ton': [], 'cic': []}, tmp_path / 'output', report)
    assert not (tmp_path / 'output').exists()


def test_unknown_raw_label_fails_instead_of_being_discarded(tmp_path):
    from collections import Counter
    frame = _fixture_frame('ton').iloc[:1].copy()
    frame['Attack'] = 'Unregistered new attack'
    with unittest.TestCase().assertRaisesRegex(ValueError, 'Unfrozen labels'):
        nf._partition_frame(frame, tmp_path / 'sample.csv', 'ton', 0, nf.TON_LABELS,
                            {'ton': Counter()}, {'ton': Counter()}, {}, None, tmp_path, 16)


def _fixture_frame(domain):
    attack = ['Benign', 'dos', 'ddos', 'password', 'xss'] if domain == 'ton' else [
        'Benign', 'DoS attacks-Hulk', 'DDOS attack-HOIC', 'FTP-BruteForce', 'SQL Injection']
    # Disjoint unique vectors; enough rows for all deterministic split cells.
    values = np.random.default_rng(40 + int(domain == 'cic')).uniform(0, 1000, (600, 39)).astype('<f4')
    frame = pd.DataFrame(values, columns=nf.MODEL_FEATURES)
    for c in nf.IDENTIFIERS:
        frame[c] = '192.0.2.1' if 'ADDR' in c else 42
    frame['Attack'] = [attack[i // 120] for i in range(600)]
    frame['Label'] = [int(i // 120 != 0) for i in range(600)]
    return frame[nf.FEATURES + ['Label', 'Attack']]


def test_synthetic_end_to_end_clean_cache_and_task0_transform(tmp_path):
    ton = tmp_path / 'ton.csv'
    _fixture_frame('ton').to_csv(ton, index=False)
    cic = tmp_path / 'cic.zip'
    with zipfile.ZipFile(cic, 'w') as archive:
        archive.writestr('source/data/NF-CSE-CIC-IDS2018-v2.csv', _fixture_frame('cic').to_csv(index=False))
        archive.writestr('source/data/NetFlow_v2_Features.csv', 'feature,description\n')
    report = tmp_path / 'report.md'
    report.write_text(nf.spec_sha256(), encoding='utf-8')
    output = tmp_path / 'prepared'
    receipt = nf.prepare_cache({'ton': [ton], 'cic': [cic]}, output, report, buckets=16, chunksize=137)
    assert receipt['status'] == 'complete'
    assert receipt['global_cleaning']['input_rows'] == 1200
    assert receipt['global_cleaning']['retained_unique_all_labels'] == 1200
    config = {'cleaned_data_root': str(output), 'frozen_mapping_sha256': nf.spec_sha256()}
    data, shifted, ys, meta = nf.prepare_setting('S4', config)
    assert data.tasks == [[0, 1], [2, 3], [4]]
    assert len(data.x_train) + len(data.x_test) == 600
    assert len(shifted) == sum(receipt['domains']['cic']['test_uncapped'].values())
    assert np.allclose(data.x_train[np.isin(data.y_train, [0, 1])].mean(axis=0), 0, atol=1e-6)
    assert len(meta['feature_alignment']) == 43
    for d in nf.DOMAINS:
        assert sum(receipt['domains'][d]['train_before_cap'].values()) + sum(receipt['domains'][d]['test_uncapped'].values()) == 600
    receipt_path = output / 'preparation.json'
    # A receipt omission/duplicate cannot remove an array from validation.
    damaged = dict(receipt)
    damaged['cache_files'] = [e for e in receipt['cache_files'] if e['file'] != 'ton/train_y.npy']
    receipt_path.write_text(json.dumps(damaged), encoding='utf-8')
    with unittest.TestCase().assertRaisesRegex(ValueError, 'eight required unique'):
        nf.prepare_setting('S4', config)
    damaged['cache_files'].append(damaged['cache_files'][0])
    receipt_path.write_text(json.dumps(damaged), encoding='utf-8')
    with unittest.TestCase().assertRaisesRegex(ValueError, 'eight required unique'):
        nf.prepare_setting('S4', config)
    receipt_path.write_text(json.dumps(receipt), encoding='utf-8')
    # Mutated data must not silently bypass the prepared-data binding.
    with (output / 'ton/train_y.npy').open('ab') as stream:
        stream.write(b'x')
    with unittest.TestCase().assertRaisesRegex(ValueError, 'hash mismatch'):
        nf.prepare_setting('S4', config)


def _run_with_temp(function):
    with tempfile.TemporaryDirectory() as directory:
        function(Path(directory))


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name, function in sorted(globals().items()):
        if name.startswith('test_') and callable(function):
            if inspect.signature(function).parameters:
                suite.addTest(unittest.FunctionTestCase(lambda f=function: _run_with_temp(f), description=name))
            else:
                suite.addTest(unittest.FunctionTestCase(function))
    return suite


if __name__ == '__main__':
    unittest.main()
