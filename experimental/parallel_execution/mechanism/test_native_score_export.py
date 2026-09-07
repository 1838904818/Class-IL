from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import native_score_export as e
import anchor_fusion as a
import prepare_fusion
import evaluate_fusion


def fixture(root):
    root.mkdir()
    torch.manual_seed(14)
    arch = {'encoder_type': 'mlp', 'd_model': 4, 'n_layers': 1, 'lora_rank': 2, 'lora_alpha': 4.0}
    seedmeta = {'feature_dim': 3, 'architecture': arch}
    encoder = e.build_bound_encoder(seedmeta)
    heads = {c: e.r.FamilyHead(4, 2, 4.0) for c in (0, 1, 2)}
    refs = {}
    def ref(path):
        return {'path': path.name, 'sha256': e.r.sha(path)}
    for role, cp, classes in (('reference', 0, [0, 1]), ('old', 1, [0, 1]), ('new', 2, [0, 1, 2])):
        states = {'mean': np.zeros(3), 'scale': np.ones(3)}
        schema = {'encoder': {}, 'heads': {}, 'normalization': {'mean': 'mean', 'scale': 'scale'}, 'cap3000_router': {}}
        for k, v in encoder.state_dict().items():
            schema['encoder'][k] = 'encoder_' + k
            states['encoder_' + k] = v.numpy()
        for c in classes:
            mapping = {}
            for k, v in heads[c].state_dict().items():
                mapping[k] = f'head_{c}_{k}'
                states[mapping[k]] = v.numpy()
            schema['heads'][str(c)] = mapping
            key = f'centroid_{c}'
            states[key] = np.full((2, 4), 0.3 * c, dtype=np.float32)
            schema['cap3000_router'][str(c)] = {'centroids': key}
        statefile = root / f'{role}-state.npz'
        np.savez(statefile, **states)
        metadata = {**seedmeta, 'checkpoint': cp, 'seen_classes': classes, 'dataset': 'synthetic-only', 'seed': 14,
                    'state_schema': schema, 'inference_state_sha256': e.r.sha(statefile)}
        metafile = root / f'{role}-metadata.json'
        e.r.atomic_json(metafile, metadata)
        refs[role] = {'metadata': ref(metafile), 'state': ref(statefile)}
    rng = np.random.default_rng(9)
    arrays = {'raw_features': rng.normal(size=(11, 3)).astype(np.float32),
              'row_ids': np.array([f'row-{i}' for i in range(11)]),
              'group_ids': np.array([f'group-{i // 2}' for i in range(11)])}
    binding = {}
    for key, v in arrays.items():
        path = root / f'{key}.npy'
        np.save(path, v, allow_pickle=False)
        binding[key] = ref(path)
    manifest = root / 'input.json'
    e.r.atomic_json(manifest, {'schema': 'native-ofra-score-pair-v1', 'evidence_kind': 'synthetic', 'batch_rows': 4,
                              'checkpoints': refs, 'arrays': binding})
    return manifest


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = fixture(self.root / 'input')

    def tearDown(self):
        self.temp.cleanup()

    def test_actual_tensor_forward_to_mechanism(self):
        out = self.root / 'out'
        report = e.run(self.manifest, out)
        self.assertTrue(report['fixed_reference_state_eligible'])
        self.assertFalse(report['historical_gpu_fidelity_verified'])
        self.assertFalse(report['training_lineage_verified'])
        p = np.load(out / 'new_head.npy')
        margin = np.load(out / 'new_head_logits.npy')
        self.assertLess(np.max(np.abs(p - 1 / (1 + np.exp(-margin)))), 2e-7)
        v = a.check_transition(*(np.load(out / f'{s}.npy') for s in ('old_head', 'old_raw', 'new_head', 'new_raw')),
                               ['0', '1'], ['0', '1', '2'], ['0', '1'], weight=.5, floor=.01, fixed_scale=1., tolerance=1e-12)
        self.assertTrue(v['controls']['reference']['conditional_invariance_pass'])
        self.assertTrue((out / 'COMPLETE.json').exists())

    def test_raw_affinity_is_distance_not_z(self):
        spec = e.r.read_json(self.manifest)
        model = e.load_bound(self.manifest.parent, spec['checkpoints']['old'], None, torch.device('cpu'))
        raw = np.load(self.manifest.parent / 'raw_features.npy')[:3]
        _, _, affinity = e.block_scores(model, raw)
        with torch.no_grad():
            emb = model['encoder'](torch.from_numpy(raw)).numpy()
        expected = np.column_stack([-np.linalg.norm(emb - model['centers'][c][0], axis=1) for c in (0, 1)])
        np.testing.assert_allclose(affinity, expected, atol=1e-6)
        self.assertTrue((affinity <= 0).all())

    def test_hash_tamper_blocks(self):
        with (self.manifest.parent / 'raw_features.npy').open('ab') as f:
            f.write(b'tamper')
        with self.assertRaises(ValueError):
            e.run(self.manifest, self.root / 'out')

    def test_login_node_blocks_before_output(self):
        with patch.object(e.socket, 'gethostname', return_value='login01'):
            with self.assertRaises(ValueError):
                e.run(self.manifest, self.root / 'out')
        self.assertFalse((self.root / 'out').exists())

    def test_no_output_overwrite(self):
        e.run(self.manifest, self.root / 'out')
        with self.assertRaises(ValueError):
            e.run(self.manifest, self.root / 'out')

    def test_inference_failure_leaves_no_complete(self):
        with patch.object(e, 'block_scores', side_effect=RuntimeError('injected')):
            with self.assertRaises(RuntimeError):
                e.run(self.manifest, self.root / 'out')
        self.assertTrue((self.root / 'out' / 'FAILED.json').exists())
        self.assertFalse((self.root / 'out' / 'COMPLETE.json').exists())

    def test_ref_state_mismatch_cannot_claim_eligibility(self):
        spec = e.r.read_json(self.manifest)
        meta = self.manifest.parent / spec['checkpoints']['new']['metadata']['path']
        statepath = self.manifest.parent / spec['checkpoints']['new']['state']['path']
        with np.load(statepath, allow_pickle=False) as z:
            state = {k: z[k].copy() for k in z.files}
        state['centroid_0'] += 1
        np.savez(statepath, **state)
        metadata = e.r.read_json(meta)
        metadata['inference_state_sha256'] = e.r.sha(statepath)
        e.r.atomic_json(meta, metadata)
        spec['checkpoints']['new']['metadata']['sha256'] = e.r.sha(meta)
        spec['checkpoints']['new']['state']['sha256'] = e.r.sha(statepath)
        e.r.atomic_json(self.manifest, spec)
        report = e.run(self.manifest, self.root / 'out')
        self.assertFalse(report['fixed_reference_state_eligible'])

    def test_row_identity_duplicates_block(self):
        spec = e.r.read_json(self.manifest)
        path = self.manifest.parent / spec['arrays']['row_ids']['path']
        ids = np.load(path)
        ids[1] = ids[0]
        np.save(path, ids)
        spec['arrays']['row_ids']['sha256'] = e.r.sha(path)
        e.r.atomic_json(self.manifest, spec)
        with self.assertRaises(ValueError):
            e.run(self.manifest, self.root / 'out')

    def fusion_pipeline(self):
        out = self.root / 'export'
        e.run(self.manifest, out)
        policy = self.root / 'policy.json'
        e.r.atomic_json(policy, {'schema': 'fixed-fusion-controls-v1', 'weight': .5, 'floor': .01, 'fixed_scale': 1.,
                                'tolerance': 1e-12, 'block_rows': 4,
                                'selection_origin': 'development_only_frozen_before_evaluation'})
        spec = prepare_fusion.prepare(out, policy, self.root / 'scores')
        pilot = self.root / 'pilot'
        a.execute(spec, pilot)
        label_values = self.root / 'labels.npy'
        np.save(label_values, np.array([str(i % 3) for i in range(11)]))
        row_ids = self.root / 'label-row-ids.npy'
        np.save(row_ids, np.load(out / 'old_row_ids.npy'))
        labels = self.root / 'labels.json'
        e.r.atomic_json(labels, {'schema': 'fusion-evaluation-labels-v1',
                                'labels': {'path': label_values.name, 'sha256': e.r.sha(label_values)},
                                'row_ids': {'path': row_ids.name, 'sha256': e.r.sha(row_ids)},
                                'score_export_receipt_sha256': a.strict_json(spec)['score_export_receipt_sha256']})
        return spec, pilot, labels

    def test_full_export_prepare_pilot_evaluate(self):
        spec, pilot, labels = self.fusion_pipeline()
        report = evaluate_fusion.evaluate(spec, pilot, labels, e.r.sha(labels), self.root / 'evaluation', evidence_kind='synthetic')
        self.assertEqual(set(report['controls']), set(evaluate_fusion.MODES))
        for row in report['controls'].values():
            final = row['new_checkpoint_all_classes']
            self.assertEqual(np.sum(final['confusion_counts']), 11)
            self.assertEqual(set(final['per_class']), {'0', '1', '2'})
        self.assertFalse(report['full_stream_forgetting_computed'])

    def test_no_labels_before_completed_pilot(self):
        spec, pilot, labels = self.fusion_pipeline()
        with (pilot / 'RESULT.json').open('ab') as f:
            f.write(b' ')
        with self.assertRaises(ValueError):
            evaluate_fusion.evaluate(spec, pilot, labels, e.r.sha(labels), self.root / 'eval', evidence_kind='synthetic')

    def test_evaluation_rejects_label_hash_drift(self):
        spec, pilot, labels = self.fusion_pipeline()
        with self.assertRaises(ValueError):
            evaluate_fusion.evaluate(spec, pilot, labels, '0' * 64, self.root / 'eval', evidence_kind='synthetic')

    def test_evaluation_rejects_missing_class(self):
        spec, pilot, labels = self.fusion_pipeline()
        descriptor = a.strict_json(labels)
        values = labels.parent / descriptor['labels']['path']
        np.save(values, np.array(['0'] * 11))
        descriptor['labels']['sha256'] = a.sha(values)
        e.r.atomic_json(labels, descriptor)
        with self.assertRaises(ValueError):
            evaluate_fusion.evaluate(spec, pilot, labels, e.r.sha(labels), self.root / 'eval', evidence_kind='synthetic')

    def test_evaluation_rejects_reordered_label_identities(self):
        spec, pilot, labels = self.fusion_pipeline()
        descriptor = a.strict_json(labels)
        rows = labels.parent / descriptor['row_ids']['path']
        values = np.load(rows)
        np.save(rows, values[::-1])
        descriptor['row_ids']['sha256'] = a.sha(rows)
        e.r.atomic_json(labels, descriptor)
        with self.assertRaises(ValueError):
            evaluate_fusion.evaluate(spec, pilot, labels, e.r.sha(labels), self.root / 'eval', evidence_kind='synthetic')

    def test_evaluation_rejects_label_source_mismatch(self):
        spec, pilot, labels = self.fusion_pipeline()
        descriptor = a.strict_json(labels)
        descriptor['score_export_receipt_sha256'] = 'a' * 64
        e.r.atomic_json(labels, descriptor)
        with self.assertRaises(ValueError):
            evaluate_fusion.evaluate(spec, pilot, labels, e.r.sha(labels), self.root / 'eval', evidence_kind='synthetic')

    def test_evaluation_metrics_known_confusion(self):
        m = evaluate_fusion.metrics(np.array([[8, 2], [1, 9]]), ['0', '1'])
        self.assertAlmostEqual(m['accuracy'], .85)
        self.assertAlmostEqual(m['balanced_accuracy'], .85)
        self.assertAlmostEqual(m['per_class']['0']['recall'], .8)

    def test_synthetic_cannot_be_relabelled_as_real(self):
        spec, pilot, labels = self.fusion_pipeline()
        with self.assertRaises(ValueError):
            evaluate_fusion.evaluate(spec, pilot, labels, e.r.sha(labels), self.root / 'eval', evidence_kind='prospective-score-evaluation')

    def test_implicit_checkpoint_dtype_conversion_forbidden(self):
        for key in ('centroid_0', 'encoder_feat.0.weight'):
            with self.subTest(key=key):
                spec = e.r.read_json(self.manifest)
                binding = spec['checkpoints']['old']
                statepath = self.manifest.parent / binding['state']['path']
                metapath = self.manifest.parent / binding['metadata']['path']
                with np.load(statepath, allow_pickle=False) as z:
                    state = {k: z[k].copy() for k in z.files}
                original = state[key].copy()
                state[key] = state[key].astype(np.float64)
                np.savez(statepath, **state)
                metadata = e.r.read_json(metapath)
                metadata['inference_state_sha256'] = e.r.sha(statepath)
                e.r.atomic_json(metapath, metadata)
                binding['state']['sha256'] = e.r.sha(statepath)
                binding['metadata']['sha256'] = e.r.sha(metapath)
                with self.assertRaises(ValueError):
                    e.load_bound(self.manifest.parent, binding, None, torch.device('cpu'))
                state[key] = original
                np.savez(statepath, **state)


if __name__ == '__main__':
    unittest.main()
