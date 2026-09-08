import copy
import unittest
import checkpoint_metrics as m


class PayloadTests(unittest.TestCase):
    def fixture(self):
        return dict(task=0, class_axis=[0, 2], confusion=[[3, 1], [1, 3]],
                    accuracy=.75, macro_f1=.75, balanced_accuracy=.75,
                    mean_signed_forgetting=None, signed_forgetting=[],
                    per_class=[dict(class_id=c, support=4, precision=.75, recall=.75, f1=.75) for c in (0, 2)])

    def payload(self, value):
        return m.checkpoint_payload(value, arm='ce', checkpoint_sha256='a' * 64, history=[])

    def test_allowlist_and_negative_forgetting(self):
        value = self.fixture()
        value['private_path'] = 'must not leave process'
        payload = self.payload(value)
        self.assertNotIn('private_path', str(payload))
        self.assertNotIn('ce/mean_signed_recall_forgetting', payload['metrics'])
        self.assertEqual(len(payload['confusion']['data']), 4)

    def test_corrupt_metrics_rejected(self):
        for key, bad in [('accuracy', .9), ('macro_f1', float('nan')), ('balanced_accuracy', .2), ('task', True)]:
            value = self.fixture()
            value[key] = bad
            with self.assertRaises(ValueError):
                self.payload(value)

    def test_bad_class_row(self):
        for key, bad in [('support', 5), ('recall', .5), ('precision', .9), ('f1', .1)]:
            value = self.fixture()
            value['per_class'][0][key] = bad
            with self.assertRaises(ValueError):
                self.payload(value)

    def test_initial_no_forgetting(self):
        value = self.fixture()
        value['mean_signed_forgetting'] = None
        self.assertNotIn('ce/mean_signed_recall_forgetting', self.payload(value)['metrics'])

    def test_identity_required(self):
        with self.assertRaises(ValueError):
            m.checkpoint_payload(self.fixture(), arm='router', checkpoint_sha256='a' * 64, history=[])
        with self.assertRaises(ValueError):
            m.checkpoint_payload(self.fixture(), arm='ce', checkpoint_sha256='unknown', history=[])

    def test_tight_tolerance(self):
        value = self.fixture()
        value['accuracy'] += 5e-10
        with self.assertRaises(ValueError):
            self.payload(value)

    def test_zero_support(self):
        value = self.fixture()
        value['confusion'][1] = [0, 0]
        with self.assertRaises(ValueError):
            self.payload(value)

    def test_history_and_forged_forgetting(self):
        prior = self.fixture()
        current = self.fixture()
        current.update(task=1, mean_signed_forgetting=0.0,
                       signed_forgetting=[dict(class_id=c, signed_recall_forgetting=0.0) for c in (0, 2)])
        kwargs = dict(arm='ce', checkpoint_sha256='a'*64, history=[prior])
        self.assertEqual(m.checkpoint_payload(current, **kwargs)['metrics']['ce/mean_signed_recall_forgetting'], 0.0)
        current['mean_signed_forgetting'] = .99
        with self.assertRaises(ValueError):
            m.checkpoint_payload(current, **kwargs)
        current['mean_signed_forgetting'] = 0.0
        current['signed_forgetting'][0]['signed_recall_forgetting'] = .1
        with self.assertRaises(ValueError):
            m.checkpoint_payload(current, **kwargs)
        with self.assertRaises(ValueError):
            m.checkpoint_payload(current, **{**kwargs, 'history':[]})


if __name__ == '__main__':
    unittest.main()
