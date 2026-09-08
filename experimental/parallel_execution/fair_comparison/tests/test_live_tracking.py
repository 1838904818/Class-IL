import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from live_tracking import Tracker


class Fake:
    def Settings(self, **kwargs):
        return kwargs

    def init(self, **kwargs):
        self.arguments = kwargs
        self.id, self.entity, self.project = (kwargs[k] for k in ('id','entity','project'))
        self.payloads = []
        return self

    def Table(self, **kwargs):
        return kwargs

    def log(self, payload):
        if getattr(self, 'fail', False):
            raise RuntimeError('simulated transport failure')
        self.payloads.append(payload)

    def finish(self, **kwargs):
        if getattr(self, 'fail_finish', False):
            raise RuntimeError('simulated finish failure')
        self.finished = kwargs


class TrackingTests(unittest.TestCase):
    def fixture(self):
        return dict(task=0, class_axis=[0], confusion=[[2]], accuracy=1., macro_f1=1.,
                    balanced_accuracy=1., signed_forgetting=[], mean_signed_forgetting=None,
                    per_class=[dict(class_id=0,support=2,precision=1.,recall=1.,f1=1.)])

    def setup_tracker(self, root):
        sdk = Fake()
        policy = dict(entity='example',project='research',allow_online=True,aggregate_only=True)
        return Tracker(Path(root)/'sender',policy,'a'*32,sdk), sdk

    def test_payload_and_duplicate(self):
        with tempfile.TemporaryDirectory() as root:
            tracker, sdk = self.setup_tracker(root)
            args = dict(arm='ce',checkpoint_sha256='b'*64,history=[])
            tracker.send(self.fixture(), **args)
            self.assertEqual(len(sdk.payloads),1)
            with self.assertRaises(ValueError):
                tracker.send(self.fixture(), **args)
            tracker.finish()
            self.assertFalse(tracker.state['remote_verified'])
            self.assertEqual(tracker.state['status'],'client_finished_remote_unverified')

    def test_uncertain_send_not_retried(self):
        with tempfile.TemporaryDirectory() as root:
            tracker, sdk = self.setup_tracker(root)
            sdk.fail = True
            args = dict(arm='ce',checkpoint_sha256='b'*64,history=[])
            with self.assertRaises(RuntimeError):
                tracker.send(self.fixture(), **args)
            state=json.loads((tracker.directory/'DELIVERY.json').read_text())
            self.assertEqual(state['status'],'send_uncertain')
            sdk.fail = False
            with self.assertRaises(RuntimeError):
                tracker.send(self.fixture(), **args)
            with self.assertRaises(FileExistsError):
                self.setup_tracker(root)
            tracker.finish()
            self.assertEqual(sdk.finished['exit_code'],1)
            self.assertEqual(tracker.state['status'],'send_uncertain')

    def test_finish_failure_persisted(self):
        with tempfile.TemporaryDirectory() as root:
            tracker, sdk = self.setup_tracker(root)
            sdk.fail_finish = True
            with self.assertRaises(RuntimeError):
                tracker.finish()
            saved = json.loads((tracker.directory/'DELIVERY.json').read_text())
            self.assertEqual(saved['status'],'finish_uncertain')
            self.assertEqual(saved['pre_finish_status'],'ready')
            self.assertFalse(saved['remote_verified'])
            with self.assertRaises(RuntimeError):
                tracker.finish()

    def test_post_send_disk_failure_closes_sender(self):
        import live_tracking
        with tempfile.TemporaryDirectory() as root:
            tracker, sdk = self.setup_tracker(root)
            original = live_tracking.r.atomic_json
            def write(path, value):
                if value['status'] == 'ready':
                    raise OSError('synthetic disk failure')
                return original(path, value)
            args = dict(arm='ce',checkpoint_sha256='b'*64,history=[])
            with patch.object(live_tracking.r, 'atomic_json', side_effect=write):
                with self.assertRaises(OSError):
                    tracker.send(self.fixture(), **args)
            self.assertEqual(len(sdk.payloads),1)
            self.assertTrue(tracker.closed)
            self.assertEqual(tracker.state['status'],'send_uncertain')
            self.assertEqual(json.loads((tracker.directory/'DELIVERY.json').read_text())['status'],'send_uncertain')
            with self.assertRaises(RuntimeError):
                tracker.send(self.fixture(), **args)


if __name__ == '__main__':
    unittest.main()
