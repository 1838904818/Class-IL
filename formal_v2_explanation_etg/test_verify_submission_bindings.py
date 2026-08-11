from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from verify_submission_bindings import _manifest_shard_paths, confined_file


class SubmissionBindingContainmentTests(unittest.TestCase):
    def test_confined_file_rejects_file_outside_authorized_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            allowed = base / "owned"
            allowed.mkdir()
            inside = allowed / "inside.txt"
            outside = base / "outside.txt"
            inside.write_text("inside", encoding="utf-8")
            outside.write_text("outside", encoding="utf-8")
            self.assertEqual(confined_file(inside, (allowed.resolve(),), "inside"), inside.resolve())
            with self.assertRaisesRegex(RuntimeError, "escapes"):
                confined_file(outside, (allowed.resolve(),), "outside")

    def test_manifest_path_escape_is_rejected_before_dataset_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            allowed = base / "owned"
            data = allowed / "dataset"
            data.mkdir(parents=True)
            outside = base / "outside.npy"
            outside.write_bytes(b"not opened by this test")
            manifest = data / "streaming_manifest.json"
            manifest.write_text(json.dumps({
                "classes": [{
                    "train": [{"path": "../../outside.npy"}],
                    "test": [{"path": "../../outside.npy"}],
                }]
            }), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "escapes"):
                _manifest_shard_paths(manifest, (allowed.resolve(),))


if __name__ == "__main__":
    unittest.main()
