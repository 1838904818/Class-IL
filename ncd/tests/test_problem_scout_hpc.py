"""CPU checks for the job boundary, not experimental evidence."""
import tempfile
import unittest
from pathlib import Path

from ncd.hpc.problem_scout_job import copy_small, under, sample, write_json


class ProblemScoutHpcTests(unittest.TestCase):
    def test_small_evidence_backup_excludes_arrays_weights_and_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, dest = root / "scratch", root / "home"
            source.mkdir()
            for name in ("result.json", "final_rows.csv", "run.log", "resources.jsonl", "weight.pt", "array.npy", ".active.lock"):
                (source / name).write_text("evidence")
            prior = {}
            self.assertEqual(copy_small(source, dest, prior), 4)
            self.assertEqual(copy_small(source, dest, prior), 0)
            self.assertEqual({p.name for p in dest.iterdir()}, {"result.json", "final_rows.csv", "run.log", "resources.jsonl"})

    def test_containment_and_atomic_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertTrue(under(root / "data", root))
            self.assertFalse(under(root.parent, root))
            write_json(root / "value.json", {"value": 3})
            self.assertFalse((root / "value.json.tmp").exists())
            self.assertIn('"value": 3', (root / "value.json").read_text())


if __name__ == "__main__":
    unittest.main()
