"""NSL extension entry regression tests; optional real prepared-cache checks."""
import ast
import hashlib
import json
import os
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NslEntrySupport(unittest.TestCase):
    def test_cross_drive_cache_path(self):
        from unittest.mock import patch
        from ncd.prepare_problem_scout_data import relative
        path = ROOT / "cache" / "data_quality.json"
        with patch("ncd.prepare_problem_scout_data.os.path.relpath", side_effect=ValueError):
            self.assertEqual(relative(path), path.resolve().as_posix())
        self.assertEqual(relative(ROOT / "cache"), "cache")
        from ncd.run_problem_scout import path_label
        with patch("ncd.run_problem_scout.os.path.relpath", side_effect=ValueError):
            self.assertEqual(path_label(path), path.resolve().as_posix())
        self.assertEqual(path_label(ROOT / "cache"), "cache")

    def test_both_entry_dataset_choices(self):
        for name in ("run_section165.py", "run_section123_fixed.py"):
            tree = ast.parse((ROOT / "ncd" / name).read_text(encoding="utf-8"))
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute)
                     and node.func.attr == "add_argument" and node.args
                     and isinstance(node.args[0], ast.Constant)
                     and node.args[0].value == "--dataset"]
            self.assertEqual(len(calls), 1)
            choices = next(k.value for k in calls[0].keywords if k.arg == "choices")
            self.assertEqual(ast.literal_eval(choices),
                             ("NSL-KDD", "UNSW-NB15", "CIC-IDS-2017", "CSE-CIC-IDS2018"))

    @unittest.skipUnless(os.environ.get("NSL_PREPARED_ROOT"), "real-cache root not supplied")
    def test_real_cache_and_permanent_holdout(self):
        import numpy as np
        from ncd.prepare_problem_scout_data import prepare_setting
        from ncd.run_problem_scout import section26_pools, smoke_subset
        data, shift_x, shift_y, _ = prepare_setting(
            "NSL-KDD", {"problem_scout_data_root": os.environ["NSL_PREPARED_ROOT"]})
        self.assertIsNone(shift_x)
        self.assertIsNone(shift_y)
        self.assertEqual(data.class_order, ["Normal", "DoS", "Probe", "R2L", "U2R"])
        self.assertEqual(data.tasks, [[0, 1], [2], [3], [4]])
        self.assertEqual(len(data.y_train), 125973)
        self.assertEqual(len(data.y_test), 22544)
        for seed in range(10, 15):
            train, valid = section26_pools(data, seed)
            again_train, again_valid = section26_pools(data, seed)
            normal = np.flatnonzero(data.y_train == 0)
            self.assertEqual(len(valid[0]), int(np.ceil(.1 * len(normal))))
            self.assertFalse(np.intersect1d(train[0], valid[0]).size)
            np.testing.assert_array_equal(np.sort(np.r_[train[0], valid[0]]), normal)
            np.testing.assert_array_equal(train[0], again_train[0])
            np.testing.assert_array_equal(valid[0], again_valid[0])
        smoke, _, _ = smoke_subset(data, None, None, limit=64)
        self.assertTrue(all(0 < n <= 64 for n in np.bincount(smoke.y_train)))
        self.assertEqual(smoke.tasks, data.tasks)

    @unittest.skipUnless(os.environ.get("NSL_SMOKE_ROOT"), "smoke artifacts not supplied")
    def test_seven_real_entry_smoke_outputs(self):
        root = Path(os.environ["NSL_SMOKE_ROOT"])
        cells = [(m, root / "smoke" / "nsl-kdd" / m / "seed1")
                 for m in ("R3-A", "R3-L")]
        cells += [(m, root / "fixed" / m)
                  for m in ("ar", "replay_bs", "freq", "wacb", "derpp")]
        for method, folder in cells:
            with self.subTest(method=method):
                summary = json.loads((folder / "summary.json").read_text())
                self.assertTrue(summary["completed"])
                self.assertFalse(summary["scientific_result"])
                self.assertEqual(summary["dataset"], "NSL-KDD")
                result = json.loads((folder / "cell" / "result.json").read_text())
                self.assertTrue(result["completed"])
                self.assertEqual(len(result["checkpoints"]), 4)
                for cp in result["checkpoints"]:
                    path = folder / "cell" / cp["checkpoint_file"]
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                                     cp["checkpoint_sha256"])
                if method in ("freq", "wacb", "derpp"):
                    default = json.loads((folder / "default_stages.json").read_text())
                    calibrated = json.loads((folder / "calibrated_stages.json").read_text())
                    self.assertEqual(len(default), 4)
                    self.assertEqual(len(calibrated), 4)
                    hashes = set()
                    for stage in calibrated:
                        self.assertEqual(set(stage["operating_points"]), {"0.01", "0.02", "0.05"})
                        hashes.add(stage["calibration"]["calibration_indices_sha256"])
                        for point in stage["operating_points"].values():
                            self.assertEqual(len(point["confusion_matrix"]), 5)
                    self.assertEqual(len(hashes), 1)


if __name__ == "__main__":
    unittest.main()
