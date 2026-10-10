from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ncd.data import PreparedData
from ncd.validate_cache_data import summarize_prepared


class NcdValidateCacheDataTest(unittest.TestCase):
    def test_summary_reports_shapes_counts_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            prepared = PreparedData(
                x_train=np.array([[-1.0], [1.0], [2.0]], dtype=np.float32),
                y_train=np.array([0, 0, 1], dtype=np.int64),
                x_test=np.array([[0.0], [3.0]], dtype=np.float32),
                y_test=np.array([0, 1], dtype=np.int64),
                class_order=["Normal", "Attack"],
                tasks=[[0], [1]],
                feature_names=["numeric__f0"],
                source_files=[str(manifest)],
            )
            summary = summarize_prepared("synthetic-contract", prepared)
            self.assertEqual(summary["train_shape"], [3, 1])
            self.assertEqual(summary["train_class_counts"], {"Normal": 2, "Attack": 1})
            self.assertEqual(summary["task0_feature_mean_max_abs"], 0.0)
            self.assertEqual(len(summary["source_files"][0]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
