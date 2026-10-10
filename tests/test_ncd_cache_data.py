from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ncd.data import _load_frozen_cache, _sample_shards, prepare_dataset


class NcdCacheDataTest(unittest.TestCase):
    def test_shard_sampling_is_exact_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = []
            for index in range(3):
                path = root / f"part-{index:03d}.npy"
                np.save(path, np.arange(index * 20, (index + 1) * 20, dtype=np.float32).reshape(10, 2))
                files.append(path)
            first = _sample_shards(files, cap=11, seed=123)
            second = _sample_shards(files, cap=11, seed=123)
            self.assertEqual(first.shape, (11, 2))
            np.testing.assert_array_equal(first, second)

    def test_cic17_cache_contract_and_caps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            cache = data_root / "cic-ids-2017"
            classes = [
                "Normal", "DoS", "DDoS", "Bruteforce", "PortScan",
                "WebAttack", "Botnet", "Infiltration",
            ]
            (cache / "manifest.json").parent.mkdir(parents=True)
            (cache / "manifest.json").write_text(
                json.dumps({"dataset": "cic-ids-2017"}), encoding="utf-8"
            )
            (cache / "feature_schema.json").write_text(
                json.dumps(
                    {
                        "class_order": classes,
                        "feature_columns": ["a", "b", "c"],
                        "feature_count": 3,
                    }
                ),
                encoding="utf-8",
            )
            for split in ("train", "test"):
                for class_id, class_name in enumerate(classes):
                    target = cache / split / class_name
                    target.mkdir(parents=True)
                    values = np.full((4, 3), class_id + (0.5 if split == "test" else 0), dtype=np.float32)
                    np.save(target / "part-000.npy", values)
            train, y_train, test, y_test, order, categorical, sources = _load_frozen_cache(
                "CIC-IDS-2017", data_root
            )
            self.assertEqual(train.shape, (32, 3))
            self.assertEqual(test.shape, (32, 3))
            self.assertEqual(order, classes)
            self.assertEqual(categorical, [])
            self.assertEqual(np.bincount(y_train).tolist(), [4] * 8)
            self.assertEqual(np.bincount(y_test).tolist(), [4] * 8)
            self.assertEqual(len(sources), 2)

            prepared = prepare_dataset("CIC-IDS-2017", data_root)
            self.assertEqual(prepared.x_train.shape, (32, 3))
            self.assertEqual(prepared.x_test.shape, (32, 3))
            self.assertEqual(prepared.tasks, [[0, 1], [2, 3], [4, 5], [6, 7]])
            self.assertTrue(np.isfinite(prepared.x_train).all())
            self.assertTrue(np.isfinite(prepared.x_test).all())
            np.testing.assert_allclose(
                prepared.x_train[np.isin(prepared.y_train, prepared.tasks[0])].mean(axis=0),
                np.zeros(3),
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
