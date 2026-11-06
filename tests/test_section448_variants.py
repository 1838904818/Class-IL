from types import SimpleNamespace
import unittest

import numpy as np

from ncd.section448_variants import (
    KMEANS_N_INIT,
    _cluster_rows,
    cap_new_class_rows,
    refresh_diverse_attack_memory,
)


class TestSection448Variants(unittest.TestCase):
    def test_kmeans_n_init_is_recorded_engineering_default(self):
        self.assertEqual(KMEANS_N_INIT, "auto")

    def test_caps_are_deterministic_and_v1b_uses_four_times_quota(self):
        ids = np.arange(40, dtype=np.int64)
        a, meta = cap_new_class_rows(ids, 8, "V1a", 17)
        b, _ = cap_new_class_rows(ids, 8, "V1a", 17)
        c, cmeta = cap_new_class_rows(ids, 8, "V1b", 17)
        self.assertEqual(len(a), 8)
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(meta["cap_rows"], 8)
        self.assertEqual(len(c), 32)
        self.assertEqual(cmeta["cap_rows"], 32)

    def test_kmeans_selects_unique_observed_rows_deterministically(self):
        rng = np.random.default_rng(41)
        x = np.concatenate([rng.normal(loc=i * 5, scale=.15, size=(20, 3)) for i in range(4)])
        ids = np.arange(100, 180, dtype=np.int64)
        picked, meta = _cluster_rows(ids, x, 4, 13)
        again, _ = _cluster_rows(ids, x, 4, 13)
        self.assertEqual(len(picked), len(np.unique(picked)))
        self.assertEqual(len(picked), 4)
        self.assertTrue(set(picked).issubset(set(ids)))
        self.assertTrue(np.array_equal(picked, again))
        self.assertEqual(meta["clusters"], 4)

    def test_diverse_memory_shrinks_old_rows_without_rereading_and_adds_new_class(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(160, 5)).astype(np.float32)
        old_train = np.arange(0, 30, dtype=np.int64)
        old_valid = np.arange(30, 35, dtype=np.int64)
        new_train = np.arange(40, 100, dtype=np.int64)
        new_valid = np.arange(100, 120, dtype=np.int64)
        old = {1: {"train_ids": old_train, "validation_ids": old_valid,
                   "train_x": x[old_train].copy(), "validation_x": x[old_valid].copy()}}
        data = SimpleNamespace(x_train=x)
        memory, details = refresh_diverse_attack_memory(
            data, {1: np.arange(0, 160), 2: new_train}, {1: np.arange(0, 160), 2: new_valid},
            old, {2}, {1: 20, 2: 20}, seed=3, task=1)
        self.assertEqual(len(memory[1]["train_ids"]), 15)
        self.assertEqual(len(memory[1]["validation_ids"]), 5)
        self.assertTrue(set(memory[1]["train_ids"]).issubset(set(old_train)))
        self.assertTrue(set(memory[1]["validation_ids"]).issubset(set(old_valid)))
        self.assertTrue(np.array_equal(memory[1]["train_x"], x[memory[1]["train_ids"]]))
        self.assertEqual(len(memory[2]["train_ids"]), 15)
        self.assertEqual(len(memory[2]["validation_ids"]), 5)
        self.assertTrue(set(memory[2]["train_ids"]).issubset(set(new_train)))
        self.assertEqual(details["1"]["source"], "previously retained TRAIN rows only")
        self.assertEqual(details["2"]["source"], "fresh current-class TRAIN rows")


if __name__ == "__main__":
    unittest.main()
