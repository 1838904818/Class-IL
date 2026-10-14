"""Data-only checks for Section21; no model fitting or historical writes."""
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd

from ncd.prepare_problem_scout_data import (clean_unsw, numeric_overlap, RawCacheView,
                                          save_dataset, prepare_setting, TABULAR_CLASSES)


class CleanDataTests(unittest.TestCase):
    def test_conflict_all_members_and_global_duplicates(self):
        frame = pd.DataFrame({"number": np.arange(40, dtype=float),
                              "category": ["a"] * 20 + ["b"] * 20})
        y = np.array([0] * 20 + [1] * 20)
        # Two same-label duplicates, and a contradictory third member of key0.
        train = pd.concat([frame.iloc[:30], frame.iloc[[0, 1]]], ignore_index=True)
        yt = np.concatenate([y[:30], [0, 0]])
        test = pd.concat([frame.iloc[30:], frame.iloc[[0, 1]]], ignore_index=True)
        ye = np.concatenate([y[30:], [1, 0]])
        tr, a, te, b, q = clean_unsw(train, yt, test, ye)
        self.assertEqual(q["raw_rows"], 44)
        self.assertEqual(q["conflict_groups_removed"], 1)
        self.assertEqual(q["conflict_member_rows_removed"], 3)
        self.assertEqual(q["same_label_excess_rows_removed"], 2)
        self.assertEqual(q["retained_rows"], 39)
        self.assertFalse((tr["number"] == 0).any())
        self.assertFalse((te["number"] == 0).any())
        self.assertEqual(len(tr) + len(te), 39)
        self.assertEqual(set(a) | set(b), {0, 1})

    def test_numeric_equality_signed_zero_and_fullbytes(self):
        a = np.array([[0., 1.], [1., 2.], [1., 3.]], dtype=np.float32)
        b = np.array([[-0., 1.], [1., 4.], [1., 3.]], dtype=np.float32)
        q = numeric_overlap(a, b, 2)
        self.assertEqual(q["test_matching_any_train_rows"], 2)

    def test_raw_projection_across_shards(self):
        with TemporaryDirectory() as temp:
            p = Path(temp)
            a = np.arange(24, dtype=np.float32).reshape(8, 3)
            np.save(p / "a.npy", a[:4]); np.save(p / "b.npy", a[4:])
            view = RawCacheView([(p / "a.npy", 0, np.array([1, 3]), 2),
                                 (p / "b.npy", 1, None, 4)], [2, 0])
            expected = a[[1, 3, 4, 5, 6, 7]][:, [2, 0]]
            np.testing.assert_array_equal(view[:], expected)
            np.testing.assert_array_equal(view[1:5], expected[1:5])
            np.testing.assert_array_equal(view[4], expected[4])
            del view

    def test_loader_requires_complete_unique_hash_receipt(self):
        with TemporaryDirectory() as temp:
            p = Path(temp) / "nsl-kdd"
            x, y = np.arange(10, dtype=np.float32).reshape(5, 2), np.arange(5, dtype=np.int64)
            info = save_dataset(p, "NSL-KDD", x, y, x, y,
                                TABULAR_CLASSES["NSL-KDD"], [[0, 1], [2], [3], [4]],
                                ["a", "b"], {})
            data, _, _, _ = prepare_setting("NSL-KDD", {"problem_scout_data_root": temp})
            np.testing.assert_array_equal(data.y_train, y)
            del data
            for files in (info["cache_files"][:-1], info["cache_files"] + [info["cache_files"][0]]):
                changed = dict(info, cache_files=files)
                (p / "data_quality.json").write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "exact expected arrays"):
                    prepare_setting("NSL-KDD", {"problem_scout_data_root": temp})

    def test_loader_rejects_array_mutation(self):
        with TemporaryDirectory() as temp:
            p = Path(temp) / "nsl-kdd"
            x, y = np.arange(10, dtype=np.float32).reshape(5, 2), np.arange(5, dtype=np.int64)
            save_dataset(p, "NSL-KDD", x, y, x, y,
                         TABULAR_CLASSES["NSL-KDD"], [[0, 1], [2], [3], [4]], ["a", "b"], {})
            np.save(p / "train_y.npy", np.zeros(5, dtype=np.int64))
            with self.assertRaisesRegex(ValueError, "hash/size differs"):
                prepare_setting("NSL-KDD", {"problem_scout_data_root": temp})


if __name__ == "__main__":
    unittest.main()
