"""Tests for the §489 benign exposure cap and stable row roles."""
from types import SimpleNamespace
import unittest

import numpy as np

from ncd.run_aacil import select_rows
from ncd.section81_exposure import ERExposure


class BenignExposureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        labels = np.repeat(np.arange(3), (62000, 256, 256))
        cls.data = SimpleNamespace(
            class_order=["Normal", "A", "B"], y_train=labels,
            x_train=np.arange(len(labels), dtype=np.float32).reshape(-1, 1),
            tasks=[[0, 1], [2]],
        )

    def test_registered_caps_are_shared_by_both_roles(self):
        for cap in (5000, 10000, 20000, 50000):
            exposure = ERExposure(self.data, 1, normal_training_rows=cap)
            for task in (0, 1):
                with self.subTest(cap=cap, task=task):
                    slots = exposure(task)
                    normal = slots[0]
                    opt, stop = normal["train_ids"], normal["validation_ids"]
                    selected = select_rows(exposure.train[0], cap, (10000 + task) * 10007)
                    self.assertEqual(len(opt) + len(stop), cap)
                    np.testing.assert_array_equal(np.sort(np.r_[opt, stop]), np.sort(selected))
                    self.assertFalse(np.intersect1d(opt, stop).size)
                    self.assertFalse(np.intersect1d(selected, exposure.calibration_ids).size)
                    self.assertNotIn(1 if task else 2, slots)

    def test_default_keeps_legacy_ids_and_order(self):
        exposure = ERExposure(self.data, 1)
        for task in (0, 1):
            actual = exposure(task)[0]
            legacy = select_rows(exposure.train[0], 50000, (10000 + task) * 10007)
            stopping = np.isin(legacy, exposure.normal_stopping_ids)
            np.testing.assert_array_equal(actual["train_ids"], legacy[~stopping])
            np.testing.assert_array_equal(actual["validation_ids"], legacy[stopping])


if __name__ == "__main__":
    unittest.main()
