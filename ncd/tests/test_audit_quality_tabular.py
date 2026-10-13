"""Exact equality must not merge close floats; conflict scopes exclude labels."""

import unittest

import numpy as np
import pandas as pd

from ncd.audit_quality_tabular import conflict_flags, exact_feature_groups


class ExactTabularAuditTests(unittest.TestCase):
    def test_signed_zero_only_and_exact_nearby_float(self):
        train = pd.DataFrame({"x": [0.0, 1.0, np.nextafter(1.0, 2.0)], "category": ["tcp", "a", "a"]})
        test = pd.DataFrame({"x": [-0.0, 1.0, 1.0], "category": ["tcp", "a", "b"]})
        groups, count = exact_feature_groups(train, test)
        self.assertEqual(count, 4)
        self.assertEqual(groups[0], groups[3])
        self.assertEqual(groups[1], groups[4])
        self.assertNotEqual(groups[1], groups[2])
        self.assertNotEqual(groups[1], groups[5])
        flags, conflict_groups = conflict_flags(groups, np.array(["a", "a", "b", "a", "b", "b"]), count)
        self.assertEqual(flags.tolist(), [False, True, False, False, True, False])
        self.assertEqual(int(conflict_groups.sum()), 1)

    def test_no_unreviewed_type_coercion_or_null_equality(self):
        with self.assertRaises(ValueError):
            exact_feature_groups(pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [1.0]}))
        with self.assertRaises(ValueError):
            exact_feature_groups(pd.DataFrame({"x": [np.nan]}), pd.DataFrame({"x": [1.0]}))


if __name__ == "__main__":
    unittest.main()
