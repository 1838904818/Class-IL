"""Inclusive Section 18 thresholds and report reconciliation boundaries."""
import unittest

from ncd.summarize_quality_section18 import decision, row


class RegisteredThresholdTests(unittest.TestCase):
    def test_exact_overall_boundary(self):
        rows = [dict(class_name="a", test_rows=100, test_overlap_rows=5)]
        self.assertEqual(decision(rows)[0], "去重重跑")
        rows[0]["test_overlap_rows"] = 4
        self.assertEqual(decision(rows)[0], "保持")

    def test_exact_class_boundary_with_low_overall(self):
        rows = [dict(class_name="rare", test_rows=5, test_overlap_rows=1),
                dict(class_name="major", test_rows=995, test_overlap_rows=0)]
        self.assertEqual(decision(rows), ("去重重跑", ["rare"]))
        rows[0]["test_rows"] = 6
        self.assertEqual(decision(rows), ("保持", []))

    def test_empty_class_does_not_trigger(self):
        rows = [dict(class_name="empty", test_rows=0, test_overlap_rows=0),
                dict(class_name="observed", test_rows=100, test_overlap_rows=0)]
        self.assertEqual(decision(rows), ("保持", []))

    def test_invalid_counts_fail_closed(self):
        with self.assertRaises(AssertionError):
            row("x", "y", "z", 2, 1, 2, 1, 0, 2, 0, 0, "retained")
        with self.assertRaises(AssertionError):
            row("x", "y", "z", 2, 1, 3, 1, 0, 0, 0, 0, "retained")


if __name__ == "__main__":
    unittest.main()
