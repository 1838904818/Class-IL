"""Small arithmetic/identity tests only; no dataset is read."""
import hashlib
import unittest
import numpy as np
from audit_inputs import canonical_sha, overlap_counts, radius, reconstruct_indices


class AuditTests(unittest.TestCase):
    def fixture(self):
        source = "a" * 64
        seed = int.from_bytes(hashlib.sha256(f"42|{source}|2|normal_to_largest_attack|v2".encode()).digest()[:8], "little")
        order = np.random.default_rng(seed).permutation(30)
        cal, fit = np.sort(order[:3]), np.sort(order[3:15])
        return source, {"id": 2, "source_train_rows": 30, "calibration_rows": 3, "fit_rows": 12,
                        "seed": seed, "calibration_indices_sha256": canonical_sha(cal.tolist()),
                        "fit_indices_sha256": canonical_sha(fit.tolist())}, cal, fit

    def test_reconstruction_must_match_both_recorded_hashes(self):
        source, record, c, f = self.fixture()
        actual_c, actual_f = reconstruct_indices(record, source, 42)
        np.testing.assert_array_equal(actual_c, c)
        np.testing.assert_array_equal(actual_f, f)

    def test_bad_fit_hash_blocks_audit(self):
        source, record, _, _ = self.fixture()
        record["fit_indices_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            reconstruct_indices(record, source, 42)

    def test_wrong_seed_blocks_audit(self):
        source, record, _, _ = self.fixture()
        with self.assertRaises(ValueError):
            reconstruct_indices(record, source, 1)

    def test_equal_counts_do_not_imply_equal_identities(self):
        result = overlap_counts(np.array([1, 2, 3]), np.array([0, 1, 4]), np.array([2, 3, 5]))
        self.assertEqual(result, {"matches_native_calibration": 1, "overlaps_native_fit": 2, "outside_both": 0})

    def test_bound_shrinks_with_group_count_not_duplicates(self):
        self.assertAlmostEqual(radius(400, 100), radius(100, 100) / 2)
        self.assertGreater(radius(2, 100), 1)

    def test_unbounded_allocation_refused(self):
        source, record, _, _ = self.fixture()
        record["source_train_rows"] = 1000001
        with self.assertRaises(ValueError):
            reconstruct_indices(record, source, 42)


if __name__ == "__main__":
    unittest.main()
