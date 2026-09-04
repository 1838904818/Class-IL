import copy
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
import warnings
import zipfile

import numpy as np

from test_validate_semantics import fixture
from validate_arrays import METHODS, ranking, read_npz, verify_rows


def arrays_fixture():
    artifact, _, _ = fixture()
    artifact["source_bindings"] = {"streaming_manifest_file_sha256": "a"*64}
    samples = []
    background = []
    for split, target, classes in (("official_test", samples, range(10)), ("task0_train_background", background, range(2))):
        for c in classes:
            for i in range(54 if split == "official_test" and c == 7 else 128):
                target.append({"dataset": "malaya-network-gt", "split": split, "class_id": c,
                               "streaming_manifest_sha256": "a"*64,
                               "sample_id_sha256": hashlib.sha256(f"{split}-{c}-{i}".encode()).hexdigest()})
    probes = {"dataset": "malaya-network-gt", "raw_feature_rows_persisted": False,
              "streaming_manifest_sha256": "a"*64, "official_test": {"samples": samples},
              "task0_train_background": {"source_classes": [0, 1], "samples": background}}
    expected, secondary = {}, {}
    for method in METHODS:
        for row in artifact["checkpoint_rows"][method]:
            c, t = row["class_id"], row["checkpoint"]
            row["probe_rows"] = 54 if c == 7 else 128
            row["background_rows"] = 256
            core = f"checkpoint_{t:03d}_class_{c:03d}"
            prefix = core if method == METHODS[0] else method + "__" + core
            arrays = expected if method == METHODS[0] else secondary
            arrays[prefix+"_mean_abs"] = np.arange(77, 0, -1, dtype=np.float32)
            arrays[prefix+"_mean_signed"] = arrays[prefix+"_mean_abs"] / 2
            if method == METHODS[0]:
                arrays[core+"_sample_id_sha256"] = np.array([s["sample_id_sha256"].encode() for s in samples if s["class_id"] == c], dtype="S64")
    analysis = {"dataset": artifact["dataset"], "seed": artifact["seed"],
                "attribution_scope": copy.deepcopy(artifact["attribution_scope"]),
                "checkpoint_rows": copy.deepcopy(artifact["checkpoint_rows"][METHODS[0]])}
    return artifact, analysis, probes, expected, secondary


class ArrayTests(unittest.TestCase):
    def test_valid_arrays_and_rare_class_probe_count(self):
        output = verify_rows(*arrays_fixture())
        self.assertEqual(output["rankings_checked"], 90)
        self.assertEqual(output["mean_vectors_checked"], 180)
        self.assertEqual(output["unique_test_probes"], 1206)
        self.assertEqual(output["background_samples"], 256)

    def test_tie_order_is_feature_index(self):
        self.assertEqual(ranking(np.ones(77), np.zeros(77)), list(range(15)))

    def test_negative_mean_absolute_rejected(self):
        with self.assertRaisesRegex(ValueError, "Negative"):
            ranking(-np.ones(77), np.zeros(77))

    def test_nonfinite_rejected(self):
        for value in (float("nan"), float("inf")):
            bad = np.ones(77)
            bad[0] = value
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                ranking(bad, np.zeros(77))

    def test_shape_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            ranking(np.ones((1, 77)), np.zeros(77))

    def test_integer_dtype_rejected(self):
        with self.assertRaisesRegex(ValueError, "floating"):
            ranking(np.ones(77, dtype=int), np.zeros(77))

    def test_jensen_bound_rejected(self):
        with self.assertRaisesRegex(ValueError, "Signed mean"):
            ranking(np.ones(77), np.full(77, 2.0))

    def test_wrong_top15_even_with_valid_feature_indices(self):
        args = arrays_fixture()
        args[0]["checkpoint_rows"][METHODS[1]][0]["top15_indices"] = list(range(1, 16))
        with self.assertRaisesRegex(ValueError, "ranking"):
            verify_rows(*args)

    def test_probe_order_not_only_membership(self):
        args = arrays_fixture()
        key = "checkpoint_000_class_000_sample_id_sha256"
        args[3][key] = args[3][key][::-1]
        with self.assertRaisesRegex(ValueError, "identity/order"):
            verify_rows(*args)

    def test_duplicate_probe_identity_rejected(self):
        args = arrays_fixture()
        samples = args[2]["official_test"]["samples"]
        samples[1]["sample_id_sha256"] = samples[0]["sample_id_sha256"]
        with self.assertRaisesRegex(ValueError, "Duplicate probe"):
            verify_rows(*args)

    def test_background_future_class_rejected(self):
        args = arrays_fixture()
        args[2]["task0_train_background"]["samples"][0]["class_id"] = 2
        with self.assertRaisesRegex(ValueError, "Unexpected probe class"):
            verify_rows(*args)

    def test_expected_analysis_row_cannot_change(self):
        args = arrays_fixture()
        args[1]["checkpoint_rows"][0]["rationale_mass"] = 0.3
        with self.assertRaisesRegex(ValueError, "assembly"):
            verify_rows(*args)

    def test_exact_npz_members(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arrays.npz"
            np.savez(path, a=np.ones(77), extra=np.zeros(77))
            with self.assertRaisesRegex(ValueError, "registry"):
                read_npz(path, {"a"})

    def test_duplicate_npz_members(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arrays.npz"
            buffer = io.BytesIO()
            np.save(buffer, np.ones(77))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr("a.npy", buffer.getvalue())
                    archive.writestr("a.npy", buffer.getvalue())
            with self.assertRaisesRegex(ValueError, "Duplicate NPZ"):
                read_npz(path, {"a"})

    def test_pickle_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arrays.npz"
            np.savez(path, a=np.array([{}], dtype=object))
            with self.assertRaises(ValueError):
                read_npz(path, {"a"})

    def test_npz_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arrays.npz"
            np.savez_compressed(path, a=np.ones(77))
            np.testing.assert_array_equal(read_npz(path, {"a"})["a"], np.ones(77))


if __name__ == "__main__":
    unittest.main()
