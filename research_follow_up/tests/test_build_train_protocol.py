from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_train_protocol import build_protocol


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BuildTrainProtocolTests(unittest.TestCase):
    def make_source(self, root: Path) -> Path:
        classes = []
        for class_id in range(4):
            directory = root / f"class_{class_id:02d}"
            directory.mkdir(parents=True)
            train = (
                np.arange(48, dtype=np.float32).reshape(12, 4) + class_id * 100
            )
            test = np.full((5, 4), class_id, dtype=np.float32)
            train_path = directory / "train.npy"
            test_path = directory / "test.npy"
            np.save(train_path, train)
            np.save(test_path, test)
            classes.append(
                {
                    "id": class_id,
                    "name": f"class-{class_id}",
                    "train": [
                        {
                            "path": f"class_{class_id:02d}/train.npy",
                            "rows": 12,
                            "sha256": sha256_file(train_path),
                        }
                    ],
                    "test": [
                        {
                            "path": f"class_{class_id:02d}/test.npy",
                            "rows": 5,
                            "sha256": sha256_file(test_path),
                        }
                    ],
                }
            )
        manifest = {
            "schema_version": 1,
            "dataset": "synthetic",
            "feature_dim": 4,
            "problem_type": "intrusion_detection",
            "metric_profile": "nids_multiclass_with_binary_detection",
            "task_semantics": "class_incremental",
            "normal_class_id": 0,
            "classes": classes,
            "tasks": [[0, 1], [2, 3]],
        }
        path = root / "streaming_manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return path

    def test_cap_calibration_and_full_test_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_source(root / "source")
            first = root / "first"
            second = root / "second"
            result_a = build_protocol(
                source,
                first,
                train_cap=5,
                calibration_fraction=0.25,
                seed=42,
                held_out_class_id=3,
                test_mode="copy",
            )
            result_b = build_protocol(
                source,
                second,
                train_cap=5,
                calibration_fraction=0.25,
                seed=42,
                held_out_class_id=3,
                test_mode="copy",
            )
            manifest_a = json.loads(Path(result_a["manifest"]).read_text(encoding="utf-8"))
            manifest_b = json.loads(Path(result_b["manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest_a["tasks"], [[0, 1], [2], [3]])
            for class_a, class_b in zip(manifest_a["classes"], manifest_b["classes"]):
                self.assertEqual(class_a["train"][0]["rows"], 5)
                self.assertEqual(class_a["train"][0]["sha256"], class_b["train"][0]["sha256"])
                source_test = source.parent / f"class_{class_a['id']:02d}" / "test.npy"
                derived_test = first / class_a["test"][0]["path"]
                self.assertEqual(sha256_file(source_test), sha256_file(derived_test))
            audit = json.loads(Path(result_a["audit"]).read_text(encoding="utf-8"))
            for item in audit["classes"]:
                self.assertEqual(item["calibration_rows"], 3)
                self.assertTrue(item["fit_calibration_disjoint"])
                self.assertTrue(item["official_test"][0]["byte_identical"])

    def test_refuses_task0_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_source(root / "source")
            with self.assertRaisesRegex(ValueError, "Task 0"):
                build_protocol(
                    source,
                    root / "bad",
                    train_cap=5,
                    calibration_fraction=0.25,
                    seed=42,
                    held_out_class_id=1,
                    test_mode="copy",
                )

    def test_refuses_existing_empty_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_source(root / "source")
            output = root / "already-exists"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                build_protocol(
                    source,
                    output,
                    train_cap=5,
                    calibration_fraction=0.25,
                    seed=42,
                    test_mode="copy",
                )

    def test_hardlink_matches_expected_hash_and_inode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_source(root / "source")
            output = root / "hardlinked"
            result = build_protocol(
                source,
                output,
                train_cap=5,
                calibration_fraction=0.25,
                seed=42,
                test_mode="hardlink",
            )
            manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
            for class_record in manifest["classes"]:
                class_id = int(class_record["id"])
                source_test = source.parent / f"class_{class_id:02d}" / "test.npy"
                derived_test = output / class_record["test"][0]["path"]
                source_stat = source_test.stat()
                derived_stat = derived_test.stat()
                self.assertEqual(source_stat.st_dev, derived_stat.st_dev)
                self.assertEqual(source_stat.st_ino, derived_stat.st_ino)
                self.assertEqual(
                    class_record["test"][0]["sha256"], sha256_file(derived_test)
                )


if __name__ == "__main__":
    unittest.main()
