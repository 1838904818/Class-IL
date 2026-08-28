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

from build_train_protocol_v2 import build_protocol_v2


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BuildTrainProtocolV2Tests(unittest.TestCase):
    def make_source(self, root: Path, *, include_normal: bool = True) -> Path:
        row_counts = [20, 7, 12, 5]
        classes = []
        for class_id, rows in enumerate(row_counts):
            directory = root / f"class_{class_id:02d}"
            directory.mkdir(parents=True)
            train = (
                np.arange(rows * 4, dtype=np.float32).reshape(rows, 4)
                + class_id * 1000
            )
            test = np.full((class_id + 2, 4), class_id, dtype=np.float32)
            train_path = directory / "train.npy"
            test_path = directory / "test.npy"
            np.save(train_path, train)
            np.save(test_path, test)
            classes.append(
                {
                    "id": class_id,
                    "name": "Benign" if class_id == 0 else f"attack-{class_id}",
                    "train": [
                        {
                            "path": f"class_{class_id:02d}/train.npy",
                            "rows": rows,
                            "sha256": sha256_file(train_path),
                        }
                    ],
                    "test": [
                        {
                            "path": f"class_{class_id:02d}/test.npy",
                            "rows": len(test),
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
            "classes": classes,
            "tasks": [[0, 1], [2, 3]],
        }
        if include_normal:
            manifest["normal_class_id"] = 0
        path = root / "streaming_manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return path

    def test_caps_only_normal_to_largest_attack_fit_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_source(root / "source")
            first = root / "first"
            second = root / "second"
            result_a = build_protocol_v2(
                source,
                first,
                calibration_fraction=0.10,
                seed=42,
                test_mode="copy",
            )
            result_b = build_protocol_v2(
                source,
                second,
                calibration_fraction=0.10,
                seed=42,
                test_mode="copy",
            )
            self.assertEqual(result_a["normal_fit_cap_rows"], 11)
            audit_a = json.loads(Path(result_a["audit"]).read_text(encoding="utf-8"))
            audit_b = json.loads(Path(result_b["audit"]).read_text(encoding="utf-8"))
            by_id = {int(item["id"]): item for item in audit_a["classes"]}
            self.assertEqual(by_id[0]["fit_rows"], 11)
            self.assertTrue(by_id[0]["fit_capped"])
            self.assertEqual(by_id[1]["fit_rows"], 6)
            self.assertEqual(by_id[2]["fit_rows"], 11)
            self.assertEqual(by_id[3]["fit_rows"], 4)
            self.assertTrue(
                all(not by_id[class_id]["fit_capped"] for class_id in (1, 2, 3))
            )
            self.assertFalse(audit_a["invariants"]["attack_fit_rows_sampled"])
            for left, right in zip(
                audit_a["classes"], audit_b["classes"], strict=True
            ):
                self.assertEqual(left["fit"]["sha256"], right["fit"]["sha256"])
                self.assertEqual(
                    left["calibration"]["sha256"], right["calibration"]["sha256"]
                )
                self.assertTrue(left["fit_calibration_disjoint"])
                self.assertTrue(left["official_test"][0]["byte_identical"])

    def test_holdout_reorders_only_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_source(root / "source")
            result = build_protocol_v2(
                source,
                root / "output",
                calibration_fraction=0.10,
                seed=42,
                held_out_class_id=3,
                test_mode="copy",
            )
            manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["tasks"], [[0, 1], [2], [3]])

    def test_requires_declared_normal_class(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_source(root / "source", include_normal=False)
            with self.assertRaisesRegex(TypeError, "normal_class_id"):
                build_protocol_v2(
                    source,
                    root / "output",
                    calibration_fraction=0.10,
                    seed=42,
                    test_mode="copy",
                )


if __name__ == "__main__":
    unittest.main()
