from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from fullcache.core import BuildOptions, build_dataset_cache, sha256_file
from fullcache.specs import DATASET_SPECS
from fullcache.verify import verify_dataset_cache
from streaming_full.data import load_manifest


RAW_CLASS_COUNTS = {
    "Bittorent": 5,
    "ChromeRDP": 3,
    "Discord": 3,
    "EA Origin": 4,
    "Microsoft Teams": 2,
    "Slack": 2,
    "Steam": 4,
    "Teamviewer": 2,
    "Webex": 3,
    "Zoom": 3,
}


class MalayaFullCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="ofra_malaya_fullcache_")
        self.root = Path(self.temporary.name)
        self.source = self.root / "malaya-network-gt" / "csv_output"
        self.source.mkdir(parents=True)
        self.output = self.root / "cache"
        self.original_spec = DATASET_SPECS["malaya-network-gt"]

    def tearDown(self) -> None:
        DATASET_SPECS["malaya-network-gt"] = self.original_spec
        self.temporary.cleanup()

    def _write_frozen_source(self) -> Path:
        feature_columns = ["protocol", *[f"f{index:02d}" for index in range(75)], "constant_feature"]
        source_hashes: dict[str, str] = {}
        test_captures: list[str] = []
        canonical_index = {
            name: index for index, name in enumerate(self.original_spec.class_order)
        }
        for raw_class, count in RAW_CLASS_COUNTS.items():
            canonical = self.original_spec.label_mapping[raw_class]
            class_index = canonical_index[canonical]
            for capture_index in range(count):
                relative = f"{raw_class}/capture-{capture_index:02d}.csv"
                path = self.source / Path(relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                row = {
                    "src_ip": f"10.{class_index}.0.{capture_index + 1}",
                    "dst_ip": "192.0.2.1",
                    "src_port": 10_000 + capture_index,
                    "dst_port": 443,
                    "timestamp": f"2026-01-01 00:{class_index:02d}:{capture_index:02d}",
                    "protocol": 6,
                    **{
                        f"f{index:02d}": float(class_index * 100 + index)
                        for index in range(75)
                    },
                    "constant_feature": 1.0,
                }
                pd.DataFrame([row], columns=[
                    "src_ip", "dst_ip", "src_port", "dst_port", "timestamp",
                    *feature_columns,
                ]).to_csv(path, index=False)
                source_hashes[relative] = sha256_file(path)
                if capture_index == 0:
                    test_captures.append(relative)

        self.assertEqual(len(source_hashes), 31)
        contract = {
            "schema_version": 1,
            "dataset": "MalayaNetwork_GT",
            "source_revision": self.original_spec.source_revision,
            "split_strategy": "one_capture_per_class_closest_to_20_percent_then_lexicographic",
            "split_seed": None,
            "class_order": list(self.original_spec.class_order),
            "test_captures": test_captures,
            "excluded_identifier_features": list(self.original_spec.drop_columns),
            "source_csv_sha256": source_hashes,
        }
        path = self.source.parent / "candidate_capture_split.json"
        path.write_text(
            json.dumps(contract, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        DATASET_SPECS["malaya-network-gt"] = replace(
            self.original_spec,
            source_contract_relative="../candidate_capture_split.json",
            bundled_contract_relative=None,
            source_contract_sha256=sha256_file(path),
        )
        return path

    def test_nested_capture_contract_build_and_exact_overlap(self) -> None:
        contract_path = self._write_frozen_source()
        manifest = build_dataset_cache(
            "malaya-network-gt",
            self.root,
            self.output,
            BuildOptions(chunk_rows=1, overlap_batch_rows=2),
            progress=None,
        )

        self.assertEqual(manifest["problem_type"], "application_classification")
        self.assertEqual(manifest["task_semantics"], "class_incremental")
        self.assertEqual(manifest["metric_profile"], "generic_multiclass")
        self.assertIsNone(manifest["normal_class_id"])
        self.assertEqual(len(manifest["raw_files"]), 31)
        self.assertTrue(
            all("/" in entry["relative_path"] for entry in manifest["raw_files"])
        )
        self.assertEqual(
            {entry["capture_split"] for entry in manifest["raw_files"]},
            {"train", "test"},
        )
        self.assertTrue(manifest["split_protocol"]["capture_disjoint"])
        self.assertEqual(manifest["split_protocol"]["test_capture_count"], 10)
        self.assertEqual(manifest["split_protocol"]["train_capture_count"], 21)
        self.assertEqual(
            manifest["sidecars"]["source_capture_split.json"],
            sha256_file(contract_path),
        )

        schema = json.loads(
            (self.output / "malaya-network-gt" / "feature_schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(schema["feature_count"], 77)
        self.assertIn("constant_feature", schema["feature_columns"])
        self.assertTrue(
            set(self.original_spec.drop_columns).isdisjoint(schema["feature_columns"])
        )

        overlap = manifest["split_overlap_audit"]
        self.assertEqual(overlap["official_split_action"], "audit_only_no_delete_no_resplit")
        self.assertEqual(overlap["overlap_unique_feature_rows"], 10)
        self.assertEqual(overlap["train_rows_in_overlap"], 21)
        self.assertEqual(overlap["test_rows_in_overlap"], 10)

        runner = load_manifest(
            self.output / "malaya-network-gt" / "streaming_manifest.json",
            verify_hashes=True,
        )
        self.assertEqual(runner.normal_class_id, None)
        self.assertEqual(runner.tasks, ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9)))
        self.assertEqual(runner.metric_profile, "generic_multiclass")

        verified = verify_dataset_cache(
            self.output,
            "malaya-network-gt",
            recompute_overlap=True,
            overlap_batch_rows=2,
        )
        self.assertEqual(verified["status"], "ok")
        self.assertTrue(verified["overlap_recomputed"])

    def test_contract_tamper_fails_before_cache_creation(self) -> None:
        contract_path = self._write_frozen_source()
        contract_path.write_text(
            contract_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "capture-contract SHA256 mismatch"):
            build_dataset_cache(
                "malaya-network-gt",
                self.root,
                self.output,
                BuildOptions(chunk_rows=1),
                progress=None,
            )
        self.assertFalse((self.output / "malaya-network-gt").exists())


if __name__ == "__main__":
    unittest.main()
