from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fullcache.core import (
    BuildOptions,
    build_dataset_cache,
    feature_hash_splits,
    sha256_file,
    stable_row_hash64,
)
from fullcache.specs import DATASET_SPECS, NSL_KDD_COLUMNS
from fullcache.verify import verify_dataset_cache
from streaming_full.data import load_manifest


HASH_LABELS = {
    "cic-ids-2017": {
        "Normal": "BENIGN",
        "DoS": "DoS Hulk",
        "DDoS": "DDoS",
        "Bruteforce": "FTP-Patator",
        "PortScan": "PortScan",
        "WebAttack": "Web Attack \x96 XSS",
        "Botnet": "Bot",
        "Infiltration": "Infiltration",
    },
    "cic-ids-2018": {
        "Normal": "Benign",
        "DoS": "DoS attacks-Hulk",
        "DDoS": "DDOS attack-HOIC",
        "Bruteforce": "FTP-BruteForce",
        "WebAttack": "Brute Force -Web",
        "Infiltration": "Infilteration",
        "Botnet": "Bot",
    },
    "cic-iot-2023": {
        "Normal": "BenignTraffic",
        "DDoS": "DDoS-UDP_Flood",
        "DoS": "DoS-TCP_Flood",
        "Recon": "Recon-PortScan",
        "WebAttack": "SqlInjection",
        "Bruteforce": "DictionaryBruteForce",
        "Spoofing": "DNS_Spoofing",
        "Mirai": "Mirai-udpplain",
    },
}


def _hash_balanced_rows(
    feature_columns: list[str], labels: dict[str, str], per_split: int = 2
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    candidate = 1
    for raw_label in labels.values():
        found = {"train": 0, "test": 0}
        while min(found.values()) < per_split:
            values = np.asarray(
                [candidate + (column * 3) for column in range(len(feature_columns))],
                dtype=np.float32,
            )
            split = str(feature_hash_splits(values[None, :])[0])
            candidate += 1
            if found[split] >= per_split:
                continue
            row = {column: float(value) for column, value in zip(feature_columns, values)}
            row["__label__"] = raw_label
            rows.append(row)
            found[split] += 1
    return rows


def _load_split(root: Path, dataset: str, split: str) -> np.ndarray:
    paths = sorted((root / dataset / split).glob("*/*.npy"))
    arrays = [np.load(path, allow_pickle=False) for path in paths]
    if not arrays:
        return np.empty((0, DATASET_SPECS[dataset].expected_feature_count), np.float32)
    return np.concatenate(arrays, axis=0)


def _load_class_split(
    root: Path, dataset: str, split: str, class_name: str
) -> np.ndarray:
    arrays = [
        np.load(path, allow_pickle=False)
        for path in sorted((root / dataset / split / class_name).glob("*.npy"))
    ]
    return np.concatenate(arrays, axis=0)


class FullCacheSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="ofra_fullcache_smoke_")
        self.root = Path(self.temporary.name)
        self.data = self.root / "datasets"
        self.output = self.root / "cache"
        self.data.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _build(
        self, dataset: str, *, strict_files: bool, chunk_rows: int = 17
    ) -> dict:
        return build_dataset_cache(
            dataset,
            self.data,
            self.output,
            BuildOptions(
                chunk_rows=chunk_rows,
                strict_files=strict_files,
                strict_unmapped_labels=True,
            ),
            progress=None,
        )

    def _assert_streaming_manifest(self, dataset: str, tasks: tuple) -> None:
        path = self.output / dataset / "streaming_manifest.json"
        manifest = load_manifest(path, verify_hashes=True)
        self.assertEqual(manifest.tasks, tasks)
        self.assertEqual(manifest.normal_class_id, 0)
        self.assertEqual(manifest.feature_dim, DATASET_SPECS[dataset].expected_feature_count)

    def test_cic18_reconciles_80_and_84_columns_and_hashes_artifacts(self) -> None:
        directory = self.data / "cic-ids-2018"
        directory.mkdir()
        features = ["Dst Port", *[f"F{i:02d}" for i in range(77)]]
        rows = _hash_balanced_rows(features, HASH_LABELS["cic-ids-2018"])
        regular_rows = []
        tuesday_rows = []
        for index, row in enumerate(rows):
            target = regular_rows if index % 2 == 0 else tuesday_rows
            item = {column: row[column] for column in features}
            item["Timestamp"] = f"2018-02-20 00:{index:02d}:00"
            item["Label"] = row["__label__"]
            if target is tuesday_rows:
                item = {
                    "Flow ID": f"flow-{index}",
                    "Src IP": "10.0.0.1",
                    "Src Port": 1000 + index,
                    "Dst IP": "10.0.0.2",
                    **item,
                }
            target.append(item)
        regular = directory / "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"
        tuesday = directory / "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"
        pd.DataFrame(regular_rows).to_csv(regular, index=False)
        pd.DataFrame(tuesday_rows).to_csv(tuesday, index=False)

        manifest = self._build("cic-ids-2018", strict_files=False, chunk_rows=5)
        schema = json.loads(
            (self.output / "cic-ids-2018" / "feature_schema.json").read_text("utf-8")
        )
        self.assertEqual(schema["feature_count"], 78)
        self.assertIn("Dst Port", schema["feature_columns"])
        self.assertNotIn("Src Port", schema["feature_columns"])
        by_name = {entry["relative_path"]: entry for entry in manifest["raw_files"]}
        self.assertEqual(by_name[regular.name]["raw_column_count"], 80)
        self.assertEqual(by_name[tuesday.name]["raw_column_count"], 84)
        self.assertEqual(
            by_name[tuesday.name]["identifier_columns_removed"],
            ["Flow ID", "Src IP", "Src Port", "Dst IP"],
        )
        for entry in manifest["raw_files"]:
            self.assertEqual(entry["sha256"], sha256_file(directory / entry["relative_path"]))
        for entry in manifest["shards"]:
            path = self.output / "cic-ids-2018" / entry["relative_path"]
            self.assertEqual(entry["sha256"], sha256_file(path))
        self._assert_streaming_manifest(
            "cic-ids-2018", ((0, 1), (2, 3), (4, 5), (6,))
        )
        verified = verify_dataset_cache(
            self.output, "cic-ids-2018", verify_raw_hashes=True, scan_rows=3
        )
        self.assertEqual(verified["status"], "ok")

    def test_cic17_identical_features_never_cross_splits(self) -> None:
        directory = self.data / "cic-ids-2017"
        directory.mkdir()
        features = ["Destination Port", *[f"F{i:02d}" for i in range(77)]]
        rows = _hash_balanced_rows(features, HASH_LABELS["cic-ids-2017"])
        duplicate = dict(rows[0])
        rows.extend([dict(duplicate), dict(duplicate)])
        frame_rows = []
        for row in rows:
            item = {column: row[column] for column in features}
            item["Label"] = row["__label__"]
            frame_rows.append(item)
        invalid = dict(frame_rows[0])
        invalid[features[3]] = "inf"
        frame_rows.append(invalid)
        path = directory / "Monday-WorkingHours.pcap_ISCX.csv"
        pd.DataFrame(frame_rows).to_csv(path, index=False, encoding="latin-1")

        manifest = self._build("cic-ids-2017", strict_files=False, chunk_rows=7)
        train_hashes = set(stable_row_hash64(_load_split(self.output, "cic-ids-2017", "train")))
        test_hashes = set(stable_row_hash64(_load_split(self.output, "cic-ids-2017", "test")))
        self.assertFalse(train_hashes & test_hashes)
        self.assertEqual(manifest["quality_counts"]["invalid_numeric_or_nonfinite_rows"], 1)
        self.assertEqual(manifest["quality_counts"]["rows_written"], len(rows))
        self._assert_streaming_manifest(
            "cic-ids-2017", ((0, 1), (2, 3), (4, 5), (6, 7))
        )

    def test_cic_iot_exceeds_legacy_attack_cap_without_sampling(self) -> None:
        directory = self.data / "cic-iot-2023"
        directory.mkdir()
        features = [f"feature_{index:02d}" for index in range(46)]
        rows = _hash_balanced_rows(features, HASH_LABELS["cic-iot-2023"])
        repeated = dict(rows[20])
        repeated["__label__"] = "DictionaryBruteForce"
        rows.extend(dict(repeated) for _ in range(50_001))
        frame_rows = []
        for row in rows:
            item = {column: row[column] for column in features}
            item["label"] = row["__label__"]
            frame_rows.append(item)
        pd.DataFrame(frame_rows).to_csv(directory / "part-00000.csv", index=False)

        manifest = self._build("cic-iot-2023", strict_files=False, chunk_rows=8_000)
        brute_rows = sum(
            manifest["row_counts"][split]["Bruteforce"]
            for split in ("train", "test")
        )
        self.assertGreater(brute_rows, 50_000)
        self.assertIsNone(manifest["quality_counts"]["class_cap"])
        self.assertFalse(manifest["quality_counts"]["sampling_applied"])
        self._assert_streaming_manifest(
            "cic-iot-2023", ((0, 1), (2, 3), (4, 5), (6, 7))
        )

    def test_nf_preserves_official_file_splits_and_filters_infinity(self) -> None:
        directory = self.data / "nf-ton-iot-v2"
        directory.mkdir()
        numeric = [f"N{index:02d}" for index in range(39)]
        raw_labels = list(DATASET_SPECS["nf-ton-iot-v2"].label_mapping)

        def make_rows(split: str) -> list[dict[str, object]]:
            result = []
            for index, label in enumerate(raw_labels):
                row = {
                    "IPV4_SRC_ADDR": f"10.0.0.{index + 1}",
                    "IPV4_DST_ADDR": "10.0.1.1",
                    "L4_SRC_PORT": 1000 + index,
                    "L4_DST_PORT": 80,
                    **{column: index + offset for offset, column in enumerate(numeric)},
                    "Label": int(label != "benign"),
                    "Attack": label,
                }
                if split == "test" and index == 0:
                    bad = dict(row)
                    bad[numeric[0]] = "inf"
                    result.append(bad)
                result.append(row)
            return result

        train = directory / "NF-ToN-IoT-v2-train.csv"
        test = directory / "NF-ToN-IoT-v2-test.csv"
        pd.DataFrame(make_rows("train")).to_csv(train, index=False)
        pd.DataFrame(make_rows("test")).to_csv(test, index=False)
        manifest = self._build("nf-ton-iot-v2", strict_files=True, chunk_rows=4)
        self.assertEqual(sum(manifest["row_counts"]["train"].values()), 10)
        self.assertEqual(sum(manifest["row_counts"]["test"].values()), 10)
        self.assertEqual(manifest["quality_counts"]["invalid_numeric_or_nonfinite_rows"], 1)
        for entry in manifest["shards"]:
            expected = "train" if "train" in entry["source_file"].lower() else "test"
            self.assertEqual(entry["split"], expected)
        overlap = manifest["split_overlap_audit"]
        self.assertEqual(overlap["overlap_unique_feature_rows"], 10)
        self.assertEqual(overlap["overlap_unique_sha256_digests"], 10)
        self.assertEqual(overlap["sha256_digest_collisions_detected"], 0)
        self.assertEqual(overlap["train_rows_in_overlap"], 10)
        self.assertEqual(overlap["test_rows_in_overlap"], 10)
        verified = verify_dataset_cache(
            self.output,
            "nf-ton-iot-v2",
            verify_raw_hashes=True,
            scan_rows=3,
            recompute_overlap=True,
            overlap_batch_rows=3,
        )
        self.assertTrue(verified["overlap_recomputed"])
        self._assert_streaming_manifest(
            "nf-ton-iot-v2", ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
        )

    def test_nsl_vocab_is_train_only_and_official_splits_are_kept(self) -> None:
        directory = self.data / "nsl-kdd"
        directory.mkdir()
        numeric = [
            column
            for column in NSL_KDD_COLUMNS
            if column not in {"protocol_type", "service", "flag", "label", "difficulty"}
        ]
        family_labels = ["normal", "neptune", "satan", "guess_passwd", "buffer_overflow"]
        train_rows = []
        for index in range(70):
            row = {column: index + offset for offset, column in enumerate(numeric)}
            row.update(
                {
                    "protocol_type": ("tcp", "udp", "icmp")[index % 3],
                    "service": f"svc{index:03d}",
                    "flag": f"flag{index % 11:02d}",
                    "label": family_labels[index % len(family_labels)],
                    "difficulty": 0,
                }
            )
            train_rows.append(row)
        test_rows = []
        for index, label in enumerate(family_labels):
            row = {column: 1000 + index + offset for offset, column in enumerate(numeric)}
            row.update(
                {
                    "protocol_type": "tcp",
                    "service": "svc000" if index else "test-only-service",
                    "flag": "flag00",
                    "label": label,
                    "difficulty": 0,
                }
            )
            test_rows.append(row)
        fallback = dict(test_rows[-1])
        fallback["label"] = "future_r2l_attack"
        fallback["service"] = "svc001"
        test_rows.append(fallback)
        pd.DataFrame(train_rows, columns=NSL_KDD_COLUMNS).to_csv(
            directory / "KDDTrain+.txt", index=False, header=False
        )
        pd.DataFrame(test_rows, columns=NSL_KDD_COLUMNS).to_csv(
            directory / "KDDTest+.txt", index=False, header=False
        )

        manifest = self._build("nsl-kdd", strict_files=True, chunk_rows=13)
        schema = json.loads(
            (self.output / "nsl-kdd" / "feature_schema.json").read_text("utf-8")
        )
        self.assertEqual(schema["feature_count"], 122)
        self.assertNotIn("service_test-only-service", schema["feature_columns"])
        self.assertEqual(
            manifest["quality_counts"]["unknown_categorical_counts"]["service"][
                "test-only-service"
            ],
            1,
        )
        self.assertEqual(manifest["quality_counts"]["defaulted_label_rows"], 1)
        self.assertEqual(sum(manifest["row_counts"]["train"].values()), 70)
        self.assertEqual(sum(manifest["row_counts"]["test"].values()), 6)
        normal_test = _load_class_split(
            self.output, "nsl-kdd", "test", "Normal"
        )
        duration_index = schema["feature_columns"].index("duration")
        unknown_row = normal_test[normal_test[:, duration_index] == 1000][0]
        service_indices = [
            index
            for index, column in enumerate(schema["feature_columns"])
            if column.startswith("service_")
        ]
        self.assertTrue(np.all(unknown_row[service_indices] == 0.0))
        self._assert_streaming_manifest(
            "nsl-kdd", ((0, 1), (2,), (3,), (4,))
        )

    def test_unsw_test_only_category_is_all_zero_not_schema_leakage(self) -> None:
        directory = self.data / "unsw-nb15"
        directory.mkdir()
        numeric = [f"M{index:02d}" for index in range(39)]
        classes = list(DATASET_SPECS["unsw-nb15"].class_order)
        train_rows = []
        for index in range(133):
            train_rows.append(
                {
                    "id": index,
                    **{column: index + offset for offset, column in enumerate(numeric)},
                    "proto": f"p{index:03d}",
                    "service": f"s{index % 13:02d}",
                    "state": f"st{index % 9:02d}",
                    "attack_cat": classes[index % len(classes)],
                    "label": int(index % len(classes) != 0),
                }
            )
        test_rows = []
        for index, class_name in enumerate(classes):
            test_rows.append(
                {
                    "id": 1000 + index,
                    **{column: 2000 + index + offset for offset, column in enumerate(numeric)},
                    "proto": "p000",
                    "service": "s00",
                    "state": "ACC" if index == 0 else "st00",
                    "attack_cat": class_name,
                    "label": int(class_name != "Normal"),
                }
            )
        pd.DataFrame(train_rows).to_csv(
            directory / "UNSW_NB15_training-set.csv", index=False
        )
        pd.DataFrame(test_rows).to_csv(
            directory / "UNSW_NB15_testing-set.csv", index=False
        )

        manifest = self._build("unsw-nb15", strict_files=True, chunk_rows=19)
        schema = json.loads(
            (self.output / "unsw-nb15" / "feature_schema.json").read_text("utf-8")
        )
        self.assertEqual(schema["feature_count"], 194)
        self.assertNotIn("state_ACC", schema["feature_columns"])
        self.assertEqual(
            manifest["quality_counts"]["unknown_categorical_counts"]["state"]["ACC"],
            1,
        )
        self.assertEqual(sum(manifest["row_counts"]["train"].values()), 133)
        self.assertEqual(sum(manifest["row_counts"]["test"].values()), 10)
        normal_test = _load_class_split(
            self.output, "unsw-nb15", "test", "Normal"
        )
        state_indices = [
            index
            for index, column in enumerate(schema["feature_columns"])
            if column.startswith("state_")
        ]
        self.assertTrue(np.all(normal_test[0, state_indices] == 0.0))
        self._assert_streaming_manifest(
            "unsw-nb15", ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
        )


if __name__ == "__main__":
    unittest.main()
