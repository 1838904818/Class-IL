from __future__ import annotations

import json
import copy
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

from sensitivity_overlap import build_overlap_view
from streaming_full.exposure_preflight import (
    exposure_preflight_from_path,
    main as exposure_preflight_main,
    validate_exposure_preflight,
)
from streaming_full.data import (
    canonical_sha256,
    load_evaluation_view,
    load_manifest,
    sha256_file,
)
from streaming_full.smoke_test import _make_synthetic_manifest
from streaming_full.summarize import summarize_campaign
from streaming_full.validation import (
    ARMS,
    OutputDirectoryLock,
    RunConfig,
    _exposure_counter_dtype,
    _validate_training_instrumentation,
    run_manifest,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _rewrite_view_manifest(path: Path, value: dict) -> None:
    value.pop("canonical_manifest", None)
    value["canonical_manifest"] = {
        "algorithm": "sha256-canonical-json-excluding-this-field-v1",
        "sha256": canonical_sha256(value),
    }
    _write_json(path, value)


def _manifest_with_label_aware_overlap(root: Path) -> Path:
    manifest_path = _make_synthetic_manifest(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def shard(class_id: int, split: str, ordinal: int = 0):
        return manifest["classes"][class_id][split][ordinal]

    train0_path = root / shard(0, "train")["path"]
    train1_path = root / shard(1, "train")["path"]
    test0_path = root / shard(0, "test")["path"]
    train0 = np.load(train0_path, allow_pickle=False)
    train1 = np.load(train1_path, allow_pickle=False)
    test0 = np.load(test0_path, allow_pickle=False)

    # same-label-only, different-label-only, and mixed-including-same.
    train0[2] = train1[2]
    test0[0] = train0[0]
    test0[1] = train1[0]
    test0[2] = train0[2]
    np.save(train0_path, train0.astype(np.float32), allow_pickle=False)
    np.save(test0_path, test0.astype(np.float32), allow_pickle=False)
    shard(0, "train")["sha256"] = sha256_file(train0_path)
    shard(0, "test")["sha256"] = sha256_file(test0_path)
    overlap_path = root / "split_overlap_audit.json"
    overlap = {
        "schema_version": 1,
        "dataset": manifest["dataset"],
        "feature_dim": manifest["feature_dim"],
        "algorithm": "sqlite_exact_float32_row_bytes_v1",
        "equality_key": (
            "complete canonical little-endian float32 feature-row bytes; "
            "no probabilistic hash is used for equality"
        ),
        "official_split_action": "audit_only_no_delete_no_resplit",
        "test_rows": 80,
        "test_rows_in_overlap": 3,
        "overlap_unique_feature_rows": 3,
    }
    overlap["canonical_report_sha256"] = canonical_sha256(overlap)
    _write_json(overlap_path, overlap)
    overlap_sha = sha256_file(overlap_path)
    manifest["source"]["split_overlap_audit_sha256"] = overlap_sha
    _write_json(manifest_path, manifest)

    fullcache_path = root / "manifest.json"
    fullcache = json.loads(fullcache_path.read_text(encoding="utf-8"))
    fullcache["sidecars"]["streaming_manifest.json"] = sha256_file(manifest_path)
    fullcache["sidecars"]["split_overlap_audit.json"] = overlap_sha
    fullcache.pop("canonical_manifest", None)
    fullcache["canonical_manifest"] = {
        "algorithm": "sha256-canonical-json-excluding-this-field-v1",
        "sha256": canonical_sha256(fullcache),
    }
    _write_json(fullcache_path, fullcache)
    return manifest_path


def _config() -> RunConfig:
    return RunConfig(
        pretrain_epochs=1,
        epochs_per_task=1,
        batch_size=16,
        eval_batch_size=7,
        learning_rate=2e-3,
        d_model=12,
        n_layers=2,
        lora_rank=2,
        lora_alpha=4.0,
        minority_threshold=1000,
        negative_ratio=4,
        exemplar_capacity=4,
        exemplar_candidate_capacity=24,
        router_cap_samples=20,
        router_lambda_quantile=0.30,
        router_max_centroids=4,
        device="cpu",
        deterministic=True,
        verify_shard_hashes=True,
        verbose=False,
    )


def _set_tasks(manifest_path: Path, tasks: list[list[int]]) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tasks"] = tasks
    _write_json(manifest_path, manifest)
    fullcache_path = manifest_path.parent / "manifest.json"
    fullcache = json.loads(fullcache_path.read_text(encoding="utf-8"))
    fullcache["streaming_runner"]["tasks"] = tasks
    fullcache["sidecars"]["streaming_manifest.json"] = sha256_file(manifest_path)
    fullcache.pop("canonical_manifest", None)
    fullcache["canonical_manifest"] = {
        "algorithm": "sha256-canonical-json-excluding-this-field-v1",
        "sha256": canonical_sha256(fullcache),
    }
    _write_json(fullcache_path, fullcache)


class StreamingViewTests(unittest.TestCase):
    def test_run_config_normalizes_integral_json_floats(self):
        native = _config()
        serialized = asdict(native)
        serialized["lora_alpha"] = 4
        serialized["focal_gamma"] = 2
        restored = RunConfig(**json.loads(json.dumps(serialized)))

        self.assertEqual(asdict(restored), asdict(native))
        self.assertIsInstance(restored.lora_alpha, float)
        self.assertIsInstance(restored.focal_gamma, float)

    def test_exposure_counter_uses_minimum_safe_unsigned_dtype(self):
        self.assertIs(_exposure_counter_dtype(10), np.uint8)
        self.assertIs(_exposure_counter_dtype(255), np.uint8)
        self.assertIs(_exposure_counter_dtype(256), np.uint16)
        self.assertIs(_exposure_counter_dtype(65_535), np.uint16)
        self.assertIs(_exposure_counter_dtype(65_536), np.uint32)
        with self.assertRaisesRegex(ValueError, "capacity"):
            _exposure_counter_dtype(2**32)

    def test_exposure_preflight_is_metadata_only_deterministic_and_verifiable(self):
        with tempfile.TemporaryDirectory(prefix="ofra_preflight_test_") as temporary:
            root = Path(temporary)
            manifest_path = _make_synthetic_manifest(root)
            _set_tasks(manifest_path, [[0, 1], [2], [3]])
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            test_shard = root / raw["classes"][0]["test"][0]["path"]
            hidden = test_shard.with_suffix(".hidden")
            test_shard.rename(hidden)
            try:
                first = exposure_preflight_from_path(manifest_path, _config())
                second = exposure_preflight_from_path(manifest_path, _config())
            finally:
                hidden.rename(test_shard)
            self.assertEqual(first, second)
            validate_exposure_preflight(first)
            families = {
                family["class_id"]: family
                for task in first["tasks_detail"]
                for family in task["families"]
            }
            probe = families[2]
            self.assertEqual(probe["positive_rows_per_epoch"], 48)
            self.assertEqual(probe["negative_candidate_pool"]["rows"], 8)
            self.assertEqual(probe["selected_negative_rows_per_epoch"], 8)
            self.assertEqual(probe["zero_negative_steps_per_epoch"], 8)
            self.assertTrue(probe["candidate_limited"])

            config_path = root / "config.json"
            _write_json(config_path, asdict(_config()))
            output_path = root / "exposure_preflight.json"
            self.assertEqual(
                exposure_preflight_main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--config-json",
                        str(config_path),
                        "--output",
                        str(output_path),
                    ]
                ),
                0,
            )
            self.assertEqual(
                exposure_preflight_main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--config-json",
                        str(config_path),
                        "--output",
                        str(output_path),
                        "--verify-only",
                    ]
                ),
                0,
            )
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")), first
            )

    def test_singleton_task_exposure_instrumentation_reports_candidate_limit(self):
        with tempfile.TemporaryDirectory(prefix="ofra_exposure_test_") as temporary:
            root = Path(temporary)
            manifest_path = _make_synthetic_manifest(root)
            _set_tasks(manifest_path, [[0, 1], [2], [3]])
            output = run_manifest(
                manifest_path,
                seeds=[31],
                output_dir=root / "result",
                config=_config(),
            )
            result = output["results"][0]
            self.assertEqual(
                output["protocol"]["exposure_preflight"],
                exposure_preflight_from_path(manifest_path, _config()),
            )
            tampered = copy.deepcopy(result)
            tampered["training_exposure_records"]["2"][
                "desired_negative_rows_per_epoch"
            ] += 1
            with self.assertRaisesRegex(RuntimeError, "differs from exposure preflight"):
                _validate_training_instrumentation(
                    tampered,
                    root / "tampered.json",
                    expected_preflight=output["protocol"]["exposure_preflight"],
                )
            for class_id in ("2", "3"):
                exposure = result["training_exposure_records"][class_id]
                self.assertTrue(exposure["candidate_limited"])
                self.assertLess(
                    exposure["negative_ratio_realized"], _config().negative_ratio
                )
                self.assertTrue(
                    all(
                        source["source_kind"] == "prior_exemplar"
                        for source in exposure["negative_sources"]
                    )
                )
                self.assertGreater(
                    sum(
                        epoch["exposure"]["zero_negative_steps"]
                        for epoch in result["training_history"][class_id]
                    ),
                    0,
                )

    def test_mask_builder_dual_view_invariance_and_formal_summary(self):
        with tempfile.TemporaryDirectory(prefix="ofra_view_test_") as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            manifest_path = _manifest_with_label_aware_overlap(data)
            view_path = build_overlap_view(
                manifest_path,
                root / "view",
                work_directory=root,
                batch_rows=11,
            )
            primary = load_manifest(manifest_path, verify_hashes=True)
            view = load_evaluation_view(view_path, primary, verify_hashes=True)
            self.assertEqual(view.excluded_rows[0], 3)
            self.assertEqual(view.retained_rows[0], 17)
            audit = json.loads(
                (view_path.parent / "duplicate_exclusion_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            class0 = audit["classes"][0]
            self.assertEqual(class0["same_label_only_rows"], 1)
            self.assertEqual(class0["different_label_only_rows"], 1)
            self.assertEqual(class0["mixed_including_same_label_rows"], 1)
            self.assertEqual(
                audit["frozen_split_overlap_reconciliation"]["actual"],
                {
                    "test_rows": 80,
                    "excluded_rows": 3,
                    "overlap_unique_feature_rows": 3,
                },
            )
            self.assertTrue(
                audit["frozen_split_overlap_reconciliation"]["verified"]
            )

            official = run_manifest(
                manifest_path,
                seeds=[17],
                output_dir=root / "official",
                config=_config(),
            )["results"][0]
            dual = run_manifest(
                manifest_path,
                seeds=[17],
                output_dir=root / "dual-one",
                config=_config(),
                evaluation_view_paths=[view_path],
            )["results"][0]
            for key in (
                "normalization",
                "pretrain_history",
                "training_history",
                "training_exposure_records",
                "training_prior_records",
                "exemplar_records",
                "router_records",
            ):
                self.assertEqual(official[key], dual[key], key)
            self.assertEqual(
                [item["views"]["official"] for item in official["checkpoints"]],
                [item["views"]["official"] for item in dual["checkpoints"]],
            )
            self.assertEqual(set(ARMS), set(dual["summary"]["views"]["official"]))
            self.assertEqual(
                set(ARMS), set(dual["summary"]["views"]["duplicate_excluded"])
            )
            for class_id, exposure in dual["training_exposure_records"].items():
                histogram = {
                    int(key): int(value)
                    for key, value in exposure[
                        "negative_exposure_multiplicity_histogram"
                    ].items()
                }
                self.assertEqual(
                    sum(histogram.values()), exposure["negative_candidate_rows"]
                )
                self.assertEqual(
                    sum(multiplicity * count for multiplicity, count in histogram.items()),
                    exposure["negative_exposures_used_by_loss"],
                )
                prior = dual["training_prior_records"][class_id]
                self.assertEqual(
                    prior["positive_exposures"],
                    exposure["positive_exposures_used_by_loss"],
                )
                self.assertEqual(
                    prior["negative_exposures"],
                    exposure["negative_exposures_used_by_loss"],
                )
                self.assertAlmostEqual(
                    prior["log_positive_to_negative_exposure_ratio"],
                    np.log(
                        prior["positive_exposures"]
                        / prior["negative_exposures"]
                    ),
                )
                for epoch in dual["training_history"][class_id]:
                    epoch_exposure = epoch["exposure"]
                    self.assertEqual(
                        sum(
                            item["selected_rows"]
                            for item in epoch_exposure["negative_selected_by_source"]
                        ),
                        epoch["negative_rows"],
                    )
                    self.assertEqual(
                        epoch["loss_diagnostics"]["positive"]["examples"],
                        epoch["positive_rows"],
                    )
                    self.assertEqual(
                        epoch["loss_diagnostics"]["negative"]["examples"],
                        epoch["negative_rows"],
                    )
            for checkpoint in dual["checkpoints"]:
                for arm in ARMS:
                    official_matrix = np.asarray(
                        checkpoint["views"]["official"]["arms"][arm][
                            "confusion_matrix"
                        ]
                    )
                    retained_matrix = np.asarray(
                        checkpoint["views"]["duplicate_excluded"]["arms"][arm][
                            "confusion_matrix"
                        ]
                    )
                    excluded_matrix = np.asarray(
                        checkpoint["view_decomposition"]["duplicate_excluded"][
                            "arms"
                        ][arm]["excluded_confusion_matrix"]
                    )
                    np.testing.assert_array_equal(
                        official_matrix, retained_matrix + excluded_matrix
                    )

            campaign = run_manifest(
                manifest_path,
                seeds=[0, 1, 2, 3, 4],
                output_dir=root / "campaign" / "synthetic_streaming_smoke_only",
                config=_config(),
                evaluation_view_paths=[view_path],
            )
            report = summarize_campaign(
                root / "campaign",
                root / "paired.json",
                expected_seeds=[0, 1, 2, 3, 4],
                expected_datasets=["synthetic_streaming_smoke_only"],
            )
            self.assertEqual(report["schema_version"], 2)
            dataset = report["datasets"]["synthetic_streaming_smoke_only"]
            self.assertIn("duplicate_excluded_minus_official", dataset["view_sensitivity"])
            self.assertEqual(campaign["summary"]["schema_version"], 2)

    def test_mask_tamper_zero_support_and_output_lock(self):
        with tempfile.TemporaryDirectory(prefix="ofra_view_tamper_") as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            manifest_path = _manifest_with_label_aware_overlap(data)
            view_path = build_overlap_view(
                manifest_path, root / "view", work_directory=root, batch_rows=13
            )
            primary = load_manifest(manifest_path, verify_hashes=True)
            view_raw = json.loads(view_path.read_text(encoding="utf-8"))
            mask_record = view_raw["classes"][0]["shards"][0]["mask"]
            mask_path = view_path.parent / mask_record["path"]
            mask = np.load(mask_path, allow_pickle=False)
            mask[4] = ~mask[4]
            np.save(mask_path, mask, allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "mask file SHA-256"):
                load_evaluation_view(view_path, primary, verify_hashes=True)

            zero_path = build_overlap_view(
                manifest_path, root / "zero-view", work_directory=root, batch_rows=13
            )
            zero_raw = json.loads(zero_path.read_text(encoding="utf-8"))
            class_zero = zero_raw["classes"][0]
            old_retained = class_zero["retained_rows"]
            for shard_record in class_zero["shards"]:
                record = shard_record["mask"]
                path = zero_path.parent / record["path"]
                mask = np.load(path, allow_pickle=False)
                mask[:] = False
                np.save(path, mask, allow_pickle=False)
                record.update(
                    {
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "true_count": 0,
                        "false_count": record["rows"],
                    }
                )
            class_zero["retained_rows"] = 0
            class_zero["excluded_rows"] = class_zero["official_rows"]
            zero_raw["totals"]["retained_rows"] -= old_retained
            zero_raw["totals"]["excluded_rows"] += old_retained
            _rewrite_view_manifest(zero_path, zero_raw)
            with self.assertRaisesRegex(ValueError, "zero retained rows"):
                load_evaluation_view(zero_path, primary, verify_hashes=True)

            escape_path = build_overlap_view(
                manifest_path,
                root / "escape-view",
                work_directory=root,
                batch_rows=13,
            )
            escape_raw = json.loads(escape_path.read_text(encoding="utf-8"))
            escape_raw["classes"][0]["shards"][0]["mask"]["path"] = (
                "../outside.keep.npy"
            )
            _rewrite_view_manifest(escape_path, escape_raw)
            with self.assertRaisesRegex(ValueError, "escapes"):
                load_evaluation_view(escape_path, primary, verify_hashes=True)

            audit_path = build_overlap_view(
                manifest_path,
                root / "audit-view",
                work_directory=root,
                batch_rows=13,
            )
            audit_raw = json.loads(audit_path.read_text(encoding="utf-8"))
            audit_file = audit_path.parent / audit_raw["selection"]["audit"]["path"]
            audit = json.loads(audit_file.read_text(encoding="utf-8"))
            audit["classes"][0]["same_label_only_rows"] += 1
            audit.pop("canonical_report_sha256", None)
            audit["canonical_report_sha256"] = canonical_sha256(audit)
            _write_json(audit_file, audit)
            audit_raw["selection"]["audit"].update(
                {
                    "sha256": sha256_file(audit_file),
                    "canonical_sha256": audit["canonical_report_sha256"],
                }
            )
            _rewrite_view_manifest(audit_path, audit_raw)
            with self.assertRaisesRegex(ValueError, "deterministic content hash"):
                load_evaluation_view(audit_path, primary, verify_hashes=True)

            lock_root = root / "locked"
            lock_root.mkdir()
            code = (
                "from pathlib import Path; import sys; "
                "from streaming_full.validation import OutputDirectoryLock; "
                f"p=Path({str(lock_root)!r}); "
                "\ntry:\n"
                "  with OutputDirectoryLock(p): pass\n"
                "except RuntimeError:\n  sys.exit(23)\n"
            )
            with OutputDirectoryLock(lock_root):
                completed = subprocess.run(
                    [sys.executable, "-c", code],
                    cwd=Path(__file__).parents[1],
                    check=False,
                )
            self.assertEqual(completed.returncode, 23)
            with OutputDirectoryLock(lock_root):
                pass


if __name__ == "__main__":
    unittest.main()
