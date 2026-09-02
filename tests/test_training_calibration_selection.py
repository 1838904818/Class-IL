from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from streaming_full.data import sha256_file
from streaming_full.smoke_test import _make_synthetic_manifest
from streaming_full.validation import (
    RunConfig,
    _checkpoint_guard_decision,
    run_manifest,
)


def _calibration_audit(manifest_path: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    classes = []
    for item in manifest["classes"]:
        class_id = int(item["id"])
        fit_values = np.vstack(
            [
                np.load(root / shard["path"], allow_pickle=False)
                for shard in item["train"]
            ]
        ).astype(np.float32)
        fit_relative = Path(f"fit_{class_id:02d}.npy")
        fit_path = root / fit_relative
        np.save(fit_path, fit_values, allow_pickle=False)
        item["train"] = [
            {
                "path": fit_relative.as_posix(),
                "rows": len(fit_values),
                "sha256": sha256_file(fit_path),
            }
        ]
        rng = np.random.default_rng(9000 + class_id)
        values = rng.normal(
            loc=float(class_id),
            scale=0.25,
            size=(6, int(manifest["feature_dim"])),
        ).astype(np.float32)
        relative = Path(f"calibration_{class_id:02d}.npy")
        path = root / relative
        np.save(path, values, allow_pickle=False)
        fit = item["train"][0]
        classes.append(
            {
                "id": class_id,
                "name": item["name"],
                "fit": fit,
                "fit_rows": fit["rows"],
                "fit_calibration_disjoint": True,
                "calibration": {
                    "path": relative.as_posix(),
                    "sha256": sha256_file(path),
                },
                "calibration_rows": len(values),
                "calibration_indices_sha256": hashlib.sha256(
                    f"synthetic-calibration-{class_id}".encode()
                ).hexdigest(),
                "official_test": [
                    {
                        "derived_path": shard["path"],
                        "rows": shard["rows"],
                        "sha256": shard["sha256"],
                        "byte_identical": True,
                    }
                    for shard in item["test"]
                ],
            }
        )
    audit = {
        "schema_version": 1,
        "algorithm": "synthetic_train_calibration_v1",
        "configuration": {"calibration_fraction": 0.10},
        "invariants": {
            "calibration_source": "source training rows only",
            "fit_calibration_disjoint": True,
            "official_test_sampled": False,
        },
        "classes": classes,
    }
    path = root / "sampling_audit.json"
    path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    manifest["source"] = {
        "builder": "synthetic-calibration-test",
        "algorithm": audit["algorithm"],
        "sampling_audit_sha256": sha256_file(path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return path


def _config() -> RunConfig:
    return RunConfig(
        pretrain_epochs=1,
        epochs_per_task=2,
        batch_size=16,
        eval_batch_size=16,
        encoder_type="mlp",
        d_model=16,
        n_layers=1,
        lora_rank=2,
        lora_alpha=4.0,
        negative_ratio=1,
        exemplar_capacity=4,
        exemplar_candidate_capacity=8,
        router_cap_samples=8,
        router_max_centroids=4,
        device="cpu",
        deterministic=True,
        verify_shard_hashes=True,
        verbose=False,
        monitor_enabled=False,
        family_checkpoint_selection="training_only_calibration_macro_f1",
        family_validation_cap_per_label=4,
        family_validation_min_per_label=2,
    )


def _guard_config() -> RunConfig:
    return replace(
        _config(),
        family_checkpoint_selection=(
            "training_only_calibration_macro_f1_recall_fpr_guard"
        ),
        family_checkpoint_min_macro_f1_gain=0.01,
        family_checkpoint_max_positive_recall_drop=0.01,
        family_checkpoint_max_negative_fpr_increase=0.01,
    )


class TrainingCalibrationSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_selection_uses_seen_training_only_calibration(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ofra_calibration_selection_") as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            manifest = _make_synthetic_manifest(cache)
            audit = _calibration_audit(manifest)
            output = run_manifest(
                manifest,
                seeds=[53],
                output_dir=root / "output",
                config=_config(),
                training_calibration_audit_path=audit,
            )
            protocol = output["protocol"]
            self.assertTrue(protocol["training_calibration"]["enabled"])
            self.assertFalse(protocol["training_calibration"]["official_test_used"])
            result = output["results"][0]
            for class_id, expected_seen in {
                "0": [0, 1],
                "1": [0, 1],
                "2": [0, 1, 2, 3],
                "3": [0, 1, 2, 3],
            }.items():
                history = result["training_history"][class_id]
                self.assertEqual(len(history), 2)
                selection = result["training_exposure_records"][class_id][
                    "checkpoint_selection"
                ]
                self.assertEqual(selection["seen_classes"], expected_seen)
                self.assertTrue(selection["applied"])
                self.assertFalse(selection["official_test_used"])
                self.assertFalse(selection["future_classes_used"])
                values = [
                    epoch["training_only_calibration"]["binary_macro_f1"]
                    for epoch in history
                ]
                expected_epoch = max(range(len(values)), key=lambda index: values[index]) + 1
                self.assertEqual(selection["selected_epoch"], expected_epoch)

    def test_recovery_preserves_best_calibration_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ofra_calibration_recovery_") as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            manifest = _make_synthetic_manifest(cache)
            audit = _calibration_audit(manifest)
            uninterrupted = run_manifest(
                manifest,
                seeds=[59],
                output_dir=root / "uninterrupted",
                config=_config(),
                training_calibration_audit_path=audit,
            )
            resumed_dir = root / "resumed"
            paused = run_manifest(
                manifest,
                seeds=[59],
                output_dir=resumed_dir,
                config=_config(),
                training_calibration_audit_path=audit,
                recovery_enabled=True,
                recovery_pause_after_checkpoints=4,
            )
            self.assertIsNotNone(paused["paused"])
            resumed = run_manifest(
                manifest,
                seeds=[59],
                output_dir=resumed_dir,
                config=_config(),
                training_calibration_audit_path=audit,
                recovery_enabled=True,
            )
            self.assertEqual(
                uninterrupted["results"][0]["deterministic_result_sha256"],
                resumed["results"][0]["deterministic_result_sha256"],
            )

    def test_calibration_audit_is_required_exactly_for_selection(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ofra_calibration_gate_") as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            manifest = _make_synthetic_manifest(cache)
            with self.assertRaisesRegex(ValueError, "must be provided exactly"):
                run_manifest(
                    manifest,
                    seeds=[61],
                    output_dir=root / "missing",
                    config=_config(),
                )

    def test_checkpoint_guard_accepts_only_a_bounded_improvement(self) -> None:
        candidate = {
            "binary_macro_f1": 0.72,
            "positive_recall": 0.79,
            "negative_false_positive_rate": 0.11,
        }
        last = {
            "binary_macro_f1": 0.70,
            "positive_recall": 0.80,
            "negative_false_positive_rate": 0.10,
        }
        decision = _checkpoint_guard_decision(
            7,
            candidate,
            10,
            last,
            min_macro_f1_gain=0.01,
            max_positive_recall_drop=0.01,
            max_negative_fpr_increase=0.01,
        )
        self.assertTrue(decision["passed"])
        self.assertEqual(decision["decision"], "restore_candidate")

    def test_checkpoint_guard_rejects_recall_or_fpr_regression(self) -> None:
        last = {
            "binary_macro_f1": 0.70,
            "positive_recall": 0.80,
            "negative_false_positive_rate": 0.10,
        }
        for candidate in (
            {
                "binary_macro_f1": 0.72,
                "positive_recall": 0.78,
                "negative_false_positive_rate": 0.10,
            },
            {
                "binary_macro_f1": 0.72,
                "positive_recall": 0.80,
                "negative_false_positive_rate": 0.12,
            },
            {
                "binary_macro_f1": 0.705,
                "positive_recall": 0.80,
                "negative_false_positive_rate": 0.10,
            },
        ):
            with self.subTest(candidate=candidate):
                decision = _checkpoint_guard_decision(
                    7,
                    candidate,
                    10,
                    last,
                    min_macro_f1_gain=0.01,
                    max_positive_recall_drop=0.01,
                    max_negative_fpr_increase=0.01,
                )
                self.assertFalse(decision["passed"])
                self.assertEqual(decision["decision"], "retain_last_epoch")

    def test_guarded_selection_records_auditable_decision(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ofra_checkpoint_guard_") as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            manifest = _make_synthetic_manifest(cache)
            audit = _calibration_audit(manifest)
            output = run_manifest(
                manifest,
                seeds=[67],
                output_dir=root / "output",
                config=_guard_config(),
                training_calibration_audit_path=audit,
            )
            result = output["results"][0]
            for record in result["training_exposure_records"].values():
                selection = record["checkpoint_selection"]
                self.assertTrue(selection["applied"])
                self.assertIsInstance(selection["guard_decision"], dict)
                self.assertIn(
                    selection["guard_decision"]["decision"],
                    {"restore_candidate", "retain_last_epoch"},
                )
                self.assertEqual(
                    selection["selected_epoch"],
                    selection["candidate_epoch"]
                    if selection["guard_decision"]["passed"]
                    else selection["max_epochs"],
                )


if __name__ == "__main__":
    unittest.main()
