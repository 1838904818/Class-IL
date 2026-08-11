"""CPU-only end-to-end checks for auditable checkpoint monitoring."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from streaming_full.data import ClassShards, load_manifest
from streaming_full.monitoring import load_checkpoint, validate_checkpoint_manifest
from streaming_full.smoke_test import _make_synthetic_manifest
from streaming_full.validation import RunConfig, run_manifest


def _config(*, enabled: bool) -> RunConfig:
    return RunConfig(
        pretrain_epochs=1,
        epochs_per_task=1,
        batch_size=16,
        eval_batch_size=16,
        encoder_type="ft_transformer",
        d_model=16,
        n_layers=1,
        ft_heads=2,
        ft_dim_head=8,
        ft_attn_dropout=0.0,
        ft_ff_dropout=0.0,
        lora_rank=2,
        lora_alpha=4.0,
        exemplar_capacity=4,
        exemplar_candidate_capacity=8,
        router_cap_samples=8,
        router_max_centroids=4,
        device="cpu",
        deterministic=True,
        verify_shard_hashes=True,
        verbose=False,
        monitor_enabled=enabled,
        monitor_official_test_per_class=3,
        monitor_task0_background_per_class=2,
    )


def _probe_rows(manifest_path: Path, contract: dict, seen: list[int]) -> np.ndarray:
    manifest = load_manifest(manifest_path, verify_hashes=True)
    sources = {
        record.class_id: ClassShards(record.test, manifest.feature_dim)
        for record in manifest.classes
    }
    try:
        rows = []
        for record in contract["official_test"]["samples"]:
            class_id = int(record["class_id"])
            if class_id not in seen:
                continue
            source = sources[class_id]
            ordinal = int(record["shard_ordinal"])
            global_index = int(source.offsets[ordinal] + int(record["local_row"]))
            rows.append(source.take(np.asarray([global_index], dtype=np.int64)))
        return np.vstack(rows).astype(np.float32, copy=False)
    finally:
        for source in sources.values():
            source.close()


def _without_run_identity(value: dict) -> dict:
    result = copy.deepcopy(value)
    for key in ("protocol_sha256", "monitoring", "deterministic_result_sha256", "timing"):
        result.pop(key, None)
    return result


class CheckpointMonitoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cls.temporary = tempfile.TemporaryDirectory(prefix="ofra_monitor_test_")
        cls.root = Path(cls.temporary.name)
        cache = cls.root / "cache"
        cache.mkdir()
        cls.manifest_path = _make_synthetic_manifest(cache)
        cls.first_dir = cls.root / "enabled_first"
        cls.second_dir = cls.root / "enabled_second"
        cls.disabled_dir = cls.root / "disabled"
        cls.first = run_manifest(
            cls.manifest_path,
            seeds=[31],
            output_dir=cls.first_dir,
            config=_config(enabled=True),
        )
        cls.second = run_manifest(
            cls.manifest_path,
            seeds=[31],
            output_dir=cls.second_dir,
            config=_config(enabled=True),
        )
        cls.disabled = run_manifest(
            cls.manifest_path,
            seeds=[31],
            output_dir=cls.disabled_dir,
            config=_config(enabled=False),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_double_run_hashes_match(self):
        self.assertEqual(
            self.first["summary"]["protocol_sha256"],
            self.second["summary"]["protocol_sha256"],
        )
        self.assertEqual(
            self.first["results"][0]["deterministic_result_sha256"],
            self.second["results"][0]["deterministic_result_sha256"],
        )
        first_monitor = self.first["results"][0]["monitoring"]
        self.assertEqual(first_monitor, self.second["results"][0]["monitoring"])
        self.assertTrue(first_monitor["enabled"])
        contract = self.first["protocol"]["monitoring"]["probe_contract"]
        self.assertFalse(contract["raw_feature_rows_persisted"])
        self.assertEqual(len(contract["task0_train_background"]["samples"]), 4)

    def test_reloaded_checkpoint_scores_are_elementwise_identical(self):
        monitor = self.first["results"][0]["monitoring"]
        contract = self.first["protocol"]["monitoring"]["probe_contract"]
        for checkpoint in monitor["checkpoints"]:
            manifest_file = self.first_dir / checkpoint[
                "checkpoint_manifest_relative_path"
            ]
            metadata = validate_checkpoint_manifest(manifest_file)
            raw = _probe_rows(
                self.manifest_path,
                contract,
                [int(value) for value in metadata["seen_classes"]],
            )
            reconstructed = load_checkpoint(manifest_file, device="cpu").score(raw)
            with np.load(
                manifest_file.parent / metadata["probe_scores_file"],
                allow_pickle=False,
            ) as saved:
                for name in (
                    "class_axis",
                    "head_scores",
                    "router_z_scores",
                    "joint_scores",
                    "predicted_class_id",
                ):
                    self.assertTrue(np.array_equal(reconstructed[name], saved[name]), name)

    def test_tampered_score_artifact_fails_closed(self):
        monitor = self.first["results"][0]["monitoring"]
        manifest_file = self.first_dir / monitor["checkpoints"][0][
            "checkpoint_manifest_relative_path"
        ]
        scores = manifest_file.parent / "probe_scores.npz"
        original = scores.read_bytes()
        changed = bytearray(original)
        changed[len(changed) // 2] ^= 0x01
        scores.write_bytes(changed)
        try:
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                validate_checkpoint_manifest(manifest_file)
        finally:
            scores.write_bytes(original)
        validate_checkpoint_manifest(manifest_file)

    def test_disabled_monitor_has_no_artifacts_or_numeric_side_effect(self):
        self.assertEqual(
            self.disabled["results"][0]["monitoring"], {"enabled": False}
        )
        self.assertFalse((self.disabled_dir / "monitoring").exists())
        self.assertEqual(
            _without_run_identity(self.first["results"][0]),
            _without_run_identity(self.disabled["results"][0]),
        )


if __name__ == "__main__":
    unittest.main()
