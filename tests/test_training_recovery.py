from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from streaming_full.smoke_test import _make_synthetic_manifest
from streaming_full.validation import RunConfig, run_manifest


def _config() -> RunConfig:
    return RunConfig(
        pretrain_epochs=2,
        epochs_per_task=2,
        batch_size=16,
        eval_batch_size=16,
        encoder_type="mlp",
        d_model=16,
        n_layers=1,
        ft_heads=2,
        ft_dim_head=8,
        ft_attn_dropout=0.1,
        ft_ff_dropout=0.1,
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
        monitor_enabled=False,
    )


class TrainingRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def _exercise_pause(self, pause_after: int) -> None:
        with tempfile.TemporaryDirectory(prefix="ofra_training_recovery_") as temporary:
            root = Path(temporary)
            cache = root / "cache"
            cache.mkdir()
            manifest = _make_synthetic_manifest(cache)
            uninterrupted = run_manifest(
                manifest,
                seeds=[41],
                output_dir=root / "uninterrupted",
                config=_config(),
            )
            resumed_dir = root / "resumed"
            paused = run_manifest(
                manifest,
                seeds=[41],
                output_dir=resumed_dir,
                config=_config(),
                recovery_enabled=True,
                recovery_pause_after_checkpoints=pause_after,
            )
            self.assertIsNotNone(paused["paused"])
            self.assertIsNone(paused["summary"])
            resumed = run_manifest(
                manifest,
                seeds=[41],
                output_dir=resumed_dir,
                config=_config(),
                recovery_enabled=True,
            )
            self.assertEqual(
                uninterrupted["summary"]["protocol_sha256"],
                resumed["summary"]["protocol_sha256"],
            )
            self.assertEqual(
                uninterrupted["results"][0]["deterministic_result_sha256"],
                resumed["results"][0]["deterministic_result_sha256"],
            )

    def test_resume_after_pretrain_epoch_matches_uninterrupted(self) -> None:
        self._exercise_pause(pause_after=2)

    def test_resume_after_family_epoch_matches_uninterrupted(self) -> None:
        self._exercise_pause(pause_after=5)

    def test_adamw_resume_after_pretrain_epoch_matches_uninterrupted(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ofra_adamw_recovery_") as temporary:
            root = Path(temporary)
            cache = root / "cache"
            cache.mkdir()
            manifest = _make_synthetic_manifest(cache)
            config = replace(
                _config(),
                optimizer_name="adamw",
                learning_rate=5e-4,
                weight_decay=1e-5,
            )
            uninterrupted = run_manifest(
                manifest,
                seeds=[47],
                output_dir=root / "uninterrupted",
                config=config,
            )
            resumed_dir = root / "resumed"
            paused = run_manifest(
                manifest,
                seeds=[47],
                output_dir=resumed_dir,
                config=config,
                recovery_enabled=True,
                recovery_pause_after_checkpoints=2,
            )
            self.assertIsNotNone(paused["paused"])
            resumed = run_manifest(
                manifest,
                seeds=[47],
                output_dir=resumed_dir,
                config=config,
                recovery_enabled=True,
            )
            self.assertEqual(
                uninterrupted["results"][0]["deterministic_result_sha256"],
                resumed["results"][0]["deterministic_result_sha256"],
            )

    def test_tampered_recovery_payload_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ofra_training_recovery_tamper_") as temporary:
            root = Path(temporary)
            cache = root / "cache"
            cache.mkdir()
            manifest = _make_synthetic_manifest(cache)
            output = root / "resumed"
            paused = run_manifest(
                manifest,
                seeds=[43],
                output_dir=output,
                config=_config(),
                recovery_enabled=True,
                recovery_pause_after_checkpoints=2,
            )
            self.assertIsNotNone(paused["paused"])
            payload = output / "recovery" / "seed_43" / "recovery_state.pt"
            original = payload.read_bytes()
            changed = bytearray(original)
            changed[len(changed) // 2] ^= 0x01
            payload.write_bytes(changed)
            with self.assertRaisesRegex(RuntimeError, "payload SHA-256 mismatch"):
                run_manifest(
                    manifest,
                    seeds=[43],
                    output_dir=output,
                    config=_config(),
                    recovery_enabled=True,
                )


if __name__ == "__main__":
    unittest.main()
