from __future__ import annotations

import tempfile
import unittest
from importlib import metadata
from pathlib import Path

import torch

from streaming_full.smoke_test import _make_synthetic_manifest
from streaming_full.validation import RunConfig, run_manifest


def _locked_ft512x12_config() -> RunConfig:
    """Small-data verification using the exact locked formal architecture."""
    return RunConfig(
        pretrain_epochs=2,
        epochs_per_task=2,
        batch_size=16,
        eval_batch_size=16,
        encoder_type="ft_transformer",
        d_model=512,
        n_layers=12,
        ft_heads=16,
        ft_dim_head=32,
        ft_attn_dropout=0.1,
        ft_ff_dropout=0.1,
        lora_rank=8,
        lora_alpha=16.0,
        exemplar_capacity=4,
        exemplar_candidate_capacity=8,
        router_cap_samples=8,
        router_max_centroids=4,
        device="cuda:0",
        deterministic=True,
        verify_shard_hashes=True,
        verbose=False,
        monitor_enabled=False,
    )


class LockedFT512x12TrainingRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("the locked FT512x12 recovery test requires CUDA")
        try:
            metadata.version("tab-transformer-pytorch")
        except metadata.PackageNotFoundError:
            self.skipTest("the locked FT512x12 dependency is not installed")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_pretrain_and_family_resume_match_uninterrupted(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ofra_ft512x12_recovery_") as temporary:
            root = Path(temporary)
            cache = root / "cache"
            cache.mkdir()
            manifest = _make_synthetic_manifest(cache)
            config = _locked_ft512x12_config()

            uninterrupted = run_manifest(
                manifest,
                seeds=[41],
                output_dir=root / "uninterrupted",
                config=config,
            )
            expected = uninterrupted["results"][0]["deterministic_result_sha256"]

            # Checkpoints: normalization=1, pretrain epoch 1=2, pretrain epoch 2=3,
            # task boundary=4, first family epoch=5. Both pauses therefore leave
            # another epoch in the same phase for the restarted run to execute.
            for label, pause_after in (("pretrain", 2), ("family", 5)):
                output = root / label
                paused = run_manifest(
                    manifest,
                    seeds=[41],
                    output_dir=output,
                    config=config,
                    recovery_enabled=True,
                    recovery_pause_after_checkpoints=pause_after,
                )
                self.assertIsNotNone(paused["paused"])
                resumed = run_manifest(
                    manifest,
                    seeds=[41],
                    output_dir=output,
                    config=config,
                    recovery_enabled=True,
                )
                self.assertEqual(
                    uninterrupted["summary"]["protocol_sha256"],
                    resumed["summary"]["protocol_sha256"],
                )
                self.assertEqual(
                    expected,
                    resumed["results"][0]["deterministic_result_sha256"],
                )


if __name__ == "__main__":
    unittest.main()
