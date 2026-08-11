"""Regression coverage for the non-reportable capacity profile."""
from __future__ import annotations

import json
import site
import tempfile
import unittest
from pathlib import Path

from streaming_full.capacity_profile import PROFILE_KIND, run_capacity_profile
from streaming_full.smoke_test import _make_synthetic_manifest
from streaming_full.validation import RunConfig


class CapacityProfileTests(unittest.TestCase):
    def test_cpu_profile_uses_bounded_formal_pretrain_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _make_synthetic_manifest(root)
            output = root / "profile.json"
            config = RunConfig(
                pretrain_epochs=1,
                epochs_per_task=0,
                batch_size=16,
                eval_batch_size=16,
                shuffle_block_rows=16,
                encoder_type="mlp",
                d_model=8,
                n_layers=1,
                device="cpu",
                deterministic=True,
                verify_shard_hashes=True,
            )
            result = run_capacity_profile(
                manifest,
                config=config,
                seed=7,
                warmup_batches=1,
                timed_batches=2,
                output_path=output,
            )
            self.assertTrue(output.is_file())
            self.assertTrue(result["non_reportable"])
            self.assertEqual(result["profile_kind"], PROFILE_KIND)
            self.assertEqual(result["input"]["sampling"]["warmup_batches"], 1)
            self.assertEqual(result["measurement"]["timed_batches"], 2)
            self.assertEqual(result["measurement"]["timed_rows"], 32)
            self.assertGreater(result["measurement"]["training_rows_per_second"], 0.0)
            self.assertIn("profile_total_seconds", result["phases"])
            self.assertNotIn("mean_loss", result["warmup"])
            self.assertNotIn("mean_loss", result["measurement"])
            serialized = json.dumps(result, sort_keys=True)
            for forbidden in ("loss", "accuracy", "forgetting", "shap", "etg"):
                self.assertNotIn(forbidden, serialized.lower())
            self.assertEqual(result["resources"], {"device": "cpu"})
            self.assertIs(
                result["environment"]["python"]["user_site_enabled"],
                bool(site.ENABLE_USER_SITE),
            )
            self.assertEqual(len(result["canonical_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
