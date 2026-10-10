from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ncd.run_exp3_phase1_dataset import (
    explanation_complete,
    training_complete,
)
from ncd.source_commit import source_commit


COMMIT = "1" * 40


class NcdExp3DatasetRunnerTest(unittest.TestCase):
    def test_packaged_source_commit_is_validated(self) -> None:
        with mock.patch.dict(os.environ, {"NCD_SOURCE_COMMIT": COMMIT}):
            self.assertEqual(source_commit(Path(".")), COMMIT)
        with mock.patch.dict(os.environ, {"NCD_SOURCE_COMMIT": "short"}):
            with self.assertRaises(ValueError):
                source_commit(Path("."))

    def test_completed_training_and_explanation_are_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "checkpoint_000.pt").write_bytes(b"checkpoint")
            (run_dir / "probe_background.npz").write_bytes(b"probe")
            training = {
                "dataset": "NSL-KDD",
                "method": "naive",
                "seed": 1,
                "git_commit": COMMIT,
                "deterministic_algorithms_enabled": True,
                "probe_background": "probe_background.npz",
                "checkpoints": [
                    {"checkpoint": 0, "path": "checkpoint_000.pt", "seen_classes": [0]}
                ],
            }
            training_path = run_dir / "training_result.json"
            training_path.write_text(json.dumps(training), encoding="utf-8")
            self.assertTrue(
                training_complete(
                    training_path,
                    dataset="NSL-KDD",
                    method="naive",
                    seed=1,
                    commit=COMMIT,
                )
            )

            records = []
            for explainer in ("expected_gradients", "feature_ablation", "grad_x_input"):
                artifact = run_dir / f"{explainer}.npz"
                artifact.write_bytes(b"artifact")
                records.append(
                    {"path": artifact.name, "explainer": explainer, "estimate_count": 100}
                )
            explanation = {
                "dataset": "NSL-KDD",
                "method": "naive",
                "seed": 1,
                "git_commit": COMMIT,
                "training_git_commit": COMMIT,
                "explainers": ["expected_gradients", "feature_ablation", "grad_x_input"],
                "repeats": 10,
                "probe_bootstraps": 10,
                "expected_gradients_nsamples": 32,
                "deterministic_algorithms_enabled": True,
                "records": records,
            }
            explanation_path = run_dir / "explanation_result.json"
            explanation_path.write_text(json.dumps(explanation), encoding="utf-8")
            self.assertTrue(
                explanation_complete(
                    explanation_path,
                    training_path=training_path,
                    dataset="NSL-KDD",
                    method="naive",
                    seed=1,
                    commit=COMMIT,
                )
            )


if __name__ == "__main__":
    unittest.main()
