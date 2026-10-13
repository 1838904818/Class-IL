"""Protect the frozen learners, checkpoint resume, and signed PI definitions."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from ncd.crr_benchmarks import METHODS
from ncd.data import PreparedData
from ncd.run_crr_real_sequence import _pi_vectors, run_cell as original_cell
from ncd.run_exp_h import _checkpoint_scorer, _dependency_vectors, _train_checkpoints


class FrozenRunnerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        rng = np.random.default_rng(821)
        self.data = PreparedData(
            x_train=rng.normal(size=(48, 5)).astype(np.float32),
            y_train=np.repeat(np.arange(4, dtype=np.int64), 12),
            x_test=rng.normal(size=(32, 5)).astype(np.float32),
            y_test=np.repeat(np.arange(4, dtype=np.int64), 8),
            class_order=["a", "b", "c", "d"], tasks=[[0, 1], [2], [3]],
            feature_names=[f"x{i}" for i in range(5)], source_files=[],
        )
        self.config = dict(
            hidden=12, batch_size=16, epochs_per_task=1, learning_rate=0.001,
            buffer_per_class=4, temperature=2.0, rcl_features=3,
            rcl_temperature=0.5, rcl_beta=1.0, rcl_prediction_weight=0.5,
            device="cpu", data_root="unused",
        )

    def test_all_frozen_methods_and_resume_match_original(self):
        for method in METHODS:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                regularization = 0.0 if method == "replay" else 0.25
                identity = dict(method=method, seed=1, regularization=regularization, setting="test")
                with patch("ncd.run_crr_real_sequence.prepare_dataset", return_value=self.data):
                    original = original_cell(self.config, "NSL-KDD", method, 1, regularization)
                paths, history = _train_checkpoints(self.data, self.config, identity, root / "full", torch.device("cpu"), False)
                self.assertEqual(history, original["rows"][0]["task_history"])
                resume = root / "resume" / "checkpoints"
                resume.mkdir(parents=True)
                for path in paths[:-1]:
                    shutil.copy2(path, resume / path.name)
                resumed_paths, resumed_history = _train_checkpoints(self.data, self.config, identity, root / "resume", torch.device("cpu"), False)
                self.assertEqual(history, resumed_history)
                full_last = torch.load(paths[-1], weights_only=True)
                resume_last = torch.load(resumed_paths[-1], weights_only=True)
                for key in full_last["model_state"]:
                    self.assertTrue(torch.equal(full_last["model_state"][key], resume_last["model_state"][key]))
                scorer = _checkpoint_scorer(full_last, torch.device("cpu"))
                pairs, self_vectors = _dependency_vectors(scorer, self.data, (0, 1, 2), 102, torch.device("cpu"), 16)
                original_pairs = _pi_vectors(scorer, self.data, (0, 1, 2), 102, torch.device("cpu"), 16)
                for pair in pairs:
                    np.testing.assert_array_equal(pairs[pair], original_pairs[pair])
                self.assertEqual(set(self_vectors), {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
