from __future__ import annotations

import unittest

import numpy as np
import torch

from ncd.model import MLP
from ncd.train_cil import (
    SUPPORTED_METHODS,
    compute_fisher,
    herding_select,
    seed_everything,
    train_epoch,
)


class NcdTrainCilTest(unittest.TestCase):
    def test_phase_i_method_set_is_explicit(self) -> None:
        self.assertEqual(
            SUPPORTED_METHODS,
            {
                "naive",
                "ewc",
                "lwf",
                "replay",
                "replay-herding",
                "replay-dpmeans",
                "icarl",
            },
        )

    def test_herding_selects_unique_bounded_indices(self) -> None:
        features = np.arange(48, dtype=np.float32).reshape(12, 4)
        selected = herding_select(features, 5)
        self.assertEqual(selected.shape, (5,))
        self.assertEqual(len(set(selected.tolist())), 5)
        self.assertTrue(np.all(selected >= 0))
        self.assertTrue(np.all(selected < len(features)))

    def test_fisher_and_ewc_training_are_finite(self) -> None:
        seed_everything(7)
        rng = np.random.default_rng(7)
        x = rng.normal(size=(24, 4)).astype(np.float32)
        y = np.tile(np.arange(3, dtype=np.int64), 8)
        model = MLP(4, 8, 3)
        fisher = compute_fisher(
            model,
            x,
            y,
            device=torch.device("cpu"),
            batch_size=8,
            sample_size=18,
            seed=11,
        )
        parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        self.assertEqual(set(fisher), set(parameters))
        for value in fisher.values():
            self.assertTrue(torch.isfinite(value).all())
            self.assertTrue((value >= 0).all())

        before = {name: value.clone() for name, value in parameters.items()}
        train_epoch(
            model,
            x,
            y,
            device=torch.device("cpu"),
            batch_size=8,
            lr=1e-3,
            epochs=1,
            seed=13,
            ewc_snapshots=[(parameters, fisher)],
            ewc_lambda=5000.0,
        )
        self.assertTrue(
            any(
                not torch.equal(before[name], parameter.detach())
                for name, parameter in model.named_parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
