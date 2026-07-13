"""Regression checks for OFRA protocol and input guardrails."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from src_v2.methods.ofra import (
    OFRAAgent,
    RawExemplarBuffer,
    focal_loss,
    run_ofra,
)


def make_agent() -> OFRAAgent:
    return OFRAAgent(
        n_features=2,
        d_model=4,
        n_layers=1,
        lora_rank=2,
        exemplar_capacity=2,
        encoder_type="mlp",
    )


def register_family(
    agent: OFRAAgent,
    class_id: int = 0,
    centroids: np.ndarray | None = None,
) -> str:
    family = f"family_{class_id}"
    agent.pool.add_family(family, n_local_classes=2)
    agent.class_to_family[class_id] = family
    agent.family_to_class[family] = class_id
    if centroids is not None:
        agent.router.centroids[family] = centroids
    return family


class BufferValidationTests(unittest.TestCase):
    def test_capacity_and_shape_validation(self):
        with self.assertRaises(ValueError):
            RawExemplarBuffer(-1)

        buffer = RawExemplarBuffer(2)
        with self.assertRaises(ValueError):
            buffer.add(
                "family_0",
                np.zeros((2, 2), dtype=np.float32),
                np.zeros(1, dtype=np.int64),
                np.zeros((2, 4), dtype=np.float32),
            )


class AgentGuardrailTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(7)
        torch.manual_seed(7)

    def test_empty_embed_and_empty_pool_return_shapes(self):
        agent = make_agent()
        embedded = agent.embed(np.empty((0, 2), dtype=np.float32))
        self.assertEqual(embedded.shape, (0, 4))
        self.assertEqual(embedded.dtype, np.float32)

        predictions, routing = agent.predict(
            np.zeros((2, 2), dtype=np.float32),
            return_routing=True,
        )
        np.testing.assert_array_equal(predictions, np.array([-1, -1]))
        self.assertEqual(routing, [])

    def test_invalid_options_fail_before_state_changes(self):
        agent = make_agent()
        X = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        y = np.array([0, 1], dtype=np.int64)

        with self.assertRaises(ValueError):
            agent.train_task(X, y, epochs=0, loss_fn="fcoal")
        self.assertEqual(agent.pool.families, [])
        self.assertEqual(agent.class_to_family, {})

        with self.assertRaises(ValueError):
            agent.predict(X, calibration="probability")
        with self.assertRaises(ValueError):
            agent.predict(X, router_weight=0.0, head_weight=0.0)

    def test_missing_or_invalid_centroids_are_rejected(self):
        bad_values = {
            "missing": None,
            "empty": np.empty((0, 4), dtype=np.float32),
            "non_finite": np.array([[0.0, np.nan, 0.0, 0.0]], dtype=np.float32),
            "wrong_dimension": np.zeros((1, 3), dtype=np.float32),
        }
        X = np.zeros((1, 2), dtype=np.float32)

        for label, value in bad_values.items():
            with self.subTest(label=label):
                agent = make_agent()
                family = register_family(agent)
                if label != "missing":
                    agent.router.centroids[family] = value
                with self.assertRaises(RuntimeError):
                    agent.predict(X, router_weight=0.0)

    def test_single_class_without_replay_fails_atomically(self):
        agent = make_agent()
        X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        y = np.zeros(2, dtype=np.int64)

        with self.assertRaises(ValueError):
            agent.train_task(X, y, epochs=0)

        self.assertEqual(agent.pool.families, [])
        self.assertEqual(agent.class_to_family, {})
        self.assertEqual(agent.family_to_class, {})
        self.assertEqual(agent.router.centroids, {})
        self.assertEqual(agent.buffer.n_total(), 0)

    def test_repeated_class_fails_atomically(self):
        agent = make_agent()
        family = register_family(
            agent,
            centroids=np.zeros((1, 4), dtype=np.float32),
        )
        agent.buffer.samples[family] = np.zeros((1, 2), dtype=np.float32)
        agent.buffer.labels[family] = np.zeros(1, dtype=np.int64)

        families_before = tuple(agent.pool.families)
        class_map_before = dict(agent.class_to_family)
        centroids_before = agent.router.centroids[family].copy()
        samples_before = agent.buffer.samples[family].copy()

        X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        y = np.array([0, 2], dtype=np.int64)
        with self.assertRaises(ValueError):
            agent.train_task(X, y, epochs=0)

        self.assertEqual(tuple(agent.pool.families), families_before)
        self.assertEqual(agent.class_to_family, class_map_before)
        np.testing.assert_array_equal(
            agent.router.centroids[family],
            centroids_before,
        )
        np.testing.assert_array_equal(
            agent.buffer.samples[family],
            samples_before,
        )

    def test_standard_path_and_single_new_class_with_replay(self):
        agent = make_agent()
        X = np.array(
            [
                [-2.0, -1.0],
                [-1.5, -0.5],
                [-1.0, -1.5],
                [1.0, 1.5],
                [1.5, 0.5],
                [2.0, 1.0],
            ],
            dtype=np.float32,
        )
        y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        agent.fit_input_stats(X)
        agent.freeze_encoder()
        agent.train_task(X, y, epochs=1)

        predictions, routing = agent.predict(X, return_routing=True)
        self.assertEqual(predictions.shape, (len(X),))
        self.assertEqual(len(routing), len(X))
        self.assertTrue(np.isfinite(predictions).all())

        X_new = np.array([[2.5, 2.0], [3.0, 2.5]], dtype=np.float32)
        y_new = np.full(2, 2, dtype=np.int64)
        agent.train_task(X_new, y_new, epochs=0)
        self.assertEqual(agent.pool.families, ["family_0", "family_1", "family_2"])

    def test_binary_head_scores_are_not_cross_normalized(self):
        agent = make_agent()
        probabilities = (0.8, 0.6)
        for class_id, probability in enumerate(probabilities):
            family = register_family(
                agent,
                class_id=class_id,
                centroids=np.zeros((1, 4), dtype=np.float32),
            )
            log_odds = math.log(probability / (1.0 - probability))
            with torch.no_grad():
                head = agent.pool.heads[family]
                head.classifier.weight.zero_()
                head.classifier.bias.copy_(
                    torch.tensor([0.0, log_odds], dtype=torch.float32)
                )

        predictions, routing = agent.predict(
            np.zeros((1, 2), dtype=np.float32),
            router_weight=0.0,
            return_routing=True,
        )
        self.assertEqual(predictions.item(), 0)
        self.assertAlmostEqual(routing[0]["head_score"], 0.8, places=6)

    def test_runner_rejects_overlapping_tasks(self):
        X = np.array(
            [[-1.0, 0.0], [1.0, 0.0], [1.5, 0.5]],
            dtype=np.float32,
        )
        y = np.array([0, 1, 1], dtype=np.int64)
        with self.assertRaises(ValueError):
            run_ofra(
                X,
                y,
                X,
                y,
                tasks=[[0, 1], [1]],
                in_dim=2,
                n_classes=2,
                encoder_type="mlp",
                pretrain_epochs=0,
                epochs_per_task=0,
            )


class LossDeviceTests(unittest.TestCase):
    def test_focal_loss_cpu_forward_and_backward(self):
        logits = torch.tensor(
            [[0.2, 0.8], [0.7, -0.1]],
            dtype=torch.float32,
            requires_grad=True,
        )
        targets = torch.tensor([1, 0], dtype=torch.long)
        loss = focal_loss(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_focal_loss_and_transformer_pretraining(self):
        device = torch.device("cuda")
        logits = torch.randn(4, 2, device=device, requires_grad=True)
        targets = torch.tensor([0, 1, 0, 1], device=device)
        loss = focal_loss(logits, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

        agent = OFRAAgent(
            n_features=4,
            d_model=8,
            n_layers=1,
            n_heads=2,
            chunk_size=2,
            lora_rank=2,
            encoder_type="transformer",
        ).to(device)
        X = np.random.default_rng(7).normal(size=(8, 4)).astype(np.float32)
        agent.fit_input_stats(X)
        agent.pretrain_encoder(X, epochs=1, batch_size=4)
        self.assertEqual(next(agent.encoder.parameters()).device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
