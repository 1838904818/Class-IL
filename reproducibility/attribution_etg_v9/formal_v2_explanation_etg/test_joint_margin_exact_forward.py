import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from formal_v2_explanation_etg.analyze import (
    JointMarginModel,
    JointScoreModel,
    audit_attribution_gradients,
    audit_checkpoint_reconstruction,
    audit_exact_forward_and_batch_partition,
    prediction_difference_audit,
)


class FakeHead(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer("weight", torch.tensor(weight, dtype=torch.float32))

    def positive_probability(self, embedding):
        return torch.sigmoid(embedding @ self.weight)


class JointMarginExactForwardTests(unittest.TestCase):
    def checkpoint(self):
        encoder = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            encoder.weight.copy_(torch.tensor([[0.3, -0.2, 0.1], [0.5, 0.4, -0.6]]))
        return types.SimpleNamespace(
            encoder=encoder,
            heads={0: FakeHead([0.7, -0.1]), 1: FakeHead([-0.3, 0.8])},
            metadata={"seen_classes": [0, 1]},
            mean=np.array([1.000000071, -2.000000093, 0.500000029], dtype=np.float64),
            scale=np.array([0.125000019, 3.000000117, 0.750000041], dtype=np.float64),
            router=types.SimpleNamespace(
                cap={
                    0: types.SimpleNamespace(
                        centroids=np.array([[0.1, 0.2], [0.10001, 0.20001]], dtype=np.float32)
                    ),
                    1: types.SimpleNamespace(
                        centroids=np.array([[0.10002, 0.20002]], dtype=np.float32)
                    ),
                }
            ),
        )

    def zero_distance_checkpoint(self):
        checkpoint = self.checkpoint()
        with torch.no_grad():
            checkpoint.encoder.weight.copy_(
                torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            )
        checkpoint.mean = np.zeros(3, dtype=np.float64)
        checkpoint.scale = np.ones(3, dtype=np.float64)
        checkpoint.router.cap[0].centroids = np.array([[0.1, 0.2]], dtype=np.float32)
        checkpoint.router.cap[1].centroids = np.array([[-0.4, 0.7]], dtype=np.float32)
        return checkpoint

    @staticmethod
    def exact_joint(checkpoint, raw):
        normalized = ((raw.astype(np.float64) - checkpoint.mean) / checkpoint.scale).astype(np.float32)
        with torch.no_grad():
            embedding = checkpoint.encoder(torch.from_numpy(normalized)).numpy().astype(np.float32)
            heads = np.column_stack(
                [checkpoint.heads[c].positive_probability(torch.from_numpy(embedding)).numpy() for c in [0, 1]]
            ).astype(np.float32)
        router_raw = np.empty((len(raw), 2), dtype=np.float32)
        for column, class_id in enumerate([0, 1]):
            centroids = checkpoint.router.cap[class_id].centroids
            squared = (
                np.einsum("ij,ij->i", embedding, embedding)[:, None]
                + np.einsum("ij,ij->i", centroids, centroids)[None, :]
                - 2.0 * embedding @ centroids.T
            )
            np.maximum(squared, 0.0, out=squared)
            router_raw[:, column] = -np.sqrt(squared.min(axis=1), dtype=np.float32)
        router = (router_raw - router_raw.mean(axis=1, keepdims=True)) / (
            router_raw.std(axis=1, keepdims=True) + 1e-8
        )
        return heads + np.float32(0.5) * router

    @classmethod
    def exact_margin(cls, checkpoint, raw, target):
        joint = cls.exact_joint(checkpoint, raw)
        other = 1 - target
        return joint[:, target] - joint[:, other]

    def test_forward_is_exact_and_gradient_is_finite(self):
        checkpoint = self.checkpoint()
        raw_np = np.array(
            [[1.2, -1.1, 0.8], [0.9, -3.4, -0.2], [1.7, 0.4, 1.1]], dtype=np.float32
        )
        for target in (0, 1):
            raw = torch.tensor(raw_np, requires_grad=True)
            model = JointMarginModel(checkpoint, target)
            observed = model(raw).detach().numpy().ravel()
            expected = self.exact_margin(checkpoint, raw_np, target)
            self.assertTrue(np.array_equal(observed, expected))
            model(raw).sum().backward()
            self.assertTrue(torch.isfinite(raw.grad).all())
            self.assertGreater(float(raw.grad.abs().sum()), 0.0)

    def test_exact_centroid_keeps_exact_forward_and_finite_gradient(self):
        checkpoint = self.zero_distance_checkpoint()
        raw_np = np.array([[0.1, 0.2, 0.0]], dtype=np.float32)
        raw = torch.tensor(raw_np, requires_grad=True)
        model = JointMarginModel(checkpoint, 0)
        observed = model(raw)
        expected = self.exact_margin(checkpoint, raw_np, 0)
        self.assertTrue(np.array_equal(observed.detach().numpy().ravel(), expected))
        observed.sum().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertGreater(float(raw.grad.abs().sum()), 0.0)

    def test_pre_attribution_gradient_audit_covers_true_class_batches(self):
        checkpoint = self.zero_distance_checkpoint()
        raw_np = np.array(
            [[0.1, 0.2, 0.0], [-0.4, 0.7, 0.0]], dtype=np.float32
        )
        rows = audit_attribution_gradients(
            0,
            checkpoint,
            raw_np,
            [{"class_id": 0}, {"class_id": 1}],
        )
        self.assertEqual([(row["checkpoint"], row["class_id"]) for row in rows], [(0, 0), (0, 1)])
        self.assertTrue(all(row["gradient_nonfinite_count"] == 0 for row in rows))
        self.assertTrue(all(row["exact_forward_value_changed"] is False for row in rows))

    def test_pre_attribution_audit_separates_same_batch_fidelity(self):
        checkpoint = self.checkpoint()
        raw_np = np.array(
            [[1.2, -1.1, 0.8], [0.9, -3.4, -0.2], [1.7, 0.4, 1.1], [1.1, -1.0, 0.7]],
            dtype=np.float32,
        )

        def score(values):
            joint = self.exact_joint(checkpoint, values)
            axis = np.array([0, 1], dtype=np.int64)
            return {
                "class_axis": axis,
                "joint_scores": joint,
                "predicted_class_id": axis[joint.argmax(axis=1)],
            }

        checkpoint.score = score
        records = [{"class_id": value} for value in (0, 0, 1, 1)]
        reconstructed = checkpoint.score(raw_np)
        fidelity, partition = audit_exact_forward_and_batch_partition(
            0, checkpoint, raw_np, records, reconstructed
        )
        self.assertEqual(len(fidelity), 2)
        self.assertEqual(len(partition), 2)
        self.assertEqual(max(row["max_abs_error"] for row in fidelity), 0.0)
        self.assertEqual(
            max(row["class_batch_joint_score_max_abs_error"] for row in partition), 0.0
        )
        observed = JointScoreModel(checkpoint)(torch.from_numpy(raw_np)).detach().numpy()
        self.assertTrue(np.array_equal(observed, reconstructed["joint_scores"]))
        with self.assertRaisesRegex(RuntimeError, "misaligned"):
            audit_exact_forward_and_batch_partition(
                0, checkpoint, raw_np, records[:-1], reconstructed
            )

    def test_pre_attribution_audit_reports_partition_reference_exceedance(self):
        checkpoint = self.checkpoint()
        raw_np = np.array(
            [[1.2, -1.1, 0.8], [0.9, -3.4, -0.2], [1.7, 0.4, 1.1], [1.1, -1.0, 0.7]],
            dtype=np.float32,
        )

        def score(values):
            joint = self.exact_joint(checkpoint, values)
            if len(values) == 2:
                joint = joint + np.float32(0.0011)
            axis = np.array([0, 1], dtype=np.int64)
            return {
                "class_axis": axis,
                "joint_scores": joint,
                "predicted_class_id": axis[joint.argmax(axis=1)],
            }

        class SameBatchProxy(torch.nn.Module):
            def __init__(self, current_checkpoint):
                super().__init__()
                self.checkpoint = current_checkpoint

            def forward(self, raw):
                values = raw.detach().cpu().numpy()
                return torch.from_numpy(self.checkpoint.score(values)["joint_scores"])

        checkpoint.score = score
        records = [{"class_id": value} for value in (0, 0, 1, 1)]
        reconstructed = checkpoint.score(raw_np)
        with patch("formal_v2_explanation_etg.analyze.JointScoreModel", SameBatchProxy):
            _, partition = audit_exact_forward_and_batch_partition(
                0, checkpoint, raw_np, records, reconstructed
            )
        self.assertFalse(partition[0]["joint_score_within_reference_tolerance"])
        self.assertGreater(partition[0]["joint_score_reference_exceedance_count"], 0)
        self.assertEqual(partition[0]["mismatch_count"], 0)
        self.assertFalse(partition[0]["equivalence_claimed"])

    def test_large_cross_device_score_swap_is_reported_without_equivalence_claim(self):
        axis = np.array([0, 1], dtype=np.int64)
        saved_joint = np.array([[1.0, 0.0]], dtype=np.float32)
        reconstructed_joint = np.array([[0.0, 1.0]], dtype=np.float32)
        saved = {
            "class_axis": axis,
            "head_scores": saved_joint.copy(),
            "router_z_scores": saved_joint.copy(),
            "joint_scores": saved_joint,
            "predicted_class_id": np.array([0], dtype=np.int64),
        }
        reconstructed = {
            "class_axis": axis,
            "head_scores": reconstructed_joint.copy(),
            "router_z_scores": reconstructed_joint.copy(),
            "joint_scores": reconstructed_joint,
            "predicted_class_id": np.array([1], dtype=np.int64),
        }
        rows = audit_checkpoint_reconstruction(0, saved, reconstructed)
        joint = next(row for row in rows if row["array"] == "joint_scores")
        prediction = next(row for row in rows if row["array"] == "predicted_class_id")
        self.assertFalse(joint["within_reference_tolerance"])
        self.assertEqual(joint["reference_exceedance_count"], 2)
        self.assertEqual(prediction["mismatch_count"], 1)
        self.assertFalse(prediction["equivalence_claimed"])
        self.assertEqual(prediction["maximum_saved_winner_gap"], 1.0)
        self.assertEqual(prediction["maximum_reconstructed_winner_gap"], 1.0)

    def test_cross_device_audit_rejects_prediction_not_matching_own_argmax(self):
        axis = np.array([0, 1], dtype=np.int64)
        joint = np.array([[1.0, 0.5]], dtype=np.float32)
        saved = {
            "class_axis": axis,
            "joint_scores": joint,
            "predicted_class_id": np.array([1], dtype=np.int64),
        }
        reconstructed = {
            "class_axis": axis,
            "joint_scores": joint.copy(),
            "predicted_class_id": np.array([0], dtype=np.int64),
        }
        with self.assertRaisesRegex(RuntimeError, "saved prediction is inconsistent"):
            prediction_difference_audit(saved, reconstructed)

    def test_same_batch_audit_rejects_non_finite_scores(self):
        checkpoint = self.checkpoint()
        raw_np = np.array(
            [[1.2, -1.1, 0.8], [0.9, -3.4, -0.2]], dtype=np.float32
        )
        axis = np.array([0, 1], dtype=np.int64)
        reconstructed = {
            "class_axis": axis,
            "joint_scores": np.array([[np.nan, 0.0], [0.0, 1.0]], dtype=np.float32),
            "predicted_class_id": np.array([0, 1], dtype=np.int64),
        }
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            audit_exact_forward_and_batch_partition(
                0,
                checkpoint,
                raw_np,
                [{"class_id": 0}, {"class_id": 1}],
                reconstructed,
            )


if __name__ == "__main__":
    unittest.main()
