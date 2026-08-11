"""Focused checks for checkpoint-level OFRA prediction diagnostics."""

from __future__ import annotations

import json
import math
import unittest

import numpy as np
import torch

from src_v2.methods.ofra import (
    DEFAULT_PREDICTION_ARMS,
    OFRAAgent,
    compute_classification_diagnostics,
    run_ofra,
)
from src_v2.multi_seed_ofra import parse_prediction_arms


class ClassificationDiagnosticTests(unittest.TestCase):
    def test_confusion_matrix_and_metrics_use_explicit_label_order(self):
        y_true = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        y_pred = np.array([0, 1, 1, 1, 0, 2], dtype=np.int64)

        result = compute_classification_diagnostics(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            class_names={0: "zero", 1: "one", 2: "two"},
        )

        self.assertEqual(result["confusion_matrix"]["labels"], [0, 1, 2])
        self.assertEqual(
            result["confusion_matrix"]["class_names"],
            ["zero", "one", "two"],
        )
        self.assertEqual(
            result["confusion_matrix"]["values"],
            [[1, 1, 0], [0, 2, 0], [1, 0, 1]],
        )
        self.assertAlmostEqual(result["accuracy"], 4 / 6)
        self.assertAlmostEqual(result["balanced_accuracy"], 2 / 3)
        self.assertAlmostEqual(result["macro_f1"], (0.5 + 0.8 + 2 / 3) / 3)
        self.assertEqual(
            [entry["support"] for entry in result["per_class"]],
            [2, 2, 2],
        )
        json.dumps(result, allow_nan=False)


class PredictionArmTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(11)
        torch.manual_seed(11)

    def test_shared_arm_predictions_match_individual_predict_calls(self):
        agent = OFRAAgent(
            n_features=2,
            d_model=4,
            n_layers=1,
            lora_rank=2,
            exemplar_capacity=2,
            encoder_type="mlp",
        )
        probabilities = (0.8, 0.6)
        centroids = (
            np.zeros((1, 4), dtype=np.float32),
            np.ones((1, 4), dtype=np.float32),
        )
        for class_id, (probability, family_centroids) in enumerate(
            zip(probabilities, centroids)
        ):
            family = f"family_{class_id}"
            agent.pool.add_family(family, n_local_classes=2)
            agent.class_to_family[class_id] = family
            agent.family_to_class[family] = class_id
            agent.router.centroids[family] = family_centroids
            log_odds = math.log(probability / (1.0 - probability))
            with torch.no_grad():
                head = agent.pool.heads[family]
                head.classifier.weight.zero_()
                head.classifier.bias.copy_(
                    torch.tensor([0.0, log_odds], dtype=torch.float32)
                )

        X = np.array([[0.0, 0.0], [1.0, -1.0]], dtype=np.float32)
        arm_predictions = agent.predict_arms(X, DEFAULT_PREDICTION_ARMS)

        for name, config in DEFAULT_PREDICTION_ARMS.items():
            expected = agent.predict(X, **config)
            np.testing.assert_array_equal(arm_predictions[name], expected)

    def test_cli_arm_parser_records_selection_and_rejects_duplicates(self):
        parsed = parse_prediction_arms("p-only,z-only,joint")
        self.assertEqual(list(parsed), ["p-only", "z-only", "joint"])
        self.assertEqual(parsed["joint"], DEFAULT_PREDICTION_ARMS["joint"])
        with self.assertRaises(ValueError):
            parse_prediction_arms("joint,joint")
        with self.assertRaises(ValueError):
            parse_prediction_arms("unknown")


class RunnerDiagnosticTests(unittest.TestCase):
    def test_each_checkpoint_contains_all_arms_and_joint_stays_compatible(self):
        X_train = np.array(
            [
                [-2.0, -2.0],
                [-1.8, -2.2],
                [-2.2, -1.8],
                [2.0, 2.0],
                [1.8, 2.2],
                [2.2, 1.8],
                [0.0, 3.0],
                [0.2, 3.2],
                [-0.2, 2.8],
            ],
            dtype=np.float32,
        )
        y_train = np.repeat(np.arange(3, dtype=np.int64), 3)
        X_test = np.array(
            [
                [-2.1, -2.0],
                [-1.9, -2.1],
                [2.1, 2.0],
                [1.9, 2.1],
                [0.0, 3.1],
                [0.1, 2.9],
            ],
            dtype=np.float32,
        )
        y_test = np.repeat(np.arange(3, dtype=np.int64), 2)
        tasks = [[0, 1], [2]]

        run_result = run_ofra(
            X_train,
            y_train,
            X_test,
            y_test,
            tasks=tasks,
            in_dim=2,
            n_classes=3,
            d_model=4,
            n_layers=1,
            lora_rank=2,
            pretrain_epochs=0,
            epochs_per_task=0,
            exemplar_capacity=2,
            encoder_type="mlp",
            prediction_arms=DEFAULT_PREDICTION_ARMS,
            class_names=["zero", "one", "two"],
        )

        self.assertEqual(len(run_result), 2)
        acc_matrix, agent = run_result
        self.assertEqual(acc_matrix.shape, (2, 2))
        diagnostics = agent.checkpoint_diagnostics
        self.assertIsNotNone(diagnostics)
        self.assertEqual(len(diagnostics["checkpoints"]), 2)

        for checkpoint_index, checkpoint in enumerate(
            diagnostics["checkpoints"]
        ):
            self.assertEqual(checkpoint["checkpoint"], checkpoint_index)
            self.assertEqual(checkpoint["task"]["index"], checkpoint_index)
            self.assertEqual(
                list(checkpoint["arms"]),
                ["p-only", "z-only", "joint"],
            )
            labels = checkpoint["seen_labels"]
            for arm_result in checkpoint["arms"].values():
                matrix_record = arm_result["confusion_matrix"]
                self.assertEqual(matrix_record["labels"], labels)
                matrix = np.asarray(matrix_record["values"], dtype=np.int64)
                self.assertEqual(matrix.shape, (len(labels), len(labels)))
                self.assertEqual(matrix.sum(), checkpoint["n_test_samples"])
                self.assertTrue(np.isfinite(arm_result["macro_f1"]))
                self.assertTrue(np.isfinite(arm_result["balanced_accuracy"]))

            joint_matrix = np.asarray(
                checkpoint["arms"]["joint"]["confusion_matrix"]["values"],
                dtype=np.int64,
            )
            label_to_index = {
                label: index for index, label in enumerate(labels)
            }
            for task_index, task in enumerate(tasks[: checkpoint_index + 1]):
                rows = [label_to_index[label] for label in task]
                correct = sum(joint_matrix[row, row] for row in rows)
                support = int(joint_matrix[rows, :].sum())
                self.assertAlmostEqual(
                    correct / support,
                    acc_matrix[checkpoint_index, task_index],
                )

        json.dumps(diagnostics, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
