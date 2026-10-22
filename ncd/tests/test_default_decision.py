"""Decision-rule and arithmetic tests, not experimental results."""
import unittest

import numpy as np
import torch

from ncd.default_decision import (evaluate_models, metrics_from_predictions, predict_gate_family,
                                  predict_single, summarize)


class DefaultDecisionTests(unittest.TestCase):
    def test_cpu_forward_batches_filter_future_without_training(self):
        model = torch.nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [100., 100.]]))
        original = model.weight.detach().clone()
        result = evaluate_models(np.array([[2., 0.], [0., 2.], [9., 9.]]),
                                 np.array([0, 1, 2]), [0, 1],
                                 ["Normal", "a", "future"], model, batch_size=1)
        self.assertEqual(result["accuracy"], 1.)
        self.assertEqual(result["evaluated_test_rows"], 2)
        self.assertFalse(model.training)
        self.assertIsNone(model.weight.grad)
        self.assertTrue(torch.equal(original, model.weight))

    def test_cpu_gate_forward_uses_saved_mapping(self):
        result = evaluate_models(np.array([[0., 0.], [0., 1.]]), np.array([0, 2]),
                                 [0, 2], ["Normal", "future", "attack"],
                                 torch.nn.Identity(), family=torch.nn.Identity(),
                                 attack_ids=[1, 2], batch_size=1)
        self.assertEqual(result["accuracy"], 1.)

    def test_single_masks_future_and_preserves_global_ids(self):
        out = predict_single(torch.tensor([[1., 999., 2.]]), [2, 0])
        self.assertEqual(out.tolist(), [2])

    def test_strict_gate_and_family_mapping(self):
        gate = torch.tensor([[0., 0.], [0., 1.], [1., 0.]])
        family = torch.tensor([[1., 999., 2.]]).repeat(3, 1)
        out = predict_gate_family(gate, family, [2, 3, 1], [0, 1, 2], 0)
        self.assertEqual(out.tolist(), [0, 1, 0])

    def test_weighted_metrics_and_signed_bwt(self):
        names = ["Normal", "a", "b"]
        first = metrics_from_predictions(np.array([0, 0, 1, 1]),
                                         np.array([0, 0, 0, 1]), [0, 1], names)
        last = metrics_from_predictions(np.array([0, 0, 1, 1, 2, 2]),
                                        np.array([0, 1, 1, 1, 2, 2]), [0, 1, 2], names)
        self.assertAlmostEqual(first["accuracy"], .75)
        self.assertAlmostEqual(first["macro_f1"], (4/5+2/3)/2)
        self.assertAlmostEqual(first["weighted_f1"], first["macro_f1"])
        self.assertEqual(first["attack_detection"], .5)
        result = summarize([first, last], [[0, 1], [2]])
        self.assertAlmostEqual(result["acc"], (.75+5/6)/2)
        self.assertEqual(result["bwt_per_class_recall"], 0.)
        self.assertEqual(result["forgetting_all_old_classes"], 0.)
        self.assertEqual(result["per_old_class"][1]["forgetting"], -.5)

    def test_reject_unseen_prediction_and_missing_stage(self):
        with self.assertRaises(ValueError):
            metrics_from_predictions(np.array([0, 1]), np.array([0, 2]),
                                     [0, 1], ["Normal", "a", "b"])
        with self.assertRaises(ValueError):
            summarize([], [[0, 1]])


if __name__ == "__main__":
    unittest.main()
