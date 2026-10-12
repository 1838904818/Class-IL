from __future__ import annotations

import unittest

import numpy as np
import torch

from ncd.crr_benchmarks import (
    METHODS,
    _class_balanced_cross_entropy,
    _distribution_change,
    _log_probability_effect,
    _pair_effects,
    _rcl_profile_loss,
    boundary_for,
    build_hijack,
    build_necessity,
)
from ncd.data import PreparedData
from ncd.model import MLP


def synthetic_data() -> PreparedData:
    rng = np.random.default_rng(7)
    train_x, train_y, test_x, test_y = [], [], [], []
    for class_id in range(4):
        center = np.zeros(5, dtype=np.float32)
        center[class_id] = 3.0
        train_x.append(center + rng.normal(0, 0.3, size=(24, 5)))
        train_y.append(np.full(24, class_id))
        test_x.append(center + rng.normal(0, 0.3, size=(12, 5)))
        test_y.append(np.full(12, class_id))
    return PreparedData(
        x_train=np.concatenate(train_x).astype(np.float32),
        y_train=np.concatenate(train_y).astype(np.int64),
        x_test=np.concatenate(test_x).astype(np.float32),
        y_test=np.concatenate(test_y).astype(np.int64),
        class_order=["a", "b", "c", "d"],
        tasks=[[0], [1], [2], [3]],
        feature_names=[f"f{i}" for i in range(5)],
        source_files=[],
    )


class CRRBenchmarkTest(unittest.TestCase):
    def test_registered_methods_are_exactly_eight(self) -> None:
        self.assertEqual(len(METHODS), 8)
        self.assertEqual(len(set(METHODS)), 8)

    def test_hijack_marker_is_only_label_correlated_signal(self) -> None:
        data = synthetic_data()
        boundary = boundary_for(data)
        buffer_x = np.concatenate(
            [data.x_train[data.y_train == c][:4] for c in boundary.old_classes]
        )
        buffer_y = np.concatenate(
            [np.full(4, c, dtype=np.int64) for c in boundary.old_classes]
        )
        buffer_x = np.concatenate([buffer_x, np.zeros((len(buffer_x), 2), dtype=np.float32)], axis=1)
        benchmark = build_hijack(
            data,
            boundary,
            buffer_x,
            buffer_y,
            rho=1.0,
            marker_columns=2,
            seed=11,
        )
        replay = benchmark.train_x[-len(buffer_y) :]
        self.assertTrue(np.all(replay[:, -2:] == 0.0))
        fresh_rows = int(benchmark.metadata["fresh_target_rows"])
        fresh_target = benchmark.train_x[-len(buffer_y) - fresh_rows : -len(buffer_y)]
        self.assertTrue(np.all(fresh_target[:, -2:] > 0.0))
        self.assertTrue(benchmark.metadata["marker_only_on_fresh_target"])

    def test_balanced_cross_entropy_equalises_class_mass(self) -> None:
        logits = torch.tensor(
            [[3.0, 0.0], [2.0, 0.0], [1.0, 0.0], [0.0, 2.0]],
            requires_grad=True,
        )
        labels = torch.tensor([0, 0, 0, 1])
        loss = _class_balanced_cross_entropy(logits, labels)
        expected = 0.5 * (
            torch.nn.functional.cross_entropy(logits[:3], labels[:3])
            + torch.nn.functional.cross_entropy(logits[3:], labels[3:])
        )
        self.assertTrue(torch.allclose(loss, expected))

    def test_necessity_preserves_f1_and_shifts_f2(self) -> None:
        data = synthetic_data()
        boundary = boundary_for(data)
        marker_columns = 2
        buffer_x = np.concatenate(
            [data.x_train[data.y_train == c][:4] for c in boundary.old_classes]
        )
        buffer_y = np.concatenate(
            [np.full(4, c, dtype=np.int64) for c in boundary.old_classes]
        )
        buffer_x = np.concatenate([buffer_x, np.zeros((len(buffer_x), marker_columns), dtype=np.float32)], axis=1)
        model = MLP(5 + marker_columns, 8, 4)
        benchmark = build_necessity(
            data,
            boundary,
            buffer_x,
            buffer_y,
            model,
            k=2,
            marker_columns=marker_columns,
            seed=13,
            device=torch.device("cpu"),
            batch_size=16,
        )
        self.assertEqual(benchmark.metadata["f1_max_abs_change_train"], 0.0)
        self.assertEqual(benchmark.metadata["f1_name"], f"f{benchmark.metadata['f1']}")
        self.assertEqual(benchmark.metadata["f2_name"], f"f{benchmark.metadata['f2']}")
        self.assertAlmostEqual(benchmark.metadata["f1_histogram_overlap"], 1.0)
        self.assertGreaterEqual(benchmark.metadata["f2_histogram_overlap"], 0.0)
        self.assertLessEqual(benchmark.metadata["f2_histogram_overlap"], 1.0)
        self.assertIn(boundary.synthetic_class, set(benchmark.train_y.tolist()))
        self.assertEqual(benchmark.eval_x.shape[1], 5 + marker_columns)

    def test_old_pair_effects_exclude_new_class_rows(self) -> None:
        clean = torch.tensor([[3.0, 1.0, 0.0], [0.0, 1.0, 3.0]])
        changed = torch.tensor([[2.0, 1.0, 1.0], [0.0, 2.0, 2.0]])
        labels = torch.tensor([0, 2])
        effects, owners = _pair_effects(clean, changed, labels, (0, 1))
        self.assertEqual(effects.shape, (1, 1))
        self.assertEqual(owners.tolist(), [0])

    def test_global_ir_student_effect_uses_new_class_normalizer(self) -> None:
        clean = torch.tensor([[3.0, 1.0, 2.0]])
        changed = torch.tensor([[2.0, 1.0, 4.0]])
        labels = torch.tensor([0])
        mask = torch.tensor([True])
        full_effect = _log_probability_effect(
            clean, changed, labels, (0, 1, 2), mask
        )
        old_only_effect = _log_probability_effect(
            clean, changed, labels, (0, 1), mask
        )
        self.assertFalse(torch.allclose(full_effect, old_only_effect))

    def test_rcl_profile_runs_over_feature_axis(self) -> None:
        clean = torch.tensor([[3.0, 1.0, 0.0], [1.0, 3.0, 0.0]])
        changed = torch.tensor([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0]])
        labels = torch.tensor([0, 1])
        mask = torch.tensor([True, True])
        delta = _distribution_change(clean, changed, labels, (0, 1), mask, 1.0)
        teacher_delta = torch.stack([delta, delta * 2.0, delta * 0.5], dim=1)
        student_delta = torch.stack([delta * 0.8, delta * 1.5, delta], dim=1)
        gated = _rcl_profile_loss(
            student_delta,
            teacher_delta,
            clean,
            (0, 1),
            mask,
            reliance_temperature=0.5,
            gated=True,
        )
        ungated = _rcl_profile_loss(
            student_delta,
            teacher_delta,
            clean,
            (0, 1),
            mask,
            reliance_temperature=0.5,
            gated=False,
        )
        self.assertTrue(torch.isfinite(gated))
        self.assertTrue(torch.isfinite(ungated))
        self.assertGreaterEqual(float(gated), 0.0)
        self.assertGreaterEqual(float(ungated), 0.0)


if __name__ == "__main__":
    unittest.main()
