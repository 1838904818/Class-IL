"""Small CPU-only tests of Section 23 primitives; no research data is used."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ncd import section23_bias as bias
from ncd.crr_benchmarks import _clean_prediction_kd
from ncd.model import MLP
from ncd.run_problem_scout import ClassReservoir
from ncd.train_cil import seed_everything


def arrays(labels):
    y = np.asarray(labels, dtype=np.int64)
    x = np.arange(len(y) * 3, dtype=np.float32).reshape(-1, 3) / 20
    return x, y


def config():
    return {"batch_size": 4, "epochs_per_task": 1, "min_steps_per_task": 5,
            "learning_rate": .001, "temperature": 2., "rcl_features": 2,
            "rcl_temperature": .5, "rcl_beta": 1., "rcl_prediction_weight": .5,
            "derpp_alpha": .3, "derpp_beta": .5, "ssil_old_replay_batch_size": 2}


class Section23BiasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        seed_everything(17)

    def test_wa_scales_new_weight_and_bias_without_clamping(self):
        model = MLP(3, 4, 5)
        with torch.no_grad():
            model.head.weight[:2].fill_(2.)
            model.head.weight[2:4].fill_(-1.)
            model.head.bias.copy_(torch.tensor([1., 2., -2., 3., 4.]))
        before = copy.deepcopy(model.state_dict())
        stats = bias.apply_wa(model, [0, 1], [2, 3])
        self.assertEqual(stats["gamma"], 2.)
        self.assertTrue(stats["bias_scaled"])
        self.assertTrue(torch.equal(model.head.weight[2:4], before["head.weight"][2:4] * 2))
        self.assertTrue(torch.equal(model.head.bias[2:4], before["head.bias"][2:4] * 2))
        for name, tensor in model.state_dict().items():
            if name.startswith("head."):
                self.assertTrue(torch.equal(tensor[[0, 1, 4]], before[name][[0, 1, 4]]))
            else:
                self.assertTrue(torch.equal(tensor, before[name]))

    def test_wa_first_task_and_invalid_norm(self):
        model = MLP(3, 4, 3)
        before = copy.deepcopy(model.state_dict())
        self.assertFalse(bias.apply_wa(model, [], [0, 1])["applied"])
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()))
        with torch.no_grad():
            model.head.weight[2].zero_()
        with self.assertRaisesRegex(ValueError, "norm is zero"):
            bias.apply_wa(model, [0, 1], [2])

    def test_separated_softmax_gradient_does_not_cross_groups(self):
        logits = torch.randn(4, 5, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 3])
        loss = bias.separated_softmax_loss(logits, labels, [0, 1], [2, 3])
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[:2, 2:], torch.zeros(2, 3)))
        self.assertTrue(torch.equal(logits.grad[2:, :2], torch.zeros(2, 2)))
        self.assertTrue(torch.equal(logits.grad[:, 4], torch.zeros(4)))
        self.assertGreater(float(logits.grad[:, :4].abs().sum()), 0)

    def test_singleton_separated_ce_and_task_kd_are_zero(self):
        logits = torch.randn(4, 3, requires_grad=True)
        teacher = torch.randn(4, 3, requires_grad=True)
        loss = bias.separated_softmax_loss(logits, torch.tensor([0, 1, 0, 1]), [0], [1])
        loss = loss + bias.taskwise_kd(logits, teacher, [[0], [1]])
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))
        self.assertIsNone(teacher.grad)

    def test_task_kd_only_distills_supplied_tasks_and_detaches_teacher(self):
        logits = torch.randn(4, 5, requires_grad=True)
        teacher = torch.randn(4, 5, requires_grad=True)
        loss = bias.taskwise_kd(logits, teacher, [[0, 1], [2]])
        loss.backward()
        self.assertGreater(float(logits.grad[:, :2].abs().sum()), 0)
        self.assertTrue(torch.equal(logits.grad[:, 2:], torch.zeros(4, 3)))
        self.assertIsNone(teacher.grad)

    def test_bic_keeps_model_frozen_until_final_fold(self):
        model = MLP(3, 4, 3)
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.weight[2].fill_(.2)
            model.head.bias.copy_(torch.tensor([0., 0., 3.]))
        x, y = arrays([0, 0, 1, 1, 2, 2])
        before = copy.deepcopy(model.state_dict())
        old_logits = model(torch.from_numpy(x)).detach()
        actual_step = torch.optim.Adam.step
        calls = []
        def check_frozen(optimizer, *args, **kwargs):
            self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()))
            self.assertEqual(sum(len(g["params"]) for g in optimizer.param_groups), 2)
            calls.append(1)
            return actual_step(optimizer, *args, **kwargs)
        with patch.object(torch.optim.Adam, "step", check_frozen):
            stats = bias.fit_bic(model, x, y, [0, 1, 2], [2], lr=.05, steps=20, batch_size=6)
        self.assertEqual(len(calls), 20)
        expected = old_logits.clone()
        expected[:, 2] = old_logits[:, 2] * stats["alpha"] + stats["beta"]
        torch.testing.assert_close(model(torch.from_numpy(x)), expected)
        self.assertLess(stats["validation_loss_after"], stats["validation_loss_before"])
        self.assertEqual(stats["examples_presented"], 120)
        self.assertTrue(model.training)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_bic_rejects_unbalanced_holdout_before_mutating_model(self):
        model = MLP(3, 4, 3)
        x, y = arrays([0, 0, 1, 2])
        before = copy.deepcopy(model.state_dict())
        with self.assertRaisesRegex(ValueError, "balanced held-out"):
            bias.fit_bic(model, x, y, [0, 1, 2], [2])
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()))

    def test_cbf_equal_without_replacement_and_new_task_distillation(self):
        model = MLP(3, 4, 3)
        cx, cy = arrays([2, 2])
        ox, oy = arrays([0, 0, 0, 1, 1, 1, 1])
        captured, teachers = [], []
        schedule, kd = bias.training_epoch_batches, bias.taskwise_kd
        before = copy.deepcopy(model)
        def capture_schedule(x, y, *args, **kwargs):
            captured.append((x.copy(), y.copy()))
            yield from schedule(x, y, *args, **kwargs)
        def capture_teacher(student, teacher, tasks, temperature):
            teachers.append((teacher.requires_grad, list(tasks)))
            return kd(student, teacher, tasks, temperature)
        with patch.object(bias, "training_epoch_batches", capture_schedule), \
             patch.object(bias, "taskwise_kd", capture_teacher):
            stats = bias.balanced_finetune(model, cx, cy, ox, oy, [[0, 1], [2]],
                                          lr=.0001, epochs=2, batch_size=4, seed=7, device="cpu")
        self.assertEqual(stats["selected_per_class"], {"0": 2, "1": 2, "2": 2})
        self.assertEqual(stats["optimizer_steps"], 4)
        self.assertEqual(stats["examples_presented"], 12)
        self.assertEqual(len(captured), 1)
        for label in range(3):
            chosen = captured[0][0][captured[0][1] == label]
            self.assertEqual(len(np.unique(chosen, axis=0)), 2)
        self.assertTrue(all(not grad and tasks == [(0, 1), (2,)] for grad, tasks in teachers))
        self.assertTrue(any(not torch.equal(v, before.state_dict()[k]) for k, v in model.state_dict().items()))

    def test_cbf_single_class_has_zero_ce_and_kd(self):
        model = MLP(3, 4, 1)
        cx, cy = arrays([0, 0, 0])
        before = copy.deepcopy(model.state_dict())
        stats = bias.balanced_finetune(model, cx, cy, np.empty((0, 3), np.float32),
                    np.empty(0, np.int64), [[0]], lr=.0001, epochs=1, batch_size=2, seed=7, device="cpu")
        self.assertEqual(stats["ce_loss_mean"], 0.)
        self.assertEqual(stats["task_kd_loss_mean"], 0.)
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()))

    def test_ssil_all_bases_keep_budget_old_only_append_and_auxiliaries(self):
        for base in ["replay", "replay_lwf", "rcl_tab", "derpp"]:
            with self.subTest(base=base):
                seed_everything(17)
                model = MLP(3, 6, 4)
                teacher = copy.deepcopy(model)
                with torch.no_grad():
                    teacher.head.bias[0] += 1.
                cx, cy = arrays([2, 2, 2, 3, 3, 3])
                ox, oy = arrays([0, 0, 1, 1])
                memory = ClassReservoir(2, 3, 4) if base == "derpp" else None
                if memory is not None:
                    memory.add(torch.from_numpy(ox), torch.from_numpy(oy), torch.zeros(4, 4), np.random.default_rng(1))
                    old_observations = memory.observations[:2].copy()
                updates, append_labels = [], []
                actual_step, ce = torch.optim.Adam.step, bias.separated_softmax_loss
                def counted(optimizer, *args, **kwargs):
                    updates.append(optimizer)
                    return actual_step(optimizer, *args, **kwargs)
                def inspected_ce(logits, labels, old, new):
                    append_labels.append(labels[-2:].detach().tolist())
                    return ce(logits, labels, old, new)
                stats = {}
                with patch.object(torch.optim.Adam, "step", counted), \
                     patch.object(bias, "separated_softmax_loss", inspected_ce):
                    returned = bias.train_ssil(model, cx, cy, [ox], [oy], memory, base, teacher,
                        (0, 1), (0, 1, 2, 3), [[0, 1], [2, 3]], config(), .2, 17,
                        torch.device("cpu"), stats, 1.)
                self.assertIs(returned, stats)
                self.assertEqual(len(updates), 5)
                self.assertEqual(len({id(o) for o in updates}), 1)
                self.assertEqual(stats["optimizer_steps"], stats["target_steps"])
                self.assertEqual(stats["baseline_steps"], 2 if base == "derpp" else 3)
                self.assertEqual(stats["examples_presented"], 16 if base == "derpp" else 18)
                self.assertEqual(stats["ssil_extra_old_examples_presented"], 10)
                self.assertTrue(all(set(values).issubset({0, 1}) for values in append_labels))
                self.assertEqual(stats["distilled_tasks"], [[0, 1]])
                self.assertGreater(stats["task_kd_loss_mean"], 0)
                if base == "replay":
                    self.assertEqual(stats["base_auxiliary_loss_mean"], 0.)
                else:
                    self.assertGreater(stats["base_auxiliary_loss_mean"], 0.)
                if memory is not None:
                    self.assertEqual(memory.observations[:2], old_observations)
                    self.assertEqual(sum(memory.observations[2:]), 16)
                    self.assertGreater(stats["base_aux_mse_examples_presented"], 0)
                    self.assertGreater(stats["base_aux_ce_examples_presented"], 0)
                self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_ssil_single_new_class_ce_does_not_leak_to_old_or_future_logits(self):
        model = MLP(3, 4, 3)
        cx, cy = arrays([2, 2, 2])
        ox, oy = arrays([0, 0, 1, 1])
        teacher = copy.deepcopy(model)
        stats = bias.train_ssil(model, cx, cy, [ox], [oy], None, "replay", teacher,
            (0, 1), (0, 1, 2), [[0, 1], [2]], config(), 0., 17, "cpu", tkd_weight=.5)
        self.assertEqual(stats["optimizer_steps"], 5)
        self.assertEqual(stats["distilled_tasks"], [[0, 1]])

    def test_lwf_auxiliary_is_existing_old_class_kd(self):
        model, teacher = MLP(3, 4, 4), MLP(3, 4, 4)
        x, y = arrays([0, 1, 2, 3])
        bx, by = torch.from_numpy(x), torch.from_numpy(y)
        student, target = model(bx), teacher(bx).detach()
        actual = bias._base_auxiliary_loss(model, teacher, bx, by, student, target,
            "replay_lwf", (0, 1), (0, 1, 2, 3), config(), .3, np.random.default_rng(3))
        expected = .3 * _clean_prediction_kd(student, target, (0, 1), 2.)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rcl_auxiliary_keeps_prediction_and_reliance_terms(self):
        model, teacher = MLP(3, 4, 4), MLP(3, 4, 4)
        x, y = arrays([0, 1, 2, 3])
        bx, by = torch.from_numpy(x), torch.from_numpy(y)
        student, target = model(bx), teacher(bx).detach()
        with patch.object(bias, "_rcl_profile_loss", return_value=student.sum() * 0 + 1.) as profile, \
             patch.object(bias, "_clean_prediction_kd", return_value=student.sum() * 0 + 2.):
            actual = bias._base_auxiliary_loss(model, teacher, bx, by, student, target,
                "rcl_tab", (0, 1), (0, 1, 2, 3), config(), .3, np.random.default_rng(3))
        self.assertAlmostEqual(float(actual), 1.3, places=6)
        self.assertEqual(profile.call_count, 1)
        self.assertTrue(profile.call_args.kwargs["gated"])
        self.assertEqual(profile.call_args.args[0].shape, (2, 2))

    def test_ssil_first_task_needs_neither_teacher_nor_old_memory(self):
        for base in ["replay", "derpp"]:
            model = MLP(3, 4, 4)
            x, y = arrays([0, 0, 1, 1])
            memory = ClassReservoir(2, 3, 4) if base == "derpp" else None
            stats = bias.train_ssil(model, x, y, [], [], memory, base, None,
                (), (0, 1), [[0, 1]], config(), 0., 17, "cpu")
            self.assertEqual(stats["optimizer_steps"], 5)
            self.assertEqual(stats["ssil_extra_old_examples_presented"], 0)
            self.assertEqual(stats["distilled_tasks"], [])
            self.assertEqual(stats["task_kd_loss_mean"], 0.)
            if memory is not None:
                self.assertEqual(sum(memory.observations), stats["examples_presented"])

    def test_nonfinite_inputs_fail(self):
        model = MLP(3, 4, 3)
        with torch.no_grad():
            model.head.weight[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            bias.apply_wa(model, [0, 1], [2])
        logits = torch.tensor([[0., float("inf")]], requires_grad=True)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            bias.separated_softmax_loss(logits, torch.tensor([0]), [], [0, 1])


if __name__ == "__main__":
    unittest.main()
