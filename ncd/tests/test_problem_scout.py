"""Small CPU-only unit checks; no historical evidence is changed."""
import copy
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from ncd.data import PreparedData
from ncd import run_problem_scout as scout
from ncd import crr_benchmarks as benchmarks
from ncd.train_cil import extract_features, herding_select


def fixture():
    y = np.repeat(np.arange(4), 5).astype(np.int64)
    x = np.arange(60, dtype=np.float32).reshape(20, 3) / 60
    return PreparedData(x, y, x.copy(), y.copy(), ["Normal", "A", "B", "C"], [[0, 1], [2], [3]], ["x", "y", "z"], [])


def config():
    return {"hidden": 8, "batch_size": 4, "eval_batch_size": 8, "epochs_per_task": 1,
            "learning_rate": .001, "buffer_per_class": 2, "temperature": 2.,
            "rcl_features": 2, "rcl_temperature": .5, "rcl_beta": 1.,
            "rcl_prediction_weight": .5, "derpp_alpha": .3, "derpp_beta": .5}


class ProblemScoutTests(unittest.TestCase):
    def test_minimum_steps_stop_mid_extra_epoch_and_keep_full_baseline(self):
        data = fixture()
        x, y = data.x_train[:10], data.y_train[:10]
        for epochs, floor, expected in [(1, 5, (5, 1, 2, 18)), (3, 5, (9, 3, 0, 30)),
                                        (1, 6, (6, 2, 0, 20)), (1, 0, (3, 1, 0, 10))]:
            stats, batches = {}, []
            for epoch, epoch_batches in scout.training_epoch_batches(x, y, 4, 17, epochs, floor, stats):
                batches.extend((epoch, bx.clone(), by.clone()) for bx, by in epoch_batches)
            self.assertEqual((stats["optimizer_steps"], stats["completed_epochs"],
                              stats["partial_epoch_batches"], stats["examples_presented"]), expected)
            self.assertEqual(stats["effective_epochs"], expected[-1] / len(y))
            self.assertEqual(len(batches), max(epochs * 3, floor))
            self.assertEqual(stats["baseline_steps"], epochs * 3)
            historical = [(epoch, bx, by) for epoch in range(epochs)
                          for bx, by in scout.loader(x, y, 4, 17 + epoch)]
            for observed, original in zip(batches, historical):
                self.assertEqual(observed[0], original[0])
                self.assertTrue(torch.equal(observed[1], original[1]))
                self.assertTrue(torch.equal(observed[2], original[2]))

    def test_all_cil_methods_count_optimizer_steps_including_rcl_branch(self):
        data, cfg = fixture(), config()
        x, y = data.x_train[:10], data.y_train[:10]
        for method in ["replay", "icarl", "replay_lwf", "rcl_tab", "derpp"]:
            scout.seed_everything(17)
            model = scout.MLP(3, 8, 4)
            teacher = copy.deepcopy(model)
            stats, actual_updates = {}, []
            original = torch.optim.Adam.step
            def counted_step(optimizer, *args, **kwargs):
                actual_updates.append(optimizer)
                return original(optimizer, *args, **kwargs)
            with patch.object(torch.optim.Adam, "step", counted_step):
                if method == "derpp":
                    memory = scout.ClassReservoir(2, 3, 4)
                    scout.train_derpp(model, x, y, memory, cfg | {"min_steps_per_task": 5},
                                      17, torch.device("cpu"), stats)
                    self.assertEqual(sum(memory.observations), 18)
                else:
                    benchmarks._train(model, x, y, method=method, old_model=teacher,
                                      old_classes=(0, 1), seen_classes=(0, 1, 2, 3), epochs=1,
                                      batch_size=4, learning_rate=.001, seed=17, regularization=.1,
                                      temperature=2., rcl_features=2, min_steps=5, training_stats=stats)
            self.assertEqual(len(actual_updates), 5, method)
            self.assertEqual(len({id(optimizer) for optimizer in actual_updates}), 1, method)
            self.assertEqual(stats["optimizer_steps"], 5, method)
            self.assertEqual(stats["completed_epochs"], 1, method)
            self.assertEqual(stats["partial_epoch_batches"], 2, method)

    def test_zero_floor_matches_historical_epoch_loader_models_and_derpp_memory(self):
        data, cfg = fixture(), config() | {"epochs_per_task": 2}
        def historical_batches(x, y, batch_size, seed, epochs, min_steps=0, training_stats=None):
            # Independent schedule used before Section22: exactly one seeded loader per epoch.
            for epoch in range(epochs):
                yield epoch, benchmarks.loader(x, y, batch_size, seed + epoch)
        for method in ["replay", "icarl", "replay_lwf", "rcl_tab", "derpp"]:
            outputs = []
            for historical in [True, False]:
                scout.seed_everything(41)
                model = scout.MLP(3, 8, 4)
                teacher = copy.deepcopy(model)
                if method == "derpp":
                    memory = scout.ClassReservoir(2, 3, 4)
                    # DER++ only needs these fields for progress printing in the historical adapter.
                    def historical_derpp_batches(*args, **kwargs):
                        stats = args[6]
                        stats["partial_epoch_batches"] = 0
                        yield from historical_batches(*args, **kwargs)
                    helper = historical_derpp_batches if historical else benchmarks.training_epoch_batches
                    with patch.object(scout, "training_epoch_batches", helper):
                        scout.train_derpp(model, data.x_train, data.y_train, memory, cfg, 41, torch.device("cpu"))
                    outputs.append((copy.deepcopy(model.state_dict()), memory.state()))
                else:
                    helper = historical_batches if historical else benchmarks.training_epoch_batches
                    with patch.object(benchmarks, "training_epoch_batches", helper):
                        benchmarks._train(model, data.x_train, data.y_train, method=method, old_model=teacher,
                                          old_classes=(0, 1), seen_classes=(0, 1, 2, 3), epochs=2,
                                          batch_size=4, learning_rate=.001, seed=41, regularization=.1,
                                          temperature=2., rcl_features=2)
                    outputs.append((copy.deepcopy(model.state_dict()), None))
            for name in outputs[0][0]:
                self.assertTrue(torch.equal(outputs[0][0][name], outputs[1][0][name]), (method, name))
            if method == "derpp":
                self.assertTrue(torch.equal(outputs[0][1]["x"], outputs[1][1]["x"]))
                self.assertTrue(torch.equal(outputs[0][1]["logits"], outputs[1][1]["logits"]))
                self.assertEqual(outputs[0][1]["observations"], outputs[1][1]["observations"])

    def test_cell_records_step_budget_and_resume_without_retraining(self):
        data, cfg = fixture(), config() | {"min_steps_per_task": 5}
        identity = {"dataset": "NSL-KDD", "method": "replay", "seed": 1, "regularization": 0.0}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "dataset" / "cell"
            result = scout.run_cell(data, None, None, {}, cfg, identity, output, torch.device("cpu"))
            for record in result["checkpoints"]:
                schedule = record["training_schedule"]
                self.assertEqual(schedule["optimizer_steps"], 5)
                self.assertEqual(schedule["rows_per_epoch"], record["train_rows"])
                self.assertEqual(schedule["baseline_steps"], 3)
                self.assertEqual(schedule["completed_epochs"], 1)
                self.assertEqual(schedule["partial_epoch_batches"], 2)
            with patch.object(scout, "_train", side_effect=AssertionError("unexpected retraining")):
                self.assertEqual(result, scout.run_cell(data, None, None, {}, cfg, identity, output, torch.device("cpu")))

    def test_indexed_preallocated_and_mmap_herding_exactly_match_history(self):
        torch.manual_seed(271)
        model = scout.MLP(7, 16, 5)
        x = np.random.default_rng(57).normal(size=(1027, 7)).astype(np.float32)
        rows = np.flatnonzero(np.arange(len(x)) % 3 != 1)
        expected_features = extract_features(model, x[rows], torch.device("cpu"), 37)
        expected = herding_select(expected_features, 200)
        with tempfile.TemporaryDirectory() as temp:
            scratch = Path(temp)
            for limit in (0, 10**9):
                seen_batches = []
                hook = model.features.register_forward_pre_hook(lambda _m, inp: seen_batches.append(len(inp[0])))
                try:
                    with patch.object(scout, "herding_select", wraps=herding_select) as select:
                        actual = scout.herding_select_indexed(model, x, rows, 200, torch.device("cpu"), 37,
                                                             mmap_threshold_bytes=limit, scratch_dir=scratch)
                        # Copy before the temporary mmap is closed is not needed:
                        # exact output indices test the full historical reducer.
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(select.call_count, 1)
                    self.assertLessEqual(max(seen_batches), 37)
                    self.assertEqual(list(scratch.iterdir()), [])
                finally:
                    hook.remove()

    def test_indexed_herding_cleans_mmap_when_feature_extraction_fails(self):
        model = scout.MLP(3, 8, 2)
        x = np.ones((9, 3), dtype=np.float32)
        original, calls = model.features.forward, []
        def failing(batch):
            calls.append(len(batch))
            if len(calls) == 2:
                raise RuntimeError("injected feature failure")
            return original(batch)
        with tempfile.TemporaryDirectory() as temp, patch.object(model.features, "forward", side_effect=failing):
            with self.assertRaisesRegex(RuntimeError, "injected feature failure"):
                scout.herding_select_indexed(model, x, np.arange(9), 2, torch.device("cpu"), 4,
                                            mmap_threshold_bytes=0, scratch_dir=Path(temp))
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_operational_metrics_and_mutually_exclusive_flow(self):
        cm = np.array([[7, 1, 2, 0], [2, 4, 3, 1], [1, 2, 6, 1], [2, 1, 0, 7]])
        m = scout.metrics_from_confusion(cm, (0, 1, 2, 3), ["Normal", "A", "B", "C"], [[0, 1], [2], [3]], [20]*4)
        self.assertAlmostEqual(m["attack_recall"], 25/30)
        self.assertAlmostEqual(m["normal_fpr"], .3)
        self.assertAlmostEqual(m["attack_family_macro_recall"], (0.4+0.6+0.7)/3)
        a = m["per_class"][1]
        self.assertEqual((a["error_to_normal"], a["error_to_newer_class"], a["error_to_other_old"]), (2, 4, 0))
        for row in m["per_class"]:
            self.assertEqual(row["error_count"], sum(row[k] for k in ["error_to_normal", "error_to_newer_class", "error_to_other_old"]))

    def test_absent_class_is_not_fake_recall(self):
        m = scout.metrics_from_confusion(np.array([[2, 0], [0, 0]]), (0, 1), ["Normal", "A"], [[0, 1]], [2, 2])
        self.assertIsNone(m["per_class"][1]["recall"])
        self.assertIsNone(m["attack_recall"])

    def test_replay_memory_never_reacquires_old_pool(self):
        x = np.array([[99., 99., 99.], [100., 100., 100.]], dtype=np.float32)
        old_x = [np.array([[3., 3., 3.]], dtype=np.float32)]
        old_y = [np.array([0], dtype=np.int64)]
        bx, by = scout.update_memory(None, x, np.array([2, 2]), old_x, old_y, "replay", config(), 1, torch.device("cpu"))
        np.testing.assert_array_equal(bx[0], old_x[0])
        np.testing.assert_array_equal(by[0], old_y[0])
        with self.assertRaises(ValueError):
            scout.update_memory(None, x, np.array([0, 0]), old_x, old_y, "replay", config(), 1, torch.device("cpu"))

    def test_reservoir_budget_state_and_preupdate_logits(self):
        memory = scout.ClassReservoir(2, 3, 4)
        x = torch.arange(90, dtype=torch.float32).reshape(30, 3)
        y = torch.tensor([0]*20+[1]*10)
        logits = torch.arange(120, dtype=torch.float32).reshape(30, 4)
        memory.add(x, y, logits, np.random.default_rng(1))
        self.assertEqual(len(memory), 4)
        self.assertEqual(memory.observations, [20, 10, 0, 0])
        restored = scout.ClassReservoir(2, 3, 4, memory.state())
        self.assertTrue(torch.equal(restored.logits, memory.logits))
        bx, by, bz = restored.sample(100, np.random.default_rng(2), torch.device("cpu"))
        for row in range(len(bx)):
            source = int(bx[row, 0] / 3)
            self.assertTrue(torch.equal(bz[row], logits[source]))
        self.assertLessEqual(memory.payload_bytes(), memory.allocated_bytes())

    def test_joint_full_seen_fresh_and_resume(self):
        data, cfg = fixture(), config()
        identity = {"dataset": "NSL-KDD", "method": "joint", "seed": 1, "regularization": 0.0}
        calls = []
        def train(model, x, y, **kwargs):
            calls.append((len(y), set(y.tolist()), next(model.parameters()).detach().clone()))
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(100)
            return 0.0
        with tempfile.TemporaryDirectory() as temp, patch.object(scout, "_train", side_effect=train):
            output = Path(temp)/"dataset"/"cell"
            result = scout.run_cell(data, None, None, {}, cfg, identity, output, torch.device("cpu"))
            self.assertEqual([a[0] for a in calls], [10, 15, 20])
            self.assertEqual([a[1] for a in calls], [{0, 1}, {0, 1, 2}, {0, 1, 2, 3}])
            self.assertTrue(all(float(a[2].abs().max()) < 1 for a in calls))
            self.assertTrue(all(c["replay_examples_after"] == 0 for c in result["checkpoints"]))
            calls.clear()
            same = scout.run_cell(data, None, None, {}, cfg, identity, output, torch.device("cpu"))
            self.assertEqual(calls, [])
            self.assertEqual(result, same)
            wrong = identity | {"seed": 2}
            with self.assertRaises(ValueError):
                scout.run_cell(data, None, None, {}, cfg, wrong, output, torch.device("cpu"))

    def test_replay_train_rows_are_current_plus_stored_only(self):
        data, cfg = fixture(), config()
        identity = {"dataset": "NSL-KDD", "method": "replay", "seed": 1, "regularization": 0.0}
        counts = []
        def train(model, x, y, **kwargs):
            counts.append(dict(zip(*np.unique(y, return_counts=True))))
            return 0.0
        with tempfile.TemporaryDirectory() as temp, patch.object(scout, "_train", side_effect=train):
            output = Path(temp)/"dataset"/"cell"
            result = scout.run_cell(data, None, None, {}, cfg, identity, output, torch.device("cpu"))
            self.assertEqual(counts, [{0:5,1:5}, {0:2,1:2,2:5}, {0:2,1:2,2:2,3:5}])
            self.assertEqual([c["replay_examples_after"] for c in result["checkpoints"]], [4,6,8])

    def test_partial_resume_does_not_retrain_saved_checkpoint(self):
        data, cfg = fixture(), config()
        identity = {"dataset": "NSL-KDD", "method": "replay", "seed": 1, "regularization": 0.0}
        with tempfile.TemporaryDirectory() as temp, patch.object(scout, "_train", return_value=0.0) as train:
            output = Path(temp)/"dataset"/"cell"
            original = scout.evaluate
            with patch.object(scout, "evaluate", side_effect=RuntimeError("evaluation interrupted")):
                with self.assertRaisesRegex(RuntimeError, "evaluation interrupted"):
                    scout.run_cell(data, None, None, {}, cfg, identity, output, torch.device("cpu"))
            saved = scout.checkpoint_load(output/"checkpoints/checkpoint_000.pt", identity)
            self.assertEqual(train.call_count, 1)
            train.reset_mock()
            result = scout.run_cell(data, None, None, {}, cfg, identity, output, torch.device("cpu"))
            self.assertEqual(train.call_count, 2)
            self.assertEqual(result["checkpoints"][0]["training_seconds"], saved["training_seconds"])
            self.assertEqual(result["training_seconds_total"], sum(c["training_seconds"] for c in result["checkpoints"]))

    def test_finite_and_smoke_bounds(self):
        with self.assertRaises(ValueError):
            scout.require_finite({"metric": float("nan")})
        data = fixture()
        cut, sx, sy = scout.smoke_subset(data, data.x_test, data.y_test, limit=2)
        self.assertEqual(len(cut.y_train), 8)
        self.assertEqual(len(sy), 8)

    def test_same_entry_smoke_reaches_final_summary_and_resumes(self):
        cfg = config() | {"datasets": ["NSL-KDD"], "methods": list(scout.METHODS), "seeds": list(scout.SEEDS),
                          "device": "cpu", "lambda_files": {"default": "selection.json"}}
        selection = {"selection_seeds": [5, 6], "git_commit": "b" * 40,
                     "lambdas": {method: 0.0 for method in scout.METHODS}}
        module = types.ModuleType("ncd.prepare_problem_scout_data")
        module.prepare_setting = lambda dataset, config: (fixture(), None, None, {"dataset": dataset})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
            (root / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
            args = ["run_problem_scout.py", "--config", str(root / "config.json"), "--output-dir", str(root / "results"),
                    "--smoke", "--min-steps-per-task", "500"]
            with patch.object(scout, "ROOT", root), patch.object(scout.sys, "argv", args), \
                 patch.dict(scout.sys.modules, {"ncd.prepare_problem_scout_data": module}), \
                 patch.object(scout, "digest", return_value="c" * 64), \
                 patch.object(scout, "source_commit", return_value="a" * 40), \
                 patch.object(scout, "_train", return_value=0.0) as train, \
                 patch.object(scout, "train_derpp", return_value=None) as derpp:
                scout.main()
                result = json.loads((root / "results/result.json").read_text())
                self.assertEqual(result["execution_status"], "complete")
                self.assertTrue(result["smoke"])
                self.assertEqual(len(result["cells"]), 6)
                self.assertEqual(result["expected"]["seeds"], [1])
                self.assertEqual(result["provenance"]["effective_config"]["min_steps_per_task"], 0)
                self.assertEqual(result["provenance"]["effective_config"]["smoke_requested_min_steps_per_task"], 500)
                self.assertEqual(result["provenance"]["effective_config"]["epochs_per_task"], 1)
                self.assertEqual(len(result["final_rows"]), 24)
                self.assertTrue((root / "results/final_rows.csv").is_file())
                train.reset_mock(); derpp.reset_mock()
                scout.main()
                self.assertEqual(train.call_count + derpp.call_count, 0)


if __name__ == "__main__":
    unittest.main()
