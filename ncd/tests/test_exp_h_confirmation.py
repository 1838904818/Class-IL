"""Section19 control-flow tests; no learner, CUDA workload, or dataset is run."""
from __future__ import annotations

import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from ncd.data import PreparedData
from ncd import run_exp_h_confirmation as runner


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        x = np.arange(30, dtype=np.float32).reshape(10, 3)
        y = np.repeat(np.arange(5, dtype=np.int64), 2)
        self.data = PreparedData(x, y, x + 1, y.copy(), list("abcde"),
                                 [[0, 1], [2, 3], [4]], ["x0", "x1", "x2"], [])
        self.meta = {"feature_alignment": [{"feature": "x0", "action": "identity"}]}
        self.units, self.classes = runner.validate_data(self.data, x, y, self.meta)
        self.config = {"device": "cpu", "exp_h_smoke": True}
        self.selection = {"lambdas": {"replay": 0.0}, "git_commit": "b" * 40}
        self.provenance = {"git_commit": "a" * 40, "source_hashes": {"fixture": "0" * 64},
                           "receipts": {"fixture": "1" * 64}, "command": "unit fixture only", "log_path": "fixture.log"}

    def _fake_train(self, data, config, identity, output, device, evaluate_only):
        paths = []
        for t in range(3):
            p = output / f"checkpoint_{t}.pt"
            torch.save({"identity": identity, "model_state": {"fixture": torch.zeros(1)},
                        "training_seconds": 10.0 * (t + 1)}, p)
            paths.append(p)
        return paths, [{"task_index": t} for t in range(3)]

    def _fake_evaluate(self, data, x_shift, y_shift, config, identity, paths, history, output, device):
        rows = []
        for transition, c in sorted(self.units):
            name = f"pi_vectors_transition_{transition:03d}.npz"
            np.savez(output / name, fixture=np.zeros(3))
            rows.append({"setting": identity["setting"], "method": identity["method"],
                         "seed": identity["seed"], "transition": transition, "class_id": c,
                         "old_pair_pi_drift": 0.1, "self_margin_pi_drift": 0.2,
                         "shift_recall_before": 0.7, "shift_recall_after": 0.6,
                         "id_recall_before": 0.8, "id_recall_after": 0.75,
                         "pi_vectors_file": name})
        retention = [{"setting": identity["setting"], "method": identity["method"],
                      "seed": identity["seed"], "class_id": c,
                      "shift_recall_final": 0.6, "shift_recall_best": 0.7} for c in self.classes]
        return rows, retention, [{"task_index": t} for t in range(3)]

    def _cell(self, output, **kwargs):
        return runner.run_cell("S4", "replay", 1, self.data, self.data.x_test, self.data.y_test,
                               self.meta, self.config, self.selection, output, self.provenance,
                               runner.array_binding(self.data, self.data.x_test, self.data.y_test),
                               self.units, self.classes, **kwargs)

    def test_five_classes_have_six_units_and_unchanged_frozen_helpers(self):
        from ncd import run_exp_h
        self.assertEqual(len(self.units), 6)
        self.assertEqual(self.classes, set(range(5)))
        self.assertIs(runner._train_checkpoints, run_exp_h._train_checkpoints)
        self.assertIs(runner._evaluate, run_exp_h._evaluate)
        self.assertEqual(tuple(runner.SEEDS), (1, 2, 3, 4, 42))
        self.assertEqual(len(runner.METHODS), 8)

    def test_resumes_valid_result_without_training_and_rejects_missing_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            with patch.object(runner, "_train_checkpoints", side_effect=self._fake_train), patch.object(runner, "_evaluate", side_effect=self._fake_evaluate):
                result = self._cell(p)
            with patch.object(runner, "_train_checkpoints", side_effect=AssertionError("must not train")), patch.object(runner, "_evaluate", side_effect=AssertionError("must not evaluate")):
                self.assertEqual(self._cell(p), result)
            self.assertEqual(result["training_seconds_total"], 30.0)
            self.assertEqual(result["elapsed_seconds"], result["elapsed_seconds_this_attempt"])
            self.assertEqual(result["source_hashes"], self.provenance["source_hashes"])
            (p / result["checkpoints"][0]).unlink()
            with self.assertRaises(FileNotFoundError):
                self._cell(p)

    def test_rejects_source_change_partial_duplicate_and_nonfinite_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            with patch.object(runner, "_train_checkpoints", side_effect=self._fake_train), patch.object(runner, "_evaluate", side_effect=self._fake_evaluate):
                good = self._cell(p)
            for mutate in (lambda d: d["identity"].update(git_commit="f" * 40),
                           lambda d: d["rows"].pop(),
                           lambda d: d["rows"].append(d["rows"][0]),
                           lambda d: d["rows"][0].update(old_pair_pi_drift=float("nan")),
                           lambda d: d.update(training_seconds_total=0.0),
                           lambda d: d.update(elapsed_seconds=-1.0)):
                bad = copy.deepcopy(good)
                mutate(bad)
                with self.assertRaises(ValueError):
                    runner.validate_cell(bad, good["identity"], p, self.units, self.classes)

    def test_resume_after_evaluation_failure_preserves_cumulative_training_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with patch.object(runner, "_train_checkpoints", side_effect=self._fake_train), \
                 patch.object(runner, "_evaluate", side_effect=RuntimeError("fixture evaluation interrupted")), \
                 patch.object(runner.time, "perf_counter", return_value=10.0):
                with self.assertRaisesRegex(RuntimeError, "evaluation interrupted"):
                    self._cell(output)
            self.assertFalse((output / "exp_h_result.json").exists())

            def restore_only(data, config, identity, path, device, evaluate_only):
                checkpoints = [path / f"checkpoint_{t}.pt" for t in range(3)]
                for checkpoint in checkpoints:
                    self.assertEqual(runner._load_checkpoint(checkpoint, identity)["identity"], identity)
                return checkpoints, [{"task_index": t} for t in range(3)]

            with patch.object(runner, "_train_checkpoints", side_effect=restore_only), \
                 patch.object(runner, "_evaluate", side_effect=self._fake_evaluate), \
                 patch.object(runner.time, "perf_counter", side_effect=[100000.0, 100002.0]):
                result = self._cell(output, evaluate_only=True)
            self.assertEqual(result["training_seconds_total"], 30.0)
            self.assertEqual(result["elapsed_seconds_this_attempt"], 2.0)
            self.assertEqual(result["elapsed_seconds"], 2.0)
            self.assertEqual(result["git_commit"], self.provenance["git_commit"])
            self.assertEqual(result["receipts"], self.provenance["receipts"])

    def test_array_binding_detects_content_change(self):
        before = runner.array_binding(self.data, self.data.x_test, self.data.y_test)
        self.data.x_train[0, 0] += 1
        after = runner.array_binding(self.data, self.data.x_test, self.data.y_test)
        self.assertNotEqual(before["x_train"]["sha256"], after["x_train"]["sha256"])

    def test_rejects_nonfinite_saved_vectors_or_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            with patch.object(runner, "_train_checkpoints", side_effect=self._fake_train), patch.object(runner, "_evaluate", side_effect=self._fake_evaluate):
                good = self._cell(p)
            vector = p / good["rows"][0]["pi_vectors_file"]
            np.savez(vector, fixture=np.full(3, float("nan")))
            with self.assertRaisesRegex(ValueError, "PI vectors"):
                runner.validate_cell(good, good["identity"], p, self.units, self.classes)
            np.savez(vector, fixture=np.zeros(3))
            torch.save({"identity": good["identity"], "model_state": {"bad": torch.tensor(float("inf"))}}, p / good["checkpoints"][0])
            with self.assertRaisesRegex(ValueError, "nonfinite tensor"):
                runner.validate_cell(good, good["identity"], p, self.units, self.classes)

    def test_frozen_config_and_mapping_are_required_before_any_model(self):
        config = json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        config["frozen_mapping_sha256"] = None
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "preparation.json").write_text(json.dumps({"frozen_mapping_sha256": "d" * 64}), encoding="utf-8")
            p = root / "config.json"
            args = argparse.Namespace(config=p, cleaned_data_root=root, device=None, smoke=False)
            p.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mapping must be frozen"):
                runner.load_config(args)
            config["frozen_mapping_sha256"] = "d" * 64
            p.write_text(json.dumps(config), encoding="utf-8")
            loaded, selection, receipt = runner.load_config(args)
            self.assertEqual(loaded["epochs_per_task"], 8)
            self.assertEqual(len(selection["lambdas"]), 8)
            self.assertEqual(receipt["config_file_sha256"], runner.digest(p))
            args.smoke = True
            loaded, _, _ = runner.load_config(args)
            self.assertEqual((loaded["epochs_per_task"], loaded["smoke_rows_per_class"]), (1, 64))
            config["floor_recall"] = 0.1
            p.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "floor_recall"):
                runner.load_config(args)

    @staticmethod
    def stats(rho4=0.3, rho5=0.1, status="complete"):
        return {"settings": {s: {"status": status, "unit_level": {runner.PRIMARY:
                 {"status": status, "rho": rho, "ci95": [rho - 0.05, rho + 0.05]}}}
                 for s, rho in [("S4", rho4), ("S5", rho5)]}}

    def test_confirmation_logic_and_pending_are_not_legacy_section17(self):
        verdict = runner.confirmation_verdict
        s1_pass = {"s1_adjusted_condition_met": True,
                   "primary_adjusted": {"status": "complete", "rho": 0.3, "ci95": [0.1, 0.5]}}
        s1_fail = {"s1_adjusted_condition_met": False,
                   "primary_adjusted": {"status": "complete", "rho": 0.1, "ci95": [0.05, 0.15]}}
        self.assertEqual(verdict(self.stats(), s1_pass)["decision"], "harm_confirmed")
        self.assertEqual(verdict(self.stats(), s1_fail)["decision"], "does_not_meet_confirmation_gate")
        self.assertEqual(verdict(self.stats(), None)["decision"], "pending")
        self.assertEqual(verdict(self.stats(status="pending"), s1_pass)["decision"], "pending")
        self.assertEqual(verdict(self.stats(0.1, 0.1), None)["decision"], "harm_not_confirmed")
        self.assertEqual(verdict(self.stats(), s1_pass, smoke=True)["decision"], "pending")

    def test_s1_condition_requires_consistent_finite_numeric_evidence(self):
        valid = {"s1_adjusted_condition_met": True,
                 "primary_adjusted": {"status": "complete", "rho": 0.3, "ci95": [0.1, 0.5]}}
        for mutate in (lambda d: d.update(s1_adjusted_condition_met=False),
                       lambda d: d["primary_adjusted"].update(rho=float("nan")),
                       lambda d: d["primary_adjusted"].update(rho=True),
                       lambda d: d["primary_adjusted"].update(ci95=[0.5, 0.1]),
                       lambda d: d["primary_adjusted"].update(ci95=[0.1]),
                       lambda d: d.pop("primary_adjusted")):
            bad = copy.deepcopy(valid)
            mutate(bad)
            with self.assertRaises(ValueError):
                runner.confirmation_verdict(self.stats(), bad)
        unidentified = {"primary_adjusted": {"status": "unidentified"}, "s1_adjusted_condition_met": None}
        self.assertEqual(runner.confirmation_verdict(self.stats(), unidentified)["decision"], "pending")

    def test_aggregation_suppresses_legacy_verdict_and_keeps_exact_input_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp = root / "exp_h_result.json"
            inp.write_text('{"execution_status":"smoke_complete"}', encoding="utf-8")
            fake = {**self.stats(), "verdict": {"decision": "harm_supported"}, "report_status": "partial"}
            with patch.object(runner, "analyze", return_value=fake):
                output = runner.aggregate({"execution_status": "smoke_complete"}, inp, root, root / "missing.json", True)
            self.assertEqual(output["verdict"]["decision"], "pending")
            self.assertNotIn("harm_supported", json.dumps(output))
            self.assertEqual(output["input_sha256"], runner.digest(inp))
            self.assertEqual(json.loads((root / "exp_h_analysis.json").read_text()), output)

    def test_smoke_control_flow_reaches_real_analyzer_without_any_model_fit(self):
        """Exercise the final aggregation path, not a real dataset/model smoke."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = argparse.Namespace(output_dir=root, log_path=root / "run.log", smoke=True,
                                      evaluate_only=False, s1_robustness=root / "missing.json")
            selection = {"lambdas": {method: 0.0 for method in runner.METHODS},
                         "git_commit": "b" * 40}
            prepared = (self.data, self.data.x_test, self.data.y_test, self.meta)
            with patch.object(runner, "load_config", return_value=(self.config, selection, {"fixture": "a" * 64})), \
                 patch.object(runner, "source_commit", return_value="a" * 40), \
                 patch("ncd.exp_h_netflow.prepare_setting", return_value=prepared), \
                 patch.object(runner, "_train_checkpoints", side_effect=self._fake_train), \
                 patch.object(runner, "_evaluate", side_effect=self._fake_evaluate):
                runner.run(args)
                with patch.object(runner, "_train_checkpoints", side_effect=AssertionError("must not retrain")), \
                     patch.object(runner, "_evaluate", side_effect=AssertionError("must not reevaluate")):
                    runner.run(args)
            combined = json.loads((root / "exp_h_result.json").read_text(encoding="utf-8"))
            analysis = json.loads((root / "exp_h_analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(combined["execution_status"], "smoke_complete")
            self.assertEqual((len(combined["cells"]), len(combined["rows"])), (16, 96))
            self.assertEqual(combined["training_seconds_total"], 16 * 30.0)
            self.assertEqual(combined["training_seconds_total"], sum(c["training_seconds_total"] for c in combined["cells"]))
            self.assertEqual(analysis["execution_status"], "smoke_complete")
            self.assertEqual(analysis["report_status"], "partial")
            self.assertEqual(analysis["verdict"]["decision"], "pending")
            self.assertEqual(analysis["input_sha256"], runner.digest(root / "exp_h_result.json"))


if __name__ == "__main__":
    unittest.main()
