"""Synthetic analysis fixtures only, never evidence of Section22 efficacy."""
import copy
import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ncd.analyze_step_budget import (
    DATASETS, METHODS, NAMED_CLASSES, SEEDS, analyze, decide, export_tables,
    fixed_best, load_cells, paired_delta, recompute, summarize,
)


def metrics(cm, classes, seen):
    rows = []
    for c in seen:
        support = sum(cm[c])
        predicted = sum(row[c] for row in cm)
        rows.append({"class_id": c, "class_name": classes[c], "support": support,
                     "recall": cm[c][c] / support if support else None,
                     "f1": 2 * cm[c][c] / (support + predicted) if support + predicted else 0.0})
    recalls = [r["recall"] for r in rows if r["recall"] is not None]
    return {"confusion_matrix": cm, "confusion_labels": list(range(len(classes))),
            "per_class": rows, "evaluated_rows": sum(map(sum, cm)),
            "macro_f1": sum(r["f1"] for r in rows) / len(rows),
            "balanced_accuracy": sum(recalls) / len(recalls) if recalls else None}


def fixture(dataset, method, seed, budget=0, recall=40, joint_recall=80):
    classes = ["Normal", *[name for d, name in NAMED_CLASSES if d == dataset]]
    tasks = [[0], list(range(1, len(classes))) ]
    n = len(classes)
    cm = [[0] * n for _ in range(n)]
    cm[0][0], cm[0][1] = 90, 10
    for c in range(1, n):
        value = joint_recall if method == "joint" else recall
        cm[c][c], cm[c][0] = value, 100 - value
    before = [[0] * n for _ in range(n)]
    before[0][0] = 100
    config = {"epochs_per_task": 8, "batch_size": 16, "buffer_per_class": 200,
              "learning_rate": .001, "smoke": False, "device": "cpu"}
    if budget:
        config["min_steps_per_task"] = budget
    checkpoints = []
    for t in range(2):
        cp = {"task_index": t, "new_classes": tasks[t], "seen_classes": [0] if t == 0 else list(range(n)),
              "train_rows": 40, "id": metrics(before if t == 0 else cm, classes, [0] if t == 0 else list(range(n)))}
        if budget:
            batches, target = 3, max(24, budget)
            partial_examples = (target % batches) * 16
            examples = (target // batches) * 40 + partial_examples
            cp["training_schedule"] = {"rows_per_epoch": 40, "batches_per_epoch": batches,
                                       "baseline_steps": 24, "target_steps": target,
                                       "optimizer_steps": target, "min_steps_per_task": budget,
                                       "requested_epochs": 8, "completed_epochs": target // batches,
                                       "partial_epoch_batches": target % batches,
                                       "partial_epoch_examples": partial_examples,
                                       "examples_presented": examples, "effective_epochs": examples / 40}
        checkpoints.append(cp)
    return {"identity": {"dataset": dataset, "method": method, "seed": seed,
                         "effective_config": config, "class_order": classes, "tasks": tasks,
                         "data_binding": {"x_test": {"sha256": "fixture", "shape": [n * 100, 2]}},
                         "regularization": 0, "git_commit": "fixture-only"},
            "class_order": classes, "tasks": tasks, "train_counts": {str(c): 40 for c in range(n)},
            "checkpoints": checkpoints, "completed": True}


def matrix(recall=40, joint_recall=80):
    baseline = [fixture(d, m, s, joint_recall=joint_recall) for d in DATASETS for m in ("joint", *METHODS) for s in SEEDS]
    arms = [[fixture(d, m, s, budget, recall, joint_recall) for d in DATASETS for m in METHODS for s in SEEDS] for budget in (500, 2000)]
    return baseline, *arms


class StepBudgetAnalysisTests(unittest.TestCase):
    def test_complete_coverage_and_threshold_A_is_inclusive(self):
        out = analyze(*matrix())
        self.assertEqual(out["section22_3"]["decision"], "A")
        self.assertTrue(out["scientific_comparisons_final"])
        self.assertEqual([out["coverage"][b]["complete_cells"] for b in ("original", "500", "2000")], [90, 75, 75])
        evidence = out["named_class_decision_evidence"]
        self.assertEqual(len(evidence), 5)
        self.assertTrue(all(r["ratio_of_five_seed_means"] == .5 for r in evidence))
        row = next(r for r in out["recall"] if r["budget"] == "2000" and r["role"] == "old")
        self.assertAlmostEqual(row["recall"]["mean"], .9)
        self.assertAlmostEqual(row["last_task_recall_change_pp"]["mean"], -10)
        self.assertEqual(row["recall"]["n"], 5)
        self.assertAlmostEqual(row["recall"]["sample_sd"], 0)

    def test_B_requires_all_five_strictly_below_twenty_percent(self):
        self.assertEqual(analyze(*matrix(recall=15))["section22_3"]["decision"], "B")
        self.assertEqual(analyze(*matrix(recall=16))["section22_3"]["decision"], "intermediate")

    def test_mixed_class_evidence_is_not_pooled_or_cherry_picked(self):
        rows = [{"dataset": d, "class_name": c, "ratio_of_five_seed_means": .7} for d, c in NAMED_CLASSES]
        rows[-1]["ratio_of_five_seed_means"] = .1
        self.assertEqual(decide(rows, True)["decision"], "intermediate")
        self.assertEqual(decide(rows[:-1], True)["decision"], "pending")

    def test_complete_named_replay_results_do_not_override_incomplete_matrix(self):
        original, s500, s2000 = matrix()
        s500.pop()
        out = analyze(original, s500, s2000)
        self.assertEqual(out["section22_3"]["decision"], "pending")
        self.assertEqual(out["execution_status"], "partial")
        self.assertFalse(out["scientific_comparisons_final"])
        row = next(r for r in out["recall"] if r["dataset"] == DATASETS[-1] and r["method"] == METHODS[-1] and r["budget"] == "500")
        self.assertEqual(row["recall"]["n"], 4)
        self.assertIsNone(row["recall"]["by_seed"]["42"])
        self.assertTrue(all(not r["complete"] for r in out["G1"] if r["dataset"] == DATASETS[-1] and r["budget"] == "500"))

    def test_zero_joint_ratio_is_undefined_not_success(self):
        out = analyze(*matrix(joint_recall=0))
        self.assertEqual(out["section22_3"]["decision"], "pending")
        for row in out["named_class_decision_evidence"]:
            self.assertIsNone(row["ratio_of_five_seed_means"])
            self.assertTrue(all(v is None for v in row["paired_ratio_by_seed"].values()))

    def test_ratio_of_means_is_distinct_from_mean_seed_ratio(self):
        original, s500, s2000 = matrix()
        for cell in original:
            identity = cell["identity"]
            if identity["method"] == "joint" and identity["seed"] == 1:
                replacement = fixture(identity["dataset"], "joint", 1, joint_recall=20)
                cell["checkpoints"][-1] = replacement["checkpoints"][-1]
        out = analyze(original, s500, s2000)
        row = out["named_class_decision_evidence"][0]
        self.assertAlmostEqual(row["ratio_of_five_seed_means"], .4 / .68)
        self.assertAlmostEqual(sum(row["paired_ratio_by_seed"].values()) / 5, .8)

    def test_fixed_method_selection_never_seed_oracle(self):
        values = {m: dict.fromkeys(SEEDS, .1) for m in METHODS}
        values["joint"] = dict.fromkeys(SEEDS, .9)
        values["replay"] = dict(zip(SEEDS, [.9, .5, .9, .5, .9]))
        values["icarl"] = dict.fromkeys(SEEDS, .8)
        result = fixed_best(values)
        self.assertEqual(result["best_cil_method"], "icarl")
        self.assertAlmostEqual(result["gap_pp"]["mean"], 10)
        del values["derpp"][42]
        self.assertFalse(fixed_best(values)["complete"])

    def test_summaries_use_exact_seed_pairs_sample_sd_and_no_imputation(self):
        a, b = {1: .9, 2: .5}, {2: .4, 3: .3}
        result = paired_delta(a, b)
        self.assertEqual(result["n"], 1)
        self.assertAlmostEqual(result["mean"], 10)
        self.assertIsNone(result["sample_sd"])
        self.assertIsNone(result["by_seed"]["1"])
        summary = summarize({1: 0, 2: 1, 3: None})
        self.assertEqual(summary["n"], 2)
        self.assertAlmostEqual(summary["sample_sd"], math.sqrt(.5))
        with self.assertRaises(ValueError):
            summarize({1: float("nan")})

    def test_stored_summary_corruption_rejected(self):
        for key in ("macro_f1", "balanced_accuracy"):
            with self.subTest(key=key):
                original, s500, s2000 = matrix()
                s2000[0]["checkpoints"][-1]["id"][key] = 999
                with self.assertRaisesRegex(ValueError, "confusion recomputation"):
                    analyze(original, s500, s2000)
        cell = fixture(DATASETS[0], "replay", 1)
        metric = cell["checkpoints"][-1]["id"]
        metric["per_class"][0]["recall"] = .123
        with self.assertRaisesRegex(ValueError, "Per-class metrics"):
            recompute(metric, cell["class_order"], [0, 1])

    def test_confusion_axis_and_noninteger_counts_rejected(self):
        cell = fixture(DATASETS[0], "replay", 1)
        metric = cell["checkpoints"][-1]["id"]
        for corrupt in ("labels", "float", "bool", "duplicate_class"):
            value = copy.deepcopy(metric)
            if corrupt == "labels":
                value["confusion_labels"].reverse()
            elif corrupt == "float":
                value["confusion_matrix"][0][0] = 90.0
            elif corrupt == "bool":
                value["confusion_matrix"][0][0] = True
            else:
                value["per_class"].append(value["per_class"][0])
            with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                recompute(value, cell["class_order"], [0, 1])

    def test_missing_test_support_is_null_recall_not_zero(self):
        m = metrics([[10, 0], [0, 0]], ["Normal", "Attack"], [0, 1])
        result = recompute(m, ["Normal", "Attack"], [0, 1])
        self.assertIsNone(result["recall"][1])
        self.assertEqual(result["balanced_accuracy"], 1)
        self.assertEqual(result["macro_f1"], .5)

    def test_schedule_rejects_wrong_optimizer_or_exposure_counts(self):
        for field in ("optimizer_steps", "target_steps", "examples_presented", "partial_epoch_examples"):
            original, s500, s2000 = matrix()
            s2000[0]["checkpoints"][0]["training_schedule"][field] += 1
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Training schedule"):
                analyze(original, s500, s2000)
        original, s500, s2000 = matrix()
        s500[0]["identity"]["effective_config"]["min_steps_per_task"] = 2000
        with self.assertRaisesRegex(ValueError, "Wrong min_steps_per_task"):
            analyze(original, s500, s2000)

    def test_scientific_configuration_and_data_binding_must_match(self):
        original, s500, s2000 = matrix()
        s500[0]["identity"]["effective_config"]["learning_rate"] = .01
        with self.assertRaisesRegex(ValueError, "Scientific config changed"):
            analyze(original, s500, s2000)
        original, s500, s2000 = matrix()
        s2000[0]["identity"]["data_binding"]["x_test"]["sha256"] = "other-data"
        with self.assertRaisesRegex(ValueError, "Data/class/task/support mismatch"):
            analyze(original, s500, s2000)

    def test_execution_device_and_thread_count_are_not_scientific_changes(self):
        original, s500, s2000 = matrix()
        s500[0]["identity"]["effective_config"].update(device="cuda", cpu_threads=2)
        self.assertEqual(analyze(original, s500, s2000)["section22_3"]["decision"], "A")

    def test_duplicate_or_extra_cells_and_truncated_checkpoint_sequence_rejected(self):
        original, s500, s2000 = matrix()
        with self.assertRaisesRegex(ValueError, "Duplicate cell"):
            analyze(original, s500 + s500[:1], s2000)
        extra = copy.deepcopy(s500[0])
        extra["identity"]["dataset"] = "CSE-CIC-IDS2018"
        with self.assertRaisesRegex(ValueError, "Unexpected cell"):
            analyze(original, s500 + [extra], s2000)
        s2000[0]["checkpoints"].pop(0)
        with self.assertRaisesRegex(ValueError, "every checkpoint"):
            analyze(original, s500, s2000)

    def test_unfinished_cells_are_explicit_and_do_not_enter_summary(self):
        original, s500, s2000 = matrix()
        s2000[0]["completed"] = False
        out = analyze(original, s500, s2000)
        self.assertEqual(out["coverage"]["2000"]["complete_cells"], 74)
        self.assertEqual(len(out["coverage"]["2000"]["unfinished_cells"]), 1)
        self.assertEqual(out["section22_3"]["decision"], "pending")

    def test_smoke_cannot_emit_formal_verdict(self):
        arms = matrix()
        for arm in arms:
            for cell in arm:
                cell["identity"]["effective_config"]["smoke"] = True
        out = analyze(*arms)
        self.assertEqual(out["execution_status"], "smoke_only")
        self.assertEqual(out["section22_3"]["decision"], "pending")
        self.assertFalse(out["scientific_comparisons_final"])

    def test_empty_input_is_pending_and_exports_headers(self):
        out = analyze([], [], [])
        self.assertEqual(out["section22_3"]["decision"], "pending")
        self.assertEqual(out["recall"], [])
        with tempfile.TemporaryDirectory() as tmp:
            export_tables(out, Path(tmp))
            with (Path(tmp) / "recall_by_seed.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 0)

    def test_input_loader_skips_indexes_and_handles_missing_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(load_cells(root / "not_started"), ([], []))
            (root / "result.json").write_text(json.dumps({"cells": [], "expected_cells": 75}))
            (root / "cell").mkdir()
            (root / "cell" / "result.json").write_text(json.dumps(fixture(DATASETS[0], "replay", 1)))
            paths, cells = load_cells(root)
            self.assertEqual(len(paths), 1)
            self.assertEqual(len(cells), 1)

    def test_cli_writes_portable_partial_outputs_without_touching_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_bytes = {}
            for name, budget in (("original", 0), ("s500", 500), ("s2000", 2000)):
                arm = root / name
                arm.mkdir()
                path = arm / "result.json"
                path.write_text(json.dumps(fixture(DATASETS[0], "replay", 1, budget)), encoding="utf-8")
                source_bytes[path] = path.read_bytes()
            script = Path(__file__).resolve().parents[1] / "analyze_step_budget.py"
            output = root / "analysis"
            command = [sys.executable, "-I", str(script), "--baseline", str(root / "original"),
                       "--steps-500", str(root / "s500"), "--steps-2000", str(root / "s2000"),
                       "--output-dir", str(output)]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout)["execution_status"], "partial")
            value = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(value["section22_3"]["decision"], "pending")
            self.assertNotIn(tmp, value["analysis_provenance"]["command"])
            self.assertEqual({p.name for p in output.iterdir()}, {
                "analysis.json", "analysis.log", "recall_by_seed.csv", "recall_summary.csv",
                "metrics_by_seed.csv", "metrics_summary.csv"})
            for path, data in source_bytes.items():
                self.assertEqual(path.read_bytes(), data)
            command[-1] = str(root / "original" / "analysis")
            rejected = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("separate from all immutable raw input roots", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
