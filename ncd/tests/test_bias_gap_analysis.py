"""Small synthetic fixtures; never evidence of experimental efficacy."""
import copy
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from ncd.analyze_bias_gap import (
    BINS, DATASETS, EXPECTED_BEST, METHODS, SEEDS, analyze, canonical,
    class_metrics, error_flow, load_arm, ratio, sha, validate_comparability,
    write_outputs,
)
from ncd.analyze_step_budget import validate_cell

CLASSES = ["Normal", "A", "B", "C", "D"]
TASKS = [[0, 1], [2, 3], [4]]
JOINT = [[90, 2, 3, 1, 4], [2, 70, 10, 6, 12], [3, 7, 80, 4, 6], [4, 2, 5, 85, 4], [4, 6, 5, 5, 80]]
BEST = [[80, 1, 4, 5, 10], [5, 60, 5, 20, 10], [2, 3, 85, 5, 5], [4, 1, 10, 75, 10], [1, 1, 1, 2, 95]]


def metric(cm, seen=None):
    seen = list(range(5)) if seen is None else seen
    rows = []
    for c in seen:
        support, predicted, tp = sum(cm[c]), sum(r[c] for r in cm), cm[c][c]
        flow = error_flow(cm, c, TASKS)
        rows.append({"class_id": c, "class_name": CLASSES[c], "support": support,
                     "precision": ratio(tp, predicted), "recall": ratio(tp, support),
                     "f1": ratio(2 * tp, support + predicted) if support + predicted else 0.0,
                     "error_to_normal": flow["counts"]["normal"],
                     "error_to_newer_class": flow["counts"]["newer_classes"],
                     "error_to_other_old": flow["counts"]["other_old_or_same_task"]})
    recalls = [r["recall"] for r in rows if r["recall"] is not None]
    return {"confusion_matrix": cm, "confusion_labels": list(range(5)), "per_class": rows,
            "evaluated_rows": sum(map(sum, cm)), "macro_f1": sum(r["f1"] for r in rows) / len(rows),
            "balanced_accuracy": sum(recalls) / len(recalls),
            "normal_fpr": ratio(sum(cm[0]) - cm[0][0], sum(cm[0]))}


def cell(dataset, method, seed):
    config = {"epochs_per_task": 8, "batch_size": 256, "buffer_per_class": 200,
              "smoke": False, "learning_rate": .001, "device": "cpu"}
    if method != "joint":
        config["min_steps_per_task"] = 2000
    identity = {"dataset": dataset, "method": method, "seed": seed,
                "effective_config": config, "effective_config_sha256": canonical(config),
                "config_sha256": "1" * 64, "data_metadata_sha256": "2" * 64,
                "git_commit": "3" * 40, "source_hashes": {"synthetic.py": "4" * 64},
                "data_binding": {"x_test": {"sha256": "5" * 64, "shape": [500, 2]}},
                "feature_names": ["a", "b"], "class_order": CLASSES, "tasks": TASKS}
    if method == "joint":
        final = copy.deepcopy(JOINT)
    elif method == EXPECTED_BEST[dataset]:
        final = copy.deepcopy(BEST)
        # Vary errors/support fractions across seeds without changing row support.
        delta = SEEDS.index(seed) * 2
        final[0][0] -= delta
        final[0][4] += delta
    else:
        final = [[100 if c == 0 else 0 for c in range(5)] for _ in range(5)]
    checkpoints = []
    for t, group in enumerate(TASKS):
        seen = sorted(c for g in TASKS[:t + 1] for c in g)
        cm = final if t == 2 else [[100 if r == c and r in seen else 0 for c in range(5)] for r in range(5)]
        cp = {"task_index": t, "new_classes": group, "seen_classes": seen,
              "id": metric(cm, seen), "identity_hash": canonical(identity), "train_rows": 512}
        if method != "joint":
            cp["training_schedule"] = {"rows_per_epoch": 512, "batches_per_epoch": 2,
                 "baseline_steps": 16, "target_steps": 2000, "optimizer_steps": 2000,
                 "min_steps_per_task": 2000, "requested_epochs": 8,
                 "completed_epochs": 1000, "partial_epoch_batches": 0,
                 "partial_epoch_examples": 0, "examples_presented": 512000, "effective_epochs": 1000.0}
        checkpoints.append(cp)
    return {"identity": identity, "completed": True, "class_order": CLASSES, "tasks": TASKS,
            "train_counts": {str(c): 100 for c in range(5)}, "checkpoints": checkpoints}


def records():
    joint = {(d, "joint", s): validate_cell(cell(d, "joint", s), "original") for d in DATASETS for s in SEEDS}
    cil = {(d, m, s): validate_cell(cell(d, m, s), "2000") for d in DATASETS for m in METHODS for s in SEEDS}
    return joint, cil


def write_arm(root, baseline=False):
    rows = []
    for d in DATASETS:
        for m in (("joint",) if baseline else METHODS):
            for s in SEEDS:
                path = root / d / f"{m}_{s}" / "result.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(cell(d, m, s)), encoding="utf-8")
                rows.append({"dataset": d, "method": m, "seed": s, "result": path.relative_to(root).as_posix(),
                             "result_sha256": sha(path), "checkpoints": 3})
    index = {"execution_status": "complete", "smoke": False, "cells": rows}
    (root / "result.json").write_text(json.dumps(index), encoding="utf-8")
    return index


class BiasGapAnalysisTests(unittest.TestCase):
    def test_precision_recall_f1_use_correct_axes(self):
        result = class_metrics(metric(JOINT), CLASSES)
        self.assertAlmostEqual(result[0]["precision"], 90 / 103)
        self.assertEqual(result[0]["recall"], .9)
        self.assertEqual(result[0]["fp"], 13)
        self.assertAlmostEqual(result[0]["f1"], 180 / 203)

    def test_undefined_denominators_remain_null(self):
        cm = [[0] * 5 for _ in range(5)]
        cm[0][0] = 10
        result = class_metrics(metric(cm), CLASSES)
        self.assertIsNone(result[1]["precision"])
        self.assertIsNone(result[1]["recall"])
        self.assertIsNone(result[1]["f1"])
        self.assertEqual(result[1]["macro_f1_term"], 0)
        self.assertTrue(result[1]["f1_zero_division"])
        self.assertIsNone(ratio(0, 0))

    def test_false_precision_and_g4_summaries_rejected(self):
        data = metric(JOINT)
        data["per_class"][0]["precision"] = .123
        with self.assertRaisesRegex(ValueError, "precision disagrees"):
            class_metrics(data, CLASSES)
        joint, cil = records()
        cil[DATASETS[0], EXPECTED_BEST[DATASETS[0]], 1]["cell"]["checkpoints"][-1]["id"]["per_class"][1]["error_to_newer_class"] += 1
        with self.assertRaisesRegex(ValueError, "G4 counts"):
            analyze(joint, cil)

    def test_g4_relative_learning_time_and_mutually_exclusive_bins(self):
        flow = error_flow(JOINT, 1, TASKS)
        self.assertEqual(flow["counts"], {"normal": 2, "newer_classes": 28, "other_old_or_same_task": 0})
        # B -> C is same-task, not newer; B -> D is newer.
        self.assertEqual(error_flow(JOINT, 2, TASKS)["counts"], {"normal": 3, "newer_classes": 6, "other_old_or_same_task": 11})

    def test_additive_signed_contributions_reconcile_every_seed(self):
        out = analyze(*records())
        for d in out["datasets"]:
            for s in SEEDS:
                rows = [r for r in out["class_by_seed"] if r["dataset"] == d["dataset"] and r["seed"] == s]
                self.assertAlmostEqual(math.fsum(r["macro_f1_contribution_pp"] for r in rows), d["macro_f1"]["joint_minus_cil_pp"]["by_seed"][str(s)], places=12)
                self.assertTrue(any(r["macro_f1_contribution_pp"] < 0 for r in rows))
        self.assertEqual(len(out["class_by_seed"]), 75)
        self.assertEqual(len(out["fp_sources_by_seed"]), 600)

    def test_best_selection_is_dataset_fixed_five_seed_mean(self):
        joint, cil = records()
        # An oracle would pick replay on seed 1, but its mean is still inferior.
        for d in DATASETS:
            m = "replay"
            cp = cil[d, m, 1]["cell"]["checkpoints"][-1]
            cp["id"] = metric([[100 if r == c else 0 for c in range(5)] for r in range(5)])
            cil[d, m, 1] = validate_cell(cil[d, m, 1]["cell"], "2000")
        out = analyze(joint, cil)
        self.assertEqual({r["dataset"]: r["best_cil_method"] for r in out["selection"]}, EXPECTED_BEST)

    def test_fp_sources_exhaust_receiver_fp_and_have_paired_differences(self):
        out = analyze(*records())
        for d in DATASETS:
            for s in SEEDS:
                for arm in ("joint", "cil"):
                    for c in CLASSES:
                        rows = [r for r in out["fp_sources_by_seed"] if (r["dataset"], r["seed"], r["arm"], r["predicted_class"]) == (d, s, arm, c)]
                        self.assertEqual(sum(r["count"] for r in rows), rows[0]["receiver_fp"])
        row = next(r for r in out["fp_sources_paired"] if r["dataset"] == DATASETS[0] and r["predicted_class"] == "D" and r["true_class"] == "Normal")
        self.assertEqual(row["count"]["by_seed"]["1"], -6)
        self.assertEqual(row["count"]["mean"], -10)
        self.assertAlmostEqual(row["count"]["sample_sd"], math.sqrt(10))

    def test_pooled_g4_ratios_are_not_mean_seed_ratios(self):
        out = analyze(*records())
        row = next(r for r in out["g4_summary"] if r["dataset"] == DATASETS[0] and r["population"] == "all_old")
        values = row["arms"]["cil"]
        pooled = values["pooled_counts"]
        self.assertEqual(sum(pooled[b] for b in BINS), pooled["errors"])
        self.assertAlmostEqual(pooled["newer_classes_fraction_of_errors"], pooled["newer_classes"] / pooled["errors"])
        self.assertNotAlmostEqual(pooled["newer_classes_fraction_of_errors"], values["seed_summary"]["newer_classes_fraction_of_errors"]["mean"], places=8)
        old = [r for r in out["g4_class_by_seed"] if r["dataset"] == DATASETS[0]]
        self.assertNotIn("D", {r["true_class"] for r in old})

    def test_normal_fpr_uses_only_normal_support(self):
        row = analyze(*records())["datasets"][0]["normal_fpr"]
        self.assertEqual(row["joint"]["mean"], .1)
        self.assertAlmostEqual(row["cil"]["mean"], .24)
        self.assertAlmostEqual(row["cil_minus_joint_pp"]["mean"], 14)

    def test_saved_class_row_order_does_not_change_label_mapping(self):
        joint, cil = records()
        expected = analyze(joint, cil)
        for record in [*joint.values(), *cil.values()]:
            record["cell"]["checkpoints"][-1]["id"]["per_class"].reverse()
        self.assertEqual(analyze(joint, cil), expected)

    def test_data_and_scientific_config_drift_rejected(self):
        for mutation in ("data", "config"):
            joint, cil = records()
            identity = cil[DATASETS[0], "replay", 1]["cell"]["identity"]
            if mutation == "data":
                identity["data_binding"]["x_test"]["sha256"] = "6" * 64
            else:
                identity["effective_config"]["learning_rate"] = .5
            with self.assertRaises(ValueError):
                validate_comparability(joint, cil)

    def test_loader_checks_exact_selected_coverage_and_source_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            index = write_arm(root)
            records_, evidence = load_arm(root)
            self.assertEqual(len(records_), 75)
            self.assertEqual(evidence["selected_checkpoints"], 225)
            path = root / index["cells"][0]["result"]
            path.write_text(path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_arm(root)

    def test_loader_rejects_missing_duplicate_escape_and_nonformal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = write_arm(root)
            cases = []
            value = copy.deepcopy(original); value["cells"].pop(); cases.append(value)
            value = copy.deepcopy(original); value["cells"].append(value["cells"][0]); cases.append(value)
            value = copy.deepcopy(original); value["cells"][0]["result"] = "../result.json"; cases.append(value)
            value = copy.deepcopy(original); value["smoke"] = True; cases.append(value)
            for value in cases:
                (root / "result.json").write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    load_arm(root)

    def test_loader_rejects_checkpoint_identity_and_effective_config_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = write_arm(root, baseline=True)
            first = original["cells"][0]
            path = root / first["result"]
            clean = json.loads(path.read_text())
            for which in ("identity", "config"):
                value = copy.deepcopy(clean)
                if which == "identity":
                    value["checkpoints"][0]["identity_hash"] = "9" * 64
                else:
                    value["identity"]["effective_config"]["learning_rate"] = .4
                path.write_text(json.dumps(value))
                first["result_sha256"] = sha(path)
                (root / "result.json").write_text(json.dumps(original))
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    load_arm(root, baseline=True)

    def test_export_json_csv_roundtrip_preserves_zero_and_null(self):
        out = analyze(*records())
        out["provenance"] = {"inputs": {"original_joint": {"selected_cells": 15}, "s2000": {"selected_cells": 75}}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_outputs(out, root)
            restored = json.loads((root / "analysis.json").read_text())
            self.assertEqual(restored, out)
            with (root / "class_by_seed.csv").open(newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 75)
            self.assertEqual(len(list(root.iterdir())), 11)


if __name__ == "__main__":
    unittest.main()
