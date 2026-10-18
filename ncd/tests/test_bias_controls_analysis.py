"""Synthetic checks only: no training, no claim of Section23 efficacy."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from ncd.analyze_bias_controls import (
    CORRECTIONS, DATASETS, EXPECTED_BEST, ID_BINDINGS, METHODS, ROLES, SEEDS,
    SOURCE_PATHS, analyze, base_role, canonical, checkpoint_metrics, closure,
    decide, digest, export, finite, groups, load_controls, posthoc_steps, summarize_costs,
)
from ncd.tests.test_bias_gap_analysis import CLASSES, TASKS, cell as baseline_cell, metric
from ncd.analyze_step_budget import validate_cell


def references():
    joint, s2000 = {}, {}
    for d in DATASETS:
        for m in ("joint", *METHODS):
            for s in SEEDS:
                cell = baseline_cell(d, m, s)
                identity = cell["identity"]
                identity.update(regularization=0.0,
                    data_binding={k: {"shape": [500, 2] if k.startswith("x_") else [500], "sha256": str(i + 1) * 64, "dtype": "<f4" if k.startswith("x_") else "<i8"} for i, k in enumerate(ID_BINDINGS)})
                for cp in cell["checkpoints"]:
                    cp["identity_hash"] = canonical(identity)
                (joint if m == "joint" else s2000)[d, m, s] = validate_cell(cell, "original" if m == "joint" else "2000")
    return joint, s2000


def spec():
    return {"base_config": "base.json", "datasets": list(DATASETS), "best_methods": EXPECTED_BEST,
        "development_seeds": [5, 6], "reporting_seeds": list(SEEDS), "epochs_per_task": 8,
        "min_steps_per_task": 2000, "batch_size": 256, "development_validation_fraction": .1,
        "bic_training_memory_per_class": 180, "bic_holdout_per_class_max": 20,
        "ssil_old_replay_batch_size": 64,
        "candidates": {"wa": [{}], "bic": [{"steps": 100}, {"steps": 300}],
                       "ssil": [{"tkd_weight": .5}, {"tkd_weight": 1}],
                       "cbf": [{"epochs": 10}, {"epochs": 30}]}}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")


def make_cell(root, d, base, correction, candidate, seed, phase, config, bound, s2000, selection_sha=None):
    value = copy.deepcopy(s2000[d, base, SEEDS[0]]["cell"])
    identity = value["identity"]
    identity.update(dataset=d, base_method=base, method=base + "+" + correction,
        correction=correction, candidate=candidate, seed=seed, phase=phase, smoke=False,
        config_sha256=bound["config_sha256"], config_input_hashes=bound["config_input_hashes"],
        source_hashes=bound["source_hashes"], git_commit=bound["git_commit"],
        hyperparameters=config["candidates"][correction][candidate], selection_seeds=[5, 6],
        evaluation_role=ROLES[phase], frozen_selection=None)
    identity["effective_config"]["ssil_old_replay_batch_size"] = 64
    identity["effective_config"]["buffer_per_class"] = 180 if correction == "bic" else 200
    if phase == "formal":
        identity["development_split"] = {"role": ROLES[phase]}
        identity["frozen_selection"] = {"path": "selection.json", "sha256": selection_sha,
            "selected_candidate": candidate, "selection_seeds": [5, 6]}
    else:
        identity["development_split"] = {"role": ROLES[phase], "seed": seed, "disjoint_indices": True,
            "fraction": .1, "training_rows": 450, "validation_rows": 50,
            "train_indices_sha256": "a" * 64, "validation_indices_sha256": "b" * 64}
        value["train_counts"] = {str(c): 90 for c in range(5)}
        for k, v in identity["data_binding"].items():
            v["shape"][0] = 450 if k.endswith("train") else 50
    relative = f"{phase}/{d}/{base}/{correction}/candidate_{candidate}_seed_{seed}/result.json"
    directory = root / relative
    for t, cp in enumerate(value["checkpoints"]):
        seen = cp["seen_classes"]
        if phase == "development":
            diagonal = 8 + candidate
            cm = [[0] * 5 for _ in range(5)]
            for c in seen:
                cm[c][c] = diagonal
                cm[c][seen[(seen.index(c) + 1) % len(seen)]] += 10 - diagonal
            cp["id"] = metric(cm, seen)
        cp.update(identity_hash=canonical(identity), correction_complete=True, evaluation_role=ROLES[phase],
            checkpoint_file=f"checkpoints/checkpoint_{t:03d}.pt", checkpoint_sha256="c" * 64,
            training_seconds=1., correction_seconds=.1, buffer_selection_seconds=.01, evaluation_seconds=.01,
            training_seconds_cumulative=999., correction={"optimizer_steps": 7 if correction in ("bic", "cbf") and t else 0},
            training_memory_counts={str(c): 10 for c in seen}, calibration_holdout_counts={})
        if correction == "wa" and t:
            cp["correction"] = {"applied": True, "gamma": 1.}
        save(directory.parent / Path(cp["checkpoint_file"]).with_suffix(".json"), cp)
    save(directory, value)
    requested = {"dataset": d, "base": base, "correction": correction, "candidate": candidate,
        "seed": seed, "phase": phase, "smoke": False}
    return {"identity": requested, "result": relative, "result_sha256": digest(directory),
        "macro_f1": value["checkpoints"][-1]["id"]["macro_f1"], "checkpoint_count": len(TASKS)}


def write_controls(root, s2000):
    config = spec()
    config_path = root.parent / (root.name + "_config.json")
    save(config_path, config)
    bound = {"git_commit": "d" * 40, "config_sha256": digest(config_path),
             "config_input_hashes": {"base.json": "e" * 64},
             "source_hashes": {p: "f" * 64 for p in SOURCE_PATHS}}
    cells, selections = [], []
    for d, base, correction in groups():
        candidates = []
        for n, parameters in enumerate(config["candidates"][correction]):
            proofs = []
            for seed in (5, 6):
                row = make_cell(root, d, base, correction, n, seed, "development", config, bound, s2000)
                cells.append(row)
                proofs.append({"seed": seed, **row})
            candidates.append({"candidate": n, "parameters": parameters,
                "mean_validation_macro_f1": sum(p["macro_f1"] for p in proofs) / 2,
                "development_results": proofs})
        selections.append({"dataset": d, "base": base, "correction": correction,
            "selected_candidate": len(candidates) - 1, "candidates": candidates})
    selection = {"schema_version": 1, "provenance": bound, "selection_seeds": [5, 6],
                 "selection_metric": "final training-held-out validation Macro-F1",
                 "tie_break": "lower candidate index", "selections": selections}
    save(root / "selection.json", selection)
    selection_sha = digest(root / "selection.json")
    for group in selections:
        for seed in SEEDS:
            cells.append(make_cell(root, group["dataset"], group["base"], group["correction"], group["selected_candidate"], seed, "formal", config, bound, s2000, selection_sha))
    index = {"smoke": False, "status": "complete", "provenance": bound,
             "expected_cells": {"development": 84, "formal": 120},
             "completed_cells": {"development": 84, "formal": 120},
             "selection_sha256": selection_sha, "cells": cells}
    save(root / "result.json", index)
    events = [dict(event=event, identity=row["identity"], attempt=1,
                   **({"process_seconds": 5., "exit_code": 0} if event == "child_exit" else {}))
              for row in cells for event in ("child_start", "child_exit")]
    (root / "driver_attempts.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    return config_path, index


def configurations(default=.1):
    return [{"dataset": d, "base_role": base_role(d, b), "correction": c,
             "closure": {"ratio_of_five_seed_means": default if d != "NSL-KDD" else None}}
            for d, b, c in groups()]


class BiasControlsAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.joint, cls.s2000 = references()
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name) / "controls"
        cls.config_path, cls.index = write_controls(cls.root, cls.s2000)
        cls.formal, cls.coverage, _ = load_controls(cls.root, cls.config_path, cls.s2000)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_complete_portable_grid_without_any_pt_payload(self):
        self.assertEqual(len(self.formal), 120)
        self.assertEqual(self.coverage["development_cells"], 84)
        self.assertTrue(self.coverage["selection_verified"])
        self.assertEqual(self.coverage["weights_verified"], 0)
        self.assertEqual(self.coverage["checkpoint_payload_status"], "not_read_recorded_hashes_only")
        self.assertFalse(list(self.root.rglob("*.pt")))

    def test_cost_totals_use_non_cumulative_timers_and_separate_elapsed(self):
        costs = self.coverage["costs"]
        totals = next(r for r in costs["rows"] if r["aggregation_level"] == "overall")
        self.assertEqual(totals["cells"], 204)
        self.assertEqual(totals["checkpoints"], 204 * len(TASKS))
        self.assertEqual(totals["training_seconds"], 204 * len(TASKS))
        self.assertAlmostEqual(totals["correction_seconds"], 204 * len(TASKS) * .1)
        self.assertEqual(totals["subprocess_seconds"], 204 * 5.)
        self.assertTrue(totals["process_timing_complete"])
        self.assertEqual(totals["failed_attempts"], 0)
        self.assertEqual(len(costs["rows"]), 1 + 2 + 48)
        phases = {r["phase"]: r for r in costs["rows"] if r["aggregation_level"] == "phase"}
        self.assertEqual(phases["development"]["cells"], 84)
        self.assertEqual(phases["formal"]["cells"], 120)
        ssil = [r for r in costs["rows"] if r["correction"] == "ssil"]
        self.assertTrue(all(r["base_optimizer_steps"] > 0 and r["posthoc_optimizer_steps"] == 0 for r in ssil))
        self.assertEqual(self.coverage["cost_attempt_log"]["sha256"], digest(self.root / "driver_attempts.jsonl"))

    def test_cost_missing_or_unclosed_attempts_are_not_complete_elapsed(self):
        record = next(iter(self.formal.values()))
        for events in (None, [], [{"event": "child_start", "identity": self.index["cells"][84]["identity"], "attempt": 1}]):
            row = summarize_costs([record], events)["rows"][0]
            self.assertIsNone(row["subprocess_seconds"])
            self.assertFalse(row["process_timing_complete"])
            self.assertEqual(row["recorded_subprocess_seconds"], 0)
            self.assertGreater(row["training_seconds"], 0)

    def test_closed_form_wa_is_zero_but_other_missing_step_counters_fail(self):
        self.assertEqual(posthoc_steps({"correction": {"gamma": 1.2}}, "wa"), 0)
        for correction in ("bic", "cbf", "ssil"):
            with self.assertRaisesRegex(ValueError, "Invalid post-hoc"):
                posthoc_steps({"correction": {}}, correction)

    def test_cost_failed_attempts_are_counted_and_duplicate_events_rejected(self):
        record = next(iter(self.formal.values()))
        identity = self.index["cells"][84]["identity"]
        events = [{"event": event, "identity": identity, "attempt": attempt,
                   **({"process_seconds": seconds, "exit_code": code} if event == "child_exit" else {})}
                  for attempt, seconds, code in ((1, 2., 1), (2, 5., 0))
                  for event in ("child_start", "child_exit")]
        row = summarize_costs([record], events)["rows"][0]
        self.assertEqual(row["subprocess_seconds"], 7.)
        self.assertEqual(row["failed_attempts"], 1)
        self.assertEqual(row["exited_attempts"], 2)
        self.assertEqual(row["training_seconds"], len(TASKS))
        with self.assertRaisesRegex(ValueError, "Duplicate cost attempt"):
            summarize_costs([record], events + [events[-1]])
        events[-1]["process_seconds"] = -1
        with self.assertRaisesRegex(ValueError, "Invalid subprocess"):
            summarize_costs([record], events)

    def test_explicit_payload_mode_refuses_unavailable_weights(self):
        with self.assertRaisesRegex(ValueError, "payload missing"):
            load_controls(self.root, self.config_path, self.s2000, "verify")

    def test_all_checkpoint_and_final_statistics_recomputed(self):
        out = analyze(self.joint, self.s2000, self.formal, self.coverage)
        self.assertTrue(out["scientific_comparisons_final"])
        self.assertEqual(len(out["configurations"]), 24)
        self.assertEqual(len(out["checkpoint_metrics"]), 24 * 3 * 5)
        initial_old = [r for r in out["checkpoint_metrics"] if r["task_index"] == 0 and r["metric"] == "old_macro_recall"]
        self.assertTrue(all(r["corrected"]["mean"] is None for r in initial_old))
        self.assertTrue(all(r["corrected"]["n"] == 0 for r in initial_old))

    def test_missing_formal_seed_is_null_and_never_final(self):
        records = dict(self.formal)
        records.pop(next(iter(records)))
        coverage = dict(self.coverage, formal_cells=119, controller_status="running")
        out = analyze(self.joint, self.s2000, records, coverage)
        self.assertEqual(out["section23_4"]["decision"], "pending")
        self.assertFalse(out["scientific_comparisons_final"])
        self.assertTrue(any(r["corrected"]["n"] == 4 for r in out["checkpoint_metrics"]))
        self.assertIsNone(out["best_corrected"][0]["configuration"])

    def test_closure_uses_fixed_best_not_replay_improvement(self):
        best, joint = dict.fromkeys(SEEDS, .8), dict.fromkeys(SEEDS, .9)
        corrected = dict.fromkeys(SEEDS, .85)
        result = closure(corrected, best, joint)
        self.assertAlmostEqual(result["ratio_of_five_seed_means"], .5)
        self.assertAlmostEqual(result["gain_over_fixed_best_pp"]["mean"], 5)
        # Replay=.60 would imply a 25pp same-base gain, but not 250% gap closure.
        self.assertIsNone(closure(corrected, joint, best)["ratio_of_five_seed_means"])
        self.assertIsNone(closure(corrected, best, best)["ratio_of_five_seed_means"])
        self.assertGreater(closure(dict.fromkeys(SEEDS, .95), best, joint)["ratio_of_five_seed_means"], 1)

    def test_ratio_of_means_is_not_mean_of_seed_ratios(self):
        best = dict.fromkeys(SEEDS, .5)
        joint = {s: .6 if s == 1 else .9 for s in SEEDS}
        corrected = dict.fromkeys(SEEDS, .6)
        result = closure(corrected, best, joint)
        self.assertAlmostEqual(result["ratio_of_five_seed_means"], .1 / .34)
        self.assertAlmostEqual(result["paired_seed_ratios"]["mean"], .4)

    def test_same_correction_same_base_role_seventy_is_inclusive(self):
        rows = configurations()
        for row in rows:
            if row["base_role"] == "s2000_best" and row["correction"] == "wa":
                row["closure"]["ratio_of_five_seed_means"] = .7
        result = decide(rows, True)
        self.assertEqual(result["decision"], "solved")
        self.assertEqual(result["strict_same_correction_same_base_role_70pct"], [{"base_role": "s2000_best", "correction": "wa"}])

    def test_cross_base_success_does_not_meet_confirmed_base_role_rule(self):
        rows = configurations()
        for row in rows:
            if row["correction"] == "bic" and (row["dataset"], row["base_role"]) in {("UNSW-NB15", "replay"), ("CIC-IDS-2017", "s2000_best")}:
                row["closure"]["ratio_of_five_seed_means"] = .8
        result = decide(rows, True)
        self.assertEqual(result["decision"], "intermediate")
        self.assertEqual(result["strict_same_correction_same_base_role_70pct"], [])
        self.assertEqual(result["cross_base_same_correction_70pct_sensitivity"], ["bic"])

    def test_selection_file_before_controller_binding_is_not_used(self):
        index = copy.deepcopy(self.index)
        index["cells"] = [r for r in index["cells"] if r["identity"]["phase"] == "development"]
        index["completed_cells"]["formal"] = 0
        index["status"] = "running"
        index.pop("selection_sha256")
        try:
            save(self.root / "result.json", index)
            formal, coverage, _ = load_controls(self.root, self.config_path, self.s2000)
            self.assertEqual(formal, {})
            self.assertFalse(coverage["selection_verified"])
            self.assertEqual(coverage["controller_sha256"], digest(self.root / "result.json"))
        finally:
            save(self.root / "result.json", self.index)

    def test_all_source_and_config_input_hash_keys_are_required(self):
        for field in ("source_hashes", "config_input_hashes"):
            index = copy.deepcopy(self.index)
            index["provenance"][field].pop(next(iter(index["provenance"][field])))
            try:
                save(self.root / "result.json", index)
                with self.assertRaises(ValueError):
                    load_controls(self.root, self.config_path, self.s2000)
            finally:
                save(self.root / "result.json", self.index)

    def test_different_correction_winners_do_not_satisfy_solved(self):
        rows = configurations()
        for row in rows:
            if row["base_role"] == "replay" and (row["dataset"], row["correction"]) in {("UNSW-NB15", "wa"), ("CIC-IDS-2017", "cbf")}:
                row["closure"]["ratio_of_five_seed_means"] = .9
        self.assertEqual(decide(rows, True)["decision"], "intermediate")

    def test_unsolved_requires_both_strictly_below_thirty(self):
        self.assertEqual(decide(configurations(.299), True)["decision"], "unsolved")
        self.assertEqual(decide(configurations(.3), True)["decision"], "intermediate")
        self.assertEqual(decide(configurations(.9), False)["decision"], "pending")

    def test_nonfinite_and_confusion_corruption_fail(self):
        for value in (math.inf, math.nan, {"x": [-math.inf]}):
            with self.assertRaises(ValueError):
                finite(value)
        cp = copy.deepcopy(next(iter(self.formal.values()))["cell"]["checkpoints"][-1])
        cp["id"]["per_class"][0]["precision"] += .1
        with self.assertRaisesRegex(ValueError, "precision disagrees"):
            checkpoint_metrics(cp, CLASSES)
        cp["id"]["confusion_matrix"][0][0] = True
        with self.assertRaisesRegex(ValueError, "integer counts"):
            checkpoint_metrics(cp, CLASSES)

    def test_smoke_and_forged_completed_coverage_fail(self):
        for mutation in ("smoke", "missing", "duplicate"):
            index = copy.deepcopy(self.index)
            if mutation == "smoke":
                index["smoke"] = True
            elif mutation == "missing":
                index["cells"].pop()
                index["completed_cells"]["formal"] -= 1
            else:
                index["cells"].append(index["cells"][-1])
            try:
                save(self.root / "result.json", index)
                with self.assertRaises(ValueError):
                    load_controls(self.root, self.config_path, self.s2000)
            finally:
                save(self.root / "result.json", self.index)

    def test_selection_formal_seed_leakage_and_winner_tampering_fail(self):
        path = self.root / "selection.json"
        original = json.loads(path.read_text())
        for mode in ("seed", "winner"):
            changed = copy.deepcopy(original)
            if mode == "seed":
                changed["selections"][0]["candidates"][0]["development_results"][0]["seed"] = 1
            else:
                changed["selections"][1]["selected_candidate"] = 0
            index = copy.deepcopy(self.index)
            try:
                save(path, changed)
                index["selection_sha256"] = digest(path)
                save(self.root / "result.json", index)
                with self.assertRaises(ValueError):
                    load_controls(self.root, self.config_path, self.s2000)
            finally:
                save(path, original); save(self.root / "result.json", self.index)

    def test_result_hash_and_sidecar_tampering_fail(self):
        row = self.index["cells"][-1]
        result_path = self.root / row["result"]
        original = result_path.read_bytes()
        try:
            result_path.write_bytes(original + b" ")
            with self.assertRaisesRegex(ValueError, "result hash mismatch"):
                load_controls(self.root, self.config_path, self.s2000)
        finally:
            result_path.write_bytes(original)
        cell_ = json.loads(original)
        sidecar = result_path.parent / Path(cell_["checkpoints"][-1]["checkpoint_file"]).with_suffix(".json")
        original_sidecar = sidecar.read_bytes()
        try:
            value = json.loads(original_sidecar); value["correction_seconds"] += .1
            save(sidecar, value)
            with self.assertRaisesRegex(ValueError, "sidecar differs"):
                load_controls(self.root, self.config_path, self.s2000)
        finally:
            sidecar.write_bytes(original_sidecar)

    def test_missing_controller_is_explicit_zero_coverage_not_fake_results(self):
        formal, coverage, _ = load_controls(self.root / "not_created", self.config_path, self.s2000)
        self.assertEqual(formal, {})
        self.assertEqual(len(coverage["missing_formal_cells"]), 120)
        out = analyze(self.joint, self.s2000, formal, coverage)
        self.assertEqual(out["section23_4"]["decision"], "pending")
        self.assertTrue(all(r["final_macro_f1"]["mean"] is None for r in out["configurations"]))

    def test_json_csv_output_is_portable_partial_and_read_only_inputs(self):
        out = analyze(self.joint, self.s2000, {}, dict(self.coverage, formal_cells=0, controller_status="running"))
        before = digest(self.root / "result.json")
        with tempfile.TemporaryDirectory() as temp:
            export(out, Path(temp))
            self.assertEqual(json.loads((Path(temp) / "analysis.json").read_text()), out)
            self.assertEqual(len(list(Path(temp).iterdir())), 8)
            self.assertTrue((Path(temp) / "costs.csv").read_text().startswith("aggregation_level,"))
        self.assertEqual(digest(self.root / "result.json"), before)


if __name__ == "__main__":
    unittest.main()
