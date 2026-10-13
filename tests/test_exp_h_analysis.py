"""Focused statistical invariants of the registered Section 17 analysis."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import rankdata

MODULE_PATH = Path(__file__).resolve().parents[1] / "ncd" / "analyze_exp_h.py"
SPEC = importlib.util.spec_from_file_location("exp_h_analysis", MODULE_PATH)
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)


def formal_payload():
    rows, retention = [], []
    for setting in ("S2", "S3"):
        for method_index, method in enumerate(h.METHODS):
            for seed_index, seed in enumerate(h.SEEDS):
                for transition, class_id in ((1, 0), (2, 0), (2, 1)):
                    drift = .05 + method_index * .07 + seed_index * .009 + transition * .02 + class_id * .01
                    rows.append({
                        "setting": setting, "method": method, "seed": seed,
                        "transition": transition, "class_id": class_id,
                        "old_pair_pi_drift": drift,
                        "self_margin_pi_drift": drift * .9 + .01 * np.sin(seed_index + method_index),
                        "shift_recall_before": .8,
                        "shift_recall_after": .8 - drift * .4 + .01 * np.cos(seed_index + transition),
                        "id_recall_before": .85,
                        "id_recall_after": .82 + .02 * np.sin(method_index + transition + seed_index),
                    })
                for class_id in (0, 1):
                    retention.append({
                        "setting": setting, "method": method, "seed": seed,
                        "class_id": class_id, "shift_recall_final": .8 - method_index * .04 - seed_index * .003,
                        "shift_recall_best": .85,
                    })
    return {"rows": rows, "retention_rows": retention, "settings": {
        name: {"status": "complete", "expected_units_per_cell": 3} for name in ("S2", "S3")}}


def constant_bootstrap(*args, **kwargs):
    return {"rho": .3, "ci95": [.2, .4], "row_count": len(args[0]), "status": "complete", "reason": None,
            "bootstrap_valid": 2000, "bootstrap_requested": 2000}


class ExpHStatisticsTests(unittest.TestCase):
    def test_partial_spearman_matches_closed_form_rank_formula(self):
        x = [1, 3, 3, 2, 5, 8, 4, 6, 9]
        y = [2, 4, 5, 4, 1, 3, 5, 8, 9]
        z = [7, 5, 3, 5, 8, 9, 2, 7, 4]
        ranks = np.asarray([rankdata(v) for v in (x, y, z)])
        c = np.corrcoef(ranks)
        expected = (c[0, 1] - c[0, 2] * c[1, 2]) / np.sqrt((1-c[0, 2]**2) * (1-c[1, 2]**2))
        self.assertAlmostEqual(h.partial_spearman(x, y, z), expected, places=12)

    def test_constant_control_equals_spearman(self):
        x, y = [1, 3, 2, 7, 6], [8, 7, 9, 4, 5]
        self.assertAlmostEqual(h.partial_spearman(x, y, [0]*5), h.spearman(x, y), places=12)

    def test_constant_residuals_nonfinite_and_small_samples_are_unidentified(self):
        self.assertIsNone(h.partial_spearman([1]*5, range(5), [0]*5))
        self.assertIsNone(h.partial_spearman(range(5), range(5), range(5)))
        self.assertIsNone(h.partial_spearman([1, 2, float("nan"), 3], [3, 2, 1, 4], [1, 1, 2, 2]))
        self.assertIsNone(h.partial_spearman([1, 2, 3], [3, 2, 1], [0, 0, 0]))

    def test_bootstrap_samples_whole_clusters_not_independent_rows(self):
        seeds = np.repeat(h.SEEDS, [2, 3, 4, 5, 6])
        rng = np.random.default_rng(971)
        for _ in range(50):
            indices = h.sample_cluster_indices(seeds, rng)
            multiplicities = []
            for seed in h.SEEDS:
                original_indices = np.flatnonzero(seeds == seed)
                counts = [int(np.sum(indices == i)) for i in original_indices]
                self.assertEqual(len(set(counts)), 1, "Every unit in a seed cluster must have identical draw multiplicity")
                multiplicities.append(counts[0])
            self.assertEqual(sum(multiplicities), 5)

    def test_real_bootstrap_reproducible_and_sign_is_decline(self):
        rows = [row for row in formal_payload()["rows"] if row["setting"] == "S3"]
        first = h.cluster_bootstrap(rows, n_bootstrap=2000, random_seed=23)
        second = h.cluster_bootstrap(rows, n_bootstrap=2000, random_seed=23)
        self.assertEqual(first, second)
        self.assertEqual(first["bootstrap_valid"], 2000)
        self.assertGreater(first["rho"], .8)
        self.assertGreater(first["ci95"][0], 0)
        with self.assertRaises(ValueError):
            h.cluster_bootstrap(rows, n_bootstrap=1999)

    @patch.object(h, "cluster_bootstrap", side_effect=constant_bootstrap)
    def test_floor_boundary_and_no_zero_imputation(self, _mock):
        payload = formal_payload()
        payload["rows"][0]["shift_recall_before"] = .199
        payload["rows"][1]["shift_recall_before"] = .2
        result = h.analyze(payload)
        sample = result["settings"]["S2"]["sample_size"]
        self.assertEqual(sample["floor_excluded_units"], 1)
        self.assertEqual(sample["included_units"], 119)
        payload["rows"][2]["old_pair_pi_drift"] = None
        bad = h.analyze(payload)
        self.assertEqual(bad["settings"]["S2"]["sample_size"]["invalid_numeric_units"], 1)
        self.assertEqual(bad["verdict"]["decision"], "pending")

    @patch.object(h, "cluster_bootstrap", side_effect=constant_bootstrap)
    def test_missing_duplicate_incomplete_and_smoke_stay_pending(self, _mock):
        for mutation in ("missing", "duplicate", "incomplete", "smoke"):
            payload = formal_payload()
            if mutation == "missing": payload["rows"].pop()
            elif mutation == "duplicate": payload["rows"].append(copy.deepcopy(payload["rows"][0]))
            elif mutation == "incomplete": payload["settings"]["S2"]["status"] = "running"
            elif mutation == "smoke": payload["settings"]["S2"]["smoke"] = True
            self.assertEqual(h.analyze(payload)["verdict"]["decision"], "pending", mutation)

    @patch.object(h, "cluster_bootstrap", side_effect=constant_bootstrap)
    def test_descriptive_missing_or_constant_does_not_add_primary_gate(self, _mock):
        payload = formal_payload()
        payload["retention_rows"].pop()
        result = h.analyze(payload)
        self.assertEqual(result["verdict"]["decision"], "harm_supported")
        payload["rows"][0]["self_margin_pi_drift"] = None
        result = h.analyze(payload)
        self.assertEqual(result["verdict"]["decision"], "harm_supported")
        self.assertEqual(result["report_status"], "partial")
        self.assertEqual(result["report_status"], "partial")
        for row in payload["retention_rows"]:
            row["shift_recall_final"] = .6
        result = h.analyze(payload)
        self.assertIsNone(result["settings"]["S2"]["method_level"]["rho_drift_vs_retention"][h.PRIMARY])
        self.assertEqual(result["verdict"]["decision"], "harm_supported")

    def test_decision_boundaries_and_secondary_not_used(self):
        def item(rho, lo, hi):
            return {"status": "complete", "unit_level": {h.PRIMARY: {"rho": rho, "ci95": [lo, hi]}, h.SECONDARY: {"rho": -1, "ci95": [-1, -1]}}}
        for decision, settings in (
            ("harm_supported", {"S2": item(.2, .01, .4), "S3": item(.4, .1, .6)}),
            ("harm_not_supported", {"S2": item(.099, -.1, .15), "S3": item(-.6, -.8, -.4)}),
            ("harm_not_supported", {"S2": item(.4, 0, .6), "S3": item(.3, -.1, .5)}),
            ("intermediate", {"S2": item(.1, .01, .2), "S3": item(.19, .1, .3)}),
            ("pending", {"S2": item(.3, .1, .5)}),
            ("pending", {"S2": item(float("nan"), .1, .5), "S3": item(.4, .1, .6)}),
        ):
            self.assertEqual(h.adjudicate(settings)["decision"], decision)

    @patch.object(h, "cluster_bootstrap", side_effect=constant_bootstrap)
    def test_runner_metadata_schema_and_scope_caveats(self, _mock):
        payload = formal_payload()
        payload["setting_metadata"] = payload.pop("settings")
        payload["settings"] = ["S2", "S3"]
        payload["git_commit"] = "source-for-test"
        payload["setting_metadata"]["S3"]["caveats"] = ["Shifted subset overlaps the ID test; not independent domain evidence"]
        payload["setting_metadata"]["S3"]["feature_alignment"] = [{"setting": "S3", "training_feature": "duration", "shift_feature": "duration", "action": "identity"}]
        result = h.analyze(payload)
        self.assertEqual(result["verdict"]["decision"], "harm_supported")
        self.assertEqual(result["provenance"]["git_commit"], "source-for-test")
        report = h.render_report(result)
        self.assertIn("overlaps the ID test", report)
        self.assertIn("duration", report)

    def test_cli_writes_partial_json_three_tables_and_scatter(self):
        payload = formal_payload()
        payload["rows"] = [row for row in payload["rows"] if row["seed"] == 1]
        payload["retention_rows"] = [row for row in payload["retention_rows"] if row["seed"] == 1]
        for metadata in payload["settings"].values():
            metadata["smoke"] = True
        with tempfile.TemporaryDirectory(prefix="exp_h_analysis_test_") as temporary:
            path = Path(temporary)
            input_path, output, report = path / "input.json", path / "analysis.json", path / "report.md"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            process = subprocess.run([
                sys.executable, str(MODULE_PATH), "--input", str(input_path),
                "--output-json", str(output), "--report", str(report),
            ], capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(process.stdout)["decision"], "pending")
            self.assertEqual(json.loads(output.read_text())["report_status"], "partial")
            markdown = report.read_text(encoding="utf-8")
            for number in (1, 2, 3):
                self.assertIn(f"## Table {number}:", markdown)
            self.assertIn("report_scatter.png", markdown)
            self.assertGreater((path / "report_scatter.png").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
