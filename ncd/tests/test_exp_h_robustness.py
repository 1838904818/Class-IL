"""Numerical and sampling checks independent of the implementation helpers."""
from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

SCRIPT = Path(__file__).resolve().parents[1] / "analyze_exp_h_robustness.py"
SPEC = importlib.util.spec_from_file_location("exp_h_robustness", SCRIPT)
ROBUST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROBUST)


def average_ranks(values):
    """Pure-Python tie ranking; deliberately do not use scipy.rankdata."""
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for index in order[start:end]:
            out[index] = (start + 1 + end) / 2.0
        start = end
    return np.array(out)


def independent_rho(rows, categories):
    """Full-dummy pseudoinverse projection, independent of drop-first lstsq."""
    x = average_ranks([r["old_pair_pi_drift"] for r in rows])
    y = average_ranks([r["shift_recall_before"] - r["shift_recall_after"] for r in rows])
    z = average_ranks([r["id_recall_after"] - r["id_recall_before"] for r in rows])
    columns = [np.ones(len(rows)), z]
    for key in categories:
        levels = sorted({r[key] for r in rows})
        for level in levels:
            columns.append(np.array([float(r[key] == level) for r in rows]))
    design = np.array(columns).T
    projector = design @ np.linalg.pinv(design)
    rx, ry = x - projector @ x, y - projector @ y
    return float(np.corrcoef(rx, ry)[0, 1])


def fixture():
    rng = np.random.default_rng(46)
    rows = []
    for seed in ROBUST.SEEDS:
        for method in ("a", "b"):
            for class_id in (0, 1):
                for transition in (1, 2):
                    x, drop, dz = rng.integers(0, 12, size=3)
                    rows.append({
                        "seed": seed, "method": method, "class_id": class_id,
                        "class_name": str(class_id), "transition": transition,
                        "old_pair_pi_drift": float(x),
                        "shift_recall_before": 0.75,
                        "shift_recall_after": 0.75 - int(drop) / 32,
                        "id_recall_before": 0.5,
                        "id_recall_after": 0.5 + int(dz) / 32,
                    })
    return rows


class RobustnessTests(unittest.TestCase):
    def test_tie_ranks_and_independent_full_dummy_projection(self):
        self.assertEqual(average_ranks([3, 1, 1, 5]).tolist(), [3, 1.5, 1.5, 4])
        rows = fixture()
        for fields in ((), ("method",), ("class_id",), ("method", "class_id")):
            result = ROBUST.partial_rank(rows, fields)
            self.assertEqual(result["status"], "complete")
            self.assertAlmostEqual(result["rho"], independent_rho(rows, fields), places=12)

    def test_z_only_matches_analytic_partial_spearman(self):
        rows = fixture()
        x = [r["old_pair_pi_drift"] for r in rows]
        y = [r["shift_recall_before"] - r["shift_recall_after"] for r in rows]
        z = [r["id_recall_after"] - r["id_recall_before"] for r in rows]
        xy, xz, yz = spearmanr(x, y).statistic, spearmanr(x, z).statistic, spearmanr(y, z).statistic
        expected = (xy - xz * yz) / math.sqrt((1 - xz ** 2) * (1 - yz ** 2))
        self.assertAlmostEqual(ROBUST.partial_rank(rows)["rho"], expected, places=12)

    def test_dummy_reference_and_row_order_do_not_change_result(self):
        rows = fixture()
        original = ROBUST.partial_rank(rows, ROBUST.ADJUSTED)["rho"]
        renamed = [dict(r, method={"a": "z", "b": "a"}[r["method"]],
                        class_id={0: 20, 1: 10}[r["class_id"]]) for r in reversed(rows)]
        self.assertAlmostEqual(original, ROBUST.partial_rank(renamed, ROBUST.ADJUSTED)["rho"], places=12)

    def test_floor_is_before_recall_and_boundary_is_included(self):
        rows = fixture()[:3]
        rows[0]["shift_recall_before"] = 0.199999
        rows[1]["shift_recall_before"] = 0.2
        rows[2]["shift_recall_before"] = 0.200001
        rows[1]["shift_recall_after"] = 0.0
        self.assertEqual(ROBUST.eligible_rows(rows), rows[1:])
        for value in (float("nan"), float("inf"), None, True):
            with self.assertRaises(ValueError):
                ROBUST.eligible_rows([dict(rows[1], old_pair_pi_drift=value)])

    def test_constant_or_fully_explained_response_is_unidentified(self):
        rows = fixture()
        constant = [dict(r, old_pair_pi_drift=1.0) for r in rows]
        self.assertIsNone(ROBUST.partial_rank(constant, ROBUST.ADJUSTED)["rho"])
        explained = [dict(r, old_pair_pi_drift=float(r["method"] == "a")) for r in rows]
        self.assertEqual(ROBUST.partial_rank(explained, ROBUST.ADJUSTED)["status"], "unidentified")
        constant_z = [dict(r, id_recall_after=r["id_recall_before"]) for r in rows]
        self.assertEqual(ROBUST.partial_rank(constant_z, ROBUST.ADJUSTED)["status"], "complete")

    def test_seed_resampling_preserves_whole_unequal_clusters(self):
        rows = fixture()
        rows = [r for r in rows if r["seed"] != 42 or r["method"] == "a"]
        selected = [42, 42, 1, 2, 42]
        indices = ROBUST.cluster_indices(rows, selected).tolist()
        independently_expected = [i for seed in selected for i, row in enumerate(rows) if row["seed"] == seed]
        self.assertEqual(indices, independently_expected)
        self.assertEqual(len(indices), 3 * 4 + 2 * 8)
        self.assertEqual([i for i in indices if rows[i]["seed"] == 42],
                         [i for i, row in enumerate(rows) if row["seed"] == 42] * 3)

    def test_bootstrap_independent_recomputation_and_all_2000_draws(self):
        rows = fixture()
        with threadpool_limits(limits=1):
            variants, records = ROBUST.bootstrap_variants(rows)
            rng = np.random.default_rng(1705)
            independent = {"z_only": [], "method_class_adjusted": []}
            for record in records:
                selected = rng.choice(np.array(ROBUST.SEEDS), size=5, replace=True)
                sample = [row for seed in selected for row in rows if row["seed"] == seed]
                self.assertEqual(record["sampled_rows"], len(sample))
                self.assertEqual([record[f"seed_draw_{i+1}"] for i in range(5)], selected.tolist())
                for name, fields in (("z_only", ()), ("method_class_adjusted", ROBUST.ADJUSTED)):
                    rho = independent_rho(sample, fields)
                    independent[name].append(rho)
                    self.assertAlmostEqual(record[name + "_rho"], rho, places=11)
            for name in independent:
                self.assertEqual(variants[name]["bootstrap_valid"], 2000)
                np.testing.assert_allclose(variants[name]["ci95"],
                                           np.quantile(independent[name], [0.025, 0.975]),
                                           rtol=0, atol=1e-12)
        with self.assertRaises(ValueError):
            ROBUST.bootstrap_variants(rows, draws=1999)
        with self.assertRaises(ValueError):
            ROBUST.bootstrap_variants([r for r in rows if r["seed"] != 42])


if __name__ == "__main__":
    unittest.main()
