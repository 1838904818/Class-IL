"""Tests for the fail-closed Section 14 gate adjudication."""

from __future__ import annotations

import unittest

from ncd.run_crr_section14_gates import _gate_verdict


def necessity_row(
    dataset: str,
    seed: int,
    method: str,
    k: float,
    c_recall: float,
    d_recall: float,
    dependency_change: float,
) -> dict:
    return {
        "dataset": dataset,
        "seed": seed,
        "method": method,
        "benchmark": "necessity",
        "k": k,
        "c_recall": c_recall,
        "d_recall": d_recall,
        "c_f2_dependency_change": dependency_change,
    }


def hijack_row(dataset: str, seed: int, step: int, rho: float, drop: float) -> dict:
    return {
        "dataset": dataset,
        "seed": seed,
        "method": "replay",
        "benchmark": "hijack",
        "rho": rho,
        "hijack_construction_step": step,
        "marker_recall_drop": drop,
        "marker_decorrelation_recall_drop": drop,
    }


def passing_rows() -> list[dict]:
    rows: list[dict] = []
    for dataset in ("NSL-KDD", "UNSW-NB15"):
        for seed in (1, 2):
            for k in (1.0, 2.0, 3.0):
                rows.append(
                    necessity_row(dataset, seed, "oracle", k, 0.8, 0.8, 0.0)
                )
                rows.append(
                    necessity_row(dataset, seed, "replay", k, 0.6, 0.6, 0.1)
                )
            for step in (1, 2, 3):
                drop = 0.2 if step >= 2 else 0.1
                for rho in (0.9, 1.0):
                    rows.append(hijack_row(dataset, seed, step, rho, drop))
    return rows


class Section14GateTest(unittest.TestCase):
    def test_selects_earliest_passing_hijack_step(self) -> None:
        verdict = _gate_verdict(passing_rows())
        self.assertTrue(verdict["passed"])
        self.assertEqual(verdict["selected_hijack_construction_step"], 2)
        self.assertEqual(verdict["decision"], "SUBMIT_FIVE_SEED_MATRIX")

    def test_fails_closed_on_any_replay_necessity_cell(self) -> None:
        rows = passing_rows()
        replay = next(
            row
            for row in rows
            if row["benchmark"] == "necessity" and row["method"] == "replay"
            and float(row["k"]) >= 2.0
        )
        replay["c_recall"] = 0.49
        verdict = _gate_verdict(rows)
        self.assertFalse(verdict["replay_necessity_passed"])
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["decision"], "STOP_AND_REPAIR_BENCHMARK")

    def test_fails_closed_on_nonpositive_dependency_change(self) -> None:
        rows = passing_rows()
        replay = next(
            row
            for row in rows
            if row["benchmark"] == "necessity" and row["method"] == "replay"
            and float(row["k"]) >= 2.0
        )
        replay["c_f2_dependency_change"] = 0.0
        verdict = _gate_verdict(rows)
        self.assertFalse(verdict["replay_necessity_passed"])
        self.assertFalse(verdict["passed"])

    def test_fails_closed_on_missing_cell(self) -> None:
        verdict = _gate_verdict(passing_rows()[:-1])
        self.assertFalse(verdict["complete"])
        self.assertFalse(verdict["passed"])


if __name__ == "__main__":
    unittest.main()
