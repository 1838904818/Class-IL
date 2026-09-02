import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent / "runtime"))

from paired_checkpoint_selection_report import (
    EXPECTED_CALIBRATION_AUDIT_SHA256,
    METRICS,
    build_report,
)
from streaming_full.validation import _training_calibration_protocol


SEEDS = [1, 2, 3, 4, 42]


def _write_seed(root: Path, arm: str, seed: int, selection: str, offset: float) -> None:
    directory = root / arm / f"seed_{seed}"
    directory.mkdir(parents=True)
    protocol_hash = f"{seed:064x}"[-64:]
    config = {
        "family_checkpoint_selection": selection,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
    }
    if selection.endswith("recall_fpr_guard"):
        config.update(
            family_checkpoint_min_macro_f1_gain=0.01,
            family_checkpoint_max_positive_recall_drop=0.01,
            family_checkpoint_max_negative_fpr_increase=0.01,
        )
    calibration = (
        None
        if selection == "last"
        else SimpleNamespace(
            path=Path("/bound/training-calibration-audit.json"),
            audit_sha256=EXPECTED_CALIBRATION_AUDIT_SHA256,
            algorithm="deterministic_stratified_training_only_split",
            calibration_fraction=0.1,
            classes=[],
        )
    )
    protocol = {
        "manifest_sha256": "a" * 64,
        "dataset": "test",
        "feature_dim": 2,
        "tasks": [[0, 1]],
        "class_names": {str(index): f"class-{index}" for index in range(8)},
        "normal_class": {"id": 0, "name": "class-0"},
        "normalization_scope": "task0_only",
        "prediction_arms": ["joint_cap3000"],
        "seeds": [seed],
        "config": config,
        "training_calibration": _training_calibration_protocol(calibration),
        "protocol_sha256": protocol_hash,
    }
    metrics = {metric: 0.5 + offset for metric in METRICS}
    records = {}
    for class_id in range(8):
        records[str(class_id)] = {
            "checkpoint_selection": {
                "official_test_used": False,
                "future_classes_used": False,
                "selected_epoch": 3 if selection != "last" else 10,
                "reason": (
                    "last_epoch_control"
                    if selection == "last"
                    else "sufficient_training_only_calibration_support"
                ),
                "applied": selection != "last",
                "mode": selection,
                **(
                    {
                        "candidate_epoch": 3,
                        "guard_decision": {
                            "decision": "restore_candidate",
                            "passed": True,
                        },
                    }
                    if selection.endswith("recall_fpr_guard")
                    else {}
                ),
            }
        }
    deterministic_hash = "c" * 64
    result = {
        "seed": seed,
        "protocol_sha256": protocol_hash,
        "deterministic_result_sha256": deterministic_hash,
        "summary": {"views": {"official": {"joint_cap3000": metrics}}},
        "training_exposure_records": records,
    }
    summary = {
        "protocol_sha256": protocol_hash,
        "aggregate": {"seeds": [seed]},
        "deterministic_result_sha256": {str(seed): deterministic_hash},
    }
    (directory / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    (directory / f"result_seed_{seed}.json").write_text(
        json.dumps(result), encoding="utf-8"
    )
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


class PairedCheckpointSelectionReportTests(unittest.TestCase):
    def test_guarded_candidate_is_paired_with_last_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in SEEDS:
                _write_seed(root, "last_epoch", seed, "last", 0.0)
                _write_seed(
                    root,
                    "guarded_checkpoint",
                    seed,
                    "training_only_calibration_macro_f1_recall_fpr_guard",
                    0.05,
                )
            report = build_report(
                root,
                baseline_arm="last_epoch",
                candidate_arm="guarded_checkpoint",
                seeds=SEEDS,
            )
            self.assertEqual(
                report["campaign"],
                "replayids-d2-checkpoint-recall-guard-paired-five-seed-v1",
            )
            self.assertAlmostEqual(
                report["metrics"]["final_macro_f1"]["paired_delta_mean"], 0.05
            )

    def test_paired_report_uses_exact_seed_contract_and_delta(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in SEEDS:
                _write_seed(root, "last_epoch", seed, "last", 0.0)
                _write_seed(
                    root,
                    "training_only_calibration",
                    seed,
                    "training_only_calibration_macro_f1",
                    0.1,
                )
            report = build_report(
                root,
                baseline_arm="last_epoch",
                candidate_arm="training_only_calibration",
                seeds=SEEDS,
            )
            self.assertAlmostEqual(
                report["metrics"]["final_macro_f1"]["paired_delta_mean"], 0.1
            )
            self.assertEqual(
                report["selected_epoch_counts_by_class"]["0"], {"3": 5}
            )
            self.assertIsNone(
                report["metrics"]["final_macro_f1"]["cohen_dz"]
            )
            json.dumps(report, allow_nan=False)

    def test_unpaired_protocol_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in SEEDS:
                _write_seed(root, "last_epoch", seed, "last", 0.0)
                _write_seed(
                    root,
                    "training_only_calibration",
                    seed,
                    "training_only_calibration_macro_f1",
                    0.1,
                )
            path = root / "training_only_calibration" / "seed_42" / "protocol.json"
            protocol = json.loads(path.read_text(encoding="utf-8"))
            protocol["config"]["learning_rate"] = 0.002
            path.write_text(json.dumps(protocol), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differ beyond"):
                build_report(
                    root,
                    baseline_arm="last_epoch",
                    candidate_arm="training_only_calibration",
                    seeds=SEEDS,
                )

    def test_real_protocol_shapes_are_arm_specific(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in SEEDS:
                _write_seed(root, "last_epoch", seed, "last", 0.0)
                _write_seed(
                    root,
                    "training_only_calibration",
                    seed,
                    "training_only_calibration_macro_f1",
                    0.1,
                )
            last_path = root / "last_epoch" / "seed_1" / "protocol.json"
            last_protocol = json.loads(last_path.read_text(encoding="utf-8"))
            self.assertEqual(last_protocol["training_calibration"], {"enabled": False})
            last_protocol["training_calibration"] = {
                "enabled": True,
                "audit_sha256": EXPECTED_CALIBRATION_AUDIT_SHA256,
            }
            last_path.write_text(json.dumps(last_protocol), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpectedly received calibration"):
                build_report(
                    root,
                    baseline_arm="last_epoch",
                    candidate_arm="training_only_calibration",
                    seeds=SEEDS,
                )
            last_protocol["training_calibration"] = {"enabled": False}
            last_path.write_text(json.dumps(last_protocol), encoding="utf-8")
            candidate_path = (
                root
                / "training_only_calibration"
                / "seed_1"
                / "protocol.json"
            )
            candidate_protocol = json.loads(
                candidate_path.read_text(encoding="utf-8")
            )
            candidate_protocol["training_calibration"] = {"enabled": False}
            candidate_path.write_text(json.dumps(candidate_protocol), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training calibration is disabled"):
                build_report(
                    root,
                    baseline_arm="last_epoch",
                    candidate_arm="training_only_calibration",
                    seeds=SEEDS,
                )

    def test_all_reported_metrics_reject_out_of_range_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in SEEDS:
                _write_seed(root, "last_epoch", seed, "last", 0.0)
                _write_seed(
                    root,
                    "training_only_calibration",
                    seed,
                    "training_only_calibration_macro_f1",
                    0.1,
                )
            result_path = (
                root
                / "training_only_calibration"
                / "seed_1"
                / "result_seed_1.json"
            )
            original = json.loads(result_path.read_text(encoding="utf-8"))
            for metric in METRICS:
                with self.subTest(metric=metric, boundary="upper"):
                    mutated = copy.deepcopy(original)
                    mutated["summary"]["views"]["official"]["joint_cap3000"][
                        metric
                    ] = 1.0001
                    result_path.write_text(json.dumps(mutated), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "out-of-range"):
                        build_report(
                            root,
                            baseline_arm="last_epoch",
                            candidate_arm="training_only_calibration",
                            seeds=SEEDS,
                        )
            for metric in (value for value in METRICS if value != "average_forgetting"):
                with self.subTest(metric=metric, boundary="lower"):
                    mutated = copy.deepcopy(original)
                    mutated["summary"]["views"]["official"]["joint_cap3000"][
                        metric
                    ] = -0.0001
                    result_path.write_text(json.dumps(mutated), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "out-of-range"):
                        build_report(
                            root,
                            baseline_arm="last_epoch",
                            candidate_arm="training_only_calibration",
                            seeds=SEEDS,
                        )
            mutated = copy.deepcopy(original)
            mutated["summary"]["views"]["official"]["joint_cap3000"][
                "average_forgetting"
            ] = -1.0001
            result_path.write_text(json.dumps(mutated), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "out-of-range"):
                build_report(
                    root,
                    baseline_arm="last_epoch",
                    candidate_arm="training_only_calibration",
                    seeds=SEEDS,
                )
            mutated = copy.deepcopy(original)
            mutated["summary"]["views"]["official"]["joint_cap3000"][
                "average_forgetting"
            ] = -0.25
            result_path.write_text(json.dumps(mutated), encoding="utf-8")
            report = build_report(
                root,
                baseline_arm="last_epoch",
                candidate_arm="training_only_calibration",
                seeds=SEEDS,
            )
            self.assertEqual(
                report["records"]["candidate"][0]["metrics"]["average_forgetting"],
                -0.25,
            )


if __name__ == "__main__":
    unittest.main()
