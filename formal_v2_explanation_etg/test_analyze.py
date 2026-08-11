from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import analyze

from analyze import (
    _json,
    _load_npz,
    _persist_resume_state,
    _reconcile_resume_state,
    audit_checkpoint_reconstruction,
    build_etg_ledger,
    jaccard,
    prediction_tie_audit,
    sensitivity_rows,
    validate_output,
)
from streaming_full.data import canonical_sha256, sha256_file


class FormalV2UnitTests(unittest.TestCase):
    def test_completed_analysis_validator_imports_from_submission_cwd(self):
        operation_dir = Path(__file__).resolve().parent
        project_dir_value = os.environ.get("OFRA_PROJECT_DIR")
        self.assertTrue(
            project_dir_value,
            "OFRA_PROJECT_DIR must identify the hash-bound project root",
        )
        project_dir = Path(project_dir_value).resolve()
        self.assertTrue((project_dir / "streaming_full" / "__init__.py").is_file())
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(operation_dir), str(project_dir)]
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "from analyze import validate_output; "
                "assert callable(validate_output)",
            ],
            cwd=project_dir,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )

    def test_jaccard(self):
        self.assertEqual(jaccard([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertEqual(jaccard([1, 2], [2, 3]), 1 / 3)

    def test_silent_drift_uses_strict_boundaries_and_full_grid(self):
        transition = {
            "delta_recall": -0.05,
            "jaccard_top5": 0.4,
            "jaccard_top10": 0.4,
            "jaccard_top15": 0.6,
            "jaccard_top20": 0.4,
        }
        grid = sensitivity_rows([transition])
        self.assertEqual(len(grid), 64)
        primary = next(row for row in grid if row["is_primary"])
        self.assertEqual(primary["eligible_transitions"], 0)
        transition["delta_recall"] = -0.049999
        primary = next(row for row in sensitivity_rows([transition]) if row["is_primary"])
        self.assertEqual((primary["events"], primary["eligible_transitions"]), (1, 1))

    def test_strict_etg_recertification_has_one_checkpoint_lag(self):
        rows = [
            {"checkpoint": 0, "class_id": 0, "class_name": "A", "admitted": True, "top15_indices": list(range(15)), "rationale_mass": .3, "random_null_95": .1, "mass_margin": .2},
            {"checkpoint": 1, "class_id": 0, "class_name": "A", "admitted": True, "top15_indices": list(range(5, 20)), "rationale_mass": .3, "random_null_95": .1, "mass_margin": .2},
            {"checkpoint": 2, "class_id": 0, "class_name": "A", "admitted": True, "top15_indices": list(range(15)), "rationale_mass": .3, "random_null_95": .1, "mass_margin": .2},
        ]
        transitions = [
            {"to_checkpoint": 1, "class_id": 0, "primary_event": True},
            {"to_checkpoint": 2, "class_id": 0, "primary_event": False},
        ]
        ledger = build_etg_ledger(rows, transitions)
        self.assertEqual([row["action"] for row in ledger], ["admission_certified", "human_review_escalation", "strict_recertified"])

    def test_manifest_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = {"analysis_protocol_sha256": ""}
            protocol["analysis_protocol_sha256"] = canonical_sha256({})
            (root / "analysis_protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
            analysis = {"analysis_protocol_sha256": protocol["analysis_protocol_sha256"]}
            analysis["canonical_sha256"] = canonical_sha256(analysis)
            (root / "analysis.json").write_text(json.dumps(analysis), encoding="utf-8")
            files = {
                name: {"sha256": sha256_file(root / name), "bytes": (root / name).stat().st_size}
                for name in ("analysis_protocol.json", "analysis.json")
            }
            manifest = {"files": files}
            manifest["canonical_sha256"] = canonical_sha256(manifest)
            (root / "analysis_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            validate_output(root)
            (root / "analysis.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "integrity"):
                validate_output(root)

    def test_cross_device_prediction_near_tie_is_audited(self):
        axis = np.asarray([0, 1], dtype=np.int64)
        saved_joint = np.asarray([[0.5004, 0.5000]], dtype=np.float32)
        reconstructed_joint = np.asarray([[0.5000, 0.5004]], dtype=np.float32)
        saved = {
            "class_axis": axis,
            "head_scores": saved_joint.copy(),
            "router_z_scores": np.zeros_like(saved_joint),
            "joint_scores": saved_joint,
            "predicted_class_id": np.asarray([0], dtype=np.int64),
        }
        reconstructed = {
            "class_axis": axis.copy(),
            "head_scores": saved["head_scores"].copy(),
            "router_z_scores": saved["router_z_scores"].copy(),
            "joint_scores": reconstructed_joint,
            "predicted_class_id": np.asarray([1], dtype=np.int64),
        }
        audit = prediction_tie_audit(
            saved, reconstructed, joint_tolerance=1e-3
        )
        self.assertFalse(audit["exact_match"])
        self.assertEqual(audit["tie_compatible_mismatch_count"], 1)
        rows = audit_checkpoint_reconstruction(4, saved, reconstructed)
        prediction = next(row for row in rows if row["array"] == "predicted_class_id")
        self.assertEqual(prediction["mismatch_count"], 1)

    def test_material_prediction_change_still_fails_closed(self):
        axis = np.asarray([0, 1], dtype=np.int64)
        saved = {
            "class_axis": axis,
            "joint_scores": np.asarray([[0.9, 0.1]], dtype=np.float32),
            "predicted_class_id": np.asarray([0], dtype=np.int64),
        }
        reconstructed = {
            "class_axis": axis.copy(),
            "joint_scores": np.asarray([[0.1, 0.9]], dtype=np.float32),
            "predicted_class_id": np.asarray([1], dtype=np.int64),
        }
        with self.assertRaisesRegex(RuntimeError, "not explained"):
            prediction_tie_audit(saved, reconstructed, joint_tolerance=1e-3)

    def test_resume_state_commits_progress_last_and_round_trips(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            progress = {
                "completed_class_checkpoint_rows": [
                    {"checkpoint": 0, "class_id": 0}
                ],
                "active": None,
            }
            rows = [{"checkpoint": 0, "class_id": 0, "admitted": True}]
            _persist_resume_state(
                root,
                attribution_arrays={"a": np.asarray([1.0, 2.0])},
                control_arrays={"b": np.asarray([[3, 4]], dtype=np.int64)},
                checkpoint_rows=rows,
                progress=progress,
            )
            np.testing.assert_array_equal(
                _load_npz(root / "resume_attributions.npz")["a"], [1.0, 2.0]
            )
            np.testing.assert_array_equal(
                _load_npz(root / "resume_etg_controls_and_masses.npz")["b"],
                [[3, 4]],
            )
            self.assertEqual(
                json.loads(
                    (root / "resume_checkpoint_rows.json").read_text(
                        encoding="utf-8"
                    )
                ),
                rows,
            )
            self.assertEqual(_json(root / "progress.json"), progress)

    def test_resume_recovers_last_commit_after_each_cross_file_interruption(self):
        old_progress = {
            "completed_class_checkpoint_rows": [
                {"checkpoint": 0, "class_id": 0}
            ],
            "active": None,
        }
        new_progress = {
            "completed_class_checkpoint_rows": [
                {"checkpoint": 0, "class_id": 0},
                {"checkpoint": 1, "class_id": 0},
            ],
            "active": None,
        }

        def state(pairs):
            attributions = {}
            controls = {"class_000_feature_sets": np.asarray([[0, 1]])}
            rows = []
            for checkpoint, class_id in pairs:
                prefix = f"checkpoint_{checkpoint:03d}_class_{class_id:03d}"
                attributions[f"{prefix}_mean_abs"] = np.asarray([1.0, 2.0])
                attributions[f"{prefix}_mean_signed"] = np.asarray([0.5, -0.5])
                attributions[f"{prefix}_sample_id_sha256"] = np.asarray([b"a" * 64])
                controls[f"{prefix}_random_masses"] = np.asarray([0.1, 0.2])
                rows.append(
                    {"checkpoint": checkpoint, "class_id": class_id, "admitted": True}
                )
            return attributions, controls, rows

        old_state = state([(0, 0)])
        new_state = state([(0, 0), (1, 0)])
        for interrupt_after_call in (1, 2, 3):
            with self.subTest(interrupt_after_call=interrupt_after_call):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    _persist_resume_state(
                        root,
                        attribution_arrays=old_state[0],
                        control_arrays=old_state[1],
                        checkpoint_rows=old_state[2],
                        progress=old_progress,
                    )
                    real_npz = analyze._deterministic_npz
                    real_json = analyze._atomic_json
                    calls = 0

                    def interrupted_npz(path, arrays):
                        nonlocal calls
                        real_npz(path, arrays)
                        calls += 1
                        if calls == interrupt_after_call:
                            raise RuntimeError("simulated interruption")

                    def interrupted_json(path, value):
                        nonlocal calls
                        real_json(path, value)
                        calls += 1
                        if calls == interrupt_after_call:
                            raise RuntimeError("simulated interruption")

                    with mock.patch.object(
                        analyze, "_deterministic_npz", side_effect=interrupted_npz
                    ), mock.patch.object(
                        analyze, "_atomic_json", side_effect=interrupted_json
                    ):
                        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                            _persist_resume_state(
                                root,
                                attribution_arrays=new_state[0],
                                control_arrays=new_state[1],
                                checkpoint_rows=new_state[2],
                                progress=new_progress,
                            )

                    recovered = _reconcile_resume_state(
                        progress=_json(root / "progress.json"),
                        attribution_arrays=_load_npz(root / "resume_attributions.npz"),
                        control_arrays=_load_npz(
                            root / "resume_etg_controls_and_masses.npz"
                        ),
                        checkpoint_rows=json.loads(
                            (root / "resume_checkpoint_rows.json").read_text(
                                encoding="utf-8"
                            )
                        ),
                    )
                    self.assertEqual(
                        {(row["checkpoint"], row["class_id"]) for row in recovered[2]},
                        {(0, 0)},
                    )
                    self.assertFalse(
                        any("checkpoint_001" in name for name in recovered[0])
                    )


if __name__ == "__main__":
    unittest.main()
