from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import l1_runner as r
import pretrain_task0 as p
import materialize_embeddings as exporter
from synthetic_smoke import make_inputs
import numpy as np
import torch


def tiny_cohort(root):
    root.mkdir()
    data = root / "transform_fit"
    data.mkdir()
    rows = 8
    raw = np.arange(rows * 4, dtype=np.float32).reshape(rows, 4) / 10
    raw[:, 3] = 3.0
    array_values = {"x": raw, "labels": np.array([0] * 4 + [1] * 4, np.int64),
                    "row_ids": np.array([r.object_hash(["train", i]) for i in range(rows)], "S64"),
                    "group_ids": np.array([r.object_hash(["group", i]) for i in range(rows)], "S64"),
                    "available_tasks": np.zeros(rows, np.int64),
                    "source_index": np.array([[0, i] for i in range(rows)], np.int64)}
    collection = {"rows": rows}
    for key, value in array_values.items():
        name = "transform_fit/" + key + ".npy"
        np.save(root / name, value, allow_pickle=False)
        collection[key] = {"path": name, "sha256": r.sha(root / name), "rows": rows,
                           "shape": list(value.shape), "dtype": value.dtype.str}
    manifest = {"kind": "derived_train_increment", "status": "COMPLETE", "increment": 0,
                "policy": {"arm": "P", "transform_fit_policy": "common_task0_prospective_fit"},
                "official_test_used_for_selection": False, "dataset_id": "tiny_task0", "feature_dim": 4,
                "feature_dtype": "float32", "classes": [{"class_id": 0}, {"class_id": 1}],
                "collections": {"transform_fit": collection, "calibration": {"path": "NEVER_OPEN.npy"}},
                "common_transform_fit_sha256": r.object_hash(collection)}
    r.atomic_json(root / "manifest.json", manifest)
    r.atomic_json(root / "COMPLETE.json", {"schema_version": 1, "status": "TRAIN_INCREMENT_COMPLETE",
                  "manifest_sha256": r.sha(root / "manifest.json"),
                  "files": {f.relative_to(root).as_posix(): r.sha(f) for f in root.rglob("*") if f.is_file()}})
    return r.sha(root / "manifest.json")


class Task0Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".test-task0-", dir=BASE)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cohort = self.root / "cohort"
        self.digest = tiny_cohort(self.cohort)
        config = r.read_json(BASE / "task0_pilot_config.json")
        config.update(epochs=2, batch_size=3, checkpoint_every_steps=4, architecture={"encoder_type": "mlp", "d_model": 4, "n_layers": 1})
        self.config = self.root / "config.json"
        r.atomic_json(self.config, config)

    def run_stage(self, name="pretrain", **kwargs):
        return p.run(self.cohort, self.digest, self.config, self.root / name, evidence_kind="synthetic", **kwargs)

    def test_actual_tiny_training_normalization_and_audit(self):
        self.assertEqual(self.run_stage()["status"], "COMPLETE")
        profile = r.read_json(self.root / "pretrain" / "PRETRAIN_PROFILE.json")
        self.assertEqual(profile["productive_optimizer_steps"], 6)
        self.assertEqual(profile["productive_raw_row_presentations"], 16)
        self.assertGreater(profile["observed_optimizer_peak_bytes"], 0)
        self.assertNotEqual(profile["initial_encoder_sha256"], profile["final_encoder_sha256"])
        with np.load(self.root / "pretrain" / "inference-state.npz", allow_pickle=False) as state:
            x = np.load(self.cohort / "transform_fit/x.npy", allow_pickle=False).astype(np.float64)
            np.testing.assert_allclose(state["normalization_mean"], x.mean(0), atol=1e-14)
            np.testing.assert_allclose(state["normalization_scale"][:3], x.std(0)[:3], atol=1e-14)
            self.assertEqual(state["normalization_scale"][3], 1.0)
        result = p.audit(self.root / "pretrain", self.cohort, self.digest)
        self.assertEqual(result["status"], "AUDIT_PASS")

    def test_pause_resume_matches_clean_state_and_cost(self):
        self.assertEqual(self.run_stage("resumed", max_steps=1)["status"], "PAUSED")
        self.assertFalse((self.root / "resumed" / "COMPLETE.json").exists())
        self.run_stage("resumed", resume=True)
        self.run_stage("clean")
        resumed = r.read_json(self.root / "resumed" / "PRETRAIN_PROFILE.json")
        clean = r.read_json(self.root / "clean" / "PRETRAIN_PROFILE.json")
        self.assertEqual(resumed["final_encoder_sha256"], clean["final_encoder_sha256"])
        self.assertEqual(resumed["final_classifier_sha256"], clean["final_classifier_sha256"])
        self.assertEqual(p.audit(self.root / "resumed", self.cohort, self.digest)["status"], "AUDIT_PASS")
        cost = r.read_json(self.root / "resumed" / "COST_LEDGER.json")
        self.assertEqual(cost["normalization_row_presentations"], 8)

    def test_failed_work_is_counted_and_exactly_resumable(self):
        with self.assertRaises(RuntimeError):
            self.run_stage("resumed", fault_after_steps=1)
        self.assertEqual(r.read_json(self.root / "resumed" / "COST_LEDGER.json")["total_uncommitted_completed_steps"], 1)
        self.run_stage("resumed", resume=True)
        self.run_stage("clean")
        a = r.read_json(self.root / "resumed" / "PRETRAIN_PROFILE.json")
        b = r.read_json(self.root / "clean" / "PRETRAIN_PROFILE.json")
        self.assertEqual(a["final_encoder_sha256"], b["final_encoder_sha256"])
        self.assertEqual(p.audit(self.root / "resumed", self.cohort, self.digest)["status"], "AUDIT_PASS")
        cost = r.read_json(self.root / "resumed" / "pretrain-cost.json")["measurements"]
        self.assertEqual(cost["optimizer_steps"], 7)
        self.assertEqual(cost["raw_row_presentations"], 19)
        self.assertFalse(cost["unknown_tail"])

    def test_no_held_out_arrays_are_opened(self):
        real_load = np.load
        seen = []
        def load(path, *args, **kwargs):
            seen.append(str(path))
            self.assertNotIn("NEVER_OPEN", str(path))
            self.assertNotIn("calibration", str(path))
            return real_load(path, *args, **kwargs)
        with patch.object(np, "load", side_effect=load):
            self.run_stage()
        self.assertTrue(seen)

    def test_changed_cohort_fails_before_output(self):
        with (self.cohort / "transform_fit/x.npy").open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaises(r.Invalid):
            self.run_stage()
        self.assertFalse((self.root / "pretrain").exists())

    def test_non_p_or_non_task0_or_test_selected_rejected(self):
        for mutation in (lambda m: m["policy"].update(arm="O"), lambda m: m.update(increment=1),
                         lambda m: m.update(official_test_used_for_selection=True)):
            source = r.read_json(self.cohort / "manifest.json")
            original = r.read_json(self.cohort / "manifest.json")
            mutation(source)
            r.atomic_json(self.cohort / "manifest.json", source)
            digest = r.sha(self.cohort / "manifest.json")
            marker = r.read_json(self.cohort / "COMPLETE.json")
            marker["manifest_sha256"] = marker["files"]["manifest.json"] = digest
            r.atomic_json(self.cohort / "COMPLETE.json", marker)
            with self.assertRaises(r.Invalid):
                p.load_cohort(self.cohort, digest)
            r.atomic_json(self.cohort / "manifest.json", original)

    def test_resume_config_drift_fails(self):
        self.run_stage(max_steps=1)
        config = r.read_json(self.config)
        config["learning_rate"] *= 2
        r.atomic_json(self.config, config)
        with self.assertRaisesRegex(r.Invalid, "drift"):
            self.run_stage(resume=True)

    def test_resume_tensor_precision_drift_fails(self):
        self.run_stage(max_steps=1)
        latest_path = self.root / "pretrain" / "LATEST.json"
        latest = r.read_json(latest_path)
        checkpoint = self.root / "pretrain" / latest["path"]
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        key = next(iter(state["model"]))
        state["model"][key] = state["model"][key].double()
        with checkpoint.open("wb") as stream:
            torch.save(state, stream)
        latest["sha256"] = r.sha(checkpoint)
        r.atomic_json(latest_path, latest)
        with self.assertRaisesRegex(r.Invalid, "dtype"):
            self.run_stage(resume=True)
        self.assertFalse((self.root / "pretrain" / "COMPLETE.json").exists())

    def test_unknown_failure_tail_is_explicit_in_cost(self):
        with self.assertRaises(RuntimeError):
            self.run_stage(fault_after_steps=1)
        journal = next((self.root / "pretrain").glob("attempt-*.jsonl"))
        records = [r.json.loads(line) for line in journal.read_bytes().splitlines()]
        records = [event for event in records if event["event"] != "attempt_failed"]
        r.atomic_bytes(journal, b"".join(r.json_bytes(e) + b"\n" for e in records))
        self.run_stage(resume=True)
        cost = r.read_json(self.root / "pretrain" / "pretrain-cost.json")["measurements"]
        self.assertTrue(cost["unknown_tail"])
        self.assertEqual(p.audit(self.root / "pretrain", self.cohort, self.digest)["status"], "AUDIT_PASS")

    def test_completed_output_and_active_writer_not_overwritten(self):
        self.run_stage(max_steps=1)
        lock = self.root / "pretrain" / "WRITER.lock"
        with lock.open("xb") as stream:
            stream.write(b"test-owner")
        with self.assertRaises(FileExistsError):
            self.run_stage(resume=True)
        self.assertEqual(lock.read_bytes(), b"test-owner")

    def test_cli_bounded_step(self):
        result = subprocess.run([sys.executable, "-B", str(BASE / "pretrain_task0.py"), str(self.cohort),
                "--manifest-sha256", self.digest, "--config", str(self.config), "--output", str(self.root / "cli"),
                "--evidence-kind", "synthetic", "--max-steps", "1"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PAUSED", result.stdout)


if __name__ == "__main__":
    unittest.main()
