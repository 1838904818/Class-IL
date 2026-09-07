import copy
import importlib.util
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import l1_runner as r
from synthetic_smoke import make_inputs
import numpy as np
import torch
import materialize_embeddings as exporter


class L1Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".test-l1-", dir=BASE)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = make_inputs(self.root / "inputs")

    def write_manifest(self, edit):
        value = r.read_json(self.manifest)
        edit(value)
        r.atomic_json(self.manifest, value)

    def mutate_bound_array(self, split, key, edit):
        manifest = r.read_json(self.manifest)
        path = self.manifest.parent / manifest["splits"][split][key]["path"]
        array = np.load(path, allow_pickle=False)
        edit(array)
        np.save(path, array, allow_pickle=False)
        manifest["splits"][split][key]["sha256"] = r.sha(path)
        receipt_path = self.manifest.parent / manifest["encoder_binding"]["export_receipt"]["path"]
        receipt = r.read_json(receipt_path)
        receipt["split_array_sha256"][split][key] = r.sha(path)
        r.atomic_json(receipt_path, receipt)
        manifest["encoder_binding"]["export_receipt"]["sha256"] = r.sha(receipt_path)
        r.atomic_json(self.manifest, manifest)

    def test_actual_cpu_training_and_full_audit(self):
        with patch.object(r, "aggregate_wandb", side_effect=AssertionError("must stay offline")):
            result = r.run(self.manifest, self.root / "run")
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(r.audit(self.root / "run", self.manifest)["status"], "AUDIT_PASS")
        self.assertEqual(len(result["metrics"]["ce"]), 2)
        self.assertEqual(result["metrics"]["ce"][0]["mean_signed_forgetting"], None)
        self.assertEqual(len(result["metrics"]["ce"][-1]["signed_forgetting"]), 2)
        for unit in result["measurements"]:
            a, b = (unit["arms"][arm] for arm in r.ARMS)
            self.assertEqual(a["counters"], b["counters"])
            self.assertEqual(a["batch_stream_sha256"], b["batch_stream_sha256"])
            self.assertGreater(a["observed_optimizer_peak_bytes"], 0)
            self.assertGreater(a["process_lifetime_peak_rss_bytes"], 0)
            self.assertEqual(a["counters"]["raw_row_presentations"], a["counters"]["binary_targets"])
            self.assertEqual(a["loss_kind"], "binary_cross_entropy")
            self.assertEqual(b["loss_kind"], "conditional_focal")
        self.assertNotEqual(result["final_head_sha256"]["ce"], result["final_head_sha256"]["conditional_focal"])

    def test_mid_epoch_failure_counts_spent_work_and_resumes(self):
        with self.assertRaises(RuntimeError):
            r.run(self.manifest, self.root / "resumed", fault_after_steps=1)
        cost = r.read_json(self.root / "resumed" / "COST_LEDGER.json")
        self.assertEqual(cost["total_uncommitted_completed_steps"], 1)
        resumed = r.run(self.manifest, self.root / "resumed", resume=True)
        clean = r.run(self.manifest, self.root / "clean")
        self.assertEqual(resumed["final_head_sha256"], clean["final_head_sha256"])
        self.assertEqual(r.audit(self.root / "resumed", self.manifest)["status"], "AUDIT_PASS")
        cost = r.read_json(self.root / "resumed" / "COST_LEDGER.json")
        self.assertEqual(cost["total_uncommitted_completed_steps"], 1)

    def test_checkpoint_optimizer_resume_is_exact(self):
        with self.assertRaises(RuntimeError):
            r.run(self.manifest, self.root / "resumed", fault_after_unit=1)
        result = r.run(self.manifest, self.root / "resumed", resume=True)
        clean = r.run(self.manifest, self.root / "clean")
        self.assertEqual(result["final_head_sha256"], clean["final_head_sha256"])
        self.assertEqual(r.audit(self.root / "resumed", self.manifest)["status"], "AUDIT_PASS")

    def test_bounded_prefix_pause_and_resume(self):
        paused = r.run(self.manifest, self.root / "run", max_units=1)
        self.assertEqual(paused["status"], "PAUSED")
        self.assertFalse((self.root / "run" / "COMPLETE.json").exists())
        self.assertEqual(r.run(self.manifest, self.root / "run", resume=True)["status"], "COMPLETE")

    def test_cli_run_and_audit(self):
        run = subprocess.run([sys.executable, "-B", str(BASE / "l1_runner.py"), "run", str(self.manifest),
                              "--output", str(self.root / "cli"), "--device", "cpu"], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        audited = subprocess.run([sys.executable, "-B", str(BASE / "l1_runner.py"), "audit", str(self.root / "cli"),
                                  "--manifest", str(self.manifest)], capture_output=True, text=True)
        self.assertEqual(audited.returncode, 0, audited.stdout + audited.stderr)

    def test_existing_output_is_never_overwritten(self):
        (self.root / "existing").mkdir()
        with self.assertRaises(FileExistsError): r.run(self.manifest, self.root / "existing")

    def test_exclusive_writer_lock(self):
        r.run(self.manifest, self.root / "run", max_units=1)
        (self.root / "run" / "WRITER.lock").write_text("synthetic-active-writer")
        with self.assertRaises(FileExistsError): r.run(self.manifest, self.root / "run", resume=True)
        self.assertEqual((self.root / "run" / "WRITER.lock").read_text(), "synthetic-active-writer")

    def test_manifest_drift_cannot_resume(self):
        r.run(self.manifest, self.root / "run", max_units=1)
        self.write_manifest(lambda m: m["config"].__setitem__("learning_rate", 0.002))
        with self.assertRaises(r.Invalid): r.run(self.manifest, self.root / "run", resume=True)

    def test_tampered_checkpoint_rejected(self):
        r.run(self.manifest, self.root / "run", max_units=1)
        latest = r.read_json(self.root / "run" / "LATEST.json")
        with (self.root / "run" / latest["path"]).open("ab") as stream: stream.write(b"corrupt")
        with self.assertRaises(r.Invalid): r.run(self.manifest, self.root / "run", resume=True)

    def test_corrupt_array_fails_before_output(self):
        m = r.read_json(self.manifest)
        with (self.manifest.parent / m["splits"]["train"]["embeddings"]["path"]).open("ab") as stream: stream.write(b"corrupt")
        with self.assertRaises(r.Invalid): r.run(self.manifest, self.root / "run")
        self.assertFalse((self.root / "run").exists())

    def test_array_changed_during_training_cannot_complete(self):
        original = r.train_unit
        changed = False
        manifest = r.read_json(self.manifest)
        path = self.manifest.parent / manifest["splits"]["train"]["embeddings"]["path"]
        def mutate_after_training(*args, **kwargs):
            nonlocal changed
            result = original(*args, **kwargs)
            if not changed:
                writable = np.load(path, mmap_mode="r+", allow_pickle=False)
                writable[0, 0] += 9
                writable.flush()
                writable._mmap.close()
                changed = True
            return result
        with patch.object(r, "train_unit", new=mutate_after_training):
            with self.assertRaisesRegex(r.Invalid, "digest mismatch"):
                r.run(self.manifest, self.root / "run")
        self.assertFalse((self.root / "run" / "COMPLETE.json").exists())

    def test_resume_model_precision_is_not_coerced(self):
        config = r.read_json(self.manifest)["config"]
        model = r.FamilyHead(4, config["rank"], config["lora_alpha"])
        state = {"heads": {arm: {"0": {k: v.double() for k, v in model.state_dict().items()}} for arm in r.ARMS}}
        with self.assertRaisesRegex(r.Invalid, "dtype"):
            r.load_heads(state, 4, config, torch.device("cpu"))

    def test_optimizer_precision_and_missing_state_fail_closed(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        saved = copy.deepcopy(optimizer.state_dict())
        wanted = torch.optim.Adam(model.parameters(), lr=0.001)
        r.strict_adam_load(wanted, saved, has_updates=True)
        saved["state"][0]["exp_avg"] = saved["state"][0]["exp_avg"].double()
        with self.assertRaisesRegex(r.Invalid, "dtype"):
            r.strict_adam_load(wanted, saved, has_updates=True)
        saved = copy.deepcopy(optimizer.state_dict())
        saved["state"].pop(0)
        with self.assertRaisesRegex(r.Invalid, "missing"):
            r.strict_adam_load(wanted, saved, has_updates=True)

    def test_batch_size_is_an_actual_maximum(self):
        manifest, arrays, _ = r.load_inputs(self.manifest)
        config = copy.deepcopy(manifest["config"])
        config["batch_size"] = 1
        with self.assertRaisesRegex(r.Invalid, "batch size"):
            r.config_check(config)
        config["batch_size"] = 2
        r.config_check(config)
        batches = r.batches_for(arrays["train"], manifest["tasks"], {}, config, 0, 0, 0)
        self.assertTrue(all(len(indices) <= config["batch_size"] for indices, _, _ in batches))

    def test_unknown_config_and_test_selected_checkpoint_rejected(self):
        self.write_manifest(lambda m: m["config"].__setitem__("checkpoint_policy", "best_test"))
        with self.assertRaises(r.Invalid): r.load_inputs(self.manifest)

    def test_future_encoder_training_rejected(self):
        self.write_manifest(lambda m: m["encoder_binding"].__setitem__("trained_tasks", [0, 1]))
        with self.assertRaises(r.Invalid): r.load_inputs(self.manifest)

    def test_actual_encoder_tensor_hash_not_only_attestation(self):
        manifest = r.read_json(self.manifest)
        manifest["encoder_binding"]["state_sha256"] = "a" * 64
        r.atomic_json(self.manifest, manifest)
        with self.assertRaisesRegex(r.Invalid, "actual encoder tensor hash"): r.load_inputs(self.manifest)

    def test_training_row_cannot_move_to_future_availability(self):
        self.mutate_bound_array("train", "available_tasks", lambda a: a.__setitem__(0, 1))
        with self.assertRaisesRegex(r.Invalid, "availability"): r.load_inputs(self.manifest)

    def test_duplicate_original_training_row_rejected(self):
        self.mutate_bound_array("train", "row_ids", lambda a: a.__setitem__(1, a[0]))
        with self.assertRaisesRegex(r.Invalid, "duplicate original"): r.load_inputs(self.manifest)

    def test_capture_group_overlap_rejected(self):
        manifest = r.read_json(self.manifest)
        train_group = np.load(self.manifest.parent / manifest["splits"]["train"]["group_ids"]["path"], allow_pickle=False)[0]
        self.mutate_bound_array("test", "group_ids", lambda a: a.__setitem__(0, train_group))
        with self.assertRaisesRegex(r.Invalid, "group leakage"): r.load_inputs(self.manifest)

    def test_online_requires_dual_optin_and_governance(self):
        with self.assertRaises(r.Invalid): r.run(self.manifest, self.root / "run", wandb_project="research", allow_online=True)
        self.assertFalse((self.root / "run").exists())

    def test_focal_boundary_is_strict_less_than(self):
        config = r.read_json(self.manifest)["config"]
        logits = torch.tensor([[0.2, -0.1], [0.0, 0.8]], requires_grad=True)
        target = torch.tensor([0, 1])
        ce, _ = r.objective(logits, target, "ce", 5, config)
        same, kind = r.objective(logits, target, "conditional_focal", 5, config)
        self.assertTrue(torch.equal(ce, same))
        self.assertEqual(kind, "binary_cross_entropy")
        focal, kind = r.objective(logits, target, "conditional_focal", 4, config)
        self.assertEqual(kind, "conditional_focal")
        self.assertNotEqual(ce.item(), focal.item())

    def test_storage_views_are_counted_once(self):
        value = torch.arange(12, dtype=torch.float32)
        inv = r.storage_inventory({"params": [value, value[:3]], "duplicate_view": [value.reshape(3, 4)]})
        self.assertEqual(inv["total_bytes"], 48)
        self.assertEqual(inv["by_category_bytes"]["duplicate_view"], 0)

    def test_confusion_metrics_are_recomputed(self):
        result = r.confusion_metrics([[3, 1], [2, 4]], [0, 1])
        self.assertAlmostEqual(result["accuracy"], 0.7)
        self.assertAlmostEqual(result["balanced_accuracy"], (0.75 + 4 / 6) / 2)

    def test_canonical_nonfinite_and_duplicate_json_rejected(self):
        path = self.root / "bad.json"
        for bad in ('{"x": [1e400]}', '{"x": NaN}', '{"x":1,"x":2}'):
            path.write_text(bad)
            with self.assertRaises(r.Invalid): r.read_json(path)

    def test_negative_sampler_has_no_future_rows_and_equal_exact_stream(self):
        m, arrays, _ = r.load_inputs(self.manifest)
        buffers = r.buffers_for(arrays["train"], m["tasks"], m["config"])
        rows = list(r.batches_for(arrays["train"], m["tasks"], buffers, m["config"], 0, 0, 0))
        all_rows = np.concatenate([x[0] for x in rows])
        self.assertTrue((arrays["train"]["available_tasks"][all_rows] == 0).all())
        self.assertEqual(len(set(all_rows.tolist())), len(all_rows))

    def test_replay_storage_contains_arrived_classes_only(self):
        m, arrays, _ = r.load_inputs(self.manifest)
        self.assertEqual(r.buffers_for(arrays["train"], [], m["config"]), {})
        old = r.buffers_for(arrays["train"], m["tasks"][:1], m["config"])
        self.assertEqual(set(old), {0, 1})
        self.assertTrue(all((arrays["train"]["available_tasks"][idx] == 0).all() for idx in old.values()))

    def raw_export_spec(self):
        manifest = r.read_json(self.manifest)
        model = exporter.MLPEncoder(4, 4, 1)
        state = {name: tensor.detach().numpy() for name, tensor in model.state_dict().items()}
        schema = {name: "encoder_" + str(i) for i, name in enumerate(state)}
        values = {schema[name]: value for name, value in state.items()}
        values.update(normalization_mean=np.zeros(4, np.float64), normalization_scale=np.ones(4, np.float64))
        np.savez(self.manifest.parent / "task0.npz", **values)
        ref = lambda name: {"path": name, "sha256": r.sha(self.manifest.parent / name)}
        metadata = {"dataset": manifest["dataset"], "checkpoint": 0, "seen_classes": [0, 1], "feature_dim": 4,
                    "architecture": {"encoder_type": "mlp", "d_model": 4, "n_layers": 1},
                    "inference_state_sha256": ref("task0.npz")["sha256"],
                    "state_schema": {"encoder": schema, "normalization": {"mean": "normalization_mean", "scale": "normalization_scale"}}}
        r.atomic_json(self.manifest.parent / "task0.json", metadata)
        raw = {k: manifest[k] for k in ("schema_version", "dataset", "evidence_kind", "tasks", "group_mode", "config", "allow_aggregate_wandb")}
        raw.update(checkpoint_metadata=ref("task0.json"), checkpoint_state=ref("task0.npz"), pretrain_cost_receipt=None,
                   raw_splits={s: {("raw_features" if k == "embeddings" else k): descriptor for k, descriptor in fields.items()}
                               for s, fields in manifest["splits"].items()})
        r.atomic_json(self.manifest.parent / "raw-export.json", raw)
        return self.manifest.parent / "raw-export.json"

    def test_real_export_core_tiny_mlp_then_actual_training(self):
        source = self.raw_export_spec()
        exported = exporter.export(source, self.root / "export", batch_size=3)
        profile = r.read_json(self.root / "export" / "EXPORT_PROFILE.json")
        self.assertEqual(profile["rows_forwarded"], 18)
        self.assertEqual(profile["encoder_state_sha256_before"], profile["encoder_state_sha256_after"])
        self.assertEqual(r.load_inputs(exported)[0]["evidence_kind"], "synthetic")
        self.assertEqual(r.run(exported, self.root / "exported-run")["status"], "COMPLETE")
        self.assertEqual(r.audit(self.root / "exported-run", exported)["status"], "AUDIT_PASS")

    def test_non_task0_export_checkpoint_rejected(self):
        source = self.raw_export_spec()
        metadata = r.read_json(self.manifest.parent / "task0.json")
        metadata["checkpoint"] = 1
        r.atomic_json(self.manifest.parent / "task0.json", metadata)
        raw = r.read_json(source)
        raw["checkpoint_metadata"]["sha256"] = r.sha(self.manifest.parent / "task0.json")
        r.atomic_json(source, raw)
        with self.assertRaises(r.Invalid): exporter.export(source, self.root / "export", batch_size=3)
        self.assertFalse((self.root / "export").exists())

    def test_string_normalization_is_not_coerced_to_numeric(self):
        source = self.raw_export_spec()
        state_path = source.parent / "task0.npz"
        with np.load(state_path, allow_pickle=False) as archive:
            state = {k: np.array(archive[k], copy=True) for k in archive.files}
        state.update(normalization_mean=np.array(["0"] * 4), normalization_scale=np.array(["1"] * 4))
        np.savez(state_path, **state)
        metadata_path = source.parent / "task0.json"
        metadata = r.read_json(metadata_path)
        metadata["inference_state_sha256"] = r.sha(state_path)
        r.atomic_json(metadata_path, metadata)
        raw = r.read_json(source)
        raw["checkpoint_state"]["sha256"] = r.sha(state_path)
        raw["checkpoint_metadata"]["sha256"] = r.sha(metadata_path)
        r.atomic_json(source, raw)
        with self.assertRaisesRegex(r.Invalid, "native float64"):
            exporter.export(source, self.root / "export", batch_size=3)
        self.assertFalse((self.root / "export").exists())

    def producer_input(self):
        source = self.raw_export_spec()
        pair = self.root / "pair"
        arm = pair / "P"
        shutil.copytree(source.parent, arm)
        (arm / source.name).rename(arm / "export-input.json")
        r.atomic_json(arm / "STARTED.json", {"stage": "offline_fit_only_export_input"})
        r.atomic_json(arm / "LINEAGE.json", {"evidence_kind": "synthetic"})
        r.atomic_json(pair / "STARTED.json", {"stage": "offline_O_P_export_pair"})
        files = {x.relative_to(arm).as_posix(): r.sha(x) for x in arm.rglob("*") if x.is_file()}
        r.atomic_json(arm / "COMPLETE.json", {"status": "RAW_EXPORT_INPUT_COMPLETE", "evidence_kind": "synthetic",
                      "manifest_sha256": files["export-input.json"], "files": files})
        r.atomic_json(pair / "COMPLETE.json", {"status": "RAW_EXPORT_PAIR_COMPLETE", "evidence_kind": "synthetic",
                      "P_export_manifest_sha256": files["export-input.json"]})
        return arm / "export-input.json"

    def test_marked_producer_complete_input_exports(self):
        source = self.producer_input()
        exported = exporter.export(source, self.root / "export", batch_size=3)
        self.assertEqual(r.load_inputs(exported)[0]["evidence_kind"], "synthetic")

    def test_marked_producer_incomplete_pair_fails_before_inference(self):
        source = self.producer_input()
        (source.parent.parent / "COMPLETE.json").unlink()
        with self.assertRaises((r.Invalid, OSError)):
            exporter.export(source, self.root / "export", batch_size=3)
        self.assertFalse((self.root / "export").exists())

    def test_marked_producer_extra_file_is_not_accepted(self):
        source = self.producer_input()
        r.atomic_json(source.parent / "uncommitted.json", {})
        with self.assertRaisesRegex(r.Invalid, "closure"):
            exporter.export(source, self.root / "export", batch_size=3)
        self.assertFalse((self.root / "export").exists())

    def test_wandb_adapter_uses_only_aggregates_and_reports_unverified_cloud(self):
        from types import SimpleNamespace
        logged = []
        class Table:
            def __init__(self, columns): self.columns, self.rows = columns, []
            def add_data(self, *values): self.rows.append(values)
        fake_run = SimpleNamespace(id="synthetic-id", url="https://example.invalid/synthetic", log=logged.append, finish=lambda: None)
        fake = SimpleNamespace(Settings=lambda **kw: kw, Table=Table, init=lambda **kw: fake_run)
        metric = r.confusion_metrics([[2, 0], [1, 1]], [0, 1])
        metric.update(task=0, mean_signed_forgetting=None)
        result = {"binding": {"synthetic": True}, "evidence_kind": "synthetic", "metrics": {arm: [metric] for arm in r.ARMS}}
        with patch.dict(sys.modules, {"wandb": fake}):
            receipt = r.aggregate_wandb(result, "synthetic_test", True)
        self.assertEqual(receipt["status"], "client_finished_remote_verification_not_performed")
        self.assertEqual(len(logged), 2)
        self.assertTrue(any("per_class" in k for k in logged[0]))
        self.assertFalse(any("row_ids" in k or "manifest" in k for payload in logged for k in payload))


if __name__ == "__main__":
    unittest.main()
