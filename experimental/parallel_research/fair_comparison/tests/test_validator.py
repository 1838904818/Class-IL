"""Small generated synthetic evidence only; never a scientific result."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

BASE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fair_accounting_validator", BASE / "validator.py")
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


def h(value):
    return hashlib.sha256(value.encode()).hexdigest()


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".synthetic-", dir=BASE)
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.n = 0
        self.rows = [h("synthetic-row-0"), h("synthetic-row-1")]
        phase = lambda name, grad: {
            "id": name, "task": 0, "gradient": grad,
            "target_kind": "multiclass" if grad else "none", "device": "cpu",
            "na_reason": None if grad else "frozen nearest-mean fitting has no optimizer",
            "limits": {k: None if k.startswith("cuda_") else 10000 for k in v.METRICS},
        }
        self.p = {
            "schema_version": 1, "status": "BOUND", "evidence_kind": "synthetic",
            "dataset": "synthetic-two-class", "locks": {}, "seeds": [7],
            "tasks": [[0, 1]], "available_labels": [[0, 1]],
            "shared_initial_encoder_sha256": {"7": h("synthetic-init")},
            "shared_task0_encoder_sha256": {"7": h("synthetic-task0")},
            "checkpoint_policy": "fixed_last_registered_update_no_test_selection",
            "arms": [{"id": a, "status": "bound", "phase_plan": [phase("shared_pretrain", True), phase("fit_0", False)]}
                     for a in ("reference", "candidate")],
            "contrasts": [{"arms": ["reference", "candidate"], "estimand": "synthetic exact equality only",
                           "residual_confounds": [], "match_metrics": ["raw_row_presentations", "optimizer_steps"]}],
            "real_measurements": "SYNTHETIC_ONLY", "submission_authorized": False,
        }
        registry = {"rows": [{"id": r, "class_id": i, "partition": "train", "available_from_task": 0}
                             for i, r in enumerate(self.rows)]}
        self.registry = v.registry_check(registry, self.p)
        specs = {a["id"]: {"loss_policy": "synthetic multiclass CE", "target_assignment": "one true label per row",
                            "sampling_policy": "synthetic ordered rows", "trainable_parameters": "synthetic pretrain only",
                            "optimizer_schedule": "synthetic one successful step", "checkpoint_policy": self.p["checkpoint_policy"],
                            "initial_head_state_sha256": {"7": h("head-" + a["id"])}}
                 for a in self.p["arms"]}
        bindings = {}
        for lock in v.LOCKS:
            value = {"synthetic_only": True, "binding": lock}
            if lock == "split_registry_sha256":
                value = registry
            elif lock == "task_order_sha256":
                value = self.p["tasks"]
            elif lock == "available_labels_sha256":
                value = self.p["available_labels"]
            elif lock == "method_specs_sha256":
                value = specs
            ref = self.put(value)
            bindings[lock] = ref
            self.p["locks"][lock] = ref["sha256"]
        self.report = {"schema_version": 1, "evidence_kind": "synthetic",
                       "protocol_sha256": v.canonical_hash(self.p), "binding_artifacts": bindings, "records": []}
        for arm in self.p["arms"]:
            for phase in arm["phase_plan"]:
                grad = phase["gradient"]
                trace = {"events": [{"kind": "microbatch" if grad else "fit_read", "rows": self.rows,
                                     "binary_targets": [], "multiclass_targets": [[0, 0], [1, 1]] if grad else []}]}
                if grad:
                    trace["events"].append({"kind": "step"})
                memory = self.memory(grad)
                counts, _ = v.trace_check(trace, self.registry, self.p, phase)
                self.report["records"].append({
                    "arm": arm["id"], "seed": 7, "phase": phase["id"],
                    "initial_encoder_sha256": self.p["shared_initial_encoder_sha256"]["7"],
                    "task0_encoder_sha256": self.p["shared_task0_encoder_sha256"]["7"],
                    "initial_head_state_sha256": specs[arm["id"]]["initial_head_state_sha256"]["7"],
                    "trace": self.put(trace), "memory": self.put(memory),
                    "counters": counts, "memory_metrics": v.memory_check(memory, phase),
                    "measurement_status": "synthetic",
                })

    def put(self, value):
        self.n += 1
        data = json.dumps(value, sort_keys=True, allow_nan=False).encode()
        name = f"synthetic-{self.n}.json"
        (self.root / name).write_bytes(data)
        return {"path": name, "sha256": hashlib.sha256(data).hexdigest()}

    def memory(self, gradient):
        return {"scope": "phase_boundary_live_storage_and_process_tree_phase_peaks", "snapshot_id": "synthetic-boundary",
                "inventory_method": "synthetic integer storage generator", "peak_method": "synthetic not hardware measurement",
                "components": [{"storage_id": c, "category": c, "numel": 2 if c != "optimizer_state" or gradient else 0,
                                "itemsize": 4, "bytes": 8 if c != "optimizer_state" or gradient else 0} for c in sorted(v.CATEGORIES)],
                "optimizer_peak_bytes": 8 if gradient else 0, "host_peak_rss_bytes": 100,
                "cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None, "device_na_reason": "cpu_only_phase"}

    def validate(self):
        return v.validate(self.p, self.report, self.root)

    def rebind(self):
        self.report["protocol_sha256"] = v.canonical_hash(self.p)

    def change_artifact(self, record, key, edit):
        value = v.artifact(self.root, record[key])
        edit(value)
        record[key] = self.put(value)

    def test_complete_synthetic_evidence_passes_only_as_synthetic(self):
        self.assertEqual(self.validate(), {"status": "ACCOUNTING_CONSISTENT", "synthetic_only": True,
                                         "records_checked": 4, "submission_authorized": False})

    def test_positive_cli_is_accounting_only(self):
        protocol_ref = self.put(self.p)
        report_ref = self.put(self.report)
        result = subprocess.run([sys.executable, "-B", str(BASE / "validator.py"),
                                 str(self.root / protocol_ref["path"]), str(self.root / report_ref["path"]),
                                 "--artifact-root", str(self.root)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(json.loads(result.stdout)["synthetic_only"])

    def test_shipped_real_template_is_blocked(self):
        result = subprocess.run([sys.executable, "-B", str(BASE / "validator.py"), str(BASE / "protocol.template.json")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "BLOCKED")

    def test_missing_real_measurement_rejected(self):
        self.report["records"][0]["measurement_status"] = "pending"
        with self.assertRaises(v.Blocked): self.validate()

    def test_stale_protocol_rejected(self):
        self.p["dataset"] = "changed"
        with self.assertRaises(v.Blocked): self.validate()

    def test_wrong_initial_encoder_rejected(self):
        self.report["records"][0]["initial_encoder_sha256"] = h("different")
        with self.assertRaises(v.Blocked): self.validate()

    def test_wrong_initial_head_rejected(self):
        self.report["records"][0]["initial_head_state_sha256"] = h("different")
        with self.assertRaises(v.Blocked): self.validate()

    def test_missing_record_rejected(self):
        self.report["records"].pop()
        with self.assertRaises(v.Blocked): self.validate()

    def test_duplicate_record_rejected(self):
        self.report["records"].append(copy.deepcopy(self.report["records"][0]))
        with self.assertRaises(v.Blocked): self.validate()

    def test_counter_cannot_be_guessed_from_epochs(self):
        self.report["records"][0]["counters"]["optimizer_steps"] = 10
        with self.assertRaises(v.Blocked): self.validate()

    def test_boolean_counter_rejected(self):
        self.report["records"][0]["counters"]["optimizer_steps"] = True
        with self.assertRaises(v.Blocked): self.validate()

    def test_unknown_counter_rejected(self):
        self.report["records"][0]["counters"]["epoch_matched"] = 1
        with self.assertRaises(v.Blocked): self.validate()

    def test_budget_exceeded_rejected(self):
        self.p["arms"][0]["phase_plan"][0]["limits"]["optimizer_steps"] = 0
        self.rebind()
        with self.assertRaises(v.Blocked): self.validate()

    def test_unavailable_labels_rejected(self):
        self.p["available_labels"] = [[0]]
        self.rebind()
        with self.assertRaises(v.Blocked): self.validate()

    def test_test_row_in_fitting_trace_rejected(self):
        reg = copy.deepcopy(self.registry)
        reg[self.rows[0]]["partition"] = "test"
        trace = v.artifact(self.root, self.report["records"][0]["trace"])
        with self.assertRaises(v.Blocked): v.trace_check(trace, reg, self.p, self.p["arms"][0]["phase_plan"][0])

    def test_future_row_rejected(self):
        reg = copy.deepcopy(self.registry)
        reg[self.rows[0]]["available_from_task"] = 1
        trace = v.artifact(self.root, self.report["records"][0]["trace"])
        with self.assertRaises(v.Blocked): v.trace_check(trace, reg, self.p, self.p["arms"][0]["phase_plan"][0])

    def test_orphan_optimizer_step_rejected(self):
        self.change_artifact(self.report["records"][0], "trace", lambda t: t["events"].insert(0, {"kind": "step"}))
        with self.assertRaises(v.Blocked): self.validate()

    def test_uncommitted_microbatch_rejected(self):
        self.change_artifact(self.report["records"][0], "trace", lambda t: t["events"].pop())
        with self.assertRaises(v.Blocked): self.validate()

    def test_nme_gradient_event_rejected(self):
        self.change_artifact(self.report["records"][1], "trace", lambda t: t["events"].append({"kind": "step"}))
        with self.assertRaises(v.Blocked): self.validate()

    def test_wrong_target_rejected(self):
        self.change_artifact(self.report["records"][0], "trace", lambda t: t["events"][0]["multiclass_targets"].__setitem__(0, [0, 1]))
        with self.assertRaises(v.Blocked): self.validate()

    def test_binary_targets_are_distinct_from_raw_rows(self):
        phase = copy.deepcopy(self.p["arms"][0]["phase_plan"][0])
        phase["target_kind"] = "binary"
        trace = {"events": [{"kind": "microbatch", "rows": self.rows,
                             "binary_targets": [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]], "multiclass_targets": []},
                            {"kind": "step"}]}
        counts, _ = v.trace_check(trace, self.registry, self.p, phase)
        self.assertEqual(counts["binary_targets"], 4)
        self.assertEqual(counts["raw_row_presentations"], 2)
        self.assertEqual(counts["optimizer_steps"], 1)

    def test_duplicate_binary_decision_rejected(self):
        phase = copy.deepcopy(self.p["arms"][0]["phase_plan"][0])
        phase["target_kind"] = "binary"
        trace = {"events": [{"kind": "microbatch", "rows": [self.rows[0]],
                             "binary_targets": [[0, 0, 1], [0, 0, 1]], "multiclass_targets": []}, {"kind": "step"}]}
        with self.assertRaises(v.Blocked): v.trace_check(trace, self.registry, self.p, phase)

    def test_repeated_raw_presentations_not_distinct_rows(self):
        phase = self.p["arms"][0]["phase_plan"][1]
        trace = {"events": [{"kind": "fit_read", "rows": [self.rows[0]] * 3, "binary_targets": [], "multiclass_targets": []}]}
        counts, _ = v.trace_check(trace, self.registry, self.p, phase)
        self.assertEqual(counts["raw_row_presentations"], 3)
        self.assertEqual(counts["unique_raw_rows"], 1)

    def test_memory_missing_category_rejected(self):
        self.change_artifact(self.report["records"][0], "memory", lambda m: m["components"].pop())
        with self.assertRaises(v.Blocked): self.validate()

    def test_memory_duplicated_storage_rejected(self):
        self.change_artifact(self.report["records"][0], "memory", lambda m: m["components"].append(m["components"][0]))
        with self.assertRaises(v.Blocked): self.validate()

    def test_memory_byte_mismatch_rejected(self):
        self.change_artifact(self.report["records"][0], "memory", lambda m: m["components"][0].__setitem__("bytes", 9))
        with self.assertRaises(v.Blocked): self.validate()

    def test_optimizer_peak_below_snapshot_rejected(self):
        self.change_artifact(self.report["records"][0], "memory", lambda m: m.__setitem__("optimizer_peak_bytes", 0))
        with self.assertRaises(v.Blocked): self.validate()

    def test_no_gradient_can_retain_optimizer_memory(self):
        phase = self.p["arms"][0]["phase_plan"][1]
        memory = self.memory(True)
        measured = v.memory_check(memory, phase)
        self.assertEqual(measured["optimizer_bytes"], 8)
        self.assertEqual(measured["optimizer_peak_bytes"], 8)
        self.assertEqual(measured["checkpoint_retained_bytes"], 40)
        record = self.report["records"][1]
        record["memory"] = self.put(memory)
        record["memory_metrics"] = measured
        self.assertEqual(self.validate()["status"], "ACCOUNTING_CONSISTENT")

    def test_cpu_cuda_must_be_na_not_fabricated_zero(self):
        self.change_artifact(self.report["records"][0], "memory", lambda m: m.__setitem__("cuda_peak_allocated_bytes", 0))
        with self.assertRaises(v.Blocked): self.validate()

    def test_cuda_peak_order_rejected(self):
        phase = copy.deepcopy(self.p["arms"][0]["phase_plan"][0])
        phase["device"] = "cuda"
        memory = self.memory(True)
        memory.update(cuda_peak_allocated_bytes=20, cuda_peak_reserved_bytes=10, device_na_reason=None)
        with self.assertRaises(v.Blocked): v.memory_check(memory, phase)

    def test_synthetic_cannot_be_relabelled_real_by_report(self):
        self.report["evidence_kind"] = "real"
        with self.assertRaises(v.Blocked): self.validate()

    def test_modified_artifact_hash_rejected(self):
        self.report["records"][0]["trace"]["sha256"] = h("stale")
        with self.assertRaises(v.Blocked): self.validate()

    def test_path_traversal_rejected(self):
        self.report["records"][0]["trace"]["path"] = "../outside.json"
        with self.assertRaises(v.Blocked): self.validate()

    def test_absolute_windows_path_rejected(self):
        self.report["records"][0]["trace"]["path"] = "C:/outside.json"
        with self.assertRaises(v.Blocked): self.validate()

    def test_duplicate_json_key_rejected(self):
        with self.assertRaises(v.Blocked): v.decode(b'{"x": 1, "x": 2}')

    def test_nonfinite_json_rejected(self):
        with self.assertRaises(v.Blocked): v.decode(b'{"x": NaN}')

    def test_nested_overflowed_json_floats_rejected(self):
        for value in (b'{"x": [1e400]}', b'{"x": {"y": -1e400}}'):
            with self.subTest(value=value), self.assertRaises(v.Blocked): v.decode(value)

    def test_shared_pretrain_cost_cannot_differ(self):
        record = self.report["records"][2]
        memory = v.artifact(self.root, record["memory"])
        memory["host_peak_rss_bytes"] = 101
        record["memory"] = self.put(memory)
        record["memory_metrics"] = v.memory_check(memory, self.p["arms"][1]["phase_plan"][0])
        with self.assertRaisesRegex(v.Blocked, "shared pretraining"): self.validate()

    def test_shared_pretrain_steps_match_even_without_optional_matching(self):
        self.p["contrasts"][0]["match_metrics"] = []
        self.rebind()
        record = self.report["records"][2]
        trace = v.artifact(self.root, record["trace"])
        trace["events"].extend(copy.deepcopy(trace["events"]))
        record["trace"] = self.put(trace)
        record["counters"], _ = v.trace_check(trace, self.registry, self.p, self.p["arms"][1]["phase_plan"][0])
        self.assertEqual(record["counters"]["optimizer_steps"], 2)
        with self.assertRaisesRegex(v.Blocked, "shared pretraining"): self.validate()

    def test_complete_phase_coverage_required(self):
        self.p["arms"][0]["phase_plan"].pop()
        self.rebind()
        with self.assertRaises(v.Blocked): self.validate()

    def test_shared_pretrain_cannot_be_omitted(self):
        self.p["arms"][0]["phase_plan"].pop(0)
        self.rebind()
        with self.assertRaises(v.Blocked): self.validate()

    def test_exact_matching_failure_rejected(self):
        record = self.report["records"][-1]
        trace = v.artifact(self.root, record["trace"])
        trace["events"][0]["rows"].append(self.rows[0])
        record["trace"] = self.put(trace)
        phase = self.p["arms"][-1]["phase_plan"][-1]
        record["counters"], _ = v.trace_check(trace, self.registry, self.p, phase)
        with self.assertRaisesRegex(v.Blocked, "unmatched axis"): self.validate()


if __name__ == "__main__":
    unittest.main()
