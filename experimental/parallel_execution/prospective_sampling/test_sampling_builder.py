"""Synthetic temporary-array end-to-end tests. No real-data preprocessing."""
import copy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import sampling_builder as b


BASE = Path(__file__).resolve().parent


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".synthetic-", dir=BASE)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.counter = 0
        self.policy = {"schema_version": 1, "arm": "P", "seed": 42, "offline_normal_cap": None,
                       "calibration_numerator": 1, "calibration_denominator": 10,
                       "transform_fit_policy": "common_task0_prospective_fit"}
        self.pfile = self.json("policy.json", self.policy)
        self.current = self.contract(0, [(0, 30), (1, 10)])
        self.ifile = self.json("increment.json", self.current)

    def json(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def feature(self, cid, n, suffix="", dtype="float32"):
        path = self.source / f"class_{cid:02d}{suffix}.npy"
        array = np.arange(n * 3, dtype=dtype).reshape(n, 3) + cid * 1000
        np.save(path, array)
        return {"path": path.name, "sha256": b.sha256_file(path), "rows": n}

    def sidecar(self, name, values):
        path = self.source / name
        np.save(path, values)
        return {"path": name, "sha256": b.sha256_file(path), "rows": len(values)}

    def contract(self, increment, classes):
        return {"schema_version": 1, "kind": "available_train_increment", "dataset_id": "synthetic",
                "namespace": "synthetic-source-v1", "increment": increment, "feature_dim": 3,
                "normal_class_id": 0, "group_mode": "row_identity_proxy",
                "classes": [{"class_id": cid, "name": f"class-{cid}", "introduced_increment": increment,
                             "partition": "train", "shards": [{"shard_id": f"class_{cid:02d}.npy",
                                "features": self.feature(cid, n), "row_ids": None, "group_ids": None, "labels": None}]}
                            for cid, n in classes]}

    def derive(self, contract=None, policy=None, previous=None, output=None, **kwargs):
        self.counter += 1
        infile = self.ifile if contract is None else self.json(f"input-{self.counter}.json", contract)
        pfile = self.pfile if policy is None else self.json(f"policy-{self.counter}.json", policy)
        output = output or self.root / f"out-{self.counter}"
        result = b.derive(infile, self.source, output, pfile, previous=previous, chunk_rows=4, **kwargs)
        return output, result

    def ids(self, directory, part):
        return tuple(bytes(r).decode("ascii") for chunk in b.iter_partition(directory, part, chunk_rows=3) for r in chunk["row_ids"])

    def test_real_array_end_to_end_conservation_and_provenance(self):
        out, manifest = self.derive()
        self.assertEqual(manifest["normal_cap"], 9)
        self.assertEqual([manifest["collections"][p]["rows"] for p in ("fit", "calibration", "omitted")], [18, 4, 18])
        parts = [set(self.ids(out, p)) for p in ("fit", "calibration", "omitted")]
        self.assertEqual(len(set.union(*parts)), 40)
        self.assertFalse(parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
        for chunk in b.iter_partition(out, "fit", chunk_rows=2):
            self.assertLessEqual(len(chunk["x"]), 2)
            for x, label, (slot, offset), rid in zip(chunk["x"], chunk["labels"], chunk["source_index"], chunk["row_ids"]):
                shard = manifest["source_shards"][slot]
                with b.mapped(self.source / shard["features"]["path"]) as source:
                    np.testing.assert_array_equal(x, source[offset])
                self.assertEqual(label, shard["class_id"])
                self.assertEqual(bytes(rid).decode(), b.row_identity("synthetic-source-v1", shard["shard_id"], int(offset)))
        self.assertFalse(manifest["capture_independent_established"])

    def test_o_p_bridge_same_calibration_attacks_and_transform(self):
        pdir, pm = self.derive()
        opolicy = dict(self.policy, arm="O", offline_normal_cap=20)
        odir, om = self.derive(policy=opolicy)
        self.assertEqual(self.ids(pdir, "calibration"), self.ids(odir, "calibration"))
        self.assertLess(set(self.ids(pdir, "fit")), set(self.ids(odir, "fit")))
        self.assertEqual(pm["common_transform_fit_sha256"], om["common_transform_fit_sha256"])
        self.assertEqual(pm["collections"]["transform_fit"], om["collections"]["transform_fit"])
        self.assertEqual(self.ids(pdir, "fit"), self.ids(pdir, "transform_fit"))

    def test_increment_freezes_cap_and_does_not_modify_previous(self):
        old, pm = self.derive()
        before = b.sha256_file(old / "manifest.json")
        future = self.contract(1, [(2, 100), (3, 7)])
        out, result = self.derive(contract=future, previous=old)
        self.assertEqual(result["normal_cap"], 9)
        self.assertEqual(result["common_transform_fit_sha256"], pm["common_transform_fit_sha256"])
        self.assertIsNone(result["collections"]["transform_fit"])
        self.assertEqual(b.sha256_file(old / "manifest.json"), before)
        self.assertEqual(result["binding"]["previous_manifest_sha256"], before)
        self.assertEqual(len(self.ids(out, "fit")), 96)

    def test_input_order_and_chunk_size_do_not_change_arrays(self):
        first, fm = self.derive()
        changed = copy.deepcopy(self.current)
        changed["classes"].reverse()
        second, sm = self.derive(contract=changed)
        self.assertEqual(fm["collections"], sm["collections"])
        third = self.root / "third"
        tm = b.derive(self.ifile, self.source, third, self.pfile, chunk_rows=1)
        self.assertEqual(fm["collections"], tm["collections"])

    def test_seed_reproduction_and_change(self):
        first, fm = self.derive()
        second, sm = self.derive()
        self.assertEqual(fm["collections"], sm["collections"])
        other, _ = self.derive(policy=dict(self.policy, seed=43))
        self.assertNotEqual(self.ids(first, "calibration"), self.ids(other, "calibration"))

    def test_more_than_toy_limit_is_supported_with_small_chunks(self):
        current = self.contract(0, [(0, 10005), (1, 20)])
        infile = self.json("larger-toy.json", current)
        out = self.root / "larger-toy"
        result = b.derive(infile, self.source, out, self.pfile, chunk_rows=128)
        self.assertEqual(sum(c["source_rows"] for c in result["classes"]), 10025)
        self.assertEqual(result["normal_cap"], 18)
        self.assertEqual(len(self.ids(out, "fit")), 36)

    def test_completed_resume_is_idempotent_and_checks_input(self):
        out, result = self.derive()
        before = b.sha256_file(out / "COMPLETE.json")
        repeated = b.derive(self.ifile, self.source, out, self.pfile, resume=True, chunk_rows=2)
        self.assertEqual(result, repeated)
        self.assertEqual(before, b.sha256_file(out / "COMPLETE.json"))
        self.json("increment.json", dict(self.current, namespace="changed"))
        with self.assertRaisesRegex(b.Blocked, "resume input"):
            b.derive(self.ifile, self.source, out, self.pfile, resume=True)

    def test_existing_output_never_overwritten(self):
        out, _ = self.derive()
        before = b.sha256_file(out / "COMPLETE.json")
        with self.assertRaises(b.Blocked):
            b.derive(self.ifile, self.source, out, self.pfile)
        self.assertEqual(before, b.sha256_file(out / "COMPLETE.json"))

    def test_nested_completion_marker_is_not_ignored_by_closure(self):
        out, _ = self.derive()
        (out / "fit" / "COMPLETE.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(b.Blocked, "closure"):
            b.load_increment(out)

    def test_contract_changes_while_reading_fail_closed(self):
        original = b.read_json
        def mutating_read(path):
            result = original(path)
            if Path(path) == self.ifile:
                self.ifile.write_text(json.dumps(dict(self.current, namespace="changed")), encoding="utf-8")
            return result
        with patch.object(b, "read_json", side_effect=mutating_read), self.assertRaisesRegex(b.Blocked, "while reading"):
            self.derive()

    def test_contract_change_during_completed_resume_is_rejected(self):
        out, _ = self.derive()
        original = b.checked_input
        def mutate_during_resume(*args, **kwargs):
            result = original(*args, **kwargs)
            self.ifile.write_text(json.dumps(dict(self.current, namespace="changed-during-resume")), encoding="utf-8")
            return result
        with patch.object(b, "checked_input", side_effect=mutate_during_resume), self.assertRaisesRegex(b.Blocked, "during validation"):
            b.derive(self.ifile, self.source, out, self.pfile, resume=True, chunk_rows=4)

    def test_bad_hash_leaves_incomplete_output_not_loadable_or_resumable(self):
        bad = copy.deepcopy(self.current)
        bad["classes"][0]["shards"][0]["features"]["sha256"] = "a" * 64
        out = self.root / "failed"
        with self.assertRaisesRegex(b.Blocked, "hash mismatch"):
            self.derive(contract=bad, output=out)
        self.assertFalse((out / "COMPLETE.json").exists())
        self.assertTrue((out / "FAILED.json").is_file())
        with self.assertRaises((OSError, b.Blocked)):
            b.load_increment(out)
        with self.assertRaises((OSError, b.Blocked)):
            b.derive(self.ifile, self.source, out, self.pfile, resume=True)

    def test_changed_completed_array_is_rejected(self):
        out, _ = self.derive()
        with (out / "fit" / "labels.npy").open("ab") as handle:
            handle.write(b"mutation")
        with self.assertRaisesRegex(b.Blocked, "artifact hash"):
            b.load_increment(out)

    def test_extra_output_file_and_external_manifest_hash_rejected(self):
        out, _ = self.derive()
        with self.assertRaisesRegex(b.Blocked, "externally bound"):
            b.load_increment(out, "a" * 64)
        (out / "unregistered.txt").write_text("not part of complete closure")
        with self.assertRaisesRegex(b.Blocked, "closure changed"):
            b.load_increment(out)

    def test_prefix_rewrite_skip_and_policy_change_rejected(self):
        old, _ = self.derive()
        cases = [(self.contract(1, [(0, 3)]), None), (self.contract(2, [(4, 3)]), None),
                 (self.contract(1, [(2, 3)]), dict(self.policy, seed=9))]
        for manifest, policy in cases:
            with self.subTest(manifest=manifest["increment"]), self.assertRaises(b.Blocked):
                self.derive(contract=manifest, policy=policy, previous=old)

    def test_future_test_unknown_metadata_and_labels_rejected(self):
        for edit in (lambda m: m.update(future_class_counts={2: 100}),
                     lambda m: m.update(official_test={}), lambda m: m.update(kind="sealed_official_test"),
                     lambda m: m["classes"][1].update(introduced_increment=1),
                     lambda m: m["classes"][1].update(partition="test")):
            current = copy.deepcopy(self.current)
            edit(current)
            with self.assertRaises(b.Blocked):
                self.derive(contract=current)

    def test_label_sidecar_mismatch_rejected(self):
        bad = copy.deepcopy(self.current)
        bad["classes"][1]["shards"][0]["labels"] = self.sidecar("wrong-label.npy", np.zeros(10, dtype="int64"))
        with self.assertRaisesRegex(b.Blocked, "label sidecar"):
            self.derive(contract=bad)

    def test_correct_integer_label_sidecars_pass(self):
        current = copy.deepcopy(self.current)
        for cls in current["classes"]:
            n = cls["shards"][0]["features"]["rows"]
            cls["shards"][0]["labels"] = self.sidecar(f"labels-{cls['class_id']}.npy", np.full(n, cls["class_id"], dtype="int64"))
        self.derive(contract=current)

    def test_row_count_shape_dtype_and_nonfinite_rejected(self):
        bad = copy.deepcopy(self.current)
        bad["classes"][0]["shards"][0]["features"]["rows"] = 31
        with self.assertRaisesRegex(b.Blocked, "row count"):
            self.derive(contract=bad)
        for array in (np.ones((30, 4), dtype="float32"), np.ones((30, 3), dtype="int64"), np.full((30, 3), np.nan, dtype="float32")):
            bad = copy.deepcopy(self.current)
            ref = self.sidecar("invalid.npy", array)
            bad["classes"][0]["shards"][0]["features"] = ref
            bad["classes"][0]["shards"][0]["shard_id"] = ref["path"]
            with self.assertRaises(b.Blocked):
                self.derive(contract=bad)

    def test_duplicate_provided_ids_and_cross_prefix_ids_rejected(self):
        bad = copy.deepcopy(self.current)
        duplicate = np.array(["a" * 64] * 30, dtype="S64")
        bad["classes"][0]["shards"][0]["row_ids"] = self.sidecar("ids-bad.npy", duplicate)
        with self.assertRaisesRegex(b.Blocked, "identity collision"):
            self.derive(contract=bad)
        old, _ = self.derive()
        future = self.contract(1, [(2, 1)])
        prior_id = self.ids(old, "fit")[0]
        future["classes"][0]["shards"][0]["row_ids"] = self.sidecar("old-id.npy", np.array([prior_id], dtype="S64"))
        with self.assertRaisesRegex(b.Blocked, "identity collision"):
            self.derive(contract=future, previous=old)

    def test_missing_group_sidecar_cannot_claim_grouped(self):
        with self.assertRaisesRegex(b.Blocked, "group claim"):
            self.derive(contract=dict(self.current, group_mode="provided_disjoint"))

    def test_provided_group_cross_split_is_rejected(self):
        current = copy.deepcopy(self.current)
        current["group_mode"] = "provided_disjoint"
        for cls in current["classes"]:
            n = cls["shards"][0]["features"]["rows"]
            cls["shards"][0]["group_ids"] = self.sidecar(f"groups-{cls['class_id']}.npy", np.array(["b" * 64] * n, dtype="S64"))
        with self.assertRaisesRegex(b.Blocked, "group crosses"):
            self.derive(contract=current)

    def test_unique_provided_groups_pass_without_claiming_truth(self):
        current = copy.deepcopy(self.current)
        current["group_mode"] = "provided_disjoint"
        for cls in current["classes"]:
            n = cls["shards"][0]["features"]["rows"]
            values = [b.object_hash(["synthetic-group", cls["class_id"], i]) for i in range(n)]
            cls["shards"][0]["group_ids"] = self.sidecar(f"groups-{cls['class_id']}.npy", np.array(values, dtype="S64"))
        _, result = self.derive(contract=current)
        self.assertTrue(result["group_boundary_checked"])
        self.assertFalse(result["capture_independent_established"])

    def test_path_escape_rejected(self):
        bad = copy.deepcopy(self.current)
        bad["classes"][0]["shards"][0]["features"]["path"] = "../outside.npy"
        bad["classes"][0]["shards"][0]["shard_id"] = "../outside.npy"
        with self.assertRaises(b.Blocked):
            self.derive(contract=bad)
        with self.assertRaisesRegex(b.Blocked, "relative path"):
            b.safe_path(self.source, "../outside.npy")

    def test_symlink_input_rejected(self):
        original = self.source / self.current["classes"][0]["shards"][0]["features"]["path"]
        link = self.source / "linked.npy"
        try:
            link.symlink_to(original)
        except OSError:
            self.skipTest("symlink creation unavailable on this host")
        bad = copy.deepcopy(self.current)
        bad["classes"][0]["shards"][0]["features"]["path"] = link.name
        bad["classes"][0]["shards"][0]["shard_id"] = link.name
        with self.assertRaisesRegex(b.Blocked, "symlink"):
            self.derive(contract=bad)

    def test_empty_singleton_and_two_row_classes(self):
        old, _ = self.derive()
        current = self.contract(1, [(2, 0), (3, 1), (4, 2)])
        _, result = self.derive(contract=current, previous=old)
        self.assertEqual([(c["fit_rows"], c["calibration_rows"]) for c in result["classes"]], [(0, 0), (1, 0), (1, 1)])

    def test_no_attack_and_offline_cap_below_common_transform_fail(self):
        with self.assertRaisesRegex(b.Blocked, "Task-0 attack"):
            self.derive(contract=self.contract(0, [(0, 3)]))
        self.feature(0, 30)
        with self.assertRaisesRegex(b.Blocked, "O must contain P"):
            self.derive(policy=dict(self.policy, arm="O", offline_normal_cap=2))

    def test_windows_reparse_point_flag_is_rejected_without_real_symlink_privilege(self):
        from types import SimpleNamespace
        original = Path.lstat
        target = self.source / "class_00.npy"
        def lstat(path, *args, **kwargs):
            if path == target:
                return SimpleNamespace(st_file_attributes=0x400, st_mode=original(path, *args, **kwargs).st_mode)
            return original(path, *args, **kwargs)
        with patch.object(Path, "lstat", lstat), self.assertRaisesRegex(b.Blocked, "reparse"):
            b.no_links(target)

    def test_contract_decoder_rejects_duplicates_nonfinite_and_bad_id_sidecar(self):
        raw = self.root / "bad-json.json"
        for content in ('{"a":1,"a":2}', '{"a":1e400}'):
            raw.write_text(content)
            with self.assertRaises(b.Blocked):
                b.read_json(raw)
        bad = copy.deepcopy(self.current)
        bad["classes"][0]["shards"][0]["row_ids"] = self.sidecar("invalid-ids.npy", np.array(["not-a-digest"] * 30, dtype="S64"))
        with self.assertRaisesRegex(b.Blocked, "SHA-256"):
            self.derive(contract=bad)

    def test_mid_derivation_mutation_cannot_produce_complete_marker(self):
        out = self.root / "mid-failure"
        original = b.materialize
        mutated = [False]
        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            if not mutated[0]:
                mutated[0] = True
                with (self.source / "class_00.npy").open("ab") as handle:
                    handle.write(b"mutated")
            return result
        with patch.object(b, "materialize", side_effect=changed), self.assertRaisesRegex(b.Blocked, "source changed"):
            self.derive(output=out)
        self.assertFalse((out / "COMPLETE.json").exists())

    def source_catalog(self):
        future = self.contract(1, [(2, 11), (3, 3)])
        classes = self.current["classes"] + future["classes"]
        return {"schema_version": 1, "dataset": "synthetic", "feature_dim": 3, "metric_profile": "nids",
                "normal_class_id": 0, "problem_type": "intrusion_detection", "source": {}, "task_semantics": "class_incremental",
                "tasks": [[0, 1], [2, 3]], "classes": [{"id": c["class_id"], "name": c["name"], "train": [c["shards"][0]["features"]],
                         "test": [self.feature(c["class_id"], 2, suffix="-test")]} for c in classes]}

    def test_adapter_changes_future_test_metadata_without_changing_prefix_contract(self):
        catalog = self.source_catalog()
        source = self.json("catalog.json", catalog)
        first = self.root / "adapter-one"
        b.adapt_replayids(source, first, expected_sha256=b.sha256_file(source))
        changed = copy.deepcopy(catalog)
        changed["classes"][2]["train"][0]["rows"] = 999
        changed["classes"][3]["name"] = "different-future"
        changed["classes"][0]["test"][0]["sha256"] = "c" * 64
        other = self.json("catalog-two.json", changed)
        second = self.root / "adapter-two"
        b.adapt_replayids(other, second, expected_sha256=b.sha256_file(other))
        self.assertEqual((first / "increment_00.json").read_bytes(), (second / "increment_00.json").read_bytes())
        one = b.derive(first / "increment_00.json", self.source, self.root / "adapter-build-1", self.pfile, chunk_rows=4)
        two = b.derive(second / "increment_00.json", self.source, self.root / "adapter-build-2", self.pfile, chunk_rows=3)
        self.assertEqual(one["collections"], two["collections"])

    def test_separate_test_verification_does_not_touch_training_or_test_bytes(self):
        catalog = self.source_catalog()
        source = self.json("catalog.json", catalog)
        adapted = self.root / "adapted"
        b.adapt_replayids(source, adapted, expected_sha256=b.sha256_file(source))
        before = {r["path"]: b.sha256_file(self.source / r["path"]) for c in catalog["classes"] for r in c["test"]}
        result = b.verify_official_test(adapted / "official_test.json", self.source, self.root / "test-binding", chunk_rows=1)
        self.assertEqual(result["total_rows"], 8)
        self.assertFalse(result["participated_in_train_selection"])
        self.assertEqual(before, {path: b.sha256_file(self.source / path) for path in before})
        with self.assertRaises(b.Blocked):
            b.derive(adapted / "official_test.json", self.source, self.root / "test-as-train", self.pfile)

    def test_cli_is_executable_and_strict_unknown_flags_rejected(self):
        out = self.root / "cli-output"
        result = subprocess.run([sys.executable, "-B", str(BASE / "sampling_builder.py"), "derive", "--input", str(self.ifile),
                                 "--source-root", str(self.source), "--output", str(out), "--policy", str(self.pfile), "--chunk-rows", "3"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "COMPLETE")
        bad = subprocess.run([sys.executable, "-B", str(BASE / "sampling_builder.py"), "check", str(out), "--future-count", "1"], capture_output=True)
        self.assertNotEqual(bad.returncode, 0)


if __name__ == "__main__":
    unittest.main()
