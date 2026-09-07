"""CPU synthetic invariant tests; no empirical repair-utility claims."""
import copy
import json
import math
import os
import subprocess
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import runner
from adapt_native_exports import adapt
from make_demo import create_demo, prospective_policy
from score_core import (ARMS, Blocked, candidate_pool, classification_report, digest, evaluate_pair,
                        fit_r1, fused, group_intervals, reachability_bound, score_path, sigmoid)
from runner import (Ledger, bound_path, evaluate_study, file_hash, load_partition, read_json, run_study,
                    source_binding, validate_manifest, validate_policy, verify_real_evidence)


def mathematical_partition(groups=64):
    classes = ["application-a", "application-b", "application-c"]
    labels = classes * groups
    logits = np.asarray([[1., 2., -4.], [-4., 3., -4.], [-4., -4., 3.]] * groups)
    return {"role": "fit", "labels": labels, "row_ids": [f"row-{i}" for i in range(len(labels))],
            "group_ids": [f"group-{g}" for g in range(groups) for _ in classes],
            "new_head_logits": logits.tolist(), "new_router_raw": np.zeros_like(logits).tolist()}, classes


class NumericTests(unittest.TestCase):
    def test_sigmoid_extreme_finite_logits(self):
        values = sigmoid(np.asarray([-10000., 0., 10000.]))
        np.testing.assert_array_equal(values, [0, .5, 1])

    def test_non_numeric_and_overflowed_score_arithmetic_fail_closed(self):
        p = prospective_policy(2, 6)
        for h, r in (([[True, False]], [[0., 0.]]), ([["1", "2"]], [[0, 0]]),
                     ([[0., 0.]], [[1e308, -1e308]])):
            with self.assertRaises(Blocked):
                fused(h, r, p["fusion"])

    def test_identity_r1_matches_baseline_exactly(self):
        part, classes = mathematical_partition(2)
        f = prospective_policy(2, 6)["fusion"]
        np.testing.assert_array_equal(fused(part["new_head_logits"], part["new_router_raw"], f),
                                      fused(part["new_head_logits"], part["new_router_raw"], f, 0, (1, 0)))

    def test_actual_gradient_fit_changes_params_and_decreases_fit_loss(self):
        part, classes = mathematical_partition(4)
        policy = prospective_policy(4, 12)
        state = fit_r1(part, classes, classes[0], policy["fusion"], policy["r1"])
        self.assertNotEqual((state["scale"], state["offset"]), (1, 0))
        self.assertLess(state["loss_history"][-1], state["loss_history"][0])
        self.assertEqual(state["fit_rows_processed"], 24 * 12)
        self.assertGreater(state["fit_wall_seconds"], 0)
        self.assertGreater(state["traced_peak_bytes"], 0)

    def test_r1_preserves_router_and_other_heads(self):
        part, classes = mathematical_partition(4)
        policy = prospective_policy(4, 12)
        original = copy.deepcopy(part)
        state = fit_r1(part, classes, classes[0], policy["fusion"], policy["r1"])
        before = fused(part["new_head_logits"], part["new_router_raw"], policy["fusion"])
        after = fused(part["new_head_logits"], part["new_router_raw"], policy["fusion"], 0, (state["scale"], state["offset"]))
        np.testing.assert_array_equal(before[:, 1:], after[:, 1:])
        self.assertEqual(part, original)

    def test_optimizer_rejects_non_fit_labels(self):
        part, classes = mathematical_partition(4)
        policy = prospective_policy(4, 12)
        for role in ("trigger", "acceptance", "subsequent"):
            with self.assertRaises(Blocked):
                fit_r1({**part, "role": role}, classes, classes[0], policy["fusion"], policy["r1"])

    def test_optimization_respects_projection_bounds(self):
        part, classes = mathematical_partition(4)
        p = prospective_policy(4, 12)
        p["r1"].update(learning_rate=10000)
        state = fit_r1(part, classes, classes[0], p["fusion"], p["r1"])
        self.assertTrue(.5 <= state["scale"] <= 2 and -2 <= state["offset"] <= 2)

    def test_actual_fit_acceptance_and_rejection_branches(self):
        part, classes = mathematical_partition(64)
        policy = prospective_policy(64, 192)
        policy["r1"].update(steps=48, learning_rate=2)
        # Deliberately loose toy tolerance exercises the software ACCEPT branch,
        # not a scientifically acceptable deployment risk level.
        policy["acceptance"].update(max_old_error_increase=.6, max_new_error_increase=.6)
        state = fit_r1(part, classes, classes[0], policy["fusion"], policy["r1"])
        acceptance = {**part, "role": "acceptance"}
        accepted = evaluate_pair(acceptance, classes, classes[:2], policy["fusion"], state, policy, "acceptance")
        self.assertEqual(accepted["decision"], "ACCEPT_RESEARCH_VARIANT")
        self.assertTrue(accepted["raw_alerts_preserved"])
        policy["acceptance"]["min_target_improvement"] = 1
        rejected = evaluate_pair(acceptance, classes, classes[:2], policy["fusion"], state, policy, "acceptance")
        self.assertEqual(rejected["decision"], "REJECT")
        self.assertEqual(accepted["baseline_scores"], rejected["baseline_scores"])

    def test_group_ci_not_fake_precision_from_duplicate_rows(self):
        v, g = np.asarray([1., -1., 0., 1.]), np.asarray(["a", "b", "c", "d"])
        a = group_intervals({"x": (v, g, -1, 1)}, .05, 2, 2)["x"]
        b = group_intervals({"x": (np.repeat(v, 100), np.repeat(g, 100), -1, 1)}, .05, 2, 2)["x"]
        self.assertEqual((a["estimate"], a["lower"], a["upper"]), (b["estimate"], b["lower"], b["upper"]))
        self.assertEqual(b["groups"], 4)

    def test_group_ci_formula_and_simultaneity(self):
        groups = [str(x) for x in range(128)]
        metrics = {"x": (np.full(128, .2), groups, -1, 1)}
        result = group_intervals(metrics, .05, 10, 2)["x"]
        radius = 2 * math.sqrt(math.log(400) / 256)
        self.assertAlmostEqual(result["upper"], .2 + radius)
        smaller = group_intervals(metrics, .05, 1, 2)["x"]
        self.assertGreater(result["upper"], smaller["upper"])

    def test_group_small_support_and_nonfinite_fail_closed(self):
        for values, groups in (([0, 0], ["one", "one"]), ([0, float("nan")], ["a", "b"])):
            with self.assertRaises(Blocked):
                group_intervals({"x": (values, groups, -1, 1)}, .05, 1, 2)

    def test_actual_confusion_and_attack_semantics(self):
        labels = ["benign", "attack", "attack", "benign"]
        predictions = ["attack", "attack", "benign", "benign"]
        ordinary = classification_report(labels, predictions, ["benign", "attack"])
        self.assertEqual(ordinary["confusion_counts"], [[1, 1], [1, 1]])
        self.assertNotIn("attack_recall", ordinary)
        attack = classification_report(labels, predictions, ["benign", "attack"], ["attack"])
        self.assertEqual(attack["attack_recall"], .5)
        self.assertEqual(attack["false_positive_rate"], .5)

    def test_reachability_high_but_no_shared_affine_solution(self):
        # Both target-a rows are individually reachable, but with b<=2:
        # row1 requires a > logit(.95)-2 > .944,
        # row2 requires a < 2-logit(.8) < .614. Impossible jointly.
        part = {"new_head_logits": [[1, math.log(19), -10], [-1, math.log(4), -10]],
                "new_router_raw": [[0, 0, 0], [0, 0, 0]]}
        policy = prospective_policy(2, 6)
        bound = reachability_bound(part, 0, policy["fusion"], policy["r1"])
        self.assertTrue(bound["can_cross_necessary_bound"].all())
        self.assertFalse(bound["jointly_achievable"])
        self.assertGreater(math.log(19) - 2, 2 - math.log(4))
        for scale in np.linspace(.5, 2, 20):
            for offset in np.linspace(-2, 2, 20):
                scores = fused(part["new_head_logits"], part["new_router_raw"], policy["fusion"], 0, (scale, offset))
                self.assertFalse(np.all(scores.argmax(axis=1) == 0))


class BundleTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parent / "_test_scratch"
        base.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.root = Path(self.temp.name).resolve()
        self.assertTrue(self.root.is_relative_to(base.resolve()))
        self.bundle = self.root / "bundle"
        self.manifest = create_demo(self.bundle, seed=11, groups=6)
        self.policy = read_json(self.bundle / "policy.json")
        self.ledger_path = self.root / "ledger.sqlite"
        self.output = self.root / "outputs"

    def tearDown(self):
        self.temp.cleanup()

    def run_decisions(self, study="test-study"):
        return run_study(self.bundle / "manifest.json", self.bundle / "policy.json", self.ledger_path, self.output, study)

    def run_evaluation(self, study="test-study"):
        return evaluate_study(self.bundle / "manifest.json", self.bundle / "policy.json",
                              self.bundle / "sealed-evaluation-labels.json", self.ledger_path, self.output, study)

    def rewrite(self, path, value):
        Path(path).write_text(json.dumps(value, allow_nan=False), encoding="utf-8")

    def native_spec(self, evidence_kind="synthetic"):
        checkpoints = {"reference": {"checkpoint": 0, "classes": self.manifest["old_classes"], "dataset": "synthetic-fixture", "seed": 11},
                       "old": {"checkpoint": 0, "classes": self.manifest["old_classes"], "dataset": "synthetic-fixture", "seed": 11},
                       "new": {"checkpoint": 1, "classes": self.manifest["classes"], "dataset": "synthetic-fixture", "seed": 11}}
        spec = {"schema": "native-score-repair-adapter-v1", "increment": 1, "probability_margin_tolerance": 1e-6, "partitions": {}}
        for role in ("trigger", "fit", "acceptance", "subsequent"):
            part = read_json(self.bundle / f"{role}.json")
            folder = self.root / f"export-{role}"
            folder.mkdir()
            values = {"group_ids": np.asarray(part["group_ids"])}
            for when in ("old", "new"):
                h = np.asarray(part[f"{when}_head_logits"], dtype=np.float64)
                values.update({f"{when}_head": sigmoid(h).astype(np.float32), f"{when}_head_logits": h,
                               f"{when}_raw": np.asarray(part[f"{when}_router_raw"], np.float32),
                               f"{when}_row_ids": np.asarray(part["row_ids"])})
            arrays = {}
            for name, value in values.items():
                path = folder / f"{name}.npy"
                np.save(path, value, allow_pickle=False)
                arrays[name] = {"path": path.name, "sha256": file_hash(path)}
            receipt = {"schema": "native-ofra-score-pair-v1", "status": "COMPLETE", "evidence_kind": evidence_kind,
                       "labels_read": False, "attributions_computed": False, "rows": len(part["row_ids"]),
                       "arrays": arrays, "checkpoints": checkpoints, "manifest_sha256": "a" * 64,
                       "historical_gpu_fidelity_verified": False, "training_lineage_verified": False}
            self.rewrite(folder / "EXPORT_RECEIPT.json", receipt)
            receipt_sha = file_hash(folder / "EXPORT_RECEIPT.json")
            self.rewrite(folder / "COMPLETE.json", {"status": "COMPLETE", "receipt_sha256": receipt_sha, "arrays": arrays})
            origin = "prospective-heldout" if role == "subsequent" else "increment-available-training"
            labels = part.get("labels", read_json(self.bundle / "sealed-evaluation-labels.json")["labels"])
            self.rewrite(folder / "labels.json", {"row_ids": part["row_ids"], "group_ids": part["group_ids"], "labels": labels, "origin": origin})
            spec["partitions"][role] = {"receipt": {"path": f"export-{role}/EXPORT_RECEIPT.json", "sha256": receipt_sha},
                                        "labels": {"path": f"export-{role}/labels.json", "sha256": file_hash(folder / "labels.json")},
                                        "origin": origin, "observed_increment": 2 if role == "subsequent" else 1,
                                        "labels_available_increment": None if role == "subsequent" else 1}
        self.rewrite(self.root / "adapter-spec.json", spec)
        return spec

    def bound_external_fixture(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["source_kind"] = "externally-verified-score-bundle"
        binding = source_binding(manifest)
        for field, schema, status in (("score_fidelity_evidence", "score-fidelity-evidence-v1", "PASSED"),
                                       ("data_governance_evidence", "data-governance-evidence-v1", "APPROVED")):
            report = {"schema": "external-research-review-report-v1", "status": status, "source_binding_sha256": binding,
                      "review_id": "SYNTHETIC-TEST-NOT-A-REAL-REVIEW", "fusion_sha256": digest(self.policy["fusion"]),
                      "numerical_fidelity_verified": True, "training_lineage_verified": True,
                      "semantics_sha256": digest(self.policy["semantics"]), "decision_partition_source": "increment-available-training",
                      "official_test_used_for_decisions": False}
            self.rewrite(self.bundle / f"{field}-report.json", report)
            evidence = {"schema": schema, "status": status, "source_binding_sha256": binding,
                        "report": {"file": f"{field}-report.json", "sha256": file_hash(self.bundle / f"{field}-report.json")}}
            self.rewrite(self.bundle / f"{field}.json", evidence)
            manifest[field] = {"file": f"{field}.json", "sha256": file_hash(self.bundle / f"{field}.json")}
        return manifest

    def test_strict_json_rejects_duplicate_nonfinite_and_overflow(self):
        path = self.root / "invalid.json"
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}', '{"x":1e400}'):
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(Blocked):
                read_json(path)

    def test_bound_paths_reject_traversal_and_simulated_reparse(self):
        with self.assertRaises(Blocked):
            bound_path(self.bundle, "../other.json")
        actual_lstat = os.lstat
        target = self.bundle / "trigger.json"
        def marked(path, *args, **kwargs):
            value = actual_lstat(path, *args, **kwargs)
            if Path(path) == target:
                return SimpleNamespace(st_mode=value.st_mode, st_file_attributes=1024)
            return value
        with patch("runner.os.lstat", side_effect=marked):
            with self.assertRaisesRegex(Blocked, "reparse"):
                bound_path(self.bundle, "trigger.json")

    def test_root_internal_symlink_is_rejected(self):
        link = self.bundle / "linked.json"
        try:
            link.symlink_to(self.bundle / "trigger.json")
        except OSError as exc:
            if os.name != "nt":
                self.skipTest(f"OS does not permit a real symlink fixture: {type(exc).__name__}")
            # Windows junction creation does not need symlink privilege. Both
            # ends are checked descendants of this test's own temporary root.
            target = self.root / "junction-target"
            target.mkdir()
            (target / "value.json").write_text("{}", encoding="utf-8")
            link = self.bundle / "linked-dir"
            self.assertTrue(target.resolve().is_relative_to(self.root) and link.parent.resolve().is_relative_to(self.root))
            created = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if created.returncode:
                self.skipTest("OS denied both symlink and local junction test fixtures")
            with self.assertRaisesRegex(Blocked, "symlink|reparse"):
                bound_path(self.bundle, "linked-dir/value.json")
        else:
            with self.assertRaisesRegex(Blocked, "symlink"):
                bound_path(self.bundle, "linked.json")

    def test_digest_strings_do_not_authorize_real_score_bundle(self):
        manifest = copy.deepcopy(self.manifest)
        manifest.update(source_kind="externally-verified-score-bundle", score_fidelity_evidence_sha256="a" * 64,
                        data_governance_evidence_sha256="b" * 64)
        with self.assertRaises(Blocked):
            verify_real_evidence(self.bundle, manifest, self.policy)

    def test_real_external_file_schema_status_source_and_report_are_bound(self):
        manifest = self.bound_external_fixture()
        verify_real_evidence(self.bundle, manifest, self.policy)  # format gate on fabricated fixtures only
        changed = copy.deepcopy(manifest)
        changed["partitions"]["trigger"]["sha256"] = "b" * 64
        with self.assertRaises(Blocked):
            verify_real_evidence(self.bundle, changed, self.policy)
        evidence_path = self.bundle / manifest["score_fidelity_evidence"]["file"]
        value = read_json(evidence_path)
        value["status"] = "PENDING"
        self.rewrite(evidence_path, value)
        manifest["score_fidelity_evidence"]["sha256"] = file_hash(evidence_path)
        with self.assertRaises(Blocked):
            verify_real_evidence(self.bundle, manifest, self.policy)

    def test_external_underlying_report_mutation_rejected(self):
        manifest = self.bound_external_fixture()
        self.rewrite(self.bundle / "score_fidelity_evidence-report.json", {"schema": "changed"})
        with self.assertRaises(Blocked):
            verify_real_evidence(self.bundle, manifest, self.policy)

    def test_native_adapter_converts_four_receipts_and_seals_labels(self):
        self.native_spec()
        manifest = adapt(self.root / "adapter-spec.json", self.root / "adapted")
        self.assertEqual(manifest["source_kind"], "synthetic-software-test")
        self.assertNotIn("labels", read_json(self.root / "adapted/subsequent.json"))
        self.assertIn("labels", read_json(self.root / "adapted/sealed-evaluation-labels.json"))
        self.assertNotIn("explanations", read_json(self.root / "adapted/trigger.json"))
        self.assertFalse(manifest["historical_gpu_fidelity_verified_by_adapter"])

    def test_real_native_complete_is_still_pending_fidelity(self):
        self.native_spec(evidence_kind="real")
        manifest = adapt(self.root / "adapter-spec.json", self.root / "adapted")
        self.assertEqual(manifest["source_kind"], "score-bundle-pending-fidelity")
        with self.assertRaises(Blocked):
            verify_real_evidence(self.root / "adapted", manifest, self.policy)

    def test_official_test_cannot_be_a_decision_partition(self):
        spec = self.native_spec()
        spec["partitions"]["fit"]["origin"] = "retrospective-official-test"
        self.rewrite(self.root / "adapter-spec.json", spec)
        with self.assertRaisesRegex(Blocked, "official test"):
            adapt(self.root / "adapter-spec.json", self.root / "adapted")
        self.assertFalse((self.root / "adapted").exists())

    def test_native_raw_score_file_hash_mutation_rejected(self):
        self.native_spec()
        np.save(self.root / "export-trigger/new_raw.npy", np.zeros((1, 3)), allow_pickle=False)
        with self.assertRaisesRegex(Blocked, "hash mismatch"):
            adapt(self.root / "adapter-spec.json", self.root / "adapted")

    def test_complete_cpu_decide_evaluate_for_all_controls(self):
        result = self.run_decisions()
        self.assertEqual(set(result["decisions"]), set(ARMS))
        self.assertEqual(result["stage"], "DECISIONS_LOCKED")
        for arm in ("random", "periodic"):
            self.assertEqual(result["decisions"][arm]["state"]["steps_completed"], 24)
        evaluated = self.run_evaluation()
        self.assertEqual(evaluated["stage"], "EVALUATED_ONCE")
        self.assertEqual(set(evaluated["arms"]), set(ARMS))
        self.assertFalse(evaluated["real_data_utility_established"])

    def test_all_active_arms_reach_actual_fit_on_controlled_score_fixture(self):
        # Mathematical branch fixture only: exact wrong old-a scores and exact
        # explanation-vector changes, not sampled real SHAP or an efficacy run.
        groups = 128
        for role in ("trigger", "fit", "acceptance", "subsequent"):
            part, classes = mathematical_partition(groups)
            part["role"] = role
            part["row_ids"] = [f"{role}-{x}" for x in part["row_ids"]]
            part["group_ids"] = [f"{role}-{x}" for x in part["group_ids"]]
            labels = part["labels"]
            part["old_head_logits"] = [[3, -3] if y == classes[0] else [-3, 3] for y in labels]
            part["old_router_raw"] = [[0, 0] for _ in labels]
            if role == "trigger":
                part["explanations"] = {c: {"method": self.policy["trigger"]["explanation_method"],
                                            "independent_draws_by_group": True,
                                            "A": np.tile([1., 0.], (4, len(labels), 1)).tolist(),
                                            "D": np.tile([0., 1.], (4, len(labels), 1)).tolist()}
                                        for c in classes[:2]}
            if role == "subsequent":
                sealed = {"row_ids": part["row_ids"], "group_ids": part["group_ids"], "labels": labels, "observed_increment": 2}
                self.rewrite(self.bundle / "sealed-evaluation-labels.json", sealed)
                self.manifest["sealed_labels_sha256"] = file_hash(self.bundle / "sealed-evaluation-labels.json")
                del part["labels"]
            self.rewrite(self.bundle / f"{role}.json", part)
            self.manifest["partitions"][role].update(sha256=file_hash(self.bundle / f"{role}.json"),
                                                    row_ids=part["row_ids"], group_ids=part["group_ids"])
        self.rewrite(self.bundle / "manifest.json", self.manifest)
        self.rewrite(self.bundle / "policy.json", prospective_policy(groups, groups * 3))
        result = self.run_decisions()
        for arm in ARMS:
            record = result["decisions"][arm]
            if arm != "audit-only":
                self.assertEqual(record["repairs_attempted"], 1, arm)
                self.assertEqual(record["state"]["steps_completed"], 24, arm)
            else:
                self.assertEqual(record["repairs_attempted"], 0)
        self.assertEqual(self.run_evaluation()["stage"], "EVALUATED_ONCE")

    def test_sealed_labels_are_not_opened_before_all_decisions(self):
        ledger = Ledger(self.ledger_path)
        ledger.register("test-study", self.manifest, self.policy)
        ledger.close()
        actual_hash = runner.file_hash
        def guarded_hash(path):
            self.assertNotEqual(Path(path).name, "sealed-evaluation-labels.json", "premature label file access")
            return actual_hash(path)
        with patch("runner.file_hash", side_effect=guarded_hash):
            with self.assertRaises(Blocked):
                self.run_evaluation()

    def test_decide_succeeds_without_evaluation_label_file(self):
        sealed = self.bundle / "sealed-evaluation-labels.json"
        moved = self.bundle / "still-sealed.json"
        sealed.rename(moved)
        result = self.run_decisions()
        self.assertEqual(result["stage"], "DECISIONS_LOCKED")

    def test_restart_cannot_repeat_decision_or_evaluation(self):
        self.run_decisions()
        with self.assertRaises(Blocked):
            self.run_decisions()
        self.run_evaluation()
        with self.assertRaises(Blocked):
            self.run_evaluation()

    def test_cross_increment_row_and_group_reuse_persists(self):
        ledger = Ledger(self.ledger_path)
        ledger.register("first", self.manifest, self.policy)
        ledger.close()
        changed = copy.deepcopy(self.manifest)
        changed["increment"] = 2
        changed["partitions"]["subsequent"]["observed_increment"] = 3
        ledger = Ledger(self.ledger_path)
        with self.assertRaisesRegex(Blocked, "cross-increment"):
            ledger.register("second", changed, self.policy)
        ledger.close()

    def test_group_leakage_and_future_label_metadata_rejected(self):
        changed = copy.deepcopy(self.manifest)
        changed["partitions"]["fit"]["group_ids"][0] = changed["partitions"]["trigger"]["group_ids"][0]
        with self.assertRaises(Blocked):
            validate_manifest(changed)
        changed = copy.deepcopy(self.manifest)
        changed["partitions"]["fit"]["labels_available_increment"] = 2
        with self.assertRaises(Blocked):
            validate_manifest(changed)

    def test_independent_score_path_exact_and_missing_inputs_explicit(self):
        part = load_partition(self.bundle, self.manifest, "trigger")
        path = score_path(part, self.manifest["classes"], self.manifest["old_classes"], self.manifest["old_classes"][0], self.policy["fusion"])
        self.assertLessEqual(path["identity_max_abs"], 1e-12)
        self.assertTrue((path["effects"]["new_rival"] <= 1e-12).all())
        part.pop("old_head_logits")
        with self.assertRaisesRegex(Blocked, "unavailable"):
            score_path(part, self.manifest["classes"], self.manifest["old_classes"], self.manifest["old_classes"][0], self.policy["fusion"])

    def test_simple_controls_have_no_attribution_dependency(self):
        part = load_partition(self.bundle, self.manifest, "trigger")
        part.pop("explanations")
        for arm in ("random", "periodic", "error-only", "disagreement-only"):
            pool = candidate_pool(part, self.manifest, self.policy, arm)
            self.assertFalse(any("attribution" in (c["reason"] or "") for c in pool["candidates"]))
        pool = candidate_pool(part, self.manifest, self.policy, "expansion-aware")
        self.assertTrue(all("attribution" in c["reason"] for c in pool["candidates"]))

    def test_malaya_application_cannot_gain_attack_metrics(self):
        changed = copy.deepcopy(self.policy)
        changed["semantics"]["attack_classes"] = ["application-a"]
        with self.assertRaises(Blocked):
            validate_policy(changed, self.manifest)

    def test_policy_loosened_after_fit_blocks_evaluation_before_labels(self):
        self.run_decisions()
        policy = read_json(self.bundle / "policy.json")
        policy["acceptance"]["min_target_improvement"] = 0
        self.rewrite(self.bundle / "policy.json", policy)
        with self.assertRaisesRegex(Blocked, "binding changed"):
            self.run_evaluation()

    def test_forged_candidate_pool_cannot_bypass_reserved_eligibility(self):
        ledger = Ledger(self.ledger_path)
        try:
            ledger.register("study", self.manifest, self.policy)
            part = load_partition(self.bundle, self.manifest, "trigger")
            pool = candidate_pool(part, self.manifest, self.policy, "expansion-aware")
            self.assertFalse(pool["ordered_targets"])
            pool["ordered_targets"] = [self.manifest["old_classes"][0]]
            ledger.save_pool("study", "expansion-aware", pool)
            with self.assertRaisesRegex(Blocked, "cannot be bypassed"):
                ledger.reserve("study", "expansion-aware", pool, self.manifest, self.policy, self.bundle)
        finally:
            ledger.close()

    def test_trigger_file_mutation_after_pool_freeze_blocks_reservation(self):
        ledger = Ledger(self.ledger_path)
        try:
            ledger.register("study", self.manifest, self.policy)
            part = load_partition(self.bundle, self.manifest, "trigger")
            pool = candidate_pool(part, self.manifest, self.policy, "random")
            ledger.save_pool("study", "random", pool)
            part["labels"][0] = "application-b"
            self.rewrite(self.bundle / "trigger.json", part)
            with self.assertRaisesRegex(Blocked, "hash mismatch"):
                ledger.reserve("study", "random", pool, self.manifest, self.policy, self.bundle)
        finally:
            ledger.close()

    def test_real_optimizer_failure_retains_step_and_budget_rollback(self):
        actual = runner.fit_r1
        def injected_failure(*args):
            original_progress = args[-1]
            def stop_after_step(steps, wall, cpu):
                original_progress(steps, wall, cpu)
                raise RuntimeError("deliberate software-test failure after one real optimizer step")
            return actual(*args[:-1], progress=stop_after_step)
        with patch("runner.fit_r1", side_effect=injected_failure):
            result = self.run_decisions()
        for arm in ("random", "periodic"):
            record = result["decisions"][arm]
            self.assertEqual(record["decision"], "FAILED_ROLLBACK")
            self.assertTrue(record["reserved_budget_retained"])
            self.assertEqual(record["last_durable_progress"]["steps_completed"], 1)
        evaluated = self.run_evaluation()
        self.assertTrue(all(x["decision"] != "ACCEPT_RESEARCH_VARIANT" for x in evaluated["arms"].values()))

    def test_insufficient_budget_does_not_train(self):
        policy = copy.deepcopy(self.policy)
        policy["budget"]["unique_labels_per_arm"] = 1
        self.rewrite(self.bundle / "policy.json", policy)
        with patch("runner.fit_r1", side_effect=AssertionError("optimizer must not run over budget")):
            result = self.run_decisions()
        self.assertEqual(result["decisions"]["random"]["decision"], "FAILED_ROLLBACK")
        self.assertFalse(result["decisions"]["random"]["reserved_budget_retained"])

    def test_audit_only_has_no_repair_and_unused_budget(self):
        result = self.run_decisions()["decisions"]["audit-only"]
        self.assertEqual(result["repairs_attempted"], 0)
        self.assertEqual(result["unused_repair_steps"], 24)
        self.assertNotIn("state", result)

    def test_partition_hash_failure_is_persistently_failed_closed(self):
        part = read_json(self.bundle / "trigger.json")
        part["labels"][0] = "application-b"
        self.rewrite(self.bundle / "trigger.json", part)
        with self.assertRaises(Blocked):
            self.run_decisions()
        ledger = Ledger(self.ledger_path)
        self.assertEqual(ledger.study("test-study")["stage"], "FAILED_CLOSED")
        ledger.close()

    def test_invalid_evaluation_labels_spend_access_and_cannot_retry(self):
        self.run_decisions()
        self.rewrite(self.bundle / "sealed-evaluation-labels.json", {"labels": []})
        with self.assertRaises(Blocked):
            self.run_evaluation()
        ledger = Ledger(self.ledger_path)
        self.assertEqual(ledger.study("test-study")["stage"], "EVALUATION_FAILED_SPENT")
        ledger.close()
        with self.assertRaises(Blocked):
            self.run_evaluation()


if __name__ == "__main__":
    unittest.main()
