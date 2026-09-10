"""Tiny synthetic models and temporary ledgers; no research data or checkpoints."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from native_target import (CallBudget, FixedContextTarget, array_sha, capture_native,
                           permutation_repeats, score_parent_shards, validate_native)
from native_r1 import adjust, fit
from native_loader import safe_file, sha, verify_runtime
from one_use import Ledger, derive_candidates, run
from pilot_core import ARMS, ROLES, digest

SETTINGS = dict(scale_min=.5, scale_max=2., offset_min=-2., offset_max=2.,
                steps=5, learning_rate=.1, identity_l2=.01, max_seconds=20.)


def scores_for(pred, axis=(0, 1, 2, 3)):
    pred = np.asarray(pred)
    p = np.full((len(pred), len(axis)), .1, dtype=np.float32)
    for i, c in enumerate(pred):
        p[i, list(axis).index(c)] = .8
    z = np.zeros_like(p)
    logits = np.zeros((len(p), len(axis), 2), dtype=np.float32)
    logits[:, :, 1] = np.log(p/(1-p)).astype(np.float32)
    return dict(class_axis=np.array(axis, dtype=np.int64), head_scores=p,
                router_z_scores=z, joint_scores=p.copy(), predicted_class_id=pred.copy(),
                binary_logits=logits)


def linear_scores(raw):
    # Batch membership affects the synthetic target, deliberately.
    p0 = .4 + .02*raw[:, 0] + .01*raw[:, 1] + .01*raw[:, 2] + .01*raw[:, 0].mean()
    p = np.stack([p0, np.full(len(raw), .3)], axis=1).astype(np.float32)
    z = np.zeros_like(p)
    axis = np.array([0, 1], dtype=np.int64)
    return dict(class_axis=axis, head_scores=p, router_z_scores=z,
                joint_scores=p.copy(), predicted_class_id=axis[p.argmax(1)])


class TargetTests(unittest.TestCase):
    def test_context_companions_and_original_are_preserved(self):
        parent = np.array([[1, 2, 1], [3, 1, 1], [2, 2, 2]], dtype=np.float32)
        calls = []
        def recorder(x):
            calls.append(x.copy()); return linear_scores(x)
        target = FixedContextTarget(recorder, parent, 1, [0, 1], 0, CallBudget(10, 10))
        replacements = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        actual = target(replacements)
        self.assertEqual(len(calls), 2)
        for a in calls:
            np.testing.assert_array_equal(a[[0, 2]], parent[[0, 2]])
        wrong = linear_scores(replacements)["joint_scores"]
        self.assertFalse(np.array_equal(actual, wrong[:, 0]-wrong[:, 1]))
        np.testing.assert_array_equal(parent, target.parent)

    def test_full_parent_shards_not_concatenated_or_prefiltered(self):
        parents = {0: [np.ones((513, 3), np.float32), np.ones((2, 3), np.float32)],
                   1: [np.ones((3, 3), np.float32)]}
        calls = []
        def record(x):
            calls.append(len(x)); return linear_scores(x)
        selected = {(0, 0, 512), (0, 1, 1), (1, 0, 2)}
        output, boundaries = score_parent_shards(record, parents, [0, 1], selected, CallBudget(10, 10))
        self.assertEqual(calls, [512, 1, 2, 3])
        self.assertEqual(set(output), selected)
        self.assertEqual(boundaries, [[0, 0, 0, 512], [0, 0, 512, 1], [0, 1, 0, 2], [1, 0, 0, 3]])

    def test_out_of_range_selection_blocks(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            score_parent_shards(linear_scores, {0: [np.ones((2, 3), np.float32)]}, [0, 1], {(0, 0, 8)}, CallBudget(10, 10))

    def test_failed_native_call_is_spent(self):
        budget = CallBudget(1, 10)
        def fail(x):
            raise RuntimeError("fixture")
        with self.assertRaises(RuntimeError):
            budget.run(fail, np.ones((3, 3), np.float32))
        self.assertEqual((budget.calls, budget.context_rows), (1, 3))
        with self.assertRaisesRegex(ValueError, "exhausted"):
            budget.run(linear_scores, np.ones((3, 3), np.float32))

    def test_scorer_cannot_mutate_inputs(self):
        def mutates(x):
            x[:] = 0
            return linear_scores(x)
        target = FixedContextTarget(mutates, np.ones((2, 3), np.float32), 0, [0, 1], 0, CallBudget(10, 10))
        with self.assertRaisesRegex(ValueError, "modified"):
            target(np.ones((1, 3), np.float32))

    def test_float64_score_surrogate_rejected(self):
        result = linear_scores(np.ones((2, 3), np.float32))
        result["joint_scores"] = result["joint_scores"].astype(np.float64)
        with self.assertRaisesRegex(ValueError, "float32"):
            validate_native(result, 2, [0, 1])

    def test_wrong_axis_and_changed_argmax_rejected(self):
        for field, bad in [("class_axis", np.array([1, 0])), ("predicted_class_id", np.ones(2, dtype=int))]:
            result = linear_scores(np.ones((2, 3), np.float32)); result[field] = bad
            with self.assertRaises(ValueError):
                validate_native(result, 2, [0, 1])

    def test_context_over512_rejected(self):
        with self.assertRaises(ValueError):
            FixedContextTarget(linear_scores, np.ones((513, 3), np.float32), 0, [0, 1], 0, CallBudget(10, 10))

    def test_masker_promotion_must_be_lossless(self):
        target = FixedContextTarget(linear_scores, np.ones((2, 3), np.float32), 0, [0, 1], 0, CallBudget(10, 10))
        x = np.ones((1, 3), np.float32)
        np.testing.assert_array_equal(target(x), target(x.astype(np.float64)))
        bad = x.astype(np.float64); bad[0, 0] += 1e-10
        with self.assertRaisesRegex(ValueError, "losslessly"):
            target(bad)

    def test_real_shap_package_on_tiny_synthetic_callable(self):
        parent = np.array([[1, 2, 1], [3, 1, 1], [2, 2, 2]], dtype=np.float32)
        target = FixedContextTarget(linear_scores, parent, 1, [0, 1], 0, CallBudget(200, 120))
        refs = np.zeros((1, 3), np.float32)
        result = permutation_repeats(target, refs, [11, 12, 13, 14])
        expected = np.array([(.02+.01/3)*3, .01, .01])
        np.testing.assert_allclose(result["attributions"], np.tile(expected, (4, 1)), atol=1e-7)
        self.assertLess(max(map(abs, result["residuals"])), 1e-7)
        self.assertFalse(result["statistical_certificate"])
        self.assertLessEqual(result["native_calls"], 200)


    def test_near_equal_float32_values_are_not_skipped_by_masker(self):
        def threshold(raw):
            h = np.stack([np.where(raw[:, 0] > 1, .8, .2), np.full(len(raw), .1)], 1).astype(np.float32)
            return dict(class_axis=np.array([0, 1]), head_scores=h, router_z_scores=np.zeros_like(h),
                        joint_scores=h.copy(), predicted_class_id=h.argmax(1))
        parent = np.array([[np.nextafter(np.float32(1), np.float32(2))]], np.float32)
        target = FixedContextTarget(threshold, parent, 0, [0, 1], 0, CallBudget(50, 120))
        result = permutation_repeats(target, np.array([[1]], np.float32), [1, 2, 3, 4])
        self.assertGreater(float(result["attributions"].min()), .5)

    def test_native_capture_reuses_forward_and_removes_hooks_on_failure(self):
        import torch
        class Model:
            metadata = {"seen_classes": [0, 1]}
            def __init__(self):
                self.heads = {c: torch.nn.Linear(3, 2) for c in [0, 1]}
                self.fail = False
            def score(self, raw):
                with torch.no_grad():
                    logits = [self.heads[c](torch.from_numpy(raw)) for c in [0, 1]]
                    if self.fail:
                        raise RuntimeError("capture failure fixture")
                    h = np.stack([torch.softmax(v, 1)[:, 1].numpy() for v in logits], 1)
                return dict(class_axis=np.array([0, 1]), head_scores=h, router_z_scores=np.zeros_like(h),
                            joint_scores=h.copy(), predicted_class_id=h.argmax(1))
        model = Model(); raw = np.ones((2, 3), np.float32)
        result = capture_native(model, raw)
        for j in [0, 1]:
            np.testing.assert_array_equal(torch.softmax(torch.from_numpy(result["binary_logits"][:, j]), 1)[:, 1].numpy(), result["head_scores"][:, j])
        self.assertTrue(all(not h._forward_hooks for h in model.heads.values()))
        model.fail = True
        with self.assertRaises(RuntimeError):
            capture_native(model, raw)
        self.assertTrue(all(not h._forward_hooks for h in model.heads.values()))


class R1Tests(unittest.TestCase):
    def test_identity_preserves_native_bytes_even_with_rounded_probabilities(self):
        native = scores_for([0, 1, 2, 3])
        # Deliberately not reconstructible from supplied logits: detect bypass.
        native["head_scores"][0, 0] = np.nextafter(native["head_scores"][0, 0], np.float32(1))
        native["joint_scores"] = native["head_scores"].copy()
        actual = adjust(native, 0, [1., 0.])
        self.assertEqual(array_sha(actual), array_sha(native["joint_scores"]))

    def test_only_target_column_changes(self):
        native = scores_for([0, 1, 2, 3]); actual = adjust(native, 1, [1.1, .3])
        np.testing.assert_array_equal(actual[:, [0, 2, 3]], native["joint_scores"][:, [0, 2, 3]])
        self.assertEqual(actual.dtype, np.float32)

    def test_fit_is_bounded_and_fixed_step(self):
        native = scores_for([1, 1, 2, 3])
        before = array_sha(native["joint_scores"])
        result = fit(native, [0, 1, 2, 3], 0, SETTINGS, "fit")
        self.assertEqual(result["steps"], 5)
        self.assertTrue(.5 <= result["params"][0] <= 2 and -2 <= result["params"][1] <= 2)
        self.assertEqual(before, array_sha(native["joint_scores"]))

    def test_non_fit_labels_and_missing_class_rejected(self):
        native = scores_for([0, 1, 2, 3])
        for labels, role in [([0, 1, 2, 3], "acceptance"), ([0, 1, 2, 2], "fit")]:
            with self.assertRaises(ValueError):
                fit(native, labels, 0, SETTINGS, role)


def identities(prefix):
    return {role: [digest([prefix, role, i]) for i in range(8)] for role in ROLES}


class ToyLoader:
    def __init__(self, ledger, study, roles, harmful=False, fail_eval=False):
        self.ledger, self.study, self.roles = ledger, study, roles
        self.harmful, self.fail_eval = harmful, fail_eval
        self.calls = []

    def data(self, role):
        y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        return dict(native=scores_for(y), labels=y, row_ids=self.roles[role])

    def trigger(self):
        self.calls.append("trigger"); assert self.ledger.stage(self.study) == "CLAIMED"
        data = self.data("trigger")
        old = scores_for([0, 0, 1, 1, 1, 1, 1, 1], [0, 1])
        data["old_native"] = old
        if self.harmful:
            data["native"] = scores_for([1, 1, 1, 1, 2, 2, 3, 3])
        return data

    def fit(self):
        self.calls.append("fit"); assert self.ledger.stage(self.study) == "FIT_ATTEMPTS_SPENT"
        return self.data("fit")

    def acceptance(self):
        self.calls.append("acceptance"); assert self.ledger.stage(self.study) == "ACCEPTANCE_SPENT"
        return self.data("acceptance")

    def evaluation(self):
        self.calls.append("evaluation"); assert self.ledger.stage(self.study) == "EVALUATION_SPENT"
        if self.fail_eval:
            raise RuntimeError("evaluation read failure fixture")
        return self.data("evaluation")


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "ledger.sqlite"
        self.ledger = Ledger(self.path)
        self.roles = identities("source")

    def tearDown(self):
        self.ledger.close(); self.tmp.cleanup()

    def test_no_harm_uses_no_fit_or_acceptance_labels(self):
        loader = ToyLoader(self.ledger, "test", self.roles)
        result = run(self.ledger, "test", "a"*64, self.roles, loader, SETTINGS, "fixed", "synthetic")
        self.assertEqual(loader.calls, ["trigger", "evaluation"])
        self.assertEqual(self.ledger.stage("test"), "COMPLETE")
        self.assertEqual(set(result["arms"]), set(ARMS))
        self.assertFalse(result["full_test_population_evaluated"])
        self.assertFalse(result["upstream_attribution_cost_included"])

    def test_ordering_and_abstention_when_attribution_missing(self):
        loader = ToyLoader(self.ledger, "test", self.roles, harmful=True)
        result = run(self.ledger, "test", "a"*64, self.roles, loader, SETTINGS, "fixed", "synthetic")
        self.assertEqual(loader.calls, ["trigger", "fit", "acceptance", "evaluation"])
        self.assertEqual(result["arms"]["error-only"]["attempts_spent"], 1)
        self.assertEqual(result["arms"]["noise-aware"]["decision"]["status"], "unavailable-attribution")
        self.assertIsNone(result["noise_aware_minus_error_only_old_error"])
        self.assertEqual(result["primary_contrast_status"], "UNAVAILABLE_ATTRIBUTION")

    def test_failed_evaluation_terminal_and_not_retryable(self):
        loader = ToyLoader(self.ledger, "test", self.roles, fail_eval=True)
        with self.assertRaises(RuntimeError):
            run(self.ledger, "test", "a"*64, self.roles, loader, SETTINGS, "fixed", "synthetic")
        self.assertEqual(self.ledger.stage("test"), "FAILED")
        with self.assertRaises(Exception):
            run(self.ledger, "test", "a"*64, self.roles, loader, SETTINGS, "fixed", "synthetic")
        self.assertEqual(loader.calls.count("evaluation"), 1)

    def test_global_row_claim_survives_restart_and_new_study_name(self):
        self.ledger.claim("first", "a"*64, self.roles)
        self.ledger.close(); self.ledger = Ledger(self.path)
        with self.assertRaises(Exception):
            self.ledger.claim("renamed", "b"*64, self.roles)
        self.assertEqual(self.ledger.stage("first"), "CLAIMED")
        with self.assertRaises(ValueError):
            self.ledger.stage("renamed")

    def test_no_early_evaluation_or_stage_skip(self):
        self.ledger.claim("first", "a"*64, self.roles)
        with self.assertRaises(ValueError):
            self.ledger.advance("first", "CLAIMED", "EVALUATION_SPENT", {})
        self.assertEqual(self.ledger.stage("first"), "CLAIMED")

    def test_role_overlap_is_not_registered(self):
        roles = identities("same"); roles["fit"] = roles["trigger"]
        with self.assertRaises(ValueError):
            self.ledger.claim("first", "a"*64, roles)

    def test_tampered_stage_row_order_fails_before_fit(self):
        loader = ToyLoader(self.ledger, "test", self.roles, harmful=True)
        original = loader.fit
        def invalid():
            data = original(); data["row_ids"] = list(reversed(data["row_ids"])); return data
        loader.fit = invalid
        with self.assertRaises(ValueError):
            run(self.ledger, "test", "a"*64, self.roles, loader, SETTINGS, "fixed", "synthetic")
        self.assertEqual(self.ledger.stage("test"), "FAILED")
        self.assertNotIn("acceptance", loader.calls)

    def test_empty_existing_database_not_silently_reset(self):
        path = Path(self.tmp.name)/"empty.sqlite"; path.write_bytes(b"")
        with self.assertRaises(ValueError):
            Ledger(path)

    def test_callback_cannot_reassign_already_claimed_rows(self):
        loader = ToyLoader(self.ledger, "test", self.roles, harmful=True)
        original = loader.trigger
        def mutate():
            result = original()
            self.roles["fit"] = list(reversed(self.roles["fit"]))
            return result
        loader.trigger = mutate
        with self.assertRaisesRegex(ValueError, "identity/order"):
            run(self.ledger, "test", "a"*64, self.roles, loader, SETTINGS, "fixed", "synthetic")
        self.assertEqual(self.ledger.stage("test"), "FAILED")
        self.assertNotIn("acceptance", loader.calls)

    def test_historical_loader_pins_before_import(self):
        root = Path(self.tmp.name)
        fixture = root / "source.py"; fixture.write_bytes(b"raise RuntimeError('do not import')\n")
        pins = [{"path": "source.py", "sha256": sha(fixture)}]
        verify_runtime(root, pins)
        fixture.write_bytes(b"raise RuntimeError('changed')\n")
        with self.assertRaisesRegex(ValueError, "runtime hash"):
            verify_runtime(root, pins)
        with self.assertRaisesRegex(ValueError, "non-relative"):
            safe_file(root, "../outside.py")


if __name__ == "__main__":
    unittest.main()
