"""Small synthetic-only tests; no real data, disk output, or training."""

from dataclasses import FrozenInstanceError, replace
import hashlib
import unittest

from prospective_sampling import (
    MAX_REFERENCE_ROWS,
    Policy,
    RowRef,
    TrainBatch,
    append_increment,
    initial_state,
)


def batch(class_id, count, increment=0, role="attack"):
    rows = tuple(RowRef(f"class-{class_id}-row-{i:05d}", increment) for i in range(count))
    digest = hashlib.sha256("|".join(row.row_id for row in rows).encode("ascii")).hexdigest()
    return TrainBatch(class_id, role, increment, rows, digest)


def task0(policy=None):
    return append_increment(
        initial_state(policy or Policy()),
        increment=0,
        batches=(batch(0, 100, role="normal"), batch(1, 20)),
    )


def selection_ids(state):
    return tuple(
        (item.class_id, item.fit_ids, item.calibration_ids, item.omitted_ids)
        for item in state.selections
    )


class ProspectiveSamplingTests(unittest.TestCase):
    def test_task0_cap_uses_only_available_attack_fitting_rows(self):
        state = task0()
        self.assertEqual(state.frozen_normal_cap, 18)
        normal, attack = state.selections
        self.assertEqual((len(normal.fit_ids), len(normal.calibration_ids), len(normal.omitted_ids)), (18, 10, 72))
        self.assertEqual((len(attack.fit_ids), len(attack.calibration_ids)), (18, 2))

    def test_cap_and_earlier_selections_remain_frozen_after_large_future_class(self):
        before = task0()
        after = append_increment(before, increment=1, batches=(batch(2, 1000, 1),))
        self.assertEqual(after.frozen_normal_cap, before.frozen_normal_cap)
        self.assertEqual(after.selections[:2], before.selections)
        self.assertEqual(before.increment, 0)

    def test_prefix_invariance_when_future_class_count_or_identity_changes(self):
        # The selector is called with released cohorts only, not either world.
        prefix = (batch(0, 100, role="normal"), batch(1, 20))
        worlds = (prefix + (batch(2, 12, 1),), prefix + (batch(90, 2000, 1), batch(91, 1, 1)))
        states = []
        for world in worlds:
            released = tuple(item for item in world if item.available_increment == 0)
            states.append(append_increment(initial_state(Policy()), increment=0, batches=released))
        self.assertEqual(states[0], states[1])

    def test_future_and_test_metadata_replacement_does_not_enter_selection(self):
        # Synthetic external registry metadata is never given to the selector.
        prefix = (batch(0, 100, role="normal"), batch(1, 20))
        registries = (
            {"future_class_counts": {2: 12}, "official_test": {"label": "A", "hash": "a" * 64}},
            {"future_class_counts": {99: 9000}, "official_test": {"label": "B", "hash": "b" * 64}},
        )
        outputs = [append_increment(initial_state(Policy()), increment=0, batches=prefix) for _ in registries]
        self.assertEqual(outputs[0], outputs[1])

    def test_full_world_input_is_rejected_not_silently_filtered(self):
        with self.assertRaisesRegex(ValueError, "newly available"):
            append_increment(initial_state(Policy()), increment=0, batches=(batch(0, 10, role="normal"), batch(1, 3), batch(2, 100, 1)))

    def test_input_batch_and_row_order_invariance(self):
        current = (batch(0, 100, role="normal"), batch(1, 20))
        reversed_input = tuple(replace(item, rows=tuple(reversed(item.rows))) for item in reversed(current))
        one = append_increment(initial_state(Policy()), increment=0, batches=current)
        two = append_increment(initial_state(Policy()), increment=0, batches=reversed_input)
        self.assertEqual(one, two)

    def test_same_seed_reproduces_exact_output(self):
        self.assertEqual(task0(Policy(seed=17)), task0(Policy(seed=17)))

    def test_different_seed_changes_synthetic_sampling(self):
        self.assertNotEqual(selection_ids(task0(Policy(seed=17))), selection_ids(task0(Policy(seed=18))))

    def test_train_binding_is_recorded_but_not_a_random_seed(self):
        current = (batch(0, 100, role="normal"), batch(1, 20))
        rebound = tuple(replace(item, source_train_sha256="f" * 64) for item in current)
        one = append_increment(initial_state(Policy()), increment=0, batches=current)
        two = append_increment(initial_state(Policy()), increment=0, batches=rebound)
        self.assertEqual(selection_ids(one), selection_ids(two))
        self.assertNotEqual(one.selections[0].source_train_sha256, two.selections[0].source_train_sha256)

    def test_fit_calibration_omitted_are_disjoint_and_conserve_source_rows(self):
        state = append_increment(task0(), increment=1, batches=(batch(2, 7, 1), batch(3, 2, 1)))
        for item in state.selections:
            fit, calibration, omitted = map(set, (item.fit_ids, item.calibration_ids, item.omitted_ids))
            self.assertFalse(fit & calibration or fit & omitted or calibration & omitted)
            self.assertEqual(fit | calibration | omitted, set(item.input_ids))
            if item.role == "attack":
                self.assertFalse(omitted)

    def test_singleton_retains_one_fit_and_reports_no_calibration(self):
        state = append_increment(task0(), increment=1, batches=(batch(2, 1, 1),))
        item = state.selections[-1]
        self.assertEqual((len(item.fit_ids), len(item.calibration_ids)), (1, 0))
        self.assertIn("singleton_class_no_calibration_support", item.warnings)

    def test_two_rows_have_one_fit_and_one_calibration(self):
        state = append_increment(task0(), increment=1, batches=(batch(2, 2, 1),))
        self.assertEqual((len(state.selections[-1].fit_ids), len(state.selections[-1].calibration_ids)), (1, 1))

    def test_empty_class_has_explicit_no_support_warning(self):
        state = append_increment(task0(), increment=1, batches=(batch(2, 0, 1),))
        self.assertEqual(state.selections[-1].fit_ids, ())
        self.assertIn("empty_class_no_fit_or_calibration_support", state.selections[-1].warnings)

    def test_no_task0_attack_fails_closed_for_task0_cap(self):
        with self.assertRaisesRegex(ValueError, "no Task-0 attack"):
            append_increment(initial_state(Policy()), increment=0, batches=(batch(0, 10, role="normal"),))

    def test_empty_task0_attack_fails_closed_for_task0_cap(self):
        with self.assertRaisesRegex(ValueError, "no Task-0 attack"):
            append_increment(initial_state(Policy()), increment=0, batches=(batch(0, 10, role="normal"), batch(1, 0)))

    def test_predeclared_fixed_cap_handles_no_attack_without_fallback(self):
        state = append_increment(initial_state(Policy(cap_mode="fixed", fixed_cap=4)), increment=0, batches=(batch(0, 10, role="normal"),))
        self.assertEqual(state.frozen_normal_cap, 4)
        self.assertEqual(len(state.selections[0].fit_ids), 4)

    def test_fixed_cap_does_not_change_attack_retention(self):
        state = task0(Policy(cap_mode="fixed", fixed_cap=4))
        self.assertEqual(len(state.selections[0].fit_ids), 4)
        self.assertEqual(len(state.selections[1].fit_ids), 18)

    def test_same_new_rng_cap_control_preserves_calibration_and_attack_rows(self):
        # A larger fixed cap is a synthetic comparator, not a selected optimum.
        control = task0(Policy(cap_mode="fixed", fixed_cap=60))
        prospective = task0()
        self.assertEqual(control.selections[1], prospective.selections[1])
        self.assertEqual(control.selections[0].calibration_ids, prospective.selections[0].calibration_ids)
        self.assertLess(set(prospective.selections[0].fit_ids), set(control.selections[0].fit_ids))

    def test_zero_calibration_fraction(self):
        state = task0(Policy(calibration_numerator=0))
        self.assertEqual(state.frozen_normal_cap, 20)
        self.assertTrue(all(not item.calibration_ids for item in state.selections))

    def test_duplicate_row_within_class_rejected(self):
        item = batch(0, 2, role="normal")
        bad = replace(item, rows=(item.rows[0], item.rows[0]))
        with self.assertRaisesRegex(ValueError, "row_id collision"):
            append_increment(initial_state(Policy()), increment=0, batches=(bad, batch(1, 3)))

    def test_duplicate_row_across_current_classes_rejected(self):
        normal = batch(0, 2, role="normal")
        bad = replace(batch(1, 2), rows=(normal.rows[0],))
        with self.assertRaisesRegex(ValueError, "row_id collision"):
            append_increment(initial_state(Policy()), increment=0, batches=(normal, bad))

    def test_duplicate_row_across_history_rejected(self):
        before = task0()
        duplicate = RowRef(before.selections[0].input_ids[0], 1)
        bad = replace(batch(2, 1, 1), rows=(duplicate,))
        with self.assertRaisesRegex(ValueError, "row_id collision"):
            append_increment(before, increment=1, batches=(bad,))

    def test_duplicate_class_current_rejected(self):
        with self.assertRaisesRegex(ValueError, "class_id collision"):
            append_increment(initial_state(Policy()), increment=0, batches=(batch(0, 2, role="normal"), batch(1, 3), batch(1, 4)))

    def test_historical_class_rewrite_rejected(self):
        with self.assertRaisesRegex(ValueError, "historical class rewrite"):
            append_increment(task0(), increment=1, batches=(batch(1, 3, 1),))

    def test_future_row_rejected_even_inside_current_batch(self):
        bad = replace(batch(1, 1), rows=(RowRef("future-row", 1),))
        with self.assertRaisesRegex(ValueError, "future train row"):
            append_increment(initial_state(Policy()), increment=0, batches=(batch(0, 3, role="normal"), bad))

    def test_test_row_rejected(self):
        for partition in ("test", "official_test", "validation"):
            with self.subTest(partition=partition), self.assertRaisesRegex(ValueError, "source-train rows"):
                RowRef("test-row", 0, partition=partition)

    def test_test_batch_rejected(self):
        with self.assertRaisesRegex(ValueError, "source-train batches"):
            replace(batch(1, 3), partition="test")

    def test_future_metadata_and_test_arguments_not_in_interface(self):
        for extra in ("future_class_counts", "official_test", "source_manifest_sha256"):
            with self.subTest(extra=extra), self.assertRaises(TypeError):
                append_increment(initial_state(Policy()), increment=0, batches=(batch(0, 2, role="normal"),), **{extra: {}})

    def test_missing_or_second_normal_rejected(self):
        for current in ((batch(1, 2),), (batch(0, 3, role="normal"), batch(1, 2, role="normal"))):
            with self.subTest(current=current), self.assertRaisesRegex(ValueError, "normal class"):
                append_increment(initial_state(Policy()), increment=0, batches=current)

    def test_nonconsecutive_increment_rejected(self):
        with self.assertRaisesRegex(ValueError, "consecutive"):
            append_increment(task0(), increment=2, batches=(batch(2, 3, 2),))

    def test_frozen_state_cannot_be_mutated(self):
        before = task0()
        with self.assertRaises(FrozenInstanceError):
            before.frozen_normal_cap = 99

    def test_reference_size_limit_rejects_real_data_scale(self):
        with self.assertRaisesRegex(ValueError, "reference row limit"):
            append_increment(initial_state(Policy(cap_mode="fixed", fixed_cap=2)), increment=0, batches=(batch(0, MAX_REFERENCE_ROWS + 1, role="normal"),))

    def test_invalid_policies_ids_and_digests_rejected(self):
        for kwargs in ({"seed": True}, {"seed": -1}, {"cap_mode": "future_max"}, {"cap_mode": "fixed", "fixed_cap": 0}, {"fixed_cap": 10}, {"calibration_denominator": 0}, {"calibration_numerator": 10}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Policy(**kwargs)
        for kwargs in ({"class_id": True}, {"source_train_sha256": "not-a-hash"}, {"role": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(batch(1, 1), **kwargs)


if __name__ == "__main__":
    unittest.main()
