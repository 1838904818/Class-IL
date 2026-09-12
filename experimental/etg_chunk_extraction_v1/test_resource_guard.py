"""Synthetic-only tests: no GPU, Slurm, SDK, model, network, or real cgroup I/O."""

import unittest
from unittest.mock import patch

import resource_guard as guard


ROOT = "/slurm/uid_42/job_123/step_0"
V2 = "29 23 0:26 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n"
V1 = "30 23 0:27 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n"


class CumulativeTests(unittest.TestCase):
    def history(self, resource="gpu", low_times=()):
        use = guard.CumulativeUse()
        for elapsed in range(0, 301, 15):
            values = dict(cpu=80, gpu=80, memory=80)
            if elapsed in low_times:
                values[resource] = 0
            reason = use.update(elapsed, **values)
        return use, reason

    def test_intermittent_low_intervals_accumulate(self):
        use, reason = self.history(low_times=(15, 60, 105, 150, 195, 240))
        self.assertEqual(reason, guard.LOW_UTILIZATION_CUMULATIVE)
        self.assertEqual(use.snapshot()["low_seconds"], dict(cpu=0, gpu=90, memory=0))
        self.assertEqual(use.snapshot()["low_fraction"]["gpu"], 0.30)
        self.assertEqual(use.measured_seconds, 300)

    def test_each_resource_independent(self):
        for resource in guard.CumulativeUse.resources:
            with self.subTest(resource=resource):
                _, reason = self.history(resource, range(15, 106, 15))
                self.assertEqual(reason, guard.LOW_UTILIZATION_CUMULATIVE)

    def test_low_fraction_below_30_percent_is_not_blocked(self):
        _, reason = self.history(low_times=(15, 60, 105, 150, 195))
        self.assertIsNone(reason)

    def test_ten_percent_is_not_low(self):
        use = guard.CumulativeUse()
        for elapsed in range(0, 301, 15):
            self.assertIsNone(use.update(elapsed, 10, 10, 10))
        self.assertEqual(sum(use.low_seconds.values()), 0)

    def test_grace_does_not_discard_low_time(self):
        use = guard.CumulativeUse()
        for elapsed in range(0, 300, 15):
            self.assertIsNone(use.update(elapsed, 80, 0, 80))
        self.assertEqual(use.update(300, 80, 80, 80), guard.LOW_UTILIZATION_CUMULATIVE)

    def test_right_endpoint_not_average_or_sample_count(self):
        use = guard.CumulativeUse()
        use.update(0, 80, 0, 80)
        use.update(5, 80, 80, 80)
        use.update(30, 80, 0, 80)
        self.assertEqual(use.low_seconds["gpu"], 25)
        self.assertEqual(use.measured_seconds, 30)

    def test_gap_not_counted_and_failure_sticky(self):
        use = guard.CumulativeUse()
        use.update(0, 80, 80, 80)
        self.assertEqual(use.update(31, 80, 80, 80), guard.MONITORING_INSUFFICIENT)
        self.assertEqual(use.measured_seconds, 0)
        self.assertEqual(use.update(46, 80, 80, 80), guard.MONITORING_INSUFFICIENT)

    def test_first_sample_late_is_insufficient(self):
        use = guard.CumulativeUse()
        self.assertEqual(use.update(31, 80, 80, 80), guard.MONITORING_INSUFFICIENT)

    def test_configured_startup_allowance_preserves_unknown_not_measured(self):
        use = guard.CumulativeUse(startup_limit_seconds=60)
        self.assertIsNone(use.update(60, 80, 80, 80))
        initial = use.snapshot()
        self.assertEqual(initial["unknown_start_seconds"], 60)
        self.assertEqual(initial["measured_seconds"], 0)
        self.assertEqual(initial["accounted_seconds"], 60)
        self.assertEqual(initial["low_seconds"], dict(cpu=0, gpu=0, memory=0))
        self.assertEqual(initial["potentially_low_seconds"], dict(cpu=60, gpu=60, memory=60))
        self.assertEqual(initial["low_fraction"], dict(cpu=1, gpu=1, memory=1))
        self.assertIn("upper-bound", initial["low_fraction_kind"])
        for elapsed in range(75, 301, 15):
            self.assertIsNone(use.update(elapsed, 80, 80, 80))
        final = use.snapshot()
        self.assertEqual(final["measured_seconds"], 240)
        self.assertEqual(final["accounted_seconds"], 300)
        self.assertEqual(final["low_fraction"], dict(cpu=.2, gpu=.2, memory=.2))
        self.assertTrue(final["decision_eligible"])

    def test_unknown_start_combines_with_observed_low_for_each_resource(self):
        for resource in guard.CumulativeUse.resources:
            with self.subTest(resource=resource):
                use = guard.CumulativeUse(startup_limit_seconds=60)
                use.update(60, 80, 80, 80)
                for elapsed in range(75, 301, 15):
                    values = dict(cpu=80, gpu=80, memory=80)
                    if elapsed in (75, 90):
                        values[resource] = 0
                    reason = use.update(elapsed, **values)
                self.assertEqual(reason, guard.LOW_UTILIZATION_CUMULATIVE)
                self.assertEqual(use.snapshot()["low_seconds"][resource], 30)
                self.assertEqual(use.snapshot()["potentially_low_seconds"][resource], 90)
                self.assertEqual(use.snapshot()["low_fraction"][resource], .3)

    def test_startup_allowance_does_not_extend_later_gap_limit(self):
        use = guard.CumulativeUse(startup_limit_seconds=60)
        use.update(45, 80, 80, 80)
        self.assertEqual(use.update(76, 80, 80, 80), guard.MONITORING_INSUFFICIENT)
        self.assertEqual(use.measured_seconds, 0)
        self.assertEqual(use.startup_unknown_seconds, 45)

    def test_excess_startup_unknown_remains_insufficient(self):
        use = guard.CumulativeUse(startup_limit_seconds=60)
        self.assertEqual(use.update(61, 80, 80, 80), guard.MONITORING_INSUFFICIENT)
        self.assertEqual(use.update(76, 80, 80, 80), guard.MONITORING_INSUFFICIENT)
        self.assertEqual(use.startup_unknown_seconds, 61)

    def test_startup_allowance_cannot_exceed_sixty_seconds(self):
        for limit in (None, True, 0, -1, 61, float("nan"), float("inf"), "60"):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                guard.CumulativeUse(startup_limit_seconds=limit)

    def test_exact_30_second_intervals_allowed(self):
        use = guard.CumulativeUse()
        for elapsed in range(0, 301, 30):
            self.assertIsNone(use.update(elapsed, 80, 80, 80))

    def test_invalid_samples_are_not_zero_and_latch_failure(self):
        for invalid in (None, True, "0", float("nan"), float("inf"), -1, 101):
            for resource in guard.CumulativeUse.resources:
                with self.subTest(invalid=invalid, resource=resource):
                    values = dict(cpu=80, gpu=80, memory=80)
                    values[resource] = invalid
                    use = guard.CumulativeUse()
                    with self.assertRaises(ValueError):
                        use.update(0, **values)
                    self.assertEqual(use.status(), guard.MONITORING_INSUFFICIENT)
                    self.assertEqual(use.measured_seconds, 0)

    def test_nonmonotonic_time_rejected(self):
        for elapsed in (-1, 0, None, float("nan"), True):
            use = guard.CumulativeUse()
            use.update(0, 80, 80, 80)
            with self.assertRaises(ValueError):
                use.update(elapsed, 80, 80, 80)

    def test_snapshot_is_copy_and_empty_fraction_missing(self):
        use = guard.CumulativeUse()
        self.assertIsNone(use.snapshot()["low_fraction"]["cpu"])
        use.snapshot()["low_seconds"]["cpu"] = 999
        self.assertEqual(use.low_seconds["cpu"], 0)


class MemorySafetyTests(unittest.TestCase):
    def test_exact_bound_and_threshold(self):
        requested = 12 * 1024**3
        threshold = int(requested * .90)
        self.assertEqual(guard.MEMORY_STOP_BYTES, threshold)
        for source in guard.OWN_MEMORY_SOURCES:
            self.assertIsNone(guard.memory_safety(threshold-1, requested,
                              measured_limit_bytes=requested, source=source))
            self.assertEqual(guard.memory_safety(threshold, requested,
                             measured_limit_bytes=requested, source=source),
                             "HOST_MEMORY_GUARD")

    def test_missing_wrong_source_and_wrong_limit_fail_closed(self):
        for overrides in ({"used_bytes": None}, {"measured_limit_bytes": None},
                          {"source": "process-rss"}, {"measured_limit_bytes": 1},
                          {"used_bytes": True}, {"requested_bytes": 0}):
            args = dict(used_bytes=0, requested_bytes=guard.REQUESTED_MEMORY_BYTES,
                        measured_limit_bytes=guard.REQUESTED_MEMORY_BYTES,
                        source="own-step-cgroup-v2")
            args.update(overrides)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                guard.memory_safety(**args)

    def test_peak_does_not_disappear_when_current_falls(self):
        self.assertEqual(guard.memory_safety(1, guard.REQUESTED_MEMORY_BYTES,
                         measured_limit_bytes=guard.REQUESTED_MEMORY_BYTES,
                         source="own-step-cgroup-v2", peak_bytes=guard.MEMORY_STOP_BYTES),
                         "HOST_MEMORY_GUARD")
        with self.assertRaises(ValueError):
            guard.memory_safety(2, 100, measured_limit_bytes=100,
                                source="own-step-cgroup-v1", peak_bytes=1)


class CgroupTests(unittest.TestCase):
    def test_v2_task_leaf_resolves_whole_own_step(self):
        own = guard.resolve_own_cgroup("0::" + ROOT + "/task_0\n", V2, "123", "0")
        self.assertEqual(own.scope_path, "/sys/fs/cgroup" + ROOT)
        self.assertEqual(own.version, 2)

    def test_v1_memory_controller_only(self):
        own = guard.resolve_own_cgroup("4:cpu,cpuacct:/foreign\n5:memory:" + ROOT +
                                      "/task_0\n", V1, "123", "0")
        self.assertEqual(own.scope_path, "/sys/fs/cgroup/memory" + ROOT)
        self.assertEqual(own.version, 1)

    def test_nonroot_mount_maps_exact_scope(self):
        mount = V2.replace(" / /sys", " /slurm /sys")
        own = guard.resolve_own_cgroup("0::" + ROOT, mount, "123", "0")
        self.assertEqual(own.scope_path, "/sys/fs/cgroup/uid_42/job_123/step_0")

    def test_namespace_slash_requires_visible_exact_identity(self):
        mount = V2.replace(" / /sys", " " + ROOT + " /sys")
        own = guard.resolve_own_cgroup("0::/", mount, "123", "0")
        self.assertEqual(own.scope_path, "/sys/fs/cgroup")
        with self.assertRaises(ValueError):
            guard.resolve_own_cgroup("0::/", V2, "123", "0")

    def test_hidden_step_parent_cannot_measure_worker_scope(self):
        mount = V2.replace(" / /sys", " " + ROOT + "/task_0 /sys")
        with self.assertRaises(ValueError):
            guard.resolve_own_cgroup("0::" + ROOT + "/task_0", mount, "123", "0")

    def test_ambiguous_mount_or_hierarchy_rejected(self):
        for memberships, mounts in (("0::" + ROOT, V2 + V2),
                                     ("0::" + ROOT + "\n0::" + ROOT, V2),
                                     ("0::" + ROOT + "\n5:memory:" + ROOT, V2 + V1)):
            with self.subTest(memberships=memberships), self.assertRaises(ValueError):
                guard.resolve_own_cgroup(memberships, mounts, "123", "0")

    def test_foreign_traversal_and_nonnumeric_identity_rejected(self):
        for member in (ROOT.replace("job_123", "job_456"),
                       ROOT.replace("step_0", "step_1"),
                       ROOT + "/../step_1", ROOT + "/step_2", ROOT + "\\task_0",
                       ROOT.replace("step_0", "extra/step_0")):
            with self.subTest(member=member), self.assertRaises(ValueError):
                guard.resolve_own_cgroup("0::" + member, V2, "123", "0")
        with self.assertRaises(ValueError):
            guard.resolve_own_cgroup("0::" + ROOT, V2, "123", "batch")

    def sample(self, version=2, **replacement):
        names = (("memory.current", "memory.peak", "memory.max") if version == 2 else
                 ("memory.usage_in_bytes", "memory.max_usage_in_bytes", "memory.limit_in_bytes"))
        values = dict(zip(names, ("100", "200", str(guard.REQUESTED_MEMORY_BYTES))))
        if version == 1:
            values["memory.use_hierarchy"] = "1"
        values.update(replacement)
        membership = ("0::" if version == 2 else "5:memory:") + ROOT + "/task_0"
        with patch.object(guard.Path, "read_text", side_effect=[membership, V2 if version == 2 else V1]), \
                patch.object(guard, "_read_scope_files", return_value=values) as reader:
            result = guard.own_memory_sample("123", "0", guard.REQUESTED_MEMORY_BYTES)
            reader.assert_called_once()
            self.assertTrue(reader.call_args.args[0].endswith(ROOT))
            self.assertNotIn("task_0", reader.call_args.args[0])
            return result

    def test_synthetic_read_v1_and_v2(self):
        for version in (1, 2):
            result = self.sample(version)
            self.assertEqual(result["current_bytes"], 100)
            self.assertEqual(result["peak_bytes"], 200)
            self.assertEqual(result["limit_bytes"], guard.REQUESTED_MEMORY_BYTES)
            self.assertEqual(result["scope"], "whole-own-computation-step")

    def test_unlimited_missing_wrong_bound_and_peak_fail(self):
        for values in ({"memory.max": "max"}, {"memory.current": ""},
                       {"memory.max": "1000"}, {"memory.peak": "99"},
                       {"memory.max": str(2**63 - 4096)}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.sample(**values)

    def test_missing_proc_blocks_before_cgroup_read(self):
        with patch.object(guard.Path, "read_text", side_effect=OSError("synthetic missing")), \
                patch.object(guard, "_read_scope_files") as reader:
            with self.assertRaises(OSError):
                guard.own_memory_sample("123", "0")
            reader.assert_not_called()

    def test_v1_requires_descendant_accounting(self):
        with self.assertRaises(ValueError):
            self.sample(version=1, **{"memory.use_hierarchy": "0"})

    def test_exact_descriptor_walk_and_bounded_read(self):
        with patch.object(guard.os, "name", "posix"), \
                patch.object(guard.os, "O_DIRECTORY", 0x10000, create=True), \
                patch.object(guard.os, "O_NOFOLLOW", 0x20000, create=True), \
                patch.object(guard.os, "open", side_effect=[100, 101, 102, 103]) as opener, \
                patch.object(guard.os, "close") as closer, \
                patch.object(guard.os, "fstat") as fstat, \
                patch.object(guard.os, "read", return_value=b"100\n") as reader:
            fstat.return_value.st_mode = guard.stat.S_IFREG
            self.assertEqual(guard._read_scope_files("/synthetic/step_0", ("memory.current",)),
                             {"memory.current": "100"})
            self.assertEqual([call.args[0] for call in opener.call_args_list],
                             ["/", "synthetic", "step_0", "memory.current"])
            self.assertTrue(all(call.args[1] & 0x20000 for call in opener.call_args_list))
            reader.assert_called_once_with(103, 257)
            self.assertEqual([call.args[0] for call in closer.call_args_list],
                             [100, 101, 103, 102])

    def test_symlink_open_failure_has_no_fallback(self):
        with patch.object(guard.os, "name", "posix"), \
                patch.object(guard.os, "O_DIRECTORY", 0x10000, create=True), \
                patch.object(guard.os, "O_NOFOLLOW", 0x20000, create=True), \
                patch.object(guard.os, "open", side_effect=[100, OSError("synthetic ELOOP")]), \
                patch.object(guard.os, "close") as closer, \
                patch.object(guard.os, "read") as reader:
            with self.assertRaises(OSError):
                guard._read_scope_files("/linked", ("memory.current",))
            closer.assert_called_once_with(100)
            reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
