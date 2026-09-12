"""Fail-closed, stdlib-only own-step resource accounting.

This module does not start processes, query Slurm, enumerate cgroups, or perform
network I/O. Call ``own_memory_sample`` only inside the allocated compute step.
All thresholds here are project controls, not quotations of cluster policy.
"""

import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


REQUESTED_MEMORY_BYTES = 12 * 1024**3
MEMORY_STOP_BYTES = int(REQUESTED_MEMORY_BYTES * 0.90)
MONITORING_INSUFFICIENT = "MONITORING_INSUFFICIENT"
LOW_UTILIZATION_CUMULATIVE = "LOW_UTILIZATION_CUMULATIVE"
OWN_MEMORY_SOURCES = ("own-step-cgroup-v1", "own-step-cgroup-v2")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _bytes(value, label, allow_zero=False):
    _require(type(value) is int and value >= (0 if allow_zero else 1),
             label + " is missing or invalid")
    return value


class CumulativeUse:
    """Independent resource-low interval fractions, never averaged together.

    Observed intervals are estimated from their right endpoints. Time before
    the first observation is unknown and conservatively potentially low for
    every resource, never labelled measured use. Its bounded startup allowance
    is at most 60 seconds; later intervals above 30 seconds are not counted.
    Missing/invalid samples or excessive gaps are fail-closed and sticky. Low
    use accrued during grace remains in the cumulative totals.
    """

    resources = ("cpu", "gpu", "memory")

    def __init__(self, startup_limit_seconds=30):
        _require(_number(startup_limit_seconds) and 0 < startup_limit_seconds <= 60,
                 "startup monitoring allowance must be positive and at most 60 seconds")
        self.startup_limit_seconds = float(startup_limit_seconds)
        self.startup_unknown_seconds = 0.0
        self.last_elapsed = None
        self.measured_seconds = 0.0
        self.low_seconds = dict.fromkeys(self.resources, 0.0)
        self.reason = None
        self.samples = 0

    def update(self, elapsed, cpu, gpu, memory):
        values = (cpu, gpu, memory)
        valid = (_number(elapsed) and elapsed >= 0
                 and all(_number(value) and 0 <= value <= 100 for value in values)
                 and (self.last_elapsed is None or elapsed > self.last_elapsed))
        if not valid:
            self.reason = MONITORING_INSUFFICIENT
            raise ValueError("missing/invalid utilization or nonmonotonic elapsed time")
        if self.last_elapsed is None:
            self.startup_unknown_seconds = float(elapsed)
            if elapsed > self.startup_limit_seconds:
                self.reason = MONITORING_INSUFFICIENT
        else:
            gap = elapsed - self.last_elapsed
            if gap > 30:
                self.reason = MONITORING_INSUFFICIENT
            else:
                self.measured_seconds += gap
                for name, value in zip(self.resources, values):
                    if value < 10:
                        self.low_seconds[name] += gap
        self.last_elapsed = float(elapsed)
        self.samples += 1
        if elapsed >= 300:
            if self.measured_seconds < 240:
                self.reason = MONITORING_INSUFFICIENT
            elif self.reason is None and any(
                seconds + self.startup_unknown_seconds >= 0.30 * (
                    self.measured_seconds + self.startup_unknown_seconds)
                for seconds in self.low_seconds.values()
            ):
                self.reason = LOW_UTILIZATION_CUMULATIVE
        return self.reason

    observe = update

    def status(self):
        return self.reason

    def snapshot(self):
        accounted = self.measured_seconds + self.startup_unknown_seconds
        return {
            "estimator": "right-endpoint-interval-estimate",
            "elapsed_seconds": self.last_elapsed,
            "measured_seconds": self.measured_seconds,
            "unknown_start_seconds": self.startup_unknown_seconds,
            "startup_limit_seconds": self.startup_limit_seconds,
            "accounted_seconds": accounted,
            "samples": self.samples,
            "low_seconds": dict(self.low_seconds),
            "low_seconds_kind": "observed-right-endpoint-estimate-only",
            "potentially_low_seconds": {
                name: seconds + self.startup_unknown_seconds
                for name, seconds in self.low_seconds.items()
            },
            "low_fraction_kind": "upper-bound-including-unknown-start-as-low",
            "low_fraction": {
                name: (seconds + self.startup_unknown_seconds) / accounted if accounted else None
                for name, seconds in self.low_seconds.items()
            },
            "decision_eligible": (self.last_elapsed is not None
                                  and self.last_elapsed >= 300
                                  and self.measured_seconds >= 240),
            "reason": self.reason,
        }


def memory_safety(used_bytes, requested_bytes, *, measured_limit_bytes, source,
                  peak_bytes=None):
    """Validate own-step evidence and stop at 90% of the bound allocation.

    Process RSS, host free memory, GPU memory, and missing readings cannot
    replace whole-step memory accounting. A recorded peak is also conservative
    stop evidence, even if current usage has since declined.
    """
    used = _bytes(used_bytes, "own-step memory.current", allow_zero=True)
    requested = _bytes(requested_bytes, "requested allocation")
    limit = _bytes(measured_limit_bytes, "own-step allocation limit")
    _require(source in OWN_MEMORY_SOURCES, "unverified memory source")
    _require(limit == requested, "own-step allocation limit does not match request")
    if peak_bytes is not None:
        peak = _bytes(peak_bytes, "own-step memory peak", allow_zero=True)
        _require(peak >= used, "memory peak precedes current usage")
        used = peak
    return "HOST_MEMORY_GUARD" if used >= int(requested * 0.90) else None


@dataclass(frozen=True)
class OwnCgroup:
    version: int
    scope_path: str
    membership_path: str
    mount_root: str
    mount_point: str
    job: str
    step: str


def _posix(value):
    _require(isinstance(value, str) and value.startswith("/")
             and "\\" not in value and "\x00" not in value
             and all(part not in (".", "..") for part in value.split("/")),
             "unsafe cgroup path")
    return PurePosixPath(value)


def _unescape_mount(value):
    return re.sub(r"\\(040|011|012|134)",
                  lambda match: chr(int(match[1], 8)), value)


def resolve_own_cgroup(proc_cgroup, proc_mountinfo, job, step):
    """Pure resolver for only self's exact Slurm computation-step ancestor.

    A self task leaf is widened only to its already-proven step ancestor, so
    monitor and workers are included. Hidden namespace identity, duplicate
    memberships/mounts, foreign IDs, and a mount hiding step scope are rejected.
    A namespace ``/`` is accepted only when mountinfo exposes the exact step.
    """
    _require(isinstance(job, str) and re.fullmatch(r"[1-9]\d*", job),
             "numeric own job required")
    _require(isinstance(step, str) and re.fullmatch(r"\d+", step),
             "numeric own computation step required")
    memberships = []
    for row in proc_cgroup.splitlines():
        fields = row.split(":", 2)
        _require(len(fields) == 3 and fields[0].isdigit(), "invalid self cgroup entry")
        hierarchy, controllers, member = fields
        if hierarchy == "0" and controllers == "":
            memberships.append((2, _posix(member)))
        elif "memory" in controllers.split(","):
            memberships.append((1, _posix(member)))
    # A hybrid system with both memory hierarchies is ambiguous, not a fallback.
    _require(len(memberships) == 1, "one own memory hierarchy required")
    version, membership = memberships[0]
    candidates = []
    for row in proc_mountinfo.splitlines():
        before, separator, after = row.partition(" - ")
        _require(bool(separator), "invalid self mountinfo entry")
        left, right = before.split(), after.split()
        _require(len(left) >= 6 and len(right) >= 3, "invalid self mountinfo fields")
        relevant = (version == 2 and right[0] == "cgroup2") or (
            version == 1 and right[0] == "cgroup" and "memory" in right[2].split(","))
        if not relevant:
            continue
        root = _posix(_unescape_mount(left[3]))
        mount = _posix(_unescape_mount(left[4]))
        # Full membership paths are preferred. The only namespace remapping
        # accepted is self '/', whose global scope is the visible mount root.
        effective = root if membership == PurePosixPath("/") else membership
        try:
            effective.relative_to(root)
        except ValueError:
            continue
        parts = effective.parts
        job_part, step_part = "job_" + job, "step_" + step
        if parts.count(job_part) != 1 or parts.count(step_part) != 1:
            continue
        job_index, step_index = parts.index(job_part), parts.index(step_part)
        if step_index != job_index + 1:
            continue
        if any(part.startswith("job_") and part != job_part for part in parts):
            continue
        if any(part.startswith("step_") and part != step_part for part in parts):
            continue
        scope = PurePosixPath(*parts[:step_index + 1])
        try:
            relative = scope.relative_to(root)
        except ValueError:
            continue  # A task-only mount cannot prove complete step memory.
        candidates.append(OwnCgroup(version, str(mount / relative), str(membership),
                                    str(root), str(mount), job, step))
    _require(len(candidates) == 1, "own step memory cgroup unavailable or ambiguous")
    return candidates[0]


def _read_scope_files(scope_path, names):
    """Open exact files by descriptor without following any symlink component."""
    _require(os.name == "posix" and hasattr(os, "O_NOFOLLOW"),
             "secure Linux cgroup reads unavailable")
    path = _posix(scope_path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for part in path.parts[1:]:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        output = {}
        for name in names:
            _require("/" not in name and "\\" not in name, "invalid metric name")
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
            try:
                _require(stat.S_ISREG(os.fstat(fd).st_mode), "nonregular cgroup metric")
                raw = os.read(fd, 257)
                _require(len(raw) <= 256, "oversized cgroup metric")
                output[name] = raw.decode("ascii").strip()
            finally:
                os.close(fd)
        return output
    finally:
        os.close(descriptor)


def own_memory_sample(job, step, requested_bytes=None):
    """Read self metadata and exact own-step memory only; no process launches.

    Exceptions mean monitoring is unavailable and must block worker startup.
    There is no process-RSS, global-memory, or foreign-cgroup fallback.
    """
    own = resolve_own_cgroup(Path("/proc/self/cgroup").read_text(encoding="ascii"),
                            Path("/proc/self/mountinfo").read_text(encoding="ascii"),
                            job, step)
    names = (("memory.current", "memory.peak", "memory.max") if own.version == 2 else
             ("memory.usage_in_bytes", "memory.max_usage_in_bytes", "memory.limit_in_bytes",
              "memory.use_hierarchy"))
    raw = _read_scope_files(own.scope_path, names)
    _require(all(re.fullmatch(r"[0-9]+", raw[name]) for name in names),
             "missing or unlimited own-step memory metric")
    if own.version == 1:
        _require(raw["memory.use_hierarchy"] == "1",
                 "v1 own-step accounting does not include worker descendants")
    current, peak, limit = (int(raw[name]) for name in names[:3])
    _require(0 <= current <= peak and 0 < limit < 2**60,
             "invalid own-step memory values or unlimited allocation")
    if requested_bytes is not None:
        _require(limit == _bytes(requested_bytes, "requested allocation"),
                 "own-step allocation limit does not match request")
    return {
        "source": "own-step-cgroup-v" + str(own.version),
        "current_bytes": current,
        "peak_bytes": peak,
        "limit_bytes": limit,
        "scope": "whole-own-computation-step",
        "scope_path": own.scope_path,
        "job": job,
        "step": step,
    }
