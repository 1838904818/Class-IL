"""One allocated dataset/seed run, with bounded telemetry and small-file backup.

The unchanged scientific runner owns data verification and checkpoint identity.
This entry never submits another job, changes a training configuration, copies
datasets/weights into HOME, or launches more than one training process.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

DATASETS = {"nsl": "NSL-KDD", "unsw": "UNSW-NB15", "cic17": "CIC-IDS-2017", "cic18": "CSE-CIC-IDS2018"}
SLUGS = {"nsl": "nsl-kdd", "unsw": "unsw-nb15", "cic17": "cic-ids-2017", "cic18": "cic-ids-2018"}
METHODS = ["joint", "replay", "icarl", "replay_lwf", "rcl_tab", "derpp"]
FULL_COUNTS = {"nsl": (125973, 22544), "unsw": (120991, 30248),
               "cic17": (2262203, 565673), "cic18": (12882081, 3255102)}


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def under(path, root):
    return path.resolve().is_relative_to(root.resolve())


def copy_small(source, destination, previous):
    """Copy only portable scalar evidence; checkpoints and arrays stay in scratch."""
    copied = 0
    for path in source.rglob("*"):
        if not path.is_file() or path.is_symlink() or path.suffix not in {".json", ".csv", ".log", ".jsonl"}:
            continue
        stat = path.stat()
        key = path.relative_to(source).as_posix()
        stamp = (stat.st_size, stat.st_mtime_ns)
        if previous.get(key) == stamp:
            continue
        target = destination / key
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        shutil.copyfile(path, temporary)
        temporary.replace(target)
        previous[key] = stamp
        copied += 1
    return copied


def visible_gpu_uuids():
    """Resolve CUDA-visible devices only; Slurm global ordinals are not NVML IDs."""
    import torch
    identifiers = []
    count = torch.cuda.device_count()
    if not 1 <= count <= 2:
        raise RuntimeError(f"expected one or two allocated CUDA-visible GPUs, got {count}")
    for index in range(count):
        value = str(getattr(torch.cuda.get_device_properties(index), "uuid", ""))
        if not value.startswith("GPU-"):
            value = "GPU-" + value
        if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value):
            raise RuntimeError("CUDA-visible device has no supported physical GPU UUID")
        identifiers.append(value.lower())
    if len(set(identifiers)) != count:
        raise RuntimeError("CUDA-visible GPU UUIDs are not unique")
    return ["GPU-" + value[4:] for value in identifiers]


def gpu_sample(identifiers):
    """No unfiltered query fallback: malformed, missing, or nonnumeric rows fail."""
    result = {"gpu_telemetry_valid": False, "gpu_query_uuids": identifiers}
    if not identifiers:
        result["gpu_measurement_error"] = "no allocated GPU UUIDs"
        return result
    command = ["nvidia-smi", "--id=" + ",".join(identifiers),
               "--query-gpu=uuid,utilization.gpu,memory.used,memory.total",
               "--format=csv,noheader,nounits"]
    try:
        run = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        result.update(gpu_csv=run.stdout.strip(), gpu_query_exit=run.returncode,
                      gpu_query_stderr=run.stderr.strip())
        if run.returncode:
            raise ValueError("UUID-targeted nvidia-smi failed")
        values = {}
        for row in csv.reader(run.stdout.strip().splitlines()):
            if len(row) != 4:
                raise ValueError("GPU CSV must contain UUID and three numeric fields")
            identifier = row[0].strip().lower()
            if identifier not in {value.lower() for value in identifiers} or identifier in values:
                raise ValueError("GPU CSV contains unexpected or duplicate UUID")
            utilization, used, total = [float(value.strip()) for value in row[1:]]
            if not all(math.isfinite(value) for value in (utilization, used, total)):
                raise ValueError("GPU CSV contains nonfinite values")
            if not (0 <= utilization <= 100 and 0 <= used <= total and total > 0):
                raise ValueError("GPU CSV numeric values are outside valid ranges")
            values[identifier] = {"uuid": "GPU-" + identifier[4:],
                                  "utilization_percent": utilization,
                                  "memory_used_mib": used, "memory_total_mib": total}
        if set(values) != {value.lower() for value in identifiers}:
            raise ValueError("GPU CSV is missing an allocated GPU")
        result.update(gpu_telemetry_valid=True,
                      gpus=[values[value.lower()] for value in identifiers])
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        result["gpu_measurement_error"] = f"{type(exc).__name__}: {exc}"
    return result


class TelemetryGuard:
    """Sampling-based guard; observed spans do not prove continuous utilization."""
    def __init__(self):
        self.failures = 0
        self.low = {}

    def observe(self, measurement, elapsed):
        if not measurement["gpu_telemetry_valid"]:
            self.failures += 1
            self.low.clear()  # Never count unobserved gaps as low utilization.
            measurement["consecutive_telemetry_failures"] = self.failures
            return "three_consecutive_gpu_telemetry_failures" if self.failures >= 3 else None
        self.failures = 0
        spans = {}
        reason = None
        for gpu in measurement["gpus"]:
            identifier = gpu["uuid"]
            if gpu["utilization_percent"] < 10:
                start, count = self.low.get(identifier, (elapsed, 0))
                count += 1
                self.low[identifier] = (start, count)
                span = max(0.0, elapsed - start)
                spans[identifier] = {"consecutive_samples": count, "observed_span_seconds": span}
                if count >= 3 and elapsed > 0 and span >= elapsed / 3:
                    reason = "sampled_gpu_under_10_percent_for_observed_elapsed_third"
            else:
                self.low.pop(identifier, None)
        measurement.update(consecutive_telemetry_failures=0, sampled_low_utilization=spans,
                           low_utilization_threshold_seconds=elapsed / 3)
        return reason


def sample(pid, output, identifiers):
    result = {"utc": utc(), "pid": pid, "disk_free_bytes": shutil.disk_usage(output).free}
    # Linux /proc reads only this task's own child and aggregate free memory.
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        result["cpu_seconds"] = (int(stat[11]) + int(stat[12])) / os.sysconf("SC_CLK_TCK")
        result["rss_bytes"] = int(stat[21]) * os.sysconf("SC_PAGE_SIZE")
        mem = Path("/proc/meminfo").read_text().splitlines()
        result["available_ram_bytes"] = next(int(line.split()[1]) * 1024 for line in mem if line.startswith("MemAvailable:"))
    except (OSError, ValueError, StopIteration):
        result["proc_measurements"] = "unavailable on this platform"
    result.update(gpu_sample(identifiers))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("code-root", "data-root", "output-root", "protected-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2, 3, 4, 42), required=True)
    parser.add_argument("--method-group", choices=("all", "joint", "nonjoint"), default="all")
    parser.add_argument("--section26", action="store_true", help="Matched FARQ matrix, including held-out Joint reference")
    parser.add_argument("--local-smoke", action="store_true")
    args = parser.parse_args()
    source = os.environ.get("NCD_SOURCE_COMMIT", "")
    if not re.fullmatch("[0-9a-f]{40}", source):
        parser.error("NCD_SOURCE_COMMIT must identify the actual complete source commit")
    roots = [args.code_root, args.data_root, args.output_root, args.protected_root]
    args.code_root, args.data_root, args.output_root, args.protected_root = [p.resolve() for p in roots]
    if args.method_group != "all" and args.dataset != "cic18":
        parser.error("method splitting is reserved for measured CSE18 runtime needs")
    if under(args.protected_root, args.output_root) or under(args.output_root, args.protected_root):
        parser.error("result and protective-copy roots must be disjoint")
    if not args.local_smoke:
        user = os.environ.get("USER", "")
        scratch = Path("/scr/user") / user
        if not user or not os.environ.get("SLURM_JOB_ID", "").isdigit():
            parser.error("formal execution requires the user's Slurm allocation")
        if not all(under(p, Path.home()) for p in (args.code_root, args.protected_root)):
            parser.error("code and protected small outputs must be in HOME")
        if not all(under(p, scratch) for p in (args.data_root, args.output_root)):
            parser.error("data and active outputs must be in the user's scratch")
    config = args.code_root / "ncd/configs/problem_scout_section21.json"
    settings = json.loads(config.read_text(encoding="utf-8"))
    if settings["device"] != "cuda" or settings["cpu_threads"] != 2 or settings["methods"] != METHODS:
        parser.error("frozen GPU configuration changed")
    if (args.code_root / settings["problem_scout_data_root"]).resolve() != args.data_root:
        parser.error("stage the frozen relative data path beside the code; do not override scientific config")
    receipt = args.data_root / SLUGS[args.dataset] / "data_quality.json"
    info = json.loads(receipt.read_text(encoding="utf-8"))
    expected = FULL_COUNTS[args.dataset]
    if info.get("status") != "complete" or info.get("dataset") != DATASETS[args.dataset]:
        parser.error("prepared data receipt is incomplete or mismatched")
    if (sum(info["train_counts"].values()), sum(info["id_counts"].values())) != expected:
        parser.error("Section21.8 requires the frozen full uncapped dataset")
    methods = METHODS if args.method_group == "all" else (["joint"] if args.method_group == "joint" else METHODS[1:])
    if args.section26:
        if args.dataset == "cic18" or args.method_group != "all":
            parser.error("Section26.9 uses NSL/UNSW/CIC17 and all seven training entries")
        methods = ["replay", "wacb", "freq", "farq", "farq_all", "derpp", "joint"]
    seed = 1 if args.local_smoke else args.seed
    suffix = Path("smoke" if args.local_smoke else "formal") / args.dataset / f"seed_{seed}" / args.method_group
    output, protected = args.output_root / suffix, args.protected_root / suffix
    output.mkdir(parents=True, exist_ok=True)
    protected.mkdir(parents=True, exist_ok=True)
    # Exclusive lock survives crashes: do not automatically erase it on retry.
    # An operator must first establish that the exact old process/job is terminal.
    lock = output / ".active.lock"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "job_id": os.environ.get("SLURM_JOB_ID"), "utc": utc()}, stream)
    attempt = output / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + os.environ.get("SLURM_JOB_ID", "local") + f"_{os.getpid()}")
    attempt.mkdir(parents=True)
    command = [sys.executable, "-I", "-u", "ncd/run_problem_scout.py", "--config", "ncd/configs/problem_scout_section21.json",
               "--output-dir", os.path.relpath(output, args.code_root),
               "--datasets", DATASETS[args.dataset], "--seeds", str(seed), "--methods", *methods]
    if args.section26:
        command.extend(["--section26", "--min-steps-per-task", "2000"])
    if args.local_smoke:
        command.append("--smoke")
    record = {"job_id": os.environ.get("SLURM_JOB_ID"), "source_commit": source,
              "dataset": DATASETS[args.dataset], "seed": seed, "methods": methods, "smoke": args.local_smoke,
              "command": command, "started_utc": utc(), "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
              "batch_entry_sha256": hashlib.sha256((args.code_root / "ncd/hpc/run_problem_scout.sbatch").read_bytes()).hexdigest(),
              "job_helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "telemetry_policy": {"device_selection": "CUDA-visible physical UUIDs only",
                                   "sample_interval_seconds": 60, "failure_limit": 3,
                                   "low_utilization_percent": 10, "minimum_low_samples": 3,
                                   "low_span_threshold": "observed child elapsed / 3",
                                   "interpretation": "sampling-based conservative guard, not proof of continuous use",
                                   "low_utilization_enforced": not args.local_smoke},
              "stdout": "stdout.log", "stderr": "stderr.log"}
    write_json(attempt / "job_start.json", record)
    started, previous, child = time.monotonic(), {}, None
    exit_code = 1
    def stop(signum, _frame):
        if child is not None and child.poll() is None:
            child.send_signal(signum)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)
    try:
        if not args.local_smoke:
            step = os.environ.get("SLURM_STEP_ID", os.environ.get("SLURM_STEPID", ""))
            if not step.isdigit() or not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
                raise RuntimeError("GPU preflight requires an actual srun step with explicit CUDA visibility")
            record["slurm_step_id"] = step
        requested_wall = os.environ.get("NCD_WALLTIME_SECONDS", "")
        if requested_wall and (not requested_wall.isdigit() or int(requested_wall) <= 0):
            raise ValueError("NCD_WALLTIME_SECONDS must be a positive integer when supplied")
        record["requested_walltime_seconds"] = int(requested_wall) if requested_wall else None
        identifiers = visible_gpu_uuids()
        record["allocated_gpu_uuids"] = identifiers
        preflight = {"event": "gpu_preflight", "utc": utc(), **gpu_sample(identifiers)}
        with (attempt / "resources.jsonl").open("a", encoding="utf-8") as telemetry:
            telemetry.write(json.dumps(preflight, allow_nan=False) + "\n")
        if not preflight["gpu_telemetry_valid"]:
            raise RuntimeError("GPU telemetry preflight failed before training: " + preflight["gpu_measurement_error"])
        record["gpu_preflight_passed"] = True
        write_json(attempt / "job_start.json", record)
        copy_small(output, protected, previous)
        guard = TelemetryGuard()
        with (attempt / "stdout.log").open("wb") as out, (attempt / "stderr.log").open("wb") as err:
            child = subprocess.Popen(command, cwd=args.code_root, stdout=out, stderr=err)
            child_started = time.monotonic()
            print(json.dumps({"event": "runner_started", "pid": child.pid, **record}), flush=True)
            while True:
                measurement = sample(child.pid, output, identifiers)
                elapsed = time.monotonic() - child_started
                measurement["child_elapsed_seconds"] = elapsed
                reason = guard.observe(measurement, elapsed)
                measurement["guard_trigger"] = reason
                with (attempt / "resources.jsonl").open("a", encoding="utf-8") as telemetry:
                    telemetry.write(json.dumps(measurement, allow_nan=False) + "\n")
                copy_small(output, protected, previous)
                if child.poll() is not None:
                    exit_code = child.returncode
                    break
                if reason and (not args.local_smoke or reason == "three_consecutive_gpu_telemetry_failures"):
                    record["guard_stop_reason"] = reason
                    record["guard_stop_child_elapsed_seconds"] = elapsed
                    raise RuntimeError("stopping this wrapper's child: " + reason)
                try:
                    exit_code = child.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    pass
        if exit_code == 0:
            result = json.loads((output / "result.json").read_text(encoding="utf-8"))
            selection = {"datasets": [DATASETS[args.dataset]], "methods": methods, "seeds": [seed]}
            if result["execution_status"] != "complete" or result["expected"] != selection or len(result["cells"]) != len(methods):
                raise ValueError("runner exited zero without the complete selected matrix")
            record["training_seconds_total"] = result["training_seconds_total"]
    except BaseException as exc:
        record["entry_error"] = f"{type(exc).__name__}: {exc}"
        exit_code = exit_code or 1
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if child is not None:
            record["child_returncode"] = child.poll()
        raise
    finally:
        record.update(exit_code=exit_code, finished_utc=utc(), elapsed_seconds_this_attempt=time.monotonic() - started)
        write_json(attempt / "job_exit.json", record)
        copy_small(output, protected, previous)
        lock.unlink()
        print(json.dumps({"event": "job_exit", **record}), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
