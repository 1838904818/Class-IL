"""One local CUDA child at a time for the reserved CSE18 reference seed2.

Calls the exact frozen reference package without changing its training or HPC
sources. The local launcher is separately bound. No formal aggregate is read or
created here, including after all seven assigned cells finish. The operator must
keep seed2 reserved exclusively to this local queue; HPC owns other seeds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ENTRY = Path(__file__).resolve()
WORK_ROOT = ENTRY.parents[2]
RESOURCE_HELPER_SHA = "bf8a5aa83b033927966d26dcf0d834ad27008fa4bfc1622a79c8a2dd5721239b"
ADAPTER_COMMIT = "270121272848087e4af85ec5290eaede0102d776"
PACKAGE_SHA = "0b0380b7cbdf5598c2c14f9f9636593596dc0d74c609a8c42575b6ca02b2fc00"
METHODS = ("aacil_kd", "freq", "derpp", "replay", "wacb", "replay_bs", "aacil")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_helpers():
    helper_path = ENTRY.with_name("run_section35_local.py")
    if sha(helper_path) != RESOURCE_HELPER_SHA:
        raise ValueError("Frozen reference loader/resource helper changed")
    spec = importlib.util.spec_from_file_location("section37_frozen_local_helpers", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("code-root", "data-root", "output-root", "protected-root", "reference-package"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True, help="Local launcher publication commit; separate from frozen adapter2701212")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-smoke", action="store_true", help="NONSCIENTIFIC CPU NSL/UNSW seed1, one epoch, <=64 rows/class, only KD child")
    parser.add_argument("--smoke-dataset", choices=("NSL-KDD", "UNSW-NB15"), default="NSL-KDD")
    parser.add_argument("--child-method", choices=METHODS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.name != "nt" or os.environ.get("SLURM_JOB_ID"):
        parser.error("This local Windows launcher must not run inside Slurm")
    if not re.fullmatch("[0-9a-f]{40}", args.source_commit):
        parser.error("Local launcher source commit must be a complete SHA")
    for name in ("code_root", "data_root", "output_root", "protected_root", "reference_package"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output_root.is_relative_to(args.protected_root) or args.protected_root.is_relative_to(args.output_root):
        parser.error("Active and protected roots must be disjoint")
    if args.code_root.parent != WORK_ROOT:
        parser.error("Frozen reference code must be directly under this workspace work directory")
    global_lock = WORK_ROOT / (".section37_cpu_smoke.lock" if args.local_smoke else ".section35_local_cuda.lock")
    if not args.child_method and global_lock.exists():
        parser.error("Local CUDA queue lock is held; no second GPU driver")
    helper = load_helpers()
    reference = helper.load_reference(args)
    settings = reference.read(args.code_root / "ncd/configs/problem_scout_section21.json")
    if (args.code_root / settings["problem_scout_data_root"]).resolve() != args.data_root:
        parser.error("Frozen relative data-cache path must resolve exactly; no config override")
    os.environ["NCD_SOURCE_COMMIT"] = ADAPTER_COMMIT
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if args.local_smoke else "0"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    methods = (METHODS[0],) if args.local_smoke else METHODS
    seed = 1 if args.local_smoke else 2
    if args.child_method:
        if args.child_method not in methods or os.environ.get("NCD_SECTION37_PARENT_PID") != str(os.getppid()):
            parser.error("Child must be this guarded queue's assigned method/seed")
        args.methods, args.seeds = [args.child_method], [seed]
        # Preserve original training/adapter identity and source hashes so local
        # and HPC cells remain source-compatible. Launcher provenance is in the
        # separate attempt records, not mislabeled as training source.
        reference.run_child(args)
        return
    cache = reference.check_cache(args.data_root, args.smoke_dataset if args.local_smoke else reference.DATASET)
    identifiers = []
    if not args.local_smoke:
        identifiers = reference.visible_gpu_uuids()
        if len(identifiers) != 1 or not reference.gpu_sample(identifiers)["gpu_telemetry_valid"]:
            raise RuntimeError("Exactly one selected local CUDA device and valid UUID telemetry required")
    output, protected = args.output_root, args.protected_root
    output.mkdir(parents=True, exist_ok=True)
    protected.mkdir(parents=True, exist_ok=True)
    with global_lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "started_at": reference.utc(), "seed": seed}, stream)
    attempt = output / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"_local_{os.getpid()}")
    attempt.mkdir(parents=True)
    record = {"status": "running", "started_at": reference.utc(), "pid": os.getpid(),
        "local_launcher_commit": args.source_commit, "local_launcher_sha256": sha(ENTRY),
        "local_resource_helper_sha256": RESOURCE_HELPER_SHA,
        "reference_adapter_commit": ADAPTER_COMMIT, "reference_package_sha256": PACKAGE_SHA,
        "methods": list(methods), "seeds": [seed], "cache": cache, "smoke": args.local_smoke,
        "non_scientific": args.local_smoke, "resume": args.resume, "gpu_uuids": identifiers,
        "scope": "Only reserved reference seed2 after its HPC assignment is terminal; no other reference seed or selected-main overlap; one child serially",
        "aggregate_policy": "No formal aggregate produced or read by local launcher", "cells": [], "current_cell": None}
    reference.write_json(attempt / "job_start.json", record)
    received, child, cell_lock, copied = [], None, None, {}

    def on_signal(number, _frame):
        received.append(number)
        if child is not None and child.poll() is None:
            child.terminate()

    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, on_signal)
    try:
        for method in methods:
            if received or (output / "STOP_REQUESTED").exists():
                raise RuntimeError("Owned stop requested; preserved all completed checkpoints")
            target = reference.cell_root(output, method, seed)
            if target.exists() and any(target.iterdir()) and not args.resume:
                raise ValueError("Existing cell requires explicit resume after prior process terminal confirmation")
            locks = output / ".cell_locks"
            locks.mkdir(exist_ok=True)
            candidate_lock = locks / f"{method}_seed{seed}.lock"
            with candidate_lock.open("x", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid(), "started_at": reference.utc()}, stream)
            cell_lock = candidate_lock
            command = [sys.executable, "-I", "-B", "-u", str(ENTRY), "--child-method", method,
                       "--source-commit", args.source_commit]
            for name in ("code_root", "data_root", "output_root", "protected_root", "reference_package"):
                command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
            if args.local_smoke:
                command.extend(["--local-smoke", "--smoke-dataset", args.smoke_dataset])
            current = {"method": method, "seed": seed, "started_at": reference.utc(), "command": command}
            record["current_cell"] = current
            prefix = attempt / f"{method}_seed{seed}"
            with prefix.with_suffix(".stdout.log").open("x", encoding="utf-8") as stdout, prefix.with_suffix(".stderr.log").open("x", encoding="utf-8") as stderr:
                child = subprocess.Popen(command, cwd=args.code_root,
                    env=dict(os.environ, NCD_SECTION37_PARENT_PID=str(os.getpid())),
                    stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
                current["pid"] = child.pid
                reference.write_json(attempt / "status.json", record)
                started, next_sample, guard = time.monotonic(), 0., reference.TelemetryGuard()
                while child.poll() is None:
                    if received or (output / "STOP_REQUESTED").exists():
                        raise RuntimeError("Owned stop requested; preserved all completed checkpoints")
                    elapsed = time.monotonic() - started
                    if not args.local_smoke and elapsed >= next_sample:
                        measurement = helper.own_process_sample(reference, child.pid, output, identifiers)
                        reason = guard.observe(measurement, elapsed)
                        measurement.update(method=method, seed=seed, child_elapsed_seconds=elapsed, guard_trigger=reason)
                        with (attempt / "resources.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(measurement, allow_nan=False) + "\n")
                        next_sample = elapsed + 60
                        if reason:
                            raise RuntimeError(reason)
                    reference.protect_owned(output, protected, attempt, methods, [seed], copied)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                current.update(exit_code=child.returncode, ended_at=reference.utc(), elapsed_seconds=time.monotonic() - started)
                if child.returncode != 0 or not reference.completed_path(output, method, seed).exists():
                    raise RuntimeError(f"Assigned cell {method}/{seed} failed; preserve logs and diagnose before resume")
            record["cells"].append(current)
            record["current_cell"] = None
            reference.write_json(attempt / "status.json", record)
            cell_lock.unlink()
            cell_lock = None
            print(f"CELL_CLOSED method={method} seed={seed} metrics=sealed", flush=True)
        if args.local_smoke:
            reference.release_if_complete(output, True, methods)
        record.update(status="complete", ended_at=reference.utc())
    except BaseException as error:
        reference.terminate(child)
        if record["current_cell"] is not None and child is not None:
            record["current_cell"].update(exit_code=child.returncode, ended_at=reference.utc(), elapsed_seconds=time.monotonic() - started)
        record.update(status="interrupted" if received or (output / "STOP_REQUESTED").exists() else "failed",
                      ended_at=reference.utc(), error_type=type(error).__name__, error=str(error))
        raise
    finally:
        reference.terminate(child)
        reference.write_json(attempt / "status.json", record)
        reference.write_json(attempt / "job_exit.json", record)
        reference.protect_owned(output, protected, attempt, methods, [seed], copied)
        if cell_lock is not None:
            cell_lock.unlink(missing_ok=True)
        global_lock.unlink(missing_ok=True)
    print(f"ASSIGNMENT_COMPLETE physical_cells={len(record['cells'])} smoke={args.local_smoke} metrics=sealed", flush=True)


if __name__ == "__main__":
    main()
