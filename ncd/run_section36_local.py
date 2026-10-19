"""Reserved local KD+WA seed42, only after the reference GPU queue is free.

The exact selected-method package supplies frozen training, selection validation,
checkpoint/resume and score-sealed child execution. This launcher never creates
or reads formal aggregates. Its commit/hash is separate from training c97f83b
and the immutable HPC wrapper045b862 identities.
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
import tarfile
import time

ENTRY = Path(__file__).resolve()
WORK_ROOT = ENTRY.parents[2]
WRAPPER_COMMIT = "045b86222c24d23389edae8ae75c0dd30fd80a42"
TRAINING_COMMIT = "c97f83b1bfa14f4d2665ff33c45c897941e9ddb3"
PACKAGE_SHA = "d7d9912bde5a8f30147b8873cd054484aca66caab57923998ae2975a656b5650"
RESOURCE_HELPER_SHA = "bf8a5aa83b033927966d26dcf0d834ad27008fa4bfc1622a79c8a2dd5721239b"
METHOD = "aacil_kd_wa"
SELECTION_RELATIVE = "ncd/results/problem_scout/section31/development_v1/selection.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_selected(args):
    if sha(args.selected_package) != PACKAGE_SHA:
        raise ValueError("Use the exact immutable selected-method 045b862 archive")
    with tarfile.open(args.selected_package, "r") as archive:
        members = archive.getmembers()
        if len(members) != 46:
            raise ValueError("Selected-method package file set changed")
        for item in members:
            path = (args.code_root / item.name).resolve()
            if not item.isfile() or not path.is_relative_to(args.code_root):
                raise ValueError("Unsafe or unsupported source archive entry")
            expected = hashlib.sha256(archive.extractfile(item).read()).hexdigest()
            if not path.is_file() or sha(path) != expected:
                raise ValueError(f"Immutable selected-method source differs: {item.name}")
    sys.path.insert(0, str(args.code_root))
    sys.path.insert(0, str(args.code_root / "ncd/hpc"))
    selected = load_module("section36_frozen_selected", args.code_root / "ncd/hpc/section32_job.py")
    os.environ["NCD_SELECTION_FILE"] = str(args.code_root / SELECTION_RELATIVE)
    selected.selected_configuration()
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("code-root", "data-root", "output-root", "protected-root", "selected-package"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True, help="Actual local-launcher publication commit, not training/wrapper commit")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-smoke", action="store_true", help="NONSCIENTIFIC CPU seed1, one epoch, <=64 rows/class")
    parser.add_argument("--smoke-dataset", choices=("NSL-KDD", "UNSW-NB15"), default="NSL-KDD")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.name != "nt" or os.environ.get("SLURM_JOB_ID"):
        parser.error("This local Windows launcher must not run inside Slurm")
    if not re.fullmatch("[0-9a-f]{40}", args.source_commit):
        parser.error("Local launcher source commit must be a complete SHA")
    for name in ("code_root", "data_root", "output_root", "protected_root", "selected_package"):
        setattr(args, name, getattr(args, name).resolve())
    if args.code_root.parent != WORK_ROOT:
        parser.error("Selected immutable code must be directly under this workspace work directory")
    if args.output_root.is_relative_to(args.protected_root) or args.protected_root.is_relative_to(args.output_root):
        parser.error("Active and protected roots must be disjoint")
    global_lock = WORK_ROOT / (".section36_cpu_smoke.lock" if args.local_smoke else ".section35_local_cuda.lock")
    # Check before any CUDA inspection; atomic creation below also closes races.
    if not args.child and global_lock.exists():
        parser.error("Local reference/selected CUDA queue lock is held; no second GPU driver")
    selected = load_selected(args)
    reference = selected.supervisor
    settings = reference.read(args.code_root / "ncd/configs/problem_scout_section21.json")
    if (args.code_root / settings["problem_scout_data_root"]).resolve() != args.data_root:
        parser.error("Frozen relative cache path must resolve exactly; no config override")
    os.environ["NCD_SOURCE_COMMIT"] = WRAPPER_COMMIT
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if args.local_smoke else "0"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    seed = 1 if args.local_smoke else 42
    args.methods, args.seeds = [METHOD], [seed]
    if args.child:
        if os.environ.get("NCD_SECTION36_PARENT_PID") != str(os.getppid()):
            parser.error("Child must be launched by this guarded local parent")
        selected.run_child(args)
        return
    helper_path = ENTRY.with_name("run_section35_local.py")
    if sha(helper_path) != RESOURCE_HELPER_SHA:
        raise ValueError("Frozen local resource helper changed")
    resources = load_module("section36_local_resources", helper_path)
    cache = reference.check_cache(args.data_root, args.smoke_dataset if args.local_smoke else reference.DATASET)
    output, protected = args.output_root, args.protected_root
    output.mkdir(parents=True, exist_ok=True)
    protected.mkdir(parents=True, exist_ok=True)
    with global_lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "started_at": reference.utc(), "method": METHOD, "seed": seed,
                   "owner": "section36_selected_seed42"}, stream)
    child, cell_lock, copied, received = None, None, {}, []
    attempt = output / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"_local_{os.getpid()}")
    attempt.mkdir(parents=True)
    record = {"status": "running", "started_at": reference.utc(), "pid": os.getpid(),
        "local_launcher_commit": args.source_commit, "local_launcher_sha256": sha(ENTRY),
        "local_resource_helper_sha256": RESOURCE_HELPER_SHA, "training_source_commit": TRAINING_COMMIT,
        "selected_wrapper_commit": WRAPPER_COMMIT, "selected_package_sha256": PACKAGE_SHA,
        "methods": [METHOD], "seeds": [seed], "cache": cache, "smoke": args.local_smoke,
        "non_scientific": args.local_smoke, "resume": args.resume, "gpu_uuids": [],
        "scope": "Only reserved selected-main seed42, serial after local reference seed42; no HPC seed1/2/3/4 overlap",
        "aggregate_policy": "No formal aggregate produced or read by local launcher", "cells": [], "current_cell": None}
    reference.write_json(attempt / "job_start.json", record)

    def on_signal(number, _frame):
        received.append(number)
        if child is not None and child.poll() is None:
            child.terminate()

    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, on_signal)
    started = time.monotonic()
    try:
        identifiers = []
        if not args.local_smoke:
            identifiers = reference.visible_gpu_uuids()
            if len(identifiers) != 1 or not reference.gpu_sample(identifiers)["gpu_telemetry_valid"]:
                raise RuntimeError("Exactly one local CUDA device and valid UUID-targeted telemetry required")
        record["gpu_uuids"] = identifiers
        if received or (output / "STOP_REQUESTED").exists():
            raise RuntimeError("Owned stop requested; preserved prior checkpoints")
        target = reference.cell_root(output, METHOD, seed)
        if target.exists() and any(target.iterdir()) and not args.resume:
            raise ValueError("Existing cell requires explicit resume after prior parent/child terminal confirmation")
        locks = output / ".cell_locks"
        locks.mkdir(exist_ok=True)
        candidate_lock = locks / f"{METHOD}_seed{seed}.lock"
        with candidate_lock.open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "started_at": reference.utc()}, stream)
        cell_lock = candidate_lock
        command = [sys.executable, "-I", "-B", "-u", str(ENTRY), "--child", "--source-commit", args.source_commit]
        for name in ("code_root", "data_root", "output_root", "protected_root", "selected_package"):
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        if args.local_smoke:
            command.extend(["--local-smoke", "--smoke-dataset", args.smoke_dataset])
        current = {"method": METHOD, "seed": seed, "started_at": reference.utc(), "command": command}
        record["current_cell"] = current
        prefix = attempt / f"{METHOD}_seed{seed}"
        with prefix.with_suffix(".stdout.log").open("x", encoding="utf-8") as stdout, prefix.with_suffix(".stderr.log").open("x", encoding="utf-8") as stderr:
            started = time.monotonic()
            child = subprocess.Popen(command, cwd=args.code_root,
                env=dict(os.environ, NCD_SECTION36_PARENT_PID=str(os.getpid())),
                stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
            current["pid"] = child.pid
            reference.write_json(attempt / "status.json", record)
            next_sample, guard = 0., reference.TelemetryGuard()
            while child.poll() is None:
                if received or (output / "STOP_REQUESTED").exists():
                    raise RuntimeError("Owned stop requested; preserved checkpoints")
                elapsed = time.monotonic() - started
                if not args.local_smoke and elapsed >= next_sample:
                    measurement = resources.own_process_sample(reference, child.pid, output, identifiers)
                    reason = guard.observe(measurement, elapsed)
                    measurement.update(method=METHOD, seed=seed, child_elapsed_seconds=elapsed, guard_trigger=reason)
                    with (attempt / "resources.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(measurement, allow_nan=False) + "\n")
                    next_sample = elapsed + 60
                    if reason:
                        raise RuntimeError(reason)
                selected.protect_owned(output, protected, attempt, [METHOD], [seed], copied)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            current.update(exit_code=child.returncode, ended_at=reference.utc(), elapsed_seconds=time.monotonic() - started)
            if child.returncode != 0 or not selected.completed_path(output, METHOD, seed).exists():
                raise RuntimeError("Assigned selected-main cell failed; preserve logs and diagnose before resume")
        record["cells"].append(current)
        record["current_cell"] = None
        reference.write_json(attempt / "status.json", record)
        print(f"CELL_CLOSED method={METHOD} seed={seed} metrics=sealed", flush=True)
        if args.local_smoke:
            selected.release_if_complete(output, True, [METHOD])
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
        reference.write_json(attempt / "job_exit.json", record)
        selected.protect_owned(output, protected, attempt, [METHOD], [seed], copied)
        if cell_lock is not None:
            cell_lock.unlink(missing_ok=True)
        global_lock.unlink(missing_ok=True)
    print(f"ASSIGNMENT_COMPLETE physical_cells={len(record['cells'])} smoke={args.local_smoke} metrics=sealed", flush=True)


if __name__ == "__main__":
    main()
