"""Serial CIC-2017 descriptive completion, using the immutable selected package.

Only five fixed KD+WA seeds and the missing Replay-BS seed42 are assigned here.
Raw training/attempts use a new root; the existing Section28 analysis is not
overwritten by training. CSE18 jobs, model results and release are untouched.
The legacy CIC17 cache loader validates its stored shift arrays but does not
evaluate them; training and saved confusion matrices use CIC17 TRAIN/TEST only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
WORK = ENTRY.parents[2]
LOADER_SHA = "3991d922667078d30e7f7a01b1fde717b8c734b8aaeb117d48dd3cfae794ad34"
RESOURCE_SHA = "bf8a5aa83b033927966d26dcf0d834ad27008fa4bfc1622a79c8a2dd5721239b"
CELLS = [("aacil_kd_wa", seed) for seed in (1, 2, 3, 4, 42)] + [("replay_bs", 42)]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_cache(ref, data_root, dataset):
    if dataset != "CIC-IDS-2017":
        return ref.check_cache(data_root, dataset)
    folder = data_root / "cic-ids-2017"
    receipt = folder / "data_quality.json"
    info = ref.read(receipt)
    counts = (sum(info["train_counts"].values()), sum(info["id_counts"].values()))
    if (info.get("status") != "complete" or info.get("dataset") != dataset
            or counts != ref.FULL_COUNTS["cic17"]):
        raise ValueError("Complete uncapped CIC17 cache required")
    required = {f"{split}_{field}.npy" for split in ("train", "test", "shift") for field in ("x", "y")}
    files = info.get("cache_files", [])
    if len(files) != 6 or {row["path"] for row in files} != required:
        raise ValueError("Legacy CIC17 receipt must identify all six arrays exactly once")
    for row in files:
        if (folder / row["path"]).stat().st_size != row["bytes"]:
            raise ValueError("Prepared array is absent or has the wrong size")
    return {"dataset": dataset, "receipt_sha256": sha(receipt),
            "train_rows": counts[0], "test_rows": counts[1], "task_count": len(info["tasks"]),
            "array_hash_validation": "Frozen loader validates six arrays in child; shift arrays are not evaluated",
            "shift_role": "Legacy cache validation only, no training, inference or metrics"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("code-root", "data-root", "output-root", "protected-root", "selected-package"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-smoke", action="store_true")
    parser.add_argument("--child-method", choices=("aacil_kd_wa", "replay_bs"))
    parser.add_argument("--child-seed", type=int, choices=(1, 2, 3, 4, 42))
    args = parser.parse_args()
    if os.name != "nt" or os.environ.get("SLURM_JOB_ID"):
        parser.error("Windows local execution only")
    if not re.fullmatch("[0-9a-f]{40}", args.source_commit):
        parser.error("Full launcher publication SHA required")
    for name in ("code_root", "data_root", "output_root", "protected_root", "selected_package"):
        setattr(args, name, getattr(args, name).resolve())
    if args.code_root.parent != WORK:
        parser.error("Frozen package must be directly under workspace work")
    if args.output_root.is_relative_to(args.protected_root) or args.protected_root.is_relative_to(args.output_root):
        parser.error("Disjoint active and protected roots required")
    lock = WORK / (".section39_cpu_smoke.lock" if args.local_smoke else ".section35_local_cuda.lock")
    if not args.child_method and lock.exists():
        parser.error("Existing local queue lock; do not duplicate or remove it")
    loader_path = ENTRY.with_name("run_section36_local.py")
    helper_path = ENTRY.with_name("run_section35_local.py")
    if sha(loader_path) != LOADER_SHA or sha(helper_path) != RESOURCE_SHA:
        raise ValueError("Frozen package loader/resource helper changed")
    loader = load("section39_package_loader", loader_path)
    selected = loader.load_selected(args)
    ref = selected.supervisor
    resources = load("section39_resources", helper_path)
    config = ref.read(args.code_root / "ncd/configs/problem_scout_section21.json")
    if (args.code_root / config["problem_scout_data_root"]).resolve() != args.data_root:
        raise ValueError("Use original relative full-cache configuration")
    # This new invocation selects CIC17 in memory; no frozen file is edited.
    ref.DATASET = "CIC-IDS-2017"
    ref.DATASET_KEYS[ref.DATASET] = "cic17"
    args.smoke_dataset = "NSL-KDD"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if args.local_smoke else "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    cells = [("aacil_kd_wa", 1), ("replay_bs", 1)] if args.local_smoke else CELLS
    if args.child_method:
        if ((args.child_method, args.child_seed) not in cells
                or os.environ.get("NCD_SECTION39_PARENT_PID") != str(os.getppid())):
            parser.error("Child must belong to this serial parent assignment")
        args.methods, args.seeds = [args.child_method], [args.child_seed]
        # Frozen callable identities remain separate from the new invocation.
        os.environ["NCD_SOURCE_COMMIT"] = loader.WRAPPER_COMMIT if args.child_method == selected.METHOD else "270121272848087e4af85ec5290eaede0102d776"
        if args.child_method == selected.METHOD:
            selected.run_child(args)
        else:
            ref.run_child(args)
        folder = ref.cell_root(args.output_root, args.child_method, args.child_seed)
        ref.write_json(folder / "section39_invocation.json", {
            "launcher_commit": args.source_commit, "launcher_sha256": sha(ENTRY),
            "actual_entrypoint": "ncd/run_section39_local.py",
            "frozen_callable": "section32_job.run_child" if args.child_method == selected.METHOD else "section31_job.run_child",
            "dataset": args.smoke_dataset if args.local_smoke else ref.DATASET,
            "method": args.child_method, "seed": args.child_seed,
            "smoke": args.local_smoke, "descriptive_development_dataset": True,
            "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
            "selection_changed": False, "CSE18_model_metrics_accessed": False,
            "legacy_shift_cache_validation_only": not args.local_smoke})
        return
    cache = check_cache(ref, args.data_root, args.smoke_dataset if args.local_smoke else ref.DATASET)
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        parser.error("Existing output requires diagnosed explicit resume, never a duplicate launch")
    if args.protected_root.exists() and not args.resume:
        parser.error("Protected root already exists")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.protected_root.mkdir(parents=True, exist_ok=True)
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "started_at": ref.utc(), "owner": "section39_cic"}, stream)
    attempt = args.output_root / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"_local_{os.getpid()}")
    attempt.mkdir(parents=True)
    record = {"status": "running", "started_at": ref.utc(), "pid": os.getpid(),
        "launcher_commit": args.source_commit, "launcher_sha256": sha(ENTRY),
        "training_commit": loader.TRAINING_COMMIT, "package_sha256": loader.PACKAGE_SHA,
        "selection_sha256": selected.SELECTION_SHA, "cache": cache,
        "smoke": args.local_smoke, "resume": args.resume,
        "planned_cells": cells, "cells": [], "current_cell": None,
        "scope": "CIC17 descriptive fixed candidate and missing Replay-BS only; no CSE18 model metrics",
        "legacy_shift_cache_role": "Frozen CIC17 loader validates stored shift arrays; they are not evaluated"}
    ref.write_json(attempt / "job_start.json", record)
    child, cell_lock, copied, received = None, None, {}, []

    def on_signal(number, _frame):
        received.append(number)
        if child is not None and child.poll() is None:
            child.terminate()

    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, on_signal)
    try:
        identifiers = [] if args.local_smoke else ref.visible_gpu_uuids()
        if not args.local_smoke and (len(identifiers) != 1 or not ref.gpu_sample(identifiers)["gpu_telemetry_valid"]):
            raise RuntimeError("One selected GPU with valid UUID telemetry required")
        record["gpu_uuids"] = identifiers
        for method, seed in cells:
            if received or (args.output_root / "STOP_REQUESTED").exists():
                raise RuntimeError("Stop requested; preserve checkpoints")
            locks = args.output_root / ".cell_locks"
            locks.mkdir(exist_ok=True)
            cell_lock = locks / f"{method}_seed{seed}.lock"
            with cell_lock.open("x", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid(), "started_at": ref.utc()}, stream)
            command = [sys.executable, "-I", "-B", "-u", str(ENTRY), "--source-commit", args.source_commit,
                       "--child-method", method, "--child-seed", str(seed)]
            for name in ("code_root", "data_root", "output_root", "protected_root", "selected_package"):
                command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
            if args.local_smoke:
                command.append("--local-smoke")
            current = {"method": method, "seed": seed, "started_at": ref.utc(), "command": command}
            record["current_cell"] = current
            prefix = attempt / f"{method}_seed{seed}"
            child, started = None, time.monotonic()
            with prefix.with_suffix(".stdout.log").open("x", encoding="utf-8") as stdout, prefix.with_suffix(".stderr.log").open("x", encoding="utf-8") as stderr:
                child = subprocess.Popen(command, cwd=args.code_root,
                    env=dict(os.environ, NCD_SECTION39_PARENT_PID=str(os.getpid())),
                    stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
                current["pid"] = child.pid
                ref.write_json(attempt / "status.json", record)
                started, next_sample, guard = time.monotonic(), 0., ref.TelemetryGuard()
                while child.poll() is None:
                    if received or (args.output_root / "STOP_REQUESTED").exists():
                        raise RuntimeError("Stop requested")
                    elapsed = time.monotonic() - started
                    if not args.local_smoke and elapsed >= next_sample:
                        measurement = resources.own_process_sample(ref, child.pid, args.output_root, identifiers)
                        reason = guard.observe(measurement, elapsed)
                        measurement.update(method=method, seed=seed, child_elapsed_seconds=elapsed, guard_trigger=reason)
                        with (attempt / "resources.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(measurement, allow_nan=False) + "\n")
                        next_sample = elapsed + 60
                        if reason:
                            raise RuntimeError(reason)
                    ref.protect_owned(args.output_root, args.protected_root, attempt, [method], [seed], copied)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                current.update(exit_code=child.returncode, ended_at=ref.utc(), elapsed_seconds=time.monotonic() - started)
                result = (selected if method == selected.METHOD else ref).completed_path(args.output_root, method, seed)
                if child.returncode != 0 or not result.exists():
                    raise RuntimeError(f"Cell {method}/{seed} failed; diagnose before explicit resume")
            record["cells"].append(current)
            record["current_cell"] = None
            ref.write_json(attempt / "status.json", record)
            ref.protect_owned(args.output_root, args.protected_root, attempt, [method], [seed], copied)
            cell_lock.unlink()
            cell_lock = None
            print(f"CELL_CLOSED method={method} seed={seed}", flush=True)
        record.update(status="complete", ended_at=ref.utc())
    except BaseException as error:
        ref.terminate(child)
        if record["current_cell"] is not None and child is not None:
            record["current_cell"].update(exit_code=child.returncode, ended_at=ref.utc(), elapsed_seconds=time.monotonic() - started)
        record.update(status="failed", ended_at=ref.utc(), error_type=type(error).__name__, error=str(error))
        raise
    finally:
        ref.terminate(child)
        ref.write_json(attempt / "status.json", record)
        ref.write_json(attempt / "job_exit.json", record)
        for method, seed in cells:
            ref.protect_owned(args.output_root, args.protected_root, attempt, [method], [seed], copied)
        if cell_lock is not None:
            cell_lock.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
    print(f"ASSIGNMENT_COMPLETE cells={len(record['cells'])} smoke={args.local_smoke}", flush=True)


if __name__ == "__main__":
    main()
