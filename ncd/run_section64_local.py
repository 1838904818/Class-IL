"""Section64 disjoint local CSE18 seeds 5 and 6, serial original-eight methods.

Owner Section64 supersedes the original release schedule without changing the
frozen training package. AKD is a separate pending extension, not an alias.
This supervisor does not read or release model aggregates.
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
import shutil
import signal
import subprocess
import sys
import tarfile
import time

ENTRY = Path(__file__).resolve()
WORK = ENTRY.parents[2]
METHODS = ("aacil_nogkd_wa", "replay_bs", "aacil", "freq", "wacb",
           "aacil_kd_wa", "replay", "derpp")
SEEDS = (5, 6)
DATASETS = ("CSE-CIC-IDS2018",)
PACKAGE_SHA = "3d5f190496f89dd07d10401f40a5e4ea2b0beaaa322e0f32dbc1f00c6dece0d1"
WRAPPER_COMMIT = "359102b09f389a8c1d59b60d973dd8129c1dd5a7"
RESOURCE_SHA = "bf8a5aa83b033927966d26dcf0d834ad27008fa4bfc1622a79c8a2dd5721239b"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_package(args):
    if sha(args.package) != PACKAGE_SHA:
        raise ValueError("Exact Section44 source package required")
    with tarfile.open(args.package, "r") as archive:
        for item in archive.getmembers():
            path = (args.code_root / item.name).resolve()
            if not item.isfile() or not path.is_relative_to(args.code_root):
                raise ValueError("Invalid source archive member")
            if sha(path) != hashlib.sha256(archive.extractfile(item).read()).hexdigest():
                raise ValueError(f"Frozen source differs: {item.name}")
    module = load("section64_local_frozen", args.code_root / "ncd/hpc/section44_job.py")
    module.source_binding()
    return module


def check_cache(ref, data_root, dataset):
    return ref.check_cache(data_root, dataset)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("code-root", "data-root", "output-root", "protected-root", "package"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=[5])
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-smoke", action="store_true")
    parser.add_argument("--child-dataset", choices=(*DATASETS, "NSL-KDD"))
    parser.add_argument("--child-seed", type=int, choices=(1, *SEEDS))
    parser.add_argument("--child-method", choices=METHODS)
    args = parser.parse_args()
    if os.name != "nt" or os.environ.get("SLURM_JOB_ID"):
        parser.error("Windows disjoint local execution only")
    if not re.fullmatch("[0-9a-f]{40}", args.source_commit):
        parser.error("Full local launcher publication commit required")
    for name in ("code_root", "data_root", "output_root", "protected_root", "package"):
        setattr(args, name, getattr(args, name).resolve())
    if args.code_root.parent != WORK:
        parser.error("Frozen package must be directly under workspace work")
    if args.output_root.is_relative_to(args.protected_root) or args.protected_root.is_relative_to(args.output_root):
        parser.error("Disjoint active/protected roots required")
    lock = WORK / (".section64_cpu_smoke.lock" if args.local_smoke else ".section35_local_cuda.lock")
    if not args.child_dataset and lock.exists():
        parser.error("Existing queue lock; never duplicate or blindly remove")
    wrapper = load_package(args)
    ref = wrapper.ref
    helper = ENTRY.with_name("run_section35_local.py")
    if sha(helper) != RESOURCE_SHA:
        raise ValueError("Frozen resource helper changed")
    resources = load("section64_local_resources", helper)
    config = ref.read(args.code_root / "ncd/configs/problem_scout_section21.json")
    if (args.code_root / config["problem_scout_data_root"]).resolve() != args.data_root:
        raise ValueError("Original relative data configuration required")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if args.local_smoke else "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["NCD_SOURCE_COMMIT"] = WRAPPER_COMMIT
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate assignment")
    cells = [("NSL-KDD", m, 1) for m in METHODS] if args.local_smoke else [(d, m, s) for d in args.datasets for s in args.seeds for m in METHODS]
    if args.child_dataset:
        if ((args.child_dataset, args.child_method, args.child_seed) not in cells
                or os.environ.get("NCD_SECTION64_LOCAL_PARENT_PID") != str(os.getppid())):
            parser.error("Child must belong to the guarded local assignment")
        METHOD = args.child_method
        wrapper.configure_supervisor()
        ref.DATASET = args.child_dataset
        wrapper.SEEDS = SEEDS
        wrapper.DISCLOSURE = "Section64 disjoint local seeds5/6; Section61-64 per-seed five-decision-cell descriptive interim, final AR judgment after25 decision cells; original40 completed and reported in full; same exposed dataset, no tuning or original selection change"
        original_config = wrapper.configuration

        def local_config(settings, method, smoke):
            config = original_config(settings, method, smoke)
            config.update(release_policy="section64_interim_five_per_seed_final_25_decision_full_40",
                          section64=True, previously_released_seeds=False, exploratory=False,
                          descriptive_development_dataset=True,
                          local_launcher_commit=args.source_commit, local_launcher_sha256=sha(ENTRY))
            return config

        wrapper.configuration = local_config
        ref.configuration = local_config
        args.methods, args.seeds = [METHOD], [args.child_seed]
        args.smoke_dataset = "NSL-KDD"
        args.output_root = args.output_root / args.child_dataset.lower().replace("-", "_")
        wrapper.run_child(args)
        ref.write_json(ref.cell_root(args.output_root, METHOD, args.child_seed) / "section64_local_invocation.json", {
            "actual_entrypoint": "ncd/run_section64_local.py", "launcher_commit": args.source_commit,
            "launcher_sha256": sha(ENTRY), "package_sha256": PACKAGE_SHA,
            "frozen_callable": "section44_job.run_child", "wrapper_commit": WRAPPER_COMMIT,
            "dataset": args.child_dataset, "method": METHOD, "seed": args.child_seed,
            "smoke": args.local_smoke, "descriptive_development_dataset": True,
            "selection_changed": False, "CSE18_released_seed_exploration": False,
            "owner_release_revision": "WORK_ORDER sections61-64",
            "cancelled_zero_cost_hpc_jobs": [489800, 489801],
            "new_seed_training": not args.local_smoke,
            "legacy_shift_cache_validation_only": args.child_dataset == "CIC-IDS-2017",
            "command": subprocess.list2cmdline([sys.executable, *sys.argv])})
        return
    caches = [check_cache(ref, args.data_root, d) for d in dict.fromkeys(d for d, _, _ in cells)]
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        parser.error("Existing output requires diagnosed explicit resume")
    if args.protected_root.exists() and not args.resume:
        parser.error("Protected root already exists")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.protected_root.mkdir(parents=True, exist_ok=True)
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "started_at": ref.utc(), "owner": "section64_local_seeds5_6"}, stream)
    attempt = args.output_root / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"_local_{os.getpid()}")
    attempt.mkdir(parents=True)
    record = {"status": "running", "started_at": ref.utc(), "pid": os.getpid(),
        "launcher_commit": args.source_commit, "launcher_sha256": sha(ENTRY),
        "package_sha256": PACKAGE_SHA, "wrapper_commit": WRAPPER_COMMIT, "caches": caches,
        "smoke": args.local_smoke, "resume": args.resume, "planned_cells": cells,
        "cells": [], "current_cell": None, "scope": "Disjoint local seeds5/6 original8 methods; AKD extension remains pending; no aggregate reads",
        "cancelled_zero_cost_hpc_jobs": [489800, 489801],
        "owner_release_revision": "WORK_ORDER sections61-64"}
    ref.write_json(attempt / "job_start.json", record)
    child, cell_lock, copied, received = None, None, {}, []

    def protect():
        ref.copy_small(attempt, args.protected_root / attempt.relative_to(args.output_root), copied.setdefault("attempt", {}))
        for dataset, METHOD, seed in cells:
            slug = dataset.lower().replace("-", "_")
            folder = ref.cell_root(args.output_root / slug, METHOD, seed)
            if folder.exists():
                ref.copy_small(folder, args.protected_root / folder.relative_to(args.output_root),
                               copied.setdefault(f"{slug}/{METHOD}/{seed}", {}))
        completion = args.output_root / "development_completion.json"
        if completion.exists():
            target = args.protected_root / completion.name
            temporary = target.with_suffix(".json.tmp")
            shutil.copyfile(completion, temporary)
            temporary.replace(target)

    def on_signal(number, _frame):
        received.append(number)
        if child is not None and child.poll() is None:
            child.terminate()

    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, on_signal)
    try:
        identifiers = [] if args.local_smoke else ref.visible_gpu_uuids()
        if not args.local_smoke and (len(identifiers) != 1 or not ref.gpu_sample(identifiers)["gpu_telemetry_valid"]):
            raise RuntimeError("One selected local CUDA UUID with valid telemetry required")
        record["gpu_uuids"] = identifiers
        for dataset, METHOD, seed in cells:
            if received or (args.output_root / "STOP_REQUESTED").exists():
                raise RuntimeError("Stop requested")
            slug = dataset.lower().replace("-", "_")
            locks = args.output_root / ".cell_locks"
            locks.mkdir(exist_ok=True)
            cell_lock = locks / f"{slug}_{METHOD}_seed{seed}.lock"
            with cell_lock.open("x", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid(), "utc": ref.utc()}, stream)
            command = [sys.executable, "-I", "-B", "-u", str(ENTRY), "--source-commit", args.source_commit,
                       "--child-dataset", dataset, "--child-seed", str(seed), "--child-method", METHOD,
                       "--seeds", *map(str, args.seeds), "--datasets", *args.datasets]
            for name in ("code_root", "data_root", "output_root", "protected_root", "package"):
                command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
            if args.local_smoke:
                command.append("--local-smoke")
            current = {"dataset": dataset, "method": METHOD, "seed": seed,
                       "started_at": ref.utc(), "command": command}
            record["current_cell"] = current
            prefix = attempt / f"{slug}_{METHOD}_seed{seed}"
            child, started = None, time.monotonic()
            with prefix.with_suffix(".stdout.log").open("x", encoding="utf-8") as stdout, prefix.with_suffix(".stderr.log").open("x", encoding="utf-8") as stderr:
                child = subprocess.Popen(command, cwd=args.code_root,
                    env=dict(os.environ, NCD_SECTION64_LOCAL_PARENT_PID=str(os.getpid())),
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
                        measurement.update(dataset=dataset, method=METHOD, seed=seed, child_elapsed_seconds=elapsed, guard_trigger=reason)
                        with (attempt / "resources.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(measurement, allow_nan=False) + "\n")
                        next_sample = elapsed + 60
                        if reason:
                            raise RuntimeError(reason)
                    protect()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                current.update(exit_code=child.returncode, ended_at=ref.utc(), elapsed_seconds=time.monotonic() - started)
                if child.returncode != 0 or not wrapper.completed_path(args.output_root / slug, METHOD, seed).exists():
                    raise RuntimeError("Cell failed; inspect preserved logs before explicit resume")
            record["cells"].append(current)
            record["current_cell"] = None
            ref.write_json(attempt / "status.json", record)
            protect()
            cell_lock.unlink()
            cell_lock = None
            print(f"CELL_CLOSED dataset={dataset} method={METHOD} seed={seed}", flush=True)
        record.update(status="complete", ended_at=ref.utc())
        ref.write_json(args.output_root / "development_completion.json", {
            "completed": True, "smoke": args.local_smoke, "cells": record["cells"],
            "descriptive_development_dataset": True, "selection_changed": False,
            "aggregate_metrics_generated": False})
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
        protect()
        if cell_lock is not None:
            cell_lock.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
    print(f"ASSIGNMENT_COMPLETE cells={len(record['cells'])} smoke={args.local_smoke}", flush=True)


if __name__ == "__main__":
    main()
