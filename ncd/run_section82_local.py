"""Serial Section82 AKD new-seed append with guarded local CUDA execution.

The original Section44 package is verified without modification. An independent
AKD entrypoint is hash-bound beside its unchanged dependencies. CSE18 seeds5-9 are a separate descriptive append, not a new method verdict.
The supervisor reads execution structure, never partial model metrics.
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
import runpy
import shutil
import signal
import subprocess
import sys
import tarfile
import time

ENTRY = Path(__file__).resolve()
WORK = ENTRY.parents[2]
METHOD = "aacil_akd_wa"
SEEDS = (5, 6, 7, 8, 9)
DATASETS = ("CSE-CIC-IDS2018",)
PACKAGE_SHA = "3d5f190496f89dd07d10401f40a5e4ea2b0beaaa322e0f32dbc1f00c6dece0d1"
WRAPPER_COMMIT = "359102b09f389a8c1d59b60d973dd8129c1dd5a7"
AKD_SHA = "25bf7262dc42cc27e5706c6d264f77b27ef9dfc8a5d89373abab254e14583fbb"
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
    module = load("section82_local_frozen", args.code_root / "ncd/hpc/section44_job.py")
    module.source_binding()
    if sha(args.code_root / "ncd/run_section82_akd.py") != AKD_SHA:
        raise ValueError("Published AKD entrypoint changed")
    return module


def check_cache(ref, data_root, dataset):
    if dataset != "CIC-IDS-2017":
        return ref.check_cache(data_root, dataset)
    folder = data_root / "cic-ids-2017"
    receipt = folder / "data_quality.json"
    info = ref.read(receipt)
    counts = sum(info["train_counts"].values()), sum(info["id_counts"].values())
    files = info.get("cache_files", [])
    required = {f"{split}_{field}.npy" for split in ("train", "test", "shift") for field in ("x", "y")}
    if (info.get("status") != "complete" or info.get("dataset") != dataset
            or counts != ref.FULL_COUNTS["cic17"] or len(files) != 6
            or {r["path"] for r in files} != required):
        raise ValueError("Complete original six-array CIC17 cache required")
    for row in files:
        if (folder / row["path"]).stat().st_size != row["bytes"]:
            raise ValueError("CIC17 array size changed")
    return {"dataset": dataset, "receipt_sha256": sha(receipt),
            "train_rows": counts[0], "test_rows": counts[1],
            "shift_role": "Legacy loader validates shift arrays only; no shift training, inference or metrics"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("code-root", "data-root", "output-root", "protected-root", "package"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-smoke", action="store_true")
    parser.add_argument("--child-dataset", choices=(*DATASETS, "NSL-KDD"))
    parser.add_argument("--child-seed", type=int, choices=(*SEEDS, 5))
    args = parser.parse_args()
    if os.name != "nt" or os.environ.get("SLURM_JOB_ID"):
        parser.error("Windows local development execution only")
    if not re.fullmatch("[0-9a-f]{40}", args.source_commit):
        parser.error("Full local launcher publication commit required")
    for name in ("code_root", "data_root", "output_root", "protected_root", "package"):
        setattr(args, name, getattr(args, name).resolve())
    if args.code_root.parent != WORK:
        parser.error("Frozen package must be directly under workspace work")
    if args.output_root.is_relative_to(args.protected_root) or args.protected_root.is_relative_to(args.output_root):
        parser.error("Disjoint active/protected roots required")
    lock = WORK / (".section82_cpu_smoke.lock" if args.local_smoke else ".section35_local_cuda.lock")
    if not args.child_dataset and lock.exists():
        parser.error("Existing queue lock; never duplicate or blindly remove")
    wrapper = load_package(args)
    ref = wrapper.ref
    helper = ENTRY.with_name("run_section35_local.py")
    if sha(helper) != RESOURCE_SHA:
        raise ValueError("Frozen resource helper changed")
    resources = load("section82_local_resources", helper)
    config = ref.read(args.code_root / "ncd/configs/problem_scout_section21.json")
    if (args.code_root / config["problem_scout_data_root"]).resolve() != args.data_root:
        raise ValueError("Original relative data configuration required")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if args.local_smoke else "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["NCD_SOURCE_COMMIT"] = args.source_commit
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("Duplicate dataset assignment")
    cells = [("NSL-KDD", 5)] if args.local_smoke else [(d, s) for d in args.datasets for s in SEEDS]
    if args.child_dataset:
        if ((args.child_dataset, args.child_seed) not in cells
                or os.environ.get("NCD_SECTION82_LOCAL_PARENT_PID") != str(os.getppid())):
            parser.error("Child must belong to the guarded local assignment")
        folder = ref.cell_root(args.output_root / args.child_dataset.lower().replace("-", "_"),
                               METHOD, args.child_seed)
        folder.mkdir(parents=True, exist_ok=True)
        raw = folder / "raw"
        entry = args.code_root / "ncd/run_section82_akd.py"
        argv = [str(entry), "--datasets", args.child_dataset, "--methods", METHOD,
                "--seeds", str(args.child_seed), "--lambda-g", "0.5",
                "--der-alpha", "1", "--der-beta", "1",
                "--data-root", str(args.data_root), "--output-dir", str(raw),
                "--device", "cpu" if args.local_smoke else "cuda",
                "--source-commit", args.source_commit]
        if args.local_smoke:
            argv.append("--smoke")
        invocation = {"actual_entrypoint": "ncd/run_section82_local.py",
            "training_entrypoint": "ncd/run_section82_akd.py",
            "launcher_commit": args.source_commit, "launcher_sha256": sha(ENTRY),
            "training_entrypoint_sha256": AKD_SHA, "dependency_package_sha256": PACKAGE_SHA,
            "dependency_wrapper_commit": WRAPPER_COMMIT,
            "dataset": args.child_dataset, "method": METHOD, "seed": args.child_seed,
            "smoke": args.local_smoke, "post_unsealing_descriptive_append": True,
            "selection_changed": False, "supervisor_reads_model_metrics": False,
            "legacy_shift_cache_validation_only": args.child_dataset == "CIC-IDS-2017",
            "command": subprocess.list2cmdline([sys.executable, "-I", "-B", *argv])}
        ref.write_json(folder / "section82_local_invocation.json", invocation)
        sys.argv = argv
        runpy.run_path(str(entry), run_name="__main__")
        matrix = ref.read(raw / "result.json")
        if (matrix["execution_status"] != "complete" or len(matrix["cells"]) != 1
                or matrix["expected"] != {"datasets": [args.child_dataset],
                    "methods": [METHOD], "seeds": [args.child_seed]}):
            raise RuntimeError("AKD cell completion or assignment mismatch")
        ref.write_json(folder / "section82_result.json", {
            "completed": True, "dataset": args.child_dataset, "method": METHOD,
            "seed": args.child_seed, "smoke": args.local_smoke,
            "raw_matrix_sha256": sha(raw / "result.json"), "invocation": invocation,
            "aggregate_metrics_generated": False})
        return
    caches = [check_cache(ref, args.data_root, d) for d in dict.fromkeys(d for d, _ in cells)]
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        parser.error("Existing output requires diagnosed explicit resume")
    if args.protected_root.exists() and not args.resume:
        parser.error("Protected root already exists")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.protected_root.mkdir(parents=True, exist_ok=True)
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "started_at": ref.utc(), "owner": "section82_akd_append"}, stream)
    attempt = args.output_root / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"_local_{os.getpid()}")
    attempt.mkdir(parents=True)
    record = {"status": "running", "started_at": ref.utc(), "pid": os.getpid(),
        "launcher_commit": args.source_commit, "launcher_sha256": sha(ENTRY),
        "dependency_package_sha256": PACKAGE_SHA, "dependency_wrapper_commit": WRAPPER_COMMIT,
        "training_entrypoint_sha256": AKD_SHA, "caches": caches,
        "smoke": args.local_smoke, "resume": args.resume, "planned_cells": cells,
        "cells": [], "current_cell": None, "scope": "5 CSE18 new-seed AKD cells; separate from original40; no tuning or revised verdict"}
    ref.write_json(attempt / "job_start.json", record)
    child, cell_lock, copied, received = None, None, {}, []

    def protect():
        ref.copy_small(attempt, args.protected_root / attempt.relative_to(args.output_root), copied.setdefault("attempt", {}))
        for dataset, seed in cells:
            slug = dataset.lower().replace("-", "_")
            folder = ref.cell_root(args.output_root / slug, METHOD, seed)
            if folder.exists():
                ref.copy_small(folder, args.protected_root / folder.relative_to(args.output_root),
                               copied.setdefault(f"{slug}/{seed}", {}))
        completion = args.output_root / "append_completion.json"
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
        for dataset, seed in cells:
            if received or (args.output_root / "STOP_REQUESTED").exists():
                raise RuntimeError("Stop requested")
            slug = dataset.lower().replace("-", "_")
            locks = args.output_root / ".cell_locks"
            locks.mkdir(exist_ok=True)
            cell_lock = locks / f"{slug}_{METHOD}_seed{seed}.lock"
            with cell_lock.open("x", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid(), "utc": ref.utc()}, stream)
            command = [sys.executable, "-I", "-B", "-u", str(ENTRY), "--source-commit", args.source_commit,
                       "--child-dataset", dataset, "--child-seed", str(seed), "--datasets", *args.datasets]
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
                    env=dict(os.environ, NCD_SECTION82_LOCAL_PARENT_PID=str(os.getpid())),
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
                if child.returncode != 0 or not (ref.cell_root(args.output_root / slug, METHOD, seed) / "section82_result.json").exists():
                    raise RuntimeError("Cell failed; inspect preserved logs before explicit resume")
            record["cells"].append(current)
            record["current_cell"] = None
            ref.write_json(attempt / "status.json", record)
            protect()
            cell_lock.unlink()
            cell_lock = None
            print(f"CELL_CLOSED dataset={dataset} method={METHOD} seed={seed}", flush=True)
        record.update(status="complete", ended_at=ref.utc())
        ref.write_json(args.output_root / "append_completion.json", {
            "completed": True, "smoke": args.local_smoke, "cells": record["cells"],
            "post_unsealing_descriptive_append": True, "selection_changed": False,
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
