"""Run the approved NSL-KDD ER-Bal-TabT seeds 10-14 on one local GPU.

This Windows-only supervisor launches one formal cell at a time, samples only
the selected local GPU and its own child, preserves small evidence outside
the active run directory, and publishes a path-redacted result after each
cell. It never submits or manages a DICC job.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "ncd/hpc/tabt_cpu_gpu_portable.py"
SEEDS = (10, 11, 12, 13, 14)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def redact(value, replacements):
    if isinstance(value, dict):
        return {key: redact(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item, replacements) for item in value]
    if isinstance(value, str):
        for original, replacement in replacements:
            value = value.replace(original, replacement)
        return value
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dependency-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protected-root", type=Path, required=True)
    parser.add_argument("--publication-root", type=Path, required=True)
    args = parser.parse_args()
    if os.name != "nt" or os.environ.get("SLURM_JOB_ID"):
        parser.error("This supervisor is for local Windows execution only")
    args.data_root = args.data_root.resolve(strict=True)
    args.dependency_root = args.dependency_root.resolve(strict=True)
    args.run_root = args.run_root.resolve()
    args.protected_root = args.protected_root.resolve()
    args.publication_root = args.publication_root.resolve()
    roots = (args.run_root, args.protected_root, args.publication_root)
    if any(a == b or a in b.parents or b in a.parents
           for index, a in enumerate(roots) for b in roots[index + 1:]):
        parser.error("Active, protected, and publication roots must be disjoint")
    if args.run_root.exists() or args.protected_root.exists():
        parser.error("Existing run/protected output requires diagnosis; overwrite is forbidden")

    # Reuse the project's UUID-targeted telemetry and sampled low-utilization guard.
    reference = load_module("nsl_er_local_reference", ROOT / "ncd/hpc/problem_scout_job.py")
    resources = load_module("nsl_er_local_resources", ROOT / "ncd/run_section35_local.py")
    import torch
    import numpy as np
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    identifiers = reference.visible_gpu_uuids()
    preflight = reference.gpu_sample(identifiers)
    if len(identifiers) != 1 or not preflight.get("gpu_telemetry_valid"):
        raise RuntimeError("One local CUDA device with valid UUID-targeted telemetry is required")
    cache_meta = json.loads((args.data_root / "nsl-kdd/data_quality.json").read_text(encoding="utf-8"))
    if cache_meta.get("status") != "complete" or cache_meta.get("dataset") != "NSL-KDD":
        raise ValueError("Prepared NSL-KDD Section21 cache is not complete")
    for entry in cache_meta["cache_files"]:
        path = args.data_root / "nsl-kdd" / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError(f"Prepared NSL cache receipt mismatch: {entry['path']}")

    args.run_root.mkdir(parents=True)
    args.protected_root.mkdir(parents=True)
    args.publication_root.mkdir(parents=True, exist_ok=True)
    replacements = [(str(args.data_root), "<prepared-data-root>"),
                    (str(args.dependency_root), "<pinned-tabt-dependencies>"),
                    (str(args.run_root), "<active-run-root>"),
                    (str(args.protected_root), "<protected-run-root>")]
    run_started = time.monotonic()
    top = {"status": "running", "started_at": reference.utc(), "pid": os.getpid(),
           "repo_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
           "dataset": "NSL-KDD", "method": "ER-Bal-TabT", "seeds": list(SEEDS),
           "memory_setting": "fixed200", "protocol": "Section21/C feature typing; 10% Normal TRAIN holdout; 1% calibrated view plus default; full TRAIN prior; no TEST calibration",
           "data_cache": {"metadata_sha256": sha(args.data_root / "nsl-kdd/data_quality.json"),
                          "cache_files": cache_meta["cache_files"]},
           "python": sys.version.split()[0], "torch": str(torch.__version__),
           "numpy": np.__version__, "platform": platform.platform(),
           "gpu_uuids": identifiers, "gpu_preflight": preflight,
           "telemetry_policy": {"sample_interval_seconds": 60, "low_utilization_percent": 10,
                                "low_span_threshold": "observed child elapsed / 3",
                                "interpretation": "sampled guard; not proof of continuous utilization"},
           "cells": [], "current_seed": None}
    write_json(args.run_root / "supervisor.json", top)
    write_json(args.protected_root / "supervisor.json", top)
    received, child = [], None

    def stop(number, _frame):
        received.append(number)
        if child is not None and child.poll() is None:
            child.terminate()

    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, stop)
    exit_code = 1
    try:
        for seed in SEEDS:
            if received or (args.run_root / "STOP_REQUESTED").exists():
                raise RuntimeError("Stop requested; preserving completed cells")
            run_dir = args.run_root / f"seed{seed}"
            protected_dir = args.protected_root / f"seed{seed}"
            run_dir.mkdir()
            protected_dir.mkdir()
            raw_dir = run_dir / "raw"
            command = [sys.executable, "-I", "-B", "-u", str(ENTRY),
                       "--dataset", "NSL-KDD", "--memory-setting", "fixed200",
                       "--seed", str(seed), "--confirmation", "--data-root", str(args.data_root),
                       "--dependency-root", str(args.dependency_root), "--output-dir", str(raw_dir)]
            record = {"seed": seed, "status": "running", "started_at": reference.utc(),
                      "command": [redact(item, replacements) for item in command]}
            top["current_seed"] = seed
            write_json(args.run_root / "supervisor.json", top)
            guard, copied, elapsed, next_sample = reference.TelemetryGuard(), {}, 0.0, 0.0
            child_env = dict(os.environ, NCD_FROZEN_ROOT=str(ROOT), NCD_DATA_ROOT=str(args.data_root),
                             NCD_SECTION81_PARENT_PID=str(os.getpid()), CUDA_VISIBLE_DEVICES="0",
                             OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                             NUMEXPR_NUM_THREADS="2", CUBLAS_WORKSPACE_CONFIG=":4096:8")
            with (run_dir / "stdout.log").open("x", encoding="utf-8") as stdout, \
                    (run_dir / "stderr.log").open("x", encoding="utf-8") as stderr:
                child = subprocess.Popen(command, cwd=ROOT, env=child_env, stdout=stdout, stderr=stderr,
                                         creationflags=subprocess.CREATE_NO_WINDOW)
                record["pid"] = child.pid
                started = time.monotonic()
                while child.poll() is None:
                    if received or (args.run_root / "STOP_REQUESTED").exists():
                        raise RuntimeError("Stop requested; preserving completed cells")
                    elapsed = time.monotonic() - started
                    if elapsed >= next_sample:
                        measurement = resources.own_process_sample(reference, child.pid, args.run_root, identifiers)
                        reason = guard.observe(measurement, elapsed)
                        measurement.update(seed=seed, child_elapsed_seconds=elapsed, guard_trigger=reason)
                        with (run_dir / "resources.jsonl").open("a", encoding="utf-8") as telemetry:
                            telemetry.write(json.dumps(measurement, allow_nan=False) + "\n")
                        next_sample = elapsed + 60
                        if reason:
                            child.terminate()
                            raise RuntimeError(reason)
                    for source in raw_dir.rglob("*") if raw_dir.exists() else []:
                        if source.is_file() and not source.is_symlink() and source.suffix in {".json", ".log", ".jsonl"}:
                            key = source.relative_to(raw_dir)
                            target = protected_dir / "raw" / key
                            target.parent.mkdir(parents=True, exist_ok=True)
                            stamp = (source.stat().st_size, source.stat().st_mtime_ns)
                            if copied.get(key) != stamp:
                                temporary = target.with_suffix(target.suffix + ".tmp")
                                shutil.copyfile(source, temporary)
                                temporary.replace(target)
                                copied[key] = stamp
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                record.update(exit_code=child.returncode, ended_at=reference.utc(),
                              elapsed_seconds=time.monotonic() - started)
            if child.returncode != 0:
                record["status"] = "failed"
                top["cells"].append(record)
                write_json(run_dir / "cell_status.json", record)
                write_json(args.run_root / "supervisor.json", top)
                write_json(args.protected_root / "supervisor.json", top)
                raise RuntimeError(f"Seed {seed} failed; inspect preserved logs, no automatic retry")
            attempt = json.loads((raw_dir / "attempt.json").read_text(encoding="utf-8"))
            summary = json.loads((raw_dir / "summary.json").read_text(encoding="utf-8"))
            result_path = raw_dir / "cell/result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if (attempt.get("status") != "complete" or attempt.get("smoke")
                    or summary.get("scientific_result") is not True
                    or result.get("completed") is not True
                    or result["identity"].get("dataset") != "NSL-KDD"
                    or result["identity"].get("seed") != seed
                    or result["identity"].get("memory_setting") != "fixed200"):
                raise RuntimeError(f"Seed {seed} output did not meet the formal-cell completion contract")
            canonical = args.publication_root / f"er_bal_tabt_seed{seed}" / "result.json"
            canonical.parent.mkdir(parents=True, exist_ok=False)
            published = redact(result, replacements)
            published["execution_runtime"] = {"wall_seconds": attempt["elapsed_seconds"],
                                                "started_at": attempt["started_at"],
                                                "ended_at": attempt["ended_at"]}
            published["execution_platform"] = {"os": platform.platform(), "python": sys.version.split()[0],
                                                "torch": str(torch.__version__), "numpy": np.__version__,
                                                "gpu_model": torch.cuda.get_device_name(0),
                                                "scope": "Local Windows RTX 4060 Ti; derived extension, not Linux C canonical if C later completes"}
            write_json(canonical, published)
            record.update(status="complete", result_sha256=sha(result_path),
                          published_result_sha256=sha(canonical), published_path=canonical.relative_to(ROOT).as_posix())
            top["cells"].append(record)
            top["current_seed"] = None
            write_json(run_dir / "cell_status.json", record)
            write_json(args.run_root / "supervisor.json", top)
            write_json(args.protected_root / "supervisor.json", top)
            child = None
        exit_code = 0
        top["status"] = "complete"
    except BaseException as error:
        top["status"] = "failed"
        top["error_type"] = type(error).__name__
        top["error"] = str(error)
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
        top["ended_at"] = reference.utc()
        top["elapsed_seconds"] = time.monotonic() - run_started
        top["exit_code"] = exit_code
        write_json(args.run_root / "supervisor.json", top)
        write_json(args.protected_root / "supervisor.json", top)


if __name__ == "__main__":
    main()
