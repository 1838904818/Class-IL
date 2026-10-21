"""One explicit B1 ER cell with the existing local CUDA resource guard.

No automatic retry, fan-out or scheduler submission. Each invocation reserves
the project-wide CUDA lock and preserves complete logs and small-file copies.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent


def load(name, relative, expected):
    path = ROOT / relative
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Existing resource helper changed")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("UNSW-NB15", "CIC-IDS-2017"), required=True)
    parser.add_argument("--memory-setting", choices=("fixed200", "percent1"), required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2, 3, 4, 42), required=True)
    for name in ("data-root", "dependency-root", "output-root", "protected-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if os.name != "nt" or os.environ.get("SLURM_JOB_ID"):
        parser.error("Windows local execution only; not an HPC launcher")
    for name in ("data_root", "dependency_root", "output_root", "protected_root"):
        setattr(args, name, getattr(args, name).resolve())
    output, protected = args.output_root, args.protected_root
    if output.is_relative_to(protected) or protected.is_relative_to(output):
        parser.error("Disjoint output/protected roots required")
    if output.exists() or protected.exists():
        parser.error("Existing cell requires diagnosis; automatic retry is forbidden")
    lock = WORK / ".section35_local_cuda.lock"
    if lock.exists():
        parser.error("Another CUDA assignment or unresolved lock exists")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    ref = load("section81_telemetry", "ncd/hpc/problem_scout_job.py",
               "6e91a541f04c5fff8a00e6e914cc95d780ba583751afa3e737639c89918d446f")
    resources = load("section81_resources", "ncd/run_section35_local.py",
                     "bf8a5aa83b033927966d26dcf0d834ad27008fa4bfc1622a79c8a2dd5721239b")
    identifiers = ref.visible_gpu_uuids()
    if len(identifiers) != 1 or not ref.gpu_sample(identifiers)["gpu_telemetry_valid"]:
        raise RuntimeError("One selected GPU with valid telemetry required")
    command = [sys.executable, "-I", "-B", "-u", str(ROOT / "ncd/run_section81_er.py"),
               "--dataset", args.dataset, "--memory-setting", args.memory_setting,
               "--seed", str(args.seed), "--data-root", str(args.data_root),
               "--dependency-root", str(args.dependency_root), "--output-dir", str(output / "run")]
    output.mkdir(parents=True)
    protected.mkdir(parents=True)
    record = {"status": "starting", "started_at": ref.utc(), "pid": os.getpid(),
              "dataset": args.dataset, "method": "er_bal_tabt", "seed": args.seed,
              "memory_setting": args.memory_setting, "command": command, "gpu_uuids": identifiers}
    child, copied, received = None, {}, []
    started = time.monotonic()
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(record, stream)

    def stop(number, _frame):
        received.append(number)

    def terminate():
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)

    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, stop)
    try:
        ref.write_json(output / "status.json", record)
        with (output / "stdout.log").open("x", encoding="utf-8") as stdout, (output / "stderr.log").open("x", encoding="utf-8") as stderr:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                env=dict(os.environ, NCD_SECTION81_PARENT_PID=str(os.getpid())),
                creationflags=subprocess.CREATE_NO_WINDOW)
            record.update(status="running", child_pid=child.pid)
            ref.write_json(output / "status.json", record)
            guard, next_sample = ref.TelemetryGuard(), 0.0
            while child.poll() is None:
                if received or (output / "STOP_REQUESTED").exists():
                    raise RuntimeError("Stop requested")
                elapsed = time.monotonic() - started
                if elapsed >= next_sample:
                    sample = resources.own_process_sample(ref, child.pid, output, identifiers)
                    reason = guard.observe(sample, elapsed)
                    sample.update(child_elapsed_seconds=elapsed, guard_trigger=reason)
                    with (output / "resources.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(sample, allow_nan=False) + "\n")
                    next_sample = elapsed + 60
                    if reason:
                        raise RuntimeError(reason)
                ref.copy_small(output, protected, copied)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            if child.returncode != 0:
                raise RuntimeError("B1 cell failed; inspect preserved logs before retry")
            attempt = json.loads((output / "run/attempt.json").read_text())
            if attempt["status"] != "complete" or attempt["smoke"]:
                raise RuntimeError("Missing formal cell completion")
            record.update(status="complete", result_sha256=attempt["result_sha256"])
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        terminate()
        record.update(ended_at=ref.utc(), elapsed_seconds=time.monotonic() - started,
                      exit_code=None if child is None else child.returncode)
        ref.write_json(output / "status.json", record)
        ref.write_json(output / "job_exit.json", record)
        ref.copy_small(output, protected, copied)
        lock.unlink()


if __name__ == "__main__":
    main()
