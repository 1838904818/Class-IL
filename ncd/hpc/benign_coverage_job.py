"""Run a bounded Section25 CIC17 D1 or fixed-final D2 sequence in one allocation.

Only the existing training entry computes scientific outputs. This wrapper
serializes seeds, keeps attempt logs, and protects small evidence, not weights.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ncd.hpc.problem_scout_job import copy_small, sample, under, utc, write_json

D2_TRAINING_SOURCE = "5657b468d0c83af58ec60dcd60ecf83bf61594f5"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("code-root", "data-root", "output-root", "protected-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--treatment", choices=("D1", "D2"), default="D1")
    parser.add_argument("--d2-budget-mode", choices=("fixed_final",))
    parser.add_argument("--training-source-commit")
    parser.add_argument("--local-smoke", action="store_true")
    args = parser.parse_args()
    source = os.environ.get("NCD_SOURCE_COMMIT", "")
    if not re.fullmatch(r"[0-9a-f]{40}", source):
        parser.error("NCD_SOURCE_COMMIT must identify the complete wrapper execution commit")
    if (args.treatment == "D2") != (args.d2_budget_mode is not None):
        parser.error("D2 requires explicit fixed_final; D1 does not accept a D2 budget mode")
    if args.treatment == "D2" and args.training_source_commit != D2_TRAINING_SOURCE:
        parser.error("D2 requires the explicit frozen --training-source-commit " + D2_TRAINING_SOURCE)
    # Preserve the original D1 interface and its historical execution identity.
    # New callers may separately identify the unchanged child training source.
    training_source = args.training_source_commit or source
    if not re.fullmatch(r"[0-9a-f]{40}", training_source):
        parser.error("training source must be a complete commit")
    for name in ("code_root", "data_root", "output_root", "protected_root"):
        setattr(args, name, getattr(args, name).resolve())
    if args.code_root != ROOT:
        parser.error("code root does not match this versioned wrapper")
    if under(args.protected_root, args.output_root) or under(args.output_root, args.protected_root):
        parser.error("active and protective-copy roots must be disjoint")
    if not args.local_smoke:
        user = os.environ.get("USER", "")
        if not user or not os.environ.get("SLURM_JOB_ID", "").isdigit():
            parser.error("formal execution requires the user's Slurm allocation")
        if not all(under(p, Path.home()) for p in (ROOT, args.protected_root)):
            parser.error("code and protected small evidence must be in HOME")
        if not all(under(p, Path("/scr/user") / user) for p in (args.data_root, args.output_root)):
            parser.error("data and active outputs must be in the user's scratch")
    config = ROOT / "ncd/configs/problem_scout_section21.json"
    settings = json.loads(config.read_text(encoding="utf-8"))
    if settings["device"] != "cuda" or settings["cpu_threads"] != 2 or settings["batch_size"] != 256:
        parser.error("frozen GPU configuration changed")
    receipt = args.data_root / "cic-ids-2017/data_quality.json"
    info = json.loads(receipt.read_text(encoding="utf-8"))
    if (info.get("status"), info.get("dataset")) != ("complete", "CIC-IDS-2017"):
        parser.error("prepared full data receipt is incomplete or mismatched")
    if (sum(info["train_counts"].values()), sum(info["id_counts"].values())) != (2262203, 565673):
        parser.error("the existing full uncapped CIC17 cache is required")
    seeds = [1] if args.local_smoke else [1, 2, 3]
    suffix = Path("smoke" if args.local_smoke else "formal") / "cic17" / args.treatment
    output, protected = args.output_root / suffix, args.protected_root / suffix
    output.mkdir(parents=True, exist_ok=True)
    protected.mkdir(parents=True, exist_ok=True)
    lock = output / ".active.lock"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "job_id": os.environ.get("SLURM_JOB_ID"), "utc": utc()}, stream)
    attempt = output / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                                    + "_" + os.environ.get("SLURM_JOB_ID", "local") + f"_{os.getpid()}")
    attempt.mkdir(parents=True)
    record = {"schema_version": 2, "job_id": os.environ.get("SLURM_JOB_ID"), "source_commit": source,
              "wrapper_source_commit": source, "training_source_commit": training_source,
              "dataset": "CIC-IDS-2017", "treatment": args.treatment, "d2_budget_mode": args.d2_budget_mode,
              "seeds": seeds, "smoke": args.local_smoke,
              "started_utc": utc(), "config_sha256": sha256(config), "data_receipt_sha256": sha256(receipt),
              "job_helper_sha256": sha256(Path(__file__)),
              "batch_entry_sha256": sha256(ROOT / "ncd/hpc/run_benign_coverage.sbatch"),
              "resources": {k: os.environ.get(k) for k in ("SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS")},
              "attempt": attempt.relative_to(output).as_posix(), "cells": [], "execution_status": "running"}
    started, previous, child = time.monotonic(), {}, None
    exit_code, stopping = 1, False
    child_env = dict(os.environ, NCD_SOURCE_COMMIT=training_source)

    def stop(signum, _frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)
    try:
        # CUDA resolves logical device zero through this allocation's visibility.
        # Query its UUID, not a global Slurm index that cgroups can remap.
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one allocated/visible CUDA device is required")
        props = torch.cuda.get_device_properties(0)
        gpu_uuid = str(getattr(props, "uuid", ""))
        if gpu_uuid and not gpu_uuid.startswith("GPU-"):
            gpu_uuid = "GPU-" + gpu_uuid
        record["cuda_device"] = {"name": props.name, "total_memory": props.total_memory, "uuid": gpu_uuid or None}
        write_json(attempt / "job_start.json", record)
        write_json(output / "result.json", record)
        for seed in seeds:
            if stopping:
                raise InterruptedError("stop requested before next seed")
            cell = output / f"seed_{seed}"
            command = [sys.executable, "-I", "-u", "ncd/run_benign_coverage.py", "--dataset", "CIC-IDS-2017",
                       "--seed", str(seed), "--treatment", args.treatment, "--output-dir", str(cell),
                       "--data-root", str(args.data_root), "--device", "cuda"]
            if args.treatment == "D2":
                command += ["--d2-budget-mode", args.d2_budget_mode]
            if args.local_smoke:
                command.append("--smoke")
            stdout, stderr = attempt / f"seed_{seed}.stdout.log", attempt / f"seed_{seed}.stderr.log"
            item = {"seed": seed, "command": command, "started_utc": utc(),
                    "stdout": stdout.relative_to(output).as_posix(), "stderr": stderr.relative_to(output).as_posix(),
                    "result": (cell / "result.json").relative_to(output).as_posix()}
            record["cells"].append(item)
            write_json(output / "result.json", record)
            with stdout.open("wb") as out, stderr.open("wb") as err:
                child = subprocess.Popen(command, cwd=ROOT, env=child_env, stdout=out, stderr=err)
                item["pid"] = child.pid
                print(json.dumps({"event": "cell_started", **item}), flush=True)
                while True:
                    measured = sample(child.pid, output)
                    measured["seed"] = seed
                    if gpu_uuid:
                        try:
                            gpu = subprocess.run(["nvidia-smi", f"--id={gpu_uuid}", "--query-gpu=uuid,utilization.gpu,memory.used,memory.total",
                                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=False)
                            measured.update(gpu_csv=gpu.stdout.strip(), gpu_query_exit=gpu.returncode, gpu_uuid=gpu_uuid)
                        except (OSError, subprocess.TimeoutExpired) as exc:
                            measured["gpu_measurement_error"] = type(exc).__name__
                    with (attempt / "resources.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(measured) + "\n")
                    copy_small(output, protected, previous)
                    try:
                        exit_code = child.wait(timeout=60)
                        break
                    except subprocess.TimeoutExpired:
                        pass
            item.update(exit_code=exit_code, finished_utc=utc())
            if exit_code:
                raise RuntimeError(f"seed {seed} exited {exit_code}; preserve this attempt and diagnose before resubmission")
            result = json.loads((cell / "result.json").read_text(encoding="utf-8"))
            identity = result["identity"]
            expected = {"dataset": "CIC-IDS-2017", "treatment": args.treatment, "seed": seed,
                        "git_commit": training_source, "d2_budget_mode": args.d2_budget_mode,
                        "smoke": args.local_smoke}
            if not result.get("completed") or any(identity.get(k) != v for k, v in expected.items()) or len(result["checkpoints"]) != 4:
                raise ValueError("training entry did not produce the complete bound cell")
            effective = identity["effective_config"]
            if (effective["epochs_per_task"], effective["min_steps_per_task"], effective["batch_size"]) != ((1, 0, 256) if args.local_smoke else (8, 2000, 256)):
                raise ValueError("effective optimization configuration differs")
            if args.local_smoke and max(result["train_counts"].values()) > 64:
                raise ValueError("local-smoke exceeded 64 rows per class")
            for checkpoint in result["checkpoints"]:
                path = cell / checkpoint["checkpoint_file"]
                if not under(path, cell) or sha256(path) != checkpoint["checkpoint_sha256"]:
                    raise ValueError("checkpoint payload hash mismatch")
            item.update(result_sha256=sha256(cell / "result.json"), completed=True,
                        training_seconds_total=result["training_seconds_total"],
                        selection_seconds_total=result["selection_seconds_total"],
                        evaluation_seconds_total=result["evaluation_seconds_total"])
            write_json(output / "result.json", record)
            copy_small(output, protected, previous)
        record["execution_status"] = "complete"
        record["training_seconds_total"] = sum(c["training_seconds_total"] for c in record["cells"])
        exit_code = 0
    except BaseException as exc:
        record.update(execution_status="failed", entry_error=f"{type(exc).__name__}: {exc}")
        exit_code = exit_code or 1
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        raise
    finally:
        record.update(exit_code=exit_code, finished_utc=utc(), elapsed_seconds_this_attempt=time.monotonic() - started)
        write_json(attempt / "job_exit.json", record)
        write_json(output / "result.json", record)
        copy_small(output, protected, previous)
        lock.unlink()
        print(json.dumps({"event": "job_exit", **record}), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
