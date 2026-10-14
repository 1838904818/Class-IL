"""One allocated dataset/seed run, with bounded telemetry and small-file backup.

The unchanged scientific runner owns data verification and checkpoint identity.
This entry never submits another job, changes a training configuration, copies
datasets/weights into HOME, or launches more than one training process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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


def sample(pid, output):
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
    # Slurm's GPU IDs are global, unlike CUDA's potentially remapped ordinals.
    # Never guess GPU 0 when an allocation's global IDs are unavailable.
    visible = (os.environ.get("SLURM_STEP_GPUS") or os.environ.get("SLURM_JOB_GPUS")
               or (os.environ.get("CUDA_VISIBLE_DEVICES", "") if not os.environ.get("SLURM_JOB_ID") else ""))
    if re.fullmatch(r"(?:\d+|GPU-[\w-]+)(?:,(?:\d+|GPU-[\w-]+))*", visible):
        gpu = ["nvidia-smi", f"--id={visible}", "--query-gpu=uuid,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
    elif not os.environ.get("SLURM_JOB_ID"):
        gpu = ["nvidia-smi", "--query-gpu=uuid,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
    else:
        gpu = None
    if gpu:
        try:
            run = subprocess.run(gpu, capture_output=True, text=True, timeout=10, check=False)
            result["gpu_csv"] = run.stdout.strip()
            result["gpu_query_exit"] = run.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["gpu_measurement_error"] = type(exc).__name__
    else:
        result["gpu_measurements"] = "unavailable: allocation visibility not identifiable"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("code-root", "data-root", "output-root", "protected-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2, 3, 4, 42), required=True)
    parser.add_argument("--method-group", choices=("all", "joint", "nonjoint"), default="all")
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
    if args.local_smoke:
        command.append("--smoke")
    record = {"job_id": os.environ.get("SLURM_JOB_ID"), "source_commit": source,
              "dataset": DATASETS[args.dataset], "seed": seed, "methods": methods, "smoke": args.local_smoke,
              "command": command, "started_utc": utc(), "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
              "batch_entry_sha256": hashlib.sha256((args.code_root / "ncd/hpc/run_problem_scout.sbatch").read_bytes()).hexdigest(),
              "job_helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
        with (attempt / "stdout.log").open("wb") as out, (attempt / "stderr.log").open("wb") as err:
            child = subprocess.Popen(command, cwd=args.code_root, stdout=out, stderr=err)
            print(json.dumps({"event": "runner_started", "pid": child.pid, **record}), flush=True)
            while True:
                with (attempt / "resources.jsonl").open("a", encoding="utf-8") as telemetry:
                    telemetry.write(json.dumps(sample(child.pid, output)) + "\n")
                copy_small(output, protected, previous)
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
