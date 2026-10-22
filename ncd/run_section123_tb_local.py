"""One CPU TB cell alongside the independent CUDA lane; no automatic retry."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def snapshot(pid=None):
    command = "$o=Get-CimInstance Win32_OperatingSystem; $c=(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum; "
    command += "$p=$null; " if pid is None else f"$p=Get-Process -Id {int(pid)} -ErrorAction Stop; "
    command += "@{available=[long]$o.FreePhysicalMemory*1024; cores=$c; cpu=$p.CPU; rss=$p.WorkingSet64} | ConvertTo-Json -Compress"
    return json.loads(subprocess.check_output(["powershell", "-NoProfile", "-Command", command],
                      text=True, creationflags=subprocess.CREATE_NO_WINDOW, timeout=15))


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("UNSW-NB15", "CIC-IDS-2017"), required=True)
    parser.add_argument("--method", choices=("ar_es", "replay_bs_es"), required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2, 3, 4, 42), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protected-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if os.name != "nt" or os.environ.get("SLURM_JOB_ID"):
        parser.error("Windows CPU launcher only")
    output, protected = args.output_root.resolve(), args.protected_root.resolve()
    if output.exists() or protected.exists() or output.is_relative_to(protected) or protected.is_relative_to(output):
        parser.error("New disjoint roots required")
    machine = snapshot()
    if machine["cores"] < 6 or machine["available"] < 6 * 2**30:
        parser.error("Need six physical cores and 6 GiB currently available for CPU+GPU lanes")
    lock = ROOT.parent / ".section123_local_cpu.lock"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(dict(pid=os.getpid()), stream)
    child = None
    started = time.monotonic()
    record = dict(status="starting", started_at=datetime.now(timezone.utc).isoformat(),
                  pid=os.getpid(), dataset=args.dataset, method=args.method, seed=args.seed,
                  cpu_threads=4, smoke=args.smoke)
    copied = {}

    def protect():
        for path in output.rglob("*"):
            if not path.is_file() or path.suffix not in (".json", ".jsonl", ".log"):
                continue
            stat = path.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
            if copied.get(path) == signature:
                continue
            target = protected / path.relative_to(output)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            copied[path] = signature

    try:
        output.mkdir(parents=True)
        protected.mkdir(parents=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = "4"
        command = [sys.executable, "-I", "-B", "-u", str(ROOT / "ncd/run_section123_tb.py"),
                   "--dataset", args.dataset, "--method", args.method, "--seed", str(args.seed),
                   "--data-root", str(args.data_root.resolve(strict=True)), "--output-dir", str(output / "run")]
        if args.smoke:
            command.append("--smoke")
        with (output / "stdout.log").open("x") as out, (output / "stderr.log").open("x") as err:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            record.update(status="running", child_pid=child.pid, command=command)
            write(output / "status.json", record)
            previous_cpu, previous_time = None, None
            while child.poll() is None:
                sample = snapshot(child.pid)
                now = time.monotonic()
                cpu, rss = sample["cpu"], sample["rss"]
                utilization = None if previous_cpu is None else (cpu - previous_cpu) / max(now - previous_time, .001) * 25
                available = sample["available"]
                guard = None
                if available < 2 * 2**30 or rss > 8 * 2**30:
                    guard = "CPU lane memory boundary"
                if shutil.disk_usage(output).free < 10 * 2**30:
                    guard = "Disk free below 10 GiB"
                if (output / "STOP_REQUESTED").exists():
                    guard = "Stop requested"
                # Section126.2: local CPU utilization is telemetry only.
                # DICC utilization guards and the independent CUDA lane are unchanged.
                sample = dict(utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=now-started,
                              cpu_allocation_percent=utilization, rss_bytes=rss,
                              available_memory_bytes=available, guard_trigger=guard)
                with (output / "resources.jsonl").open("a") as stream:
                    stream.write(json.dumps(sample, allow_nan=False) + "\n")
                protect()
                if guard:
                    raise RuntimeError(guard)
                previous_cpu, previous_time = cpu, now
                try:
                    child.wait(timeout=5 if args.smoke else 60)
                except subprocess.TimeoutExpired:
                    pass
            if child.returncode != 0:
                raise RuntimeError(f"TB child exited {child.returncode}")
            summary = json.loads((output / "run/summary.json").read_text())
            if not summary["completed"]:
                raise RuntimeError("Missing completed summary")
            record.update(status="complete", exit_code=0)
    except BaseException as exc:
        record.update(status="failed", error=str(exc))
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
        if output.exists():
            record.update(elapsed_seconds=time.monotonic()-started, ended_at=datetime.now(timezone.utc).isoformat())
            write(output / "job_exit.json", record)
            write(output / "status.json", record)
            if protected.exists():
                protect()
        lock.unlink()


if __name__ == "__main__":
    main()
