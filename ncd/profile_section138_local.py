"""Measure one local, non-scientific section138 entry smoke with full logs.

These small-data measurements are not a full-data walltime or memory estimate.
Requires psutil (validated with 7.2.2).
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if (psutil.cpu_count(logical=False) or 0) < 4:
        raise RuntimeError("At least four physical CPU cores required")
    if psutil.virtual_memory().available < 6 * 1024**3:
        raise RuntimeError("Less than 6 GiB available memory")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-I", "-B", "-u",
               str(Path(__file__).with_name("run_section138.py")),
               "--configuration", args.configuration, "--dataset", args.dataset,
               "--seed", "1", "--data-root", str(args.data_root.resolve(strict=True)),
               "--output-root", str(root / "outputs"), "--smoke"]
    start = time.perf_counter()
    peaks, cpu, reason = 0, {}, None
    with (root / "stdout.log").open("x") as stdout, (root / "stderr.log").open("x") as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        proc = psutil.Process(child.pid)
        with (root / "resources.jsonl").open("x") as telemetry:
            while child.poll() is None:
                rss = 0
                try:
                    for process in [proc, *proc.children(recursive=True)]:
                        try:
                            rss += process.memory_info().rss
                            timing = process.cpu_times()
                            cpu[(process.pid, process.create_time())] = timing.user + timing.system
                        except psutil.NoSuchProcess:
                            pass
                except psutil.NoSuchProcess:
                    pass
                peaks = max(peaks, rss)
                available = psutil.virtual_memory().available
                telemetry.write(json.dumps(dict(elapsed_seconds=time.perf_counter()-start,
                    rss_bytes=rss, available_bytes=available)) + "\n")
                telemetry.flush()
                if available < 2 * 1024**3:
                    reason = "Available memory fell below 2 GiB"
                    child.terminate()
                    break
                time.sleep(.1)
        code = child.wait()
    summaries = list((root / "outputs").rglob("summary.json"))
    elapsed = time.perf_counter() - start
    result = dict(command=command, exit_code=code, guard_reason=reason,
        scientific_result=False, elapsed_seconds=elapsed, sampled_peak_rss_bytes=peaks,
        psutil_version=psutil.__version__,
        sampled_cpu_seconds=sum(cpu.values()), sampling_interval_seconds=.1,
        allocated_cpu_threads=4, summary_count=len(summaries),
        stdout_log="stdout.log", stderr_log="stderr.log", telemetry_log="resources.jsonl",
        limitation="64 rows per class, one epoch; not a full-data resource estimate")
    (root / "profile.json").write_text(json.dumps(result, indent=2) + "\n")
    if code or reason or len(summaries) != 1:
        raise RuntimeError("Profile failed; inspect retained logs")
    print(json.dumps({k: v for k, v in result.items() if k != "command"}, indent=2))


if __name__ == "__main__":
    main()
