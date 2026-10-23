"""One bounded four-thread worker of the disjoint CIC screening lane."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ncd.section138_config import configurations
from ncd.source_commit import source_commit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--worker", type=int, required=True)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--smoke", action="store_true", help="One non-scientific cell through this worker")
    a = p.parse_args()
    if not 0 <= a.worker < a.workers or 4*a.workers > (psutil.cpu_count(logical=False) or 0):
        p.error("Worker partition exceeds physical CPU capacity")
    root = a.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    commit = source_commit(ROOT)
    queue = [(cfg.identifier, seed) for cfg in configurations() for seed in (1, 2)]
    for number, (configuration, seed) in enumerate(queue):
        if number % a.workers != a.worker:
            continue
        if source_commit(ROOT) != commit:
            raise RuntimeError("Source changed: stop before the next cell")
        if psutil.virtual_memory().available < 6*1024**3 or shutil.disk_usage(root).free < 20*1024**3:
            raise RuntimeError("Insufficient RAM or disk for next cell")
        cell = root / "screen/cic-ids-2017" / configuration / f"seed{seed}"
        command = [sys.executable, "-I", "-B", "-u", str(ROOT / "ncd/run_section138.py"),
            "--configuration", configuration, "--dataset", "CIC-IDS-2017", "--seed", str(seed),
            "--data-root", str(a.data_root.resolve(strict=True)), "--output-root", str(root), "--lane", "local"]
        if cell.exists():
            command.append("--resume")
        if a.smoke:
            command.append("--smoke")
        attempt = root / "logs" / f"{configuration}-seed{seed}-{uuid.uuid4().hex}"
        attempt.mkdir(parents=True)
        started = time.perf_counter()
        reason, peak = None, 0
        with (attempt / "stdout.log").open("x") as out, (attempt / "stderr.log").open("x") as err:
            child = subprocess.Popen(command, stdout=out, stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            process = psutil.Process(child.pid)
            with (attempt / "resources.jsonl").open("x") as telemetry:
                while child.poll() is None:
                    try:
                        rss = process.memory_info().rss
                        peak = max(peak, rss)
                        cpu = process.cpu_times()
                        telemetry.write(json.dumps(dict(elapsed=time.perf_counter()-started,
                            rss_bytes=rss, cpu_seconds=cpu.user+cpu.system)) + "\n")
                        telemetry.flush()
                    except psutil.NoSuchProcess:
                        break
                    if psutil.virtual_memory().available < 2*1024**3 or shutil.disk_usage(root).free < 5*1024**3:
                        reason = "RAM or disk safety floor reached"
                        child.terminate()
                        break
                    time.sleep(1)
            code = child.wait()
        receipt = dict(command=command, git_commit=commit, seed=seed, configuration=configuration,
            worker=a.worker, workers=a.workers, exit_code=code, guard_reason=reason,
            elapsed_seconds=time.perf_counter()-started, sampled_peak_rss_bytes=peak,
            stdout_log="stdout.log", stderr_log="stderr.log", resource_log="resources.jsonl")
        (attempt / "exit.json").write_text(json.dumps(receipt, indent=2)+"\n")
        if code or reason:
            raise RuntimeError("Cell failed; retained logs require diagnosis before retry")
        if a.smoke:
            return


if __name__ == "__main__":
    main()
