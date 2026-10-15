"""Run the Section 22 step-floor controls sequentially on one local GPU.

The underlying Section 21 runner owns checkpoint resume and scientific settings.
This entry point changes only the minimum optimizer updates per task and excludes
Joint/CSE18. Complete child stdout and stderr are retained, including startup
errors. An advisory process lock prevents duplicate runs of the same output root.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ncd.source_commit import source_commit

DATASETS = ["NSL-KDD", "UNSW-NB15", "CIC-IDS-2017"]
METHODS = ["replay", "replay_lwf", "rcl_tab", "icarl", "derpp"]
SEEDS = [1, 2, 3, 4, 42]


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def execution_lock(path):
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("ncd/configs/problem_scout_section21.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    commit = source_commit(ROOT)
    # Keep a single source identity across a multi-hour run even if documentation
    # is committed in the checkout while a child is training.
    environment = os.environ.copy()
    environment["NCD_SOURCE_COMMIT"] = commit
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[key] = "2"
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    record = {"schema_version": 1, "git_commit": commit, "smoke": args.smoke,
              "expected_cells": 30 if args.smoke else 150,
              "started_at": timestamp(), "process_id": os.getpid(),
              "status": "running", "attempts": []}
    with execution_lock(out / ".execution.lock"):
        # Attempts are immutable log entries; a resume adds a new driver attempt.
        history = out / "driver_attempts.jsonl"
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": "start", **record}) + "\n")
        for floor in (2000, 500):
            child = out / f"steps_{floor}_v1"
            child.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-I", "-u", "ncd/run_problem_scout.py",
                       "--config", str(args.config), "--output-dir", Path(os.path.relpath(child, ROOT)).as_posix(),
                       "--datasets", *DATASETS, "--methods", *METHODS,
                       "--seeds", *map(str, SEEDS), "--min-steps-per-task", str(floor)]
            if args.data_root:
                command += ["--data-root", args.data_root]
            if args.smoke:
                command += ["--smoke"]
            attempt = {"floor": floor, "started_at": timestamp(),
                       "stdout": f"steps_{floor}_v1/console.log",
                       "stderr": f"steps_{floor}_v1/stderr.log"}
            record["attempts"].append(attempt)
            save(out / "result.json", record)
            print(f"BUDGET_START floor={floor} smoke={args.smoke} source={commit}", flush=True)
            started = time.perf_counter()
            with (child / "console.log").open("a", encoding="utf-8") as stdout, \
                    (child / "stderr.log").open("a", encoding="utf-8") as stderr:
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
                attempt["process_id"] = process.pid
                save(out / "result.json", record)
                code = process.wait()
            attempt.update(exit_code=code, finished_at=timestamp(),
                           process_seconds=time.perf_counter() - started)
            with history.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"event": "child_exit", **attempt}) + "\n")
            if code:
                record.update(status="failed", finished_at=timestamp())
                save(out / "result.json", record)
                print(f"BUDGET_FAILED floor={floor} exit_code={code}", flush=True)
                return code
            result = json.loads((child / "result.json").read_text(encoding="utf-8"))
            expected = 15 if args.smoke else 75
            if result["execution_status"] != "complete" or len(result["cells"]) != expected:
                raise RuntimeError(f"child exited without complete {expected}-cell summary")
            save(out / "result.json", record)
            print(f"BUDGET_COMPLETE floor={floor} cells={expected}", flush=True)
        record.update(status="complete", finished_at=timestamp())
        save(out / "result.json", record)
        print(f"STEP_BUDGET_COMPLETE cells={record['expected_cells']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
