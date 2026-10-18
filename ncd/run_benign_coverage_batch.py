"""Sequential local Section25 scheduling; no new training or selection logic."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"NSL-KDD": ("nsl_kdd", [1, 2, 3, 4, 42]),
            "CIC-IDS-2017": ("cic_ids_2017", [1, 2, 3])}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    for attempt in range(40):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.05)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--treatments", nargs="+", choices=("D1", "D2", "D3"), default=["D1", "D3"])
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    parser.add_argument("--d2-budget-mode", choices=("staged_archive", "fixed_final"))
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        raise ValueError("An explicit full source commit is required")
    if ("D2" in args.treatments) != (args.d2_budget_mode is not None):
        raise ValueError("D2 requires a separately authorized explicit budget interpretation")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".driver.lock").open("a+b")
    if (output / ".driver.lock").stat().st_size == 0:
        lock.write(b"0"); lock.flush()
    lock.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    planned = [{"dataset": dataset, "seed": seed, "treatment": treatment,
                "relative_path": f"{slug}/{treatment}_seed{seed}"}
               for treatment in args.treatments for dataset in args.datasets
               for slug, seeds in [DATASETS[dataset]] for seed in seeds]
    state = {"status": "running", "source_commit": args.source_commit,
             "driver_pid": os.getpid(), "started_utc": stamp(),
             "d2_budget_mode": args.d2_budget_mode, "planned_cells": len(planned),
             "completed_cells": [], "current_cell": None,
             "scheduler_sha256": sha256(Path(__file__)),
             "training_entry_sha256": sha256(ROOT / "ncd/run_benign_coverage.py")}
    env = dict(os.environ, NCD_SOURCE_COMMIT=args.source_commit)
    attempts = output / "driver_attempts.jsonl"

    def event(value):
        with attempts.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time_utc": stamp(), **value}, sort_keys=True) + "\n")

    def completed(cell):
        path = output / cell["relative_path"] / "result.json"
        if not path.exists():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        ident = value["identity"]
        if (not value.get("completed") or ident["smoke"] or ident["git_commit"] != args.source_commit
                or any(ident[k] != cell[k] for k in ("dataset", "seed", "treatment"))
                or ident["source_hashes"]["ncd/run_benign_coverage.py"] != state["training_entry_sha256"]
                or len(value["checkpoints"]) != len(value["tasks"])):
            raise ValueError("Existing completed cell differs from this batch")
        for cp in value["checkpoints"]:
            weight = path.parent / cp["checkpoint_file"]
            if sha256(weight) != cp["checkpoint_sha256"]:
                raise ValueError("Existing completed checkpoint changed")
        return {**cell, "result_sha256": sha256(path)}

    event({"event": "driver_started", "pid": os.getpid(), "source_commit": args.source_commit})
    try:
        for cell in planned:
            done = completed(cell)
            if done:
                state["completed_cells"].append(done)
                write_json(output / "result.json", state)
                print("CELL_SKIP", cell, flush=True)
                continue
            folder = output / cell["relative_path"]
            folder.mkdir(parents=True, exist_ok=True)
            attempt = 1
            while (folder / f"console.attempt_{attempt:04d}.log").exists():
                attempt += 1
            stdout = folder / f"console.attempt_{attempt:04d}.log"
            stderr = folder / f"stderr.attempt_{attempt:04d}.log"
            command = [sys.executable, "-I", "-u", "ncd/run_benign_coverage.py",
                       "--dataset", cell["dataset"], "--seed", str(cell["seed"]),
                       "--treatment", cell["treatment"], "--output-dir", os.path.relpath(folder, ROOT),
                       "--coverage-cache", os.path.relpath(folder.parent / "normal_coverage.json", ROOT)]
            if cell["treatment"] == "D2":
                command += ["--d2-budget-mode", args.d2_budget_mode]
            if args.data_root:
                command += ["--data-root", str(args.data_root)]
            started = time.perf_counter()
            with stdout.open("wb") as out, stderr.open("wb") as err:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err,
                                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                state["current_cell"] = {**cell, "pid": child.pid, "attempt": attempt}
                write_json(output / "result.json", state)
                event({"event": "cell_started", **state["current_cell"]})
                print("CELL_START", state["current_cell"], flush=True)
                code = child.wait()
            event({"event": "cell_exited", **cell, "attempt": attempt,
                   "returncode": code, "wall_seconds": time.perf_counter() - started})
            if code:
                raise RuntimeError(f"Cell exited {code}: {cell['relative_path']}; inspect preserved stderr")
            done = completed(cell)
            if done is None:
                raise RuntimeError("Child exited successfully without a completed result")
            state["completed_cells"].append(done)
            state["current_cell"] = None
            write_json(output / "result.json", state)
        state.update(status="complete", ended_utc=stamp(), current_cell=None)
        write_json(output / "result.json", state)
        event({"event": "driver_complete", "completed_cells": len(state["completed_cells"])})
        print("BATCH_COMPLETE", len(state["completed_cells"]), flush=True)
    except Exception as exc:
        state.update(status="failed", ended_utc=stamp(), error=f"{type(exc).__name__}: {exc}")
        write_json(output / "result.json", state)
        event({"event": "driver_failed", "error": state["error"]})
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
