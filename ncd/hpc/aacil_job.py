"""Serial AA-CIL allocation with the frozen UUID telemetry guard and a hard stop.

This wrapper does not change the scientific runner or submit any other job.
The local same-entry smoke uses two CPU threads, one seed/epoch and <=64/class.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from problem_scout_job import (  # Frozen resource policy; never weaken its checks.
    TelemetryGuard, copy_small, gpu_sample, sample, under, utc,
    visible_gpu_uuids, write_json,
)

DATASETS = {"nsl": "NSL-KDD", "unsw": "UNSW-NB15", "cic17": "CIC-IDS-2017"}
METHODS = ("aacil", "ts_mem", "aa_noattmem")
TRAINING_SOURCE = "414136b418cbb5f682875309b00e4f8151240bfe"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("code-root", "data-root", "output-root", "protected-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, choices=(1, 2, 3, 4, 42), required=True)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--local-smoke", action="store_true")
    args = parser.parse_args()
    wrapper_source = os.environ.get("NCD_SOURCE_COMMIT", "")
    if not re.fullmatch("[0-9a-f]{40}", wrapper_source):
        parser.error("NCD_SOURCE_COMMIT must identify the actual wrapper commit")
    if os.environ.get("NCD_TRAINING_SOURCE_COMMIT") != TRAINING_SOURCE:
        parser.error("The frozen AA-CIL training source must be explicit")
    deadline = datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00"))
    if deadline.tzinfo is None or deadline <= datetime.now(timezone.utc):
        parser.error("The hard deadline must be a future timezone-aware timestamp")
    if len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Repeated methods/seeds would duplicate work")
    roots = (args.code_root, args.data_root, args.output_root, args.protected_root)
    code, data, output, protected = [p.resolve() for p in roots]
    if under(output, protected) or under(protected, output):
        parser.error("Active and protective output roots must be disjoint")
    if code != ROOT.resolve():
        parser.error("code-root must be this versioned wrapper checkout")
    if not args.local_smoke:
        user = os.environ.get("USER", "")
        if not user or not os.environ.get("SLURM_JOB_ID", "").isdigit():
            parser.error("Formal execution requires the user's Slurm allocation")
        if not all(under(p, Path.home()) for p in (code, protected)):
            parser.error("Code and protected small results must be in HOME")
        if not all(under(p, Path("/scr/user") / user) for p in (data, output)):
            parser.error("Prepared data and active results must be in the user's scratch")
    config = code / "ncd/configs/problem_scout_section21.json"
    settings = json.loads(config.read_text(encoding="utf-8"))
    if (code / settings["problem_scout_data_root"]).resolve() != data:
        parser.error("Keep the frozen relative data path; do not override the scientific config")
    if settings["cpu_threads"] != 2 or settings["batch_size"] != 256:
        parser.error("The frozen CPU and scientific batch settings changed")
    seeds = [1] if args.local_smoke else args.seeds
    output.mkdir(parents=True, exist_ok=True)
    protected.mkdir(parents=True, exist_ok=True)
    lock = output / ".active.lock"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "job_id": os.environ.get("SLURM_JOB_ID"), "utc": utc()}, stream)
    attempt = output / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"_{os.getpid()}")
    attempt.mkdir(parents=True)
    command = [sys.executable, "-I", "-u", "ncd/run_aacil.py", "--config",
               "ncd/configs/problem_scout_section21.json", "--output-dir", str(output),
               "--datasets", DATASETS[args.dataset], "--methods", *args.methods,
               "--seeds", *map(str, seeds), "--gate-step-policy", "matched_s2000",
               "--calibration-policy", "permanent_normal_holdout_outside_m",
               "--device", "cpu" if args.local_smoke else "cuda"]
    if args.local_smoke:
        command.append("--smoke")
    record = {"job_id": os.environ.get("SLURM_JOB_ID"), "wrapper_source_commit": wrapper_source,
              "training_source_commit": TRAINING_SOURCE, "dataset": DATASETS[args.dataset],
              "methods": args.methods, "seeds": seeds, "smoke": args.local_smoke,
              "command": command, "started_utc": utc(), "deadline_utc": deadline.isoformat(),
              "telemetry_policy": {"source": "frozen problem_scout_job.TelemetryGuard",
                                   "sample_interval_seconds": 60, "failure_limit": 3,
                                   "minimum_low_samples": 3, "low_utilization_percent": 10,
                                   "low_span_threshold": "observed child elapsed / 3",
                                   "gpu_preflight_required": not args.local_smoke,
                                   "interpretation": "conservative sampling guard, not proof of continuous use"}}
    write_json(attempt / "job_start.json", record)
    started, previous, child = time.monotonic(), {}, None
    exit_code, received_signal = 1, None

    def stop(signum, _frame):
        nonlocal received_signal
        received_signal = signum
        if child is not None and child.poll() is None:
            child.terminate()

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)
    try:
        identifiers = []
        if not args.local_smoke:
            step = os.environ.get("SLURM_STEP_ID", os.environ.get("SLURM_STEPID", ""))
            if not step.isdigit() or not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
                raise RuntimeError("GPU preflight requires an actual srun step with explicit CUDA visibility")
            identifiers = visible_gpu_uuids()
            if len(identifiers) != 1:
                raise RuntimeError("This serial workload requires exactly one allocated visible GPU")
            preflight = {"event": "gpu_preflight", "utc": utc(), **gpu_sample(identifiers)}
            with (attempt / "resources.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(preflight, allow_nan=False) + "\n")
            if not preflight["gpu_telemetry_valid"]:
                raise RuntimeError("GPU telemetry unavailable before training: " + preflight["gpu_measurement_error"])
            record.update(allocated_gpu_uuids=identifiers, gpu_preflight_passed=True, slurm_step_id=step)
        else:
            record["gpu_preflight_passed"] = None
            record["smoke_limit"] = "CPU-only end-to-end scientific-entry smoke, not a GPU utilization measurement"
        write_json(attempt / "job_start.json", record)
        environment = dict(os.environ, NCD_SOURCE_COMMIT=TRAINING_SOURCE,
                           OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        guard = TelemetryGuard()
        with (attempt / "stdout.log").open("wb") as out, (attempt / "stderr.log").open("wb") as err:
            child = subprocess.Popen(command, cwd=code, env=environment, stdout=out, stderr=err)
            record["child_pid"] = child.pid
            print(json.dumps({"event": "runner_started", **record}), flush=True)
            child_started, next_sample = time.monotonic(), 0.0
            while child.poll() is None:
                elapsed = time.monotonic() - child_started
                if datetime.now(timezone.utc) >= deadline:
                    record["guard_stop_reason"] = "owner_hard_deadline"
                    exit_code = 124
                    raise TimeoutError("Owner hard deadline reached; completed seeds remain usable")
                if received_signal:
                    raise RuntimeError(f"Wrapper received signal {received_signal}")
                if not args.local_smoke and elapsed >= next_sample:
                    measurement = sample(child.pid, output, identifiers)
                    measurement["child_elapsed_seconds"] = elapsed
                    reason = guard.observe(measurement, elapsed)
                    measurement["guard_trigger"] = reason
                    with (attempt / "resources.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(measurement, allow_nan=False) + "\n")
                    next_sample = elapsed + 60
                    if reason:
                        record["guard_stop_reason"] = reason
                        raise RuntimeError("Stopping own child: " + reason)
                copy_small(output, protected, previous)
                try:
                    child.wait(timeout=min(5, max(.1, (deadline-datetime.now(timezone.utc)).total_seconds())))
                except subprocess.TimeoutExpired:
                    pass
            exit_code = child.returncode
        if exit_code == 0:
            result = json.loads((output / "result.json").read_text(encoding="utf-8"))
            expected = {"datasets": [DATASETS[args.dataset]], "methods": args.methods, "seeds": seeds}
            if result["execution_status"] != "complete" or result["expected"] != expected or len(result["cells"]) != len(args.methods)*len(seeds):
                raise RuntimeError("Child exited zero without the selected complete matrix")
            record["completed_cells"] = len(result["cells"])
            record["training_seconds_total"] = result["training_seconds_total"]
        return exit_code
    except BaseException as exc:
        record["entry_error"] = f"{type(exc).__name__}: {exc}"
        exit_code = exit_code or 1
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        raise
    finally:
        record.update(exit_code=exit_code, child_returncode=child.poll() if child else None,
                      finished_utc=utc(), elapsed_seconds_this_attempt=time.monotonic()-started)
        write_json(attempt / "job_exit.json", record)
        copy_small(output, protected, previous)
        lock.unlink()
        print(json.dumps({"event": "job_exit", **record}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
