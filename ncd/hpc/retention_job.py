"""Serial Section29.8 CIC17 execution with frozen selection and UUID telemetry.

One child owns one method and all five formal seeds. This entry never submits
jobs, changes the selected scientific parameters, or copies weights to HOME.
Local smoke uses the same entry, CPU, seed5, one epoch and <=64 rows/class.
"""
from __future__ import annotations

import argparse
import hashlib
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
from problem_scout_job import (
    TelemetryGuard, copy_small, gpu_sample, sample, under, utc,
    visible_gpu_uuids, write_json,
)

METHODS = ("aacil_kd", "aacil_iso", "aacil_wa", "aacil_cbf", "aacil_ncm")
FORMAL_SEEDS = (1, 2, 3, 4, 42)
OWNER_DEADLINE = datetime.fromisoformat("2026-10-19T21:50:26+08:00")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_parameters(path, training_source, smoke):
    if path is None:
        if not smoke:
            raise ValueError("Formal execution requires the immutable completed selection.json")
        return {m: {"lambda_g": .5, "der_alpha": 1., "der_beta": 1.,
                    "cbf_epochs": 1, "cbf_lr_multiplier": .1} for m in METHODS}, None
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("section") != "29.8" or value.get("completed") is not True
            or value.get("selection_complete") is not True
            or value.get("selection_dataset") != "UNSW-NB15"
            or value.get("selection_seeds") != [5, 6]
            or value.get("test_metrics_used_for_selection") is not False
            or value.get("source_commit") != training_source):
        raise ValueError("Selection identity/completeness does not match the frozen training source")
    rows = value["per_version"]
    if len(rows) != len(METHODS) or {r["method"] for r in rows} != set(METHODS):
        raise ValueError("Selection must fix exactly all five registered versions")
    result = {}
    for row in rows:
        p = row["configuration"]
        if (set(p) != {"lambda_g", "der_alpha", "der_beta", "cbf_epochs", "cbf_lr_multiplier"}
                or p["lambda_g"] not in (.5, 1., 2.) or p["der_alpha"] != 1.
                or p["der_beta"] != 1. or p["cbf_epochs"] not in (1, 5)
                or p["cbf_lr_multiplier"] != .1):
            raise ValueError("Frozen selection contains unregistered parameter values")
        result[row["method"]] = dict(p)
    return result, digest(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("code-root", "data-root", "output-root", "protected-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--dataset", choices=("cic17",), default="cic17")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(FORMAL_SEEDS))
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--selection-sha256", help="Expected immutable selection SHA256, required formally")
    parser.add_argument("--training-source-commit", required=True)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--local-smoke", action="store_true")
    args = parser.parse_args()
    wrapper_source = os.environ.get("NCD_SOURCE_COMMIT", "")
    training_source = args.training_source_commit
    if not all(re.fullmatch("[0-9a-f]{40}", s) for s in (wrapper_source, training_source)):
        parser.error("Wrapper and training sources must both be explicit full commits")
    deadline = datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00"))
    if deadline.tzinfo is None or not datetime.now(timezone.utc) < deadline <= OWNER_DEADLINE:
        parser.error("Deadline must be future, timezone-aware, and no later than 19Oct21:50:26 MYT")
    if len(set(args.methods)) != len(args.methods):
        parser.error("Duplicate methods would duplicate execution")
    if not args.local_smoke and args.seeds != list(FORMAL_SEEDS):
        parser.error("Each formal method must run exactly seeds1/2/3/4/42 in that order")
    code, data, output, protected = [p.resolve() for p in
        (args.code_root, args.data_root, args.output_root, args.protected_root)]
    if code != ROOT.resolve():
        parser.error("code-root must be this versioned wrapper checkout")
    if under(output, protected) or under(protected, output):
        parser.error("Active and protective roots must be disjoint")
    if not args.local_smoke:
        user = os.environ.get("USER", "")
        if not user or not os.environ.get("SLURM_JOB_ID", "").isdigit():
            parser.error("Formal execution requires the user's Slurm allocation")
        if not all(under(p, Path.home()) for p in (code, protected)):
            parser.error("Code and protected small results must be in HOME")
        if not all(under(p, Path("/scr/user") / user) for p in (data, output)):
            parser.error("Data and active results must be in the user's scratch")
        if args.selection is None or not args.selection_sha256:
            parser.error("Formal execution requires --selection and --selection-sha256")
    selection = args.selection.resolve() if args.selection else None
    parameters, selection_hash = selected_parameters(selection, training_source, args.local_smoke)
    if args.selection_sha256 and selection_hash != args.selection_sha256:
        parser.error("Immutable selection SHA256 mismatch")
    config = code / "ncd/configs/problem_scout_section21.json"
    settings = json.loads(config.read_text(encoding="utf-8"))
    if (code / settings["problem_scout_data_root"]).resolve() != data:
        parser.error("Preserve the frozen relative data path; do not override the scientific config")
    if settings["cpu_threads"] != 2 or settings["batch_size"] != 256:
        parser.error("Frozen CPU/scientific batch settings changed")
    seeds = [5] if args.local_smoke else list(FORMAL_SEEDS)
    output.mkdir(parents=True, exist_ok=True)
    protected.mkdir(parents=True, exist_ok=True)
    lock = output / ".active.lock"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "job_id": os.environ.get("SLURM_JOB_ID"), "utc": utc()}, stream)
    attempt = output / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"_{os.getpid()}")
    attempt.mkdir(parents=True)
    record = {"job_id": os.environ.get("SLURM_JOB_ID"), "wrapper_source_commit": wrapper_source,
        "training_source_commit": training_source, "dataset": "CIC-IDS-2017", "methods": args.methods,
        "seeds": seeds, "smoke": args.local_smoke, "selection_sha256": selection_hash,
        "selection_parameters": {m:parameters[m] for m in args.methods}, "started_utc": utc(),
        "deadline_utc": deadline.isoformat(), "config_sha256": digest(config),
        "wrapper_sha256": digest(Path(__file__)),
        "batch_entry_sha256": digest(code / "ncd/hpc/run_retention.sbatch"),
        "training_entry_sha256": digest(code / "ncd/run_aacil_retention.py"),
        "telemetry_helper_sha256": digest(code / "ncd/hpc/problem_scout_job.py"),
        "smoke_parameter_scope": "non-scientific fixed defaults; not a selected configuration" if args.local_smoke and selection is None else None,
        "telemetry_policy": {"source": "frozen problem_scout_job.TelemetryGuard",
            "sample_interval_seconds": 60, "failure_limit": 3, "minimum_low_samples": 3,
            "low_utilization_percent": 10, "low_span_threshold": "observed child elapsed / 3",
            "gpu_preflight_required": not args.local_smoke,
            "interpretation": "conservative sampling guard, not proof of continuous use"},
        "child_attempts": []}
    previous, child = {}, None
    started, exit_code, received_signal = time.monotonic(), 1, None

    def stop(signum, _frame):
        nonlocal received_signal
        received_signal = signum
        if child is not None and child.poll() is None:
            child.terminate()

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)
    write_json(attempt / "job_start.json", record)
    try:
        if selection:
            # Preserve original selection bytes; never replace an earlier binding.
            target = output / "selection.json"
            if target.exists() and digest(target) != selection_hash:
                raise ValueError("Existing output is bound to a different immutable selection")
            if not target.exists():
                with target.open("xb") as stream:
                    stream.write(selection.read_bytes())
        identifiers = []
        if not args.local_smoke:
            step = os.environ.get("SLURM_STEP_ID", os.environ.get("SLURM_STEPID", ""))
            if not step.isdigit() or not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
                raise RuntimeError("GPU preflight requires srun and explicit CUDA visibility")
            identifiers = visible_gpu_uuids()
            if len(identifiers) != 1:
                raise RuntimeError("Serial workload requires exactly one allocated visible GPU")
            preflight = {"event": "gpu_preflight", "utc": utc(), **gpu_sample(identifiers)}
            with (attempt / "resources.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(preflight, allow_nan=False) + "\n")
            if not preflight["gpu_telemetry_valid"]:
                raise RuntimeError("GPU telemetry unavailable before training: " + preflight["gpu_measurement_error"])
            record.update(allocated_gpu_uuids=identifiers, gpu_preflight_passed=True, slurm_step_id=step)
        else:
            record.update(gpu_preflight_passed=None, smoke_limit="CPU-only same-entry smoke, not GPU resource measurement")
        write_json(attempt / "job_start.json", record)
        environment = dict(os.environ, NCD_SOURCE_COMMIT=training_source,
            OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        for method in args.methods:
            if datetime.now(timezone.utc) >= deadline or received_signal:
                raise TimeoutError("Owner deadline/signal reached before next method")
            if selection and digest(selection) != selection_hash:
                raise ValueError("Immutable selection changed during serial execution")
            p = parameters[method]
            child_output = output / method
            command = [sys.executable, "-I", "-u", "ncd/run_aacil_retention.py", "--config",
                "ncd/configs/problem_scout_section21.json", "--output-dir", str(child_output),
                "--datasets", "CIC-IDS-2017", "--methods", method, "--seeds", *map(str, seeds),
                "--lambda-g", str(p["lambda_g"]), "--der-alpha", str(p["der_alpha"]),
                "--der-beta", str(p["der_beta"]), "--cbf-epochs", str(p["cbf_epochs"]),
                "--cbf-lr-multiplier", str(p["cbf_lr_multiplier"]),
                "--source-commit", training_source, "--deadline-iso", deadline.isoformat(),
                "--device", "cpu" if args.local_smoke else "cuda"]
            if args.local_smoke:
                command.append("--smoke")
            child_attempt = attempt / method
            child_attempt.mkdir()
            child_record = {"method": method, "command": command, "started_utc": utc(),
                "parameters": p, "selection_sha256": selection_hash, "status": "running"}
            record["child_attempts"].append(child_record)
            child_started, next_sample, child_exit = time.monotonic(), 0.0, None
            child = None
            guard = TelemetryGuard()
            try:
                with (child_attempt / "stdout.log").open("wb") as out, (child_attempt / "stderr.log").open("wb") as err:
                    child = subprocess.Popen(command, cwd=code, env=environment, stdout=out, stderr=err)
                    child_record["child_pid"] = child.pid
                    write_json(child_attempt / "child_start.json", child_record)
                    write_json(attempt / "job_start.json", record)
                    print(json.dumps({"event": "runner_started", **child_record}), flush=True)
                    while child.poll() is None:
                        elapsed = time.monotonic() - child_started
                        if datetime.now(timezone.utc) >= deadline:
                            child_record["guard_stop_reason"] = "owner_hard_deadline"
                            exit_code = 124
                            raise TimeoutError("Owner hard deadline reached; completed seeds remain usable")
                        if received_signal:
                            raise RuntimeError(f"Wrapper received signal {received_signal}")
                        if not args.local_smoke and elapsed >= next_sample:
                            measurement = sample(child.pid, output, identifiers)
                            measurement["child_elapsed_seconds"] = elapsed
                            reason = guard.observe(measurement, elapsed)
                            measurement["guard_trigger"] = reason
                            with (child_attempt / "resources.jsonl").open("a", encoding="utf-8") as stream:
                                stream.write(json.dumps(measurement, allow_nan=False) + "\n")
                            next_sample = elapsed + 60
                            if reason:
                                child_record["guard_stop_reason"] = reason
                                raise RuntimeError("Stopping own child: " + reason)
                        copy_small(output, protected, previous)
                        try:
                            child.wait(timeout=min(5, max(.1, (deadline-datetime.now(timezone.utc)).total_seconds())))
                        except subprocess.TimeoutExpired:
                            pass
                    child_exit = child.returncode
                if child_exit != 0:
                    exit_code = child_exit
                    raise RuntimeError(f"Scientific child exited {child_exit}; no automatic resubmission")
                result_path = child_output / "result.json"
                value = json.loads(result_path.read_text(encoding="utf-8"))
                expected = {"datasets": ["CIC-IDS-2017"], "methods": [method], "seeds": seeds}
                if (value["execution_status"] != "complete" or value["expected"] != expected
                        or len(value["cells"]) != len(seeds) or value["selection_only"]
                        or value["provenance"]["git_commit"] != training_source):
                    raise RuntimeError("Child exited zero without a complete source-matched formal/smoke matrix")
                child_record.update(status="complete", completed_cells=len(value["cells"]),
                    result_sha256=digest(result_path), training_seconds_total=value["training_seconds_total"])
            except BaseException as error:
                child_record.update(status="failed", entry_error=f"{type(error).__name__}: {error}")
                if child is not None and child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                raise
            finally:
                child_record.update(child_returncode=child.poll() if child else None,
                    finished_utc=utc(), elapsed_seconds_this_attempt=time.monotonic()-child_started)
                write_json(child_attempt / "child_exit.json", child_record)
                copy_small(output, protected, previous)
        exit_code = 0
        record.update(execution_status="complete", completed_cells=sum(c["completed_cells"] for c in record["child_attempts"]),
            training_seconds_total=sum(c["training_seconds_total"] for c in record["child_attempts"]))
        return exit_code
    except BaseException as error:
        record.update(execution_status="failed", entry_error=f"{type(error).__name__}: {error}")
        exit_code = exit_code or 1
        raise
    finally:
        record.update(exit_code=exit_code, finished_utc=utc(), elapsed_seconds_this_attempt=time.monotonic()-started)
        write_json(attempt / "job_exit.json", record)
        write_json(output / "result.json", record)
        copy_small(output, protected, previous)
        lock.unlink()
        print(json.dumps({"event": "job_exit", **record}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
