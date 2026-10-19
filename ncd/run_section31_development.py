"""Serial compact-family development selection, then optional formal reporting.

All selection inputs are UNSW/NSL seeds5/6 fixed TRAIN-validation metrics.
The 16 candidate configurations and four physical external references require
80 development cells. Replay+LA is a logical calibrated-total-tau Replay alias.
No formal child starts before the entire development grid has completed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("UNSW-NB15", "NSL-KDD")
DEV_SEEDS = (5, 6)
FORMAL_SEEDS = (1, 2, 3, 4, 42)
METHODS = ("aacil_kd_weak", "aacil_kd_wa", "aacil_kd_cbf", "aacil_ema_kd", "aacil_dual")
EXTERNAL = ("replay", "derpp", "wacb", "freq", "replay_la")
CANDIDATE_ENTRY = "ncd/run_aacil_compact.py"
EXTERNAL_ENTRY = "ncd/section31_validation.py"
METRICS = ("final_balanced_accuracy", "mean_incremental_balanced_accuracy", "forgetting")


def now():
    return datetime.now().astimezone().isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label(path):
    return path.resolve().relative_to(ROOT).as_posix()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path, value):
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    for attempt in range(40):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.05)


def exclusive_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


@contextmanager
def driver_lock(path):
    """An OS-held lock prevents simultaneous resumed drivers on this root."""
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def configurations():
    result = []
    for method in METHODS:
        for lam in (.25, .5):
            for alpha in ((.25, .5) if method == "aacil_kd_weak" else (1.,)):
                for beta in ((.25, .5) if method == "aacil_kd_weak" else (1.,)):
                    config = {"lambda_g": lam, "der_alpha": alpha, "der_beta": beta,
                        "cbf_epochs": 1, "cbf_lr_multiplier": .1,
                        "ema_decay": .999, "ema_consistency": 1.}
                    name = f"{method}_lg{lam:g}_a{alpha:g}_b{beta:g}".replace(".", "p")
                    result.append({"candidate_id": name, "method": method,
                        "configuration": config, "candidate_order": len(result)})
    return result


def require_complete_group(root, dataset, method, seeds, source_commit, development, configuration=None):
    matrix = read(root/"result.json")
    if matrix.get("execution_status") != "complete" or matrix.get("smoke"):
        raise ValueError("Selection/reporting requires a complete nonsmoke child matrix")
    if matrix.get("expected") != {"datasets": [dataset], "methods": [method], "seeds": list(seeds)}:
        raise ValueError("Child matrix dataset/method/seed selection differs")
    cells = []
    for seed in seeds:
        path = root/dataset.lower().replace("-", "_")/f"{method}_seed{seed}"/"result.json"
        row = read(path)
        identity = row["identity"]
        if not row.get("completed") or tuple(identity.get(k) for k in ("dataset", "method", "seed", "git_commit")) != (dataset, method, seed, source_commit):
            raise ValueError("Completed child identity/source differs")
        for source, expected_hash in identity.get("source_hashes", {}).items():
            if digest(ROOT/source) != expected_hash:
                raise ValueError("Completed child imports differ from the current fixed source")
        for key, expected_value in (configuration or {}).items():
            bound_key = {"der_alpha": "derpp_alpha", "der_beta": "derpp_beta"}.get(key, key)
            if identity["effective_config"].get(bound_key) != expected_value:
                raise ValueError("Completed candidate configuration differs from its assigned grid point")
        stamp = datetime.fromisoformat(row["completed_at"].replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("Completion timestamp is not timezone-aware")
        record = {"seed": seed, "result": label(path), "result_sha256": digest(path),
            "completed_at": row["completed_at"], "source_commit": source_commit,
            "training_seconds": row.get("training_seconds_total"),
            "cost_seconds": {k: v for k, v in row.items() if k.endswith("_seconds_total")}}
        if development:
            if row.get("test_evaluation_performed") is not False or not row.get("selection_only"):
                raise ValueError("Development selection forbids TEST-evaluated results")
            record["validation_metrics"] = {k: row["selection_metrics"][k] for k in METRICS}
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in record["validation_metrics"].values()):
                raise ValueError("Missing/nonfinite selection metric; do not drop a seed or configuration")
        if method in METHODS:
            capacity = row.get("architecture_capacity")
            if not isinstance(capacity, dict):
                raise ValueError("Compact candidate is missing architecture capacity accounting")
            record["architecture_capacity"] = capacity
        cells.append(record)
    result = {"dataset": dataset, "method": method, "source_results": cells,
        "matrix_result": label(root/"result.json"), "matrix_sha256": digest(root/"result.json"),
        "training_seconds": sum(c["training_seconds"] for c in cells) if all(c["training_seconds"] is not None for c in cells) else None}
    if development:
        result["validation_mean"] = {k: statistics.mean(c["validation_metrics"][k] for c in cells) for k in METRICS}
    return result


def choose(external, candidates, source_commit):
    winners, ceilings = {}, {}
    for dataset in DATASETS:
        if set(external[dataset]) != set(EXTERNAL):
            raise ValueError("All five logical external validation references are required")
        winner = min(EXTERNAL, key=lambda m: (external[dataset][m]["validation_mean"]["forgetting"], EXTERNAL.index(m)))
        winners[dataset] = {"method": winner, **external[dataset][winner]}
        ceilings[dataset] = winners[dataset]["validation_mean"]["forgetting"] + .01
    evaluated = []
    for candidate in candidates:
        if set(candidate["datasets"]) != set(DATASETS):
            raise ValueError("Each candidate configuration requires both development datasets")
        checks = {dataset: candidate["datasets"][dataset]["validation_mean"]["forgetting"] <= ceilings[dataset] for dataset in DATASETS}
        evaluated.append(candidate | {"eligible": all(checks.values()), "forgetting_constraint_by_dataset": checks,
            "average_validation_final_bacc": statistics.mean(candidate["datasets"][d]["validation_mean"]["final_balanced_accuracy"] for d in DATASETS),
            "average_validation_forgetting": statistics.mean(candidate["datasets"][d]["validation_mean"]["forgetting"] for d in DATASETS)})
    if len(evaluated) != len(configurations()) or {c["candidate_id"] for c in evaluated} != {c["candidate_id"] for c in configurations()}:
        raise ValueError("The entire fixed candidate grid must complete before selection")
    versions = []
    for method in METHODS:
        rows = [c for c in evaluated if c["method"] == method]
        eligible = [c for c in rows if c["eligible"]]
        selected = min(eligible, key=lambda c: (-c["average_validation_final_bacc"], c["candidate_order"])) if eligible else min(rows, key=lambda c: (c["average_validation_forgetting"], -c["average_validation_final_bacc"], c["candidate_order"]))
        versions.append({"method": method, "selected_candidate_id": selected["candidate_id"],
            "eligible": bool(eligible), "configuration": selected["configuration"],
            "average_validation_final_bacc": selected["average_validation_final_bacc"],
            "average_validation_forgetting": selected["average_validation_forgetting"],
            "datasets": selected["datasets"], "forgetting_constraint_by_dataset": selected["forgetting_constraint_by_dataset"],
            "selection_reason": "eligible_maximum_average_validation_final_bacc" if eligible else "ineligible_minimum_average_signed_forgetting_descriptive_fallback"})
    eligible = [v for v in versions if v["eligible"]]
    if eligible:
        best = max(v["average_validation_final_bacc"] for v in eligible)
        tied = [v for v in eligible if best - v["average_validation_final_bacc"] <= .01 + 1e-12]
        # All no-extra-parameter methods have equal full-head capacity. EMA and
        # DUAL are ordered after them exactly as the owner specified.
        selected = min(tied, key=lambda v: METHODS.index(v["method"]))
    else:
        tied = []
        selected = min(versions, key=lambda v: (v["average_validation_forgetting"], -v["average_validation_final_bacc"], METHODS.index(v["method"])))
    return {"schema_version": 1, "completed": True, "selection_complete": True,
        "completed_at": now(), "source_commit": source_commit,
        "selection_datasets": list(DATASETS), "selection_seeds": list(DEV_SEEDS),
        "test_metrics_used_for_selection": False, "validation_policy": "Fixed 10% attack TRAIN validation plus permanent Normal calibration, outside M, never optimized; alpha=.02 per stage",
        "external_references": external, "external_lowest_forgetting_winner": winners,
        "forgetting_ceiling_by_dataset": ceilings, "forgetting_tolerance_fraction": .01,
        "rule": "Both datasets' validation forgetting <= fixed external five-method minimum mean plus .01; per candidate maximize two-dataset mean final validation BAcc",
        "main_rule": "Maximum mean validation final BAcc among eligible candidates; within .01 of best apply fewer-components/parameters priority",
        "simplicity_tie_order": list(METHODS), "simplicity_tied_versions": [v["method"] for v in tied],
        "configuration_exact_tie_rule": "Declared grid order",
        "fallback_rule": "If no eligible configuration/version, minimum two-dataset mean signed validation forgetting; descriptive constraint failure, not successful selection",
        "candidates": evaluated, "per_version": versions, "selected_main": selected,
        "main_selection_status": "eligible" if eligible else "forgetting_constraint_not_satisfied",
        "formal_datasets_local": list(DATASETS), "formal_seeds": list(FORMAL_SEEDS),
        "formal_interpretation": "Results on exposed development datasets, not blind confirmation",
        "all_five_versions_formally_reported": True,
        "CSE18": "Separate HPC execution only; no CSE metrics enter development selection"}


def pid_exists(pid):
    if os.name == "nt":
        # os.kill(pid, 0) is not a harmless existence probe on Windows.
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32))
        kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return ctypes.get_last_error() != 87  # Invalid PID is gone; access failures fail closed.
        try:
            code = ctypes.c_uint32()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--config", type=Path, default=Path("ncd/configs/problem_scout_section21.json"))
    parser.add_argument("--output-root", type=Path, default=Path("ncd/results/problem_scout/section31/development_v1"))
    parser.add_argument("--formal-root", type=Path, default=Path("ncd/results/problem_scout/section31/formal_local_v1"))
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--development-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        parser.error("A complete fixed source commit is required")
    output, formal, logs = ((ROOT/p).resolve() for p in (args.output_root, args.formal_root, args.log_root))
    config = (ROOT/args.config).resolve()
    for path in (output, formal):
        label(path)
    if output == formal or output in formal.parents or formal in output.parents:
        parser.error("Development and formal roots must be distinct and non-nested")
    if not config.is_file() or any(not (ROOT/p).is_file() for p in (CANDIDATE_ENTRY, EXTERNAL_ENTRY)):
        parser.error("Configuration or child entry is unavailable")
    identity = {"source_commit": args.source_commit, "config": label(config), "config_sha256": digest(config),
        "driver_sha256": digest(Path(__file__)), "candidate_entry_sha256": digest(ROOT/CANDIDATE_ENTRY),
        "external_entry_sha256": digest(ROOT/EXTERNAL_ENTRY), "planned_candidate_configurations": configurations(),
        "formal_root": label(formal), "development_only": args.development_only}
    if args.resume:
        if not (output/"queue.json").is_file() or not logs.is_dir():
            parser.error("Resume requires preserved queue state and log directory")
        state = read(output/"queue.json")
        if state["identity"] != identity:
            parser.error("Queue source/config/grid/mode changed; never mix attempts")
        # A previous driver may have died while its child continued. An existing
        # PID is conservatively treated as live, including PID reuse.
        for attempt in state["attempts"]:
            if attempt.get("status") == "running" and attempt.get("pid"):
                if pid_exists(attempt["pid"]):
                    parser.error("A recorded child PID still exists; no duplicate resume")
    else:
        if any(path.exists() for path in (output, formal, logs)):
            parser.error("Output/formal/log roots must be new; use explicit resume only for the same immutable queue")
        for path in (output, formal, logs):
            path.mkdir(parents=True)
        state = {"schema_version": 1, "identity": identity, "status": "running", "started_at": now(),
            "selection_complete": False, "current_attempt": None, "attempts": [],
            "external_references": {}, "completed_candidates": [], "formal_completed_groups": [],
            "planned_development_physical_cells": 80, "planned_formal_physical_cells": 50,
            "one_owned_gpu_child_at_a_time": True, "CSE18_local_excluded": True}
    child = None
    session_started = time.perf_counter()

    def save():
        state.update(driver_pid=os.getpid(), current_session_elapsed_seconds=time.perf_counter()-session_started,
                     recorded_attempt_wall_seconds=sum(a.get("elapsed_seconds", 0.) for a in state["attempts"]))
        atomic_json(output/"queue.json", state)

    def run(name, entry, child_root, dataset, method, seeds, configuration, development):
        nonlocal child
        previous = [a for a in state["attempts"] if a["name"] == name]
        if previous and previous[-1].get("status") == "completed":
            return require_complete_group(child_root, dataset, method, seeds, args.source_commit, development, configuration)
        command = [sys.executable, "-I", "-B", "-u", entry, "--datasets", dataset, "--methods", method,
            "--seeds", *map(str, seeds), "--config", label(config), "--output-dir", label(child_root),
            "--source-commit", args.source_commit, "--device", "cuda"]
        if entry == CANDIDATE_ENTRY:
            for key, value in configuration.items():
                command += ["--"+key.replace("_", "-"), str(value)]
            if development:
                command.append("--selection-only")
        attempt_id = f"{name}_attempt{len(previous)+1:03d}"
        attempt = {"name": name, "attempt_id": attempt_id, "phase": "development" if development else "formal",
            "dataset": dataset, "method": method, "seeds": list(seeds), "configuration": configuration,
            "command": ["python", *command[1:]], "started_at": now(), "status": "starting",
            "output": label(child_root), "console_log": attempt_id+".console.log", "stderr_log": attempt_id+".stderr.log"}
        state["attempts"].append(attempt)
        state.update(current_attempt=attempt_id, status="running")
        save()
        env = os.environ.copy()
        env["NCD_SOURCE_COMMIT"] = args.source_commit
        began = time.perf_counter()
        with (logs/attempt["console_log"]).open("xb") as stdout, (logs/attempt["stderr_log"]).open("xb") as stderr:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            attempt.update(pid=child.pid, status="running")
            save()
            print("CHILD_STARTED "+json.dumps({"name": attempt_id, "pid": child.pid}), flush=True)
            code = child.wait()
            child = None
        attempt.update(exit_code=code, elapsed_seconds=time.perf_counter()-began, finished_at=now(), status="process_exited")
        state["current_attempt"] = None
        save()
        if code != 0:
            raise RuntimeError(f"Child {attempt_id} exited {code}; preserve logs and diagnose before explicit resume")
        result = require_complete_group(child_root, dataset, method, seeds, args.source_commit, development, configuration)
        attempt.update(status="completed", cells=result["source_results"])
        save()
        print("GROUP_COMPLETE "+json.dumps({"dataset": dataset, "method": method, "phase": attempt["phase"]}), flush=True)
        return result

    with driver_lock(output/"queue.lock"):
        save()
        try:
            external = {}
            for dataset in DATASETS:
                external[dataset] = {}
                for method in EXTERNAL[:-1]:
                    slug = dataset.lower().replace("-", "_")
                    external[dataset][method] = run(f"external_{slug}_{method}", EXTERNAL_ENTRY,
                        output/"external"/slug/method, dataset, method, DEV_SEEDS, {}, True)
                    state["external_references"] = external
                    save()
                external[dataset]["replay_la"] = external[dataset]["replay"] | {
                    "method": "replay_la", "logical_alias_of": "replay", "additional_training_seconds": 0.,
                    "alias_reason": "Same Replay checkpoint and Normal-calibrated total tau; no extra forward"}
            candidates = []
            for spec in configurations():
                data = {}
                for dataset in DATASETS:
                    slug = dataset.lower().replace("-", "_")
                    data[dataset] = run(f"candidate_{slug}_{spec['candidate_id']}", CANDIDATE_ENTRY,
                        output/"candidates"/spec["candidate_id"]/slug, dataset, spec["method"], DEV_SEEDS, spec["configuration"], True)
                candidates.append(spec | {"datasets": data})
                state["completed_candidates"] = candidates
                save()
            selection_path = output/"selection.json"
            if selection_path.exists():
                selection = read(selection_path)
                expected_selection = choose(external, candidates, args.source_commit)
                expected_selection["completed_at"] = selection["completed_at"]
                for key in ("development_recorded_attempt_wall_seconds", "cost_accounting"):
                    expected_selection[key] = selection[key]
                if selection != expected_selection:
                    raise ValueError("Immutable selection no longer matches exact completed validation inputs")
            else:
                selection = choose(external, candidates, args.source_commit)
                selection["development_recorded_attempt_wall_seconds"] = sum(a.get("elapsed_seconds", 0.) for a in state["attempts"] if a["phase"] == "development")
                selection["cost_accounting"] = "All physical development attempts retained; cell training/calibration totals are nested within child walltime, not added to it; ReplayLA alias charged zero additional training"
                exclusive_json(selection_path, selection)
            state.update(selection_complete=True, selection_result=label(selection_path), selection_sha256=digest(selection_path),
                selected_main=selection["selected_main"], main_selection_status=selection["main_selection_status"])
            save()
            print("SELECTION_COMPLETE "+json.dumps({"selection": label(selection_path), "method": selection["selected_main"]["method"], "status": selection["main_selection_status"]}), flush=True)
            if not args.development_only:
                completed = []
                for dataset in DATASETS:
                    slug = dataset.lower().replace("-", "_")
                    for version in selection["per_version"]:
                        method = version["method"]
                        result = run(f"formal_{slug}_{method}", CANDIDATE_ENTRY, formal/slug/method,
                            dataset, method, FORMAL_SEEDS, version["configuration"], False)
                        completed.append(result | {"selected_candidate_id": version["selected_candidate_id"], "development_eligible": version["eligible"]})
                        state["formal_completed_groups"] = completed
                        save()
                        print("FORMAL_VERSION_COMPLETE "+dataset+" "+method, flush=True)
            state.update(status="complete", finished_at=now(), current_attempt=None)
            save()
            print("QUEUE_COMPLETE", flush=True)
        except BaseException as error:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            state.update(status="stopped_on_error", error_type=type(error).__name__, stopped_at=now())
            save()
            raise


if __name__ == "__main__":
    main()
