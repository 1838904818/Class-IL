"""Sequential local controller for the Section 23 development/formal matrix.

All candidate selection uses development seeds 5 and 6 and the child runner's
training-held-out validation metric. Formal seeds are never read for selection.
The child runner owns training/checkpoint resume; this controller owns the grid,
frozen selection, source/config checks, and complete per-attempt process logs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ncd.source_commit import source_commit

DATASETS = ["NSL-KDD", "UNSW-NB15", "CIC-IDS-2017"]
BEST_METHODS = {"NSL-KDD": "replay_lwf", "UNSW-NB15": "rcl_tab", "CIC-IDS-2017": "derpp"}
CORRECTIONS = ["wa", "bic", "ssil", "cbf"]
DEVELOPMENT_SEEDS = [5, 6]
REPORTING_SEEDS = [1, 2, 3, 4, 42]
# Analysis/report changes do not invalidate a training resume. These files do.
SOURCE_PATHS = [
    "ncd/run_bias_correction.py", "ncd/section23_bias.py", "ncd/run_bias_control.py",
    "ncd/run_problem_scout.py", "ncd/crr_benchmarks.py", "ncd/model.py",
    "ncd/prepare_problem_scout_data.py", "ncd/train_cil.py", "ncd/data.py",
    "ncd/source_commit.py", "ncd/run_crr_real_sequence.py",
]


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # A short Windows reader/AV lock must not discard a completed training cell.
    # Retain atomic replacement, and fail visibly after a bounded two seconds.
    for attempt in range(41):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 40:
                raise
            time.sleep(.05)


@contextmanager
def execution_lock(path):
    """One live controller per output root, on Windows and Unix."""
    with Path(path).open("a+b") as handle:
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
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def validate_config(config):
    """Refuse accidental seed leakage or a different Section 23 comparison."""
    if config.get("datasets") != DATASETS:
        raise ValueError("Section 23 requires the three fixed datasets in order")
    if config.get("best_methods") != BEST_METHODS:
        raise ValueError("best_methods must match the fixed Section 22 S2000 winners")
    if config.get("development_seeds") != DEVELOPMENT_SEEDS:
        raise ValueError("candidate selection requires only development seeds 5 and 6")
    if config.get("reporting_seeds") != REPORTING_SEEDS:
        raise ValueError("formal reporting seeds must be 1, 2, 3, 4, 42")
    if (config.get("epochs_per_task"), config.get("min_steps_per_task"), config.get("batch_size")) != (8, 2000, 256):
        raise ValueError("Section 23 requires max(8 epochs, 2000 steps), batch 256")
    candidates = config.get("candidates", {})
    if set(candidates) != set(CORRECTIONS):
        raise ValueError("expected exactly wa, bic, ssil, cbf candidate groups")
    if {key: len(value) for key, value in candidates.items()} != {"wa": 1, "bic": 2, "ssil": 2, "cbf": 2}:
        raise ValueError("expected the bounded 1/2/2/2 candidate grid")
    if any(not isinstance(item, dict) for values in candidates.values() for item in values):
        raise ValueError("each candidate must be a parameter dictionary")


def cell_identity(dataset, base, correction, candidate, seed, phase, smoke=False):
    return {"dataset": dataset, "base": base, "correction": correction,
            "candidate": candidate, "seed": seed, "phase": phase, "smoke": bool(smoke)}


def cell_path(identity):
    dataset = identity["dataset"].lower().replace("-", "_")
    return Path(identity["phase"]) / dataset / identity["base"] / identity["correction"] / \
        f"candidate_{identity['candidate']}_seed_{identity['seed']}"


def development_cells(config, smoke=False):
    for dataset in config["datasets"]:
        for base in ("replay", config["best_methods"][dataset]):
            for correction in CORRECTIONS:
                candidates = range(1 if smoke else len(config["candidates"][correction]))
                for candidate in candidates:
                    for seed in ([5] if smoke else config["development_seeds"]):
                        yield cell_identity(dataset, base, correction, candidate, seed, "development", smoke)


def formal_cells(config, selection):
    chosen = {(row["dataset"], row["base"], row["correction"]): row["selected_candidate"]
              for row in selection["selections"]}
    for dataset in config["datasets"]:
        for base in ("replay", config["best_methods"][dataset]):
            for correction in CORRECTIONS:
                candidate = chosen[(dataset, base, correction)]
                for seed in config["reporting_seeds"]:
                    yield cell_identity(dataset, base, correction, candidate, seed, "formal")


def provenance(config_path, root=ROOT, previous=None):
    """Keep the original commit across documentation-only HEAD advances."""
    old_commit = previous.get("git_commit") if previous else None
    override = os.environ.get("NCD_SOURCE_COMMIT", "").strip().lower()
    if old_commit and override and override != old_commit:
        raise ValueError("resume NCD_SOURCE_COMMIT differs from the original run")
    commit = old_commit or source_commit(root)
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("source commit must be a full 40-character SHA-1")
    spec = load(config_path)
    base_path = root / spec["base_config"]
    base = load(base_path)
    inputs = {spec["base_config"], *base["lambda_files"].values()}
    current = {"git_commit": commit, "config_sha256": digest(config_path),
               "config_input_hashes": {path: digest(root / path) for path in sorted(inputs)},
               "source_hashes": {path: digest(root / path) for path in SOURCE_PATHS}}
    if previous is not None and current != previous:
        raise ValueError("training source/config changed; preserve this run and use a new output root")
    return current


def _bound_value(result, name):
    for container in (result.get("identity", {}), result.get("provenance", {}), result):
        if name in container:
            return container[name]
    raise ValueError(f"child result lacks {name}")


def validate_child(path, identity, bound, require_complete=True, selection_sha256=None):
    """Read JSON/weights without unpickling; reject identity or resume drift."""
    path = Path(path)
    result = load(path)
    for key, expected in identity.items():
        if _bound_value(result, "base_method" if key == "base" else key) != expected:
            raise ValueError(f"child {path.name} {key} does not match the requested cell")
    for key in ("git_commit", "config_sha256"):
        if _bound_value(result, key) != bound[key]:
            raise ValueError(f"child {key} does not match controller provenance")
    if _bound_value(result, "config_input_hashes") != bound["config_input_hashes"]:
        raise ValueError("child base configuration/lambda inputs do not match controller provenance")
    hashes = _bound_value(result, "source_hashes")
    if any(hashes.get(name) != value for name, value in bound["source_hashes"].items()):
        raise ValueError("child training source hashes do not match controller provenance")
    expected_role = ("training_holdout_not_official_ID_test" if identity["phase"] == "development"
                     else "official_ID_test_not_used_for_selection")
    if _bound_value(result, "evaluation_role") != expected_role:
        raise ValueError("child evaluation role is inconsistent with development/formal phase")
    if identity["phase"] == "formal":
        selected = _bound_value(result, "frozen_selection")
        if not selection_sha256 or not isinstance(selected, dict) or selected.get("sha256") != selection_sha256:
            raise ValueError("formal child does not match the frozen development selection")
    complete = result.get("completed") is True or result.get("execution_status") == "complete"
    if not complete:
        if require_complete:
            raise ValueError("child exited without a complete result")
        return None
    checkpoints = result.get("checkpoints", [])
    if not checkpoints:
        raise ValueError("complete child has no checkpoints")
    tasks = result.get("identity", {}).get("tasks")
    if tasks is not None and len(checkpoints) != len(tasks):
        raise ValueError("complete child checkpoint count does not match its tasks")
    identity_hash = hashlib.sha256(json.dumps(result["identity"], sort_keys=True,
                                  separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    for index, checkpoint in enumerate(checkpoints):
        if "checkpoint_file" not in checkpoint or "checkpoint_sha256" not in checkpoint:
            raise ValueError("complete child checkpoint is missing its file/hash")
        weight_path = (path.parent / checkpoint["checkpoint_file"]).resolve()
        if not weight_path.is_relative_to(path.parent.resolve()):
            raise ValueError("checkpoint path escapes the cell directory")
        if not weight_path.is_file() or digest(weight_path) != checkpoint["checkpoint_sha256"]:
            raise ValueError("complete child checkpoint file is missing or changed")
        if checkpoint.get("identity_hash") != identity_hash or checkpoint.get("task_index") != index:
            raise ValueError("checkpoint identity/task index does not match the complete cell")
        if checkpoint.get("evaluation_role") != expected_role:
            raise ValueError("checkpoint evaluation role does not match the cell")
        sidecar = weight_path.with_suffix(".json")
        if not sidecar.is_file() or load(sidecar) != checkpoint:
            raise ValueError("checkpoint metrics sidecar is missing or changed")
    score = checkpoints[-1].get("id", {}).get("macro_f1")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("final child Macro-F1 must be a finite proportion")
    return {"identity": identity, "result_sha256": digest(path), "macro_f1": float(score),
            "checkpoint_count": len(checkpoints)}


def freeze_selection(config, out, bound):
    """Build/recheck the immutable choices using development paths only."""
    rows = []
    for dataset in config["datasets"]:
        for base in ("replay", config["best_methods"][dataset]):
            for correction in CORRECTIONS:
                candidates = []
                for candidate, parameters in enumerate(config["candidates"][correction]):
                    evidence = []
                    for seed in DEVELOPMENT_SEEDS:
                        identity = cell_identity(dataset, base, correction, candidate, seed, "development")
                        relative = cell_path(identity) / "result.json"
                        proof = validate_child(out / relative, identity, bound)
                        evidence.append({"seed": seed, "result": relative.as_posix(), **proof})
                    candidates.append({"candidate": candidate, "parameters": parameters,
                                       "mean_validation_macro_f1": math.fsum(e["macro_f1"] for e in evidence) / len(evidence),
                                       "development_results": evidence})
                winner = max(candidates, key=lambda row: (row["mean_validation_macro_f1"], -row["candidate"]))
                rows.append({"dataset": dataset, "base": base, "correction": correction,
                             "selected_candidate": winner["candidate"], "candidates": candidates})
    selection = {"schema_version": 1, "provenance": bound,
                 "selection_seeds": DEVELOPMENT_SEEDS,
                 "selection_metric": "final training-held-out validation Macro-F1",
                 "tie_break": "lower candidate index", "selections": rows}
    path = out / "selection.json"
    if path.exists():
        if load(path) != selection:
            raise ValueError("frozen selection or development evidence changed; refusing formal continuation")
    else:
        save(path, selection)
    return selection


def relative_argument(path, root=ROOT):
    return Path(os.path.relpath(Path(path).resolve(), root)).as_posix()


def child_command(identity, config_path, directory, root=ROOT, selection_path=None):
    command = [sys.executable, "-I", "-u", "ncd/run_bias_correction.py",
               "--config", relative_argument(config_path, root),
               "--output-dir", relative_argument(directory, root)]
    for key in ("dataset", "base", "correction", "candidate", "seed", "phase"):
        command += [f"--{key}", str(identity[key])]
    if identity["smoke"]:
        command += ["--smoke"]
    if identity["phase"] == "formal":
        if selection_path is None:
            raise ValueError("formal child requires frozen development selection")
        command += ["--selection", relative_argument(selection_path, root)]
    return command


def _event(out, event, **fields):
    entry = {"time": timestamp(), "event": event, **fields}
    line = json.dumps(entry, allow_nan=False)
    with (out / "driver_attempts.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    with (out / "driver.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def run_cell(identity, config_path, out, bound, record, environment, root=ROOT):
    # Detect edits during a multi-hour matrix before launching the next cell.
    provenance(config_path, root, bound)
    directory = out / cell_path(identity)
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    selection_path = out / "selection.json" if identity["phase"] == "formal" else None
    selection_sha256 = record.get("selection_sha256") if selection_path is not None else None
    if selection_path is not None and digest(selection_path) != selection_sha256:
        raise ValueError("frozen selection changed during the formal matrix")
    proof = validate_child(result_path, identity, bound, require_complete=False,
                           selection_sha256=selection_sha256) if result_path.exists() else None
    if proof is not None:
        _event(out, "cell_skipped_complete", identity=identity)
        return proof
    number = 1
    while (directory / f"console.attempt_{number:04d}.log").exists() or \
            (directory / f"stderr.attempt_{number:04d}.log").exists():
        number += 1
    command = child_command(identity, config_path, directory, root, selection_path)
    stdout_path = directory / f"console.attempt_{number:04d}.log"
    stderr_path = directory / f"stderr.attempt_{number:04d}.log"
    attempt = {"identity": identity, "attempt": number, "command": ["python", *command[1:]],
               "stdout": stdout_path.relative_to(out).as_posix(),
               "stderr": stderr_path.relative_to(out).as_posix(), "started_at": timestamp()}
    record.update(current_cell=attempt, active_phase=identity["phase"])
    save(out / "result.json", record)
    _event(out, "child_start", **attempt)
    started = time.perf_counter()
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=stdout, stderr=stderr)
        attempt["process_id"] = process.pid
        save(out / "result.json", record)
        code = process.wait()
    attempt.update(exit_code=code, finished_at=timestamp(), process_seconds=time.perf_counter() - started)
    save(out / "result.json", record)
    _event(out, "child_exit", **attempt)
    if code:
        raise RuntimeError(f"child failed with exit code {code}; diagnose the retained logs before resuming")
    provenance(config_path, root, bound)
    return validate_child(result_path, identity, bound, selection_sha256=selection_sha256)


def run(config_path, out, smoke=False, root=ROOT):
    config_path, out = Path(config_path).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with execution_lock(out / ".execution.lock"):
        prior = load(out / "result.json") if (out / "result.json").exists() else None
        if prior is not None and prior.get("smoke") != bool(smoke):
            raise ValueError("smoke and scientific runs require different output roots")
        config = load(config_path)
        validate_config(config)
        bound = provenance(config_path, root, prior.get("provenance") if prior else None)
        record = {"schema_version": 1, "status": "running", "smoke": bool(smoke),
                  "provenance": bound, "process_id": os.getpid(),
                  "started_at": prior["started_at"] if prior else timestamp(),
                  "attempt_started_at": timestamp(), "driver_attempt": 1 + (prior.get("driver_attempt", 0) if prior else 0),
                  "expected_cells": {"development": 24 if smoke else 84, "formal": 0 if smoke else 120},
                  "completed_cells": {"development": 0, "formal": 0}, "cells": [],
                  "active_phase": "development", "current_cell": None,
                  "command": ["python", "-I", "-u", "ncd/run_bias_control.py", "--config", relative_argument(config_path, root),
                              "--output-dir", relative_argument(out, root)] + (["--smoke"] if smoke else [])}
        environment = os.environ.copy()
        environment["NCD_SOURCE_COMMIT"] = bound["git_commit"]
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            environment[key] = "2"
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONUNBUFFERED"] = "1"
        save(out / "result.json", record)
        _event(out, "driver_start", process_id=os.getpid(), driver_attempt=record["driver_attempt"],
               git_commit=bound["git_commit"], smoke=smoke)
        try:
            phases = [("development", list(development_cells(config, smoke)))]
            for phase, identities in phases:
                for identity in identities:
                    proof = run_cell(identity, config_path, out, bound, record, environment, root)
                    record["cells"].append({"result": (cell_path(identity) / "result.json").as_posix(), **proof})
                    record["completed_cells"][phase] += 1
                    record.update(current_cell=None)
                    save(out / "result.json", record)
                if phase == "development" and not smoke:
                    selection = freeze_selection(config, out, bound)
                    record["selection_sha256"] = digest(out / "selection.json")
                    _event(out, "selection_frozen", selection_sha256=record["selection_sha256"], groups=len(selection["selections"]))
                    phases.append(("formal", list(formal_cells(config, selection))))
                save(out / "result.json", record)
            if record["completed_cells"] != record["expected_cells"]:
                raise RuntimeError("controller matrix count does not match the declared grid")
            record.update(status="complete", active_phase=None, current_cell=None, finished_at=timestamp())
            save(out / "result.json", record)
            _event(out, "driver_complete", completed_cells=record["completed_cells"], smoke=smoke)
            return 0
        except Exception as error:
            record.update(status="failed", finished_at=timestamp(), error=f"{type(error).__name__}: {error}")
            save(out / "result.json", record)
            _event(out, "driver_failed", error=record["error"], completed_cells=record["completed_cells"])
            traceback.print_exc()
            return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("ncd/configs/bias_correction_section23.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    return run(args.config, args.output_dir, args.smoke)


if __name__ == "__main__":
    raise SystemExit(main())
