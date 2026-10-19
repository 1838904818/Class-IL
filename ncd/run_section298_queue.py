"""Section29.8 bounded development/selection/formal queue, one GPU child at a time.

Only the runner's permanent-TRAIN-validation metrics select configurations.
New roots and exclusive log/selection creation preserve all earlier attempts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ENTRY = "ncd/run_aacil_retention.py"
HARD_DEADLINE = dt.datetime.fromisoformat("2026-10-19T21:50:26+08:00")
DEV_SEEDS = [5, 6]
FORMAL_SEEDS = [1, 2, 3, 4, 42]
VERSIONS = ["aacil_kd", "aacil_iso", "aacil_wa", "aacil_cbf", "aacil_ncm"]
LAMBDAS = [.5, 1., 2.]
FORMAL_DATASETS = ["UNSW-NB15", "NSL-KDD"]


def now():
    return dt.datetime.now().astimezone()


def label(path):
    return path.resolve().relative_to(ROOT).as_posix()


def atomic_json(path, value):
    """Mutable driver state, not the immutable final selection."""
    temporary = path.with_name(path.name+f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    for attempt in range(40):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.05)


def exclusive_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def specifications():
    candidates = []
    for method in VERSIONS:
        for lam in LAMBDAS:
            for epochs in ([1, 5] if method == "aacil_cbf" else [1]):
                name = f"{method}_lg{lam:g}".replace(".", "p")
                if method == "aacil_cbf":
                    name += f"_cbf{epochs}"
                candidates.append({"candidate_id": name, "method": method,
                    "lambda_g": lam, "der_alpha": 1., "der_beta": 1.,
                    "cbf_epochs": epochs, "cbf_lr_multiplier": .1,
                    "candidate_order": len(candidates)})
    return candidates


def choose(baseline, candidates, source_commit):
    threshold = baseline["validation_mean"]["final_balanced_accuracy"]-.03
    evaluated, per_version = [], []
    for candidate in candidates:
        candidate = candidate | {"eligible": candidate["validation_mean"]["final_balanced_accuracy"] >= threshold}
        evaluated.append(candidate)
    for method in VERSIONS:
        available = [c for c in evaluated if c["method"] == method]
        eligible = [c for c in available if c["eligible"]]
        if eligible:
            selected = min(eligible, key=lambda c: (c["validation_mean"]["forgetting"], c["candidate_order"]))
            reason = "eligible_minimum_validation_forgetting"
        else:
            selected = min(available, key=lambda c: (-c["validation_mean"]["final_balanced_accuracy"],
                c["validation_mean"]["forgetting"], c["candidate_order"]))
            reason = "ineligible_fallback_maximum_validation_final_bacc_not_success"
        per_version.append({"method": method, "selected_candidate_id": selected["candidate_id"],
            "eligible": bool(eligible), "selection_reason": reason,
            "configuration": {k:selected[k] for k in ("lambda_g", "der_alpha", "der_beta", "cbf_epochs", "cbf_lr_multiplier")},
            "validation_mean": selected["validation_mean"], "source_results": selected["source_results"]})
    eligible_main = [v for v in per_version if v["eligible"]]
    main = min(eligible_main, key=lambda v: (v["validation_mean"]["forgetting"], VERSIONS.index(v["method"]))) if eligible_main else None
    return {"completed": True, "selection_complete": True, "completed_at": now().isoformat(),
        "section": "29.8", "source_commit": source_commit, "selection_dataset": "UNSW-NB15",
        "selection_seeds": DEV_SEEDS, "test_metrics_used_for_selection": False,
        "validation_policy": "Permanent stratified 10% TRAIN attack validation outside M, never optimized; permanent Normal calibration pool; no test tuning",
        "original_aacil": baseline, "eligibility_final_bacc_floor": threshold,
        "rule": "Per version: among configurations with mean final validation BAcc >= original AA mean minus .03, minimize mean signed validation forgetting; then select the eligible version with lowest forgetting",
        "configuration_tie_rule": "Declared candidate order: lambda_G .5,1,2; CBF epochs1,5",
        "main_version_tie_rule": VERSIONS,
        "ineligible_fallback_rule": "For mandatory formal reporting only: maximum validation final BAcc, then minimum validation forgetting, then declared candidate order; not an eligible success",
        "candidates": evaluated, "per_version": per_version,
        "selected_main": main, "main_selection_status": "eligible" if main else "no_eligible_version",
        "formal_datasets_local": FORMAL_DATASETS, "formal_seeds": FORMAL_SEEDS,
        "all_five_versions_formally_reported": True,
        "CIC17": "Not scheduled locally; separate explicitly assigned HPC execution only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--deadline", default=HARD_DEADLINE.isoformat())
    parser.add_argument("--output-root", type=Path, default=Path("ncd/results/problem_scout/section29/retention_queue_v1"))
    parser.add_argument("--formal-root", type=Path, default=Path("ncd/results/problem_scout/section29/retention_local_v1"))
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("ncd/configs/problem_scout_section21.json"))
    args = parser.parse_args()
    deadline = dt.datetime.fromisoformat(args.deadline.replace("Z", "+00:00"))
    if deadline.tzinfo is None or deadline > HARD_DEADLINE:
        parser.error("Deadline must include timezone and cannot exceed 2026-10-19T21:50:26+08:00")
    if deadline <= now():
        parser.error("The fixed experiment deadline has already passed; no child may start")
    output, formal, logs = ((ROOT/p).resolve() for p in (args.output_root, args.formal_root, args.log_root))
    if output == formal or output in formal.parents or formal in output.parents:
        parser.error("Queue and formal roots must be distinct non-nested new directories")
    # Output bindings are public repo-relative references; raw external logs stay outside publications.
    label(output); label(formal)
    config = (ROOT/args.config).resolve()
    if not config.is_file() or not (ROOT/ENTRY).is_file():
        parser.error("Configuration or retention runner is unavailable")
    if any(path.exists() for path in (output, formal, logs)):
        parser.error("All output/formal/log roots must be new; prior attempts are never overwritten")
    for path in (output, formal, logs):
        path.mkdir(parents=True)
    specifications_all = specifications()
    state = {"schema_version": 1, "section": "29.8", "status": "running",
        "started_at": now().isoformat(), "driver_pid": os.getpid(), "source_commit": args.source_commit,
        "runner": ENTRY, "deadline": deadline.isoformat(), "absolute_deadline_maximum": HARD_DEADLINE.isoformat(),
        "selection_complete": False, "attempts": [], "completed_candidates": [], "formal_completed_groups": [],
        "planned_candidate_configurations": specifications_all, "candidate_count": len(specifications_all),
        "planned_dev_cells_including_original_AA": 2*(1+len(specifications_all)),
        "planned_formal_cells_local": len(VERSIONS)*len(FORMAL_SEEDS)*len(FORMAL_DATASETS),
        "development_seeds": DEV_SEEDS, "formal_seeds": FORMAL_SEEDS, "formal_dataset_order": FORMAL_DATASETS,
        "version_order": VERSIONS, "one_owned_child_at_a_time": True,
        "CIC17_local_excluded": True, "new_roots_no_overwrite": True}
    running_child = None
    began = time.perf_counter()

    def save():
        state["elapsed_seconds"] = time.perf_counter()-began
        state["recorded_child_elapsed_seconds"] = sum(a.get("elapsed_seconds", 0.) for a in state["attempts"])
        atomic_json(output/"queue.json", state)

    def stop(reason):
        state.update(status=reason, finished_at=now().isoformat())
        if not state["selection_complete"] and not (output/"selection.incomplete.json").exists():
            exclusive_json(output/"selection.incomplete.json", {
                "completed": False, "selection_complete": False, "status": reason,
                "recorded_at": now().isoformat(), "selected_main": None,
                "reason": "The complete bounded development grid did not finish; no configuration/version selection is valid",
                "original_aacil": state.get("original_aacil"), "completed_candidates": state["completed_candidates"],
                "planned_candidate_configurations": specifications_all, "source_commit": args.source_commit})
        save()

    def run(name, child_root, dataset, method, seeds, configuration, dev):
        nonlocal running_child
        left = (deadline-now()).total_seconds()
        if left <= 0:
            stop("hard_deadline"); return None
        command = [sys.executable, "-I", "-B", "-u", ENTRY,
            "--datasets", dataset, "--methods", method, "--seeds", *map(str, seeds),
            "--config", label(config), "--output-dir", label(child_root),
            "--lambda-g", str(configuration["lambda_g"]), "--der-alpha", "1", "--der-beta", "1",
            "--cbf-epochs", str(configuration["cbf_epochs"]), "--cbf-lr-multiplier", ".1",
            "--source-commit", args.source_commit, "--deadline-iso", deadline.isoformat(), "--device", "cuda"]
        if dev:
            command.append("--selection-only")
        env = os.environ.copy()
        env["NCD_SOURCE_COMMIT"] = args.source_commit
        attempt = {"name": name, "phase": "development" if dev else "formal", "dataset": dataset,
            "method": method, "seeds": seeds, "configuration": configuration,
            "started_at": now().isoformat(), "output": label(child_root), "status": "starting",
            "console_log": name+".console.log", "stderr_log": name+".stderr.log",
            "command": ["python", *command[1:]]}
        state["attempts"].append(attempt); state["current_attempt"] = name; save()
        start = time.perf_counter()
        with (logs/(name+".console.log")).open("xb") as stdout, (logs/(name+".stderr.log")).open("xb") as stderr:
            running_child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            attempt.update(pid=running_child.pid, status="running"); save()
            print("CHILD_STARTED "+json.dumps(attempt), flush=True)
            try:
                code = running_child.wait(timeout=max(.001, (deadline-now()).total_seconds()))
            except subprocess.TimeoutExpired:
                running_child.terminate()
                try:
                    code = running_child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    running_child.kill(); code = running_child.wait()
                attempt["status"] = "hard_deadline_terminated_owned_child"
            else:
                attempt["status"] = "process_exited"
            running_child = None
        attempt.update(exit_code=code, finished_at=now().isoformat(), elapsed_seconds=time.perf_counter()-start)
        state["current_attempt"] = None; save()
        print("CHILD_EXITED "+json.dumps(attempt), flush=True)
        if attempt["status"] == "hard_deadline_terminated_owned_child":
            stop("hard_deadline"); return None
        matrix_path = child_root/"result.json"
        if code != 0 or not matrix_path.exists():
            stop("stopped_on_child_failure"); return None
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        if matrix.get("execution_status") != "complete":
            stop("stopped_on_incomplete_child"); return None
        cells = []
        for seed in seeds:
            result_path = child_root/dataset.lower().replace("-", "_")/f"{method}_seed{seed}"/"result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            identity = result.get("identity", {})
            if not result.get("completed") or (identity.get("dataset"), identity.get("method"), identity.get("seed")) != (dataset, method, seed):
                raise ValueError("Child did not produce the requested completed physical cell")
            if identity.get("git_commit") != args.source_commit:
                raise ValueError("Child source commit differs from the queue's fixed source")
            record = {"seed": seed, "result": label(result_path), "completed_at": result["completed_at"],
                      "training_seconds": result.get("training_seconds_total"),
                      "source_commit": identity["git_commit"],
                      "cost_seconds": {k:v for k,v in result.items() if k.endswith("_seconds_total")}}
            if dev:
                if result.get("test_evaluation_performed") is not False:
                    raise ValueError("Development result does not explicitly exclude test evaluation")
                metrics = result["selection_metrics"]
                record["validation_metrics"] = {k:metrics[k] for k in (
                    "final_balanced_accuracy", "mean_incremental_balanced_accuracy", "forgetting")}
                if any(not math.isfinite(v) for v in record["validation_metrics"].values()):
                    raise ValueError("Nonfinite validation selection metric; no seed/configuration may be silently removed")
            cells.append(record)
        attempt.update(status="completed", cells=cells); save()
        return {"source_results": cells, "matrix_result": label(matrix_path),
                "child_elapsed_seconds": attempt["elapsed_seconds"],
                "training_seconds": sum(c["training_seconds"] for c in cells) if all(c["training_seconds"] is not None for c in cells) else None,
                **({"validation_mean": {k:statistics.mean(c["validation_metrics"][k] for c in cells)
                    for k in cells[0]["validation_metrics"]}} if dev else {})}

    save()
    try:
        baseline_config = {"lambda_g": .5, "der_alpha": 1., "der_beta": 1., "cbf_epochs": 1, "cbf_lr_multiplier": .1}
        baseline = run("dev_original_aacil", output/"development"/"original_aacil", "UNSW-NB15", "aacil", DEV_SEEDS, baseline_config, True)
        if baseline is None:
            return
        state["original_aacil"] = baseline; save()
        candidates = []
        for spec in specifications_all:
            result = run("dev_"+spec["candidate_id"], output/"development"/spec["candidate_id"],
                "UNSW-NB15", spec["method"], DEV_SEEDS, spec, True)
            if result is None:
                return
            candidates.append(spec | result)
            state["completed_candidates"] = candidates; save()
        selection = choose(baseline, candidates, args.source_commit)
        selection["development_queue_elapsed_seconds"] = time.perf_counter()-began
        selection["development_child_elapsed_seconds"] = sum(a["elapsed_seconds"] for a in state["attempts"])
        selection["cost_accounting"] = "All development candidates, original-AA reference, and eventual formal execution are retained; nested times are not additive to queue/child walltime"
        exclusive_json(output/"selection.json", selection)
        state.update(selection_complete=True, selection_result=label(output/"selection.json"),
                     selected_main=selection["selected_main"], main_selection_status=selection["main_selection_status"]); save()
        print("SELECTION_COMPLETE "+json.dumps({"selection": label(output/"selection.json"), "selected_main": selection["selected_main"]}), flush=True)
        for dataset in FORMAL_DATASETS:
            for selected in selection["per_version"]:
                method = selected["method"]
                name = "formal_"+dataset.lower().replace("-", "_")+"_"+method
                result = run(name, formal/dataset.lower().replace("-", "_")/method,
                    dataset, method, FORMAL_SEEDS, selected["configuration"], False)
                if result is None:
                    return
                state["formal_completed_groups"].append({"dataset": dataset, "method": method,
                    "selected_candidate_id": selected["selected_candidate_id"], "development_eligible": selected["eligible"], **result})
                save(); print("FORMAL_VERSION_COMPLETE "+dataset+" "+method, flush=True)
        state.update(status="complete", finished_at=now().isoformat()); save()
        print("QUEUE_COMPLETE", flush=True)
    except BaseException as error:
        if running_child is not None and running_child.poll() is None:
            running_child.terminate()
            try:
                running_child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                running_child.kill(); running_child.wait()
        state["error_type"] = type(error).__name__
        # Full traceback goes to the preserved parent stderr, not public sanitized state.
        stop("stopped_on_driver_error")
        raise


if __name__ == "__main__":
    main()
