"""Bounded CSE18 reference adapter and one-child Slurm supervisor.

Frozen training functions are imported, never edited. This new adapter extends
their dataset entry to the full Section21 CSE18 cache and records its own source
identity separately. Reference results remain sealed until all seven physical
reference groups (five seeds each) are complete; Replay+LA is derived, not trained.
No scheduler submission, retry, selection, or hyperparameter search occurs here.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from problem_scout_job import (FULL_COUNTS, SLUGS, TelemetryGuard, copy_small,
                               gpu_sample, sample, under, utc, visible_gpu_uuids,
                               write_json)

DATASET = "CSE-CIC-IDS2018"
METHODS = ("replay", "derpp", "wacb", "freq", "replay_bs", "aacil", "aacil_kd")
SEEDS = (1, 2, 3, 4, 42)
FROZEN = {
    "ncd/run_problem_scout.py": ("b835a9309d8aa3bfb20e4e30eb426cb8b076e9d1", "2d7f5cafe12469166424caf3c78a5c6784c6d6be379c93d59510721d04f4f0fa"),
    "ncd/run_replay_bs.py": (None, "b398d0f2a5c7c75d9363ea3c93bee36a74e3183814873480cc9c9b2429cf81f2"),
    "ncd/run_aacil.py": ("414136b418cbb5f682875309b00e4f8151240bfe", "8e8264d56149012c454fcc4691c1159ccc3f4d69ffdbb57ecda6aaa77c88f8ee"),
    "ncd/run_aacil_retention.py": ("7418b7d418f9974e88cba5dcc66c6126df117264", "7947c110d5dc75dc7dcdba6a0a3d49026ca1595d41332da19191cff5b3cac2b8"),
}
DATASET_KEYS = {"NSL-KDD": "nsl", "UNSW-NB15": "unsw", DATASET: "cic18"}
BASELINE = {"replay", "derpp", "wacb", "freq"}
ENTRY = "ncd/hpc/section31_job.py"
SENTINEL = "9999-12-31T23:59:59+00:00"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cell_root(output, method, seed):
    return output / "cells" / method / f"seed_{seed}"


def completed_path(output, method, seed):
    return cell_root(output, method, seed) / "section31_result.json"


def source_binding():
    from ncd import run_aacil_retention as retention, run_replay_bs as bs
    paths = set(retention.SOURCE_PATHS) | set(bs.SOURCES) | {
        ENTRY, "ncd/hpc/run_section31.sbatch", "ncd/hpc/problem_scout_job.py",
        "ncd/analyze_stagewise_forgetting.py"}
    hashes = {path: sha(ROOT / path) for path in sorted(paths)}
    for path, (_, expected) in FROZEN.items():
        if hashes[path] != expected:
            raise ValueError(f"Frozen reference entry differs: {path}")
    return hashes


def check_cache(data_root, dataset):
    folder = data_root / SLUGS[DATASET_KEYS[dataset]]
    receipt = folder / "data_quality.json"
    info = read(receipt)
    if info.get("status") != "complete" or info.get("dataset") != dataset:
        raise ValueError("Prepared full-cache receipt is absent, incomplete, or mismatched")
    counts = (sum(info["train_counts"].values()), sum(info["id_counts"].values()))
    if counts != FULL_COUNTS[DATASET_KEYS[dataset]]:
        raise ValueError("Section21.8 requires the full uncapped TRAIN/TEST cache")
    required = {"train_x.npy", "train_y.npy", "test_x.npy", "test_y.npy"}
    files = info.get("cache_files", [])
    if len(files) != 4 or {row["path"] for row in files} != required:
        raise ValueError("Full-cache receipt must identify exactly four arrays")
    for row in files:
        if (folder / row["path"]).stat().st_size != row["bytes"]:
            raise ValueError("Prepared full-cache array is absent or has the wrong size")
    return {"dataset": dataset, "receipt_sha256": sha(receipt),
            "train_rows": counts[0], "test_rows": counts[1],
            "task_count": len(info["tasks"]),
            "array_hash_validation": "Frozen prepare_setting validates complete arrays inside the compute child"}


class ProgressOnly:
    """Allow structural progress, never dataset scores, into child stdout."""
    prefixes = ("TASK_START ", "CHECKPOINT_", "CELL_", "G_EPOCH_DONE ",
                "F_EPOCH_DONE ", "DERPP_EPOCH_", "HERDING_FEATURES ")

    def __init__(self, target):
        self.target, self.pending = target, ""

    def write(self, value):
        self.pending += value
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            if line.startswith(self.prefixes):
                self.target.write(line + "\n")
                self.target.flush()
        return len(value)

    def flush(self):
        self.target.flush()


def configuration(settings, method, smoke):
    from ncd import run_problem_scout as ps, run_aacil as aa
    from ncd import run_aacil_retention as retention
    from ncd.calibrate_aacil_baselines import CALIBRATION_PROTOCOL
    config = dict(settings)
    config.update(section31=True, section31_schema_version=1, smoke=smoke,
                  device="cpu" if smoke else "cuda", cpu_threads=2,
                  cpu_interop_threads=1, epochs_per_task=1 if smoke else 8,
                  min_steps_per_task=0 if smoke else 2000,
                  normal_training_rows=50000, owner_deadline=None)
    if method in BASELINE:
        config.update(section26=True, section26_schema_version=2,
                      section26_access_policy=ps.SECTION26_POLICY)
    else:
        config.update(section27=True, section27_schema_version=1,
                      section27_access_policy=aa.POLICY,
                      gate_step_policy="matched_s2000",
                      calibration_policy="permanent_normal_holdout_outside_m")
    if method == "replay_bs":
        config.update(section28=True, section28_method=method, post_hoc=True,
                      calibration_protocol=CALIBRATION_PROTOCOL, deadline=SENTINEL,
                      deadline_compatibility_note="Frozen Replay-BS API requires a timestamp; far-future sentinel disables the historical owner cutoff only. Slurm walltime/signals/telemetry remain active.")
    if method == "aacil_kd":
        config.update(section29=True, section29_schema_version=1,
                      section298=True, section298_schema_version=1,
                      lambda_g=.5, derpp_alpha=1., derpp_beta=1.,
                      derpp_allocation_adaptation=retention.ADAPTATION,
                      derpp_coefficient_source="Frozen KD reference: lambdaG=.5, alpha=beta=1; no CSE18 selection",
                      cbf_epochs=1, cbf_lr_multiplier=.1, deadline_iso=None,
                      selection_only=False,
                      selection_aggregation="not applicable to frozen CSE18 reference",
                      selection_validation_policy="Permanent Normal holdout plus lawful retained attack validation memory; evaluation only")
    return config


def calibrated_stages(data, result, folder, config, device, original_train_counts):
    """Use saved AA/BS CMs; calibrate baseline checkpoints using Normal only."""
    import numpy as np
    import torch
    from ncd.model import MLP
    from ncd import run_problem_scout as ps, run_replay_bs as bs
    from ncd.calibrate_aacil_baselines import select_taus, CALIBRATION_PROTOCOL
    from ncd.analyze_stagewise_forgetting import stage_metrics, aggregate_trajectory
    stages, costs = [], []
    identity = result["identity"]
    names = result["class_order"]
    _, validation = ps.section26_pools(data, identity["seed"])
    ids = validation[names.index("Normal")]
    # Smoke changes the sampled rows, not the prepared TRAIN prior. In formal
    # execution this is exactly the same count vector as the uncapped data.
    counts = np.asarray(original_train_counts, dtype=np.int64)
    if counts.shape != (len(names),) or np.any(counts <= 0):
        raise ValueError("Original prepared TRAIN prior has invalid class counts")
    destination = folder.parent / "calibrated_stages"
    destination.mkdir(exist_ok=True)
    for cp in result["checkpoints"]:
        target = destination / f"task_{cp['task_index']:03d}.json"
        bound = {"checkpoint_sha256": cp["checkpoint_sha256"],
                 "identity_sha256": ps.canonical_hash(identity), "alpha": .02}
        if target.exists():
            stage = read(target)
            if stage.get("binding") != bound or not stage.get("completed"):
                raise ValueError("Calibrated checkpoint cache changed")
            stages.append(stage)
            continue
        started = time.monotonic()
        if "0.02" in cp.get("operating_points", {}):
            cm = cp["operating_points"]["0.02"]["confusion_matrix"]
            calibration = cp.get("calibration", cp.get("normal_holdout_calibration"))
            mode = "saved_alpha2_confusion_no_new_forward"
            extra = 0.0
        else:
            seen = cp["seen_classes"]
            state = ps.checkpoint_load(folder / cp["checkpoint_file"], identity)
            model = MLP(data.x_train.shape[1], int(config["hidden"]), len(names)).to(device)
            model.load_state_dict(state["model_state"])
            prior = torch.from_numpy(np.log(counts/counts.sum()).astype(np.float32)[seen])
            holdout_logits = bs.logits_for(model, data.x_train, ids, seen, config, device)
            grid, selected, queries, gap = select_taus(holdout_logits, prior, seen.index(names.index("Normal")))
            point = selected["0.02"]
            rows = np.flatnonzero(np.isin(data.y_test, seen))
            logits = bs.logits_for(model, data.x_test, rows, seen, config, device)
            predicted = np.asarray(seen)[(logits + point["total_tau"] * prior).argmax(1).numpy()]
            n = len(names)
            cm = np.bincount(data.y_test[rows] * n + predicted, minlength=n*n).reshape(n, n)
            calibration = {"protocol": CALIBRATION_PROTOCOL, "operating_point": point,
                           "grid": grid, "search_evaluations": queries,
                           "minimum_normal_log_prior_gap": gap,
                           "calibration_rows": len(ids),
                           "calibration_indices_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                           "original_train_counts": counts.tolist(),
                           "prior_role": "Original complete prepared TRAIN labels before permanent holdout or smoke subsampling; no TEST role",
                           "outside_M": True, "never_optimized": True,
                           "feature_bytes": int(len(ids)*data.x_train.shape[1]*4),
                           "logical_label_bytes": int(len(ids)*8), "index_bytes": int(len(ids)*8),
                           "test_used_for_threshold_selection": False}
            extra = time.monotonic()-started
            mode = "checkpoint_only_normal_holdout_calibration_then_test_evaluation"
            del model, state, logits, holdout_logits
        stage = {"completed": True, "binding": bound, "task_index": cp["task_index"],
                 "mode": mode, "calibration": calibration,
                 "metrics": stage_metrics(cm, cp["seen_classes"], names),
                 "additional_calibration_evaluation_seconds": extra}
        write_json(target, stage)
        stages.append(stage)
        print(f"CHECKPOINT_CALIBRATED task={cp['task_index']} alpha=0.02", flush=True)
    costs = [s["additional_calibration_evaluation_seconds"] for s in stages]
    return stages, aggregate_trajectory(stages, result["tasks"], names), sum(costs)


def run_child(args):
    import numpy as np
    import sklearn
    import torch
    from ncd import run_problem_scout as ps, run_aacil as aa
    from ncd import run_replay_bs as bs, run_aacil_retention as retention
    from ncd.prepare_problem_scout_data import prepare_setting
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    dataset = args.smoke_dataset if args.local_smoke else DATASET
    method, seed = args.methods[0], args.seeds[0]
    folder = cell_root(args.output_root, method, seed)
    folder.mkdir(parents=True, exist_ok=True)
    settings_path = ROOT / "ncd/configs/problem_scout_section21.json"
    settings = read(settings_path)
    config = configuration(settings, method, args.local_smoke)
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Formal child requires CUDA")
    hashes = source_binding()
    start = time.monotonic()
    data, _, _, metadata = prepare_setting(dataset, config)
    original_train_counts = np.bincount(data.y_train, minlength=len(data.class_order))
    if args.local_smoke:
        data, _, _ = ps.smoke_subset(data, None, None)
    loading_seconds = time.monotonic()-start
    train, valid = ps.section26_pools(data, seed)
    provenance = {"git_commit": os.environ["NCD_SOURCE_COMMIT"],
                  "adapter_entrypoint": ENTRY, "effective_config": config,
                  "config_sha256": sha(settings_path),
                  "effective_config_sha256": ps.canonical_hash(config),
                  "source_hashes": hashes,
                  "frozen_imports": {path: {"historical_commit": commit, "sha256": digest}
                                     for path, (commit, digest) in FROZEN.items()},
                  "runtime_versions": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                                       "numpy": np.__version__, "sklearn": sklearn.__version__, "cuda": torch.version.cuda}}
    identity = provenance | {"dataset": dataset, "method": method, "seed": seed,
        "class_order": data.class_order, "tasks": data.tasks, "feature_names": data.feature_names,
        "data_binding": ps.binding(data, None, None), "data_metadata_sha256": ps.canonical_hash(metadata),
        "original_train_counts": original_train_counts.tolist(),
        "calibration_prior_role": "Original complete prepared TRAIN labels before permanent holdout or smoke subsampling; no TEST role",
        "section26_holdout": {str(c): {"train_count": len(train[c]), "validation_count": len(valid[c]),
            "validation_indices_sha256": hashlib.sha256(valid[c].tobytes()).hexdigest()} for c in valid}}
    if method in BASELINE:
        identity.update(regularization=0., section26_access_policy=ps.SECTION26_POLICY,
                        offline_joint_reference=False, lambda_file=None, lambda_file_sha256=None,
                        lambda_source_commit=None)
    else:
        identity["section27_access_policy"] = aa.POLICY
    if method == "aacil_kd":
        identity.update(section29_method=method, section298_schema_version=1,
                        effective_lambda_g=.5, family_type=retention.family_kind(method))
    if method == "replay_bs":
        identity["section28_access_policy"] = aa.POLICY
    raw = folder / "raw"
    raw.mkdir(exist_ok=True)
    with redirect_stdout(ProgressOnly(sys.stdout)):
        if method in BASELINE:
            result = ps.run_cell(data, None, None, metadata, config, identity, raw, device)
        elif method == "aacil":
            result = aa.run_cell(data, config, identity, raw, device)
        elif method == "aacil_kd":
            result = retention.run_cell(data, config, identity, raw, device)
        else:
            result = bs.run_cell(data, original_train_counts, config, identity, raw, device)
    completed = folder / "section31_result.json"
    if completed.exists():
        old = read(completed)
        if (not old.get("completed") or old.get("identity") != identity
                or old.get("raw_result_sha256") != sha(raw / "result.json")):
            raise ValueError("Completed Section31 cell provenance changed")
        for checkpoint in result["checkpoints"]:
            path = raw / checkpoint["checkpoint_file"]
            if sha(path) != checkpoint["checkpoint_sha256"] or read(path.with_suffix(".json")) != checkpoint:
                raise ValueError("Completed checkpoint or sidecar changed")
        print(f"CELL_SKIP method={method} seed={seed} metrics=sealed", flush=True)
        return
    stages, metrics, calibration_cost = calibrated_stages(data, result, raw, config, device, original_train_counts)
    output = {"completed": True, "completed_at": utc(), "schema_version": 1,
              "dataset": dataset, "method": method, "seed": seed, "smoke": args.local_smoke,
              "identity": identity, "raw_result": "raw/result.json", "raw_result_sha256": sha(raw / "result.json"),
              "actual_entrypoint": ENTRY, "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
              "checkpoint_resume": "Frozen run_cell validates and reuses completed task checkpoints; unsaved task work is repeated and retained in attempt logs",
              "owner_deadline": None, "metrics": metrics, "stages": stages,
              "cost_seconds": {key: value for key, value in result.items() if key.endswith("_seconds_total")},
              "adapter_cost_seconds": {"data_validation_loading_this_attempt": loading_seconds,
                                       "additional_calibration_evaluation_successful_stages": calibration_cost},
              "disclosure": "Sealed until all seven reference five-seed groups complete; no CSE18 tuning"}
    write_json(folder / "section31_result.json", output)
    print(f"CELL_COMPLETE method={method} seed={seed} checkpoints={len(stages)} metrics=sealed", flush=True)


def release_if_complete(output, smoke, methods):
    required_methods = tuple(methods) if smoke else METHODS
    required_seeds = (1,) if smoke else SEEDS
    paths = [completed_path(output, method, seed) for method in required_methods for seed in required_seeds]
    # No partial-group statistics: the whole reference set must be closed first.
    if not all(path.exists() for path in paths):
        return False
    prior = output / "reference_summary.json"
    if prior.exists():
        previous = read(prior)
        bindings = [{"path": path.relative_to(output).as_posix(), "sha256": sha(path)} for path in paths]
        if (not previous.get("completed") or bool(previous.get("smoke")) != smoke
                or previous.get("source_results") != bindings):
            raise ValueError("Released reference aggregate source changed")
        return True
    lock = output / ".aggregate.lock"
    try:
        stream = lock.open("x", encoding="utf-8")
    except FileExistsError:
        return False
    try:
        with stream:
            stream.write(json.dumps({"pid": os.getpid(), "utc": utc()}))
        rows = [read(path) for path in paths]
        common = rows[0]["identity"]
        for row, path in zip(rows, paths):
            if not row.get("completed") or bool(row.get("smoke")) != smoke:
                raise ValueError("Incomplete or incompatible reference cell cannot be released")
            if path != completed_path(output, row["method"], row["seed"]):
                raise ValueError("Reference identity does not match its assigned cell path")
            if not smoke and row.get("dataset") != DATASET:
                raise ValueError("Formal reference release must contain CSE18 only")
            if any(row["identity"].get(key) != common.get(key) for key in
                   ("git_commit", "source_hashes", "config_sha256", "data_binding",
                    "data_metadata_sha256", "class_order", "tasks", "original_train_counts")):
                raise ValueError("Reference release mixes source, data or task bindings")
            if sha(path.parent / row["raw_result"]) != row["raw_result_sha256"]:
                raise ValueError("Completed reference source changed")
        groups = []
        for method in required_methods:
            group = [row for row in rows if row["method"] == method]
            if sorted(row["seed"] for row in group) != sorted(required_seeds):
                raise ValueError("Reference group seed set differs")
            keys = ("average_incremental_accuracy", "final_balanced_accuracy", "forgetting_all_old_classes",
                    "final_attack_detection", "final_macro_f1", "final_normal_fpr")
            metrics = {}
            for key in keys:
                values = [row["metrics"][key] for row in group]
                metrics[key] = {"n": len(values), "mean": statistics.mean(values) if all(v is not None for v in values) else None,
                                "sample_sd": statistics.stdev(values) if len(values)>1 and all(v is not None for v in values) else None}
            groups.append({"method": method, "seeds": list(required_seeds), "metrics": metrics,
                           "cost_seconds": [row["cost_seconds"] for row in group]})
        if "replay" in required_methods:
            original = next(group for group in groups if group["method"] == "replay")
            groups.append(original | {"method": "replay_la", "derived_from": "replay",
                "cost_seconds": [], "training_cost_double_charged": False,
                "calibration_definition": "LA offset1 plus residual tau equals the same Normal-selected total tau; exact alpha2 prediction alias, no second training"})
        release = {"completed": True, "released_at": utc(), "smoke": smoke,
                   "release_rule": "All seven physical reference groups and all five seeds complete before any formal aggregate release",
                   "group_count": len(groups), "physical_cells": len(rows), "groups": groups,
                   "source_results": [{"path": path.relative_to(output).as_posix(), "sha256": sha(path)} for path in paths],
                   "test_selected": False, "primary_selected_candidate_included": False}
        write_json(output / "reference_summary.json", release)
        print(f"REFERENCE_AGGREGATE_RELEASED physical_cells={len(rows)} smoke={smoke}", flush=True)
        return True
    finally:
        lock.unlink(missing_ok=True)


def terminate(child):
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)


def protect_owned(output, protected, attempt, methods, seeds, copied):
    """Separate jobs copy only their disjoint cells/attempt, never peer temp files."""
    sources = [attempt] + [cell_root(output, method, seed) for method in methods for seed in seeds]
    for source in sources:
        if not source.exists():
            continue
        relative = source.relative_to(output)
        destination = protected / relative
        destination.mkdir(parents=True, exist_ok=True)
        copy_small(source, destination, copied.setdefault(relative.as_posix(), {}))
    summary = output / "reference_summary.json"
    if summary.exists():
        target = protected / summary.name
        temporary = target.with_suffix(f".json.{os.getpid()}.tmp")
        shutil.copyfile(summary, temporary)
        temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("code-root", "data-root", "output-root", "protected-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--resume", action="store_true", help="Resume only after the prior owned job is confirmed terminal; stale locks require operator reconciliation")
    parser.add_argument("--local-smoke", action="store_true")
    parser.add_argument("--smoke-dataset", choices=DATASET_KEYS, default=DATASET,
                        help="Only smoke may use an available NSL/UNSW full cache before its <=64/class subset")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("code_root", "data_root", "output_root", "protected_root"):
        setattr(args, name, getattr(args, name).resolve())
    if args.code_root != ROOT.resolve():
        parser.error("Code root must identify this exact immutable package")
    source = os.environ.get("NCD_SOURCE_COMMIT", "")
    if not re.fullmatch("[0-9a-f]{40}", source):
        parser.error("NCD_SOURCE_COMMIT must identify the new complete adapter/wrapper commit")
    if len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate method or seed assignment")
    if args.local_smoke:
        args.seeds = [1]
    settings = read(ROOT / "ncd/configs/problem_scout_section21.json")
    if (ROOT / settings["problem_scout_data_root"]).resolve() != args.data_root:
        parser.error("Stage the existing cache at the frozen relative config path; no config override")
    if (settings["batch_size"], settings["buffer_per_class"], settings["epochs_per_task"], settings["cpu_threads"]) != (256, 200, 8, 2):
        parser.error("Frozen runtime configuration changed")
    if under(args.output_root, args.protected_root) or under(args.protected_root, args.output_root):
        parser.error("Active scratch and protected small-output roots must be disjoint")
    if not args.local_smoke:
        user = os.environ.get("USER", "")
        scratch = Path("/scr/user") / user
        if not user or not os.environ.get("SLURM_JOB_ID", "").isdigit():
            parser.error("Formal execution requires the owner's Slurm allocation")
        if not all(under(path, Path.home()) for path in (args.code_root, args.protected_root)):
            parser.error("Code and protected small evidence must be in HOME")
        if not all(under(path, scratch) for path in (args.data_root, args.output_root)):
            parser.error("Data and active results must be in the owner's scratch")
        if (os.environ.get("SLURM_NNODES"), os.environ.get("SLURM_NTASKS", "1"), os.environ.get("SLURM_CPUS_PER_TASK")) != ("1", "1", "2"):
            parser.error("One node, one task and two CPUs are required")
    if args.child:
        if len(args.methods) != 1 or len(args.seeds) != 1:
            parser.error("A child owns exactly one physical cell")
        if os.environ.get("NCD_SECTION31_PARENT_PID") != str(os.getppid()):
            parser.error("Child mode must be launched by this guarded supervisor")
        run_child(args)
        return
    cache = check_cache(args.data_root, args.smoke_dataset if args.local_smoke else DATASET)
    identifiers = []
    if not args.local_smoke:
        step = os.environ.get("SLURM_STEP_ID", os.environ.get("SLURM_STEPID", ""))
        if not step.isdigit() or not os.environ.get("CUDA_VISIBLE_DEVICES"):
            parser.error("Launch in a single srun step with the allocated CUDA device")
        identifiers = visible_gpu_uuids()
        if len(identifiers) != 1 or not gpu_sample(identifiers)["gpu_telemetry_valid"]:
            parser.error("Exactly one allocated CUDA UUID and valid UUID-targeted telemetry are required")
    output, protected = args.output_root, args.protected_root
    output.mkdir(parents=True, exist_ok=True)
    protected.mkdir(parents=True, exist_ok=True)
    attempt = output / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"_{os.environ.get('SLURM_JOB_ID', 'local')}_{os.getpid()}")
    attempt.mkdir(parents=True)
    record = {"started_at": utc(), "job_id": os.environ.get("SLURM_JOB_ID"), "pid": os.getpid(),
              "adapter_source_commit": source, "methods": args.methods, "seeds": args.seeds,
              "smoke": args.local_smoke, "cache": cache, "owner_deadline": None,
              "resource_scope": "One GPU child serially; account-wide quota and disjoint job assignments checked by submitter",
              "resume": args.resume, "cells": [], "status": "running", "gpu_uuids": identifiers}
    write_json(attempt / "job_start.json", record)
    received, child, lock, copied = [], None, None, {}

    def on_signal(number, _frame):
        received.append(number)
        if child is not None and child.poll() is None:
            child.terminate()

    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, on_signal)
    try:
        for method in args.methods:
            for seed in args.seeds:
                if received:
                    raise RuntimeError(f"Received signal {received[-1]}; preserved checkpoints")
                target = cell_root(output, method, seed)
                if target.exists() and any(target.iterdir()) and not args.resume:
                    raise ValueError("Existing cell requires explicit --resume; never overwrite a prior attempt")
                locks = output / ".cell_locks"
                locks.mkdir(exist_ok=True)
                candidate_lock = locks / f"{method}_seed{seed}.lock"
                with candidate_lock.open("x", encoding="utf-8") as stream:
                    json.dump({"job_id": record["job_id"], "pid": os.getpid(), "utc": utc()}, stream)
                lock = candidate_lock
                command = [sys.executable, "-I", "-B", "-u", ENTRY, "--child",
                           "--code-root", str(args.code_root), "--data-root", str(args.data_root),
                           "--output-root", str(output), "--protected-root", str(protected),
                           "--methods", method, "--seeds", str(seed)]
                if args.local_smoke:
                    command.extend(["--local-smoke", "--smoke-dataset", args.smoke_dataset])
                current = {"method": method, "seed": seed, "started_at": utc(), "command": command}
                record["current_cell"] = current
                write_json(attempt / "status.json", record)
                prefix = attempt / f"{method}_seed{seed}"
                with prefix.with_suffix(".stdout.log").open("x", encoding="utf-8") as stdout, prefix.with_suffix(".stderr.log").open("x", encoding="utf-8") as stderr:
                    environment = dict(os.environ, NCD_SECTION31_PARENT_PID=str(os.getpid()))
                    child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
                    current["pid"] = child.pid
                    write_json(attempt / "status.json", record)
                    started, next_sample, guard = time.monotonic(), 0., TelemetryGuard()
                    while child.poll() is None:
                        elapsed = time.monotonic()-started
                        if received:
                            raise RuntimeError(f"Received signal {received[-1]}; preserved checkpoints")
                        if not args.local_smoke and elapsed >= next_sample:
                            measurement = sample(child.pid, output, identifiers)
                            reason = guard.observe(measurement, elapsed)
                            measurement.update(child_elapsed_seconds=elapsed, method=method, seed=seed, guard_trigger=reason)
                            with (attempt / "resources.jsonl").open("a", encoding="utf-8") as stream:
                                stream.write(json.dumps(measurement, allow_nan=False) + "\n")
                            next_sample = elapsed + 60
                            if reason:
                                raise RuntimeError(reason)
                        protect_owned(output, protected, attempt, args.methods, args.seeds, copied)
                        try:
                            child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            pass
                    current.update(exit_code=child.returncode, ended_at=utc(), elapsed_seconds=time.monotonic()-started)
                    if child.returncode != 0 or not completed_path(output, method, seed).exists():
                        raise RuntimeError(f"Cell {method}/{seed} did not complete; inspect preserved stderr before any retry")
                record["cells"].append(current)
                record["current_cell"] = None
                write_json(attempt / "status.json", record)
                lock.unlink()
                lock = None
                print(f"CELL_CLOSED method={method} seed={seed} metrics=sealed", flush=True)
                release_if_complete(output, args.local_smoke, args.methods)
        record.update(status="complete", ended_at=utc())
    except BaseException as error:
        terminate(child)
        if record.get("current_cell") is not None and child is not None:
            record["current_cell"].update(exit_code=child.returncode, ended_at=utc(),
                                          elapsed_seconds=time.monotonic()-started)
        record.update(status="interrupted" if received else "failed", ended_at=utc(),
                      error_type=type(error).__name__, error=str(error), received_signals=received)
        raise
    finally:
        terminate(child)
        write_json(attempt / "job_exit.json", record)
        protect_owned(output, protected, attempt, args.methods, args.seeds, copied)
        if lock is not None:
            lock.unlink(missing_ok=True)
    print(f"ASSIGNMENT_COMPLETE physical_cells={len(record['cells'])} smoke={args.local_smoke}", flush=True)


if __name__ == "__main__":
    main()
