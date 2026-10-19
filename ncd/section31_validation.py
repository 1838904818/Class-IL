"""Validation-only external references for the compact-family development study.

The four physical learners use the unchanged Section26 training/memory routine.
Evaluation is replaced by fixed, seed-specific TRAIN holdouts; TEST features are
never forwarded. Replay+LA is an explicit calibrated-total-tau alias of Replay,
not a fifth physical training run. This entry does not launch other processes.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import sklearn
import torch
from ncd import run_problem_scout as base
from ncd.calibrate_aacil_baselines import select_taus
from ncd.run_aacil import SOURCE_PATHS as AA_SOURCES, row_storage
from ncd.run_aacil_retention import stagewise_metrics
from ncd.source_commit import source_commit
from ncd.train_cil import Tee

DATASETS = ("UNSW-NB15", "NSL-KDD")
METHODS = ("replay", "derpp", "wacb", "freq")
DEV_SEEDS = (5, 6)
SOURCE_PATHS = tuple(dict.fromkeys((*AA_SOURCES,
    "ncd/run_aacil_retention.py", "ncd/aacil_iso.py", "ncd/aacil_retention.py",
    "ncd/calibrate_aacil_baselines.py", "ncd/analyze_farq_operating_curves.py",
    "ncd/section31_validation.py")))
POLICY = ("Fixed permanent stratified 10% attack TRAIN holdout plus the permanent "
          "Normal TRAIN calibration pool; seen classes only; evaluation only, "
          "outside replay M; no TEST forward or test-based selection")


def now():
    return datetime.now().astimezone().isoformat()


def calibrated_validation_evaluator(data, valid, config, prior_counts=None):
    """Return a run_problem_scout.evaluate-compatible validation-only callback.

    The callback checks object identity so passing a TEST view cannot silently
    become validation. Its one-entry cache makes Replay+LA a cost-free alias.
    """
    normal = data.class_order.index("Normal")
    normal_rows = valid[normal]
    calibration_x = data.x_train[normal_rows]
    batch = int(config["eval_batch_size"])
    counts = np.bincount(data.y_train, minlength=len(data.class_order))
    prior_counts = counts if prior_counts is None else np.asarray(prior_counts)
    full_prior = torch.tensor(np.log(prior_counts / prior_counts.sum()), dtype=torch.float32)
    cache = {}

    def forward(scorer, features, seen, device):
        parts = []
        scorer.eval()
        with torch.no_grad():
            for start in range(0, len(features), batch):
                logits = scorer(torch.from_numpy(np.asarray(features[start:start + batch])).to(device))
                if not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite validation/calibration logits")
                parts.append(logits[:, list(seen)].detach().cpu())
        return torch.cat(parts)

    def evaluate(scorer, x, y, seen, current_data, current_config, device, log_prior=None):
        if x is None and y is None:
            return None
        if current_data is not data or x is not data.x_test or y is not data.y_test:
            raise ValueError("Only the substituted fixed TRAIN-validation view may be evaluated")
        if cache.get("scorer") is scorer and cache.get("seen") == tuple(seen):
            result = copy.deepcopy(cache["result"])
            result["replay_la_calibrated_total_tau_alias"] = log_prior is not None
            return result
        if log_prior is not None:
            raise ValueError("Replay+LA must reuse an already calibrated Replay point")
        local_normal = list(seen).index(normal)
        prior = full_prior[list(seen)]
        started = time.perf_counter()
        normal_logits = forward(scorer, calibration_x, seen, device)
        normal_forward_seconds = time.perf_counter() - started
        started = time.perf_counter()
        grid, selected, probes, gap = select_taus(normal_logits, prior, local_normal)
        calibration_seconds = time.perf_counter() - started
        point = selected["0.02"]
        rows = np.flatnonzero(np.isin(y, seen))
        started = time.perf_counter()
        logits = forward(scorer, x[rows], seen, device)
        prediction = np.asarray(seen)[(logits + point["total_tau"] * prior).argmax(1).numpy()]
        n = len(data.class_order)
        cm = np.bincount(y[rows] * n + prediction, minlength=n * n).reshape(n, n)
        result = base.metrics_from_confusion(cm, seen, data.class_order, data.tasks, counts)
        validation_seconds = time.perf_counter() - started
        seen_attack_rows = sum(len(valid[c]) for c in seen if c != normal)
        storage = row_storage(seen_attack_rows, data.x_train.shape[1])
        storage.update(outside_replay_M=True, never_optimized=True,
                       includes_rows_also_retained_as_validation=True,
                       storage_scope="Full fixed attack validation pool; overlap with retained VAL is not independent storage")
        result.update(total_available_rows=len(y), excluded_unseen_rows=len(y)-len(rows),
            primary_alpha=.02, evaluation_role="fixed_train_validation_only", test_evaluation_performed=False,
            calibration={"point": point, "grid": grid, "search_evaluations": probes,
                "normal_indices_sha256": hashlib.sha256(normal_rows.tobytes()).hexdigest(),
                "minimum_normal_log_prior_gap": gap,
                "pool_storage": row_storage(len(normal_rows), data.x_train.shape[1]),
                "pool_outside_M": True, "pool_never_optimized": True},
            selection_attack_holdout_storage=storage,
            normal_forward_seconds=normal_forward_seconds, calibration_seconds=calibration_seconds,
            validation_seconds=validation_seconds, replay_la_calibrated_total_tau_alias=False)
        cache.clear()
        cache.update(scorer=scorer, seen=tuple(seen), result=copy.deepcopy(result))
        return result

    return evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=DEV_SEEDS, default=list(DEV_SEEDS))
    parser.add_argument("--config", type=Path, default=ROOT/"ncd/configs/problem_scout_section21.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    os.environ["NCD_SOURCE_COMMIT"] = args.source_commit
    commit = source_commit(ROOT)
    if any(len(values) != len(set(values)) for values in (args.datasets, args.methods, args.seeds)):
        parser.error("Duplicate dataset/method/seed selections are forbidden")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if (int(config["buffer_per_class"]), int(config["batch_size"]), int(config["epochs_per_task"])) != (200, 256, 8):
        parser.error("External references require the existing 200/class, batch256, eight-epoch configuration")
    config.update(section26=True, section26_schema_version=2,
        section26_access_policy=base.SECTION26_POLICY, section31_development=True,
        selection_only=True, selection_validation_policy=POLICY, primary_alpha=.02,
        device=args.device, min_steps_per_task=2000, smoke=args.smoke)
    if args.smoke:
        config.update(epochs_per_task=1, min_steps_per_task=0)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance = {"git_commit": commit, "effective_config": config,
        "config_sha256": base.digest(args.config), "effective_config_sha256": base.canonical_hash(config),
        "source_hashes": {path: base.digest(ROOT/path) for path in SOURCE_PATHS},
        "runtime_versions": {"python": sys.version.split()[0], "torch": str(torch.__version__),
            "numpy": np.__version__, "sklearn": sklearn.__version__, "cuda": torch.version.cuda}}
    expected = {"datasets": args.datasets, "methods": args.methods, "seeds": args.seeds}
    matrix_path = output/"result.json"
    if matrix_path.exists():
        existing = json.loads(matrix_path.read_text(encoding="utf-8"))
        if existing["provenance"] != provenance or existing["expected"] != expected:
            raise ValueError("Validation matrix source/config/selection changed; preserve this root")
    snapshot = {"schema_version": 1, "section31_development": True, "selection_only": True,
        "test_evaluation_performed": False, "execution_status": "in_progress", "smoke": args.smoke,
        "provenance": provenance, "expected": expected,
        "expected_cells": len(args.datasets)*len(args.methods)*len(args.seeds), "cells": [],
        "started_at": now(), "current_cell": None,
        "command": subprocess.list2cmdline(["python", "-I", "ncd/section31_validation.py", *sys.argv[1:]]),
        "log_path": base.path_label(output/"run.log"), "validation_policy": POLICY,
        "logical_external_methods": [*METHODS, "replay_la"],
        "replay_la": "Exact calibrated-total-tau Replay alias; no extra training or forward cost"}
    original_evaluate, original_save = base.evaluate, base.save_json
    try:
        with (output/"run.log").open("a", encoding="utf-8") as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            from ncd.prepare_problem_scout_data import prepare_setting
            for dataset in args.datasets:
                raw, _, _, metadata = prepare_setting(dataset, config)
                original_counts = np.bincount(raw.y_train, minlength=len(raw.class_order))
                if args.smoke:
                    raw, _, _ = base.smoke_subset(raw, None, None)
                for method in args.methods:
                    for seed in args.seeds:
                        train, valid = base.section26_pools(raw, seed)
                        ids = np.concatenate([valid[c] for c in sorted(valid)])
                        # The base routine's conventional x_test field now contains TRAIN holdout rows only.
                        data = replace(raw, x_test=raw.x_train[ids].copy(), y_test=raw.y_train[ids].copy())
                        identity = provenance | {"dataset": dataset, "method": method, "seed": seed,
                            "class_order": data.class_order, "tasks": data.tasks, "feature_names": data.feature_names,
                            "data_binding": base.binding(data, None, None),
                            "data_binding_x_test_role": "substituted fixed TRAIN validation; not original TEST",
                            "calibration_prior_counts": original_counts.tolist(),
                            "calibration_prior_role": "Original prepared TRAIN frequencies; smoke retains these because equal per-class truncation would remove the strictly-largest Normal prior",
                            "data_metadata_sha256": base.canonical_hash(metadata), "regularization": 0.,
                            "lambda_file": None, "lambda_file_sha256": None, "lambda_source_commit": None,
                            "section26_access_policy": base.SECTION26_POLICY, "selection_only": True,
                            "section26_holdout": {str(c): {"train_count": len(train[c]), "validation_count": len(valid[c]),
                                "validation_indices_sha256": hashlib.sha256(valid[c].tobytes()).hexdigest()} for c in valid}}
                        base.evaluate = calibrated_validation_evaluator(data, valid, config, original_counts)

                        def save_with_validation_semantics(path, value):
                            if isinstance(value, dict) and "id" in value and "task_index" in value:
                                allocation = value["allocation"]
                                normal = data.class_order.index("Normal")
                                retained = allocation["actual_validation_counts"]
                                attack_overlap = sum(int(n) for c, n in retained.items() if int(c) != normal)
                                normal_overlap = int(retained[str(normal)])
                                attack_rows = sum(len(valid[c]) for c in value["seen_classes"] if c != normal)
                                value["id"]["selection_attack_holdout_storage"].update(
                                    overlap_with_retained_validation_rows=attack_overlap,
                                    additional_unique_rows_outside_M=attack_rows-attack_overlap,
                                    additional_unique_storage=row_storage(attack_rows-attack_overlap, data.x_train.shape[1]))
                                value["id"]["calibration"].update(
                                    overlap_with_retained_normal_validation_rows=normal_overlap,
                                    additional_unique_rows_outside_M=len(valid[normal])-normal_overlap,
                                    additional_unique_storage=row_storage(len(valid[normal])-normal_overlap, data.x_train.shape[1]))
                                value.update(validation_operating_points={"0.02": value["id"]},
                                    test_evaluation_performed=False, evaluation_role="fixed_train_validation_only")
                                if "la_id" in value:
                                    value["la_id"] = copy.deepcopy(value["id"])
                                    value["la_id"]["replay_la_calibrated_total_tau_alias"] = True
                                    value["la_definition"] = "Replay+LA calibrated total tau aliases Replay; no extra forward/training"
                            if isinstance(value, dict) and value.get("completed") and value.get("identity") == identity:
                                value.update(selection_only=True, test_evaluation_performed=False,
                                    selection_metrics=stagewise_metrics([r["id"] for r in value["checkpoints"]], data.tasks),
                                    validation_policy=POLICY, evaluation_role="fixed_train_validation_only",
                                    command=snapshot["command"], log_path=snapshot["log_path"],
                                    replay_la_alias=method == "replay")
                                for key in ("normal_forward_seconds", "calibration_seconds", "validation_seconds"):
                                    value[key+"_total"] = sum(r["id"][key] for r in value["checkpoints"])
                            return original_save(path, value)

                        base.save_json = save_with_validation_semantics
                        cell_dir = output/dataset.lower().replace("-", "_")/f"{method}_seed{seed}"
                        snapshot["current_cell"] = {"dataset": dataset, "method": method, "seed": seed, "started_at": now()}
                        original_save(matrix_path, snapshot)
                        result = base.run_cell(data, None, None, metadata, config, identity, cell_dir, device, args.evaluate_only)
                        if result.get("test_evaluation_performed") is not False or not result.get("selection_only"):
                            raise ValueError("Completed external reference is not validation-only")
                        snapshot["cells"].append({"dataset": dataset, "method": method, "seed": seed,
                            "result": (cell_dir/"result.json").relative_to(output).as_posix(),
                            "result_sha256": base.digest(cell_dir/"result.json"),
                            "selection_metrics": result["selection_metrics"], "training_seconds_total": result["training_seconds_total"]})
                        snapshot["current_cell"] = None
                        if len(snapshot["cells"]) == snapshot["expected_cells"]:
                            snapshot.update(execution_status="complete", completed_at=now())
                        original_save(matrix_path, snapshot)
                        del data
                del raw
            print(f"MATRIX_COMPLETE cells={len(snapshot['cells'])}/{snapshot['expected_cells']} selection_only=True smoke={args.smoke}", flush=True)
    finally:
        base.evaluate, base.save_json = original_evaluate, original_save


if __name__ == "__main__":
    main()
