"""Section28 post-hoc Replay-BS: one classifier with the AA-CIL G input rows.

Reuses frozen data, memory and training helpers; it does not modify AA-CIL.
Threshold selection uses only permanent Normal TRAIN holdout predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from ncd.model import MLP
from ncd.run_aacil import (
    SOURCE_PATHS as AA_SOURCES, DATASETS, SEEDS, POLICY, now, sync,
    select_rows, row_storage, attack_targets, refresh_attack_memory,
    memory_to_tensors, memory_from_tensors, train_model,
)
from ncd.run_problem_scout import (
    binding, canonical_hash, checkpoint_load, checkpoint_save, digest,
    metrics_from_confusion, path_label, require_finite, save_json,
    section26_pools, smoke_subset,
)
from ncd.calibrate_aacil_baselines import select_taus, CALIBRATION_PROTOCOL
from ncd.source_commit import source_commit
from ncd.train_cil import Tee, seed_everything

SOURCES = (*AA_SOURCES, "ncd/run_replay_bs.py", "ncd/calibrate_aacil_baselines.py",
           "ncd/analyze_farq_operating_curves.py")


class DeadlineReached(RuntimeError):
    pass


def command():
    return subprocess.list2cmdline(["python", "-I", "ncd/run_replay_bs.py", *sys.argv[1:]])


def logits_for(model, x, rows, seen, config, device):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(rows), int(config["eval_batch_size"])):
            xx = torch.from_numpy(np.asarray(x[rows[start:start+int(config["eval_batch_size"])]] )).to(device)
            logits = model(xx)[:, seen]
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite Replay-BS logits")
            parts.append(logits.cpu())
    return torch.cat(parts)


def evaluate(model, data, seen, calibration_ids, original_counts, config, device):
    normal = data.class_order.index("Normal")
    # Original prepared TRAIN labels, including the heldout fraction, as in LA.
    prior = torch.from_numpy(np.log(original_counts / original_counts.sum()).astype(np.float32)[seen])
    sync(device)
    start = time.perf_counter()
    calibration_logits = logits_for(model, data.x_train, calibration_ids, seen, config, device)
    grid, selected, queries, gap = select_taus(calibration_logits, prior, seen.index(normal))
    calibration_seconds = time.perf_counter() - start
    start = time.perf_counter()
    rows = np.flatnonzero(np.isin(data.y_test, seen))
    test_logits = logits_for(model, data.x_test, rows, seen, config, device)
    points = {}
    n = len(data.class_order)
    for key, point in selected.items():
        pred = np.asarray(seen)[(test_logits + point["total_tau"] * prior).argmax(1).numpy()]
        cm = np.bincount(data.y_test[rows] * n + pred, minlength=n*n).reshape(n, n)
        value = metrics_from_confusion(cm, seen, data.class_order, data.tasks, original_counts)
        eligible = [r for r in value["per_class"] if r["class_id"] != normal and r["support"] >= 20]
        value.update(total_available_rows=len(data.y_test), excluded_unseen_rows=len(data.y_test)-len(rows),
                     attack_family_macro_recall_support_ge20=float(np.mean([r["recall"] for r in eligible])) if eligible else None,
                     attack_guard_excluded_classes=[r["class_id"] for r in value["per_class"] if r["class_id"] != normal and r["support"] < 20])
        points[key] = value
    sync(device)
    return ({"protocol": CALIBRATION_PROTOCOL, "grid": grid, "search_evaluations": queries,
             "operating_points": selected, "seen_class_columns": seen,
             "minimum_normal_log_prior_gap": gap, "original_train_counts": original_counts.tolist(),
             "prior_role": "Original prepared TRAIN labels before permanent holdout or smoke subset; no TEST role",
             "calibration_rows": len(calibration_ids),
             "calibration_indices_sha256": hashlib.sha256(calibration_ids.tobytes()).hexdigest()},
            points, calibration_seconds, time.perf_counter()-start)


def run_cell(data, original_counts, config, identity, output, device):
    final = output / "result.json"
    if final.exists():
        result = json.loads(final.read_bytes())
        if not result.get("completed") or result["identity"] != identity:
            raise ValueError("Existing Replay-BS cell identity or completion differs")
        print(f"CELL_SKIP {identity['dataset']} seed={identity['seed']}", flush=True)
        return result
    seed = identity["seed"]
    normal = data.class_order.index("Normal")
    train, valid = section26_pools(data, seed)
    calibration_ids = valid[normal].copy()
    memory, records = {}, []
    seed_everything(seed * 10000)
    model = MLP(data.x_train.shape[1], int(config["hidden"]), len(data.class_order)).to(device)
    for task_index, task in enumerate(data.tasks):
        if datetime.now().astimezone() >= datetime.fromisoformat(config["deadline"]):
            raise DeadlineReached("Fixed Section28 deadline reached before next task; unfinished cell retained")
        seen = [c for part in data.tasks[:task_index+1] for c in part]
        new = [c for c in task if c != normal]
        checkpoint = output / "checkpoints" / f"checkpoint_{task_index:03d}.pt"
        sidecar = checkpoint.with_suffix(".json")
        task_seed = seed * 10000 + task_index
        seed_everything(task_seed)
        if checkpoint.exists():
            state = checkpoint_load(checkpoint, identity)
            model.load_state_dict(state["model_state"])
            memory = memory_from_tensors(state["attack_memory"])
            if state["task_index"] != task_index or state["seen_classes"] != seen:
                raise ValueError("Replay-BS checkpoint task differs")
            print(f"CHECKPOINT_RESUME {identity['dataset']} seed={seed} task={task_index}", flush=True)
        else:
            start = time.perf_counter()
            rng = task_seed * 10007 + normal
            normal_rows = select_rows(train[normal], 50000, rng)
            if np.intersect1d(normal_rows, calibration_ids).size:
                raise AssertionError("Permanent Normal holdout entered optimization")
            xx = [data.x_train[normal_rows].copy()]
            yy = [np.full(len(normal_rows), normal, dtype=np.int64)]
            used = {}
            for c in seen:
                if c == normal:
                    continue
                ids, values = (train[c], data.x_train[train[c]].copy()) if c in new else (memory[c]["train_ids"], memory[c]["train_x"])
                used[str(c)] = ids.copy()
                xx.append(values)
                yy.append(np.full(len(ids), c, dtype=np.int64))
            training_x, training_y = np.concatenate(xx), np.concatenate(yy)
            selection_seconds = time.perf_counter() - start
            print(f"TASK_START {identity['dataset']} replay_bs seed={seed} task={task_index} rows={len(training_y)}", flush=True)
            seconds, schedule = train_model(model, training_x, training_y, config, task_seed, device)
            del training_x, training_y, xx, yy
            start = time.perf_counter()
            targets = attack_targets("aacil", seen, normal)
            memory = refresh_attack_memory(data, train, valid, memory, new, targets, seed, task_index)
            selection_seconds += time.perf_counter() - start
            nt = {str(c): len(v["train_ids"]) for c, v in memory.items()}
            nv = {str(c): len(v["validation_ids"]) for c, v in memory.items()}
            actual = sum(nt.values()) + sum(nv.values())
            storage = row_storage(actual, data.x_train.shape[1])
            normal_storage = row_storage(len(normal_rows), data.x_train.shape[1])
            normal_storage.update(outside_replay_M=True, refreshed_each_task=True, calibration_overlap_rows=0)
            calibration_storage = row_storage(len(calibration_ids), data.x_train.shape[1])
            calibration_storage.update(outside_replay_M=True, shared_across_methods=True, never_optimized=True,
                                       overlap_with_retained_TS_normal_validation_rows=0,
                                       additional_unique_rows_outside_M=len(calibration_ids))
            allocation = {"policy": "replay_bs_all_attack_memory", "capacity_total": 200*len(seen),
                          "actual_total": actual, "unfilled_capacity": 200*len(seen)-actual,
                          "attack_target_counts": {str(c): q for c, q in targets.items()},
                          "actual_attack_training_counts": nt, "actual_attack_validation_counts": nv,
                          "normal_refreshed": True, "normal_refresh_rng_seed": rng,
                          "normal_training_count": len(normal_rows), "normal_training_outside_replay_memory": True,
                          "old_attack_training_pool_access": False, "old_attack_validation_pool_access": False,
                          "replay_memory_storage": storage, "normal_training_storage": normal_storage,
                          "attack_memory_counted_copies": 1}
            state = {"identity": identity, "task_index": task_index, "seen_classes": seen, "new_classes": list(task),
                     "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                     "attack_memory": memory_to_tensors(memory),
                     "normal_training_row_ids": torch.from_numpy(normal_rows.copy()),
                     "calibration_row_ids": torch.from_numpy(calibration_ids.copy()),
                     "optimization_attack_row_ids": {c: torch.from_numpy(v.copy()) for c, v in used.items()},
                     "optimization_attack_counts": {c: len(v) for c, v in used.items()},
                     "training_seconds": seconds, "training_schedule": schedule,
                     "buffer_selection_seconds": selection_seconds, "allocation": allocation,
                     "calibration_pool_storage": calibration_storage,
                     "input_matching": "Exactly AA-CIL G input rows and order at the same dataset/seed/task; multiclass labels instead of binary labels"}
            checkpoint_save(checkpoint, state)
            print(f"CHECKPOINT_TRAINED {identity['dataset']} seed={seed} task={task_index} seconds={seconds:.3f}", flush=True)
        if sidecar.exists():
            record = json.loads(sidecar.read_bytes())
            if record["identity_hash"] != canonical_hash(identity):
                raise ValueError("Replay-BS sidecar identity differs")
        else:
            calibration, points, cal_seconds, eval_seconds = evaluate(model, data, seen, calibration_ids, original_counts, config, device)
            record = {k: v for k, v in state.items() if k not in {
                "identity", "model_state", "attack_memory", "normal_training_row_ids", "calibration_row_ids", "optimization_attack_row_ids"}}
            record.update(checkpoint_file=checkpoint.relative_to(output).as_posix(), checkpoint_sha256=digest(checkpoint),
                          identity_hash=canonical_hash(identity), normal_holdout_calibration=calibration,
                          operating_points=points, id=points["0.02"], primary_alpha=0.02,
                          calibration_seconds=cal_seconds, evaluation_seconds=eval_seconds)
            save_json(sidecar, record)
        records.append(record)
        print(f"CHECKPOINT_COMPLETE {identity['dataset']} seed={seed} task={task_index}", flush=True)
    result = {"schema_version": 1, "completed": True, "completed_at": now(), "post_hoc": True,
              "identity": identity, "class_order": data.class_order, "tasks": data.tasks,
              "checkpoints": records, "command": command(), "log_path": path_label(output / "run.log"),
              "train_counts": {str(c): int(v) for c, v in enumerate(original_counts)},
              "cost_scope": "Successful optimization/selection/calibration/evaluation components; loading excluded",
              "limitations": ["Post-hoc Section27.10/28 control; does not replace the original seven-baseline registered verdict.",
                              "One multiclass network, same AA-CIL G input rows; not compute-matched to G+F combined.",
                              "Calibrated alpha is not a guarantee of equal test FPR; permanent Normal holdout and benign stream storage are outside M."]}
    for name in ("training_seconds", "buffer_selection_seconds", "calibration_seconds", "evaluation_seconds"):
        result[name+"_total"] = sum(r[name] for r in records)
    require_finite(result)
    save_json(final, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=["NSL-KDD", "UNSW-NB15"])
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--config", type=Path, default=ROOT/"ncd/configs/problem_scout_section21.json")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--deadline", default="2026-10-19T21:50:26+08:00")
    args = parser.parse_args()
    config = json.loads(args.config.read_bytes())
    config.update(device=args.device, section28=True, section28_method="replay_bs", post_hoc=True,
                  epochs_per_task=1 if args.smoke else 8, min_steps_per_task=0 if args.smoke else 2000,
                  batch_size=256, buffer_per_class=200, normal_training_rows=50000, smoke=args.smoke,
                  calibration_policy="permanent_normal_holdout_outside_m", calibration_protocol=CALIBRATION_PROTOCOL,
                  deadline=args.deadline)
    if datetime.fromisoformat(args.deadline).tzinfo is None:
        parser.error("--deadline requires an explicit timezone")
    if args.data_root:
        config["problem_scout_data_root"] = args.data_root
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seeds = [1] if args.smoke else args.seeds
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance = {"git_commit": source_commit(ROOT), "entrypoint": "ncd/run_replay_bs.py",
                  "source_state": "new_independent_entrypoint_bound_by_source_hashes; git_commit_is_base_revision",
                  "effective_config": config, "effective_config_sha256": canonical_hash(config),
                  "config_sha256": digest(args.config), "source_hashes": {f: digest(ROOT/f) for f in SOURCES},
                  "runtime_versions": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": str(torch.__version__), "cuda": torch.version.cuda}}
    expected = {"datasets": args.datasets, "methods": ["replay_bs"], "seeds": seeds}
    result_path = output / "result.json"
    if result_path.exists():
        previous = json.loads(result_path.read_bytes())
        if previous["provenance"] != provenance or previous["expected"] != expected:
            raise ValueError("Replay-BS matrix changed; preserve old output and use a new path")
    snapshot = {"schema_version": 1, "post_hoc": True, "smoke": args.smoke, "provenance": provenance,
                "execution_status": "in_progress", "expected": expected, "expected_cells": len(args.datasets)*len(seeds),
                "current_cell": None, "cells": [], "command": command(), "log_path": path_label(output/"run.log")}
    with (output/"run.log").open("a", encoding="utf-8") as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        try:
            from ncd.prepare_problem_scout_data import prepare_setting
            for dataset in args.datasets:
                data, _, _, metadata = prepare_setting(dataset, config)
                counts = np.bincount(data.y_train, minlength=len(data.class_order))
                if args.smoke:
                    data, _, _ = smoke_subset(data, None, None)
                data_binding = binding(data, None, None)
                for seed in seeds:
                    train, valid = section26_pools(data, seed)
                    identity = provenance | {"dataset": dataset, "method": "replay_bs", "seed": seed,
                        "class_order": data.class_order, "tasks": data.tasks, "feature_names": data.feature_names,
                        "section28_access_policy": POLICY, "data_binding": data_binding,
                        "data_metadata_sha256": canonical_hash(metadata), "original_train_counts": counts.tolist(),
                        "section26_holdout": {str(c): {"train_count": len(train[c]), "validation_count": len(valid[c]),
                            "validation_indices_sha256": hashlib.sha256(valid[c].tobytes()).hexdigest()} for c in valid}}
                    cell = output/dataset.lower().replace("-", "_")/f"replay_bs_seed{seed}"
                    cell.mkdir(parents=True, exist_ok=True)
                    snapshot["current_cell"] = {"dataset": dataset, "seed": seed, "started_at": now()}
                    save_json(result_path, snapshot)
                    with (cell/"run.log").open("a", encoding="utf-8") as cell_log, redirect_stdout(Tee(sys.stdout, cell_log)), redirect_stderr(Tee(sys.stderr, cell_log)):
                        result = run_cell(data, counts, config, identity, cell, device)
                    snapshot["cells"].append({"dataset": dataset, "method": "replay_bs", "seed": seed,
                        "result": (cell/"result.json").relative_to(output).as_posix(), "result_sha256": digest(cell/"result.json"),
                        "completed_at": result["completed_at"], "training_seconds_total": result["training_seconds_total"]})
                    snapshot["current_cell"] = None
                    save_json(result_path, snapshot)
                del data
            snapshot.update(execution_status="complete", completed_at=now(), training_seconds_total=sum(r["training_seconds_total"] for r in snapshot["cells"]))
            save_json(result_path, snapshot)
            print(f"MATRIX_COMPLETE cells={len(snapshot['cells'])}/{snapshot['expected_cells']} smoke={args.smoke}", flush=True)
        except DeadlineReached as error:
            snapshot.update(execution_status="deadline_reached", stopped_at=now(), error=str(error))
            save_json(result_path, snapshot)
            print(str(error), flush=True)
        except BaseException as error:
            snapshot.update(execution_status="failed", failed_at=now(), error_type=type(error).__name__, error=str(error))
            save_json(result_path, snapshot)
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
