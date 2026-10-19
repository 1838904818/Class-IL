"""Section27 two-stage, availability-asymmetric class-incremental NIDS.

Independent entry point; no Section26 training file is changed. This is an
RQ2 comparison, not evidence that the research innovation gate is satisfied.
The gate schedule and calibration-budget policy must be explicit CLI inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import sklearn
import torch

from ncd.crr_benchmarks import _train
from ncd.model import MLP
from ncd.run_problem_scout import (
    binding, canonical_hash, checkpoint_load, checkpoint_save, digest,
    metrics_from_confusion, path_label, require_finite, save_json,
    section26_pools, smoke_subset,
)
from ncd.source_commit import source_commit
from ncd.train_cil import Tee, seed_everything

METHODS = ("aacil", "ts_mem", "aa_noattmem")
DATASETS = ("NSL-KDD", "UNSW-NB15", "CIC-IDS-2017")
SEEDS = (1, 2, 3, 4, 42)
ALPHAS = (0.01, 0.02, 0.05)
POLICY = "benign_refresh_shared_attack_memory_only"
SOURCE_PATHS = (
    "ncd/run_aacil.py", "ncd/run_problem_scout.py",
    "ncd/prepare_problem_scout_data.py", "ncd/crr_benchmarks.py",
    "ncd/run_crr_real_sequence.py", "ncd/model.py", "ncd/train_cil.py",
    "ncd/data.py", "ncd/source_commit.py", "ncd/groundtruth.py",
    "src/methods/replay_dpmeans.py", "ncd/exp_h_data.py",
    "ncd/audit_quality_tabular.py",
)


def now():
    return datetime.now().astimezone().isoformat()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def select_rows(pool, count, seed):
    """Same deterministic permutation prefix for all compared treatments."""
    order = np.random.default_rng(seed).permutation(len(pool))[:count]
    return np.sort(np.asarray(pool, dtype=np.int64)[order])


def slot_split(capacity):
    validation = max(5, math.ceil(capacity * 0.1))
    return max(0, capacity - validation), validation


def row_storage(rows, feature_count):
    """Dense sample accounting, independent of tensor/container serialization."""
    feature_bytes, label_bytes, index_bytes = rows * feature_count * 4, rows * 8, rows * 8
    return {
        "rows": rows, "feature_count": feature_count,
        "feature_dtype": "float32", "logical_label_dtype": "int64", "index_dtype": "int64",
        "feature_bytes": feature_bytes, "logical_label_bytes": label_bytes, "index_bytes": index_bytes,
        "logical_feature_label_index_bytes": feature_bytes + label_bytes + index_bytes,
        "feature_and_index_array_bytes": feature_bytes + index_bytes,
        "label_storage": "constant Normal label or per-class attack key; no retained dense label array",
        "retained_dense_label_array_bytes": 0,
        "byte_scope": "sample arrays only; excludes model weights, metadata, serialization overhead and transient training copies",
    }


def attack_targets(method, seen, normal):
    attacks = sorted(c for c in seen if c != normal)
    if not attacks:
        raise ValueError("Section27 requires an attack in the first task")
    total = 200 * len(seen)
    if method != "aacil":
        return {c: 200 for c in attacks}
    base, remainder = divmod(total, len(attacks))
    return {c: base + int(j < remainder) for j, c in enumerate(attacks)}


def refresh_attack_memory(data, train, valid, old, new_attacks, targets, seed, task):
    """Read current attack pools only; old attacks use stored feature rows."""
    memory = {}
    for c, capacity in sorted(targets.items()):
        nt, nv = slot_split(capacity)
        rng_seed = (seed * 10000 + task) * 10007 + c
        if c in new_attacks:
            ti = select_rows(train[c], nt, rng_seed)
            vi = select_rows(valid[c], nv, rng_seed + 53)
            tx, vx = data.x_train[ti].copy(), data.x_train[vi].copy()
        else:
            previous = old[c]
            # Positions index the retained arrays, not the original TRAIN pool.
            tp = select_rows(np.arange(len(previous["train_ids"])), nt, rng_seed)
            vp = select_rows(np.arange(len(previous["validation_ids"])), nv, rng_seed + 53)
            ti, vi = previous["train_ids"][tp].copy(), previous["validation_ids"][vp].copy()
            tx, vx = previous["train_x"][tp].copy(), previous["validation_x"][vp].copy()
            if not set(ti).issubset(previous["train_ids"]) or not set(vi).issubset(previous["validation_ids"]):
                raise AssertionError("Old attack memory was replenished")
        if set(ti).intersection(vi) or len(ti) + len(vi) > capacity:
            raise AssertionError("Attack training/validation memory overlap or overflow")
        memory[c] = {"train_ids": ti, "validation_ids": vi,
                     "train_x": tx, "validation_x": vx}
    return memory


def memory_to_tensors(memory):
    return {str(c): {k: torch.from_numpy(v.copy()) for k, v in item.items()}
            for c, item in memory.items()}


def memory_from_tensors(memory):
    return {int(c): {k: v.cpu().numpy().copy() for k, v in item.items()}
            for c, item in memory.items()}


def train_model(model, x, y, config, seed, device):
    stats = {}
    sync(device)
    started = time.perf_counter()
    _train(model, x, y, method="replay", old_model=None, old_classes=(),
           seen_classes=tuple(range(model.head.out_features)),
           epochs=int(config["epochs_per_task"]), batch_size=int(config["batch_size"]),
           learning_rate=float(config["learning_rate"]), seed=seed,
           regularization=0.0, temperature=float(config["temperature"]),
           min_steps=int(config["min_steps_per_task"]), training_stats=stats)
    sync(device)
    elapsed = time.perf_counter() - started
    expected = max(int(config["min_steps_per_task"]),
                   int(config["epochs_per_task"]) * math.ceil(len(y) / int(config["batch_size"])))
    if stats["optimizer_steps"] != expected or stats["target_steps"] != expected:
        raise AssertionError("Optimizer step accounting mismatch")
    stats["output_classes_in_ce"] = model.head.out_features
    stats["single_output_ce_is_degenerate"] = model.head.out_features == 1
    return elapsed, stats


def gate_scores(gate, x, device, batch_size):
    gate.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            logits = gate(torch.from_numpy(np.asarray(x[start:start + batch_size])).to(device))
            margin = logits[:, 1] - logits[:, 0]
            if not torch.isfinite(margin).all():
                raise ValueError("Nonfinite gate calibration scores")
            parts.append(margin.cpu().numpy())
    if not parts:
        raise ValueError("Empty benign calibration set")
    return np.concatenate(parts)


def calibrate_gate(gate, x, row_ids, device, batch_size):
    scores = gate_scores(gate, x, device, batch_size)
    ordered = np.sort(scores)
    result = {}
    for alpha in ALPHAS:
        allowed = int(Fraction(str(alpha)) * len(scores))
        threshold = float(ordered[len(ordered) - allowed - 1])
        false_alarms = int(np.count_nonzero(scores > threshold))
        if false_alarms > allowed:
            raise AssertionError("Calibration tie handling exceeds alpha")
        result[str(alpha)] = {
            "target_fpr": alpha, "threshold": threshold,
            "decision": "attack iff gate_attack_logit - gate_normal_logit > threshold",
            "calibration_rows": len(scores), "allowed_false_alarms": allowed,
            "actual_false_alarms": false_alarms, "actual_fpr": false_alarms / len(scores),
            "order_statistic_zero_based": len(ordered) - allowed - 1,
            "ties": "strict greater-than; no randomized tie breaking",
            "row_ids_sha256": hashlib.sha256(np.asarray(row_ids, dtype=np.int64).tobytes()).hexdigest(),
            "threshold_source": "permanent Normal TRAIN holdout only; no test labels",
        }
    return result


def evaluate_pair(gate, family, data, seen, attack_ids, calibration, config, device):
    n = len(data.class_order)
    normal = data.class_order.index("Normal")
    attack_columns = [attack_ids.index(c) for c in seen if c != normal]
    predicted_ids = np.asarray([attack_ids[c] for c in attack_columns])
    selected = np.flatnonzero(np.isin(data.y_test, seen))
    cms = {key: np.zeros((n, n), dtype=np.int64) for key in calibration}
    gate.eval(); family.eval()
    with torch.no_grad():
        for start in range(0, len(selected), int(config["eval_batch_size"])):
            rows = selected[start:start + int(config["eval_batch_size"])]
            x = torch.from_numpy(np.asarray(data.x_test[rows])).to(device)
            gl, fl = gate(x), family(x)
            if not torch.isfinite(gl).all() or not torch.isfinite(fl).all():
                raise ValueError("Nonfinite test logits")
            margin = (gl[:, 1] - gl[:, 0]).cpu().numpy()
            attacks = predicted_ids[fl[:, attack_columns].argmax(dim=1).cpu().numpy()]
            for key, point in calibration.items():
                prediction = np.where(margin > point["threshold"], attacks, normal)
                cms[key] += np.bincount(data.y_test[rows] * n + prediction, minlength=n*n).reshape(n, n)
    result = {}
    counts = np.bincount(data.y_train, minlength=n)
    for key, cm in cms.items():
        value = metrics_from_confusion(cm, seen, data.class_order, data.tasks, counts)
        rows = [r for r in value["per_class"] if r["class_id"] != normal and r["support"] >= 20]
        value.update(total_available_rows=len(data.y_test), excluded_unseen_rows=len(data.y_test)-len(selected),
                     attack_family_macro_recall_support_ge20=float(np.mean([r["recall"] for r in rows])) if rows else None,
                     attack_guard_excluded_classes=[r["class_id"] for r in value["per_class"]
                                                   if r["class_id"] != normal and r["support"] < 20])
        result[key] = value
    return result


def run_cell(data, config, identity, output, device, evaluate_only=False):
    output.mkdir(parents=True, exist_ok=True)
    final_path = output / "result.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        if result.get("identity") != identity or not result.get("completed") or len(result["checkpoints"]) != len(data.tasks):
            raise ValueError("Completed cell source/config/completeness changed")
        for record in result["checkpoints"]:
            path = output / record["checkpoint_file"]
            if digest(path) != record["checkpoint_sha256"] or json.loads(path.with_suffix(".json").read_text()) != record:
                raise ValueError("Completed checkpoint or sidecar changed")
        print(f"CELL_SKIP {identity['dataset']} {identity['method']} seed={identity['seed']}", flush=True)
        return result
    normal = data.class_order.index("Normal")
    if normal not in data.tasks[0]:
        raise ValueError("Current Section27 datasets require Normal in task0")
    seed, method = identity["seed"], identity["method"]
    attack_ids = [c for c in range(len(data.class_order)) if c != normal]
    family_mapping = {c: j for j, c in enumerate(attack_ids)}
    train, valid = section26_pools(data, seed)
    calibration_ids = valid[normal].copy()
    calibration_x = data.x_train[calibration_ids].copy()
    memory, records, cumulative = {}, [], 0.0
    seed_everything(seed * 10000)
    gate = MLP(data.x_train.shape[1], int(config["hidden"]), 2).to(device)
    family = MLP(data.x_train.shape[1], int(config["hidden"]), len(attack_ids)).to(device)
    for task_index, task in enumerate(data.tasks):
        seen = [c for part in data.tasks[:task_index+1] for c in part]
        new_attacks = [c for c in task if c != normal]
        checkpoint = output / "checkpoints" / f"checkpoint_{task_index:03d}.pt"
        metric_path = checkpoint.with_suffix(".json")
        task_seed = seed * 10000 + task_index
        seed_everything(task_seed)
        if checkpoint.exists():
            state = checkpoint_load(checkpoint, identity)
            if state["task_index"] != task_index or state["seen_classes"] != seen:
                raise ValueError("Checkpoint task order mismatch")
            gate.load_state_dict(state["gate_model_state"])
            family.load_state_dict(state["family_model_state"])
            memory = memory_from_tensors(state["attack_memory"])
            cumulative = state["training_seconds_cumulative"]
            print(f"CHECKPOINT_RESUME {identity['dataset']} {method} seed={seed} task={task_index}", flush=True)
        else:
            if evaluate_only:
                raise FileNotFoundError(checkpoint)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            selection_started = time.perf_counter()
            normal_rng = task_seed * 10007 + normal
            normal_limit = slot_split(200)[0] if method == "ts_mem" else 50000
            normal_rows = select_rows(train[normal], normal_limit, normal_rng)
            normal_validation = select_rows(valid[normal], slot_split(200)[1], normal_rng+53) if method == "ts_mem" else np.empty(0, dtype=np.int64)
            if set(normal_rows).intersection(calibration_ids):
                raise AssertionError("Normal calibration data entered optimization")
            normal_x = data.x_train[normal_rows].copy()
            attack_xs, attack_ys, used_ids = [], [], {}
            for c in seen:
                if c == normal:
                    continue
                if c in new_attacks:
                    ids, x = train[c], data.x_train[train[c]].copy()
                else:
                    ids, x = memory[c]["train_ids"], memory[c]["train_x"]
                used_ids[str(c)] = ids.copy()
                attack_xs.append(x)
                attack_ys.append(np.full(len(ids), family_mapping[c], dtype=np.int64))
            attack_x, family_y = np.concatenate(attack_xs), np.concatenate(attack_ys)
            optimization_attack_counts = {c: len(ids) for c, ids in used_ids.items()}
            preparation_seconds = time.perf_counter() - selection_started
            gx = np.concatenate([normal_x, attack_x])
            gy = np.concatenate([np.zeros(len(normal_x), dtype=np.int64), np.ones(len(attack_x), dtype=np.int64)])
            print(f"TASK_START {identity['dataset']} {method} seed={seed} task={task_index} gate_rows={len(gy)} family_rows={len(family_y)}", flush=True)
            gate_seconds, gate_schedule = train_model(gate, gx, gy, config, task_seed, device)
            family_seconds, family_schedule = train_model(family, attack_x, family_y, config, task_seed, device)
            del gx, gy, attack_x, family_y, attack_xs, attack_ys
            selection_started = time.perf_counter()
            targets = attack_targets(method, seen, normal)
            memory = refresh_attack_memory(data, train, valid, memory, new_attacks, targets, seed, task_index)
            selection_seconds = preparation_seconds + time.perf_counter() - selection_started
            tn = {str(c): len(m["train_ids"]) for c, m in memory.items()}
            vn = {str(c): len(m["validation_ids"]) for c, m in memory.items()}
            actual_attack = sum(tn.values()) + sum(vn.values())
            normal_slots = len(normal_rows) + len(normal_validation) if method == "ts_mem" else 0
            actual_total = actual_attack + normal_slots
            capacity = 200 * len(seen)
            if actual_total > capacity:
                raise AssertionError("Shared replay memory exceeds total budget")
            memory_bytes = sum(v.nbytes for m in memory.values() for v in m.values())
            if method == "ts_mem":
                memory_bytes += normal_x.nbytes + normal_rows.nbytes + len(normal_validation) * (data.x_train.shape[1]*4+8)
            replay_storage = row_storage(actual_total, data.x_train.shape[1])
            if memory_bytes != replay_storage["feature_and_index_array_bytes"]:
                raise AssertionError("Replay sample byte accounting mismatch")
            normal_storage = row_storage(len(normal_rows), data.x_train.shape[1])
            normal_storage.update(
                outside_replay_M=method != "ts_mem", refreshed_each_task=True,
                calibration_overlap_rows=0, requested_rows=normal_limit,
                checkpoint_persistence="feature and index arrays" if method == "ts_mem" else "indices only; benign training features are ephemeral")
            calibration_storage = row_storage(len(calibration_ids), data.x_train.shape[1])
            calibration_storage.update(
                outside_replay_M=True, shared_across_methods=True, never_optimized=True,
                checkpoint_persistence="indices only; features reloaded from the same bound permanent holdout",
                overlap_with_retained_TS_normal_validation_rows=len(normal_validation),
                additional_unique_rows_outside_M=len(calibration_ids)-len(normal_validation))
            cumulative += gate_seconds + family_seconds
            allocation = {
                "policy": method, "capacity_total": capacity, "actual_total": actual_total,
                "attack_target_counts": {str(c): n for c, n in targets.items()},
                "actual_attack_training_counts": tn, "actual_attack_validation_counts": vn,
                "normal_replay_target": 200 if method == "ts_mem" else 0,
                "normal_replay_training_count": len(normal_rows) if method == "ts_mem" else 0,
                "normal_replay_validation_count": len(normal_validation),
                "normal_gate_training_count": len(normal_rows), "normal_gate_requested_rows": normal_limit,
                "normal_training_outside_replay_memory": method != "ts_mem",
                "normal_refreshed": True, "normal_refresh_rng_seed": normal_rng,
                "normal_share_not_transferred_to_attacks": method == "aa_noattmem",
                "attack_memory_shared_between_gate_and_family": True,
                "attack_memory_counted_copies": 1, "replay_memory_payload_bytes": memory_bytes,
                "replay_memory_storage": replay_storage,
                "normal_gate_training_storage": normal_storage,
                "old_attack_training_pool_access": False, "old_attack_validation_pool_access": False,
                "training_row_ids": {str(c): m["train_ids"].tolist() for c, m in memory.items()},
                "validation_row_ids": {str(c): m["validation_ids"].tolist() for c, m in memory.items()},
                "normal_training_row_ids_sha256": hashlib.sha256(normal_rows.tobytes()).hexdigest(),
                "normal_replay_validation_row_ids": normal_validation.tolist(),
                "unfilled_capacity": capacity - actual_total,
            }
            state = {
                "identity": identity, "task_index": task_index, "seen_classes": seen,
                "attack_class_ids": attack_ids, "new_classes": list(task),
                "gate_model_state": {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()},
                "family_model_state": {k: v.detach().cpu().clone() for k, v in family.state_dict().items()},
                "attack_memory": memory_to_tensors(memory),
                "normal_training_row_ids": torch.from_numpy(normal_rows.copy()),
                "normal_replay_validation_row_ids": torch.from_numpy(normal_validation.copy()),
                "normal_replay_train_x": torch.from_numpy(normal_x.copy()) if method == "ts_mem" else None,
                "normal_replay_validation_x": torch.from_numpy(data.x_train[normal_validation].copy()) if method == "ts_mem" else None,
                "calibration_row_ids": torch.from_numpy(calibration_ids.copy()),
                "optimization_attack_row_ids": {c: torch.from_numpy(ids.copy()) for c, ids in used_ids.items()},
                "optimization_attack_counts": optimization_attack_counts,
                "gate_optimization_label_counts": {"normal": len(normal_rows), "attack": sum(optimization_attack_counts.values())},
                "allocation": allocation,
                "gate_training_seconds": gate_seconds, "family_training_seconds": family_seconds,
                "training_seconds": gate_seconds + family_seconds,
                "training_seconds_cumulative": cumulative, "buffer_selection_seconds": selection_seconds,
                "gate_training_schedule": gate_schedule, "family_training_schedule": family_schedule,
                "family_training_outputs": "all attack classes including future; same unmasked-head CE convention as Replay",
                "gate_binary_labels": {"normal": 0, "attack": 1},
                "calibration_policy": config["calibration_policy"],
                "calibration_pool_rows": len(calibration_ids),
                "calibration_pool_storage": calibration_storage,
                "calibration_pool_budget": "outside M; common permanent Normal holdout; never optimized",
                "calibration_overlap_with_retained_normal_validation": len(normal_validation),
                "calibration_additional_unique_rows_outside_M": len(calibration_ids) - len(normal_validation),
                "ephemeral_normal_training_feature_and_id_bytes": normal_x.nbytes + normal_rows.nbytes if method != "ts_mem" else 0,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
            }
            checkpoint_save(checkpoint, state)
            del normal_x
            print(f"CHECKPOINT_TRAINED {identity['dataset']} {method} seed={seed} task={task_index} gate_seconds={gate_seconds:.3f} family_seconds={family_seconds:.3f}", flush=True)
        checkpoint_hash = digest(checkpoint)
        if metric_path.exists():
            record = json.loads(metric_path.read_text(encoding="utf-8"))
            if record["identity_hash"] != canonical_hash(identity) or record["checkpoint_sha256"] != checkpoint_hash:
                raise ValueError("Checkpoint metric provenance changed")
        else:
            sync(device)
            started = time.perf_counter()
            calibration = calibrate_gate(gate, calibration_x, calibration_ids, device, int(config["eval_batch_size"]))
            sync(device)
            calibration_seconds = time.perf_counter() - started
            started = time.perf_counter()
            points = evaluate_pair(gate, family, data, seen, attack_ids, calibration, config, device)
            sync(device)
            evaluation_seconds = time.perf_counter() - started
            record = {key: value for key, value in state.items() if key not in {
                "identity", "gate_model_state", "family_model_state", "attack_memory",
                "normal_training_row_ids", "normal_replay_validation_row_ids", "normal_replay_train_x",
                "normal_replay_validation_x", "calibration_row_ids", "optimization_attack_row_ids"}}
            record.update(checkpoint_file=checkpoint.relative_to(output).as_posix(),
                          checkpoint_sha256=checkpoint_hash, identity_hash=canonical_hash(identity),
                          calibration=calibration, operating_points=points, id=points["0.02"],
                          primary_alpha=0.02, calibration_seconds=calibration_seconds,
                          evaluation_seconds=evaluation_seconds,
                          evaluation_scope="calibration-set alpha is not a guarantee of equal test FPR")
            save_json(metric_path, record)
            print(f"CHECKPOINT_EVALUATED {identity['dataset']} {method} seed={seed} task={task_index}", flush=True)
        records.append(record)
    result = {
        "schema_version": 1, "completed": True, "completed_at": now(), "identity": identity,
        "class_order": data.class_order, "tasks": data.tasks, "checkpoints": records,
        "train_counts": {str(c): int(np.count_nonzero(data.y_train == c)) for c in range(len(data.class_order))},
        "training_seconds_total": sum(r["training_seconds"] for r in records),
        "gate_training_seconds_total": sum(r["gate_training_seconds"] for r in records),
        "family_training_seconds_total": sum(r["family_training_seconds"] for r in records),
        "buffer_selection_seconds_total": sum(r["buffer_selection_seconds"] for r in records),
        "calibration_seconds_total": sum(r["calibration_seconds"] for r in records),
        "evaluation_seconds_total": sum(r["evaluation_seconds"] for r in records),
        "cost_scope": "successful task optimization, selection, calibration and evaluation separately; loading and failed attempts excluded",
        "command": subprocess.list2cmdline(["python", "-I", "ncd/run_aacil.py", *sys.argv[1:]]),
        "log_path": path_label(output / "run.log"),
        "limitations": ["Two networks and extra benign exposure are not compute-matched to one-stage baselines.",
                        "Calibration and refreshed benign-stream rows outside replay M disclose float32 features, logical int64 labels and int64 indices separately; constant/class-key labels are not a retained dense array.",
                        "Section27 is an RQ2 sub-study and does not close RQ1, RQ3 or the substantive innovation gate."],
    }
    require_finite(result)
    save_json(final_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "ncd/configs/problem_scout_section21.json")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--gate-step-policy", required=True, choices=("matched_s2000",),
                        help="Explicit owner choice: G also max(8 epochs,2000 updates); no implicit default")
    parser.add_argument("--calibration-policy", required=True, choices=("permanent_normal_holdout_outside_m",),
                        help="Explicit owner choice: full common Normal holdout, separately accounted outside replay M")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if int(config["buffer_per_class"]) != 200 or int(config["batch_size"]) != 256 or int(config["epochs_per_task"]) != 8:
        parser.error("Section27 fixes 200*seen memory, batch256, and8epochs")
    if args.data_root:
        config["problem_scout_data_root"] = args.data_root
    if args.device:
        config["device"] = args.device
    config.update(section27=True, section27_schema_version=1, section27_access_policy=POLICY,
                  gate_step_policy=args.gate_step_policy, calibration_policy=args.calibration_policy,
                  normal_training_rows=50000, min_steps_per_task=2000, smoke=args.smoke,
                  cpu_threads=2, cpu_interop_threads=1)
    seeds = [1] if args.smoke else args.seeds
    if args.smoke:
        config.update(epochs_per_task=1, min_steps_per_task=0)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "git_commit": source_commit(ROOT), "effective_config": config,
        "config_sha256": digest(args.config), "effective_config_sha256": canonical_hash(config),
        "source_hashes": {p: digest(ROOT/p) for p in SOURCE_PATHS},
        "runtime_versions": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                             "numpy": np.__version__, "sklearn": sklearn.__version__, "cuda": torch.version.cuda},
    }
    expected = {"datasets": args.datasets, "methods": args.methods, "seeds": seeds}
    result_path = output / "result.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if previous["provenance"] != provenance or previous["expected"] != expected:
            raise ValueError("Matrix source/config/selection changed; preserve old output and use a new directory")
    snapshot = {"schema_version": 1, "execution_status": "in_progress", "smoke": args.smoke,
                "provenance": provenance, "expected": expected,
                "expected_cells": len(args.datasets)*len(args.methods)*len(seeds), "cells": [],
                "current_cell": None, "command": subprocess.list2cmdline(["python", "-I", "ncd/run_aacil.py", *sys.argv[1:]]),
                "log_path": path_label(output / "run.log")}
    with (output / "run.log").open("a", encoding="utf-8") as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        try:
            from ncd.prepare_problem_scout_data import prepare_setting
            for dataset in args.datasets:
                data, _, _, metadata = prepare_setting(dataset, config)
                if args.smoke:
                    data, _, _ = smoke_subset(data, None, None)
                data_hashes = binding(data, None, None)
                for method in args.methods:
                    for seed in seeds:
                        train, valid = section26_pools(data, seed)
                        identity = provenance | {"dataset": dataset, "method": method, "seed": seed,
                            "class_order": data.class_order, "tasks": data.tasks, "feature_names": data.feature_names,
                            "data_binding": data_hashes, "data_metadata_sha256": canonical_hash(metadata),
                            "section27_access_policy": POLICY,
                            "section26_holdout": {str(c): {"train_count": len(train[c]), "validation_count": len(valid[c]),
                                "validation_indices_sha256": hashlib.sha256(valid[c].tobytes()).hexdigest()} for c in valid}}
                        cell_dir = output / dataset.lower().replace("-", "_") / f"{method}_seed{seed}"
                        cell_dir.mkdir(parents=True, exist_ok=True)
                        snapshot["current_cell"] = {"dataset": dataset, "method": method, "seed": seed, "started_at": now()}
                        save_json(result_path, snapshot)
                        with (cell_dir / "run.log").open("a", encoding="utf-8") as cell_log, redirect_stdout(Tee(sys.stdout, cell_log)), redirect_stderr(Tee(sys.stderr, cell_log)):
                            result = run_cell(data, config, identity, cell_dir, device, args.evaluate_only)
                        snapshot["cells"].append({"dataset": dataset, "method": method, "seed": seed,
                            "result": (cell_dir / "result.json").relative_to(output).as_posix(),
                            "result_sha256": digest(cell_dir / "result.json"), "completed_at": result["completed_at"],
                            "checkpoints": len(result["checkpoints"]), "training_seconds_total": result["training_seconds_total"]})
                        snapshot["current_cell"] = None
                        save_json(result_path, snapshot)
                del data
            snapshot.update(execution_status="complete", completed_at=now(),
                            training_seconds_total=sum(c["training_seconds_total"] for c in snapshot["cells"]))
            save_json(result_path, snapshot)
            print(f"MATRIX_COMPLETE cells={len(snapshot['cells'])}/{snapshot['expected_cells']} smoke={args.smoke}", flush=True)
        except BaseException as error:
            snapshot.update(execution_status="failed", failed_at=now(), error_type=type(error).__name__, error=str(error))
            save_json(result_path, snapshot)
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
