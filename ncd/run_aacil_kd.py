"""Section29 AA-CIL-KD/noGKD; independent entry, frozen AA sources unchanged.

This is a task-boundary shared-memory DER++ objective adaptation, not the
online-reservoir implementation. New retained attack rows receive post-task
G/F logits at admission; old targets are retained without refresh. Selection
uses only permanent Normal holdout plus retained attack validation rows.
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
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import sklearn
import torch
import torch.nn.functional as F

from ncd import run_aacil as aa
from ncd.aacil_iso import IsolatedAttackFamily, train_iso_family
from ncd.crr_benchmarks import _class_balanced_cross_entropy, training_epoch_batches
from ncd.model import MLP
from ncd.run_problem_scout import (
    binding, canonical_hash, checkpoint_load, checkpoint_save, digest,
    path_label, require_finite, save_json, section26_pools, smoke_subset,
)
from ncd.source_commit import source_commit
from ncd.train_cil import Tee, seed_everything

METHODS = ("aacil_kd", "aacil_nogkd", "aacil_iso")
DATASETS = ("UNSW-NB15", "NSL-KDD", "CIC-IDS-2017")
FORMAL_SEEDS = (1, 2, 3, 4, 42)
TUNING_SEEDS = (5, 6)
COEFFICIENTS = (0.5, 1.0, 2.0)
SOURCE_PATHS = (*aa.SOURCE_PATHS, "ncd/run_aacil_kd.py", "ncd/aacil_iso.py")
ADAPTATION = (
    "Task-boundary AA shared attack memory and unchanged AA combined-current/"
    "retained-attack CE stream/step budget; F adds alpha stored-logit MSE and "
    "beta replay CE using two independently sampled old-memory minibatches. "
    "New retained rows store post-task G/F logits at admission; old targets "
    "are subset-retained, never refreshed. This is not online-reservoir DER++."
)


def command():
    return subprocess.list2cmdline(["python", "-I", "ncd/run_aacil_kd.py", *sys.argv[1:]])


def logits_numpy(model, x, device, batch_size):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            values = model(torch.from_numpy(np.asarray(x[start:start + batch_size])).to(device))
            if not torch.isfinite(values).all():
                raise ValueError("Nonfinite teacher/entry logits")
            parts.append(values.cpu().numpy())
    return np.concatenate(parts) if parts else np.empty((0, model.head.out_features), dtype=np.float32)


def assert_schedule(stats, rows, config):
    expected = max(int(config["min_steps_per_task"]),
                   int(config["epochs_per_task"]) * math.ceil(rows / int(config["batch_size"])))
    if stats["optimizer_steps"] != expected or stats["target_steps"] != expected:
        raise AssertionError("Optimizer-step accounting changed")


def train_gate(model, x, y, targets, mask, config, seed, device, coefficient):
    """Identical data/order/control flow for KD and noGKD; only lambda differs."""
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    stats = {}
    positions = np.arange(len(y), dtype=np.int64)[:, None]
    aa.sync(device)
    started = time.perf_counter()
    for epoch, batches in training_epoch_batches(
            positions, y, int(config["batch_size"]), seed, int(config["epochs_per_task"]),
            int(config["min_steps_per_task"]), stats):
        model.train()
        for bp, by in batches:
            indices = bp[:, 0].numpy()
            bx = torch.from_numpy(np.asarray(x[indices])).to(device)
            by = by.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = _class_balanced_cross_entropy(logits, by)
            eligible = mask[indices]
            if np.any(eligible):
                selected = torch.from_numpy(eligible).to(device)
                target = torch.from_numpy(np.asarray(targets[indices][eligible])).to(device)
                loss = loss + coefficient * F.mse_loss(logits[selected], target)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite gate KD loss")
            loss.backward()
            optimizer.step()
        print(f"G_EPOCH_DONE epoch={epoch + 1} steps={stats['optimizer_steps']} lambda={coefficient}", flush=True)
    aa.sync(device)
    elapsed = time.perf_counter() - started
    assert_schedule(stats, len(y), config)
    stats.update(output_classes_in_ce=2, kd_lambda=coefficient,
                 kd_eligible_rows=int(np.count_nonzero(mask)),
                 kd_reduction="mean squared raw logits over eligible rows and both output columns")
    return elapsed, stats


def train_family(model, x, y, old_memory, mapping, config, seed, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    rng = np.random.default_rng(seed + 8317)
    stats = {}
    # Index the shared retained arrays; no second persistent feature memory.
    slots = [(c, j) for c, item in sorted(old_memory.items()) for j in range(len(item["train_ids"]))]

    def sample():
        picked = rng.choice(len(slots), min(int(config["batch_size"]), len(slots)), replace=False)
        rows = [slots[int(i)] for i in picked]
        bx = np.stack([old_memory[c]["train_x"][j] for c, j in rows])
        by = np.asarray([mapping[c] for c, _ in rows], dtype=np.int64)
        target = np.stack([old_memory[c]["f_logits"][j] for c, j in rows])
        return torch.from_numpy(bx).to(device), torch.from_numpy(by).to(device), torch.from_numpy(target).to(device)

    aa.sync(device)
    started = time.perf_counter()
    for epoch, batches in training_epoch_batches(
            x, y, int(config["batch_size"]), seed, int(config["epochs_per_task"]),
            int(config["min_steps_per_task"]), stats):
        model.train()
        for bx, by in batches:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _class_balanced_cross_entropy(model(bx), by)
            if slots:
                mx, _, mz = sample()
                loss = loss + float(config["derpp_alpha"]) * F.mse_loss(model(mx), mz)
                mx, my, _ = sample()
                loss = loss + float(config["derpp_beta"]) * _class_balanced_cross_entropy(model(mx), my)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite family DER++ loss")
            loss.backward()
            optimizer.step()
        print(f"F_EPOCH_DONE epoch={epoch + 1} steps={stats['optimizer_steps']} old_memory={len(slots)}", flush=True)
    aa.sync(device)
    elapsed = time.perf_counter() - started
    assert_schedule(stats, len(y), config)
    stats.update(output_classes_in_ce=model.head.out_features,
                 derpp_alpha=float(config["derpp_alpha"]), derpp_beta=float(config["derpp_beta"]),
                 old_replay_rows=len(slots), independent_memory_minibatches=2,
                 base_stream="same AA current plus retained-old attack rows", adaptation=ADAPTATION)
    return elapsed, stats


def attach_entry_logits(old, memory, new_attacks, gate, family, device, batch_size, store_family_targets=True):
    """Only selected new rows forward; old targets retain exact row correspondence."""
    aa.sync(device)
    started = time.perf_counter()
    forward_seconds = 0.0
    for c, item in memory.items():
        if c in new_attacks:
            aa.sync(device)
            forward_start = time.perf_counter()
            item["g_logits"] = logits_numpy(gate, item["train_x"], device, batch_size)
            item["f_logits"] = (logits_numpy(family, item["train_x"], device, batch_size)
                                if store_family_targets else np.empty((len(item["train_ids"]), 0), dtype=np.float32))
            aa.sync(device)
            forward_seconds += time.perf_counter() - forward_start
        else:
            previous = old[c]
            lookup = {int(row): j for j, row in enumerate(previous["train_ids"])}
            positions = np.asarray([lookup[int(row)] for row in item["train_ids"]], dtype=np.int64)
            item["g_logits"] = previous["g_logits"][positions].copy()
            item["f_logits"] = previous["f_logits"][positions].copy()
        if len(item["g_logits"]) != len(item["train_ids"]) or len(item["f_logits"]) != len(item["train_ids"]):
            raise AssertionError("Stored target/retained-row mismatch")
    aa.sync(device)
    elapsed = time.perf_counter() - started
    return forward_seconds, max(0.0, elapsed - forward_seconds)


def validation_view(data, calibration_x, memory, normal):
    xs, ys = [calibration_x], [np.full(len(calibration_x), normal, dtype=np.int64)]
    for c, item in sorted(memory.items()):
        if len(item["validation_ids"]):
            xs.append(item["validation_x"])
            ys.append(np.full(len(item["validation_ids"]), c, dtype=np.int64))
    # Deliberately never reference original x_test/y_test in this selection path.
    return SimpleNamespace(x_test=np.concatenate(xs), y_test=np.concatenate(ys),
                           y_train=data.y_train, class_order=data.class_order, tasks=data.tasks)


def run_cell(data, config, identity, output, device, evaluate_only=False):
    output.mkdir(parents=True, exist_ok=True)
    final_path = output / "result.json"
    if final_path.exists():
        previous = json.loads(final_path.read_text(encoding="utf-8"))
        if previous.get("identity") != identity or not previous.get("completed") or len(previous["checkpoints"]) != len(data.tasks):
            raise ValueError("Completed cell source/config/completeness changed")
        for record in previous["checkpoints"]:
            path = output / record["checkpoint_file"]
            if digest(path) != record["checkpoint_sha256"] or json.loads(path.with_suffix(".json").read_text()) != record:
                raise ValueError("Completed checkpoint/sidecar changed")
        print(f"CELL_SKIP {identity['dataset']} {identity['method']} seed={identity['seed']}", flush=True)
        return previous
    normal = data.class_order.index("Normal")
    if normal not in data.tasks[0]:
        raise ValueError("Normal must occur in task0")
    seed, method = identity["seed"], identity["method"]
    coefficient = 0.0 if method == "aacil_nogkd" else float(config["lambda_g"])
    isolated = method == "aacil_iso"
    attack_ids = [c for c in range(len(data.class_order)) if c != normal]
    mapping = {c: j for j, c in enumerate(attack_ids)}
    train, valid = section26_pools(data, seed)
    calibration_ids = valid[normal].copy()
    calibration_x = data.x_train[calibration_ids].copy()
    memory, records, cumulative = {}, [], 0.0
    seed_everything(seed * 10000)
    gate = MLP(data.x_train.shape[1], int(config["hidden"]), 2).to(device)
    family = (IsolatedAttackFamily(data.x_train.shape[1], int(config["hidden"]), len(attack_ids), rank=8)
              if isolated else MLP(data.x_train.shape[1], int(config["hidden"]), len(attack_ids))).to(device)
    for task_index, task in enumerate(data.tasks):
        seen = [c for part in data.tasks[:task_index + 1] for c in part]
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
            memory = aa.memory_from_tensors(state["attack_memory"])
            cumulative = state["training_seconds_cumulative"]
            print(f"CHECKPOINT_RESUME {method} seed={seed} task={task_index}", flush=True)
        else:
            if evaluate_only:
                raise FileNotFoundError(checkpoint)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            normal_rng = task_seed * 10007 + normal
            normal_rows = aa.select_rows(train[normal], int(config["normal_training_rows"]), normal_rng)
            if set(normal_rows).intersection(calibration_ids):
                raise AssertionError("Normal holdout entered optimization")
            normal_x = data.x_train[normal_rows].copy()
            attack_xs, attack_ys, used_ids, old_g_targets, old_masks = [], [], {}, [], []
            for c in seen:
                if c == normal:
                    continue
                if c in new_attacks:
                    ids, x = train[c], data.x_train[train[c]].copy()
                    gt = np.zeros((len(ids), 2), dtype=np.float32)
                    eligible = np.zeros(len(ids), dtype=bool)
                else:
                    ids, x = memory[c]["train_ids"], memory[c]["train_x"]
                    gt = memory[c]["g_logits"]
                    eligible = np.ones(len(ids), dtype=bool)
                used_ids[str(c)] = ids.copy()
                attack_xs.append(x)
                attack_ys.append(np.full(len(ids), mapping[c], dtype=np.int64))
                old_g_targets.append(gt)
                old_masks.append(eligible)
            attack_x, family_y = np.concatenate(attack_xs), np.concatenate(attack_ys)
            gx = np.concatenate([normal_x, attack_x])
            gy = np.concatenate([np.zeros(len(normal_x), dtype=np.int64), np.ones(len(attack_x), dtype=np.int64)])
            preparation_seconds = time.perf_counter() - started
            aa.sync(device)
            teacher_start = time.perf_counter()
            # G is still the previous checkpoint here; cache its outputs before
            # any update. This identical precomputation also runs for noGKD.
            benign_targets = (logits_numpy(gate, normal_x, device, int(config["eval_batch_size"]))
                              if task_index else np.zeros((len(normal_x), 2), dtype=np.float32))
            aa.sync(device)
            teacher_seconds = time.perf_counter() - teacher_start if task_index else 0.0
            gtargets = np.concatenate([benign_targets, *old_g_targets])
            gmask = np.concatenate([np.full(len(normal_x), bool(task_index), dtype=bool), *old_masks])
            if task_index == 0 and np.any(gmask):
                raise AssertionError("First task must have no KD teacher")
            print(f"TASK_START {identity['dataset']} {method} seed={seed} task={task_index} gate_rows={len(gy)} family_rows={len(family_y)}", flush=True)
            gate_seconds, gate_schedule = train_gate(gate, gx, gy, gtargets, gmask, config, task_seed, device, coefficient)
            if isolated:
                family_seconds, family_schedule = train_iso_family(
                    family, attack_x, family_y, [mapping[c] for c in new_attacks], config, task_seed, device)
                assert_schedule(family_schedule, len(family_y), config)
            else:
                family_seconds, family_schedule = train_family(family, attack_x, family_y, memory, mapping, config, task_seed, device)
            del gx, gy, gtargets, gmask, benign_targets, attack_x, family_y, attack_xs, attack_ys
            started = time.perf_counter()
            targets = aa.attack_targets("aacil", seen, normal)
            next_memory = aa.refresh_attack_memory(data, train, valid, memory, new_attacks, targets, seed, task_index)
            selection_seconds = preparation_seconds + time.perf_counter() - started
            entry_forward_seconds, logit_storage_seconds = attach_entry_logits(
                memory, next_memory, new_attacks, gate, family, device, int(config["eval_batch_size"]),
                store_family_targets=not isolated)
            memory = next_memory
            tn = {str(c): len(m["train_ids"]) for c, m in memory.items()}
            vn = {str(c): len(m["validation_ids"]) for c, m in memory.items()}
            actual_total = sum(tn.values()) + sum(vn.values())
            capacity = 200 * len(seen)
            if actual_total > capacity:
                raise AssertionError("Shared replay capacity exceeded")
            sample_bytes = sum(m[k].nbytes for m in memory.values()
                               for k in ("train_ids", "validation_ids", "train_x", "validation_x"))
            logit_bytes = sum(m[k].nbytes for m in memory.values() for k in ("g_logits", "f_logits"))
            replay_storage = aa.row_storage(actual_total, data.x_train.shape[1])
            if sample_bytes != replay_storage["feature_and_index_array_bytes"]:
                raise AssertionError("Replay sample byte accounting mismatch")
            normal_storage = aa.row_storage(len(normal_rows), data.x_train.shape[1])
            normal_storage.update(outside_replay_M=True, refreshed_each_task=True, calibration_overlap_rows=0,
                                  requested_rows=int(config["normal_training_rows"]), checkpoint_persistence="indices only; ephemeral benign training features")
            calibration_storage = aa.row_storage(len(calibration_ids), data.x_train.shape[1])
            calibration_storage.update(outside_replay_M=True, shared_across_methods=True, never_optimized=True,
                                       checkpoint_persistence="indices only; bound permanent holdout",
                                       overlap_with_retained_TS_normal_validation_rows=0,
                                       additional_unique_rows_outside_M=len(calibration_ids))
            allocation = {
                "policy": "aacil_attack_only_shared_memory", "capacity_total": capacity, "actual_total": actual_total,
                "attack_target_counts": {str(c): n for c, n in targets.items()},
                "actual_attack_training_counts": tn, "actual_attack_validation_counts": vn,
                "normal_replay_target": 0, "normal_replay_training_count": 0, "normal_replay_validation_count": 0,
                "normal_gate_training_count": len(normal_rows), "normal_gate_requested_rows": int(config["normal_training_rows"]),
                "normal_training_outside_replay_memory": True, "normal_refreshed": True,
                "normal_refresh_rng_seed": normal_rng, "normal_share_not_transferred_to_attacks": False,
                "attack_memory_shared_between_gate_and_family": True, "attack_memory_counted_copies": 1,
                "replay_memory_payload_bytes": sample_bytes + logit_bytes, "replay_sample_payload_bytes": sample_bytes,
                "replay_logit_target_bytes": logit_bytes, "replay_memory_storage": replay_storage,
                "normal_gate_training_storage": normal_storage,
                "old_attack_training_pool_access": False, "old_attack_validation_pool_access": False,
                "training_row_ids": {str(c): m["train_ids"].tolist() for c, m in memory.items()},
                "validation_row_ids": {str(c): m["validation_ids"].tolist() for c, m in memory.items()},
                "normal_training_row_ids_sha256": hashlib.sha256(normal_rows.tobytes()).hexdigest(),
                "normal_replay_validation_row_ids": [], "unfilled_capacity": capacity - actual_total,
                "stored_logits_scope": ("training memory rows only; float32 G[2]; isolated F has no target logits"
                                        if isolated else "training memory rows only; float32 G[2] and F[all_attack_classes]; no duplicate features"),
                "stored_logits_policy": "post-task at new-row admission; retained old row targets unchanged",
            }
            cumulative += gate_seconds + family_seconds
            counts = {c: len(ids) for c, ids in used_ids.items()}
            state = {
                "identity": identity, "task_index": task_index, "seen_classes": seen, "new_classes": list(task),
                "attack_class_ids": attack_ids,
                "gate_model_state": {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()},
                "family_model_state": {k: v.detach().cpu().clone() for k, v in family.state_dict().items()},
                "attack_memory": aa.memory_to_tensors(memory),
                "normal_training_row_ids": torch.from_numpy(normal_rows.copy()),
                "normal_replay_validation_row_ids": torch.empty(0, dtype=torch.int64),
                "normal_replay_train_x": None, "normal_replay_validation_x": None,
                "calibration_row_ids": torch.from_numpy(calibration_ids.copy()),
                "optimization_attack_row_ids": {c: torch.from_numpy(ids.copy()) for c, ids in used_ids.items()},
                "optimization_attack_counts": counts,
                "gate_optimization_label_counts": {"normal": len(normal_rows), "attack": sum(counts.values())},
                "allocation": allocation,
                "gate_training_seconds": gate_seconds, "family_training_seconds": family_seconds,
                "training_seconds": gate_seconds + family_seconds, "training_seconds_cumulative": cumulative,
                "buffer_selection_seconds": selection_seconds, "teacher_forward_seconds": teacher_seconds,
                "memory_logit_forward_seconds": entry_forward_seconds, "logit_storage_seconds": logit_storage_seconds,
                "gate_training_schedule": gate_schedule, "family_training_schedule": family_schedule,
                "family_type": "isolated_attack_heads" if isolated else "mlp_derpp",
                "family_metadata": family.metadata() if isolated else {
                    "family_type": "mlp_derpp", "output_attack_classes": len(attack_ids),
                    "total_parameters": sum(p.numel() for p in family.parameters())},
                "family_training_outputs": ("independent binary heads; inference uses positive softmax probability argmax over seen attacks; no router"
                                            if isolated else "all attack outputs including future; same unmasked-head CE convention as AA"),
                "gate_binary_labels": {"normal": 0, "attack": 1},
                "calibration_policy": config["calibration_policy"], "calibration_pool_rows": len(calibration_ids),
                "calibration_pool_storage": calibration_storage, "calibration_pool_budget": "outside M; permanent Normal TRAIN holdout; never optimized",
                "calibration_overlap_with_retained_normal_validation": 0,
                "calibration_additional_unique_rows_outside_M": len(calibration_ids),
                "ephemeral_normal_training_feature_and_id_bytes": normal_x.nbytes + normal_rows.nbytes,
                "gate_benign_teacher": "previous-task G before any current-task updates; cached current benign logits" if task_index else "none: first task KD zero",
                "effective_lambda_g": coefficient, "derpp_adaptation": None if isolated else ADAPTATION,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
            }
            checkpoint_save(checkpoint, state)
            del normal_x
        checkpoint_hash = digest(checkpoint)
        if metric_path.exists():
            record = json.loads(metric_path.read_text(encoding="utf-8"))
            if record["identity_hash"] != canonical_hash(identity) or record["checkpoint_sha256"] != checkpoint_hash:
                raise ValueError("Checkpoint metric provenance changed")
        else:
            aa.sync(device)
            started = time.perf_counter()
            calibration = aa.calibrate_gate(gate, calibration_x, calibration_ids, device, int(config["eval_batch_size"]))
            aa.sync(device)
            calibration_seconds = time.perf_counter() - started
            started = time.perf_counter()
            valid_data = validation_view(data, calibration_x, memory, normal)
            validation_points = aa.evaluate_pair(gate, family, valid_data, seen, attack_ids, calibration, config, device)
            aa.sync(device)
            validation_seconds = time.perf_counter() - started
            evaluation_seconds = 0.0
            points = {}
            if not config["selection_only"]:
                started = time.perf_counter()
                points = aa.evaluate_pair(gate, family, data, seen, attack_ids, calibration, config, device)
                aa.sync(device)
                evaluation_seconds = time.perf_counter() - started
            record = {k: v for k, v in state.items() if k not in {
                "identity", "gate_model_state", "family_model_state", "attack_memory", "normal_training_row_ids",
                "normal_replay_validation_row_ids", "normal_replay_train_x", "normal_replay_validation_x",
                "calibration_row_ids", "optimization_attack_row_ids"}}
            record.update(checkpoint_file=checkpoint.relative_to(output).as_posix(), checkpoint_sha256=checkpoint_hash,
                          identity_hash=canonical_hash(identity), calibration=calibration, operating_points=points,
                          id=points.get("0.02"), primary_alpha=0.02, calibration_seconds=calibration_seconds,
                          evaluation_seconds=evaluation_seconds, validation_seconds=validation_seconds,
                          validation_operating_points=validation_points,
                          validation_balanced_accuracy=validation_points["0.02"]["balanced_accuracy"],
                          validation_scope="permanent Normal TRAIN holdout plus current retained attack validation memory; never optimized",
                          test_evaluation_performed=not config["selection_only"],
                          evaluation_scope="test forward forbidden in selection_only; calibration alpha is not guaranteed test FPR")
            save_json(metric_path, record)
            print(f"CHECKPOINT_EVALUATED {method} seed={seed} task={task_index} selection_only={config['selection_only']}", flush=True)
        records.append(record)
    result = {"schema_version": 1, "section29_schema_version": 1, "completed": True, "completed_at": aa.now(),
              "identity": identity, "class_order": data.class_order, "tasks": data.tasks, "checkpoints": records,
              "family_type": "isolated_attack_heads" if isolated else "mlp_derpp",
              "train_counts": {str(c): int(np.count_nonzero(data.y_train == c)) for c in range(len(data.class_order))},
              "command": command(), "log_path": path_label(output / "run.log"),
              "selection_only": config["selection_only"], "test_evaluation_performed": not config["selection_only"],
              "selection_score": float(np.mean([r["validation_balanced_accuracy"] for r in records])),
              "selection_score_aggregation": "mean stagewise alpha=.02 validation BAcc; candidate score is mean of seeds5/6",
              "final_validation_balanced_accuracy": records[-1]["validation_balanced_accuracy"],
              "selection_tiebreak": "ascending lambda_g, then derpp_alpha, then derpp_beta",
              "cost_scope": "successful stage optimization, selection, teacher forwards, logit storage, calibration and evaluation separately; excludes loading/serialization/failed attempts",
              "limitations": ["ISO freezes the first-task encoder and learned binary heads; no router; no F DER targets." if isolated else ADAPTATION,
                              "Exploratory Section29 method; no tuning on formal test metrics.",
                              "Two networks and refreshed benign data are not compute-matched to one-stage baselines.",
                              "Does not close RQ1/RQ3 or establish substantive novelty."]}
    for key in ("training", "gate_training", "family_training", "buffer_selection", "teacher_forward",
                "memory_logit_forward", "logit_storage", "calibration", "evaluation", "validation"):
        result[key + "_seconds_total"] = sum(r[key + "_seconds"] for r in records)
    require_finite(result)
    save_json(final_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "ncd/configs/problem_scout_section21.json")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=["UNSW-NB15"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["aacil_kd"])
    parser.add_argument("--seeds", nargs="+", type=int, choices=(*FORMAL_SEEDS, *TUNING_SEEDS))
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--lambda-g", type=float, choices=COEFFICIENTS, required=True)
    parser.add_argument("--der-alpha", type=float, choices=COEFFICIENTS, required=True)
    parser.add_argument("--der-beta", type=float, choices=COEFFICIENTS, required=True)
    parser.add_argument("--gate-step-policy", choices=("matched_s2000",), default="matched_s2000")
    parser.add_argument("--calibration-policy", choices=("permanent_normal_holdout_outside_m",), default="permanent_normal_holdout_outside_m")
    parser.add_argument("--selection-only", action="store_true", help="No test forward; selection validation only, seeds5/6 on UNSW")
    parser.add_argument("--smoke", action="store_true", help="NONSCIENTIFIC: seed5, one epoch, <=64 rows/class, minsteps0")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    seeds = args.seeds or list(TUNING_SEEDS if args.selection_only else FORMAL_SEEDS)
    if args.smoke:
        seeds = [5]
    if args.selection_only and (args.datasets != ["UNSW-NB15"] or args.methods != ["aacil_kd"] or any(s not in TUNING_SEEDS for s in seeds)):
        parser.error("Selection is UNSW AA-CIL-KD seeds5/6 only")
    if not args.selection_only and not args.smoke and any(s not in FORMAL_SEEDS for s in seeds):
        parser.error("Formal runs use only seeds1/2/3/4/42")
    if len(seeds) != len(set(seeds)) or len(args.methods) != len(set(args.methods)) or len(args.datasets) != len(set(args.datasets)):
        parser.error("Duplicate matrix selections are not allowed")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if int(config["buffer_per_class"]) != 200 or int(config["batch_size"]) != 256 or int(config["epochs_per_task"]) != 8:
        parser.error("Section29 retains AA memory200*seen, batch256, epochs8")
    if args.data_root:
        config["problem_scout_data_root"] = args.data_root
    if args.device:
        config["device"] = args.device
    config.update(section27=True, section27_schema_version=1, section27_access_policy=aa.POLICY,
                  section29=True, section29_schema_version=1, lambda_g=args.lambda_g,
                  derpp_alpha=args.der_alpha, derpp_beta=args.der_beta, derpp_allocation_adaptation=ADAPTATION,
                  derpp_coefficient_source="Section29 predeclared .5/1/2 grid; selection-only UNSW seeds5/6",
                  gate_step_policy=args.gate_step_policy, calibration_policy=args.calibration_policy,
                  normal_training_rows=50000, min_steps_per_task=2000, smoke=args.smoke,
                  selection_only=args.selection_only, selection_aggregation="mean_incremental_validation_bacc_then_mean_seeds5_6",
                  cpu_threads=2, cpu_interop_threads=1)
    if args.smoke:
        config.update(epochs_per_task=1, min_steps_per_task=0)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance = {"git_commit": source_commit(ROOT), "effective_config": config,
                  "config_sha256": digest(args.config), "effective_config_sha256": canonical_hash(config),
                  "source_hashes": {p: digest(ROOT / p) for p in SOURCE_PATHS},
                  "runtime_versions": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                                       "numpy": np.__version__, "sklearn": sklearn.__version__, "cuda": torch.version.cuda}}
    expected = {"datasets": args.datasets, "methods": args.methods, "seeds": seeds}
    result_path = output / "result.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if previous["provenance"] != provenance or previous["expected"] != expected:
            raise ValueError("Matrix source/config/selection changed; use a new output directory")
    snapshot = {"schema_version": 1, "section29_schema_version": 1, "execution_status": "in_progress", "smoke": args.smoke,
                "selection_only": args.selection_only, "provenance": provenance, "expected": expected,
                "expected_cells": len(args.datasets) * len(args.methods) * len(seeds), "cells": [],
                "current_cell": None, "command": command(), "log_path": path_label(output / "run.log")}
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
                            "section27_access_policy": aa.POLICY, "section29_method": method,
                            "effective_lambda_g": 0.0 if method == "aacil_nogkd" else args.lambda_g,
                            "family_type": "isolated_attack_heads" if method == "aacil_iso" else "mlp_derpp",
                            "section26_holdout": {str(c): {"train_count": len(train[c]), "validation_count": len(valid[c]),
                                "validation_indices_sha256": hashlib.sha256(valid[c].tobytes()).hexdigest()} for c in valid}}
                        cell_dir = output / dataset.lower().replace("-", "_") / f"{method}_seed{seed}"
                        cell_dir.mkdir(parents=True, exist_ok=True)
                        snapshot["current_cell"] = {"dataset": dataset, "method": method, "seed": seed, "started_at": aa.now()}
                        save_json(result_path, snapshot)
                        with (cell_dir / "run.log").open("a", encoding="utf-8") as cell_log, redirect_stdout(Tee(sys.stdout, cell_log)), redirect_stderr(Tee(sys.stderr, cell_log)):
                            result = run_cell(data, config, identity, cell_dir, device, args.evaluate_only)
                        snapshot["cells"].append({"dataset": dataset, "method": method, "seed": seed,
                            "result": (cell_dir / "result.json").relative_to(output).as_posix(),
                            "result_sha256": digest(cell_dir / "result.json"), "completed_at": result["completed_at"],
                            "checkpoints": len(result["checkpoints"]), "training_seconds_total": result["training_seconds_total"],
                            "selection_score": result["selection_score"]})
                        snapshot["current_cell"] = None
                        save_json(result_path, snapshot)
                del data
            snapshot.update(execution_status="complete", completed_at=aa.now(),
                            training_seconds_total=sum(c["training_seconds_total"] for c in snapshot["cells"]))
            if args.selection_only:
                snapshot.update(selection_score=float(np.mean([c["selection_score"] for c in snapshot["cells"]])),
                                selection_complete=not args.smoke and sorted(seeds) == list(TUNING_SEEDS),
                                selection_aggregation="mean stagewise validation BAcc per seed, then mean seeds5/6",
                                candidate=[args.lambda_g, args.der_alpha, args.der_beta],
                                selection_tiebreak="ascending lambda_g, then der_alpha, then der_beta; across all27 completed candidates")
            save_json(result_path, snapshot)
            print(f"MATRIX_COMPLETE cells={len(snapshot['cells'])}/{snapshot['expected_cells']} smoke={args.smoke} selection_only={args.selection_only}", flush=True)
        except BaseException as error:
            snapshot.update(execution_status="failed", failed_at=aa.now(), error_type=type(error).__name__, error=str(error))
            save_json(result_path, snapshot)
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
