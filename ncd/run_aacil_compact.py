"""Fixed-capacity compact retention candidates; prior frozen sources unchanged.

Selection mode accepts only UNSW/NSL seeds5/6 and performs no test forward.
Permanent Normal calibration and fixed development attack holdouts are never
optimized and are accounted outside replay memory. Formal old attacks remain
restricted to retained TRAIN/VAL memory. All F output rows are preallocated,
unseen outputs are masked, and F capacity is asserted after every task.
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
import sklearn
import torch
from ncd import run_aacil as aa
from ncd import run_aacil_retention as retention
from ncd.aacil_compact import (METHODS, CompactFamily, train_compact_family,
                               attach_compact_logits, compact_balanced_finetune)
from ncd.aacil_retention import apply_weight_alignment
from ncd.model import MLP
from ncd.run_aacil_retention import (check_deadline, logits_numpy, train_gate,
                                     validation_view, stagewise_metrics)
from ncd.run_problem_scout import (binding, canonical_hash, checkpoint_load,
                                  checkpoint_save, digest, path_label,
                                  require_finite, save_json, section26_pools, smoke_subset)
from ncd.source_commit import source_commit
from ncd.train_cil import Tee, seed_everything

DATASETS = ("UNSW-NB15", "NSL-KDD", "CIC-IDS-2017", "CSE-CIC-IDS2018")
FORMAL_SEEDS = (1, 2, 3, 4, 42)
TUNING_SEEDS = (5, 6)
COEFFICIENTS = (0.25, 0.5)
SOURCE_PATHS = (*retention.SOURCE_PATHS, "ncd/aacil_compact.py", "ncd/run_aacil_compact.py")
ADAPTATION = ("Task-boundary shared attack-memory DER++ adaptation: current and retained-old attack balanced CE, "
              "stored raw-logit MSE only over seen-at-admission outputs, independent old-memory replay CE. "
              "DUAL applies the latter two losses only to the stable contribution of the joint head; "
              "no plastic-branch distillation. Not online-reservoir DER++. No task-growing branches.")
EMA_ADAPTATION = ("One slow EMA F teacher inspired by Arani, Sarfraz and Zonooz, Learning Fast Learning Slow, "
                  "ICLR2022 https://openreview.net/forum?id=uxxFrDwrE7Y. Their CLS-ER has two semantic memories; "
                  "this uses exactly one EMA copy, fixed decay .999 and consistency MSE weight1 on old replay. "
                  "Student CE learns all seen attacks; inference uses EMA only. This is a simplification, not replication.")


def command():
    return subprocess.list2cmdline(["python", "-I", "ncd/run_aacil_compact.py", *sys.argv[1:]])


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
    coefficient = float(config["lambda_g"])
    der_family = method != "aacil_ema_kd"
    kind = method
    attack_ids = [c for c in range(len(data.class_order)) if c != normal]
    mapping = {c: j for j, c in enumerate(attack_ids)}
    train, valid = section26_pools(data, seed)
    calibration_ids = valid[normal].copy()
    calibration_x = data.x_train[calibration_ids].copy()
    memory, records, cumulative = {}, [], 0.0
    seed_everything(seed * 10000)
    gate = MLP(data.x_train.shape[1], int(config["hidden"]), 2).to(device)
    family = CompactFamily(data.x_train.shape[1], int(config["hidden"]), len(attack_ids),
                           method, ema_decay=config["ema_decay"]).to(device)
    for task_index, task in enumerate(data.tasks):
        check_deadline(config)
        seen = [c for part in data.tasks[:task_index + 1] for c in part]
        new_attacks = [c for c in task if c != normal]
        family.set_seen([mapping[c] for c in seen if c != normal])
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
            family.assert_capacity()
            evaluation_family = family
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
            check_deadline(config)
            family_seconds, family_schedule = train_compact_family(
                family, attack_x, family_y, memory, mapping, config, task_seed, device)
            base_family_seconds = family_seconds
            del gx, gy, gtargets, gmask, benign_targets, attack_x, family_y, attack_xs, attack_ys
            started = time.perf_counter()
            targets = aa.attack_targets("aacil", seen, normal)
            next_memory = aa.refresh_attack_memory(data, train, valid, memory, new_attacks, targets, seed, task_index)
            selection_seconds = preparation_seconds + time.perf_counter() - started
            cbf_seconds, wa_seconds, ncm_seconds = 0.0, 0.0, 0.0
            postprocess, prototypes = {}, {}
            if method == "aacil_kd_wa":
                postprocess = apply_weight_alignment(family, [mapping[c] for c in seen if c != normal and c not in new_attacks],
                                                    [mapping[c] for c in new_attacks])
                wa_seconds = postprocess["elapsed_seconds"]
            elif method == "aacil_kd_cbf":
                check_deadline(config)
                cbf_seconds, postprocess = compact_balanced_finetune(family, next_memory, mapping, config, task_seed, device)
                family_seconds += cbf_seconds
                selection_seconds += postprocess["selection_seconds"]
            evaluation_family = family
            entry_forward_seconds, logit_storage_seconds = attach_compact_logits(
                memory, next_memory, new_attacks, gate, family, device, int(config["eval_batch_size"]))
            memory = next_memory
            tn = {str(c): len(m["train_ids"]) for c, m in memory.items()}
            vn = {str(c): len(m["validation_ids"]) for c, m in memory.items()}
            actual_total = sum(tn.values()) + sum(vn.values())
            capacity = 200 * len(seen)
            if actual_total > capacity:
                raise AssertionError("Shared replay capacity exceeded")
            sample_bytes = sum(m[k].nbytes for m in memory.values()
                               for k in ("train_ids", "validation_ids", "train_x", "validation_x"))
            logit_bytes = sum(m[k].nbytes for m in memory.values() for k in ("g_logits", "f_logits", "f_target_mask"))
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
            fixed_attack_holdout = {str(c): valid[c].tolist() for c in seen if c != normal} if config["selection_only"] else {}
            fixed_holdout_count = sum(len(ids) for ids in fixed_attack_holdout.values())
            retained_holdout_overlap = sum(len(m["validation_ids"]) for m in memory.values()) if config["selection_only"] else 0
            selection_storage = aa.row_storage(fixed_holdout_count, data.x_train.shape[1])
            selection_storage.update(outside_replay_M=True, never_optimized=True,
                                     active_only_during_selection=bool(config["selection_only"]),
                                     includes_retained_validation_rows=retained_holdout_overlap,
                                     additional_unique_rows_outside_M=fixed_holdout_count-retained_holdout_overlap,
                                     checkpoint_persistence="row IDs only; features read from bound TRAIN holdout for evaluation")
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
                "old_attack_training_pool_access": False,
                "old_attack_validation_pool_access": bool(config["selection_only"]),
                "old_attack_validation_pool_access_scope": "fixed permanent evaluation-only attack holdout outside M" if config["selection_only"] else "none; retained VAL subset only",
                "old_attack_validation_pool_access_for_memory": False,
                "training_row_ids": {str(c): m["train_ids"].tolist() for c, m in memory.items()},
                "validation_row_ids": {str(c): m["validation_ids"].tolist() for c, m in memory.items()},
                "normal_training_row_ids_sha256": hashlib.sha256(normal_rows.tobytes()).hexdigest(),
                "normal_replay_validation_row_ids": [], "unfilled_capacity": capacity - actual_total,
                "stored_logits_scope": ("training memory rows only; float32 G[2], raw F[all_attack_classes], bool seen-at-admission target mask; no duplicate features"
                                        if der_family else "training memory rows only; float32 G[2]; F has no target logits"),
                "stored_logits_policy": "post-task at new-row admission; retained old row targets unchanged",
            }
            cumulative += gate_seconds + family_seconds
            counts = {c: len(ids) for c, ids in used_ids.items()}
            state = {
                "identity": identity, "task_index": task_index, "seen_classes": seen, "new_classes": list(task),
                "attack_class_ids": attack_ids,
                "gate_model_state": {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()},
                "family_model_state": {k: v.detach().cpu().clone() for k, v in family.state_dict().items()},
                "family_ncm_prototypes": {str(c): value.detach().cpu().clone() for c, value in prototypes.items()},
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
                "base_family_training_seconds": base_family_seconds, "cbf_training_seconds": cbf_seconds,
                "weight_alignment_seconds": wa_seconds, "ncm_prototype_seconds": ncm_seconds,
                "training_seconds": gate_seconds + family_seconds, "training_seconds_cumulative": cumulative,
                "buffer_selection_seconds": selection_seconds, "teacher_forward_seconds": teacher_seconds,
                "memory_logit_forward_seconds": entry_forward_seconds, "logit_storage_seconds": logit_storage_seconds,
                "gate_training_schedule": gate_schedule, "family_training_schedule": family_schedule,
                "family_type": kind, "family_postprocess": postprocess,
                "family_metadata": family.metadata(),
                "family_training_outputs": "fixed full attack head from initialization; unseen classes masked in CE and inference",
                "gate_binary_labels": {"normal": 0, "attack": 1},
                "calibration_policy": config["calibration_policy"], "calibration_pool_rows": len(calibration_ids),
                "calibration_pool_storage": calibration_storage, "calibration_pool_budget": "outside M; permanent Normal TRAIN holdout; never optimized",
                "calibration_overlap_with_retained_normal_validation": 0,
                "calibration_additional_unique_rows_outside_M": len(calibration_ids),
                "selection_attack_validation_row_ids": fixed_attack_holdout,
                "selection_attack_holdout_storage": selection_storage,
                "selection_validation_policy": config["selection_validation_policy"],
                "ephemeral_normal_training_feature_and_id_bytes": normal_x.nbytes + normal_rows.nbytes,
                "gate_benign_teacher": "previous-task G before any current-task updates; cached current benign logits" if task_index else "none: first task KD zero",
                "effective_lambda_g": coefficient, "derpp_adaptation": ADAPTATION if der_family else EMA_ADAPTATION,
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
            valid_data = validation_view(data, calibration_x, memory, normal, valid, seen, config["selection_only"])
            validation_points = aa.evaluate_pair(gate, evaluation_family, valid_data, seen, attack_ids, calibration, config, device)
            aa.sync(device)
            validation_seconds = time.perf_counter() - started
            evaluation_seconds = 0.0
            points = {}
            if not config["selection_only"]:
                started = time.perf_counter()
                points = aa.evaluate_pair(gate, evaluation_family, data, seen, attack_ids, calibration, config, device)
                aa.sync(device)
                evaluation_seconds = time.perf_counter() - started
            record = {k: v for k, v in state.items() if k not in {
                "identity", "gate_model_state", "family_model_state", "family_ncm_prototypes", "attack_memory", "normal_training_row_ids",
                "normal_replay_validation_row_ids", "normal_replay_train_x", "normal_replay_validation_x",
                "calibration_row_ids", "optimization_attack_row_ids"}}
            record.update(checkpoint_file=checkpoint.relative_to(output).as_posix(), checkpoint_sha256=checkpoint_hash,
                          identity_hash=canonical_hash(identity), calibration=calibration, operating_points=points,
                          id=points.get("0.02"), primary_alpha=0.02, calibration_seconds=calibration_seconds,
                          evaluation_seconds=evaluation_seconds, validation_seconds=validation_seconds,
                          validation_operating_points=validation_points,
                          validation_balanced_accuracy=validation_points["0.02"]["balanced_accuracy"],
                          validation_scope=config["selection_validation_policy"] if config["selection_only"] else "permanent Normal TRAIN holdout plus current retained attack validation memory; never optimized",
                          test_evaluation_performed=not config["selection_only"],
                          evaluation_scope="test forward forbidden in selection_only; calibration alpha is not guaranteed test FPR")
            save_json(metric_path, record)
            print(f"CHECKPOINT_EVALUATED {method} seed={seed} task={task_index} selection_only={config['selection_only']}", flush=True)
        records.append(record)
    validation_metrics = stagewise_metrics([r["validation_operating_points"]["0.02"] for r in records], data.tasks)
    per_stage_capacity = [r["family_metadata"] for r in records]
    for key in ("training_parameters", "inference_parameters", "optimizer_parameters", "extra_components"):
        if len({row[key] for row in per_stage_capacity}) != 1:
            raise AssertionError("Family capacity/component count changed across stages")
    result = {"schema_version": 1, "section31_schema_version": 1, "completed": True, "completed_at": aa.now(),
              "architecture_capacity": per_stage_capacity[-1] | {"per_stage_counts": per_stage_capacity},
              "identity": identity, "class_order": data.class_order, "tasks": data.tasks, "checkpoints": records,
              "family_type": kind,
              "train_counts": {str(c): int(np.count_nonzero(data.y_train == c)) for c in range(len(data.class_order))},
              "command": command(), "log_path": path_label(output / "run.log"),
              "selection_only": config["selection_only"], "test_evaluation_performed": not config["selection_only"],
              "selection_metrics": validation_metrics if config["selection_only"] else None,
              "validation_metrics": validation_metrics,
              "test_cil_metrics": None if config["selection_only"] else stagewise_metrics([r["operating_points"]["0.02"] for r in records], data.tasks),
              "final_validation_balanced_accuracy": records[-1]["validation_balanced_accuracy"],
              "selection_rule": "UNSW and NSL seeds5/6 only: each dataset validation forgetting <= external-five minimum + .01; maximize cross-dataset mean final validation BAcc; simplicity within .01; implemented by development queue",
              "cost_scope": "training=gate+family; family=base_family+CBF (nested, not additive again); EMA consistency forward/update included in base_family; buffer_selection includes CBF subset selection; WA and gate-teacher/entry-logit/calibration/evaluation separate; prototype cost is zero; excludes loading/serialization/failed attempts",
              "limitations": [ADAPTATION if der_family else EMA_ADAPTATION,
                              "Compact retention candidate; fixed attack validation outside M only during selection, never optimization or test-based tuning.",
                              "Two networks and refreshed benign data are not compute-matched to one-stage baselines.",
                              "Does not close RQ1/RQ3 or establish substantive novelty."]}
    for key in ("training", "gate_training", "family_training", "base_family_training", "cbf_training", "weight_alignment", "ncm_prototype", "buffer_selection", "teacher_forward",
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
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["aacil_kd_weak"])
    parser.add_argument("--seeds", nargs="+", type=int, choices=(*FORMAL_SEEDS, *TUNING_SEEDS))
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--lambda-g", type=float, choices=COEFFICIENTS, required=True)
    parser.add_argument("--der-alpha", type=float, choices=(0.25, 0.5, 1.0), default=1.0)
    parser.add_argument("--der-beta", type=float, choices=(0.25, 0.5, 1.0), default=1.0)
    parser.add_argument("--cbf-epochs", type=int, choices=(1,), default=1)
    parser.add_argument("--ema-decay", type=float, choices=(0.999,), default=0.999)
    parser.add_argument("--ema-consistency", type=float, choices=(1.0,), default=1.0)
    parser.add_argument("--cbf-lr-multiplier", type=float, choices=(0.1,), default=0.1)
    parser.add_argument("--source-commit", help="Explicit full frozen source commit")
    parser.add_argument("--deadline-iso", help="Owner hard deadline, timezone-aware ISO timestamp")
    parser.add_argument("--gate-step-policy", choices=("matched_s2000",), default="matched_s2000")
    parser.add_argument("--calibration-policy", choices=("permanent_normal_holdout_outside_m",), default="permanent_normal_holdout_outside_m")
    parser.add_argument("--selection-only", action="store_true", help="No test forward; selection validation only, seeds5/6 on UNSW and NSL")
    parser.add_argument("--smoke", action="store_true", help="NONSCIENTIFIC: seed5, one epoch, <=64 rows/class, minsteps0")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.source_commit:
        if len(args.source_commit) != 40 or any(c not in "0123456789abcdef" for c in args.source_commit.lower()):
            parser.error("source-commit must be a full hexadecimal SHA")
        os.environ["NCD_SOURCE_COMMIT"] = args.source_commit
    if args.deadline_iso:
        try:
            parsed_deadline = datetime.fromisoformat(args.deadline_iso)
        except ValueError:
            parser.error("deadline-iso must be an ISO timestamp")
        if parsed_deadline.tzinfo is None:
            parser.error("deadline-iso requires an explicit time zone")
    seeds = args.seeds or list(TUNING_SEEDS if args.selection_only else FORMAL_SEEDS)
    if args.smoke:
        seeds = [5]
    if args.selection_only and (not set(args.datasets).issubset({"UNSW-NB15", "NSL-KDD"}) or any(s not in TUNING_SEEDS for s in seeds)):
        parser.error("Selection is UNSW/NSL seeds5/6 only")
    for method in args.methods:
        if method == "aacil_kd_weak":
            if args.der_alpha not in (.25, .5) or args.der_beta not in (.25, .5):
                parser.error("KD-weak uses alpha,beta in {.25,.5}")
        elif args.der_alpha != 1.0 or args.der_beta != 1.0:
            parser.error("Non-weak methods keep alpha=beta=1; EMA uses its separate consistency coefficient")
    if not args.selection_only and not args.smoke and any(s not in FORMAL_SEEDS for s in seeds):
        parser.error("Formal runs use only seeds1/2/3/4/42")
    if len(seeds) != len(set(seeds)) or len(args.methods) != len(set(args.methods)) or len(args.datasets) != len(set(args.datasets)):
        parser.error("Duplicate matrix selections are not allowed")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if int(config["buffer_per_class"]) != 200 or int(config["batch_size"]) != 256 or int(config["epochs_per_task"]) != 8:
        parser.error("Compact retention retains AA memory200*seen, batch256, epochs8")
    if args.data_root:
        config["problem_scout_data_root"] = args.data_root
    if args.device:
        config["device"] = args.device
    config.update(section27=True, section27_schema_version=1, section27_access_policy=aa.POLICY,
                  section31=True, section31_schema_version=1, lambda_g=args.lambda_g,
                  derpp_alpha=args.der_alpha, derpp_beta=args.der_beta, derpp_allocation_adaptation=ADAPTATION,
                  derpp_coefficient_source="UNSW/NSL validation seeds5/6: G lambda .25/.5; KD-weak alpha,beta .25/.5; other KD variants alpha=beta=1",
                  ema_decay=args.ema_decay, ema_consistency=args.ema_consistency,
                  family_parameter_constraint="constant through all stages; train and inference <=2x full eventual original attack-head F",
                  future_output_policy="unseen attack outputs masked; all parameter rows reserved at initialization",
                  cbf_epochs=args.cbf_epochs, cbf_lr_multiplier=args.cbf_lr_multiplier,
                  deadline_iso=args.deadline_iso,
                  gate_step_policy=args.gate_step_policy, calibration_policy=args.calibration_policy,
                  normal_training_rows=50000, min_steps_per_task=2000, smoke=args.smoke,
                  selection_only=args.selection_only, selection_aggregation="mean seeds5/6 final validation BAcc and signed old-class forgetting separately per dataset",
                  selection_validation_policy="fixed permanent stratified10percent attack TRAIN holdout plus permanent Normal TRAIN holdout; only seen classes; evaluation only; attack holdout outside M during selection only",
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
    snapshot = {"schema_version": 1, "section31_schema_version": 1, "execution_status": "in_progress", "smoke": args.smoke,
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
                        check_deadline(config)
                        train, valid = section26_pools(data, seed)
                        identity = provenance | {"dataset": dataset, "method": method, "seed": seed,
                            "class_order": data.class_order, "tasks": data.tasks, "feature_names": data.feature_names,
                            "data_binding": data_hashes, "data_metadata_sha256": canonical_hash(metadata),
                            "section27_access_policy": aa.POLICY, "section31_method": method, "section31_schema_version": 1,
                            "effective_lambda_g": args.lambda_g,
                            "family_type": method,
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
                            "selection_metrics": result["selection_metrics"]})
                        snapshot["current_cell"] = None
                        save_json(result_path, snapshot)
                del data
            snapshot.update(execution_status="complete", completed_at=aa.now(),
                            training_seconds_total=sum(c["training_seconds_total"] for c in snapshot["cells"]))
            if args.selection_only:
                keys = ("final_balanced_accuracy", "mean_incremental_balanced_accuracy", "forgetting")
                per_dataset_method = {}
                for dataset in args.datasets:
                    for method in args.methods:
                        cells = [c for c in snapshot["cells"] if c["dataset"] == dataset and c["method"] == method]
                        per_dataset_method.setdefault(dataset, {})[method] = {
                            k: float(np.mean([c["selection_metrics"][k] for c in cells])) for k in keys}
                        per_dataset_method[dataset][method].update(n_seeds=len(cells), seeds=[c["seed"] for c in cells])
                snapshot.update(selection_metrics_by_dataset_method=per_dataset_method,
                                selection_complete=not args.smoke and sorted(seeds) == list(TUNING_SEEDS),
                                selection_aggregation="seeds5/6 mean separately per dataset and candidate; joint selection belongs to queue",
                                candidate={"lambda_g": args.lambda_g, "derpp_alpha": args.der_alpha, "derpp_beta": args.der_beta,
                                           "ema_decay": args.ema_decay, "ema_consistency": args.ema_consistency,
                                           "cbf_epochs": args.cbf_epochs, "cbf_lr_multiplier": args.cbf_lr_multiplier},
                                selection_rule="both datasets satisfy external-min validation forgetting +1pp, then maximum mean final BAcc with simplicity within1pp")
            save_json(result_path, snapshot)
            print(f"MATRIX_COMPLETE cells={len(snapshot['cells'])}/{snapshot['expected_cells']} smoke={args.smoke} selection_only={args.selection_only}", flush=True)
        except BaseException as error:
            snapshot.update(execution_status="failed", failed_at=aa.now(), error_type=type(error).__name__, error=str(error))
            save_json(result_path, snapshot)
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
