"""Section138 MLP trajectory: independent G/F, fixed F, checkpointed stages."""
import json
import time

import numpy as np
import torch

from ncd import run_aacil as aa
from ncd.aacil_compact import train_compact_family, attach_compact_logits
from ncd.default_decision import evaluate_models, summarize
from ncd.run_problem_scout import checkpoint_load, checkpoint_save, digest, save_json
from ncd.section81_exposure import ERExposure
from ncd.section138_models import build_models, postprocess_family
from ncd.section138_training import train_gate_schedule
from ncd.train_cil import seed_everything


def run_cell(data, config, identity, output, configuration, *, model_builder=None,
             gate_trainer=None):
    output.mkdir(parents=True, exist_ok=True)
    final = output / "result.json"
    if final.exists():
        result = json.loads(final.read_text())
        if result["identity"] != identity or not result["completed"]:
            raise ValueError("Existing result identity differs")
        for cp in result["checkpoints"]:
            if digest(output / cp["checkpoint_file"]) != cp["checkpoint_sha256"]:
                raise ValueError("Completed checkpoint changed")
        return result
    seed = identity["seed"]
    device = torch.device("cpu")
    exposure = ERExposure(data, seed)
    normal = exposure.normal
    attacks = [c for c in range(len(data.class_order)) if c != normal]
    mapping = {c: i for i, c in enumerate(attacks)}
    switches = configuration.switches()
    first = exposure(0)
    first_x = []
    for c in sorted(first):
        first_x.append(first[c]["train_x"])
        if c == normal and configuration.gate_schedule == 1:
            first_x.append(first[c]["validation_x"])
    seed_everything(seed * 10000)
    gate, family, spec = (model_builder or build_models)(configuration, identity["dataset"], data.feature_names,
                                      np.concatenate(first_x), len(attacks))
    memory, records = {}, []
    for index, task in enumerate(data.tasks):
        started = time.perf_counter()
        task_seed = seed * 10000 + index
        seed_everything(task_seed)
        seen = [c for part in data.tasks[:index + 1] for c in part]
        new = [c for c in task if c != normal]
        family.set_seen([mapping[c] for c in seen if c != normal])
        checkpoint = output / "checkpoints" / f"checkpoint_{index:03d}.pt"
        if checkpoint.exists():
            state = checkpoint_load(checkpoint, identity)
            if state["task_index"] != index or state["seen_classes"] != seen or state["feature_spec"] != spec:
                raise ValueError("Checkpoint schema or task mismatch")
            gate.load_state_dict(state["gate_state"], strict=True)
            family.load_state_dict(state["family_state"], strict=True)
            memory = aa.memory_from_tensors(state["attack_memory"])
        else:
            current = exposure(index)
            slots = {c: current[c] if c in current else memory[c] for c in seen}
            normal_ids = aa.select_rows(exposure.train[normal], switches["normal_training_rows"], task_seed * 10007 + normal)
            if configuration.gate_schedule == 2:
                normal_ids = normal_ids[~np.isin(normal_ids, exposure.normal_stopping_ids)]
            ids = np.concatenate([normal_ids, *[slots[c]["train_ids"] for c in seen if c != normal]])
            gx = data.x_train[ids].copy()
            gy = (data.y_train[ids] != normal).astype(np.int64)
            stop_ids = np.concatenate([slots[c]["validation_ids"] for c in seen])
            gate_seconds, gate_stats = (gate_trainer or train_gate_schedule)(
                gate, gx, gy, schedule=switches["gate_schedule"], config=config, seed=task_seed,
                device=device, training_ids=ids, stopping_ids=stop_ids,
                calibration_ids=exposure.calibration_ids, stopping_x=data.x_train[stop_ids],
                stopping_y=(data.y_train[stop_ids] != normal).astype(np.int64))
            fx = np.concatenate([slots[c]["train_x"] for c in seen if c != normal])
            fy = np.concatenate([np.full(len(slots[c]["train_ids"]), mapping[c], dtype=np.int64)
                                 for c in seen if c != normal])
            tree_family = switches.get("family") == "histgb"
            if tree_family:
                from ncd.section156_models import train_histgb_family
                family_ids = np.concatenate([slots[c]["train_ids"] for c in seen if c != normal])
                family_stop = np.concatenate([slots[c]["validation_ids"] for c in seen if c != normal])
                family_seconds, family_stats = train_histgb_family(family, fx, fy,
                    stopping_x=data.x_train[family_stop],
                    stopping_y=np.asarray([mapping[int(c)] for c in data.y_train[family_stop]]),
                    seed=task_seed, training_ids=family_ids, stopping_ids=family_stop,
                    calibration_ids=exposure.calibration_ids)
            else:
                family_seconds, family_stats = train_compact_family(
                    family, fx, fy, memory, mapping, config, task_seed, device)
            next_memory = aa.refresh_attack_memory(data, exposure.train, exposure.valid, memory, new,
                aa.attack_targets("aacil", seen, normal), seed, index)
            if tree_family:
                post = dict(operations=[], logits_stored=False, weight_alignment=False, cbf=False)
            else:
                post = postprocess_family(family, configuration.retention,
                    [mapping[c] for c in seen if c != normal and c not in new], [mapping[c] for c in new],
                    next_memory, mapping, config, task_seed, device)
                attach_compact_logits(memory, next_memory, new, gate, family, device, 4096)
            memory = next_memory
            state = dict(identity=identity, task_index=index, seen_classes=seen, feature_spec=spec,
                gate_state=gate.state_dict(), family_state=family.state_dict(),
                attack_memory=aa.memory_to_tensors(memory), gate_training_seconds=gate_seconds,
                family_training_seconds=family_seconds, gate_schedule=gate_stats,
                family_schedule=family_stats, postprocessing=post,
                optimization_ids=torch.from_numpy(ids), stopping_ids=torch.from_numpy(stop_ids))
            checkpoint_save(checkpoint, state)
        calibration = aa.calibrate_gate(gate, data.x_train[exposure.calibration_ids],
                                        exposure.calibration_ids, device, 4096)
        points = aa.evaluate_pair(gate, family, data, seen, attacks, calibration, config, device)
        default = evaluate_models(data.x_test, data.y_test, seen, data.class_order, gate,
                                  family=family, attack_ids=attacks)
        record = dict(task_index=index, seen_classes=seen,
            checkpoint_file=checkpoint.relative_to(output).as_posix(), checkpoint_sha256=digest(checkpoint),
            gate_schedule=state["gate_schedule"], family_schedule=state["family_schedule"],
            gate_training_seconds=state["gate_training_seconds"],
            family_training_seconds=state["family_training_seconds"], postprocessing=state["postprocessing"],
            calibration=calibration, operating_points=points, default_decision=default,
            stage_attempt_seconds=time.perf_counter() - started)
        save_json(checkpoint.with_suffix(".json"), record)
        records.append(record)
        print(f"STAGE_COMPLETE task={index}", flush=True)
    result = dict(completed=True, scientific_result=not config["smoke"], identity=identity,
                  checkpoints=records, tasks=data.tasks, class_order=data.class_order,
                  default_summary=summarize([r["default_decision"] for r in records], data.tasks))
    save_json(final, result)
    return result
