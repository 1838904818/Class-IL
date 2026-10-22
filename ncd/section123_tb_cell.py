"""TRAIN-only TB cells with bounded replay, stage checkpoints and both views."""
from __future__ import annotations

import json
import numpy as np
import torch
from torch.nn import functional as F

from ncd import run_aacil as aa, run_replay_bs as bs
from ncd.aacil_compact import CompactFamily, attach_compact_logits
from ncd.aacil_retention import apply_weight_alignment
from ncd.crr_benchmarks import _class_balanced_cross_entropy as balanced_ce
from ncd.default_decision import evaluate_models, summarize
from ncd.model import MLP
from ncd.run_problem_scout import checkpoint_load, checkpoint_save, digest, save_json
from ncd.section81_exposure import ERExposure
from ncd.section123_es_training import assert_disjoint_roles, train_es
from ncd.train_cil import seed_everything


def assemble(exposure, index, memory):
    current = exposure(index)
    seen = [c for task in exposure.data.tasks[:index + 1] for c in task]
    slots = {c: current[c] if c in current else memory[c] for c in seen}
    arrays = {}
    for role in ("train", "validation"):
        arrays[role + "_x"] = np.concatenate([slots[c][role + "_x"] for c in seen])
        arrays[role + "_y"] = np.concatenate([
            np.full(len(slots[c][role + "_ids"]), c, dtype=np.int64) for c in seen])
        arrays[role + "_ids"] = np.concatenate([slots[c][role + "_ids"] for c in seen])
    assert_disjoint_roles(arrays["train_ids"], arrays["validation_ids"], exposure.calibration_ids)
    return arrays, seen


def family_objective(memory, mapping, seed, device):
    rng = np.random.default_rng(seed + 8317)
    slots = [(c, j) for c, m in sorted(memory.items()) for j in range(len(m["train_ids"]))]

    def sample():
        picked = rng.choice(len(slots), min(256, len(slots)), replace=False)
        pairs = [slots[int(i)] for i in picked]
        values = (np.stack([memory[c]["train_x"][j] for c, j in pairs]),
                  np.asarray([mapping[c] for c, _ in pairs], dtype=np.int64),
                  np.stack([memory[c]["f_logits"][j] for c, j in pairs]),
                  np.stack([memory[c]["f_target_mask"][j] for c, j in pairs]))
        return tuple(torch.from_numpy(v).to(device) for v in values)

    def loss(model, x, y):
        value = balanced_ce(model.mask(model.student_logits(x)), y)
        if slots:
            xx, _, target, mask = sample()
            if not torch.all(mask.any(1)):
                raise ValueError("Missing retained F targets")
            value = value + F.mse_loss(model.entry_logits(xx)[mask], target[mask])
            xx, yy, _, _ = sample()
            value = value + balanced_ce(model.mask(model.student_logits(xx)), yy)
        return value
    return loss


def run_tb(data, counts, config, identity, output):
    method, seed = identity["method"], int(identity["seed"])
    if method not in ("ar_es", "replay_bs_es") or config["hidden"] != 128:
        raise ValueError("TB methods require registered 2x128 architecture")
    output.mkdir(parents=True, exist_ok=True)
    final = output / "result.json"
    if final.exists():
        previous = json.loads(final.read_text())
        if previous["identity"] != identity or not previous["completed"]:
            raise ValueError("Existing TB cell differs")
        for row in previous["checkpoints"]:
            if digest(output / row["checkpoint_file"]) != row["checkpoint_sha256"]:
                raise ValueError("Existing TB checkpoint changed")
        return previous
    device = torch.device("cpu")
    exposure = ERExposure(data, seed)
    normal = exposure.normal
    attacks = [c for c in range(len(data.class_order)) if c != normal]
    mapping = {c: i for i, c in enumerate(attacks)}
    seed_everything(seed * 10000)
    model = MLP(data.x_train.shape[1], 128, 2 if method == "ar_es" else len(data.class_order))
    family = CompactFamily(data.x_train.shape[1], 128, len(attacks), "aacil_kd_wa") if method == "ar_es" else None
    memory, records = {}, []
    for index, task in enumerate(data.tasks):
        task_seed = seed * 10000 + index
        seed_everything(task_seed)
        seen = [c for part in data.tasks[:index + 1] for c in part]
        new = [c for c in task if c != normal]
        checkpoint = output / "checkpoints" / f"checkpoint_{index:03d}.pt"
        if family is not None:
            family.set_seen([mapping[c] for c in seen if c != normal])
        if checkpoint.exists():
            state = checkpoint_load(checkpoint, identity)
            if state["task_index"] != index or state["seen_classes"] != seen:
                raise ValueError("TB checkpoint task mismatch")
            model.load_state_dict(state["model_state"])
            if family is not None:
                family.load_state_dict(state["family_state"])
            memory = aa.memory_from_tensors(state["attack_memory"])
        else:
            arrays, _ = assemble(exposure, index, memory)
            x, y = arrays["train_x"], arrays["train_y"]
            vx, vy = arrays["validation_x"], arrays["validation_y"]
            common = dict(seed=task_seed, device=device, config=config, smoke=config["smoke"])
            labels, vlabels = ((y != normal).astype(np.int64), (vy != normal).astype(np.int64)) if family is not None else (y, vy)
            schedule = train_es(model, x, labels, vx, vlabels,
                objective=lambda m, xx, yy: balanced_ce(m(xx), yy), logits=lambda m, xx: m(xx),
                seen_classes=[0, 1] if family is not None else seen, **common)
            family_schedule, wa = None, None
            if family is not None:
                selected, vselected = y != normal, vy != normal
                fy = np.asarray([mapping[int(c)] for c in y[selected]], dtype=np.int64)
                fvy = np.asarray([mapping[int(c)] for c in vy[vselected]], dtype=np.int64)
                family_schedule = train_es(family, x[selected], fy, vx[vselected], fvy,
                    objective=family_objective(memory, mapping, task_seed, device),
                    logits=lambda m, xx: m.student_logits(xx),
                    seen_classes=[mapping[c] for c in seen if c != normal], **common)
                wa = apply_weight_alignment(family, [mapping[c] for c in seen if c != normal and c not in new],
                                            [mapping[c] for c in new])
            next_memory = aa.refresh_attack_memory(data, exposure.train, exposure.valid, memory, new,
                aa.attack_targets("aacil", seen, normal), seed, index)
            if family is not None:
                attach_compact_logits(memory, next_memory, new, model, family, device, 4096)
            memory = next_memory
            state = dict(identity=identity, task_index=index, seen_classes=seen,
                model_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                family_state=None if family is None else {k: v.detach().cpu().clone() for k, v in family.state_dict().items()},
                attack_memory=aa.memory_to_tensors(memory), training_schedule=schedule,
                family_training_schedule=family_schedule, weight_alignment=wa,
                optimization_row_ids=torch.from_numpy(arrays["train_ids"]),
                stopping_row_ids=torch.from_numpy(arrays["validation_ids"]))
            checkpoint_save(checkpoint, state)
            del arrays, x, y, vx, vy
        if family is None:
            calibration, points, _, _ = bs.evaluate(model, data, seen, exposure.calibration_ids, counts, config, device)
        else:
            calibration = aa.calibrate_gate(model, data.x_train[exposure.calibration_ids], exposure.calibration_ids, device, 4096)
            points = aa.evaluate_pair(model, family, data, seen, attacks, calibration, config, device)
        default = evaluate_models(data.x_test, data.y_test, seen, data.class_order, model,
                                  family=family, attack_ids=attacks if family is not None else None)
        record = dict(task_index=index, seen_classes=seen, checkpoint_file=checkpoint.relative_to(output).as_posix(),
            checkpoint_sha256=digest(checkpoint), training_schedule=state["training_schedule"],
            family_training_schedule=state["family_training_schedule"], weight_alignment=state["weight_alignment"],
            calibration=calibration, operating_points=points, default_decision=default)
        save_json(checkpoint.with_suffix(".json"), record)
        records.append(record)
        print(f"TB_STAGE_COMPLETE method={method} seed={seed} task={index}", flush=True)
    result = dict(completed=True, scientific_result=not config["smoke"], identity=identity,
        tasks=data.tasks, class_order=data.class_order, checkpoints=records,
        default_summary=summarize([r["default_decision"] for r in records], data.tasks),
        exposure_policy=exposure.description())
    save_json(final, result)
    return result
