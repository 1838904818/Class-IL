"""Fixed-schedule AR-Joint cell; independent of the active TB and GPU entries."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ncd import run_aacil as aa
from ncd.aacil_retention import apply_weight_alignment
from ncd.default_decision import metrics_from_predictions, summarize
from ncd.run_problem_scout import checkpoint_save, checkpoint_load, digest, save_json, section26_pools
from ncd.section123_joint import JointNetwork, train_fixed
from ncd.train_cil import seed_everything


class GateView(nn.Module):
    def __init__(self, joint):
        super().__init__()
        self.joint = joint

    def forward(self, x):
        return self.joint(x)[0]


def evaluate_joint(model, data, seen, attacks, calibration, config):
    """One shared encoder forward per TEST batch for both decision views."""
    normal, n = data.class_order.index("Normal"), len(data.class_order)
    rows = np.flatnonzero(np.isin(data.y_test, seen))
    cms = {key: np.zeros((n, n), dtype=np.int64) for key in calibration}
    default_predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), config["eval_batch_size"]):
            ids = rows[start:start + config["eval_batch_size"]]
            x = torch.as_tensor(np.array(data.x_test[ids], copy=True), dtype=torch.float32)
            gate, family = model(x)
            if not torch.isfinite(gate).all() or not torch.isfinite(family).all():
                raise ValueError("Nonfinite joint inference")
            predicted = np.asarray(attacks)[model.mask_family(family).argmax(1).numpy()]
            margin = (gate[:, 1] - gate[:, 0]).numpy()
            default_predictions.append(np.where(gate.softmax(1)[:, 1].numpy() > .5, predicted, normal))
            for key, point in calibration.items():
                p = np.where(margin > point["threshold"], predicted, normal)
                cms[key] += np.bincount(data.y_test[ids] * n + p, minlength=n*n).reshape(n, n)
    counts = np.bincount(data.y_train, minlength=n)
    points = {key: aa.metrics_from_confusion(cm, seen, data.class_order, data.tasks, counts)
              for key, cm in cms.items()}
    default = metrics_from_predictions(data.y_test[rows], np.concatenate(default_predictions), seen, data.class_order)
    return points, default


def run_joint_fixed(data, config, identity, output):
    """Original AR fixed pool, bounded old memory, persistent admission logits."""
    if config.get("schedule") != "fixed":
        raise ValueError("ES schedule integration is not implemented; refuse substitution")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError("Completed joint output must not be rerun")
    normal = data.class_order.index("Normal")
    attacks = [c for c in range(len(data.class_order)) if c != normal]
    mapping = {c: j for j, c in enumerate(attacks)}
    seed = int(identity["seed"])
    train, valid = section26_pools(data, seed)
    calibration_ids = valid[normal]
    seed_everything(seed * 10000)
    model = JointNetwork(data.x_train.shape[1], len(attacks))
    memory, records = {}, []
    for index, task in enumerate(data.tasks):
        task_seed = seed * 10000 + index
        seed_everything(task_seed)
        seen = [c for part in data.tasks[:index+1] for c in part]
        new = [c for c in task if c != normal]
        model.set_seen([mapping[c] for c in seen if c != normal])
        checkpoint = output / "checkpoints" / f"checkpoint_{index:03d}.pt"
        if checkpoint.exists():
            state = checkpoint_load(checkpoint, identity)
            if state["task_index"] != index or state["seen_classes"] != seen:
                raise ValueError("Joint checkpoint task mismatch")
            model.load_state_dict(state["model_state"])
            memory = aa.memory_from_tensors(state["attack_memory"])
        else:
            benign_ids = aa.select_rows(train[normal], 50000, task_seed * 10007 + normal)
            xs, ys = [data.x_train[benign_ids]], [np.full(len(benign_ids), normal)]
            used = [benign_ids]
            for c in seen:
                if c == normal:
                    continue
                ids = train[c] if c in new else memory[c]["train_ids"]
                values = data.x_train[ids] if c in new else memory[c]["train_x"]
                xs.append(values); ys.append(np.full(len(ids), c)); used.append(ids)
            used_ids = np.concatenate(used)
            if np.intersect1d(used_ids, calibration_ids).size:
                raise ValueError("Calibration entered joint training")
            x, y = np.concatenate(xs), np.concatenate(ys).astype(np.int64)
            by = (y != normal).astype(np.int64)
            fy = np.asarray([mapping.get(int(c), -1) for c in y], dtype=np.int64)
            retained = None
            if memory:
                retained = (np.concatenate([m["train_x"] for m in memory.values()]),
                    np.concatenate([np.full(len(m["train_ids"]), mapping[c], dtype=np.int64) for c, m in memory.items()]),
                    np.concatenate([m["f_logits"] for m in memory.values()]),
                    np.concatenate([m["f_target_mask"] for m in memory.values()]))
            schedule = train_fixed(model, x, by, fy, retained, seed=task_seed, config=config, smoke=config["smoke"])
            wa = apply_weight_alignment(model, [mapping[c] for c in seen if c != normal and c not in new], [mapping[c] for c in new])
            next_memory = aa.refresh_attack_memory(data, train, valid, memory, new,
                aa.attack_targets("aacil", seen, normal), seed, index)
            with torch.inference_mode():
                for c, slot in next_memory.items():
                    if c in new:
                        _, scores = model(torch.as_tensor(slot["train_x"], dtype=torch.float32))
                        slot["f_logits"] = scores.numpy().copy()
                        slot["f_target_mask"] = np.broadcast_to(model.seen_mask.numpy(), scores.shape).copy()
                    else:
                        positions = {int(v): j for j, v in enumerate(memory[c]["train_ids"])}
                        take = [positions[int(v)] for v in slot["train_ids"]]
                        for key in ("f_logits", "f_target_mask"):
                            slot[key] = memory[c][key][take].copy()
            memory = next_memory
            state = dict(identity=identity, task_index=index, seen_classes=seen,
                model_state=model.state_dict(), attack_memory=aa.memory_to_tensors(memory),
                training_schedule=schedule, weight_alignment=wa)
            checkpoint_save(checkpoint, state)
        calibration = aa.calibrate_gate(GateView(model), data.x_train[calibration_ids], calibration_ids, torch.device("cpu"), 4096)
        points, default = evaluate_joint(model, data, seen, attacks, calibration, config)
        record = dict(task_index=index, seen_classes=seen, checkpoint_file=checkpoint.relative_to(output).as_posix(),
            checkpoint_sha256=digest(checkpoint), training_schedule=state["training_schedule"],
            weight_alignment=state["weight_alignment"], calibration=calibration, operating_points=points, default_decision=default)
        save_json(checkpoint.with_suffix(".json"), record)
        records.append(record)
        print(f"JOINT_STAGE_COMPLETE seed={seed} task={index}", flush=True)
    result = dict(completed=True, scientific_result=not config["smoke"], identity=identity,
        tasks=data.tasks, class_order=data.class_order, checkpoints=records,
        parameters=sum(p.numel() for p in model.parameters()),
        default_summary=summarize([r["default_decision"] for r in records], data.tasks))
    save_json(output / "result.json", result)
    return result
