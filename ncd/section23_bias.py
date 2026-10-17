"""Section 23 bias-correction primitives; historical runners are unchanged.

These are explicit shared-MLP/Adam adaptations, not full reproductions of the
image-classification protocols. SS-IL retains the selected base auxiliary loss
and the shared minibatch-class-balanced CE. Fine-tuning and calibration costs
are additional to the base task budget. Only current data and stored old memory
may be supplied; test data must never be used for fitting or selection.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from ncd.crr_benchmarks import (
    _class_balanced_cross_entropy, _clean_prediction_kd, _distribution_change,
    _local_targets, _old_mask, _rcl_profile_loss, training_epoch_batches,
)


def _finite(value, name):
    if isinstance(value, torch.Tensor):
        valid = bool(torch.isfinite(value).all())
    else:
        valid = bool(np.isfinite(value).all())
    if not valid:
        raise ValueError(f"nonfinite {name}")


def _check_model(model):
    _check_tensors(tuple(model.parameters()), "model parameters")


def _check_tensors(tensors, name):
    # One device-to-host synchronization per group, not one per parameter.
    checks = [torch.isfinite(tensor).all() for tensor in tensors]
    if checks and not bool(torch.stack(checks).all()):
        raise ValueError(f"nonfinite {name}")


def _step(loss, optimizer, parameters):
    _finite(loss, "loss")
    loss.backward()
    parameters = tuple(parameters)
    _check_tensors([p.grad for p in parameters if p.grad is not None], "gradient")
    optimizer.step()
    _check_tensors(parameters, "updated parameter")


def _ids(classes, width):
    ids = tuple(int(c) for c in classes)
    if len(set(ids)) != len(ids) or any(c < 0 or c >= width for c in ids):
        raise ValueError("invalid or repeated class IDs")
    return ids


def _xy(x, y, *, allow_empty=False):
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("invalid feature/label shapes")
    if not allow_empty and not len(y):
        raise ValueError("empty training data")
    if x.dtype != np.float32 or y.dtype != np.int64:
        raise ValueError("expected float32 features and int64 labels")
    _finite(x, "features")
    return x, y


def _positive(value, name):
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def taskwise_kd(student, teacher, tasks, temperature=2.0):
    """Sum per-task KL, including exact zero for a singleton task.

    Teacher logits are always detached. A task not supplied here is not distilled.
    """
    _positive(temperature, "temperature")
    if student.shape != teacher.shape or student.ndim != 2:
        raise ValueError("student/teacher logit shapes differ")
    _finite(student, "student logits")
    _finite(teacher, "teacher logits")
    loss = student.sum() * 0.0
    used = set()
    for task in tasks:
        group = _ids(task, student.shape[1])
        if not group or used.intersection(group):
            raise ValueError("tasks must be nonempty and disjoint")
        used.update(group)
        if len(group) == 1:
            continue
        target = F.softmax(teacher.detach()[:, list(group)] / temperature, dim=1)
        predicted = F.log_softmax(student[:, list(group)] / temperature, dim=1)
        loss = loss + F.kl_div(predicted, target, reduction="batchmean") * temperature**2
    _finite(loss, "taskwise KD")
    return loss


def separated_softmax_loss(logits, labels, old_classes, new_classes):
    """Class-balanced CE in disjoint old/new spaces, weighted by group rows."""
    old = _ids(old_classes, logits.shape[1])
    new = _ids(new_classes, logits.shape[1])
    if set(old).intersection(new) or not new or not len(labels):
        raise ValueError("old/new class spaces must be disjoint with nonempty new data")
    if logits.ndim != 2 or len(logits) != len(labels):
        raise ValueError("invalid logits/labels")
    _finite(logits, "classification logits")
    _local_targets(labels, old + new)
    loss = logits.sum() * 0.0
    for group in (old, new):
        mask = _old_mask(labels, group)
        count = int(mask.sum())
        if count and len(group) > 1:
            targets = _local_targets(labels[mask], group)
            loss = loss + count / len(labels) * _class_balanced_cross_entropy(
                logits[mask][:, list(group)], targets)
    _finite(loss, "separated CE")
    return loss


def apply_wa(model, old_classes, new_classes):
    """Align new head weight norms; apply the same factor to its bias."""
    _check_model(model)
    old = _ids(old_classes, model.head.out_features)
    new = _ids(new_classes, model.head.out_features)
    if not new or set(old).intersection(new):
        raise ValueError("WA requires disjoint old/new class IDs")
    if not old:
        return {"gamma": 1.0, "applied": False, "reason": "first task", "bias_scaled": False}
    with torch.no_grad():
        old_norm = model.head.weight[list(old)].norm(p=2, dim=1).mean()
        new_norm = model.head.weight[list(new)].norm(p=2, dim=1).mean()
        _finite(old_norm, "WA old weight norm")
        _finite(new_norm, "WA new weight norm")
        if float(new_norm) <= 0:
            raise ValueError("WA new-class mean weight norm is zero")
        gamma = old_norm / new_norm
        _finite(gamma, "WA gamma")
        weight = model.head.weight[list(new)] * gamma
        _finite(weight, "WA aligned weight")
        if model.head.bias is not None:
            bias = model.head.bias[list(new)] * gamma
            _finite(bias, "WA aligned bias")
            model.head.bias[list(new)] = bias
        model.head.weight[list(new)] = weight
    _check_model(model)
    return {"gamma": float(gamma), "applied": True,
            "old_mean_weight_norm": float(old_norm), "new_mean_weight_norm": float(new_norm),
            "bias_scaled": model.head.bias is not None}


def fit_bic(model, val_x, val_y, seen, new, lr=0.01, steps=100,
            batch_size=256, device="cpu", seed=0):
    """Fit two scalars on held-out data, then fold the transform into the head.

    The caller owns train/holdout separation. Validation must contain an equal,
    nonzero number of examples for every seen class. No model parameter is in
    the calibration optimizer; the model is changed only by the final fold.
    """
    val_x, val_y = _xy(val_x, val_y)
    _positive(lr, "learning rate")
    if steps < 1 or batch_size < 1:
        raise ValueError("BiC requires positive steps and batch size")
    seen, new = _ids(seen, model.head.out_features), _ids(new, model.head.out_features)
    if not new or not set(new).issubset(seen):
        raise ValueError("BiC new classes must be a nonempty subset of seen")
    if model.head.bias is None:
        raise ValueError("BiC fold requires the shared MLP's bias parameter")
    classes, counts = np.unique(val_y, return_counts=True)
    if set(classes.tolist()) != set(seen) or len(set(counts.tolist())) != 1:
        raise ValueError("BiC requires balanced held-out examples for every seen class")
    _check_model(model)
    was_training = model.training
    model.eval()
    _sync(device)
    started = time.perf_counter()
    try:
        with torch.no_grad():
            logits = torch.cat([model(torch.from_numpy(val_x[i:i + batch_size]).to(device))
                                for i in range(0, len(val_y), batch_size)]).detach()
        _finite(logits, "held-out logits")
        targets = _local_targets(torch.from_numpy(val_y).to(device), seen)
        alpha = torch.nn.Parameter(logits.new_tensor(1.0))
        beta = torch.nn.Parameter(logits.new_tensor(0.0))
        optimizer = torch.optim.Adam([alpha, beta], lr=lr)
        mask = logits.new_zeros(logits.shape[1])
        mask[list(new)] = 1.0
        def corrected(z):
            return z * (1 + mask * (alpha - 1)) + mask * beta
        before = float(F.cross_entropy(logits[:, list(seen)], targets))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        order, cursor, examples = torch.randperm(len(val_y), generator=generator), 0, 0
        for _ in range(steps):
            if cursor == len(val_y):
                order, cursor = torch.randperm(len(val_y), generator=generator), 0
            indices = order[cursor:cursor + batch_size].to(device)
            cursor += len(indices)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(corrected(logits[indices])[:, list(seen)], targets[indices])
            _step(loss, optimizer, [alpha, beta])
            examples += len(indices)
        with torch.no_grad():
            after = float(F.cross_entropy(corrected(logits)[:, list(seen)], targets))
            _finite(after, "BiC final validation loss")
            weight = model.head.weight[list(new)] * alpha.detach()
            bias = model.head.bias[list(new)] * alpha.detach() + beta.detach()
            _finite(weight, "BiC folded weight")
            _finite(bias, "BiC folded bias")
            model.head.weight[list(new)] = weight
            model.head.bias[list(new)] = bias
        _check_model(model)
        _sync(device)
        return {"alpha": float(alpha.detach()), "beta": float(beta.detach()),
                "folded": True, "optimizer_steps": int(steps), "examples_presented": examples,
                "validation_rows": len(val_y), "validation_per_class": int(counts[0]),
                "validation_loss_before": before, "validation_loss_after": after,
                "training_seconds": time.perf_counter() - started}
    finally:
        model.train(was_training)


def _seen_tasks(tasks, seen):
    wanted = set(int(c) for c in seen)
    groups = [tuple(int(c) for c in task if int(c) in wanted) for task in tasks]
    groups = [group for group in groups if group]
    flat = [c for group in groups for c in group]
    if set(flat) != wanted or len(flat) != len(wanted):
        raise ValueError("task partition does not cover class space exactly once")
    return groups


def balanced_finetune(model, current_x, current_y, old_x, old_y, tasks,
                      lr, epochs, batch_size, seed, device):
    """Random balanced subset, low-LR Adam, full CE plus taskwise KD (T=2)."""
    current_x, current_y = _xy(current_x, current_y)
    old_x, old_y = _xy(old_x, old_y, allow_empty=True)
    _positive(lr, "learning rate")
    if epochs < 1 or batch_size < 1 or current_x.shape[1] != old_x.shape[1]:
        raise ValueError("invalid fine-tuning budget or feature shape")
    current_classes, current_counts = np.unique(current_y, return_counts=True)
    old_classes, old_counts = np.unique(old_y, return_counts=True)
    if set(current_classes.tolist()).intersection(old_classes.tolist()):
        raise ValueError("current classes overlap old memory classes")
    available = dict(zip(current_classes.tolist(), current_counts.tolist()))
    available.update(zip(old_classes.tolist(), old_counts.tolist()))
    classes = sorted(available)
    _ids(classes, model.head.out_features)
    per_class = min(200, min(available.values()))
    rng = np.random.default_rng(seed)
    selected_x, selected_y = [], []
    for c in classes:
        source_x, source_y = (current_x, current_y) if c in current_classes else (old_x, old_y)
        selected = rng.choice(np.flatnonzero(source_y == c), per_class, replace=False)
        selected_x.append(source_x[selected].copy())
        selected_y.append(np.full(per_class, c, dtype=np.int64))
    x, y = np.concatenate(selected_x), np.concatenate(selected_y)
    groups = _seen_tasks(tasks, classes)
    teacher = copy.deepcopy(model).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    _check_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stats, ce_total, kd_total = {}, 0.0, 0.0
    _sync(device)
    started = time.perf_counter()
    for _, batches in training_epoch_batches(x, y, batch_size, seed, epochs, 0, stats):
        model.train()
        for bx, by in batches:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx)
            with torch.no_grad():
                target = teacher(bx)
            ce = _class_balanced_cross_entropy(logits, by)
            kd = taskwise_kd(logits, target, groups, 2.0)
            _step(ce + kd, optimizer, model.parameters())
            ce_total += float(ce.detach())
            kd_total += float(kd.detach())
    _sync(device)
    stats.update(training_seconds=time.perf_counter() - started, balanced_rows=len(y),
                 selected_per_class={str(int(c)): per_class for c in classes},
                 selection="random without replacement from current rows and stored old memory",
                 teacher="post-base-training snapshot", distilled_tasks=[list(g) for g in groups],
                 learning_rate=float(lr), temperature=2.0,
                 ce_loss_mean=ce_total / stats["optimizer_steps"],
                 task_kd_loss_mean=kd_total / stats["optimizer_steps"])
    return stats


def _buffer_arrays(buffer_x, buffer_y, input_dim):
    if isinstance(buffer_x, np.ndarray):
        return _xy(buffer_x, buffer_y, allow_empty=True)
    if not buffer_x:
        return np.empty((0, input_dim), np.float32), np.empty(0, np.int64)
    return _xy(np.concatenate(buffer_x), np.concatenate(buffer_y), allow_empty=True)


def _base_auxiliary_loss(model, teacher, bx, by, logits, teacher_logits,
                         base_method, old_classes, seen_classes, config, regularization, rng):
    loss = logits.sum() * 0.0
    if teacher is None or not old_classes or base_method == "replay":
        return loss
    temperature = float(config["temperature"])
    if base_method == "replay_lwf":
        return regularization * _clean_prediction_kd(logits, teacher_logits, old_classes, temperature)
    if base_method != "rcl_tab":
        raise ValueError("unsupported base auxiliary loss")
    old_rows = _old_mask(by, old_classes)
    if not bool(old_rows.any()):
        return loss
    student_deltas, teacher_deltas = [], []
    features = rng.choice(bx.shape[1], min(int(config["rcl_features"]), bx.shape[1]), replace=False)
    for feature in features:
        changed = bx.clone()
        changed[:, int(feature)] = changed[torch.randperm(len(bx), device=bx.device), int(feature)]
        student_changed = model(changed)
        with torch.no_grad():
            teacher_changed = teacher(changed)
        student_deltas.append(_distribution_change(logits, student_changed, by, seen_classes,
                                                   old_rows, float(config["rcl_beta"])))
        teacher_deltas.append(_distribution_change(teacher_logits, teacher_changed, by, old_classes,
                                                   old_rows, float(config["rcl_beta"])))
    reliance = _rcl_profile_loss(torch.stack(student_deltas, dim=1), torch.stack(teacher_deltas, dim=1),
                                 teacher_logits, old_classes, old_rows,
                                 reliance_temperature=float(config["rcl_temperature"]), gated=True)
    prediction = _clean_prediction_kd(logits, teacher_logits, old_classes, temperature)
    return float(config["rcl_prediction_weight"]) * prediction + regularization * reliance


def train_ssil(model, current_x, current_y, buffer_x, buffer_y, memory, base_method,
               teacher, old_classes, seen_classes, tasks, config, regularization,
               task_seed, device, training_stats=None, tkd_weight=1.0):
    """Separated CE + old-task KD, with retained base auxiliaries and extra old rows.

    Budget rows are current+stored-memory for non-DER bases and current only for
    DER++. The additional old minibatch never adds optimizer steps. DER++ keeps
    two independent auxiliary memory draws and admits only current stream rows.
    """
    if base_method not in {"replay", "replay_lwf", "rcl_tab", "derpp"}:
        raise ValueError("unsupported SS-IL base")
    current_x, current_y = _xy(current_x, current_y)
    old_classes = _ids(old_classes, model.head.out_features)
    seen_classes = _ids(seen_classes, model.head.out_features)
    if not set(old_classes).issubset(seen_classes):
        raise ValueError("old classes are not seen classes")
    new_classes = tuple(c for c in seen_classes if c not in old_classes)
    if set(np.unique(current_y).tolist()) != set(new_classes):
        raise ValueError("current rows must cover exactly the new classes")
    if old_classes and teacher is None:
        raise ValueError("SS-IL requires a previous-task teacher when old classes exist")
    _positive(tkd_weight, "TKD weight")
    _positive(config["learning_rate"], "learning rate")
    extra_size = int(config.get("ssil_old_replay_batch_size", 64))
    if extra_size < 1:
        raise ValueError("SS-IL old replay batch size must be positive")
    if int(config["epochs_per_task"]) < 1:
        raise ValueError("SS-IL requires at least one full base epoch")
    if base_method == "derpp":
        if memory is None:
            raise ValueError("DER++ requires its class reservoir")
        xs, ys = [], []
        for c in old_classes:
            count = min(memory.per_class, memory.observations[c])
            if count:
                xs.append(memory.x[c, :count].numpy().copy())
                ys.append(np.full(count, c, dtype=np.int64))
        old_x, old_y = _buffer_arrays(xs, ys, current_x.shape[1])
        x, y = current_x, current_y
    else:
        old_x, old_y = _buffer_arrays(buffer_x, buffer_y, current_x.shape[1])
        x, y = np.concatenate([current_x, old_x]), np.concatenate([current_y, old_y])
    if set(np.unique(old_y).tolist()) != set(old_classes):
        raise ValueError("stored memory must cover exactly the old classes")
    old_tasks = _seen_tasks(tasks, old_classes)
    _check_model(model)
    if teacher is not None:
        teacher.eval()
        _check_model(teacher)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    base_rng = np.random.default_rng(task_seed + 319)
    der_rng = np.random.default_rng(task_seed + 8317)
    extra_rng = np.random.default_rng(task_seed + 23023)
    stats = training_stats if training_stats is not None else {}
    extra_count = current_count = base_old_count = mse_count = replay_ce_count = 0
    ce_total = kd_total = auxiliary_total = 0.0
    _sync(device)
    started = time.perf_counter()
    for _, batches in training_epoch_batches(x, y, int(config["batch_size"]), task_seed,
            int(config["epochs_per_task"]), int(config.get("min_steps_per_task", 0)), stats):
        model.train()
        for bx, by in batches:
            bx, by = bx.to(device), by.to(device)
            base_n = len(by)
            base_old_count += int(_old_mask(by, old_classes).sum())
            current_count += int(_old_mask(by, new_classes).sum())
            if len(old_y):
                take = extra_rng.choice(len(old_y), extra_size, replace=len(old_y) < extra_size)
                all_x = torch.cat([bx, torch.from_numpy(old_x[take]).to(device)])
                all_y = torch.cat([by, torch.from_numpy(old_y[take]).to(device)])
                extra_count += len(take)
            else:
                all_x, all_y = bx, by
            optimizer.zero_grad(set_to_none=True)
            logits = model(all_x)
            ce = separated_softmax_loss(logits, all_y, old_classes, new_classes)
            if teacher is not None and old_classes:
                with torch.no_grad():
                    teacher_logits = teacher(all_x)
                kd = taskwise_kd(logits, teacher_logits, old_tasks, 2.0)
            else:
                teacher_logits = None
                kd = logits.sum() * 0.0
            auxiliary = logits.sum() * 0.0
            if base_method == "derpp":
                if len(memory):
                    mx, _, mz = memory.sample(int(config["batch_size"]), der_rng, device)
                    auxiliary = auxiliary + float(config["derpp_alpha"]) * F.mse_loss(model(mx), mz)
                    mse_count += len(mx)
                    mx, my, _ = memory.sample(int(config["batch_size"]), der_rng, device)
                    auxiliary = auxiliary + float(config["derpp_beta"]) * _class_balanced_cross_entropy(model(mx), my)
                    replay_ce_count += len(mx)
            else:
                auxiliary = _base_auxiliary_loss(model, teacher, bx, by, logits[:base_n],
                    teacher_logits[:base_n] if teacher_logits is not None else None,
                    base_method, old_classes, seen_classes, config, regularization, base_rng)
            _step(ce + float(tkd_weight) * kd + auxiliary, optimizer, model.parameters())
            if base_method == "derpp":
                memory.add(bx, by, logits[:base_n], der_rng)
            ce_total += float(ce.detach())
            kd_total += float(kd.detach())
            auxiliary_total += float(auxiliary.detach())
    _sync(device)
    steps = stats["optimizer_steps"]
    stats.update(training_seconds=time.perf_counter() - started, base_method=base_method,
                 current_examples_presented=current_count, base_old_examples_presented=base_old_count,
                 ssil_extra_old_examples_presented=extra_count,
                 ssil_old_batch_size=extra_size, ssil_old_sampling_replacement=0 < len(old_y) < extra_size,
                 base_aux_mse_examples_presented=mse_count, base_aux_ce_examples_presented=replay_ce_count,
                 optimization_input_presentations=stats["examples_presented"] + extra_count + mse_count + replay_ce_count,
                 task_kd_weight=float(tkd_weight), task_kd_temperature=2.0,
                 distilled_tasks=[list(g) for g in old_tasks], separated_ce_loss_mean=ce_total / steps,
                 task_kd_loss_mean=kd_total / steps, base_auxiliary_loss_mean=auxiliary_total / steps)
    return stats
