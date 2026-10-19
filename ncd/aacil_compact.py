"""Fixed-capacity attack-family candidates for the compact retention protocol.

All output rows are allocated at initialization; unseen rows are masked from
training CE and inference. EMA-KD is a one-teacher simplification inspired by
Arani, Sarfraz and Zonooz (ICLR 2022), not a CLS-ER reproduction (the original
uses two semantic memories). DUAL has two fixed encoders and one joint head;
its stable-logit loss cannot backpropagate through the plastic encoder.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ncd import run_aacil as aa
from ncd.crr_benchmarks import _class_balanced_cross_entropy, training_epoch_batches
from ncd.model import MLP
from ncd.run_aacil_retention import assert_schedule, check_deadline, logits_numpy


METHODS = ("aacil_kd_weak", "aacil_kd_wa", "aacil_kd_cbf", "aacil_ema_kd", "aacil_dual")


class CompactFamily(nn.Module):
    def __init__(self, input_dim, hidden, n_classes, method, ema_decay=0.999):
        super().__init__()
        if method not in METHODS:
            raise ValueError("Unknown fixed-capacity family method")
        self.method, self.hidden = method, int(hidden)
        self.ema_decay = float(ema_decay)
        self.reference_parameters = (input_dim * hidden + hidden + hidden * hidden + hidden
                                     + hidden * n_classes + n_classes)
        if method == "aacil_dual":
            self.stable = MLP(input_dim, hidden, n_classes).features
            self.plastic = MLP(input_dim, hidden, n_classes).features
            self.classifier = nn.Linear(2 * hidden, n_classes)
        else:
            self.student = MLP(input_dim, hidden, n_classes)
            if method == "aacil_ema_kd":
                self.ema = copy.deepcopy(self.student).requires_grad_(False)
        self.register_buffer("seen_mask", torch.zeros(n_classes, dtype=torch.bool))
        self.initial_parameter_count = sum(p.numel() for p in self.parameters())
        self.assert_capacity()

    @property
    def head(self):
        return self.classifier if self.method == "aacil_dual" else self.student.head

    def set_seen(self, columns):
        columns = sorted(set(int(c) for c in columns))
        if not columns or min(columns) < 0 or max(columns) >= len(self.seen_mask):
            raise ValueError("Invalid seen attack columns")
        next_mask = torch.zeros_like(self.seen_mask)
        next_mask[columns] = True
        if torch.any(self.seen_mask & ~next_mask):
            raise ValueError("A previously seen attack output disappeared")
        self.seen_mask.copy_(next_mask)

    def mask(self, logits):
        return logits.masked_fill(~self.seen_mask[None, :], -1.0e6)

    def student_logits(self, x):
        if self.method == "aacil_dual":
            return self.classifier(torch.cat([self.stable(x), self.plastic(x)], dim=1))
        return self.student(x)

    def stable_logits(self, x):
        if self.method != "aacil_dual":
            raise ValueError("Stable contribution is defined only for DUAL")
        # Joint head stable block and bias only: no plastic features or gradients.
        return F.linear(self.stable(x), self.classifier.weight[:, :self.hidden], self.classifier.bias)

    def entry_logits(self, x):
        return self.stable_logits(x) if self.method == "aacil_dual" else self.student_logits(x)

    def forward(self, x):
        # Evaluation and exported decisions use exactly one F. EMA student is
        # deliberately not ensembled with the teacher at inference.
        logits = self.ema(x) if self.method == "aacil_ema_kd" else self.student_logits(x)
        return self.mask(logits)

    def update_ema(self):
        if self.method != "aacil_ema_kd":
            return
        with torch.no_grad():
            for teacher, student in zip(self.ema.parameters(), self.student.parameters()):
                teacher.mul_(self.ema_decay).add_(student, alpha=1.0 - self.ema_decay)

    def assert_capacity(self):
        count = sum(p.numel() for p in self.parameters())
        if count != self.initial_parameter_count or count > 2 * self.reference_parameters:
            raise AssertionError("F parameter count changed or exceeded twice the reference F")
        return count

    def metadata(self):
        total = self.assert_capacity()
        inference = self.reference_parameters if self.method == "aacil_ema_kd" else total
        return {
            "family_type": self.method, "total_parameters": total,
            "training_parameters": total, "optimizer_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "inference_parameters": inference, "reference_full_attack_head_parameters": self.reference_parameters,
            "parameter_ratio_training": total / self.reference_parameters,
            "parameter_ratio_inference": inference / self.reference_parameters,
            "extra_components": int(self.method in ("aacil_ema_kd", "aacil_dual")),
            "constant_across_tasks": True, "output_attack_classes": self.head.out_features,
            "seen_attack_columns": torch.where(self.seen_mask)[0].cpu().tolist(),
            "head_policy": "all eventual attack rows allocated at task zero; unseen rows masked from CE and inference",
            "inference_policy": "EMA copy only; no student ensemble" if self.method == "aacil_ema_kd" else "one joint attack-family F",
            "dual_budget": "two full-width encoders plus one concatenation head = 2*reference_F - output_bias_count" if self.method == "aacil_dual" else None,
            "teacher_transient_model_copies": 0,
        }


def train_compact_family(model, x, y, old_memory, mapping, config, seed, device):
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=float(config["learning_rate"]))
    rng, stats = np.random.default_rng(seed + 8317), {}
    slots = [(c, j) for c, item in sorted(old_memory.items()) for j in range(len(item["train_ids"]))]

    def sample():
        picked = rng.choice(len(slots), min(int(config["batch_size"]), len(slots)), replace=False)
        rows = [slots[int(i)] for i in picked]
        bx = np.stack([old_memory[c]["train_x"][j] for c, j in rows])
        by = np.asarray([mapping[c] for c, _ in rows], dtype=np.int64)
        target = np.stack([old_memory[c]["f_logits"][j] for c, j in rows])
        mask = np.stack([old_memory[c]["f_target_mask"][j] for c, j in rows])
        return tuple(torch.from_numpy(v).to(device) for v in (bx, by, target, mask))

    aa.sync(device)
    started = time.perf_counter()
    for epoch, batches in training_epoch_batches(x, y, int(config["batch_size"]), seed,
                                               int(config["epochs_per_task"]), int(config["min_steps_per_task"]), stats):
        model.train()
        for bx, by in batches:
            check_deadline(config)
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _class_balanced_cross_entropy(model.mask(model.student_logits(bx)), by)
            if slots:
                mx, _, mz, mask = sample()
                if model.method == "aacil_ema_kd":
                    with torch.no_grad():
                        target = model.ema(mx)
                    current = model.student_logits(mx)
                    loss = loss + float(config["ema_consistency"]) * F.mse_loss(current[:, model.seen_mask], target[:, model.seen_mask])
                else:
                    current = model.entry_logits(mx)
                    if not torch.any(mask):
                        raise ValueError("Retained logit target has no valid seen-at-admission columns")
                    loss = loss + float(config["derpp_alpha"]) * F.mse_loss(current[mask], mz[mask])
                    mx, my, _, _ = sample()
                    replay_logits = model.stable_logits(mx) if model.method == "aacil_dual" else model.student_logits(mx)
                    loss = loss + float(config["derpp_beta"]) * _class_balanced_cross_entropy(model.mask(replay_logits), my)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite compact-family loss")
            loss.backward()
            optimizer.step()
            model.update_ema()
        print(f"F_COMPACT_EPOCH_DONE method={model.method} epoch={epoch + 1} steps={stats['optimizer_steps']} old_memory={len(slots)}", flush=True)
    aa.sync(device)
    elapsed = time.perf_counter() - started
    assert_schedule(stats, len(y), config)
    model.assert_capacity()
    stats.update(output_classes_allocated=model.head.out_features, output_classes_in_ce=int(model.seen_mask.sum()),
                 single_seen_output_ce_is_degenerate=int(model.seen_mask.sum()) == 1,
                 old_replay_rows=len(slots), derpp_alpha=config["derpp_alpha"], derpp_beta=config["derpp_beta"],
                 ema_decay=config["ema_decay"] if model.method == "aacil_ema_kd" else None,
                 ema_consistency=config["ema_consistency"] if model.method == "aacil_ema_kd" else None,
                 masked_mse_policy="only output columns seen when retained row entered memory; raw finite logits",
                 dual_distillation_scope="stable features and stable block of joint head only; plastic branch receives CE only" if model.method == "aacil_dual" else None)
    return elapsed, stats


def attach_compact_logits(old, memory, new_attacks, gate, family, device, batch_size):
    aa.sync(device)
    started, forward_seconds = time.perf_counter(), 0.0
    for c, item in memory.items():
        if c in new_attacks:
            aa.sync(device)
            forward_start = time.perf_counter()
            item["g_logits"] = logits_numpy(gate, item["train_x"], device, batch_size)
            family.eval()
            parts = []
            if family.method != "aacil_ema_kd":
                with torch.no_grad():
                    for start in range(0, len(item["train_x"]), batch_size):
                        bx = torch.from_numpy(np.asarray(item["train_x"][start:start + batch_size])).to(device)
                        parts.append(family.entry_logits(bx).cpu().numpy())
                item["f_logits"] = np.concatenate(parts) if parts else np.empty((0, family.head.out_features), dtype=np.float32)
                item["f_target_mask"] = np.broadcast_to(family.seen_mask.cpu().numpy(), item["f_logits"].shape).copy()
            else:
                item["f_logits"] = np.empty((len(item["train_ids"]), 0), dtype=np.float32)
                item["f_target_mask"] = np.empty((len(item["train_ids"]), 0), dtype=bool)
            aa.sync(device)
            forward_seconds += time.perf_counter() - forward_start
        else:
            lookup = {int(row): j for j, row in enumerate(old[c]["train_ids"])}
            positions = np.asarray([lookup[int(row)] for row in item["train_ids"]], dtype=np.int64)
            for key in ("g_logits", "f_logits", "f_target_mask"):
                item[key] = old[c][key][positions].copy()
        if any(len(item[key]) != len(item["train_ids"]) for key in ("g_logits", "f_logits", "f_target_mask")):
            raise AssertionError("Retained logit target/row correspondence changed")
        if not np.isfinite(item["f_logits"]).all():
            raise ValueError("Nonfinite entry logits")
    aa.sync(device)
    return forward_seconds, max(0.0, time.perf_counter() - started - forward_seconds)


def compact_balanced_finetune(model, memory, mapping, config, seed, device):
    """One 0.1-LR epoch using retained TRAIN only; no test/VAL/pool refill."""
    started = time.perf_counter()
    nonempty = {c: item for c, item in memory.items() if len(item["train_ids"])}
    quota = min((len(item["train_ids"]) for item in nonempty.values()), default=0)
    xs, ys, selected = [], [], {}
    for c, item in sorted(nonempty.items()):
        pos = np.random.default_rng(seed + 10007 * (mapping[c] + 1)).permutation(len(item["train_ids"]))[:quota]
        pos.sort()
        xs.append(item["train_x"][pos].copy())
        ys.append(np.full(quota, mapping[c], dtype=np.int64))
        selected[str(c)] = item["train_ids"][pos].tolist()
    selection_seconds = time.perf_counter() - started
    stats = {"method": "retained_train_balanced_finetuning", "epochs": 1, "learning_rate_multiplier": 0.1,
             "balanced_rows_per_nonempty_class": quota, "selected_train_row_ids": selected,
             "selection_seconds": selection_seconds, "validation_rows_used": 0, "old_pool_access": False,
             "optimizer_steps": 0, "training_seconds": 0.0}
    if not xs:
        return 0.0, stats
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1 * float(config["learning_rate"]))
    aa.sync(device)
    start = time.perf_counter()
    model.train()
    for _, batches in training_epoch_batches(np.concatenate(xs), np.concatenate(ys), int(config["batch_size"]), seed, 1, 0, stats):
        for bx, by in batches:
            check_deadline(config)
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _class_balanced_cross_entropy(model.mask(model.student_logits(bx)), by)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite compact CBF loss")
            loss.backward()
            optimizer.step()
    aa.sync(device)
    elapsed = time.perf_counter() - start
    stats["training_seconds"] = elapsed
    model.assert_capacity()
    return elapsed, stats
