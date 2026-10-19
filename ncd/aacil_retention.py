"""Section29.8 post-task F controls for the existing tabular ncd MLP.

These helpers do not train the gate, access a dataset pool, or launch a job.
WA/CBF/NCM share a plain-Replay family-training stage owned by the caller.
Memory operations consume retained attack TRAIN arrays only, never validation
rows. They are adaptations, not exact reproductions of the cited vision runs.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ncd.crr_benchmarks import _train
from ncd.model import MLP


def _sync(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _columns(values, width):
    columns = tuple(sorted(int(c) for c in values))
    if len(columns) != len(set(columns)) or any(c < 0 or c >= width for c in columns):
        raise ValueError("Invalid or duplicate attack-only output columns")
    return columns


def _retained_train(memory, mapping):
    """Only read train_x/train_ids; no validation or original-pool access."""
    result = {}
    for class_id, item in sorted(memory.items(), key=lambda pair: int(pair[0])):
        class_id = int(class_id)
        if class_id not in mapping:
            raise ValueError("Memory contains a class outside the attack mapping")
        column = int(mapping[class_id])
        if column in result:
            raise ValueError("Attack mapping is not one-to-one")
        x = np.asarray(item["train_x"])
        ids = np.asarray(item["train_ids"], dtype=np.int64)
        if x.ndim != 2 or ids.ndim != 1 or len(x) != len(ids):
            raise ValueError("Retained attack TRAIN features/IDs are not aligned")
        result[column] = (x, ids, class_id)
    return result


def apply_weight_alignment(model: MLP, old_columns, new_columns) -> dict:
    """Scale new head weight rows by mean old/new weight-norm ratio.

    Biases are deliberately unchanged, as are old and future weight rows. The
    first task has no old reference and is recorded as an unapplied correction.
    """
    if not isinstance(model.head, nn.Linear):
        raise TypeError("WA expects the ncd MLP linear family head")
    old = _columns(old_columns, model.head.out_features)
    new = _columns(new_columns, model.head.out_features)
    if set(old).intersection(new):
        raise ValueError("WA old and new attack sets overlap")
    device = model.head.weight.device
    _sync(device)
    started = time.perf_counter()
    stats = {
        "method": "weight_alignment", "old_attack_columns": list(old),
        "new_attack_columns": list(new), "bias_treatment": "unchanged",
        "formula": "new W rows *= mean_old_L2_norm / mean_new_L2_norm",
        "applied": False, "gamma": 1.0, "optimizer_steps": 0,
    }
    if not old or not new:
        stats["reason"] = "no old or no new attack class; no alignment reference"
    else:
        with torch.no_grad():
            old_norms = model.head.weight[list(old)].norm(p=2, dim=1)
            new_norms = model.head.weight[list(new)].norm(p=2, dim=1)
            old_mean, new_mean = old_norms.mean(), new_norms.mean()
            if not torch.isfinite(old_norms).all() or not torch.isfinite(new_norms).all():
                raise ValueError("Nonfinite WA weight norms")
            if float(new_mean.item()) <= 0:
                raise ValueError("WA is undefined for zero mean new-class weight norm")
            gamma = old_mean / new_mean
            model.head.weight[list(new)] = model.head.weight[list(new)] * gamma
            stats.update(applied=True, gamma=float(gamma.item()),
                         old_norms_before=old_norms.cpu().tolist(),
                         new_norms_before=new_norms.cpu().tolist(),
                         old_mean_norm=float(old_mean.item()),
                         new_mean_norm_before=float(new_mean.item()),
                         new_mean_norm_after=float(model.head.weight[list(new)].norm(p=2, dim=1).mean().item()))
    _sync(device)
    stats["elapsed_seconds"] = time.perf_counter() - started
    stats["cost_scope"] = "post-task alignment only; no extra training"
    return stats


def class_balanced_finetune(model: MLP, memory, mapping, config, seed, device,
                           epochs=None):
    """CBF on a deterministic equal-size subset of retained attack TRAIN rows.

    Returns optimizer time only. Selection time and total helper time are
    separately recorded; do not add the total again to its subcomponents.
    Empty retained classes are disclosed and never replenished from old data.
    """
    epochs = int(config["cbf_epochs"] if epochs is None else epochs)
    multiplier = float(config.get("cbf_lr_multiplier", 0.1))
    if epochs not in (1, 5) or not math.isclose(multiplier, 0.1, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Registered CBF uses epochs in {1,5} and 0.1 times base LR")
    device = torch.device(device)
    _sync(device)
    started = time.perf_counter()
    retained = _retained_train(memory, mapping)
    nonempty = {c: values for c, values in retained.items() if len(values[0])}
    empty = sorted(set(retained) - set(nonempty))
    quota = min((len(values[0]) for values in nonempty.values()), default=0)
    chosen_ids, xs, ys = {}, [], []
    for column, (x, ids, _) in sorted(nonempty.items()):
        positions = np.random.default_rng(int(seed) + 10007 * (column + 1)).permutation(len(ids))[:quota]
        positions.sort()
        xs.append(x[positions].copy())
        ys.append(np.full(quota, column, dtype=np.int64))
        chosen_ids[str(column)] = ids[positions].tolist()
    selection_seconds = time.perf_counter() - started
    stats = {
        "method": "class_balanced_finetuning", "epochs": epochs,
        "learning_rate": float(config["learning_rate"]) * multiplier,
        "learning_rate_multiplier": multiplier,
        "retained_train_counts": {str(c): len(values[0]) for c, values in retained.items()},
        "balanced_rows_per_nonempty_class": quota,
        "balanced_training_rows": quota * len(nonempty),
        "selected_train_row_ids": chosen_ids,
        "empty_retained_attack_columns": empty,
        "balance_scope": "equal row counts among nonempty retained attack TRAIN classes",
        "validation_rows_used": 0, "old_pool_access": False,
        "additional_minimum_steps": 0, "selection_seconds": selection_seconds,
        "training_seconds": 0.0, "optimizer_steps": 0,
    }
    if not nonempty:
        stats.update(skipped=True, reason="no retained attack TRAIN rows",
                     elapsed_seconds=time.perf_counter() - started)
        return 0.0, stats
    x, y = np.concatenate(xs), np.concatenate(ys)
    train_stats = {}
    _sync(device)
    training_started = time.perf_counter()
    _train(model, x, y, method="replay", old_model=None, old_classes=(),
           seen_classes=tuple(range(model.head.out_features)), epochs=epochs,
           batch_size=int(config["batch_size"]), learning_rate=stats["learning_rate"],
           seed=int(seed), regularization=0.0,
           temperature=float(config.get("temperature", 2.0)),
           min_steps=0, training_stats=train_stats)
    _sync(device)
    training_seconds = time.perf_counter() - training_started
    stats.update(skipped=False, training_seconds=training_seconds,
                 optimizer_steps=train_stats["optimizer_steps"],
                 schedule=train_stats, elapsed_seconds=time.perf_counter() - started,
                 output_classes_in_ce=model.head.out_features,
                 ce_scope="unchanged plain-Replay full attack head; minibatch class-balanced CE",
                 cost_scope="selection and training are disjoint; elapsed includes both and is not additive")
    return training_seconds, stats


def build_ncm_prototypes(model: MLP, memory, mapping, device, batch_size):
    """Current-encoder means from retained attack TRAIN only, with L2 geometry."""
    device = torch.device(device)
    if int(batch_size) <= 0:
        raise ValueError("NCM prototype batch size must be positive")
    _sync(device)
    started = time.perf_counter()
    retained = _retained_train(memory, mapping)
    prototypes, empty, zero_norm = {}, [], []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for column, (x, _, _) in sorted(retained.items()):
                if len(x) == 0:
                    empty.append(column)
                    continue
                total = None
                for start in range(0, len(x), int(batch_size)):
                    bx = torch.from_numpy(np.asarray(x[start:start + int(batch_size)])).to(device)
                    features = F.normalize(model.features(bx), p=2, dim=1)
                    if not torch.isfinite(features).all():
                        raise ValueError("Nonfinite NCM retained-row features")
                    value = features.sum(dim=0)
                    total = value if total is None else total + value
                mean = total / len(x)
                if float(mean.norm().item()) == 0.0:
                    zero_norm.append(column)
                prototypes[column] = F.normalize(mean, p=2, dim=0).detach().cpu()
    finally:
        model.train(was_training)
    _sync(device)
    elapsed = time.perf_counter() - started
    stats = {
        "method": "nearest_class_mean", "elapsed_seconds": elapsed,
        "optimizer_steps": 0, "prototype_columns": sorted(prototypes),
        "retained_train_counts": {str(c): len(values[0]) for c, values in retained.items()},
        "empty_retained_attack_columns": empty, "zero_norm_actual_means": zero_norm,
        "unavailable_output_columns": sorted(set(range(model.head.out_features)) - set(prototypes)),
        "validation_rows_used": 0, "old_pool_access": False,
        "prototype_rule": "mean of per-row L2-normalized current features, then L2-normalize mean",
        "score": "negative squared Euclidean distance between normalized features and prototypes",
        "missing_class_rule": "no invented prototype; unavailable output score -1e6; all missing fails",
        "prototype_bytes": sum(p.numel() * p.element_size() for p in prototypes.values()),
        "cost_scope": "post-task prototype feature inference and construction, not training",
    }
    return prototypes, stats


class NCMAttackScores(nn.Module):
    """Wrap the still-trainable MLP for NCM evaluation; no router or new model."""
    def __init__(self, model: MLP, prototypes: dict[int, torch.Tensor]):
        super().__init__()
        if not prototypes:
            raise ValueError("NCM inference is unavailable without any retained prototype")
        columns = _columns(prototypes.keys(), model.head.out_features)
        self.model = model
        device = next(model.parameters()).device
        matrix = torch.stack([prototypes[c].detach() for c in columns]).to(device)
        if matrix.ndim != 2 or matrix.shape[1] != model.head.in_features or not torch.isfinite(matrix).all():
            raise ValueError("Invalid NCM prototype dimensions or values")
        self.register_buffer("prototype_columns", torch.tensor(columns, dtype=torch.long, device=device))
        self.register_buffer("prototype_matrix", matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = F.normalize(self.model.features(x), p=2, dim=1)
        scores = -torch.sum((features[:, None, :] - self.prototype_matrix[None, :, :]) ** 2, dim=2)
        full = features.new_full((len(x), self.model.head.out_features), -1.0e6)
        return full.index_copy(1, self.prototype_columns, scores)
