"""Section29.6 attack-family isolation; no gate, pool access, or launcher.

The caller supplies current attack rows plus legal retained attack TRAIN rows.
One task has one F optimizer/step budget, shared by all newly introduced heads.
The first attack task jointly trains the encoder and those heads; afterwards
the encoder and every completed head are permanently frozen. This is a tabular
MLP adaptation, not a reproduction of historical OFRA experiments.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn

from ncd.crr_benchmarks import (
    _class_balanced_cross_entropy, training_epoch_batches,
)
from ncd.model import MLP


class _FamilyHead(nn.Module):
    def __init__(self, hidden: int, rank: int):
        super().__init__()
        self.down = nn.Linear(hidden, rank, bias=False)
        self.up = nn.Linear(rank, hidden, bias=False)
        nn.init.zeros_(self.up.weight)
        self.binary = nn.Linear(hidden, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.binary(features + self.up(self.down(features)))


class IsolatedAttackFamily(nn.Module):
    """Full attack-column scores from independent positive probabilities.

    Labels and head IDs are the caller's fixed zero-based attack-only mapping.
    The caller must still restrict inference to observed attack columns. State
    consists entirely of tensors, including freeze flags and head membership,
    so existing checkpoint detach/cpu serialization remains compatible.
    """

    def __init__(self, input_dim: int, hidden: int, n_attacks: int,
                 rank: int = 8, seed: int | None = None):
        super().__init__()
        if input_dim <= 0 or hidden <= 0 or n_attacks <= 0 or rank != 8:
            raise ValueError("ISO requires positive dimensions and fixed rank 8")
        self.hidden, self.n_attacks, self.rank = hidden, n_attacks, rank
        # CPU-only RNG context keeps family construction from changing G RNG.
        # If no seed is supplied, use the caller's already-seeded CPU state.
        with torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.default_generator.manual_seed(int(seed))
            self.encoder = MLP(input_dim, hidden, n_attacks).features
        self.family_heads = nn.ModuleDict()
        self.register_buffer("encoder_ready", torch.tensor(False))
        self.register_buffer("trained_heads", torch.zeros(n_attacks, dtype=torch.bool))
        self.register_buffer("head_initialization_seeds", torch.full((n_attacks,), -1, dtype=torch.long))
        self.register_buffer("encoder_initialization_seed", torch.tensor(-1 if seed is None else int(seed), dtype=torch.long))
        self.register_buffer("tasks_completed", torch.tensor(0, dtype=torch.long))

    def add_head(self, column: int, seed: int) -> None:
        column = int(column)
        if not 0 <= column < self.n_attacks or str(column) in self.family_heads:
            raise ValueError("ISO head already exists or is outside the attack mapping")
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(int(seed))
            head = _FamilyHead(self.hidden, self.rank)
        self.family_heads[str(column)] = head.to(next(self.encoder.parameters()).device)
        self.head_initialization_seeds[column] = int(seed)

    def metadata(self) -> dict:
        """Checkpoint-friendly descriptive state; does not run inference."""
        return {
            "family_type": "isolated_attack_heads", "rank": self.rank,
            "hidden": self.hidden, "n_attack_columns": self.n_attacks,
            "encoder_ready": bool(self.encoder_ready.item()),
            "encoder_initialization_seed": int(self.encoder_initialization_seed.item()),
            "encoder_initialization_seed_minus_one_means": "caller-seeded CPU RNG state",
            "encoder_frozen": not any(p.requires_grad for p in self.encoder.parameters()),
            "encoder_parameters": sum(p.numel() for p in self.encoder.parameters()),
            "tasks_completed": int(self.tasks_completed.item()),
            "heads": {name: {
                "attack_column": int(name), "rank": self.rank,
                "initialization_seed": int(self.head_initialization_seeds[int(name)].item()),
                "completed": bool(self.trained_heads[int(name)].item()),
                "frozen": not any(p.requires_grad for p in head.parameters()),
                "parameters": sum(p.numel() for p in head.parameters()),
            } for name, head in self.family_heads.items()},
            "score": "independent binary-head positive softmax probability",
            "router": False,
            "total_parameters": sum(p.numel() for p in self.parameters()),
        }

    def restore_freeze_flags(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(not bool(self.encoder_ready.item()))
        for name, head in self.family_heads.items():
            for parameter in head.parameters():
                parameter.requires_grad_(not bool(self.trained_heads[int(name)].item()))

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        columns = sorted({int(key.split(".")[1]) for key in state_dict
                          if key.startswith("family_heads.")})
        for column in columns:
            if str(column) not in self.family_heads:
                self.add_head(column, seed=0)
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.restore_freeze_flags()
        return result

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.family_heads:
            raise RuntimeError("ISO inference requires at least one trained head")
        features = self.encoder(inputs)
        scores = features.new_zeros((len(inputs), self.n_attacks))
        for name, head in self.family_heads.items():
            scores[:, int(name)] = torch.softmax(head(features), dim=1)[:, 1]
        return scores


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_iso_family(family: IsolatedAttackFamily, x: np.ndarray, y: np.ndarray,
                     new_attack_columns, config: dict, seed: int, device):
    """Train all new heads jointly, returning actual elapsed F time and steps.

    Every new head receives its own one-versus-other-attacks labels. Their
    class-balanced binary CE losses are averaged before one optimizer step;
    neither steps nor elapsed time are multiplied by the number of heads.
    An initial single-family task has no attack negatives. That limitation is
    recorded explicitly rather than introducing benign or invented negatives.
    """
    device = torch.device(device)
    columns = tuple(sorted(int(c) for c in new_attack_columns))
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("ISO task requires unique newly introduced attack columns")
    labels = np.asarray(y)
    if (len(x) != len(labels) or len(labels) == 0 or labels.ndim != 1
            or not np.issubdtype(labels.dtype, np.integer)):
        raise ValueError("ISO requires aligned nonempty attack TRAIN features and integer labels")
    if any(c < 0 or c >= family.n_attacks for c in columns):
        raise ValueError("ISO new head is outside the fixed attack-column mapping")
    previous = {int(name) for name in family.family_heads}
    if previous.intersection(columns):
        raise ValueError("A completed ISO attack head may never be retrained")
    allowed = previous.union(columns)
    if not set(int(c) for c in np.unique(labels)).issubset(allowed):
        raise ValueError("ISO inputs contain an unintroduced attack column")
    if any(not np.any(labels == c) for c in columns):
        raise ValueError("Every new ISO head requires actual positive training rows")
    if bool(family.encoder_ready.item()) != bool(previous):
        raise ValueError("ISO encoder/head checkpoint state is inconsistent")
    if any(not bool(family.trained_heads[c].item()) for c in previous):
        raise ValueError("Previous ISO head is not marked completed and frozen")

    _sync(device)
    started = time.perf_counter()
    for column in columns:
        family.add_head(column, seed=int(seed) + 10007 * (column + 1))
    first_attack_task = not bool(family.encoder_ready.item())
    family.restore_freeze_flags()
    trainable = [p for p in family.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=float(config["learning_rate"]))
    stats = {}
    loss_sum = None
    for _, batches in training_epoch_batches(
            x, labels, int(config["batch_size"]), int(seed),
            int(config["epochs_per_task"]), int(config["min_steps_per_task"]), stats):
        family.train()
        if not first_attack_task:
            family.encoder.eval()
        for name in previous:
            family.family_heads[str(name)].eval()
        for bx, by in batches:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            features = family.encoder(bx)
            if not first_attack_task:
                features = features.detach()
            losses = torch.stack([
                _class_balanced_cross_entropy(family.family_heads[str(c)](features),
                                             (by == c).long())
                for c in columns
            ])
            if not torch.isfinite(losses).all():
                raise ValueError("Nonfinite ISO binary-head training loss")
            losses.mean().backward()
            optimizer.step()
            loss_sum = losses.detach().clone() if loss_sum is None else loss_sum + losses.detach()
    expected_steps = max(int(config["min_steps_per_task"]),
                         int(config["epochs_per_task"]) * math.ceil(len(labels) / int(config["batch_size"])))
    if stats["optimizer_steps"] != expected_steps or stats["target_steps"] != expected_steps:
        raise AssertionError("ISO shared F task-step accounting mismatch")
    family.encoder_ready.fill_(True)
    for column in columns:
        family.trained_heads[column] = True
    family.tasks_completed.add_(1)
    family.restore_freeze_flags()
    family.eval()
    _sync(device)
    elapsed = time.perf_counter() - started

    mean_losses = (loss_sum / expected_steps).cpu().tolist() if expected_steps else [None] * len(columns)
    encoder_parameters = sum(p.numel() for p in family.encoder.parameters())
    head_stats = {}
    for index, column in enumerate(columns):
        head = family.family_heads[str(column)]
        positive = int(np.count_nonzero(labels == column))
        head_stats[str(column)] = {
            "positive_training_rows": positive,
            "other_attack_negative_training_rows": len(labels) - positive,
            "no_attack_negatives": positive == len(labels),
            "optimizer_steps_shared_with_task": expected_steps,
            "mean_minibatch_binary_ce": mean_losses[index],
            "adapter_parameters": sum(p.numel() for p in head.down.parameters())
                                  + sum(p.numel() for p in head.up.parameters()),
            "binary_head_parameters": sum(p.numel() for p in head.binary.parameters()),
            "exclusive_training_seconds": None,
            "time_scope": "joint task loop; exclusive per-head time is not measured",
            "frozen_after_task": True,
        }
    stats.update(
        family_type="frozen_mlp_encoder_rank8_isolated_binary_heads",
        score="independent head softmax positive probability; argmax over seen attacks; no router",
        objective="mean of new-head minibatch-class-balanced binary cross-entropies",
        new_attack_columns=list(columns), frozen_old_attack_columns=sorted(previous),
        one_optimizer_step_per_task_batch=True, per_head_budgets_are_not_serially_added=True,
        encoder_trained_this_task=first_attack_task,
        encoder_optimizer_steps=expected_steps if first_attack_task else 0,
        encoder_exclusive_training_seconds=None if first_attack_task else 0.0,
        encoder_time_scope="first-task joint encoder/head time is included once in family_training_seconds",
        encoder_parameters=encoder_parameters,
        encoder_frozen_after_task=True, old_heads_updated=False,
        family_training_seconds=elapsed, shared_task_training_seconds=elapsed,
        head_training=head_stats,
        total_family_parameters=sum(p.numel() for p in family.parameters()),
        rank=family.rank, normal_rows_used=0,
        access_scope="caller-provided current attack TRAIN plus retained old attack TRAIN only",
        first_task_single_attack_limitation=(first_attack_task and len(np.unique(labels)) == 1),
    )
    return elapsed, stats
