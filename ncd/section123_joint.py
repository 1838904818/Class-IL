"""Section123 shared-encoder candidate primitive; not a formal run entrypoint."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ncd.crr_benchmarks import _class_balanced_cross_entropy


class JointNetwork(nn.Module):
    def __init__(self, input_dim, attack_classes, hidden=128):
        super().__init__()
        if input_dim < 1 or attack_classes < 1 or hidden != 128:
            raise ValueError("Section123 requires a 2x128 encoder and nonempty outputs")
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU())
        self.gate = nn.Linear(hidden, 2)
        self.family = nn.Linear(hidden, attack_classes)
        self.register_buffer("seen_mask", torch.zeros(attack_classes, dtype=torch.bool))

    @property
    def head(self):
        """Expose only F to the existing AR weight-alignment implementation."""
        return self.family

    def set_seen(self, columns):
        columns = sorted(set(int(c) for c in columns))
        if not columns or columns[0] < 0 or columns[-1] >= len(self.seen_mask):
            raise ValueError("Invalid family columns")
        mask = torch.zeros_like(self.seen_mask)
        mask[columns] = True
        if torch.any(self.seen_mask & ~mask):
            raise ValueError("Seen classes cannot disappear")
        self.seen_mask.copy_(mask)

    def forward(self, x):
        features = self.encoder(x)
        return self.gate(features), self.family(features)

    def mask_family(self, scores):
        if not torch.any(self.seen_mask):
            raise ValueError("Set seen families before training or inference")
        return scores.masked_fill(~self.seen_mask[None, :], -1e6)

    def default_prediction(self, x, attack_ids, normal):
        if len(attack_ids) != len(self.seen_mask) or normal in attack_ids:
            raise ValueError("Attack output mapping differs")
        gate, family = self(x)
        mapping = torch.as_tensor(attack_ids, device=x.device, dtype=torch.long)
        prediction = mapping[self.mask_family(family).argmax(1)]
        return torch.where(gate.softmax(1)[:, 1] > 0.5, prediction,
                           torch.full_like(prediction, int(normal)))


def joint_loss(model, x, binary_y, family_y, *, der_batch=None, replay_batch=None):
    """No G retention loss. DER and replay CE operate exclusively on F.

    family_y is -1 on benign rows; memory callers must supply attack-only rows.
    DER and replay batches are independent samples, matching the existing AR.
    """
    if (binary_y.shape != family_y.shape or binary_y.ndim != 1
            or len(binary_y) != len(x) or not torch.all((binary_y == 0) | (binary_y == 1))):
        raise ValueError("Invalid binary/family labels")
    attack = binary_y == 1
    if torch.any(family_y[~attack] != -1):
        raise ValueError("Benign rows cannot carry family labels")
    gate, family = model(x)
    loss = _class_balanced_cross_entropy(gate, binary_y)
    if torch.any(attack):
        loss = loss + F.cross_entropy(model.mask_family(family[attack]), family_y[attack])
    if (der_batch is None) != (replay_batch is None):
        raise ValueError("DER and replay samples must both be supplied or absent")
    if der_batch is not None:
        memory_x, target, target_mask = der_batch
        _, scores = model(memory_x)
        if (target.shape != scores.shape or target_mask.shape != scores.shape
                or target_mask.dtype != torch.bool or not torch.all(target_mask.any(1))
                or torch.any(target_mask & ~model.seen_mask[None, :])):
            raise ValueError("Invalid seen-at-admission DER mask")
        loss = loss + F.mse_loss(scores[target_mask], target.detach()[target_mask])
        replay_x, replay_y = replay_batch
        _, replay_scores = model(replay_x)
        loss = loss + F.cross_entropy(model.mask_family(replay_scores), replay_y)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite AR-Joint objective")
    return loss


def train_fixed(model, x, binary_y, family_y, memory, *, seed, config, smoke=False):
    """Joint fixed-schedule optimizer; callers own exposure and memory admission.

    Memory is attack-only (x, family_y, stored_logits, seen_at_admission_mask).
    Two independent samples preserve the existing DER++ replay convention.
    This primitive does not constitute a validated experiment entrypoint.
    """
    import time
    import numpy as np
    from ncd.run_aacil_retention import check_deadline

    if len(x) != len(binary_y) or len(x) != len(family_y) or not len(x):
        raise ValueError("Nonempty aligned joint training arrays required")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite training features")
    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    batch_size = 256
    steps_per_epoch = (len(x) + batch_size - 1) // batch_size
    target_steps = steps_per_epoch if smoke else max(2000, 8 * steps_per_epoch)
    if memory is not None:
        mx, my, targets, masks = memory
        if not len(mx) or not (len(mx) == len(my) == len(targets) == len(masks)):
            raise ValueError("Invalid retained attack memory")
        if np.any(my < 0):
            raise ValueError("Benign rows forbidden in family memory")

    def sample():
        ids = rng.choice(len(mx), min(batch_size, len(mx)), replace=False)
        return tuple(torch.as_tensor(a[ids], device=device) for a in memory)

    started = time.perf_counter()
    steps, epochs = 0, []
    model.train()
    while steps < target_steps:
        order = rng.permutation(len(x))
        total, rows, epoch_steps = 0., 0, 0
        for start in range(0, len(order), batch_size):
            check_deadline(config)
            ids = order[start:start + batch_size]
            xx = torch.as_tensor(x[ids], dtype=torch.float32, device=device)
            yy = torch.as_tensor(binary_y[ids], dtype=torch.long, device=device)
            fy = torch.as_tensor(family_y[ids], dtype=torch.long, device=device)
            der, replay = None, None
            if memory is not None:
                dx, _, dt, dm = sample()
                rx, ry, _, _ = sample()
                der, replay = (dx.float(), dt.float(), dm.bool()), (rx.float(), ry.long())
            optimizer.zero_grad(set_to_none=True)
            loss = joint_loss(model, xx, yy, fy, der_batch=der, replay_batch=replay)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ids)
            rows += len(ids)
            epoch_steps += 1
            steps += 1
            if steps >= target_steps:
                break
        epochs.append(dict(steps=epoch_steps, rows=rows, mean_loss=total / rows,
                           complete_epoch=rows == len(x)))
    model.eval()
    return dict(optimizer="Adam", learning_rate=.001, batch_size=batch_size,
                requested_epochs=1 if smoke else 8, minimum_steps=0 if smoke else 2000,
                optimizer_steps=steps, epoch_history=epochs, smoke=smoke,
                elapsed_seconds=time.perf_counter()-started,
                parameters=sum(p.numel() for p in model.parameters()))
