"""Section123 TB schedule with unchanged caller-supplied optimization losses.

This is a primitive, not an experiment entrypoint. The caller must construct
disjoint TRAIN stopping roles and bounded old-class validation memory. Neither
calibration nor TEST arrays may be supplied here.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from ncd.run_aacil_retention import check_deadline


def assert_disjoint_roles(training_ids, stopping_ids, calibration_ids):
    roles = []
    for values in (training_ids, stopping_ids, calibration_ids):
        values = np.asarray(values)
        if values.ndim != 1 or values.dtype.kind not in "iu" or np.any(values < 0):
            raise ValueError("Role IDs must be nonnegative integer vectors")
        if len(np.unique(values)) != len(values):
            raise ValueError("Duplicate row within role")
        roles.append(values)
    if not len(roles[0]) or not len(roles[1]):
        raise ValueError("Training and stopping roles must be nonempty")
    for i in range(3):
        for j in range(i):
            if np.intersect1d(roles[i], roles[j]).size:
                raise ValueError("Optimization, stopping and calibration roles overlap")


def train_es(model, x, y, validation_x, validation_y, *, objective,
             logits, seen_classes, seed, device, config, smoke=False):
    """AdamW/plateau/ES; objective(model, xb, yb) preserves each method's loss.

    Validation is ordinary seen-class CE, independently for AR G and F, and
    contains no DER sampling or retention term. Best *whole* state is restored.
    """
    seen = sorted(set(int(c) for c in seen_classes))
    if not seen or seen[0] < 0:
        raise ValueError("Nonempty nonnegative seen classes required")
    for xx, yy in ((x, y), (validation_x, validation_y)):
        if xx.ndim != 2 or yy.ndim != 1 or not len(yy) or len(xx) != len(yy):
            raise ValueError("Nonempty aligned arrays required")
        if yy.dtype.kind not in "iu" or not np.isin(yy, seen).all():
            raise ValueError("Labels must be seen integer classes")
        if not np.isfinite(xx).all():
            raise ValueError("Nonfinite features")
    if x.shape[1] != validation_x.shape[1]:
        raise ValueError("Training and stopping feature dimensions differ")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5)
    rng = np.random.default_rng(int(seed))
    best_loss, best_epoch, best_state, stale = math.inf, None, None, 0
    history, steps = [], 0
    started = time.perf_counter()
    for epoch in range(1, (1 if smoke else 100) + 1):
        model.train()
        order = rng.permutation(len(y))
        training_sum = 0.0
        for start in range(0, len(order), 256):
            check_deadline(config)
            ids = order[start:start + 256]
            xb = torch.as_tensor(x[ids], dtype=torch.float32, device=device)
            yb = torch.as_tensor(y[ids], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model, xb, yb)
            if loss.ndim or not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite or nonscalar TB objective")
            loss.backward()
            optimizer.step()
            steps += 1
            training_sum += float(loss.detach()) * len(ids)
        model.eval()
        validation_sum = 0.0
        with torch.no_grad():
            for start in range(0, len(validation_y), 256):
                check_deadline(config)
                xx = torch.as_tensor(validation_x[start:start + 256], dtype=torch.float32, device=device)
                yy = torch.as_tensor(validation_y[start:start + 256], dtype=torch.long, device=device)
                scores = logits(model, xx)
                if scores.ndim != 2 or scores.shape[1] <= seen[-1] or not torch.isfinite(scores).all():
                    raise ValueError("Invalid TB stopping logits")
                allowed = torch.zeros(scores.shape[1], dtype=torch.bool, device=device)
                allowed[seen] = True
                loss = F.cross_entropy(scores.masked_fill(~allowed[None, :], -1e6), yy)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite TB stopping loss")
                validation_sum += float(loss) * len(yy)
        value = validation_sum / len(validation_y)
        scheduler.step(value)
        history.append(dict(epoch=epoch, training_loss=training_sum / len(y),
                            validation_loss=value, learning_rate=optimizer.param_groups[0]["lr"]))
        if value < best_loss:
            best_loss, best_epoch, stale = value, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 10:
            break
    if best_state is None:
        raise RuntimeError("No finite stopping checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return dict(elapsed_seconds=time.perf_counter() - started, optimizer="AdamW",
                initial_learning_rate=5e-4, weight_decay=1e-5, batch_size=256,
                max_epochs=1 if smoke else 100, epochs_completed=len(history),
                optimizer_steps=steps, best_epoch=best_epoch, best_validation_loss=best_loss,
                restored_best_validation_state=True, early_stopping_patience=10,
                early_stopped=stale >= 10, scheduler=dict(name="ReduceLROnPlateau", factor=0.1, patience=5),
                training_rows=len(y), validation_rows=len(validation_y), seen_classes=seen,
                test_used=False, smoke=bool(smoke), epoch_history=history)
