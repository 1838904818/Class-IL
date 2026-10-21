"""B1 ER optimization primitive; receives TRAIN/validation, never TEST.

The caller owns bounded replay construction, disjoint row validation, task
checkpointing and the process resource guard. This is not a runnable entrypoint.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from ncd.run_aacil_retention import check_deadline


def train_er(model, train_x, train_y, validation_x, validation_y, *,
             seen_classes, seed, device, config, smoke=False):
    """Shuffle current plus retained replay together, restore best validation CE.

    Fixed registered schedule; only the explicit non-scientific smoke reduces
    the epoch ceiling. No class-balanced CE, fixed replay ratio, or TEST stopping.
    """
    seen = tuple(sorted(set(int(c) for c in seen_classes)))
    if not seen or seen[0] < 0 or seen[-1] >= model.head.out_features:
        raise ValueError("Invalid ER seen classes")
    for x, y in ((train_x, train_y), (validation_x, validation_y)):
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not len(y):
            raise ValueError("Nonempty aligned training and validation arrays required")
        if not np.issubdtype(y.dtype, np.integer) or not np.isin(y, seen).all():
            raise ValueError("ER labels must be integer seen-class IDs")
    if train_x.shape[1] != validation_x.shape[1]:
        raise ValueError("ER training/validation schema differs")
    allowed = torch.zeros(model.head.out_features, dtype=torch.bool, device=device)
    allowed[list(seen)] = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5)
    rng = np.random.default_rng(int(seed))
    max_epochs, batch_size = (1 if smoke else 100), 256
    best_loss, best_epoch, best_state, stale = math.inf, None, None, 0
    epochs, steps = [], 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    def batch_loss(x, y, indices):
        xb = torch.as_tensor(np.asarray(x[indices], dtype=np.float32), device=device)
        yb = torch.as_tensor(np.asarray(y[indices], dtype=np.int64), device=device)
        if not torch.isfinite(xb).all():
            raise ValueError("Nonfinite ER input")
        logits = model(xb).masked_fill(~allowed.unsqueeze(0), -1e6)
        loss = F.cross_entropy(logits, yb)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite ER cross entropy")
        return loss

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(len(train_y))
        training_sum = 0.0
        for start in range(0, len(order), batch_size):
            check_deadline(config)
            indices = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(train_x, train_y, indices)
            loss.backward()
            optimizer.step()
            steps += 1
            training_sum += float(loss.detach()) * len(indices)
        model.eval()
        validation_sum = 0.0
        with torch.no_grad():
            for start in range(0, len(validation_y), batch_size):
                check_deadline(config)
                stop = min(start + batch_size, len(validation_y))
                loss = batch_loss(validation_x, validation_y, slice(start, stop))
                validation_sum += float(loss) * (stop - start)
        validation_loss = validation_sum / len(validation_y)
        scheduler.step(validation_loss)
        epochs.append({"epoch": epoch, "training_loss": training_sum / len(train_y),
                       "validation_loss": validation_loss,
                       "learning_rate": optimizer.param_groups[0]["lr"]})
        if validation_loss < best_loss:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 10:
            break
    if best_state is None:
        raise RuntimeError("No finite ER validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"elapsed_seconds": time.perf_counter() - started,
            "optimizer": "AdamW", "initial_learning_rate": 5e-4,
            "weight_decay": 1e-5, "batch_size": batch_size,
            "max_epochs": max_epochs, "epochs_completed": len(epochs),
            "optimizer_steps": steps, "best_epoch": best_epoch,
            "best_validation_loss": best_loss, "restored_best_validation_state": True,
            "scheduler": {"name": "ReduceLROnPlateau", "factor": 0.1, "patience": 5},
            "early_stopping_patience": 10, "early_stopped": stale >= 10,
            "training_rows": len(train_y), "validation_rows": len(validation_y),
            "seen_classes": list(seen), "test_used": False, "smoke": bool(smoke),
            "epoch_history": epochs}
