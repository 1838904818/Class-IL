"""Section 21: matched prediction-failure scouting, not a proposed mechanism.

DER++ objective follows Buzzega et al. (NeurIPS 2020) and the authors' Mammoth
implementation. The allocation is explicitly adapted to the shared 200/class
budget: independent within-class reservoirs instead of a global reservoir.
All CE terms use the existing minibatch-class-balanced CE. No PI is computed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from ncd.crr_benchmarks import _class_balanced_cross_entropy, _train, training_epoch_batches
from ncd.model import MLP
from ncd.run_crr_real_sequence import _scorer, _write_csv
from ncd.source_commit import source_commit
from ncd.train_cil import Tee, herding_select, loader, seed_everything

METHODS = ("joint", "replay", "icarl", "replay_lwf", "rcl_tab", "derpp", "wacb", "freq", "farq", "farq_all")
SECTION26_POLICY = "benign_refresh_attack_memory_only"
DATASETS = ("NSL-KDD", "UNSW-NB15", "CIC-IDS-2017", "CSE-CIC-IDS2018")
SEEDS = (1, 2, 3, 4, 42)
DERPP_REFERENCE = "https://github.com/aimagelab/mammoth/blob/e75a491c69fd729edeb01431afb753d9157d9a81/models/derpp.py"
POLICIES = {
    "joint": "Fresh initialization at EVERY task checkpoint; all prepared training rows of all seen classes; same epochs/optimizer/CE.",
    "memory": "At most buffer_per_class stored examples per seen class; old full training pools are never revisited by CL learners.",
    "selection_change_from_history": "Unlike historical _select_buffer, old classes retain only previously stored examples; only new classes access their current full training pool.",
    "derpp": "Official CE + alpha logit MSE + beta replay CE, two independently drawn buffer minibatches, pre-update stored logits; classwise reservoir allocation replaces official global reservoir to match 200/class budget.",
    "ce": "Existing inverse-current-minibatch-frequency class-balanced CE for every CE term, including DER++.",
    "evaluation": "Task-known seen-label set at checkpoint, no task identity at inference; future/unseen true classes excluded. All supported seen classes reported, no recall floor exclusions.",
    "error_flow": "Mutually exclusive incorrect predictions: Normal first, then a class introduced later than the true class, then other old/same-task classes.",
    "cost": "training_seconds is synchronized wall time for optimization only, cumulative across completed checkpoints; selection/evaluation seconds separate. Failed/interrupted attempts remain in logs and are not counted in successful training totals.",
    "resource_measurement": "CUDA allocated/reserved peaks cover optimization only, not exemplar selection or evaluation; host RAM peak is not measured. Large exemplar embedding matrices use temporary disk-backed mmap, whose resident file pages remain OS-managed.",
    "herding_storage": "Current-class rows are gathered one inference batch at a time; float32 embeddings are preallocated or temporarily memory-mapped. The historical herding_select mean, dot products, order, precision and selection steps are unchanged.",
    "rng": "Deterministic task-local seed = seed*10000+task_index; checkpoint-boundary resume uses exactly the same task-local seed.",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def path_label(path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), ROOT)).as_posix()


def save_json(path: Path, value) -> None:
    require_finite(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    for attempt in range(40):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.05)


def require_finite(value) -> None:
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all().item():
            raise ValueError("nonfinite tensor")
    elif isinstance(value, np.ndarray):
        if not np.isfinite(value).all():
            raise ValueError("nonfinite array")
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError("nonfinite scalar")
    elif isinstance(value, dict):
        for item in value.values():
            require_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            require_finite(item)


def binding(data, x_shift, y_shift) -> dict:
    result = {}
    for name, value in [("x_train", data.x_train), ("y_train", data.y_train),
                        ("x_test", data.x_test), ("y_test", data.y_test),
                        ("x_shift", x_shift), ("y_shift", y_shift)]:
        if value is not None:
            a = np.ascontiguousarray(value)
            result[name] = {"shape": list(a.shape), "dtype": a.dtype.str,
                            "sha256": hashlib.sha256(memoryview(a).cast("B")).hexdigest()}
    return result


def smoke_subset(data, x_shift, y_shift, limit=64):
    def take(x, y):
        if x is None:
            return None, None
        idx = np.concatenate([np.flatnonzero(y == c)[:limit] for c in np.unique(y)])
        return x[idx], y[idx]
    fields = {}
    for split in ("train", "test"):
        fields["x_" + split], fields["y_" + split] = take(getattr(data, "x_" + split), getattr(data, "y_" + split))
    return replace(data, **fields), *take(x_shift, y_shift)


class ClassReservoir:
    """DER++ memory with standard reservoir admission separately per class."""
    def __init__(self, per_class: int, input_dim: int, n_classes: int, state=None, track_indices=False):
        self.per_class, self.input_dim, self.n_classes = per_class, input_dim, n_classes
        self.x = torch.zeros((n_classes, per_class, input_dim), dtype=torch.float32)
        self.logits = torch.zeros((n_classes, per_class, n_classes), dtype=torch.float32)
        self.observations = [0] * n_classes
        self.row_ids = torch.full((n_classes, per_class), -1, dtype=torch.int64) if track_indices else None
        if state:
            if state["per_class"] != per_class or state["input_dim"] != input_dim or state["n_classes"] != n_classes:
                raise ValueError("DER++ buffer shape changed on resume")
            self.x, self.logits = state["x"].clone(), state["logits"].clone()
            self.observations = list(state["observations"])
            if track_indices:
                self.row_ids = state["row_ids"].clone()

    def __len__(self):
        return sum(min(self.per_class, n) for n in self.observations)

    def state(self):
        result = {"per_class": self.per_class, "input_dim": self.input_dim, "n_classes": self.n_classes,
                  "x": self.x, "logits": self.logits, "observations": self.observations}
        if self.row_ids is not None:
            result["row_ids"] = self.row_ids
        return result

    def add(self, x, y, logits, rng, row_ids=None):
        # Saved logits come from the forward BEFORE the optimizer update.
        x, y, logits = x.detach().cpu(), y.detach().cpu(), logits.detach().cpu()
        labels = y.numpy()
        for label in np.unique(labels):
            rows = np.flatnonzero(labels == label)
            counts = self.observations[label] + np.arange(len(rows))
            slots = rng.integers(0, counts + 1)
            slots = np.where(counts < self.per_class, counts, slots)
            self.observations[label] += len(rows)
            selected = np.flatnonzero(slots < self.per_class)
            # Repeated admissions to the same slot keep the LAST stream item.
            _, reverse = np.unique(slots[selected][::-1], return_index=True)
            kept = selected[len(selected) - 1 - reverse]
            target = torch.from_numpy(slots[kept])
            source = torch.from_numpy(rows[kept])
            self.x[int(label), target] = x[source]
            self.logits[int(label), target] = logits[source]
            if self.row_ids is not None:
                if row_ids is None:
                    raise ValueError("Tracked DER++ admission requires original training row IDs")
                self.row_ids[int(label), target] = row_ids.detach().cpu()[source]

    def sample(self, count, rng, device):
        slots = [(c, i) for c, n in enumerate(self.observations) for i in range(min(n, self.per_class))]
        picked = rng.choice(len(slots), min(count, len(slots)), replace=False)
        classes = torch.tensor([slots[i][0] for i in picked], dtype=torch.long)
        positions = torch.tensor([slots[i][1] for i in picked], dtype=torch.long)
        return (self.x[classes, positions].to(device), classes.to(device),
                self.logits[classes, positions].to(device))

    def payload_bytes(self):
        # Class labels are encoded by the tensor's class axis, not duplicated.
        return len(self) * (4 * (self.input_dim + self.n_classes) + (8 if self.row_ids is not None else 0)) + 8 * self.n_classes

    def allocated_bytes(self):
        return (self.x.numel() + self.logits.numel()) * 4 + 8 * self.n_classes + (self.row_ids.numel() * 8 if self.row_ids is not None else 0)


def train_derpp(model, x, y, memory, config, task_seed, device, training_stats=None, row_ids=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    rng = np.random.default_rng(task_seed + 8317)
    schedule = training_stats if training_stats is not None else {}
    # Integer row-index loader preserves the same deterministic shuffle without
    # feeding audit IDs into the model or changing the DER++ objective.
    loader_x = np.arange(len(x), dtype=np.int64)[:, None] if row_ids is not None else x
    for epoch, batches in training_epoch_batches(
            loader_x, y, int(config["batch_size"]), task_seed, int(config["epochs_per_task"]),
            int(config.get("min_steps_per_task", 0)), schedule):
        model.train()
        for bx, by in batches:
            batch_ids = None
            if row_ids is not None:
                positions = bx[:, 0].numpy()
                bx = torch.from_numpy(np.asarray(x[positions]))
                batch_ids = torch.from_numpy(row_ids[positions])
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = _class_balanced_cross_entropy(logits, by)
            if len(memory):
                mx, _, mz = memory.sample(int(config["batch_size"]), rng, device)
                loss = loss + float(config["derpp_alpha"]) * F.mse_loss(model(mx), mz)
                mx, my, _ = memory.sample(int(config["batch_size"]), rng, device)
                loss = loss + float(config["derpp_beta"]) * _class_balanced_cross_entropy(model(mx), my)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite DER++ loss")
            loss.backward()
            optimizer.step()
            memory.add(bx, by, logits, rng, batch_ids)
        if schedule["partial_epoch_batches"]:
            print(f"DERPP_EPOCH_PARTIAL epoch={epoch + 1} batches={schedule['partial_epoch_batches']} "
                  f"optimizer_steps={schedule['optimizer_steps']} buffer_rows={len(memory)}", flush=True)
        else:
            print(f"DERPP_EPOCH_DONE epoch={epoch + 1} buffer_rows={len(memory)}", flush=True)


def herding_select_indexed(model, x, rows, count, device, batch_size, *, mmap_threshold_bytes, scratch_dir=None):
    """Same embeddings/herding order as extract_features(x[rows]), without full copies."""
    if not len(rows):
        return np.empty(0, dtype=np.int64)
    if scratch_dir is not None:
        scratch_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    features = None
    with tempfile.TemporaryDirectory(prefix="herding-", dir=scratch_dir) as temp:
        try:
            with torch.no_grad():
                for start in range(0, len(rows), batch_size):
                    batch = torch.from_numpy(x[rows[start:start + batch_size]]).to(device)
                    part = model.features(batch).cpu().numpy()
                    require_finite(part)
                    if features is None:
                        shape = (len(rows), part.shape[1])
                        size = int(np.prod(shape)) * part.dtype.itemsize
                        if size > mmap_threshold_bytes:
                            features = np.memmap(Path(temp) / "features.dat", mode="w+", dtype=part.dtype, shape=shape)
                            storage = "mmap"
                        else:
                            features = np.empty(shape, dtype=part.dtype)
                            storage = "memory"
                        print(f"HERDING_FEATURES rows={len(rows)} columns={part.shape[1]} bytes={size} storage={storage}", flush=True)
                    features[start:start + len(part)] = part
            # ndarray view preserves the exact historical float32 reductions;
            # it does not allocate another embedding matrix for a memmap.
            return herding_select(np.asarray(features), count)
        finally:
            if isinstance(features, np.memmap):
                features.flush()
                features._mmap.close()


def update_memory(model, current_x, current_y, buffer_x, buffer_y, method, config, task_seed, device, *, scratch_dir=None):
    """Accept ONLY current-task rows plus existing memory, never a full dataset."""
    xs, ys = [x.copy() for x in buffer_x], [y.copy() for y in buffer_y]
    old_ids = {int(y[0]) for y in ys if len(y)}
    for c in np.unique(current_y):
        if int(c) in old_ids:
            raise ValueError("new-task rows overlap stored old classes")
        rows = np.flatnonzero(current_y == c)
        count = min(int(config["buffer_per_class"]), len(rows))
        if method == "icarl":
            selected = herding_select_indexed(
                model, current_x, rows, count, device, int(config["batch_size"]),
                mmap_threshold_bytes=int(config.get("herding_mmap_threshold_bytes", 256 * 1024 * 1024)),
                scratch_dir=scratch_dir)
        else:
            selected = np.sort(np.random.default_rng(task_seed * 10007 + int(c)).choice(len(rows), count, replace=False))
        xs.append(current_x[rows[selected]].copy())
        ys.append(np.full(count, c, dtype=np.int64))
    return xs, ys


def memory_size(buffer_x, buffer_y):
    return sum(len(y) for y in buffer_y), sum(x.nbytes + y.nbytes for x, y in zip(buffer_x, buffer_y))


def section26_pools(data, seed):
    """Permanent TRAIN-only holdout; same split for every compared method."""
    train, valid = {}, {}
    for c in range(len(data.class_order)):
        rows = np.flatnonzero(data.y_train == c)
        n = max(5, int(np.ceil(len(rows) * .1)))
        if n >= len(rows):
            raise ValueError("Section26 needs at least one training row after holdout")
        shuffled = np.random.default_rng(seed * 100003 + c * 1009 + 17).permutation(rows)
        valid[c], train[c] = shuffled[:n], np.sort(shuffled[n:])
    return train, valid


def effective_policies(config):
    if not config.get("section26", False):
        return POLICIES
    return POLICIES | {
        "joint": "Offline upper-bound exception: fresh initialization each task on all seen TRAIN rows excluding common holdout; revisits old attacks, unlike deployment methods; not one of seven replay methods.",
        "memory": "M=200*seen classes includes training replay and retained validation. Requested per-class quota may exceed200; unavailable rows leave capacity unused.",
        "selection_change_from_history": "Section26.9: Normal training/validation pools are refreshed each task with common task-dependent RNG; old attacks use stored replay/validation only, never old attack pools. New attacks use current-task TRAIN only.",
        "derpp": "Unchanged CE+alpha logit MSE+beta replay CE and two independent minibatches;180 training plus20 validation slots/class. Old attack samples/logits retained; refreshed Normal target logits computed under the current post-task model.",
        "validation": "Permanent max(5,ceil(10%)) per-class TRAIN holdout. Allocation uses old retained validation and at most20 heldout rows for each new class; retained validation is counted inside M.",
        "replay_count_scope": "replay_examples/bytes fields measure optimization replay only; allocation.actual_total includes retained validation. Selection time includes validation scoring and archive sampling.",
        "freq": "Original cumulative seen-class TRAIN counts only;20 slots/class plus remaining capacity proportional to counts, largest remainder with lower class-ID ties; multiclass simplification, not exact ECBRS.",
        "scope": "Section26.9 benign stream available/old attacks inaccessible deployment comparison under RQ2/RO2; Joint explicitly offline; no OFRA motivation or RQ1/RQ3 completion.",
    }


def section26_memory(model, data, train, valid, bx, by, stored_ids, vx, vy, valid_ids,
                     task, seen, method, config, seed, t, device):
    """Refresh benign data; old attacks can only shrink/reuse stored rows."""
    normal, count = data.class_order.index("Normal"), len(seen)
    per_class = int(config["buffer_per_class"])
    total = per_class * count
    old_valid = {int(y[0]): x for x, y in zip(vx, vy) if len(y)}
    old_train = {int(y[0]): x for x, y in zip(bx, by) if len(y)}
    errors, evaluated, wrong = {}, {}, {}
    model.eval()
    with torch.no_grad():
        for c in seen:
            values = data.x_train[valid[c][:20]] if c in task else old_valid[c]
            if not len(values):
                raise ValueError("No heldout validation examples for a seen class")
            scores = model(torch.from_numpy(np.asarray(values)).to(device))[:, list(seen)]
            if not torch.isfinite(scores).all():
                raise ValueError("Nonfinite heldout scores")
            pred = np.asarray(seen)[scores.argmax(1).cpu().numpy()]
            wrong[c], evaluated[c] = int(np.count_nonzero(pred != c)), len(values)
            errors[c] = wrong[c] / evaluated[c]
    f = errors[normal]
    attacks = sorted(c for c in seen if c != normal)
    m = float(np.mean([errors[c] for c in attacks])) if attacks else 0.0
    q = 1.0 / count
    caps = {c: per_class for c in seen}
    if method in {"wacb", "farq"}:
        q = .5 if method == "wacb" else float(np.clip(f / (f + m) if f + m else q, q, .6))
        nc = min(int(np.floor(q * total)), total - 20 * len(attacks))
        base, rem = divmod(total - nc, len(attacks))
        caps = {normal: nc, **{c: base + (i < rem) for i, c in enumerate(attacks)}}
    frequencies = {c: len(train[c]) + len(valid[c]) for c in seen}
    if method == "freq" or (method == "farq_all" and sum(errors.values())):
        weights = frequencies if method == "freq" else errors
        remaining = total - 20 * count
        raw = {c: remaining * weights[c] / sum(weights.values()) for c in seen}
        caps = {c: 20 + int(np.floor(raw[c])) for c in seen}
        order = sorted(seen, key=lambda c: (-(raw[c] - np.floor(raw[c])), c))
        for c in order[:total - sum(caps.values())]:
            caps[c] += 1
        q = caps[normal] / total
    xs, ys, vxs, vys, tn, vn, selected_ids, selected_valid_ids = [], [], [], [], {}, {}, {}, {}
    for c in sorted(seen):
        nv = max(5, int(np.ceil(caps[c] * .1)))
        rng_seed = (seed * 10000 + t) * 10007 + c
        if c == normal or c in task:
            vp = np.random.default_rng(rng_seed + 53).permutation(len(valid[c]))[:nv]
            chosen_v = np.sort(valid[c][vp]); values_v = data.x_train[chosen_v].copy()
        else:
            vp = np.sort(np.random.default_rng(rng_seed + 53).permutation(len(valid_ids[c]))[:nv])
            chosen_v = valid_ids[c][vp]; values_v = old_valid[c][vp].copy()
        nt = max(0, caps[c] - nv)
        if method == "joint":
            chosen_t = np.empty(0, dtype=np.int64); values_t = np.empty((0, data.x_train.shape[1]), dtype=np.float32)
        elif c == normal or (c in task and method != "derpp"):
            tp = np.random.default_rng(rng_seed).permutation(len(train[c]))[:nt]
            chosen_t = np.sort(train[c][tp]); values_t = data.x_train[chosen_t].copy()
        else:
            tp = np.arange(len(stored_ids[c])) if len(stored_ids[c]) <= nt else np.sort(np.random.default_rng(rng_seed).choice(len(stored_ids[c]), nt, replace=False))
            chosen_t = stored_ids[c][tp]; values_t = old_train[c][tp].copy()
        xs.append(values_t); ys.append(np.full(len(chosen_t), c, dtype=np.int64))
        vxs.append(values_v); vys.append(np.full(len(chosen_v), c, dtype=np.int64))
        tn[str(c)], vn[str(c)] = len(chosen_t), len(chosen_v)
        selected_ids[c], selected_valid_ids[c] = chosen_t, chosen_v
    detail = {"policy": method, "f": f, "m": m, "q_requested": q,
              "m_definition": "macro mean P(predicted family != true attack family)",
              "target_counts": {str(c): int(caps[c]) for c in seen},
              "actual_training_counts": tn, "actual_validation_counts": vn,
              "validation_eval_counts": {str(c): evaluated[c] for c in seen},
              "validation_class_errors": {str(c): errors[c] for c in seen},
              "validation_error_counts": {str(c): wrong[c] for c in seen},
              "heldout_pool_counts": {str(c): len(valid[c]) for c in seen},
              "frequency_counts": {str(c): frequencies[c] for c in seen},
              "frequency_rule": "20/class floor; remaining slots proportional to original TRAIN frequency; largest remainder, ties lower class ID",
              "capacity_total": total, "actual_total": sum(tn.values()) + sum(vn.values()),
              "old_training_pool_access_classes": [normal] if normal not in task and method != "joint" else [],
              "old_validation_pool_access_classes": [normal] if normal not in task else [],
              "old_attack_training_pool_access": False, "old_attack_validation_pool_access": False,
              "pool_access_flags_scope": "replay/validation update; offline Joint training access listed separately",
              "normal_refreshed": True, "normal_training_refresh_applicable": method != "joint",
              "normal_refresh_rng_seed": (seed * 10000 + t) * 10007 + normal,
              "training_row_ids": {str(c): selected_ids[c].tolist() for c in seen},
              "validation_row_ids": {str(c): selected_valid_ids[c].tolist() for c in seen},
              "offline_joint_reference": method == "joint", "reference_not_replay_budget": method == "joint",
              "offline_joint_old_training_pool_access_classes": [int(c) for c in seen if c not in task] if method == "joint" else [],
              "validation_role": "permanent TRAIN holdout; never optimization or training replay; retained validation counts inside M",
              "archive_access": SECTION26_POLICY}
    detail["normal_actual_fraction_of_capacity"] = (tn[str(normal)] + vn[str(normal)]) / total
    detail["normal_actual_fraction_of_used"] = (tn[str(normal)] + vn[str(normal)]) / detail["actual_total"]
    return xs, ys, selected_ids, vxs, vys, selected_valid_ids, detail


def metrics_from_confusion(cm, seen, class_order, tasks, train_counts):
    n = len(class_order)
    if cm.shape != (n, n) or np.any(cm < 0):
        raise ValueError("invalid confusion matrix")
    intro = {c: t for t, task in enumerate(tasks) for c in task}
    normal = class_order.index("Normal")
    rows = []
    for c in seen:
        support, predicted, tp = int(cm[c].sum()), int(cm[:, c].sum()), int(cm[c, c])
        fn, fp = support - tp, predicted - tp
        miss_normal = int(cm[c, normal]) if c != normal else 0
        miss_newer = sum(int(cm[c, k]) for k in seen if k != c and k != normal and intro[k] > intro[c])
        miss_other = fn - miss_normal - miss_newer
        rows.append({"class_id": int(c), "class_name": class_order[c], "introduced_task": intro[c],
                     "train_count": int(train_counts[c]), "support": support,
                     "recall": tp / support if support else None,
                     "precision": tp / predicted if predicted else None,
                     "f1": 2 * tp / (2 * tp + fn + fp) if 2 * tp + fn + fp else 0.0,
                     "error_to_newer_class": miss_newer, "error_to_normal": miss_normal,
                     "error_to_other_old": miss_other, "error_count": fn})
    attack = [c for c in seen if c != normal]
    attack_n = int(cm[attack].sum()) if attack else 0
    attack_tp = int(cm[np.ix_(attack, attack)].sum()) if attack else 0
    normal_n = int(cm[normal].sum())
    recalls = [r["recall"] for r in rows if r["recall"] is not None]
    return {"confusion_matrix": cm.tolist(), "confusion_labels": list(range(n)),
            "evaluated_rows": int(cm.sum()), "macro_f1": float(np.mean([r["f1"] for r in rows])) if rows else None,
            "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
            "attack_recall": attack_tp / attack_n if attack_n else None,
            "attack_recall_definition": "P(predicted_label != Normal | true_label != Normal)",
            "attack_family_macro_recall": float(np.mean([r["recall"] for r in rows if r["class_id"] != normal and r["recall"] is not None])) if any(r["class_id"] != normal and r["recall"] is not None for r in rows) else None,
            "attack_support": attack_n, "normal_fpr": (normal_n - int(cm[normal, normal])) / normal_n if normal_n else None,
            "normal_support": normal_n, "per_class": rows}


def evaluate(scorer, x, y, seen, data, config, device, log_prior=None):
    if x is None or y is None:
        return None
    selected = np.flatnonzero(np.isin(y, seen))
    n = len(data.class_order)
    cm = np.zeros((n, n), dtype=np.int64)
    scorer.eval()
    with torch.no_grad():
        for start in range(0, len(selected), int(config["eval_batch_size"])):
            idx = selected[start:start + int(config["eval_batch_size"])]
            logits = scorer(torch.from_numpy(np.asarray(x[idx])).to(device))
            if log_prior is not None:
                logits = logits + log_prior
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite evaluation scores")
            local = logits[:, list(seen)].argmax(dim=1).cpu().numpy()
            pred = np.asarray(seen)[local]
            cm += np.bincount(y[idx] * n + pred, minlength=n * n).reshape(n, n)
    result = metrics_from_confusion(cm, seen, data.class_order, data.tasks,
                                    np.bincount(data.y_train, minlength=n))
    result["total_available_rows"] = len(y)
    result["excluded_unseen_rows"] = len(y) - len(selected)
    return result


def checkpoint_load(path, identity):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("identity") != identity:
        raise ValueError(f"checkpoint provenance mismatch: {path}")
    require_finite(state)
    return state


def checkpoint_save(path, state):
    require_finite(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".pt.tmp")
    torch.save(state, temp)
    temp.replace(path)


def run_cell(data, x_shift, y_shift, metadata, config, identity, output, device, evaluate_only=False):
    output.mkdir(parents=True, exist_ok=True)
    final_path = output / "result.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        if result.get("identity") != identity or not result.get("completed") or len(result.get("checkpoints", [])) != len(data.tasks):
            raise ValueError("completed cell identity/completeness mismatch")
        for record in result["checkpoints"]:
            if digest(output / record["checkpoint_file"]) != record["checkpoint_sha256"]:
                raise ValueError("completed cell checkpoint changed")
            saved_metric = (output / record["checkpoint_file"]).with_suffix(".json")
            if json.loads(saved_metric.read_text(encoding="utf-8")) != record:
                raise ValueError("completed cell differs from checkpoint metrics")
        require_finite(result)
        print(f"CELL_RESUME {identity['dataset']} {identity['method']} {identity['seed']}", flush=True)
        return result
    method, seed = identity["method"], identity["seed"]
    section26 = bool(config.get("section26", False))
    if section26 and config.get("section26_access_policy") != SECTION26_POLICY:
        raise ValueError("Section26.9 requires the new benign-only-refresh policy")
    literature_confirmation = bool(config.get("section178_literature", False))
    if literature_confirmation and (section26 or method != "derpp"):
        raise ValueError("Section178 DER++ requires non-refresh memory path")
    train_pools, valid_pools = section26_pools(data, seed) if section26 or literature_confirmation else ({}, {})
    validation_x, validation_y = [], []
    buffer_ids, validation_ids = {}, {}
    model = MLP(data.x_train.shape[1], int(config["hidden"]), len(data.class_order)).to(device)
    buffer_x, buffer_y, previous_seen, records = [], [], (), []
    reservoir_cap = int(config["buffer_per_class"]) - (max(5, int(np.ceil(config["buffer_per_class"] * .1))) if section26 else 0)
    memory = ClassReservoir(reservoir_cap, data.x_train.shape[1], len(data.class_order), track_indices=section26) if method == "derpp" else None
    cumulative = 0.0
    for t, task in enumerate(data.tasks):
        checkpoint = output / "checkpoints" / f"checkpoint_{t:03d}.pt"
        metric_path = output / "checkpoints" / f"checkpoint_{t:03d}.json"
        seen = tuple(c for part in data.tasks[:t + 1] for c in part)
        task_seed = seed * 10000 + t
        seed_everything(task_seed)
        if checkpoint.exists():
            state = checkpoint_load(checkpoint, identity)
            if state["task_index"] != t or state["seen_classes"] != list(seen):
                raise ValueError("checkpoint task order mismatch")
            model.load_state_dict(state["model_state"])
            buffer_x = [a.numpy().copy() for a in state["buffer_x"]]
            buffer_y = [a.numpy().copy() for a in state["buffer_y"]]
            if section26:
                validation_x = [a.numpy().copy() for a in state["validation_x"]]
                validation_y = [a.numpy().copy() for a in state["validation_y"]]
                buffer_ids = {int(c): a.numpy().copy() for c, a in state["buffer_row_ids"].items()}
                validation_ids = {int(c): a.numpy().copy() for c, a in state["validation_row_ids"].items()}
            if method == "derpp":
                memory = ClassReservoir(reservoir_cap, data.x_train.shape[1], len(data.class_order), state["derpp_memory"], track_indices=section26)
            cumulative = state["training_seconds_cumulative"]
            print(f"CHECKPOINT_RESUME {identity['dataset']} {method} seed={seed} task={t}", flush=True)
        else:
            if evaluate_only:
                raise FileNotFoundError(checkpoint)
            if t == 0 or method == "joint":
                # Joint is never warm-started, even after a resumed checkpoint.
                model = MLP(data.x_train.shape[1], int(config["hidden"]), len(data.class_order)).to(device)
            selected_classes = seen if method == "joint" else tuple(task)
            if section26 or literature_confirmation:
                indices = np.sort(np.concatenate([train_pools[c] for c in selected_classes]))
                current_x, current_y = data.x_train[indices], data.y_train[indices]
            elif method == "joint" and len(seen) == len(data.class_order):
                current_x, current_y = data.x_train, data.y_train
            else:
                mask = np.isin(data.y_train, selected_classes)
                current_x, current_y = data.x_train[mask], data.y_train[mask]
            train_x, train_y = current_x, current_y
            before_n, before_bytes = memory_size(buffer_x, buffer_y)
            if memory is not None:
                before_n, before_bytes = len(memory), memory.payload_bytes()
            if method not in {"joint", "derpp"} and buffer_x:
                train_x = np.concatenate([current_x, *buffer_x])
                train_y = np.concatenate([current_y, *buffer_y])
            teacher = copy.deepcopy(model).to(device).eval() if t and method not in {"joint", "replay", "derpp", "wacb", "freq", "farq", "farq_all"} else None
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            print(f"CHECKPOINT_TRAINING {identity['dataset']} {method} seed={seed} task={t} train_rows={len(train_y)} replay_rows={before_n}", flush=True)
            min_steps = int(config.get("min_steps_per_task", 0))
            training_schedule = {} if min_steps else None
            if method == "derpp":
                if training_schedule is None:
                    train_derpp(model, current_x, current_y, memory, config, task_seed, device,
                                row_ids=indices if section26 else None)
                else:
                    train_derpp(model, current_x, current_y, memory, config, task_seed, device,
                                training_stats=training_schedule, row_ids=indices if section26 else None)
            else:
                _train(model, train_x, train_y, method="replay" if method in {"joint", "wacb", "freq", "farq", "farq_all"} else method,
                       old_model=teacher, old_classes=previous_seen, seen_classes=seen,
                       epochs=int(config["epochs_per_task"]), batch_size=int(config["batch_size"]),
                       learning_rate=float(config["learning_rate"]), seed=task_seed,
                       regularization=float(identity["regularization"]), temperature=float(config["temperature"]),
                       rcl_features=int(config["rcl_features"]), rcl_temperature=float(config["rcl_temperature"]),
                       rcl_beta=float(config["rcl_beta"]), rcl_prediction_weight=float(config["rcl_prediction_weight"]),
                       **({"min_steps": min_steps, "training_stats": training_schedule} if min_steps else {}))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            training_seconds = time.perf_counter() - started
            cumulative += training_seconds
            peak_alloc = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
            del teacher
            started = time.perf_counter()
            if section26:
                prior_x, prior_y, prior_ids = buffer_x, buffer_y, buffer_ids
                if memory is not None:
                    prior_ids = {c: memory.row_ids[c, :min(reservoir_cap, memory.observations[c])].numpy().copy() for c in seen}
                    prior_x = [memory.x[c, :len(prior_ids[c])].numpy().copy() for c in sorted(seen)]
                    prior_y = [np.full(len(prior_ids[c]), c, dtype=np.int64) for c in sorted(seen)]
                new_x, new_y, buffer_ids, validation_x, validation_y, validation_ids, allocation = section26_memory(
                    model, data, train_pools, valid_pools, prior_x, prior_y, prior_ids,
                    validation_x, validation_y, validation_ids, task, seen, method, config, seed, t, device)
                if method not in {"joint", "derpp"}:
                    buffer_x, buffer_y = new_x, new_y
                elif memory is not None:
                    normal = data.class_order.index("Normal")
                    normal_x = new_x[sorted(seen).index(normal)]
                    n = len(normal_x)
                    memory.x[normal].zero_(); memory.logits[normal].zero_(); memory.row_ids[normal].fill_(-1)
                    with torch.no_grad():
                        normal_tensor = torch.from_numpy(normal_x).to(device)
                        memory.x[normal, :n] = normal_tensor.cpu()
                        memory.logits[normal, :n] = model(normal_tensor).detach().cpu()
                    memory.row_ids[normal, :n] = torch.from_numpy(buffer_ids[normal])
                    memory.observations[normal] = n
                    allocation["derpp_normal_logit_target"] = "current post-task model on newly refreshed Normal rows; old attack stored logits unchanged"
                del new_x, new_y, prior_x, prior_y, prior_ids
            elif method not in {"joint", "derpp"}:
                buffer_x, buffer_y = update_memory(model, current_x, current_y, buffer_x, buffer_y, method, config, task_seed, device,
                                                   scratch_dir=output / "selection_tmp")
            selection_seconds = time.perf_counter() - started
            after_n, after_bytes = memory_size(buffer_x, buffer_y)
            if memory is not None:
                after_n, after_bytes = len(memory), memory.payload_bytes()
            state = {"identity": identity, "task_index": t, "seen_classes": list(seen),
                     "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                     "buffer_x": [torch.from_numpy(a.copy()) for a in buffer_x],
                     "buffer_y": [torch.from_numpy(a.copy()) for a in buffer_y],
                     "derpp_memory": memory.state() if memory is not None else None,
                     "training_seconds": training_seconds, "training_seconds_cumulative": cumulative,
                     "buffer_selection_seconds": selection_seconds,
                     "peak_cuda_allocated_bytes": peak_alloc, "peak_cuda_reserved_bytes": peak_reserved,
                     "replay_examples_before": before_n, "replay_examples_after": after_n,
                     "replay_bytes_before": before_bytes, "replay_bytes_after": after_bytes,
                     "replay_allocated_bytes_after": memory.allocated_bytes() if memory is not None else after_bytes,
                     "train_rows": len(train_y), "new_training_rows": len(current_y),
                     "train_class_counts": {str(c): int(np.sum(train_y == c)) for c in seen}}
            if section26:
                state.update(validation_x=[torch.from_numpy(a.copy()) for a in validation_x],
                             validation_y=[torch.from_numpy(a.copy()) for a in validation_y],
                             buffer_row_ids={str(c): torch.from_numpy(a.copy()) for c, a in buffer_ids.items()},
                             validation_row_ids={str(c): torch.from_numpy(a.copy()) for c, a in validation_ids.items()},
                             allocation=allocation)
            if training_schedule is not None:
                baseline_steps = int(config["epochs_per_task"]) * ((len(train_y) + int(config["batch_size"]) - 1) // int(config["batch_size"]))
                expected_steps = max(baseline_steps, min_steps)
                if training_schedule.get("optimizer_steps") != expected_steps or training_schedule.get("target_steps") != expected_steps:
                    raise ValueError("training optimizer-step count does not match requested budget")
                state["training_schedule"] = training_schedule
            checkpoint_save(checkpoint, state)
            print(f"CHECKPOINT_TRAINED {identity['dataset']} {method} seed={seed} task={t} train_seconds={training_seconds:.3f}", flush=True)
            del train_x, train_y, current_x, current_y
        checkpoint_hash = digest(checkpoint)
        if metric_path.exists():
            record = json.loads(metric_path.read_text(encoding="utf-8"))
            if record.get("identity_hash") != canonical_hash(identity) or record.get("checkpoint_sha256") != checkpoint_hash:
                raise ValueError("checkpoint metrics provenance mismatch")
        else:
            started = time.perf_counter()
            scorer = _scorer(model, method, buffer_x, buffer_y, len(data.class_order), device, int(config["batch_size"]))
            record = {k: state[k] for k in ("task_index", "seen_classes", "training_seconds", "training_seconds_cumulative",
                                           "buffer_selection_seconds", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes",
                                           "replay_examples_before", "replay_examples_after", "replay_bytes_before", "replay_bytes_after",
                                           "replay_allocated_bytes_after",
                                           "train_rows", "new_training_rows", "train_class_counts")}
            if "training_schedule" in state:
                record["training_schedule"] = state["training_schedule"]
            if section26:
                record["allocation"] = state["allocation"]
            record.update(new_classes=list(task), checkpoint_file=checkpoint.relative_to(output).as_posix(),
                          checkpoint_sha256=checkpoint_hash, identity_hash=canonical_hash(identity),
                          id=evaluate(scorer, data.x_test, data.y_test, seen, data, config, device),
                          shift=evaluate(scorer, x_shift, y_shift, seen, data, config, device))
            if section26 and method == "replay":
                counts = np.bincount(data.y_train, minlength=len(data.class_order))
                prior = torch.tensor(np.log(counts / counts.sum()), dtype=torch.float32, device=device)
                record["la_id"] = evaluate(scorer, data.x_test, data.y_test, seen, data, config, device, prior)
                record["la_definition"] = "logit + log(original TRAIN prior), tau=1; Section25 adaptation, not Menon Eq9 reproduction"
            record["evaluation_seconds"] = time.perf_counter() - started
            del scorer
            save_json(metric_path, record)
            print(f"CHECKPOINT_EVALUATED {identity['dataset']} {method} seed={seed} task={t}", flush=True)
        require_finite(record)
        records.append(record)
        previous_seen = seen
    result = {"schema_version": 1, "completed": True, "identity": identity,
              "class_order": data.class_order, "tasks": data.tasks,
              "train_counts": {str(c): int(np.sum(data.y_train == c)) for c in range(len(data.class_order))},
              "checkpoints": records, "training_seconds_total": cumulative,
              "setting_metadata": metadata, "policies": effective_policies(config),
              "command": subprocess.list2cmdline(["python", "-I", "ncd/run_problem_scout.py", *sys.argv[1:]]),
              "log_path": path_label(output.parents[1] / "run.log")}
    if section26:
        result["completed_at"] = datetime.now().astimezone().isoformat()
        result["section26_policy"] = SECTION26_POLICY
        result["section26_schema_version"] = 2
    save_json(final_path, result)
    print(f"CELL_COMPLETE {identity['dataset']} {method} seed={seed}", flush=True)
    return result


def final_rows(result):
    rows = []
    for split in ("id", "shift"):
        metrics = result["checkpoints"][-1][split]
        if metrics is None:
            continue
        for row in metrics["per_class"]:
            rows.append({k: result["identity"][k] for k in ("dataset", "method", "seed")} |
                        {"split": split} | row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "ncd/configs/problem_scout_section21.json")
    parser.add_argument("--data-root", help="Prepared Section21 cache root, relative to repository")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS)
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--min-steps-per-task", type=int,
                        help="Minimum optimizer updates per task; retain all configured epochs before extra batches")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--section26", action="store_true", help="Section26.9 benign refresh, old-attack memory-only, matched permanent holdout")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.data_root:
        config["problem_scout_data_root"] = args.data_root
    if args.device:
        config["device"] = args.device
    if args.min_steps_per_task is not None:
        if args.min_steps_per_task < 0:
            parser.error("--min-steps-per-task must be nonnegative")
        if args.min_steps_per_task:
            config["min_steps_per_task"] = args.min_steps_per_task
        else:
            config.pop("min_steps_per_task", None)
    if int(config.get("min_steps_per_task", 0)) < 0:
        parser.error("config min_steps_per_task must be nonnegative")
    datasets, methods, seeds = args.datasets or config["datasets"], args.methods or config["methods"], args.seeds or config["seeds"]
    if args.section26:
        config["section26"] = True
        config["section26_access_policy"] = SECTION26_POLICY
        config["section26_schema_version"] = 2
        if not args.methods:
            methods = ["replay", "wacb", "freq", "farq", "farq_all", "derpp", "joint"]
        if any(m not in {"replay", "wacb", "freq", "farq", "farq_all", "derpp", "joint"} for m in methods):
            parser.error("Section26 only supports its matched comparisons")
        if "CSE-CIC-IDS2018" in datasets:
            parser.error("Section26 excludes CSE18")
        if int(config["buffer_per_class"]) != 200:
            parser.error("Section26 fixes M=200*seen, including validation memory")
    elif any(m in {"wacb", "freq", "farq", "farq_all"} for m in methods):
        parser.error("Quota methods require --section26")
    if args.smoke:
        seeds = [1]
        config.update(epochs_per_task=1, buffer_per_class=int(config["buffer_per_class"]) if args.section26 else min(16, int(config["buffer_per_class"])))
        if config.get("min_steps_per_task", 0):
            config["smoke_requested_min_steps_per_task"] = config["min_steps_per_task"]
            config["min_steps_per_task"] = 0
    config["smoke"] = bool(args.smoke)
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required but not available")
    torch.set_num_threads(int(config.get("cpu_threads", 2)))
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_paths = ["ncd/run_problem_scout.py", "ncd/prepare_problem_scout_data.py", "ncd/crr_benchmarks.py",
                    "ncd/run_crr_real_sequence.py", "ncd/model.py", "ncd/train_cil.py", "ncd/data.py",
                    "ncd/source_commit.py", "ncd/groundtruth.py", "src/methods/replay_dpmeans.py",
                    "ncd/exp_h_data.py", "ncd/audit_quality_tabular.py"]
    provenance = {"git_commit": source_commit(ROOT), "effective_config": config,
                  "config_sha256": digest(args.config), "effective_config_sha256": canonical_hash(config),
                  "source_hashes": {p: digest(ROOT / p) for p in source_paths},
                  "runtime_versions": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                                       "numpy": np.__version__, "sklearn": sklearn.__version__,
                                       "cuda": torch.version.cuda}, "derpp_reference": DERPP_REFERENCE}
    result_path = out / "result.json"
    if result_path.exists():
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        if prior["provenance"] != provenance or prior["expected"] != {"datasets": datasets, "methods": methods, "seeds": seeds}:
            raise ValueError("matrix resume source/config/selection changed; preserve old output and use new directory")
    with (out / "run.log").open("a", encoding="utf-8") as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        try:
            from ncd.prepare_problem_scout_data import prepare_setting
            cells, rows = [], []
            total = len(datasets) * len(methods) * len(seeds)
            for dataset in datasets:
                data, x_shift, y_shift, metadata = prepare_setting(dataset, config)
                if args.section26:
                    x_shift, y_shift = None, None
                if args.smoke:
                    data, x_shift, y_shift = smoke_subset(data, x_shift, y_shift)
                data_hashes = binding(data, x_shift, y_shift)
                if args.section26:
                    selection_path, selection = None, {"lambdas": {}, "git_commit": None}
                else:
                    selection_path = ROOT / config["lambda_files"].get(dataset, config["lambda_files"]["default"])
                    selection = json.loads(selection_path.read_text(encoding="utf-8"))
                    if selection.get("selection_seeds") != [5, 6]:
                        raise ValueError("expected previously registered calibration seeds 5,6")
                for method in methods:
                    for seed in seeds:
                        regularization = selection["lambdas"].get(method, 0.0)
                        identity = provenance | {"dataset": dataset, "method": method, "seed": seed,
                                                 "class_order": data.class_order, "tasks": data.tasks,
                                                 "feature_names": data.feature_names, "data_binding": data_hashes,
                                                 "data_metadata_sha256": canonical_hash(metadata),
                                                 "regularization": regularization, "lambda_file": path_label(selection_path) if selection_path else None,
                                                 "lambda_file_sha256": digest(selection_path) if selection_path else None, "lambda_source_commit": selection["git_commit"]}
                        if args.section26:
                            identity["section26_access_policy"] = SECTION26_POLICY
                            identity["offline_joint_reference"] = method == "joint"
                            train_pools, valid_pools = section26_pools(data, seed)
                            identity["section26_holdout"] = {
                                str(c): {"train_count": len(train_pools[c]), "validation_count": len(valid_pools[c]),
                                         "validation_indices_sha256": hashlib.sha256(valid_pools[c].tobytes()).hexdigest()}
                                for c in valid_pools}
                            del train_pools, valid_pools
                        slug = dataset.lower().replace("-", "_")
                        cell_dir = out / slug / f"{method}_seed{seed}"
                        result = run_cell(data, x_shift, y_shift, metadata, config, identity, cell_dir, device, args.evaluate_only)
                        rows.extend(final_rows(result))
                        cells.append({"dataset": dataset, "method": method, "seed": seed,
                                      "result": (cell_dir / "result.json").relative_to(out).as_posix(),
                                      "result_sha256": digest(cell_dir / "result.json"),
                                      "checkpoints": len(result["checkpoints"]), "training_seconds_total": result["training_seconds_total"]})
                        snapshot = {"schema_version": 1, "execution_status": "complete" if len(cells) == total else "in_progress",
                                    "smoke": args.smoke, "provenance": provenance,
                                    "expected": {"datasets": datasets, "methods": methods, "seeds": seeds},
                                    "expected_cells": total, "cells": cells, "final_rows": rows,
                                    "training_seconds_total": sum(c["training_seconds_total"] for c in cells),
                                    "command": subprocess.list2cmdline(["python", "-I", "ncd/run_problem_scout.py", *sys.argv[1:]]),
                                    "log_path": path_label(out / "run.log"), "policies": effective_policies(config)}
                        save_json(result_path, snapshot)
                        _write_csv(out / "final_rows.csv", rows)
                # Avoid retaining a full prepared dataset while the next one loads.
                del data, x_shift, y_shift
            print(f"MATRIX_COMPLETE cells={len(cells)}/{total} smoke={args.smoke} result={path_label(result_path)}", flush=True)
        except Exception:
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
