"""B1 bounded replay primitives with no access to discarded old-class pools.

Fresh slots must be supplied only for newly admitted classes. Recurrent benign
exposure remains a separate current stream; its retained slot is not replenished.
"""
from __future__ import annotations

import math

import numpy as np


def capacities(setting, method, seen, normal, admission_training_counts):
    seen = sorted(set(int(c) for c in seen))
    if normal not in seen or method not in ("er_bal_tabt", "ar_tabt", "ar_mlp"):
        raise ValueError("Invalid B1 memory identity")
    classes = seen if method == "er_bal_tabt" else [c for c in seen if c != normal]
    if not classes:
        raise ValueError("B1 requires an attack in Task 0")
    if setting == "fixed200":
        total = 200 * len(seen)
        base, remainder = divmod(total, len(classes))
        return {c: base + int(i < remainder) for i, c in enumerate(classes)}
    if setting == "percent1":
        if any(c not in admission_training_counts or admission_training_counts[c] <= 0
               for c in classes):
            raise ValueError("Missing class training-flow count at admission")
        # Integer floor, without an undocumented minimum-sample inflation.
        return {c: int(admission_training_counts[c]) // 100 for c in classes}
    raise ValueError("Unknown registered B1 memory setting")


def _check_slot(slot):
    required = {"train_ids", "train_x", "validation_ids", "validation_x"}
    if set(slot) != required:
        raise ValueError("Unexpected replay slot fields")
    for split in ("train", "validation"):
        ids, x = slot[split + "_ids"], slot[split + "_x"]
        if (ids.ndim != 1 or x.ndim != 2 or len(ids) != len(x)
                or not np.issubdtype(ids.dtype, np.integer)
                or len(np.unique(ids)) != len(ids) or (ids < 0).any()
                or not np.isfinite(x).all()):
            raise ValueError("Invalid replay row identity or shape")
    if slot["train_x"].shape[1] != slot["validation_x"].shape[1]:
        raise ValueError("Replay schema differs between splits")
    if np.intersect1d(slot["train_ids"], slot["validation_ids"]).size:
        raise ValueError("Replay optimization and validation overlap")


def retain_memory(old, fresh, targets, *, seed, task):
    """Sample current-class slots; old-class selection sees retained arrays only.

    The caller separates permanent calibration and supplies fresh TRAIN/validation
    IDs before this function. No data object, full pool, or TEST argument is accepted.
    Validation consumes the stated memory budget, as in the original AR accounting.
    """
    if set(old) & set(fresh) or set(old) | set(fresh) != set(targets):
        raise ValueError("Old classes cannot be replenished or silently dropped")
    result = {}
    for c, capacity in sorted(targets.items()):
        if int(capacity) != capacity or capacity < 0:
            raise ValueError("Replay capacity must be a nonnegative integer")
        source = fresh[c] if c in fresh else old[c]
        _check_slot(source)
        # Preserve original AR split at capacity >= 5; smaller 1% budgets cannot
        # allocate more validation rows than their entire registered capacity.
        nv = min(capacity, max(5, math.ceil(capacity * 0.1)))
        counts = {"train": capacity - nv, "validation": nv}
        slot = {}
        for split, count in counts.items():
            rng_seed = (seed * 10000 + task) * 10007 + c
            if split == "validation":
                rng_seed += 53
            ids = source[split + "_ids"]
            positions = np.random.default_rng(rng_seed).permutation(len(ids))[:count]
            positions = positions[np.argsort(ids[positions])]
            slot[split + "_ids"] = ids[positions].copy()
            slot[split + "_x"] = source[split + "_x"][positions].copy()
        _check_slot(slot)
        if len(slot["train_ids"]) + len(slot["validation_ids"]) > capacity:
            raise AssertionError("B1 replay exceeded its budget")
        result[c] = slot
    all_train = [s["train_ids"] for s in result.values()]
    all_valid = [s["validation_ids"] for s in result.values()]
    for rows in (all_train, all_valid):
        if rows:
            joined = np.concatenate(rows)
            if len(np.unique(joined)) != len(joined):
                raise ValueError("One replay row assigned to multiple classes")
    if result and np.intersect1d(np.concatenate(all_train), np.concatenate(all_valid)).size:
        raise ValueError("Cross-class replay training/validation overlap")
    return result


def assemble_er_stage(current, retained, *, normal, calibration_ids):
    """Build ER inputs from current exposure and bounded retained slots only.

    Current contains newly admitted attacks and the recurrent benign stream.
    Old attacks cannot appear in current. Benign rows already exposed in the
    current stream are not duplicated by replay; validation never enters the
    optimization split or permanent Normal calibration pool.
    """
    if normal not in current or (set(current) & set(retained)) - {normal}:
        raise ValueError("ER current exposure must not revisit old attacks")
    calibration_ids = np.asarray(calibration_ids)
    if calibration_ids.ndim != 1 or not np.issubdtype(calibration_ids.dtype, np.integer):
        raise ValueError("Invalid permanent calibration row identities")
    for source in (current, retained):
        for slot in source.values():
            _check_slot(slot)
    arrays = {}
    for split in ("train", "validation"):
        rows, features, labels = [], [], []
        for c in sorted(set(current) | set(retained)):
            first = current.get(c)
            second = retained.get(c)
            chosen = []
            if first is not None:
                chosen.append((first[split + "_ids"], first[split + "_x"]))
            if second is not None:
                ids, x = second[split + "_ids"], second[split + "_x"]
                if first is not None:
                    common, left, right = np.intersect1d(
                        ids, first[split + "_ids"], return_indices=True)
                    if len(common) and not np.array_equal(x[left], first[split + "_x"][right]):
                        raise ValueError("Recurrent benign row changed its feature values")
                    keep = ~np.isin(ids, first[split + "_ids"])
                    ids, x = ids[keep], x[keep]
                chosen.append((ids, x))
            for ids, x in chosen:
                rows.append(ids)
                features.append(x)
                labels.append(np.full(len(ids), c, dtype=np.int64))
        ids = np.concatenate(rows)
        if len(np.unique(ids)) != len(ids) or np.intersect1d(ids, calibration_ids).size:
            raise ValueError("Duplicate ER rows or permanent calibration leakage")
        arrays[split + "_ids"] = ids
        arrays[split + "_x"] = np.concatenate(features)
        arrays[split + "_y"] = np.concatenate(labels)
    if np.intersect1d(arrays["train_ids"], arrays["validation_ids"]).size:
        raise ValueError("ER optimization and stopping validation overlap")
    return arrays
