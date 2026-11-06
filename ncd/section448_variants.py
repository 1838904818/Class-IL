"""Section 448 screening interventions for the Section 156 AR-Stateless arm."""
from __future__ import annotations

import hashlib

import numpy as np
from sklearn.cluster import KMeans

from ncd import run_aacil as aa

VARIANTS = ("V1a", "V1b", "V2", "V3")
KMEANS_N_INIT = "auto"


def row_ids_sha256(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def cap_new_class_rows(ids, quota, variant, seed):
    """Cap only newly introduced optimization rows; validation roles stay fixed."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown §448 variant: {variant}")
    multiplier = 4 if variant in ("V1b", "V3") else 1
    cap = int(quota) * multiplier
    ids = np.asarray(ids, dtype=np.int64)
    selected = aa.select_rows(ids, min(len(ids), cap), int(seed))
    return selected, {"quota_rows": int(quota), "cap_rows": cap,
                      "available_rows": int(len(ids)), "selected_rows": int(len(selected)),
                      "selected_ids_sha256": row_ids_sha256(selected)}


def _cluster_rows(ids, x, count, seed):
    """Choose one nearest observed row per fitted cluster, with stable ties."""
    ids = np.asarray(ids, dtype=np.int64)
    x = np.asarray(x)
    if x.ndim != 2 or len(ids) != len(x) or len(np.unique(ids)) != len(ids):
        raise ValueError("Invalid row identities or features for diversity selection")
    k = min(int(count), len(ids))
    if k < 1:
        return np.empty(0, dtype=np.int64), {"clusters": 0, "candidate_rows": int(len(ids))}
    if k == len(ids):
        picked = np.sort(ids.copy())
        return picked, {"clusters": k, "candidate_rows": int(len(ids)),
                        "selected_rows": int(len(picked)), "selection": "all candidates retained"}
    fitted = KMeans(n_clusters=k, random_state=int(seed), n_init=KMEANS_N_INIT,
                    algorithm="lloyd").fit(x)
    picked = []
    for cluster in range(k):
        members = np.flatnonzero(fitted.labels_ == cluster)
        if not len(members):
            raise RuntimeError("KMeans produced an empty cluster; do not alter k or retry silently")
        center = fitted.cluster_centers_[cluster]
        distances = np.sum((x[members].astype(np.float64) - center.astype(np.float64)) ** 2,
                           axis=1, dtype=np.float64)
        best_distance = distances.min()
        ties = members[distances == best_distance]
        picked.append(int(ids[ties].min()))
    picked = np.asarray(sorted(picked), dtype=np.int64)
    if len(picked) != k or len(np.unique(picked)) != k:
        raise AssertionError("Diversity selector did not return one unique observed row per cluster")
    return picked, {"clusters": k, "candidate_rows": int(len(ids)),
                    "selected_rows": int(len(picked)), "selection": "nearest observed row within its KMeans cluster"}


def refresh_diverse_attack_memory(data, train, valid, old, new_attacks, targets, seed, task):
    """Use KMeans for attack TRAIN memory; old classes can only shrink retained rows."""
    memory, details = {}, {}
    new_attacks = set(new_attacks)
    for c, capacity in sorted(targets.items()):
        train_quota, valid_quota = aa.slot_split(capacity)
        task_seed = (seed * 10000 + task) * 10007 + c
        if c in new_attacks:
            pool = np.asarray(train[c], dtype=np.int64)
            selected, selection = _cluster_rows(pool, data.x_train[pool], train_quota, task_seed)
            valid_ids = aa.select_rows(valid[c], valid_quota, task_seed + 53)
            train_ids = selected
            train_x, valid_x = data.x_train[train_ids].copy(), data.x_train[valid_ids].copy()
            source = "fresh current-class TRAIN rows"
        else:
            previous = old[c]
            prior_ids = np.asarray(previous["train_ids"], dtype=np.int64)
            if train_quota < len(prior_ids):
                selected, selection = _cluster_rows(prior_ids, previous["train_x"], train_quota, task_seed)
            else:
                selected = np.sort(prior_ids.copy())
                selection = {"clusters": int(len(selected)), "candidate_rows": int(len(prior_ids)),
                             "selected_rows": int(len(selected)), "selection": "retained; no expansion"}
            valid_prior = np.asarray(previous["validation_ids"], dtype=np.int64)
            if valid_quota < len(valid_prior):
                positions = aa.select_rows(np.arange(len(valid_prior)), valid_quota, task_seed + 53)
                valid_ids = valid_prior[positions].copy()
            else:
                valid_ids = np.sort(valid_prior.copy())
            train_ids, valid_ids = selected, valid_ids
            # IDs may have inherited an older order; map through retained state only.
            lookup = {int(row): i for i, row in enumerate(prior_ids)}
            positions = np.asarray([lookup[int(row)] for row in train_ids], dtype=np.int64)
            train_x = previous["train_x"][positions].copy()
            valid_lookup = {int(row): i for i, row in enumerate(valid_prior)}
            valid_x = previous["validation_x"][np.asarray([valid_lookup[int(row)] for row in valid_ids])].copy()
            source = "previously retained TRAIN rows only"
            if not set(train_ids).issubset(set(prior_ids)) or not set(valid_ids).issubset(set(valid_prior)):
                raise AssertionError("Diversity memory attempted to reread an old attack pool")
        if len(set(train_ids).intersection(valid_ids)) or len(train_ids) + len(valid_ids) > capacity:
            raise AssertionError("Attack training/validation memory overlap or capacity overflow")
        memory[c] = {"train_ids": train_ids, "validation_ids": valid_ids,
                     "train_x": train_x, "validation_x": valid_x}
        details[str(c)] = {**selection, "train_quota": int(train_quota),
                           "validation_quota": int(valid_quota), "train_rows": int(len(train_ids)),
                           "validation_rows": int(len(valid_ids)), "source": source,
                           "train_ids_sha256": row_ids_sha256(train_ids),
                           "validation_ids_sha256": row_ids_sha256(valid_ids)}
    return memory, details
