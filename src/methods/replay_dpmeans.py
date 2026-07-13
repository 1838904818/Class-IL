"""Replay with DP-Means exemplar selection — the paper's core contribution (D7).

Standard Replay picks exemplars uniformly at random, which can miss underrepresented
sub-structures (rare attack variants, drifted traffic patterns).

DP-Means (Kulis & Jordan 2012) is a non-parametric clustering method derived from
a Dirichlet Process mixture model with a distance threshold λ:
  - If a point is > λ away from all existing centroids, it spawns a NEW cluster.
  - Otherwise it joins the nearest centroid.
  - λ is auto-set to the 90th-percentile pairwise feature distance, balancing
    granularity vs. buffer size.

After clustering each class's feature space, we pick ONE exemplar per cluster
(the sample closest to the centroid), then keep at most `buffer_per_class`
exemplars by retaining the largest clusters.

Why this is better than random Replay:
  - Uniform random sampling over-represents the dominant mode of each class.
  - Herding (iCaRL) greedily matches the *mean* but still uses a single global mode.
  - DP-Means preserves multiple sub-clusters, which matters when attack traffic
    is multi-modal (e.g. slow vs. fast DoS variants).
"""
import numpy as np
import torch

from src.config import (
    EPOCHS_PER_TASK, REPLAY_BUFFER_PER_CLASS,
    DPMEANS_LAMBDA, DPMEANS_MAX_ITER,
)
from src.methods.base import build_model, train_one_task, subset_by_classes
from src.metrics import evaluate_task


# ---------------------------------------------------------------------------
# DP-Means clustering
# ---------------------------------------------------------------------------

def _auto_lambda(features: np.ndarray, quantile: float = 0.30,
                 max_sample: int = 500) -> float:
    """Auto-select λ as the `quantile`-th pairwise distance on a random sub-sample.

    Low quantile (0.30) ensures we create many fine-grained clusters, so the
    budget k is then enforced via the greedy diverse subsample path.
    High quantile (0.90) created too few clusters (< k), breaking replay quality.
    """
    if len(features) > max_sample:
        idx = np.random.choice(len(features), max_sample, replace=False)
        features = features[idx]
    dists = []
    for i in range(len(features)):
        d = np.linalg.norm(features[i] - features, axis=1)
        dists.extend(d[d > 0].tolist())
    return float(np.quantile(dists, quantile)) if dists else 1.0


def _dpmeans_cluster(features: np.ndarray, lam: float,
                     max_iter: int = DPMEANS_MAX_ITER):
    """Online DP-Means clustering.

    Args:
        features: (N, D)
        lam:      distance threshold for spawning new clusters
        max_iter: number of Lloyd-style refinement iterations

    Returns:
        centroids: (K, D)
        labels:   (N,) cluster assignment for each point
    """
    N = len(features)
    centroids = [features[0].copy()]
    labels = np.zeros(N, dtype=np.int32)

    for _ in range(max_iter):
        # E-step: assign each point to nearest centroid, or spawn new cluster
        new_labels = np.zeros(N, dtype=np.int32)
        for n in range(N):
            x = features[n]
            cents = np.array(centroids)
            dists = np.linalg.norm(x - cents, axis=1)
            nearest = int(np.argmin(dists))
            if dists[nearest] > lam:
                centroids.append(x.copy())
                new_labels[n] = len(centroids) - 1
            else:
                new_labels[n] = nearest

        # M-step: recompute centroids
        K = len(centroids)
        new_cents = []
        for k in range(K):
            pts = features[new_labels == k]
            if len(pts) > 0:
                new_cents.append(pts.mean(axis=0))
            else:
                new_cents.append(centroids[k])  # keep unchanged if empty

        if np.array_equal(new_labels, labels):
            labels = new_labels
            centroids = new_cents
            break
        labels = new_labels
        centroids = new_cents

    return np.array(centroids), labels


def _dpmeans_select(features: np.ndarray, k: int,
                    lam: float | None = None,
                    max_cluster_points: int = 3000) -> np.ndarray:
    """Select up to k representative exemplars using DP-Means clustering.

    For large feature sets, clustering is done on a random subsample
    (`max_cluster_points`) for speed, but the final exemplar for each
    cluster is chosen from the FULL set (closest sample to centroid).

    Returns indices into `features`.
    """
    k = min(k, len(features))
    if len(features) <= k:
        return np.arange(len(features))

    # Subsample for clustering speed on large classes
    N = len(features)
    if N > max_cluster_points:
        sub_idx = np.random.choice(N, max_cluster_points, replace=False)
        features_sub = features[sub_idx]
    else:
        sub_idx = np.arange(N)
        features_sub = features

    if lam is None:
        lam = _auto_lambda(features_sub)

    centroids, _ = _dpmeans_cluster(features_sub, lam=lam)
    K = len(centroids)

    # For each cluster centroid, find the CLOSEST sample in the full feature set
    # (ensures exemplars are real training samples, not just sub-sample artifacts)
    selected = []
    for c_idx in range(K):
        centroid = centroids[c_idx]
        dists = np.linalg.norm(features - centroid, axis=1)
        nearest = int(np.argmin(dists))
        selected.append(nearest)

    # Deduplicate
    seen: set = set()
    deduped: list = []
    for idx in selected:
        if idx not in seen:
            seen.add(idx)
            deduped.append(idx)
    selected = deduped

    # Case 1: fewer cluster-representatives than budget → fill up with random samples
    if len(selected) < k:
        remaining_pool = [i for i in range(N) if i not in seen]
        if remaining_pool:
            extra = np.random.choice(
                remaining_pool,
                size=min(k - len(selected), len(remaining_pool)),
                replace=False,
            ).tolist()
            selected = selected + extra

    # Case 2: more cluster-representatives than budget → greedy diverse subsample
    elif len(selected) > k:
        kept = [selected[0]]
        remaining = selected[1:]
        while len(kept) < k and remaining:
            kept_feats = features[kept]
            dists_to_kept = np.array([
                np.min(np.linalg.norm(features[i] - kept_feats, axis=1))
                for i in remaining
            ])
            next_idx = remaining[int(np.argmax(dists_to_kept))]
            kept.append(next_idx)
            remaining.remove(next_idx)
        selected = kept

    return np.array(selected)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_replay_dpmeans(X_tr, y_tr, X_te, y_te, tasks, in_dim, n_classes,
                       buffer_per_class: int = REPLAY_BUFFER_PER_CLASS):
    """Replay with DP-Means exemplar selection (paper contribution)."""
    model = build_model(in_dim, n_classes)
    acc_matrix = np.full((len(tasks), len(tasks)), np.nan)
    buf_X: list[np.ndarray] = []
    buf_y: list[np.ndarray] = []

    for i, task_i in enumerate(tasks):
        Xi, yi = subset_by_classes(X_tr, y_tr, task_i)

        # Mix in past exemplars
        if buf_X:
            X_combined = np.vstack([Xi] + buf_X)
            y_combined = np.concatenate([yi] + buf_y)
        else:
            X_combined, y_combined = Xi, yi

        train_one_task(model, X_combined, y_combined, epochs=EPOCHS_PER_TASK)

        # DP-Means exemplar selection in feature space
        model.eval()
        with torch.no_grad():
            feats_all = model.extract_features(
                torch.from_numpy(Xi.astype(np.float32))
            ).numpy()

        lam = DPMEANS_LAMBDA  # None → auto-computed per class

        for c in task_i:
            mask = (yi == c)
            if mask.sum() == 0:
                continue
            feats_c = feats_all[mask]
            raw_idx_c = np.where(mask)[0]

            sel_local = _dpmeans_select(feats_c, buffer_per_class, lam=lam)
            sel_global = raw_idx_c[sel_local]

            buf_X.append(Xi[sel_global])
            buf_y.append(yi[sel_global])

        for j, task_j in enumerate(tasks[: i + 1]):
            acc_matrix[i, j] = evaluate_task(model, X_te, y_te, task_j)

    return acc_matrix, model
