from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import MatrixReservoir, array_sha256


def squared_distances(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    centroids = np.asarray(centroids, dtype=np.float32)
    result = (
        np.einsum("ij,ij->i", values, values)[:, None]
        + np.einsum("ij,ij->i", centroids, centroids)[None, :]
        - 2.0 * values @ centroids.T
    )
    np.maximum(result, 0.0, out=result)
    return result


def auto_lambda(values: np.ndarray, quantile: float, rng: np.random.Generator) -> float:
    values = np.asarray(values, dtype=np.float32)
    if len(values) > 500:
        values = values[rng.choice(len(values), 500, replace=False)]
    if len(values) < 2:
        return 1.0
    distances = np.sqrt(squared_distances(values, values), dtype=np.float32)
    upper = distances[np.triu_indices(len(values), k=1)]
    positive = upper[upper > 0]
    return float(np.quantile(positive, quantile)) if len(positive) else 1.0


def bounded_dpmeans(
    values: np.ndarray,
    *,
    quantile: float,
    max_centroids: int,
    max_iter: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """Full DP-Means on a bounded reservoir, followed by stable top-K.

    ``max_centroids`` limits only the retained routing state. Discovery itself
    is uncapped, matching the frozen cap-arm protocol for at most 3000 rows.
    """
    values = np.asarray(values, dtype=np.float32)
    if not len(values):
        raise ValueError("cannot fit a router on zero rows")
    lam = auto_lambda(values, quantile, rng)
    centroid_storage = np.empty_like(values)
    centroid_storage[0] = values[0]
    centroid_count = 1
    labels = np.full(len(values), -1, dtype=np.int32)
    for _ in range(max_iter):
        new_labels = np.zeros(len(values), dtype=np.int32)
        for index, row in enumerate(values):
            current = centroid_storage[:centroid_count]
            distance = np.linalg.norm(current - row, axis=1)
            nearest = int(np.argmin(distance))
            if distance[nearest] > lam and centroid_count < len(values):
                centroid_storage[centroid_count] = row
                new_labels[index] = centroid_count
                centroid_count += 1
            else:
                new_labels[index] = nearest
        updated = centroid_storage[:centroid_count].copy()
        for centroid_id in range(centroid_count):
            selected = values[new_labels == centroid_id]
            if len(selected):
                updated[centroid_id] = selected.mean(axis=0)
        if np.array_equal(new_labels, labels):
            centroid_storage[:centroid_count] = updated
            labels = new_labels
            break
        centroid_storage[:centroid_count] = updated
        labels = new_labels
    discovered = centroid_storage[:centroid_count].copy()
    discovered_counts = np.bincount(labels, minlength=centroid_count).astype(np.int64)
    retained_indices = np.argsort(-discovered_counts, kind="stable")[:max_centroids]
    retained = discovered[retained_indices]
    retained_counts = discovered_counts[retained_indices]
    stats = {
        "discovered_centroid_count": int(centroid_count),
        "discovered_centroid_sha256": array_sha256(discovered),
        "discovered_count_sum": int(discovered_counts.sum()),
        "retained_centroid_count": int(len(retained)),
        "retained_count_sum": int(retained_counts.sum()),
        "retention": "stable_top_k_by_cluster_size_after_uncapped_discovery",
    }
    return retained, retained_counts, lam, stats


class OnePassBudgetRouter:
    """Bounded-state cumulative-mean refinement from an auditable state."""

    algorithm = "incremental_budgeted_centroid_refinement_v3"

    def __init__(
        self,
        width: int,
        *,
        initial_centroids: np.ndarray,
        initial_counts: np.ndarray | None = None,
        initial_lambda: float,
        max_centroids: int,
    ):
        self.width = int(width)
        self.max_centroids = int(max_centroids)
        centroids = np.asarray(initial_centroids, dtype=np.float32)
        if (
            centroids.ndim != 2
            or centroids.shape[1] != self.width
            or not 0 < len(centroids) <= self.max_centroids
            or not np.isfinite(centroids).all()
        ):
            raise ValueError("invalid shared router initial centroids")
        if not np.isfinite(initial_lambda) or initial_lambda <= 0:
            raise ValueError("invalid shared router initial lambda")
        self.centroids = centroids.copy()
        if initial_counts is None:
            counts = np.zeros(len(centroids), dtype=np.int64)
        else:
            counts = np.asarray(initial_counts, dtype=np.int64)
            if counts.shape != (len(centroids),) or (counts < 0).any():
                raise ValueError("invalid shared router initial counts")
            counts = counts.copy()
        self.counts = counts
        self.lam = float(initial_lambda)
        self.initial_centroid_count = int(len(centroids))
        self.initial_centroid_sha256 = array_sha256(centroids)
        self.initial_count_sha256 = array_sha256(counts)
        self.base_rows = int(counts.sum())
        self.added_rows = 0

    def update(self, batch: np.ndarray) -> None:
        values = np.asarray(batch, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.width or not len(values):
            raise ValueError("invalid router batch")
        self.added_rows += len(values)
        self._refine(values)

    def _refine(self, values: np.ndarray) -> None:
        remaining = np.asarray(values, dtype=np.float32)
        while len(self.centroids) < self.max_centroids and len(remaining):
            distance_sq = squared_distances(remaining, self.centroids)
            nearest_sq = distance_sq.min(axis=1)
            farthest = int(np.argmax(nearest_sq))
            if float(np.sqrt(nearest_sq[farthest])) <= self.lam:
                break
            self.centroids = np.vstack([self.centroids, remaining[farthest]])
            self.counts = np.append(self.counts, 0)
        labels = squared_distances(remaining, self.centroids).argmin(axis=1)
        for centroid_id in np.unique(labels):
            selected = remaining[labels == centroid_id]
            old_count = int(self.counts[centroid_id])
            new_count = old_count + len(selected)
            self.centroids[centroid_id] = (
                self.centroids[centroid_id] * old_count + selected.sum(axis=0)
            ) / new_count
            self.counts[centroid_id] = new_count
    def finalise(self) -> tuple[np.ndarray, np.ndarray, float]:
        total_rows = self.base_rows + self.added_rows
        if total_rows <= 0:
            raise RuntimeError("cannot finalise a router without refinement rows")
        if int(self.counts.sum()) != total_rows:
            raise RuntimeError(
                f"one-pass accounting mismatch: counts={self.counts.sum()}, rows={total_rows}"
            )
        return self.centroids.copy(), self.counts.copy(), float(self.lam)


@dataclass
class FamilyRouterState:
    centroids: np.ndarray
    counts: np.ndarray
    lam: float
    stats: dict


class DualRouter:
    def __init__(self):
        self.cap: dict[int, FamilyRouterState] = {}
        self.uncapped: dict[int, FamilyRouterState] = {}

    @staticmethod
    def _distances(embeddings: np.ndarray, states: dict[int, FamilyRouterState], classes: list[int]) -> np.ndarray:
        result = np.empty((len(embeddings), len(classes)), dtype=np.float32)
        for column, class_id in enumerate(classes):
            distance_sq = squared_distances(embeddings, states[class_id].centroids)
            result[:, column] = -np.sqrt(distance_sq.min(axis=1), dtype=np.float32)
        return result

    def scores(self, embeddings: np.ndarray, classes: list[int], variant: str) -> np.ndarray:
        states = self.cap if variant == "cap3000" else self.uncapped
        raw = self._distances(embeddings, states, classes)
        mean = raw.mean(axis=1, keepdims=True)
        std = raw.std(axis=1, keepdims=True) + 1e-8
        return (raw - mean) / std


def cap_state_from_reservoir(
    reservoir: MatrixReservoir,
    *,
    quantile: float,
    max_centroids: int,
    rng: np.random.Generator,
) -> FamilyRouterState:
    samples = reservoir.array()
    selected_indices = reservoir.indices()
    centroids, counts, lam, fit_stats = bounded_dpmeans(
        samples,
        quantile=quantile,
        max_centroids=max_centroids,
        max_iter=10,
        rng=rng,
    )
    return FamilyRouterState(
        centroids=centroids,
        counts=counts,
        lam=lam,
        stats={
            "algorithm": "full_dpmeans_then_stable_topk_initial_fit_v2",
            "sample_total": int(reservoir.seen),
            "sample_used": int(reservoir.retained),
            "reservoir_sha256": array_sha256(samples),
            "selected_index_sha256": array_sha256(np.sort(selected_indices)),
            "selected_index_count": int(reservoir.retained),
            "selected_indices_unique": bool(
                len(np.unique(selected_indices)) == len(selected_indices)
            ),
            "selected_index_min": int(selected_indices.min()),
            "selected_index_max": int(selected_indices.max()),
            "fit_initial_centroid_count": int(len(centroids)),
            "fit_initial_centroid_sha256": array_sha256(centroids),
            "fit_initial_count_sha256": array_sha256(counts),
            "lambda": float(lam),
            **fit_stats,
        },
    )


def matched_cap_state(
    initial: FamilyRouterState,
    reservoir: MatrixReservoir,
    *,
    max_centroids: int,
    batch_size: int,
) -> FamilyRouterState:
    samples = reservoir.array()
    indices = reservoir.indices()
    if len(indices) != len(np.unique(indices)):
        raise RuntimeError("cap reservoir contains duplicate stream indices")
    order = np.argsort(indices, kind="stable")
    ordered = samples[order]
    router = OnePassBudgetRouter(
        ordered.shape[1],
        initial_centroids=initial.centroids,
        initial_counts=np.zeros(len(initial.centroids), dtype=np.int64),
        initial_lambda=initial.lam,
        max_centroids=max_centroids,
    )
    for start in range(0, len(ordered), batch_size):
        router.update(ordered[start : start + batch_size])
    centroids, counts, lam = router.finalise()
    return FamilyRouterState(
        centroids=centroids,
        counts=counts,
        lam=lam,
        stats={
            **initial.stats,
            "algorithm": "full_dpmeans_topk_plus_original_order_matched_refinement_v3",
            "matched_refinement_rows": int(len(ordered)),
            "matched_refinement_order": "ascending_original_stream_index",
            "centroid_count": int(len(centroids)),
            "centroid_sha256": array_sha256(centroids),
            "count_sha256": array_sha256(counts),
            "count_sum": int(counts.sum()),
        },
    )


def uncapped_state(
    router: OnePassBudgetRouter,
    *,
    selected_index_sha256: str,
) -> FamilyRouterState:
    centroids, counts, lam = router.finalise()
    total_rows = router.base_rows + router.added_rows
    return FamilyRouterState(
        centroids=centroids,
        counts=counts,
        lam=lam,
        stats={
            "algorithm": "matched_cap_state_plus_complement_refinement_v3",
            "sample_total": int(total_rows),
            "sample_used": int(total_rows),
            "base_cap_rows": int(router.base_rows),
            "added_rows": int(router.added_rows),
            "cap_selected_index_sha256": selected_index_sha256,
            "shared_initial_centroid_count": router.initial_centroid_count,
            "shared_initial_centroid_sha256": router.initial_centroid_sha256,
            "shared_initial_count_sha256": router.initial_count_sha256,
            "shared_initial_lambda": float(router.lam),
            "centroid_count": int(len(centroids)),
            "centroid_sha256": array_sha256(centroids),
            "count_sha256": array_sha256(counts),
            "lambda": float(lam),
            "count_sum": int(counts.sum()),
            "memory_complexity": "O(batch*K + K*d)",
            "complement_refinement_passes": 1,
            "router_construction_passes_over_class_shards": 2,
        },
    )
