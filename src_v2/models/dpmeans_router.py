"""DPMeans Router — extends Phase I exemplar DPMeans to LoRA routing.

Phase I usage:  cluster class features → pick one exemplar per cluster (Replay)
Phase II usage: maintain per-family centroids → route new flows to the right LoRA

Three new responsibilities beyond Phase I:
  (1) Routing — given embedding e, return most likely family name
  (2) New-family detection — if e is far from ALL centroids, flag as unknown
  (3) Federated merging — combine centroids from multiple sites into a
      consistent global set via DPMeans-on-centroids (hierarchical clustering)

The clustering core is identical to Phase I (auto-λ via 30th-percentile pairwise
distance, Lloyd-style refinement), reused via the import below.
"""
from __future__ import annotations

import numpy as np
import torch

from src.methods.replay_dpmeans import _auto_lambda, _dpmeans_cluster


class DPMeansRouter:
    """Per-family centroid memory with routing + spawn-detection.

    Internally stores:
        family → list[centroid embeddings]   (variable count per family)
        family → list[count]                 (samples per centroid, for EMA updates)

    The router can:
        .fit_family(family, embeddings)        — initial fit for a family
        .update_family(family, embeddings)     — EMA update centroids
        .route(embedding) → (family, distance) — argmin family for a new sample
        .is_novel(embedding, novelty_margin) → bool  — spawn-detection
        .federated_merge(other_router)         — combine centroids across sites
    """

    def __init__(
        self,
        lambda_quantile: float = 0.30,
        novelty_factor: float = 1.5,
        max_centroids_per_family: int = 32,
        ema_momentum: float = 0.9,
    ):
        self.lambda_quantile = lambda_quantile
        self.novelty_factor = novelty_factor
        self.max_centroids_per_family = max_centroids_per_family
        self.ema_momentum = ema_momentum

        # Internal state
        self.centroids: dict[str, np.ndarray] = {}   # family → (K_f, D)
        self.counts: dict[str, np.ndarray] = {}      # family → (K_f,)
        self.global_lambda: float | None = None      # learnt from all families combined

    # ------------------------------------------------------------------ fit
    def fit_family(self, family: str, embeddings: np.ndarray) -> int:
        """Cluster `embeddings` for `family` and store the centroids.

        Returns the number of centroids discovered.
        """
        if len(embeddings) == 0:
            return 0

        lam = _auto_lambda(embeddings, quantile=self.lambda_quantile)
        # Update global novelty threshold (running max)
        if self.global_lambda is None:
            self.global_lambda = lam
        else:
            self.global_lambda = max(self.global_lambda, lam)

        centroids, labels = _dpmeans_cluster(embeddings, lam=lam)

        # Cap at max_centroids_per_family by retaining largest clusters
        if len(centroids) > self.max_centroids_per_family:
            cluster_sizes = np.bincount(labels, minlength=len(centroids))
            top_k = np.argsort(cluster_sizes)[-self.max_centroids_per_family:]
            centroids = centroids[top_k]
            # Remap counts for kept clusters
            counts = cluster_sizes[top_k]
        else:
            counts = np.bincount(labels, minlength=len(centroids))

        self.centroids[family] = centroids.astype(np.float32)
        self.counts[family] = counts.astype(np.float32)
        return len(centroids)

    # ------------------------------------------------------------------ EMA update
    def update_family(self, family: str, embeddings: np.ndarray):
        """Online EMA refinement of family centroids.

        For each new embedding, snap to nearest centroid of this family, then
        EMA-update that centroid:
            c ← momentum · c + (1 - momentum) · e
        Maintains a "moving average prototype" for streaming deployment.
        """
        if family not in self.centroids:
            self.fit_family(family, embeddings)
            return
        cents = self.centroids[family]
        m = self.ema_momentum
        for e in embeddings:
            dists = np.linalg.norm(cents - e, axis=1)
            j = int(np.argmin(dists))
            cents[j] = m * cents[j] + (1 - m) * e
            self.counts[family][j] += 1.0
        self.centroids[family] = cents

    # ------------------------------------------------------------------ routing
    def route(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Return (family, min_distance) for a single embedding.

        Returns (None, +inf) if no family centroids exist yet.
        """
        best_family, best_dist = None, float("inf")
        for f, cents in self.centroids.items():
            d = float(np.min(np.linalg.norm(cents - embedding, axis=1)))
            if d < best_dist:
                best_family, best_dist = f, d
        return best_family, best_dist

    def route_batch(self, embeddings: np.ndarray) -> tuple[list[str | None], np.ndarray]:
        """Vectorised routing for a batch of embeddings.

        Returns (families, distances) where families[i] is the family for
        embeddings[i] and distances[i] is its min distance to that family's
        nearest centroid.
        """
        N = len(embeddings)
        if not self.centroids:
            return [None] * N, np.full(N, np.inf)

        all_min_d = np.full(N, np.inf)
        best_fam: list[str | None] = [None] * N
        for f, cents in self.centroids.items():
            # cdist: (N, K_f)
            d = np.linalg.norm(
                embeddings[:, None, :] - cents[None, :, :], axis=2
            ).min(axis=1)
            mask = d < all_min_d
            all_min_d[mask] = d[mask]
            for i in np.where(mask)[0]:
                best_fam[i] = f
        return best_fam, all_min_d

    # ------------------------------------------------------------------ novelty detection
    def is_novel(self, embedding: np.ndarray) -> bool:
        """A flow is `novel` if its distance to ALL known centroids exceeds
        novelty_factor · global_lambda.

        Use case: spawn a new LoRA when sustained novelty is observed.
        """
        if self.global_lambda is None or not self.centroids:
            return True
        _, d = self.route(embedding)
        return d > self.novelty_factor * self.global_lambda

    def novelty_score(self, embeddings: np.ndarray) -> np.ndarray:
        """Per-sample novelty score in [0, ∞):
            score = min_distance / (novelty_factor · global_lambda)
        > 1.0 means novel by the threshold rule.
        """
        if self.global_lambda is None or not self.centroids:
            return np.full(len(embeddings), np.inf)
        _, dists = self.route_batch(embeddings)
        return dists / (self.novelty_factor * self.global_lambda)

    # ------------------------------------------------------------------ federated
    def federated_merge(self, other: "DPMeansRouter") -> None:
        """Merge another site's centroids into ours.

        For each family present in either router, we concatenate centroids and
        re-cluster them with DP-Means (hierarchical: cluster of clusters).
        This bounds the global centroid count per family.
        """
        all_families = set(self.centroids) | set(other.centroids)
        for f in all_families:
            mine = self.centroids.get(f, np.zeros((0, 1)))
            theirs = other.centroids.get(f, np.zeros((0, 1)))
            # Align embedding dim if needed
            if mine.shape[1] != theirs.shape[1] and mine.size > 0 and theirs.size > 0:
                continue  # mismatch — skip
            if mine.size == 0:
                combined = theirs
            elif theirs.size == 0:
                combined = mine
            else:
                combined = np.vstack([mine, theirs])

            if len(combined) == 0:
                continue
            # Re-cluster the union (hierarchical DP-Means)
            lam = _auto_lambda(combined, quantile=self.lambda_quantile)
            new_cents, _ = _dpmeans_cluster(combined, lam=lam)
            if len(new_cents) > self.max_centroids_per_family:
                # Random downsample (preserves coverage; could refine later)
                idx = np.random.choice(
                    len(new_cents), self.max_centroids_per_family, replace=False
                )
                new_cents = new_cents[idx]
            self.centroids[f] = new_cents.astype(np.float32)
            self.counts[f] = np.ones(len(new_cents), dtype=np.float32)

        # Update global lambda
        self.global_lambda = max(
            self.global_lambda or 0.0,
            other.global_lambda or 0.0,
        )

    # ------------------------------------------------------------------ stats
    def n_centroids_total(self) -> int:
        return sum(c.shape[0] for c in self.centroids.values())

    def comm_payload_bytes(self) -> int:
        """Approximate bytes to transmit centroids over the network (float32)."""
        return sum(c.nbytes for c in self.centroids.values())

    def summary(self) -> str:
        lines = [
            f"DPMeansRouter: λ_global={self.global_lambda:.3f}"
            if self.global_lambda
            else "DPMeansRouter: empty"
        ]
        for f, c in self.centroids.items():
            lines.append(f"  {f:20s}: {c.shape[0]:3d} centroids, dim={c.shape[1]}")
        lines.append(f"  total centroids: {self.n_centroids_total()}")
        lines.append(f"  comm payload:    {self.comm_payload_bytes() / 1024:.1f} KB")
        return "\n".join(lines)
