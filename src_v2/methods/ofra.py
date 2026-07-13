"""OFRA — Oracle-Free Routed Adapters for Class-IL Intrusion Detection.

Method: a Class-IL framework that achieves near-zero catastrophic forgetting
by combining:
  (1) A supervised-pretrained encoder, frozen after Task 0
  (2) Per-attack-family LoRA adapter + binary classifier head
  (3) Greedy farthest-first raw-exemplar buffer for cross-task replay negatives
  (4) DPMeans router maintaining per-family centroid memory in embedding space
  (5) Joint inference using independent binary-head probabilities and router scores

The frozen encoder and isolated adapters prevent later parameter updates from
changing previously learned family modules. At joint inference, however, new
competing heads, independent head probabilities, and evolving router-centroid
normalisation can still change the selected family.

Components:
  encoder  — frozen MLPEncoder (or FlowTransformer)
  pool     — LoRAPool with per-family binary head
  router   — DPMeansRouter for inference routing + novelty detection
  buffer   — RawExemplarBuffer for cross-task replay (farthest-first sampling)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import BATCH_SIZE, EPOCHS_PER_TASK, LR
from src.methods.base import subset_by_classes, to_loader


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.75,
) -> torch.Tensor:
    """Focal loss for binary classification (Lin et al. 2017).

    L = -alpha * (1 - p_t)^gamma * log(p_t)

    For Class-IL minority families (e.g. U2R with 52 samples), focal loss
    massively up-weights hard positives the model would otherwise miss.
    `alpha=0.75` weights positives more than negatives (counteracts the
    intrinsic neg:pos imbalance even after our 4:1 sampling).
    """
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)  # probability assigned to the true class
    # alpha weighting per class
    alpha_t = torch.where(
        targets == 1,
        torch.full_like(logits[:, 0], alpha),
        torch.full_like(logits[:, 0], 1.0 - alpha),
    )
    return (alpha_t * (1.0 - pt) ** gamma * ce).mean()
from src_v2.models.dpmeans_router import DPMeansRouter
from src_v2.models.lora import LoRAPool
from src_v2.models.mlp_encoder import MLPEncoder
from src_v2.models.transformer_encoder import (
    FlowTransformer,
    masked_feature_reconstruction_loss,
)


# ===========================================================================
# Raw exemplar buffer — Tier 3 contribution
# ===========================================================================
class RawExemplarBuffer:
    """Per-family raw training sample buffer.

    Stores up to `capacity` actual flow samples per family for cross-task
    replay. Selection uses greedy farthest-first traversal on embeddings.
    """

    def __init__(self, capacity_per_family: int = 50):
        if capacity_per_family < 0:
            raise ValueError("capacity_per_family must be non-negative")
        self.capacity = int(capacity_per_family)
        self.samples: dict[str, np.ndarray] = {}   # family → (k, D) raw features
        self.labels: dict[str, np.ndarray] = {}    # family → (k,) global class ids

    def add(self, family: str, X: np.ndarray, y: np.ndarray, embeddings: np.ndarray):
        """Diverse sub-sample to `capacity` using farthest-first traversal."""
        X = np.asarray(X)
        y = np.asarray(y)
        embeddings = np.asarray(embeddings)
        if X.ndim != 2 or embeddings.ndim != 2 or y.ndim != 1:
            raise ValueError("X and embeddings must be 2-D and y must be 1-D")
        if not (len(X) == len(y) == len(embeddings)):
            raise ValueError("X, y, and embeddings must contain the same number of rows")
        if len(X) and (not np.isfinite(X).all() or not np.isfinite(embeddings).all()):
            raise ValueError("X and embeddings must contain only finite values")
        k = min(self.capacity, len(X))
        if k == 0:
            return
        # Greedy farthest-first selection on embeddings (cheap & diverse)
        sel = [int(np.random.randint(0, len(embeddings)))]
        embeds_arr = embeddings
        while len(sel) < k:
            kept = embeds_arr[sel]
            # min distance from each candidate to kept set
            d = np.linalg.norm(embeds_arr[:, None, :] - kept[None, :, :], axis=2).min(axis=1)
            d[sel] = -np.inf  # exclude already-selected
            sel.append(int(np.argmax(d)))
        self.samples[family] = X[sel].astype(np.float32)
        self.labels[family] = y[sel].astype(np.int64)

    def all(self) -> tuple[np.ndarray, np.ndarray]:
        """Concatenate all family exemplars."""
        if not self.samples:
            d = next(iter(self.samples.values()), np.zeros((0, 1))).shape[1] if self.samples else 1
            return np.zeros((0, d), dtype=np.float32), np.zeros(0, dtype=np.int64)
        X = np.vstack(list(self.samples.values()))
        y = np.concatenate(list(self.labels.values()))
        return X, y

    def all_except(self, family: str) -> tuple[np.ndarray, np.ndarray]:
        """All exemplars except those of `family`."""
        Xs, ys = [], []
        for f, X in self.samples.items():
            if f != family:
                Xs.append(X)
                ys.append(self.labels[f])
        if not Xs:
            return np.zeros((0, 1), dtype=np.float32), np.zeros(0, dtype=np.int64)
        return np.vstack(Xs), np.concatenate(ys)

    def n_total(self) -> int:
        return sum(len(v) for v in self.samples.values())


# ===========================================================================
# Main agent (single site)
# ===========================================================================
class OFRAAgent(nn.Module):
    """Single-site OFRA agent.

    Components:
      encoder  — frozen MLPEncoder or FlowTransformer
      pool     — LoRAPool with one LoRA adapter and binary head per family
      router   — DPMeansRouter centroid memory used in the joint score
      buffer   — farthest-first raw exemplars for replay negatives
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        chunk_size: int = 1,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        router_lambda_quantile: float = 0.30,
        novelty_factor: float = 1.5,
        exemplar_capacity: int = 50,
        encoder_type: str = "transformer",
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        # v0.4 — choose encoder backbone
        if encoder_type == "mlp":
            self.encoder: nn.Module = MLPEncoder(
                n_features=n_features,
                d_model=d_model,
                n_layers=n_layers,
            )
        elif encoder_type == "transformer":
            self.encoder = FlowTransformer(
                n_features=n_features,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                chunk_size=chunk_size,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        self.encoder_type = encoder_type
        # Binary per-family heads (n_local_classes=2) provide independent
        # positive-class probabilities at inference.
        self.pool = LoRAPool(d_model=d_model, rank=lora_rank, alpha=lora_alpha)
        self.router = DPMeansRouter(
            lambda_quantile=router_lambda_quantile,
            novelty_factor=novelty_factor,
        )
        self.buffer = RawExemplarBuffer(capacity_per_family=exemplar_capacity)

        self.class_to_family: dict[int, str] = {}
        self.family_to_class: dict[str, int] = {}
        self._encoder_frozen = False

        self.register_buffer("feature_mean", torch.zeros(n_features))
        self.register_buffer("feature_std", torch.ones(n_features))
        self._stats_fitted = False

    # ------------------------------------------------------------------ stats
    def _validate_features(
        self,
        X: np.ndarray,
        *,
        name: str = "X",
        allow_empty: bool = False,
    ) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"{name} must be a 2-D feature matrix")
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"{name} has {X.shape[1]} features; expected {self.n_features}"
            )
        if not allow_empty and len(X) == 0:
            raise ValueError(f"{name} must contain at least one row")
        if len(X) and not np.isfinite(X).all():
            raise ValueError(f"{name} must contain only finite values")
        return X

    def fit_input_stats(self, X: np.ndarray):
        X = self._validate_features(X, name="X")
        mean = X.mean(axis=0).astype(np.float32)
        std = X.std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
        self.feature_mean.copy_(torch.from_numpy(mean).to(self.feature_mean.device))
        self.feature_std.copy_(torch.from_numpy(std).to(self.feature_std.device))
        self._stats_fitted = True

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        if not self._stats_fitted:
            return X
        m = self.feature_mean.cpu().numpy()
        s = self.feature_std.cpu().numpy()
        return ((X - m) / s).astype(np.float32)

    # ------------------------------------------------------------------ pretraining
    def pretrain_encoder(
        self,
        X: np.ndarray,
        epochs: int = 5,
        batch_size: int = BATCH_SIZE,
        lr: float = 1e-3,
        mask_prob: float = 0.15,
        max_samples: int | None = None,
        verbose: bool = False,
    ):
        X = self._validate_features(X, name="X")
        if epochs < 0:
            raise ValueError("epochs must be non-negative")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive when provided")
        if max_samples is not None and len(X) > max_samples:
            idx = np.random.choice(len(X), max_samples, replace=False)
            X = X[idx]
        X = self._normalize(X)
        if verbose:
            print(f"  pretrain on {len(X)} samples (z-normalised)")

        self.encoder.train()
        opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        dev = next(self.encoder.parameters()).device
        Xt = torch.from_numpy(X.astype(np.float32)).to(dev)
        n = len(Xt)
        for ep in range(epochs):
            perm = torch.randperm(n, device=dev)
            losses = []
            for i in range(0, n, batch_size):
                bx = Xt[perm[i : i + batch_size]]
                loss = masked_feature_reconstruction_loss(
                    self.encoder, bx, mask_prob=mask_prob
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if verbose:
                print(f"  pretrain epoch {ep+1}/{epochs}: MSE={np.mean(losses):.4f}")

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        self._encoder_frozen = True

    def supervised_pretrain_encoder(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int,
        epochs: int = 5,
        batch_size: int = BATCH_SIZE,
        lr: float = 1e-3,
        verbose: bool = False,
    ):
        """Supervised pretraining on Task 0 / bootstrap data.

        Only meaningful for MLPEncoder. For FlowTransformer, this is a no-op
        (use `pretrain_encoder` for self-supervised MLM-style pretraining
        instead).
        """
        if not isinstance(self.encoder, MLPEncoder):
            if verbose:
                print("  supervised_pretrain skipped (not MLP encoder)")
            return
        X = self._validate_features(X, name="X")
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be 1-D and have the same number of rows as X")
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError("y must contain integer class identifiers")
        if n_classes <= 1:
            raise ValueError("n_classes must be greater than one")
        if epochs < 0 or batch_size <= 0:
            raise ValueError("epochs must be non-negative and batch_size must be positive")
        Xn = self._normalize(X)
        self.encoder.supervised_pretrain(
            Xn, y, n_classes=n_classes,
            epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose,
        )

    # ------------------------------------------------------------------ helpers
    @torch.no_grad()
    def embed(self, X: np.ndarray, batch: int = BATCH_SIZE,
              normalized: bool = False) -> np.ndarray:
        self.encoder.eval()
        X = self._validate_features(X, name="X", allow_empty=True)
        if batch <= 0:
            raise ValueError("batch must be positive")
        if len(X) == 0:
            return np.empty((0, self.d_model), dtype=np.float32)
        if not normalized:
            X = self._normalize(X)
        dev = next(self.encoder.parameters()).device
        Xt = torch.from_numpy(X.astype(np.float32)).to(dev)
        outs = []
        for i in range(0, len(Xt), batch):
            outs.append(self.encoder(Xt[i : i + batch]).cpu().numpy())
        return np.concatenate(outs, axis=0)

    @staticmethod
    def _family_name(c: int) -> str:
        return f"family_{c}"

    def _scalar_logits(self, embed: torch.Tensor) -> torch.Tensor:
        """Return (B, F) matrix of per-family confidence, one column per family.

        For binary heads (n_local_classes=2) this is the difference between
        positive and negative logits. For scalar heads (n_local_classes=1)
        this is the raw scalar. `head.scalar()` handles both cases.
        """
        logits = []
        for f in self.pool.families:
            head = self.pool.heads[f]
            s = head.scalar(embed)  # (B,)
            logits.append(s)
        return torch.stack(logits, dim=1)  # (B, F)

    def _family_index_of_class(self, y_global: np.ndarray) -> np.ndarray:
        """Map global class id → family-index in self.pool.families order."""
        out = np.full(len(y_global), -1, dtype=np.int64)
        for fi, fname in enumerate(self.pool.families):
            cls_of_f = self.family_to_class.get(fname, -1)
            out[y_global == cls_of_f] = fi
        return out

    # ------------------------------------------------------------------ Class-IL training (v0.5)
    def train_task(
        self,
        X_task: np.ndarray,
        y_task: np.ndarray,
        epochs: int = EPOCHS_PER_TASK,
        lr: float = LR,
        loss_fn: str = "focal",   # "ce" or "focal" (v0.5: focal default)
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,
        minority_threshold: int = 1000,
        verbose: bool = False,
    ):
        """v0.5 per-family binary training with raw-exemplar replay + focal loss.

        Training:
          - Each new family head is trained independently as a BINARY classifier
            (positive = own class samples, negative = current task other classes
             + raw exemplars from old families).
          - Old families' parameters stay completely frozen (true isolation).
        Inference:
          - Joint score over independent binary-head probabilities and router
            z-scores (in predict()).

        This decouples per-family optimization from cross-family competition,
        which was the failure mode of v0.2's joint softmax training.
        """
        X_task = self._validate_features(X_task, name="X_task")
        y_task = np.asarray(y_task)
        if y_task.ndim != 1 or len(y_task) != len(X_task):
            raise ValueError(
                "y_task must be 1-D and have the same number of rows as X_task"
            )
        if not np.issubdtype(y_task.dtype, np.integer):
            raise ValueError("y_task must contain integer class identifiers")
        if loss_fn not in {"focal", "ce"}:
            raise ValueError(
                f"Unsupported loss_fn={loss_fn!r}; expected 'focal' or 'ce'"
            )
        if epochs < 0:
            raise ValueError("epochs must be non-negative")
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be a positive finite value")
        if not np.isfinite(focal_gamma) or focal_gamma < 0:
            raise ValueError("focal_gamma must be a non-negative finite value")
        if not np.isfinite(focal_alpha) or not 0.0 <= focal_alpha <= 1.0:
            raise ValueError("focal_alpha must be a finite value in [0, 1]")
        if minority_threshold <= 0:
            raise ValueError("minority_threshold must be positive")

        new_classes = sorted(set(y_task.tolist()))
        repeated = [
            c for c in new_classes
            if c in self.class_to_family or self.pool.has(self._family_name(c))
        ]
        if repeated:
            raise ValueError(
                "Disjoint Class-IL tasks cannot repeat class identifiers: "
                f"{repeated}"
            )

        X_task_n = self._normalize(X_task)
        old_X, _old_y = self.buffer.all()
        old_usable = (
            len(old_X) > 0
            and old_X.ndim == 2
            and old_X.shape[1] == X_task_n.shape[1]
        )
        if len(old_X) > 0 and not old_usable:
            raise RuntimeError("Replay buffer has an incompatible feature shape")
        old_negative_count = len(old_X) if old_usable else 0
        without_negatives = [
            c
            for c in new_classes
            if np.count_nonzero(y_task != c) + old_negative_count == 0
        ]
        if without_negatives:
            raise ValueError(
                "Cannot train binary family heads without negative samples: "
                f"{without_negatives}"
            )

        # ----- step 1: register families after all protocol checks
        newly_added: list[str] = []
        for c in new_classes:
            fname = self.class_to_family.get(c) or self._family_name(c)
            self.class_to_family[c] = fname
            self.family_to_class[fname] = c
            if not self.pool.has(fname):
                # Use binary head (n_local_classes=2) for stable per-family training
                self.pool.add_family(fname, n_local_classes=2)
                newly_added.append(fname)
        # New family modules are created on CPU — keep pool on encoder's device
        self.pool.to(next(self.encoder.parameters()).device)

        if verbose:
            print(f"  classes {new_classes} → families "
                  f"{[self.class_to_family[c] for c in new_classes]}")
            print(f"  newly added LoRA heads: {newly_added}")

        # ----- step 2: get raw exemplars from old families (Tier 3)
        # These are stored as normalized features in self.buffer
        if verbose and len(old_X) > 0:
            print(f"  old-family exemplars in buffer: {len(old_X)}")

        # ----- step 3: train each new family as a CLASS-BALANCED binary classifier
        # Critical fix vs v0.2: pos:neg ratio held at 1:K (K=neg_ratio) so the
        # head doesn't collapse to "always predict negative" on minority classes
        # (R2L=995 + U2R=52 caused this in earlier iterations).
        neg_ratio = 4  # negatives = neg_ratio × positives
        self.encoder.eval()
        for fname in newly_added:
            cls_of_f = self.family_to_class[fname]
            pos_mask = (y_task == cls_of_f)
            X_pos = X_task_n[pos_mask]
            n_pos = len(X_pos)
            if n_pos == 0:
                continue

            # Collect all candidate negatives (current task others + old exemplars)
            neg_parts = [X_task_n[y_task != cls_of_f]]
            if len(old_X) > 0 and old_X.shape[1] == X_task_n.shape[1]:
                neg_parts.append(old_X.astype(np.float32))
            X_neg_all = np.vstack(neg_parts) if neg_parts else np.zeros((0, X_task_n.shape[1]))

            # Per-epoch resampling: each epoch sees a fresh random neg subset
            self.pool.freeze_all_except(fname)
            params = [p for p in self.pool.heads[fname].parameters() if p.requires_grad]
            opt = torch.optim.Adam(params, lr=lr)

            for ep in range(epochs):
                # Class-balanced sampling
                target_neg = min(len(X_neg_all), n_pos * neg_ratio)
                if target_neg > 0:
                    idx = np.random.choice(len(X_neg_all), target_neg, replace=False)
                    X_neg_ep = X_neg_all[idx]
                else:
                    X_neg_ep = np.zeros((0, X_task_n.shape[1]), dtype=np.float32)
                X_bin = np.vstack([X_pos, X_neg_ep]).astype(np.float32)
                y_bin = np.concatenate([
                    np.ones(n_pos, dtype=np.int64),
                    np.zeros(len(X_neg_ep), dtype=np.int64),
                ])
                loader = to_loader(X_bin, y_bin)

                losses, correct, tp, fp, fn, total = [], 0, 0, 0, 0, 0
                # Use focal loss on minority families (n_pos < threshold)
                use_focal = (loss_fn == "focal") and (n_pos < minority_threshold)
                _dev = next(self.encoder.parameters()).device
                for bx, by in loader:
                    bx, by = bx.to(_dev), by.to(_dev)
                    with torch.no_grad():
                        e = self.encoder(bx)
                    logits = self.pool.forward_single(e, fname)  # (B, 2)
                    if use_focal:
                        loss = focal_loss(logits, by, gamma=focal_gamma, alpha=focal_alpha)
                    else:
                        loss = F.cross_entropy(logits, by)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    losses.append(loss.item())
                    preds = logits.argmax(dim=1)
                    correct += (preds == by).sum().item()
                    tp += ((preds == 1) & (by == 1)).sum().item()
                    fp += ((preds == 1) & (by == 0)).sum().item()
                    fn += ((preds == 0) & (by == 1)).sum().item()
                    total += len(by)
                if verbose:
                    acc = correct / max(total, 1)
                    rec = tp / max(tp + fn, 1)
                    pre = tp / max(tp + fp, 1)
                    print(f"    [{fname}] ep {ep+1}/{epochs}: "
                          f"loss={np.mean(losses):.4f}, acc={acc:.3f}, "
                          f"P={pre:.3f}, R={rec:.3f} "
                          f"(n_pos={n_pos}, n_neg={len(X_neg_ep)})")

        # Re-freeze all
        self.pool.freeze_all()

        # ----- step 4: update DPMeans router + farthest-first exemplar buffer
        with torch.no_grad():
            embeds = self.embed(X_task_n, normalized=True)
        for c in new_classes:
            mask = (y_task == c)
            if mask.sum() == 0:
                continue
            fname = self.class_to_family[c]
            if fname in self.router.centroids:
                self.router.update_family(fname, embeds[mask])
            else:
                n_cents = self.router.fit_family(fname, embeds[mask])
                if verbose:
                    print(f"    router[{fname}] fit with {n_cents} centroids")
            # Tier 3: raw exemplar buffer (normalized features)
            self.buffer.add(fname, X_task_n[mask], y_task[mask], embeds[mask])

    # ------------------------------------------------------------------ inference (v0.4)
    @torch.no_grad()
    def predict(
        self,
        X: np.ndarray,
        return_routing: bool = False,
        novelty_threshold: float | None = None,
        router_weight: float = 0.5,
        head_weight: float = 1.0,
        calibration: str = "softmax_prob",
    ):
        """Joint inference over per-family binary heads.

        Head scalar logits are not on the same scale across families because
        each binary head is fitted independently. The default therefore uses
        each head's own positive-class probability in [0, 1]. These values are
        independent confidence scores, not a distribution across families.

        calibration:
            "softmax_prob" — independent P(positive | family head) from each
                             binary head; no cross-family normalization
            "raw_scalar"   — legacy raw positive-minus-negative logit

        Joint score:
            score_f = head_weight · head_prob[f] + router_weight · router_z[f]
        """
        if calibration not in {"softmax_prob", "raw_scalar"}:
            raise ValueError(
                f"Unsupported calibration={calibration!r}; expected "
                "'softmax_prob' or 'raw_scalar'"
            )
        for name, value in (
            ("router_weight", router_weight),
            ("head_weight", head_weight),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative finite value")
        if router_weight == 0 and head_weight == 0:
            raise ValueError("router_weight and head_weight cannot both be zero")
        if novelty_threshold is not None and (
            not np.isfinite(novelty_threshold) or novelty_threshold < 0
        ):
            raise ValueError(
                "novelty_threshold must be a non-negative finite value"
            )

        X = self._validate_features(X, name="X", allow_empty=True)
        preds = np.full(len(X), -1, dtype=np.int64)
        if len(X) == 0 or not self.pool.families:
            return (preds, []) if return_routing else preds

        bad_mappings = [
            family for family in self.pool.families
            if family not in self.family_to_class
        ]
        if bad_mappings:
            raise RuntimeError(
                f"Missing class mappings for family heads: {bad_mappings}"
            )
        bad_centroids = []
        for family in self.pool.families:
            cents = self.router.centroids.get(family)
            if (
                not isinstance(cents, np.ndarray)
                or cents.ndim != 2
                or len(cents) == 0
                or cents.shape[1] != self.d_model
                or not np.isfinite(cents).all()
            ):
                bad_centroids.append(family)
        if bad_centroids:
            raise RuntimeError(
                f"Missing or invalid router centroids for: {bad_centroids}"
            )

        self.encoder.eval()
        embeds_np = self.embed(X)
        embeds = torch.from_numpy(embeds_np.astype(np.float32)).to(
            next(self.encoder.parameters()).device)
        N = len(X)
        F_n = len(self.pool.families)

        # ---- independent per-head positive confidence
        head_scores = np.zeros((N, F_n), dtype=np.float32)
        for fi, fname in enumerate(self.pool.families):
            head = self.pool.heads[fname]
            logits = head(embeds)  # (N, 2)
            if calibration == "softmax_prob":
                # P("yes this family" | sample)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                head_scores[:, fi] = probs
            elif calibration == "raw_scalar":
                head_scores[:, fi] = (logits[:, 1] - logits[:, 0]).cpu().numpy()
        if not np.isfinite(head_scores).all():
            raise FloatingPointError("Non-finite family-head scores encountered")

        # ---- router scores
        router_scores = np.empty((N, F_n), dtype=np.float32)
        for fi, fname in enumerate(self.pool.families):
            cents = self.router.centroids[fname]
            d = np.linalg.norm(embeds_np[:, None, :] - cents[None, :, :], axis=2).min(axis=1)
            router_scores[:, fi] = -d
        if not np.isfinite(router_scores).all():
            raise FloatingPointError("Non-finite router scores encountered")
        rs_mean = router_scores.mean(axis=1, keepdims=True)
        rs_std = router_scores.std(axis=1, keepdims=True) + 1e-8
        router_scores_z = (router_scores - rs_mean) / rs_std

        joint = head_weight * head_scores + router_weight * router_scores_z
        if not np.isfinite(joint).all():
            raise FloatingPointError("Non-finite joint scores encountered")
        chosen_fi = joint.argmax(axis=1)
        for i, fi in enumerate(chosen_fi):
            fname = self.pool.families[fi]
            preds[i] = self.family_to_class.get(fname, -1)

        if novelty_threshold is not None and self.router.global_lambda is not None:
            _, min_dists = self.router.route_batch(embeds_np)
            novel = min_dists > novelty_threshold * self.router.global_lambda
            preds[novel] = -2

        if return_routing:
            routing = [
                {
                    "family": self.pool.families[chosen_fi[i]],
                    "router_score": float(router_scores[i, chosen_fi[i]]),
                    "head_score": float(head_scores[i, chosen_fi[i]]),
                }
                for i in range(N)
            ]
            return preds, routing
        return preds


# ===========================================================================
# Task runner (Class-IL benchmark interface compatible with Phase I)
# ===========================================================================
def run_ofra(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    tasks: list[list[int]],
    in_dim: int,
    n_classes: int,
    d_model: int = 128,
    n_layers: int = 4,
    chunk_size: int = 8,
    lora_rank: int = 8,
    pretrain_epochs: int = 5,
    pretrain_samples: int | None = None,
    epochs_per_task: int = 10,
    exemplar_capacity: int = 50,
    encoder_type: str = "transformer",
    loss_fn: str = "focal",
    supervised_pretrain_task0: bool = True,
    verbose: bool = False,
):
    """Drop-in replacement for Phase I runners.

    The feature statistics are fitted once on the full training split before
    Task 0. This is transductive preprocessing: labels from later tasks are not
    used, but their unlabeled feature distribution contributes to scaling.

    encoder_type:
        "transformer" — FlowTransformer + self-supervised MLM pretraining (v0.3)
        "mlp"         — Phase I MLP + supervised Task-0 pretraining (v0.4, Option B)
    """
    if encoder_type not in {"transformer", "mlp"}:
        raise ValueError(
            f"Unsupported encoder_type={encoder_type!r}; expected "
            "'transformer' or 'mlp'"
        )
    if loss_fn not in {"focal", "ce"}:
        raise ValueError(
            f"Unsupported loss_fn={loss_fn!r}; expected 'focal' or 'ce'"
        )
    if in_dim <= 0:
        raise ValueError("in_dim must be positive")
    if n_classes <= 1:
        raise ValueError("n_classes must be greater than one")
    if pretrain_epochs < 0 or epochs_per_task < 0:
        raise ValueError("epoch counts must be non-negative")
    if len(tasks) == 0:
        raise ValueError("tasks must contain at least one non-empty task")

    X_tr = np.asarray(X_tr)
    X_te = np.asarray(X_te)
    y_tr = np.asarray(y_tr)
    y_te = np.asarray(y_te)
    for split_name, X, y in (
        ("training", X_tr, y_tr),
        ("test", X_te, y_te),
    ):
        if X.ndim != 2 or X.shape[1] != in_dim:
            raise ValueError(
                f"{split_name} features must have shape (n, {in_dim})"
            )
        if len(X) == 0 or not np.isfinite(X).all():
            raise ValueError(
                f"{split_name} features must be non-empty and finite"
            )
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError(
                f"{split_name} labels must be 1-D and aligned with features"
            )
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError(
                f"{split_name} labels must contain integer class identifiers"
            )

    seen_classes: set[int] = set()
    for task_index, task in enumerate(tasks):
        if len(task) == 0:
            raise ValueError(f"tasks[{task_index}] must not be empty")
        if any(not isinstance(c, (int, np.integer)) for c in task):
            raise ValueError(f"tasks[{task_index}] must contain integer classes")
        task_classes = set(task)
        if len(task_classes) != len(task):
            raise ValueError(f"tasks[{task_index}] contains duplicate classes")
        overlap = seen_classes.intersection(task_classes)
        if overlap:
            raise ValueError(
                "Disjoint Class-IL tasks cannot repeat classes across tasks: "
                f"{sorted(overlap)}"
            )
        seen_classes.update(task_classes)

    agent = OFRAAgent(
        n_features=in_dim,
        d_model=d_model,
        n_layers=n_layers,
        chunk_size=chunk_size,
        lora_rank=lora_rank,
        exemplar_capacity=exemplar_capacity,
        encoder_type=encoder_type,
    )
    # Deliberately fit on the full training split; see transductive note above.
    agent.fit_input_stats(X_tr)
    if verbose:
        print(f"  Fitted z-score stats on {len(X_tr)} samples")
        print(f"  Encoder type: {encoder_type}")

    # ----- pretraining
    if encoder_type == "transformer" and pretrain_epochs > 0:
        if verbose:
            print(f"  Pretraining (MLM, {pretrain_epochs} epochs"
                  f"{' on '+str(pretrain_samples)+' samples' if pretrain_samples else ' on full data'})...")
        agent.pretrain_encoder(
            X_tr,
            epochs=pretrain_epochs,
            max_samples=pretrain_samples,
            verbose=verbose,
        )
    elif encoder_type == "mlp" and supervised_pretrain_task0 and len(tasks) > 0:
        # Supervised pretrain on Task 0 data
        task0 = tasks[0]
        X_t0, y_t0 = subset_by_classes(X_tr, y_tr, task0)
        if verbose:
            print(f"  Supervised pretrain on Task 0 ({task0}, "
                  f"n={len(X_t0)}, {pretrain_epochs} epochs)...")
        agent.supervised_pretrain_encoder(
            X_t0, y_t0,
            n_classes=n_classes,
            epochs=pretrain_epochs,
            verbose=verbose,
        )

    agent.freeze_encoder()

    if verbose:
        n_enc = sum(p.numel() for p in agent.encoder.parameters())
        print(f"  Encoder params (frozen): {n_enc:,}")

    acc_matrix = np.full((len(tasks), len(tasks)), np.nan)
    for i, task_i in enumerate(tasks):
        Xi, yi = subset_by_classes(X_tr, y_tr, task_i)
        if verbose:
            print(f"  Task {i}: classes {task_i}, n={len(Xi)}")
        agent.train_task(
            Xi,
            yi,
            epochs=epochs_per_task,
            loss_fn=loss_fn,
            verbose=verbose,
        )

        for j, task_j in enumerate(tasks[: i + 1]):
            Xj, yj = subset_by_classes(X_te, y_te, task_j)
            if len(Xj) == 0:
                continue
            preds = agent.predict(Xj)
            acc_matrix[i, j] = float((preds == yj).mean())

    return acc_matrix, agent
