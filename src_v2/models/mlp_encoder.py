"""MLP encoder for FedMAC-IDS Option B (v0.4).

Drop-in replacement for FlowTransformer that uses the Phase I MLP backbone.
The motivation is to isolate whether the LoRA + DPMeans router architecture
works independent of the encoder. Phase I's MLP already proved capable of
98% supervised accuracy on NSL-KDD, so this lifts the encoder ceiling.

Pretraining:
  Unlike FlowTransformer's self-supervised masked-feature reconstruction
  (which assumes a powerful transformer), MLPEncoder uses *supervised
  pretraining on Task 0* — a realistic deployment scenario in which the
  initial model is trained on bootstrap data (Normal + most-frequent
  attack family).

Architecture:
  in_dim → Linear(in_dim, hidden) → ReLU
        → Linear(hidden, hidden) → ReLU
  output: (B, hidden) feature vector
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class MLPEncoder(nn.Module):
    """2-layer MLP feature extractor with supervised pretrain support.

    Args:
        n_features: input dimension D
        d_model:    hidden / output dimension (default 128, same as Phase I)
        n_layers:   number of Linear+ReLU layers (default 2)
        dropout:    dropout between layers (default 0)
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        layers = []
        prev = n_features
        for i in range(n_layers):
            layers.append(nn.Linear(prev, d_model))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = d_model
        self.feat = nn.Sequential(*layers)

        # Temporary classification head used during supervised pretraining only;
        # discarded once we freeze the encoder.
        self.pretrain_head: nn.Linear | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, d_model) features."""
        return self.feat(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Alias matching FlowTransformer interface."""
        return self.forward(x)

    # ------------------------------------------------------------------ supervised pretrain
    def supervised_pretrain(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int,
        epochs: int = 5,
        batch_size: int = 256,
        lr: float = 1e-3,
        verbose: bool = False,
    ):
        """Train the encoder + a temporary classifier on supervised labels.

        Useful before Class-IL begins: bootstrap the encoder so its features
        are informative for the first few classes encountered (e.g., Task 0
        of NSL-KDD = Normal + DoS).
        """
        _dev = next(self.feat.parameters()).device
        if self.pretrain_head is None or self.pretrain_head.out_features != n_classes:
            self.pretrain_head = nn.Linear(self.d_model, n_classes)
        self.pretrain_head = self.pretrain_head.to(_dev)

        Xt = torch.from_numpy(X.astype(np.float32)).to(_dev)
        yt = torch.from_numpy(y.astype(np.int64)).to(_dev)
        loader = DataLoader(
            TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True
        )

        # Train both feature MLP and head
        params = list(self.feat.parameters()) + list(self.pretrain_head.parameters())
        opt = torch.optim.Adam(params, lr=lr)

        self.train()
        for ep in range(epochs):
            losses, correct, total = [], 0, 0
            for bx, by in loader:
                feats = self.feat(bx)
                logits = self.pretrain_head(feats)
                loss = F.cross_entropy(logits, by)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
                correct += (logits.argmax(dim=1) == by).sum().item()
                total += len(by)
            if verbose:
                acc = correct / max(total, 1)
                print(f"  supervised pretrain epoch {ep+1}/{epochs}: "
                      f"loss={np.mean(losses):.4f}, acc={acc:.4f}")

        # Discard pretrain head — only the encoder's `feat` is kept
        self.pretrain_head = None
        self.eval()
