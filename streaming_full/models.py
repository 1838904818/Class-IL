from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    def __init__(self, n_features: int, d_model: int = 128, n_layers: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for _ in range(n_layers):
            layers.extend([nn.Linear(width, d_model), nn.ReLU()])
            width = d_model
        self.feat = nn.Sequential(*layers)
        self.d_model = int(d_model)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.feat(values)


class FamilyHead(nn.Module):
    """Output-space LoRA rank-r adapter followed by a two-logit head."""

    def __init__(self, d_model: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.A = nn.Parameter(torch.empty(rank, d_model))
        self.B = nn.Parameter(torch.zeros(d_model, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        delta = self.scaling * (embedding @ self.A.T @ self.B.T)
        return self.classifier(embedding + delta)

    def positive_probability(self, embedding: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(embedding), dim=1)[:, 1]


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha: float = 0.75,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, reduction="none")
    probability = torch.softmax(logits, dim=1)
    pt = probability.gather(1, target[:, None]).squeeze(1)
    alpha_t = torch.where(
        target == 1,
        torch.full_like(pt, alpha),
        torch.full_like(pt, 1.0 - alpha),
    )
    return (alpha_t * (1.0 - pt).pow(gamma) * ce).mean()
