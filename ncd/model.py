"""MLP and real decision-score adapters used by NCD."""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(inputs))


class DecisionScore(nn.Module):
    """Expose the actual prediction score used by Replay or iCaRL."""

    def __init__(
        self,
        model: MLP,
        inference: str,
        n_classes: int,
        prototypes: dict[int, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.model = model
        self.inference = inference
        self.n_classes = n_classes
        prototypes = prototypes or {}
        seen = sorted(prototypes)
        self.register_buffer("seen_ids", torch.tensor(seen, dtype=torch.long))
        if seen:
            self.register_buffer("prototype_matrix", torch.stack([prototypes[c] for c in seen]))
        else:
            self.register_buffer("prototype_matrix", torch.empty((0, model.head.in_features)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.inference == "logit":
            return self.model(inputs)
        if self.inference != "nme" or self.prototype_matrix.numel() == 0:
            raise RuntimeError("NME score requested without prototypes")
        features = torch.nn.functional.normalize(self.model.features(inputs), dim=1)
        prototypes = torch.nn.functional.normalize(self.prototype_matrix, dim=1)
        seen_scores = -torch.sum((features[:, None, :] - prototypes[None, :, :]) ** 2, dim=2)
        scores = torch.full(
            (inputs.shape[0], self.n_classes),
            -1.0e6,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        return scores.index_copy(1, self.seen_ids, seen_scores)
