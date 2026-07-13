"""LoRA (Low-Rank Adaptation) for parameter-efficient Class-IL.

Each attack family gets its own LoRA delta on top of a frozen backbone.

Mathematics
===========
Given a frozen linear layer  W ∈ R^{d_out × d_in}, LoRA learns a low-rank update
        ΔW = B A,   A ∈ R^{r × d_in}, B ∈ R^{d_out × r},   r ≪ min(d_in, d_out)
The forward pass becomes
        y = (W + α/r · B A) x
where α is a scaling factor.

Parameter cost: O(r(d_in + d_out)) vs O(d_in d_out) for full fine-tuning.
At r=8, d_in=d_out=128: 2048 params per LoRA vs 16384 for full layer.

For Class-IL:
  - Frozen backbone encoder (FlowTransformer) → shared knowledge
  - One LoRA per attack family → task-specific knowledge, no interference
  - At inference, combine independent family-head confidence with router score
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Single LoRA adapter
# ---------------------------------------------------------------------------
class LoRAAdapter(nn.Module):
    """Low-rank delta applied to a linear projection.

    Standalone (does not wrap an nn.Linear) — designed to be applied additively
    over a frozen encoder's pooled output:
        embed_adapted = embed_frozen + LoRA(embed_frozen)

    This is a generalisation: instead of patching specific layers inside the
    encoder, we adapt the *output embedding space* directly. Simpler and
    sufficient for Class-IL routing.
    """

    def __init__(self, d_model: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # B is zero-initialised so ΔW = 0 at start (standard LoRA practice)
        self.A = nn.Parameter(torch.empty(rank, d_model))
        self.B = nn.Parameter(torch.zeros(d_model, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d_model) → (B, d_model)  (additive delta)"""
        return self.scaling * (x @ self.A.T @ self.B.T)

    def num_params(self) -> int:
        return self.A.numel() + self.B.numel()


# ---------------------------------------------------------------------------
# Family-specific classifier head
# ---------------------------------------------------------------------------
class FamilyHead(nn.Module):
    """Per-family LoRA and classifier head.

    OFRA uses two logits per family and converts them to that head's own
    positive-class probability. The resulting confidences are independent;
    they are not normalized into a distribution across families. A one-logit
    form remains available for compatibility with earlier experiments.

    Architecture:
      embed → embed + LoRA(embed) → Linear(d_model, n_local_classes)
    """

    def __init__(
        self,
        d_model: int,
        n_local_classes: int = 1,  # one-logit compatibility default
        rank: int = 8,
        alpha: float = 16.0,
    ):
        super().__init__()
        self.lora = LoRAAdapter(d_model, rank=rank, alpha=alpha)
        # OFRA passes n_local_classes=2; one-logit heads remain supported for
        # compatibility with earlier stored experiments.
        self.n_local_classes = max(1, n_local_classes) if n_local_classes != 2 else 2
        self.classifier = nn.Linear(d_model, self.n_local_classes)

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        """Return (B, n_local_classes) logits. OFRA uses two-logit binary
        heads; the one-logit form is retained for compatibility.
        """
        adapted = embed + self.lora(embed)
        return self.classifier(adapted)

    def scalar(self, embed: torch.Tensor) -> torch.Tensor:
        """Return (B,) scalar confidence for this family.

        Works for both n_local_classes=1 (raw scalar) and n_local_classes=2
        (positive-class logit minus negative-class logit).
        """
        logits = self.forward(embed)
        if self.n_local_classes == 1:
            return logits.squeeze(-1)
        return logits[:, 1] - logits[:, 0]

    def num_params(self) -> int:
        return (
            self.lora.num_params()
            + sum(p.numel() for p in self.classifier.parameters())
        )


# ---------------------------------------------------------------------------
# LoRA pool: one head per attack family, dynamically grown
# ---------------------------------------------------------------------------
class LoRAPool(nn.Module):
    """Dictionary of family-specific LoRA heads.

    New attack family → call `add_family(name)`. Existing families are FROZEN
    when training a new one (the parameter-isolation principle: no forgetting
    because old parameters never change).

    Routing:
      During training of family f, only LoRA[f] is updated.
      During inference, OFRA combines every binary head's independent positive
      confidence with the corresponding DPMeans router score.
    """

    def __init__(self, d_model: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.alpha = alpha
        # `heads` is an nn.ModuleDict so PyTorch tracks gradients per family
        self.heads = nn.ModuleDict()
        # `families` preserves insertion order for global class index mapping
        self.families: list[str] = []

    # ------------------------------------------------------------------ add
    def add_family(self, name: str, n_local_classes: int = 2):
        if name in self.heads:
            raise ValueError(f"family {name} already in pool")
        head = FamilyHead(
            d_model=self.d_model,
            n_local_classes=n_local_classes,
            rank=self.rank,
            alpha=self.alpha,
        )
        self.heads[name] = head
        self.families.append(name)
        return head

    def has(self, name: str) -> bool:
        return name in self.heads

    # ------------------------------------------------------------------ freeze
    def freeze_all_except(self, name: str | None):
        """Freeze all heads except the named one (or all if name is None)."""
        for fname, head in self.heads.items():
            requires = (fname == name)
            for p in head.parameters():
                p.requires_grad = requires

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------ forward
    def forward_single(self, embed: torch.Tensor, family: str) -> torch.Tensor:
        """Logits from one specific family head: (B, n_local_classes)."""
        return self.heads[family](embed)

    def forward_all(self, embed: torch.Tensor) -> dict[str, torch.Tensor]:
        """Logits from every family head: {family_name → (B, n_local_classes)}."""
        return {f: self.heads[f](embed) for f in self.families}

    def num_params_per_family(self) -> int:
        if not self.families:
            return 0
        return next(iter(self.heads.values())).num_params()

    def num_params_total(self) -> int:
        return sum(h.num_params() for h in self.heads.values())

    # ------------------------------------------------------------------ state dict for federation
    def lora_state_dict(self) -> dict:
        """Extract only the LoRA params (A, B) for federated transmission.

        Excludes classifier weights (kept private per site to preserve
        site-specific label distribution) and bias terms.
        """
        out = {}
        for fname, head in self.heads.items():
            out[fname] = {
                "A": head.lora.A.detach().clone(),
                "B": head.lora.B.detach().clone(),
            }
        return out

    def load_lora_state(self, state: dict):
        """Load shared LoRA params from federated aggregation."""
        with torch.no_grad():
            for fname, params in state.items():
                if fname in self.heads:
                    self.heads[fname].lora.A.copy_(params["A"])
                    self.heads[fname].lora.B.copy_(params["B"])
