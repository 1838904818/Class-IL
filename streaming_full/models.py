from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ofra_encoders import FTTransformerEncoder


ENCODER_TYPES = frozenset({"mlp", "ft_transformer", "tabm"})


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


class TabMMeanEncoder(nn.Module):
    """TabM backbone adapted to OFRA's single-embedding encoder contract.

    TabM emits one embedding per ensemble member with shape ``(B, K, D)``.
    OFRA's existing family heads and routers consume ``(B, D)``, so this
    additive comparison arm averages member embeddings without changing any
    downstream OFRA component.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int,
        n_layers: int,
        *,
        k: int,
        dropout: float,
    ):
        super().__init__()
        try:
            import tabm
        except ImportError as error:
            raise RuntimeError(
                "encoder_type='tabm' requires the separately declared tabm package"
            ) from error
        self.model = tabm.TabM.make(
            n_num_features=n_features,
            d_out=None,
            arch_type="tabm",
            k=k,
            d_block=d_model,
            n_blocks=n_layers,
            dropout=dropout,
        )
        self.n_features = int(n_features)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.k = int(k)
        self.dropout = float(dropout)
        self.tabm_version = getattr(tabm, "__version__", "unknown")

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        member_embeddings = self.model(values)
        if member_embeddings.ndim != 3 or member_embeddings.shape[1] != self.k:
            raise RuntimeError(
                "TabM encoder must emit (batch, ensemble_member, embedding)"
            )
        return member_embeddings.mean(dim=1)

    def architecture_record(self) -> dict[str, object]:
        return {
            "encoder_type": "tabm",
            "package_version": self.tabm_version,
            "arch_type": "tabm",
            "input_representation": "all_standardised_continuous_features",
            "n_features": self.n_features,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "k": self.k,
            "dropout": self.dropout,
            "ofra_adapter": "mean_of_member_embeddings",
            "member_output_shape": "batch_by_k_by_d_model",
            "ofra_output_shape": "batch_by_d_model",
            "parameters": int(sum(p.numel() for p in self.parameters())),
        }


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


def build_encoder(
    *,
    encoder_type: str,
    n_features: int,
    d_model: int,
    n_layers: int,
    ft_heads: int,
    ft_dim_head: int,
    ft_attn_dropout: float,
    ft_ff_dropout: float,
    ft_num_residual_streams: int,
    tabm_k: int = 16,
    tabm_dropout: float = 0.1,
) -> nn.Module:
    """Construct one encoder without changing downstream OFRA modules."""
    if encoder_type == "mlp":
        return MLPEncoder(
            n_features=n_features,
            d_model=d_model,
            n_layers=n_layers,
        )
    if encoder_type == "ft_transformer":
        return FTTransformerEncoder(
            n_features=n_features,
            d_model=d_model,
            depth=n_layers,
            heads=ft_heads,
            dim_head=ft_dim_head,
            attn_dropout=ft_attn_dropout,
            ff_dropout=ft_ff_dropout,
            num_residual_streams=ft_num_residual_streams,
        )
    if encoder_type == "tabm":
        return TabMMeanEncoder(
            n_features=n_features,
            d_model=d_model,
            n_layers=n_layers,
            k=tabm_k,
            dropout=tabm_dropout,
        )
    raise ValueError(
        f"unknown encoder_type={encoder_type!r}; expected one of "
        f"{sorted(ENCODER_TYPES)}"
    )


def model_architecture_record(
    *,
    encoder_type: str,
    n_features: int,
    n_classes: int,
    d_model: int,
    n_layers: int,
    lora_rank: int,
    lora_alpha: float,
    ft_heads: int,
    ft_dim_head: int,
    ft_attn_dropout: float,
    ft_ff_dropout: float,
    ft_num_residual_streams: int,
    tabm_k: int = 16,
    tabm_dropout: float = 0.1,
) -> dict[str, object]:
    """Return architecture and parameter counts bound into the protocol."""
    encoder = build_encoder(
        encoder_type=encoder_type,
        n_features=n_features,
        d_model=d_model,
        n_layers=n_layers,
        ft_heads=ft_heads,
        ft_dim_head=ft_dim_head,
        ft_attn_dropout=ft_attn_dropout,
        ft_ff_dropout=ft_ff_dropout,
        ft_num_residual_streams=ft_num_residual_streams,
        tabm_k=tabm_k,
        tabm_dropout=tabm_dropout,
    )
    encoder_parameters = sum(parameter.numel() for parameter in encoder.parameters())
    head = FamilyHead(d_model=d_model, rank=lora_rank, alpha=lora_alpha)
    family_head_parameters = sum(parameter.numel() for parameter in head.parameters())
    if isinstance(encoder, (FTTransformerEncoder, TabMMeanEncoder)):
        encoder_record = encoder.architecture_record()
    else:
        encoder_record = {
            "encoder_type": "mlp",
            "input_representation": "all_standardised_continuous_features",
            "n_features": int(n_features),
            "d_model": int(d_model),
            "n_layers": int(n_layers),
            "activation": "ReLU",
            "parameters": int(encoder_parameters),
        }
    return {
        "encoder": encoder_record,
        "ofra_downstream_unchanged": {
            "family_head": "output-space LoRA plus two-logit classifier",
            "router": "DualRouter",
            "exemplar_selector": "deterministic reservoir plus farthest-first",
        },
        "parameter_counts": {
            "encoder": int(encoder_parameters),
            "family_head_per_class": int(family_head_parameters),
            "family_heads_at_final_checkpoint": int(
                family_head_parameters * n_classes
            ),
            "encoder_plus_final_family_heads": int(
                encoder_parameters + family_head_parameters * n_classes
            ),
            "non_parameter_state_excluded": ["router", "exemplars"],
        },
    }


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
