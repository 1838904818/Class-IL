"""Fail-closed adapter for lucidrains' FT-Transformer implementation.

OFRA's cache builders already convert every input column to a finite float32
feature and the runner standardises those features using frozen Task-0
statistics.  This adapter therefore declares no categorical columns and sends
all columns through the upstream numerical tokenizer.
"""
from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


FT_TRANSFORMER_UPSTREAM: dict[str, object] = {
    "distribution": "tab-transformer-pytorch",
    "version": "0.6.1",
    "repository": "https://github.com/lucidrains/tab-transformer-pytorch",
    "git_commit": "10c258aa7ecf8c7e948e38c104a87caed49a6a9a",
    "git_tag": "0.6.1",
    "license": "MIT",
    "implementation_file": "tab_transformer_pytorch/ft_transformer.py",
    "implementation_sha256": (
        "db62c6e258467bb2d85b738fe1839f0b4279ec92f0bdbb83400ddd42fadd4d42"
    ),
    "pypi_wheel_sha256": (
        "4f350e3e4c8f17869eb4825d5b0db8006078ce3ba645d80e768f6f9281b5d263"
    ),
}


def verify_ft_transformer_dependency() -> dict[str, object]:
    """Return the installed source record, failing on version/source drift."""
    expected_version = str(FT_TRANSFORMER_UPSTREAM["version"])
    actual_version = metadata.version(
        str(FT_TRANSFORMER_UPSTREAM["distribution"])
    )
    if actual_version != expected_version:
        raise RuntimeError(
            "FT-Transformer dependency version mismatch: "
            f"expected {expected_version}, found {actual_version}"
        )

    from tab_transformer_pytorch import ft_transformer as upstream_module

    source_path = Path(upstream_module.__file__).resolve()
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    expected_sha256 = str(FT_TRANSFORMER_UPSTREAM["implementation_sha256"])
    if source_sha256 != expected_sha256:
        raise RuntimeError(
            "FT-Transformer implementation hash mismatch: "
            f"expected {expected_sha256}, found {source_sha256}"
        )
    return {
        **FT_TRANSFORMER_UPSTREAM,
        "installed_version": actual_version,
        "installed_implementation_sha256": source_sha256,
    }


class FTTransformerEncoder(nn.Module):
    """Map standardised continuous tabular features to one OFRA embedding.

    The upstream ``FTTransformer`` is used without vendoring or changing its
    attention blocks.  ``categories=()`` is intentional: cache preprocessing
    has already represented every field numerically.  ``dim_out=d_model``
    makes the upstream CLS readout the embedding consumed by the unchanged
    OFRA family heads, router, and exemplar selector.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        depth: int = 4,
        heads: int = 4,
        dim_head: int = 16,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        num_residual_streams: int = 1,
    ) -> None:
        super().__init__()
        integer_positive = {
            "n_features": n_features,
            "d_model": d_model,
            "depth": depth,
            "heads": heads,
            "dim_head": dim_head,
            "num_residual_streams": num_residual_streams,
        }
        for name, value in integer_positive.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("attn_dropout", attn_dropout),
            ("ff_dropout", ff_dropout),
        ):
            if not 0.0 <= float(value) < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")

        verify_ft_transformer_dependency()
        from tab_transformer_pytorch import FTTransformer

        self.n_features = int(n_features)
        self.d_model = int(d_model)
        self.categories: tuple[int, ...] = ()
        self.depth = int(depth)
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.attn_dropout = float(attn_dropout)
        self.ff_dropout = float(ff_dropout)
        self.num_residual_streams = int(num_residual_streams)
        self.backbone = FTTransformer(
            categories=self.categories,
            num_continuous=self.n_features,
            dim=self.d_model,
            dim_out=self.d_model,
            depth=self.depth,
            heads=self.heads,
            dim_head=self.dim_head,
            attn_dropout=self.attn_dropout,
            ff_dropout=self.ff_dropout,
            num_residual_streams=self.num_residual_streams,
        )
        self.pretrain_head: nn.Linear | None = None

    def _embedding_only_forward(
        self, categories: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """Run the pinned backbone without stacking unused attention maps.

        Upstream 0.6.1 always calls ``transformer(..., return_attn=True)`` even
        when ``FTTransformer.forward(return_attn=False)`` was requested. The
        resulting ``torch.stack`` is not used by OFRA and scales as
        ``depth x batch x heads x tokens x tokens``. This adapter reproduces
        the upstream embedding, CLS, transformer, and readout operations while
        requesting the transformer's existing output-only path. Parameters
        and trainable computations are unchanged.
        """
        backbone = self.backbone
        categories = categories + backbone.num_special_tokens
        tokens = backbone.embedding(
            (categories, values),
            sum_discrete_sets=False,
            sum_continuous=False,
            concat_discrete_continuous=True,
        )
        cls_tokens = backbone.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = backbone.transformer(tokens, return_attn=False)
        return backbone.to_logits(tokens[:, 0])

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != self.n_features:
            raise ValueError(
                "FTTransformerEncoder input must have shape "
                f"(batch, {self.n_features})"
            )
        categories = torch.empty(
            (values.shape[0], 0), dtype=torch.long, device=values.device
        )
        embedding = self._embedding_only_forward(categories, values)
        if embedding.shape != (values.shape[0], self.d_model):
            raise RuntimeError(
                "upstream FTTransformer returned an unexpected embedding shape: "
                f"{tuple(embedding.shape)}"
            )
        return embedding

    def supervised_pretrain(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int,
        epochs: int = 5,
        batch_size: int = 256,
        lr: float = 1e-3,
        verbose: bool = False,
    ) -> None:
        """Task-0 bootstrap matching the MLP candidate's training objective."""
        device = next(self.backbone.parameters()).device
        self.pretrain_head = nn.Linear(self.d_model, n_classes).to(device)
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64)),
            ),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=(device.type == "cuda"),
        )
        optimizer = torch.optim.Adam(
            [*self.backbone.parameters(), *self.pretrain_head.parameters()],
            lr=lr,
        )
        self.train()
        for epoch in range(epochs):
            loss_sum = 0.0
            correct = 0
            rows = 0
            for values, target in loader:
                values = values.to(device, non_blocking=(device.type == "cuda"))
                target = target.to(device, non_blocking=(device.type == "cuda"))
                logits = self.pretrain_head(self(values))
                loss = F.cross_entropy(logits, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.item()) * len(target)
                correct += int((logits.argmax(dim=1) == target).sum().item())
                rows += len(target)
            if verbose:
                print(
                    f"  supervised pretrain epoch {epoch + 1}/{epochs}: "
                    f"loss={loss_sum / max(rows, 1):.4f}, "
                    f"acc={correct / max(rows, 1):.4f}"
                )
        self.pretrain_head = None
        self.eval()

    def architecture_record(self) -> dict[str, object]:
        return {
            "encoder_type": "ft_transformer",
            "input_representation": "all_standardised_continuous_features",
            "categories": [],
            "num_continuous": self.n_features,
            "dim": self.d_model,
            "dim_out": self.d_model,
            "depth": self.depth,
            "heads": self.heads,
            "dim_head": self.dim_head,
            "attn_dropout": self.attn_dropout,
            "ff_dropout": self.ff_dropout,
            "num_residual_streams": self.num_residual_streams,
            "attention_output_policy": (
                "upstream_equivalent_embedding_only; per-layer attention matrices "
                "are not stacked because OFRA does not consume them"
            ),
            "residual_design": (
                "standard_residual_connections; upstream mHC disabled"
                if self.num_residual_streams == 1
                else "upstream_mHC"
            ),
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "upstream": verify_ft_transformer_dependency(),
        }
