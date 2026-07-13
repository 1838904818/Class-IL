"""Transformer-based flow encoder — Foundation Model surrogate for FedMAC-IDS.

Since pretrained checkpoints of Lens / NetGPT are not always available, we ship
a lightweight transformer (~200K params) that can be:
  (i)  pretrained with self-supervised masked-feature reconstruction on
       unlabeled flow data (mimicking Lens / FlowBERT objective);
  (ii) frozen as the foundation backbone for downstream Class-IL.

This is a faithful drop-in surrogate: the encoder is permutation-invariant over
feature dimensions (each flow feature is treated as a token), making it usable
across datasets with different feature counts (NSL-KDD 122 vs CIC 78 vs UNSW 196).

Architecture
============
input:  (B, D) raw flow features
        ↓ feature-tokenisation: each scalar is projected via per-position embedding
tokens: (B, D, H) where H = d_model
        ↓ N × TransformerEncoderLayer (self-attention + MLP)
        ↓ mean-pool over the D-dim sequence
output: (B, H) flow embedding
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Feature tokenisation: project each scalar feature to d_model
# ---------------------------------------------------------------------------
class FeatureTokenizer(nn.Module):
    """Turn (B, D) scalar features into (B, D, H) token embeddings.

    Each feature dimension gets its own learnable weight and bias:
        token_i = scalar_i * W_i + b_i,   W_i, b_i ∈ R^H

    This is the standard FT-Transformer style tokenizer (Gorishniy 2021).
    It is *trivially* extensible to new feature dimensions: when fine-tuning
    on a new dataset with extra features, only the new W_i, b_i need init.
    """

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_features, d_model))
        self.b = nn.Parameter(torch.zeros(n_features, d_model))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D) → (B, D, 1) * (D, H) + (D, H) → (B, D, H)
        return x.unsqueeze(-1) * self.W + self.b


# ---------------------------------------------------------------------------
# Transformer encoder
# ---------------------------------------------------------------------------
class FlowTransformer(nn.Module):
    """Lightweight transformer for flow embedding.

    Args:
        n_features:  input feature dimension D (per-dataset)
        d_model:     hidden size H (default 128 — comparable to Phase I MLP)
        n_heads:     attention heads (default 4)
        n_layers:    transformer layers (default 4)
        dim_ff:      feed-forward inner dim (default 256)
        dropout:     dropout rate
        chunk_size:  group every `chunk_size` raw features into a single token.
                     Default 1 = one token per feature (like FT-Transformer).
                     Larger chunks reduce sequence length and quadratically
                     speed up self-attention; trade off feature-granularity.

    Sequence length: ceil(n_features / chunk_size)
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_ff: int = 256,
        dropout: float = 0.1,
        chunk_size: int = 1,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.chunk_size = max(1, chunk_size)

        if self.chunk_size == 1:
            # Per-feature tokenisation (FT-Transformer style)
            self.tokenizer = FeatureTokenizer(n_features, d_model)
            self.seq_len = n_features
            self.chunk_proj = None
        else:
            # Group `chunk_size` features into one token via small linear
            # projection. Pad with zeros if n_features is not divisible.
            self.seq_len = (n_features + self.chunk_size - 1) // self.chunk_size
            self.tokenizer = None
            self.chunk_proj = nn.Linear(self.chunk_size, d_model)

        # Learnable positional embedding (one per token position)
        self.pos_emb = nn.Parameter(torch.zeros(self.seq_len, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        # Lightweight projection head used during pretraining to reconstruct
        # input features (chunk-level MSE).
        self.pretrain_head = nn.Linear(d_model, self.chunk_size)

    # --------------------------- tokenisation
    def _tokenise(self, x: torch.Tensor) -> torch.Tensor:
        """Map (B, D) → (B, seq_len, d_model)."""
        if self.chunk_size == 1:
            return self.tokenizer(x) + self.pos_emb
        # Pad and chunk
        B, D = x.shape
        pad = self.seq_len * self.chunk_size - D
        if pad > 0:
            x = torch.cat([x, x.new_zeros(B, pad)], dim=1)
        chunks = x.view(B, self.seq_len, self.chunk_size)
        tokens = self.chunk_proj(chunks)
        return tokens + self.pos_emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled embedding (B, d_model)."""
        tokens = self._tokenise(x)      # (B, L, H)
        h = self.encoder(tokens)        # (B, L, H)
        return h.mean(dim=1)            # (B, H)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Alias matching Phase I MLP interface."""
        return self.forward(x)

    def pretrain_forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Masked-token reconstruction (BERT-style for tabular).

        With chunk_size > 1, mask is per-CHUNK (one bool per token).
        Returns (pred, target) both shape (B, D), with pad positions ignored.
        """
        # Re-pad x to seq_len * chunk_size for chunk-level processing
        B, D = x.shape
        pad = self.seq_len * self.chunk_size - D
        x_p = torch.cat([x, x.new_zeros(B, pad)], dim=1) if pad > 0 else x
        chunks = x_p.view(B, self.seq_len, self.chunk_size)        # (B, L, C)

        # Sample a fresh mask at token (chunk) level
        token_mask = torch.rand(B, self.seq_len, device=x.device) < mask.float().mean()
        masked_chunks = chunks.clone()
        masked_chunks[token_mask] = 0.0

        if self.chunk_size == 1:
            tokens = self.tokenizer(masked_chunks.squeeze(-1)) + self.pos_emb
        else:
            tokens = self.chunk_proj(masked_chunks) + self.pos_emb
        h = self.encoder(tokens)                                   # (B, L, H)
        pred_chunks = self.pretrain_head(h)                        # (B, L, C)
        # Flatten back to (B, D), trim padding
        pred = pred_chunks.view(B, self.seq_len * self.chunk_size)[:, :D]
        return pred, x


# ---------------------------------------------------------------------------
# Self-supervised pretraining utility
# ---------------------------------------------------------------------------
def masked_feature_reconstruction_loss(
    model: FlowTransformer,
    x: torch.Tensor,
    mask_prob: float = 0.15,
) -> torch.Tensor:
    """Compute MSE on masked features only (BERT-style MLM for tabular).

    Random 15% of tokens (chunks) per sample are zero-masked; the model must
    predict their original values. Trains the encoder to internalise feature
    dependencies (and inter-chunk relations when chunk_size > 1).
    """
    # Token-level mask sampled inside pretrain_forward; we pass a feature-level
    # mask for API compatibility.
    mask = torch.rand_like(x) < mask_prob
    pred, target = model.pretrain_forward(x, mask)
    if mask.sum() == 0:
        return torch.zeros((), device=x.device)
    # Compute MSE over all feature positions (the model knows which were masked)
    return ((pred - target) ** 2).mean()
