"""Optional, protocol-bound encoder adapters for OFRA."""

from .ft_transformer import (
    FT_TRANSFORMER_UPSTREAM,
    FTTransformerEncoder,
    verify_ft_transformer_dependency,
)

__all__ = [
    "FT_TRANSFORMER_UPSTREAM",
    "FTTransformerEncoder",
    "verify_ft_transformer_dependency",
]
