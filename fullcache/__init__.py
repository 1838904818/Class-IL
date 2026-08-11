"""Streaming, uncapped feature-cache builder for the OFRA validation run."""

from .core import BuildOptions, build_dataset_cache, write_root_manifest
from .specs import DATASET_SPECS, DatasetSpec

__all__ = [
    "BuildOptions",
    "DATASET_SPECS",
    "DatasetSpec",
    "build_dataset_cache",
    "write_root_manifest",
]

__version__ = "1.0.0"
