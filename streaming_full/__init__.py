"""Streaming, shard-backed OFRA full-validation runner.

This package is intentionally independent from ``src_v2.methods.ofra`` and
the legacy dataset loaders.  It consumes a versioned manifest of class-wise
``.npy`` shards and keeps both router variants on one shared training
trajectory.
"""

from .validation import RunConfig, run_manifest

__all__ = ["RunConfig", "run_manifest"]
