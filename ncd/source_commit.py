"""Resolve the source commit for checkout and packaged HPC execution."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def source_commit(repo_root: Path) -> str:
    """Return the exact source commit, including from a git-free archive."""
    packaged = os.environ.get("NCD_SOURCE_COMMIT", "").strip().lower()
    if packaged:
        if COMMIT_PATTERN.fullmatch(packaged) is None:
            raise ValueError("NCD_SOURCE_COMMIT must be a complete 40-character SHA-1")
        return packaged
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip().lower()
    if COMMIT_PATTERN.fullmatch(commit) is None:
        raise RuntimeError(f"unexpected git commit value: {commit!r}")
    return commit
