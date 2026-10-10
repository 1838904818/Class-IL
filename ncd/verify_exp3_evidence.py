"""Verify repository-safe Exp 3 evidence declared by a result summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any


FORBIDDEN_CONTENT = (
    re.compile(rb"/scr/user/", re.IGNORECASE),
    re.compile(rb"[A-Za-z]:[\\/]Users[\\/]", re.IGNORECASE),
    re.compile(rb"BEGIN [A-Z ]*PRIVATE KEY", re.IGNORECASE),
    re.compile(rb"(?:api|access)[_-]?token\s*[:=]", re.IGNORECASE),
    re.compile(rb"(?:api[_-]?key|password|passwd)\s*[:=]", re.IGNORECASE),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_repository_path(repo_root: Path, declared_path: str) -> Path:
    pure = PurePosixPath(declared_path)
    if pure.is_absolute() or not declared_path or "\\" in declared_path:
        raise ValueError(f"artifact path must be a non-empty POSIX repository path: {declared_path!r}")
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"artifact path contains an unsafe component: {declared_path!r}")

    root = repo_root.resolve()
    resolved = root.joinpath(*pure.parts).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"artifact escapes repository root: {declared_path!r}") from exc
    return resolved


def verify(summary_path: Path, repo_root: Path) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    protected = payload.get("protected_repository_artifacts")
    if not isinstance(protected, dict):
        raise ValueError("summary has no protected_repository_artifacts object")
    records = protected.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("protected_repository_artifacts.files must be a non-empty list")

    verified: list[dict[str, str]] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("artifact record must be an object")
        declared_path = record.get("path")
        expected = record.get("sha256")
        if not isinstance(declared_path, str) or declared_path in seen:
            raise ValueError(f"artifact path is missing or duplicated: {declared_path!r}")
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError(f"artifact SHA-256 is invalid: {declared_path!r}")

        artifact = resolve_repository_path(repo_root, declared_path)
        if not artifact.is_file() or artifact.is_symlink():
            raise ValueError(f"artifact is missing, not a regular file, or a symlink: {declared_path}")
        actual = sha256_file(artifact)
        if actual != expected:
            raise ValueError(
                f"artifact SHA-256 mismatch: {declared_path}; expected={expected}; actual={actual}"
            )
        content = artifact.read_bytes()
        if any(pattern.search(content) for pattern in FORBIDDEN_CONTENT):
            raise ValueError(f"artifact contains a forbidden host path or credential pattern: {declared_path}")

        seen.add(declared_path)
        verified.append({"path": declared_path, "sha256": actual})

    return {
        "status": "VERIFIED_REPOSITORY_SAFE_EXP3_EVIDENCE",
        "summary": summary_path.resolve().as_posix(),
        "verified_files": len(verified),
        "artifacts": verified,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = verify(args.summary, args.repo_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
