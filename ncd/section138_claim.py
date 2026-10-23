"""Atomic cell ownership on one shared filesystem, without lease stealing.

Both lanes must access the same backing directory. Separate copies of a claim
directory do not coordinate machines. Interrupted claims require reconciliation.
"""
import json
import os
from pathlib import Path
import re
import socket
import time
import uuid


def claim(root, cell, *, lane, source_commit):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", cell):
        raise ValueError("Unsafe cell identifier")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("Full source commit required")
    root = Path(root).resolve(strict=True)
    target = root / cell
    try:
        target.mkdir()
    except FileExistsError:
        return None
    owner = dict(token=uuid.uuid4().hex, lane=lane, source_commit=source_commit,
                 host=socket.gethostname(), pid=os.getpid(), started_unix=time.time())
    # A crash before this write leaves an occupied directory, never a free cell.
    with (target / "owner.json").open("x") as handle:
        json.dump(owner, handle)
        handle.flush()
        os.fsync(handle.fileno())
    return owner


def finish(root, cell, owner, *, result_sha256, exit_code):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", cell):
        raise ValueError("Unsafe cell identifier")
    if exit_code != 0 or not re.fullmatch(r"[0-9a-f]{64}", result_sha256):
        raise ValueError("Only a successful, hashed result can finish a claim")
    target = Path(root).resolve(strict=True) / cell
    actual = json.loads((target / "owner.json").read_text())
    if actual != owner:
        raise ValueError("Claim owner mismatch")
    # Exclusive creation prevents a second publication from replacing the first.
    with (target / "done.json").open("x") as handle:
        json.dump(dict(owner_token=owner["token"], result_sha256=result_sha256,
                       finished_unix=time.time(), exit_code=0), handle)
        handle.flush()
        os.fsync(handle.fileno())
