"""Hash-bound historical checkpoint loading. Importing this module runs no model."""
import hashlib
import importlib
import json
from pathlib import Path
import socket
import sys
from pilot_core import require


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def safe_file(root, relative):
    root = Path(root).absolute()
    rel = Path(relative)
    require(not rel.is_absolute() and not rel.drive and ".." not in rel.parts
            and all(":" not in p for p in rel.parts), "non-relative input binding")
    path = root / rel
    for p in (path, *path.parents):
        require(not p.is_symlink() and (not p.exists() or
                not getattr(p.lstat(), "st_file_attributes", 0) & 0x400), "linked input binding")
    require(path.is_file() and path.resolve().is_relative_to(root.resolve()), "missing bound file")
    return path


def verify_runtime(root, pins):
    require(pins and len({r["path"] for r in pins}) == len(pins), "runtime closure missing or duplicate")
    for reference in pins:
        require(sha(safe_file(root, reference["path"])) == reference["sha256"], "historical runtime hash mismatch")


def load_native(runtime_root, checkpoint_root, checkpoint, device="cpu"):
    """Real execution needs separate review; no authorization is inferred here."""
    require(not socket.gethostname().split(".")[0].lower().startswith("login"),
            "checkpoint loading on login nodes is forbidden")
    require(checkpoint in (0, 1) and device in ("cpu", "cuda"), "pilot checkpoint/device")
    folder = Path(__file__).resolve().parent
    lineage = json.loads((folder / "LINEAGE_METADATA.json").read_text(encoding="utf-8"))
    plan = json.loads((folder / "PREPARED_INPUTS.json").read_text(encoding="utf-8"))
    require(lineage["status"] == "METADATA_CHAIN_VERIFIED_NOT_INFERENCE_FIDELITY"
            and len(lineage["source_files"]) == 16, "lineage receipt contract")
    verify_runtime(runtime_root, lineage["source_files"])
    binding = next(r for r in plan["checkpoints"] if r["checkpoint"] == checkpoint)
    require(binding["seed"] == 1, "training seed changed")
    for name, expected in binding["sha256"].items():
        require(sha(safe_file(checkpoint_root, name)) == expected, "checkpoint binding changed")
    root = Path(runtime_root).resolve()
    # Never silently reuse a module imported from another checkout.
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] in ("streaming_full", "ofra_encoders"):
            path = getattr(module, "__file__", None)
            require(path is not None and Path(path).resolve().is_relative_to(root), "stale runtime import")
    sys.path.insert(0, str(root))
    try:
        monitoring = importlib.import_module("streaming_full.monitoring")
        require(Path(monitoring.__file__).resolve() == root / "streaming_full/monitoring.py", "wrong native module")
        # Original loader validates saved manifest, state and probe hashes and
        # reconstructs the original encoder, binary heads and cap3000 Router.
        model = monitoring.load_checkpoint(Path(checkpoint_root)/"checkpoint_manifest.json", device=device)
        require(model.metadata["seen_classes"] == binding["seen_classes"] and model.metadata["seed"] == 1,
                "loaded checkpoint identity differs")
        verify_runtime(root, lineage["source_files"])
        for name, expected in binding["sha256"].items():
            require(sha(safe_file(checkpoint_root, name)) == expected, "checkpoint changed while loading")
        return model
    finally:
        sys.path.remove(str(root))
