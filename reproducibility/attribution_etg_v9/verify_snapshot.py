"""Verify this offline source snapshot, optionally run synthetic unit tests."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PACKAGES = ("formal_v2_explanation_etg", "streaming_full", "ofra_encoders")
TEST_MODULES = (
    "test_joint_margin_exact_forward",
    "test_preflight_score_fidelity",
    "test_attribution_robustness",
    "test_aggregate_attribution_robustness",
    "test_completed_output_resume",
    "test_verify_multiseed_bindings",
)


def verify() -> set[str]:
    document = json.loads((ROOT / "SOURCE_PROVENANCE.json").read_text(encoding="utf-8"))
    if document.get("schema_version") != "ofra_offline_source_snapshot_v1":
        raise RuntimeError("Unsupported source inventory")
    registered: set[str] = set()
    for row in document["files"]:
        name = row["file"]
        posix = PurePosixPath(name)
        if posix.is_absolute() or ".." in posix.parts or "\\" in name or ":" in name:
            raise RuntimeError("Invalid source inventory path")
        if name in registered:
            raise RuntimeError(f"Duplicate source: {name}")
        registered.add(name)
        path = ROOT / name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise RuntimeError(f"Source escapes snapshot: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError(f"Source hash mismatch: {name}")
    actual = set()
    for path in ROOT.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("Symlinks are not allowed in the snapshot")
        if path.is_file() and path != ROOT / "SOURCE_PROVENANCE.json":
            actual.add(path.relative_to(ROOT).as_posix())
    if actual != registered:
        raise RuntimeError(f"Inventory mismatch: extra={sorted(actual-registered)}, missing={sorted(registered-actual)}")
    print(f"SOURCE_INVENTORY=PASS ({len(registered)} files)")
    return registered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    registered = verify()
    if not args.self_test:
        return 0
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "2"
    sys.path.insert(0, str(ROOT))
    import numpy
    import torch
    torch.set_num_threads(2)
    print(f"TEST_ENVIRONMENT=Python {sys.version.split()[0]}, torch {torch.__version__}, numpy {numpy.__version__}; CPU only")
    names = ["formal_v2_explanation_etg." + name for name in TEST_MODULES]
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful() or result.testsRun != 46 or result.skipped:
        return 1
    imported = set()
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] not in PACKAGES or not getattr(module, "__file__", None):
            continue
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(ROOT):
            raise RuntimeError(f"Dependency escaped isolated snapshot: {name}")
        relative = path.relative_to(ROOT).as_posix()
        if relative not in registered:
            raise RuntimeError(f"Unregistered imported source: {relative}")
        imported.add(relative)
    print(f"LOCAL_IMPORT_CLOSURE=PASS ({len(imported)} loaded source files)")
    verify()
    print("SYNTHETIC_UNIT_TESTS=PASS; full data reproduction and numerical equivalence NOT established")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
