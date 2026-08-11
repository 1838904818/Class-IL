"""Deterministic provenance helpers for OFRA benchmark runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import torch


DATASET_INPUTS = {
    "NSL-KDD": ("nsl-kdd", "*.txt"),
    "UNSW-NB15": ("unsw-nb15", "*.csv"),
    "CIC-IDS-2017": ("cic-ids-2017", "*.csv"),
    "CIC-IDS-2018": ("cic-ids-2018", "*.csv"),
    "CIC-IoT-2023": ("cic-iot-2023", "*.csv"),
    "NF-ToN-IoT-v2": ("nf-ton-iot-v2", "*.csv"),
}


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(root: Path, files: list[Path]) -> dict:
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
        for path in files
    ]
    return {
        "files": entries,
        "file_count": len(entries),
        "total_bytes": int(sum(entry["bytes"] for entry in entries)),
        "manifest_sha256": canonical_sha256(entries),
    }


def code_manifest(code_root: Path) -> dict:
    files: list[Path] = []
    for folder in ("src", "src_v2"):
        files.extend(
            path
            for path in (code_root / folder).rglob("*.py")
            if "__pycache__" not in path.parts
        )
    files.sort(key=lambda path: path.relative_to(code_root).as_posix())
    return _manifest(code_root, files)


def dataset_manifest(data_root: Path, dataset_name: str) -> dict:
    folder, pattern = DATASET_INPUTS[dataset_name]
    dataset_root = data_root / folder
    files = sorted(dataset_root.glob(pattern), key=lambda path: path.name)
    if not files:
        raise FileNotFoundError(
            f"No inputs matching {pattern!r} found for {dataset_name} in "
            f"{dataset_root}"
        )
    manifest = _manifest(dataset_root, files)
    manifest.update({"dataset": dataset_name, "folder": folder, "pattern": pattern})
    return manifest


def array_manifest(array: np.ndarray, rows_per_chunk: int = 65_536) -> dict:
    value = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(canonical_sha256(list(value.shape)).encode("ascii"))
    if value.ndim == 0:
        digest.update(np.ascontiguousarray(value).tobytes())
    elif value.flags.c_contiguous:
        digest.update(memoryview(value).cast("B"))
    else:
        for start in range(0, len(value), rows_per_chunk):
            digest.update(
                np.ascontiguousarray(value[start:start + rows_per_chunk]).tobytes()
            )
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "sha256": digest.hexdigest(),
    }


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def environment_manifest() -> dict:
    cuda_available = torch.cuda.is_available()
    nvidia = {"driver_version": None, "gpu_uuid": None}
    if cuda_available:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version,uuid",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            driver_version, gpu_uuid = completed.stdout.splitlines()[0].split(",", 1)
            nvidia = {
                "driver_version": driver_version.strip(),
                "gpu_uuid": gpu_uuid.strip(),
            }
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scikit-learn": _package_version("scikit-learn"),
            "torch": _package_version("torch"),
        },
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        **nvidia,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "environment_controls": {
            key: os.environ.get(key)
            for key in (
                "PYTHONHASHSEED",
                "CUBLAS_WORKSPACE_CONFIG",
                "CUDA_DEVICE_ORDER",
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "FULL_DATA",
                "CIC17_NORMAL_CAP",
                "MAX_PER_CLASS",
            )
        },
    }
