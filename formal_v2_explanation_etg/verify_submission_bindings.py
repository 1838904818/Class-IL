"""Fail-closed verification of code-independent inputs within owned roots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from streaming_full.data import load_manifest, sha256_file
from streaming_full.monitoring import validate_checkpoint_manifest


def confined_file(path: Path, roots: tuple[Path, ...], label: str) -> Path:
    candidate = path.resolve(strict=True)
    if not candidate.is_file():
        raise RuntimeError(f"{label} is not a regular file")
    if not any(candidate.is_relative_to(root) for root in roots):
        raise RuntimeError(f"{label} escapes the authorized account roots")
    return candidate


def _manifest_shard_paths(stream: Path, roots: tuple[Path, ...]) -> list[Path]:
    raw = json.loads(stream.read_text(encoding="utf-8"))
    classes = raw.get("classes")
    if not isinstance(classes, list):
        raise RuntimeError("streaming manifest classes are invalid")
    paths: list[Path] = []
    for class_index, record in enumerate(classes):
        if not isinstance(record, dict):
            raise RuntimeError("streaming manifest class record is invalid")
        for split in ("train", "test"):
            shards = record.get(split)
            if not isinstance(shards, list):
                raise RuntimeError(f"streaming manifest {split} shards are invalid")
            for shard_index, item in enumerate(shards):
                relative = item if isinstance(item, str) else item.get("path") if isinstance(item, dict) else None
                if not isinstance(relative, str):
                    raise RuntimeError("streaming manifest shard path is invalid")
                candidate = confined_file(
                        stream.parent / relative,
                        roots,
                        f"classes[{class_index}].{split}[{shard_index}]",
                    )
                if not candidate.is_relative_to(stream.parent):
                    raise RuntimeError("dataset shard escapes the bound cache directory")
                paths.append(candidate)
    # load_manifest also reads these fixed sidecars for the ofra-fullcache source.
    for filename in ("manifest.json", "split_overlap_audit.json"):
        confined_file(stream.parent / filename, roots, f"streaming sidecar {filename}")
    return paths


def verify(path: Path, *, allowed_home: Path, allowed_scratch: Path) -> dict:
    roots = (
        allowed_home.resolve(strict=True),
        allowed_scratch.resolve(strict=True),
    )
    binding_path = confined_file(path, roots, "submission bindings")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if binding.get("schema_version") != "ofra_submission_bindings_v1":
        raise RuntimeError("unsupported submission-binding schema")
    if int(binding.get("upstream_job_id", -1)) != 388991:
        raise RuntimeError("unexpected upstream job id")
    result_root = allowed_scratch.resolve() / "ofra-etg/mvp/388991/malaya-ft512x12-formal"
    cache_root = allowed_scratch.resolve() / (
        "ofra-etg/transfers/20260730/core/"
        "ofra_formal_v3_cache_20260716/malaya-network-gt"
    )
    expected_files = {
        "training_protocol": result_root / "protocol.json",
        "training_result": result_root / "result_seed_1.json",
        "training_summary": result_root / "summary.json",
        "probe_manifest": result_root / "monitoring/seed_1/probe_manifest.json",
        "streaming_manifest": cache_root / "streaming_manifest.json",
        "fullcache_manifest": cache_root / "manifest.json",
        "feature_schema": cache_root / "feature_schema.json",
        "split_overlap_audit": cache_root / "split_overlap_audit.json",
    }
    if set(binding.get("files", {})) != set(expected_files):
        raise RuntimeError("submission binding file registry is not exact")
    for name, record in binding["files"].items():
        candidate = confined_file(Path(record["path"]), roots, f"bound file {name}")
        if candidate != expected_files[name].resolve(strict=True):
            raise RuntimeError(f"bound file path is not canonical: {name}")
        actual = sha256_file(candidate)
        if actual != record["sha256"]:
            raise RuntimeError(f"input binding mismatch: {name}")
    stream = confined_file(
        Path(binding["files"]["streaming_manifest"]["path"]),
        roots,
        "streaming manifest",
    )
    prechecked_shards = _manifest_shard_paths(stream, roots)
    manifest = load_manifest(stream, verify_hashes=True)
    if manifest.manifest_sha256 != binding["files"]["streaming_manifest"]["sha256"]:
        raise RuntimeError("streaming manifest identity mismatch")
    verified_checkpoints = []
    for record in binding["checkpoint_artifacts"]:
        checkpoint = int(record["checkpoint"])
        manifest_path = confined_file(
            Path(record["manifest_path"]), roots, "checkpoint manifest"
        )
        expected_manifest = (
            result_root
            / f"monitoring/seed_1/checkpoint_{checkpoint:03d}/checkpoint_manifest.json"
        ).resolve(strict=True)
        if manifest_path != expected_manifest:
            raise RuntimeError("checkpoint manifest path is not canonical")
        if sha256_file(manifest_path) != record["manifest_file_sha256"]:
            raise RuntimeError("checkpoint manifest file mismatch")
        raw_checkpoint = json.loads(manifest_path.read_text(encoding="utf-8"))
        for field in ("inference_state_file", "probe_scores_file"):
            relative = raw_checkpoint.get(field)
            if not isinstance(relative, str):
                raise RuntimeError(f"checkpoint {field} is invalid")
            internal = confined_file(
                manifest_path.parent / relative,
                roots,
                f"checkpoint {record['checkpoint']} {field}",
            )
            if internal.parent != manifest_path.parent:
                raise RuntimeError("checkpoint internal file escapes its checkpoint directory")
        metadata = validate_checkpoint_manifest(manifest_path)
        for field in (
            "canonical_sha256",
            "inference_state_sha256",
            "probe_scores_sha256",
        ):
            expected_field = (
                "manifest_canonical_sha256"
                if field == "canonical_sha256"
                else field
            )
            if metadata[field] != record[expected_field]:
                raise RuntimeError(
                    f"checkpoint {record['checkpoint']} binding mismatch: {field}"
                )
        verified_checkpoints.append(int(record["checkpoint"]))
    return {
        "status": "verified",
        "upstream_job_id": int(binding["upstream_job_id"]),
        "checkpoint_count": len(verified_checkpoints),
        "checkpoints": verified_checkpoints,
        "dataset_shards": sum(
            len(record.train) + len(record.test) for record in manifest.classes
        ),
        "prechecked_dataset_shards": len(prechecked_shards),
        "authorized_roots": [str(root) for root in roots],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--allowed-home", type=Path, required=True)
    parser.add_argument("--allowed-scratch", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(
        args.bindings,
        allowed_home=args.allowed_home,
        allowed_scratch=args.allowed_scratch,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
