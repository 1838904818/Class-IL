#!/usr/bin/env python3
"""Fail-closed verification of five source-bound training seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from formal_v2_explanation_etg.attribution_scope import ATTRIBUTION_TARGET_SCOPE
from streaming_full.data import canonical_sha256, load_manifest, sha256_file
from streaming_full.monitoring import validate_checkpoint_manifest, validate_monitoring_result


SEEDS = (1, 2, 3, 4, 42)
JOBS = {1: 388991, 2: 394503, 3: 394646, 4: 394745, 42: 412039}


def confined(path: Path, roots: tuple[Path, ...], label: str) -> Path:
    value = path.resolve(strict=True)
    if not value.is_file() or not any(value.is_relative_to(root) for root in roots):
        raise RuntimeError(f"{label} is outside authorized regular files")
    return value


def validate_binding_shape(binding: dict) -> None:
    if binding.get("schema_version") != "ofra_attribution_multiseed_bindings_v2":
        raise RuntimeError("unsupported binding schema")
    if binding.get("dataset") != "malaya-network-gt":
        raise RuntimeError("binding dataset mismatch")
    if tuple(binding.get("analysis_seeds", [])) != (1, 2, 3, 4, 42):
        raise RuntimeError("analysis seed registry mismatch")
    if tuple(binding.get("computed_seeds", [])) != SEEDS:
        raise RuntimeError("computed seed registry mismatch")
    expected_contract = {
        "score_target": "joint_cap3000 class margin",
        "explainer_primary": "shap.GradientExplainer Expected Gradients",
        "robustness_methods": [
            "expected_gradients",
            "feature_ablation",
            "gradient_x_input",
        ],
        "nsamples": 64,
        "primary_top_k": 15,
        "primary_jaccard_threshold": 0.7,
        "primary_allowed_recall_drop": 0.05,
        "etg_scope": "offline_post_hoc_non_suppressing_ledger_only",
        "attribution_target_scope": ATTRIBUTION_TARGET_SCOPE,
        "cross_device_equivalence_claimed": False,
        "batch_partition_equivalence_claimed": False,
        "selection_policy": "fixed protocol; no test-driven model or threshold selection",
    }
    if binding.get("analysis_contract") != expected_contract:
        raise RuntimeError("analysis contract mismatch or numerical equivalence claim")
    records = binding.get("seeds")
    if not isinstance(records, list) or tuple(int(row.get("seed", -1)) for row in records) != SEEDS:
        raise RuntimeError("seed binding records are not exact or ordered")
    if any(int(row.get("upstream_job_id", -1)) != JOBS[int(row["seed"])] for row in records):
        raise RuntimeError("upstream training job registry mismatch")


def verify(binding_path: Path, operation_dir: Path, home: Path, scratch: Path) -> dict:
    roots = (home.resolve(strict=True), scratch.resolve(strict=True))
    binding_path = confined(binding_path, roots, "bindings")
    operation_dir = operation_dir.resolve(strict=True)
    if not operation_dir.is_relative_to(roots[0]):
        raise RuntimeError("operation directory is outside HOME")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    validate_binding_shape(binding)
    cache = scratch.resolve() / "ofra-etg/transfers/20260730/core/ofra_formal_v3_cache_20260716/malaya-network-gt"
    shared_expected = {
        "streaming_manifest": cache / "streaming_manifest.json",
        "fullcache_manifest": cache / "manifest.json",
        "feature_schema": cache / "feature_schema.json",
        "split_overlap_audit": cache / "split_overlap_audit.json",
    }
    if set(binding.get("shared_files", {})) != set(shared_expected):
        raise RuntimeError("shared file registry is not exact")
    for name, expected in shared_expected.items():
        record = binding["shared_files"][name]
        path = confined(Path(record["path"]), roots, f"shared {name}")
        if path != expected.resolve(strict=True) or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"shared binding mismatch: {name}")
    manifest = load_manifest(shared_expected["streaming_manifest"], verify_hashes=True)
    if manifest.dataset != "malaya-network-gt":
        raise RuntimeError("loaded manifest dataset mismatch")

    verified = []
    for record in binding["seeds"]:
        seed = int(record["seed"])
        if seed == 1:
            root = scratch.resolve() / "ofra-etg/mvp/388991/malaya-ft512x12-formal"
        else:
            root = scratch.resolve() / f"ofra-etg/final-ft512x12-20260809/malaya-network-gt/seed-{seed}"
        if Path(record["result_root"]).resolve(strict=True) != root.resolve(strict=True):
            raise RuntimeError(f"seed {seed}: result root mismatch")
        expected_files = {
            "training_protocol": root / "protocol.json",
            "training_result": root / f"result_seed_{seed}.json",
            "training_summary": root / "summary.json",
            "probe_manifest": root / f"monitoring/seed_{seed}/probe_manifest.json",
        }
        if set(record.get("files", {})) != set(expected_files):
            raise RuntimeError(f"seed {seed}: file registry mismatch")
        for name, expected in expected_files.items():
            entry = record["files"][name]
            path = confined(Path(entry["path"]), roots, f"seed {seed} {name}")
            if path != expected.resolve(strict=True) or sha256_file(path) != entry["sha256"]:
                raise RuntimeError(f"seed {seed}: file binding mismatch: {name}")
        protocol = json.loads(expected_files["training_protocol"].read_text(encoding="utf-8"))
        result = json.loads(expected_files["training_result"].read_text(encoding="utf-8"))
        protocol_hash = canonical_sha256({k: v for k, v in protocol.items() if k != "protocol_sha256"})
        if protocol.get("protocol_sha256") != protocol_hash or result.get("protocol_sha256") != protocol_hash:
            raise RuntimeError(f"seed {seed}: protocol canonical mismatch")
        if int(result.get("seed", -1)) != seed:
            raise RuntimeError(f"seed {seed}: training result scope mismatch")
        without_timing = {
            key: value for key, value in result.items() if key not in {"timing", "deterministic_result_sha256"}
        }
        if result.get("deterministic_result_sha256") != canonical_sha256(without_timing):
            raise RuntimeError(f"seed {seed}: deterministic result mismatch")
        validate_monitoring_result(result, output_base=root, expected_protocol=protocol["monitoring"])
        checkpoint_records = record.get("checkpoint_artifacts")
        if not isinstance(checkpoint_records, list) or len(checkpoint_records) != 5:
            raise RuntimeError(f"seed {seed}: checkpoint registry mismatch")
        for checkpoint_entry in checkpoint_records:
            index = int(checkpoint_entry["checkpoint"])
            expected = root / f"monitoring/seed_{seed}/checkpoint_{index:03d}/checkpoint_manifest.json"
            path = confined(Path(checkpoint_entry["manifest_path"]), roots, "checkpoint manifest")
            if path != expected.resolve(strict=True) or sha256_file(path) != checkpoint_entry["manifest_file_sha256"]:
                raise RuntimeError(f"seed {seed}: checkpoint manifest mismatch")
            metadata = validate_checkpoint_manifest(path)
            for field, bound_field in (
                ("canonical_sha256", "manifest_canonical_sha256"),
                ("inference_state_sha256", "inference_state_sha256"),
                ("probe_scores_sha256", "probe_scores_sha256"),
            ):
                if metadata[field] != checkpoint_entry[bound_field]:
                    raise RuntimeError(f"seed {seed}: checkpoint internal binding mismatch")
        verified.append(seed)

    return {"status": "verified", "computed_seeds": verified, "analysis_seeds": [1, 2, 3, 4, 42]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--operation-dir", type=Path, required=True)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.bindings, args.operation_dir, args.home, args.scratch), sort_keys=True))


if __name__ == "__main__":
    main()
