"""Read-only ETG input audit: metadata, identities and count feasibility only.

Never imports an experiment driver, reads feature/checkpoint tensors, fits a
model, computes attribution, creates split arrays or contacts a remote service.
JSON is emitted to stdout; the caller decides where to archive it.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        h = hashlib.file_digest(handle, "sha256")
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "input escapes its declared root")
    require(path.is_file(), "required input absent")
    require(path.stat().st_size <= 32 * 1024 * 1024, "metadata input exceeds 32 MiB bound")
    return path


def reconstruct_indices(record, source_sha, master_seed):
    """Independently mirror the bound D2 index arithmetic, then verify digests."""
    n, cal, fit = record["source_train_rows"], record["calibration_rows"], record["fit_rows"]
    require(type(n) is int and 0 < n <= 1000000, "index-allocation ceiling exceeded")
    require(all(type(v) is int and 0 <= v <= n for v in (cal, fit)) and cal + fit <= n,
            "invalid recorded partition counts")
    payload = f"{master_seed}|{source_sha}|{record['id']}|normal_to_largest_attack|v2"
    seed = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")
    require(seed == record["seed"], "historical seed reconstruction mismatch")
    order = np.random.default_rng(seed).permutation(n)
    c = np.sort(order[:cal])
    f = np.sort(order[cal:cal + fit])
    require(canonical_sha(c.tolist()) == record["calibration_indices_sha256"], "calibration index digest mismatch")
    require(canonical_sha(f.tolist()) == record["fit_indices_sha256"], "fitting index digest mismatch")
    require(np.intersect1d(c, f).size == 0, "historical fit/calibration overlap")
    return c, f


def overlap_counts(prospective, calibration, fitting):
    return {"matches_native_calibration": int(np.intersect1d(prospective, calibration).size),
            "overlaps_native_fit": int(np.intersect1d(prospective, fitting).size),
            "outside_both": int(len(prospective) - np.intersect1d(prospective, np.union1d(calibration, fitting)).size)}


def radius(groups, family_size, alpha=0.05):
    require(groups > 0 and family_size > 0 and 0 < alpha < 1, "invalid illustrative bound inputs")
    return 2 * math.sqrt(math.log(2 * family_size / alpha) / (2 * groups))


def audit(root):
    root = Path(root).resolve()
    aliases = {
        "native_sampling_audit": "work/replayids_score_readiness_20260907/sampling_audit.json",
        "native_manifest": "work/replayids_score_readiness_20260907/streaming_manifest.json",
        "native_builder": "work/replayids_score_readiness_20260907/build_train_protocol_v2.py",
        "native_helper": "work/replayids_score_readiness_20260907/build_train_protocol.py",
        "source_manifest": "work/replayids_ofra_seed42_build/ofra_streaming_cache_v2/streaming_manifest.json",
        "data_report": "work/replayids_ofra_seed42_build/dataset/data_report.json",
        "native_protocol": "work/github_public_release_20260903/results/balanced-replay-five-seed/replayids_d2_seed1_protocol.json",
    }
    bindings = {}

    def bind(name, relative, expected=None):
        path = contained(root, relative)
        actual = sha(path)
        if expected is not None:
            require(actual == expected, f"hash mismatch: {name}")
        bindings[name] = actual
        return path

    def read(name, relative, expected=None):
        return json.loads(bind(name, relative, expected).read_text(encoding="utf-8"))

    d2 = read("native_sampling_audit", aliases["native_sampling_audit"],
              "697750d599f448ba1120ba60decac98abb607204eec1a431073673cc3b124b9b")
    protocol = read("native_protocol", aliases["native_protocol"])
    native = read("native_manifest", aliases["native_manifest"], protocol["manifest_sha256"])
    source = read("source_manifest", aliases["source_manifest"], d2["source_manifest"]["sha256"])
    bind("native_builder", aliases["native_builder"], d2["builder"]["sha256"])
    bind("native_helper", aliases["native_helper"], d2["helper_dependency"]["sha256"])
    data = read("data_report", aliases["data_report"],
                "c5213fcdb0a0e27655b03e178c411f0911727bb3b63ffbf3a0f34308126b0d9e")
    source_classes = {c["id"]: c for c in source["classes"]}
    native_classes = {c["id"]: c for c in native["classes"]}
    historical = {c["id"]: c for c in d2["classes"]}
    records, group_modes = [], []
    for inc in range(4):
        relative = f"work/prospective_real_stage_20260908/P_{inc:02d}"
        m = read(f"P_{inc:02d}_manifest", f"{relative}/manifest.json")
        other_relative = f"work/prospective_real_stage_20260908/O_{inc:02d}"
        other = read(f"O_{inc:02d}_manifest", f"{other_relative}/manifest.json")
        read(f"P_{inc:02d}_contract", f"work/prospective_real_stage_20260908/contracts/increment_{inc:02d}.json",
             m["binding"]["input_sha256"])
        require(m["increment"] == inc, "increment identity mismatch")
        group_modes.append(m["group_mode"])
        arrays = {}
        for name in ("labels", "source_index", "row_ids", "group_ids"):
            ref = m["collections"]["calibration"][name]
            other_ref = other["collections"]["calibration"][name]
            require(ref == other_ref, "P/O calibration metadata descriptors differ")
            bind(f"O_{inc:02d}_calibration_{name}", f"{other_relative}/{other_ref['path']}", ref["sha256"])
            path = bind(f"P_{inc:02d}_calibration_{name}", f"{relative}/{ref['path']}", ref["sha256"])
            values = np.load(path, allow_pickle=False)
            require(list(values.shape) == ref["shape"] and values.dtype.str == ref["dtype"], "array descriptor mismatch")
            arrays[name] = values
        n = m["collections"]["calibration"]["rows"]
        require(all(len(v) == n for v in arrays.values()), "unaligned metadata arrays")
        require(len(np.unique(arrays["row_ids"])) == n, "duplicate calibration row identity")
        groups_equal_rows = bool(np.array_equal(arrays["row_ids"], arrays["group_ids"]))
        for c in m["classes"]:
            cid = c["class_id"]
            mask = arrays["labels"] == cid
            require(int(mask.sum()) == c["calibration_rows"], "class-count mismatch")
            slots = np.unique(arrays["source_index"][mask, 0])
            require(len(slots) == 1, "this audit supports one bound source shard per class")
            shard = m["source_shards"][int(slots[0])]
            require(shard["class_id"] == cid, "source-slot class mismatch")
            require(shard["features"]["sha256"] == source_classes[cid]["train"][0]["sha256"], "source shard mismatch")
            offsets = arrays["source_index"][mask, 1]
            expected_ids = np.asarray([canonical_sha(["source-row-v1", m["namespace"], shard["shard_id"], int(offset)])
                                       for offset in offsets], dtype="S64")
            require(np.array_equal(expected_ids, arrays["row_ids"][mask]), "row identity does not match original shard offsets")
            old = historical[cid]
            require(old["source_train_rows"] == c["source_rows"], "source count mismatch")
            require(native_classes[cid]["train"][0]["sha256"] == old["fit"]["sha256"], "native fitting-array binding mismatch")
            require(len(np.unique(offsets)) == len(offsets) and np.all(offsets >= 0)
                    and np.all(offsets < old["source_train_rows"]), "invalid prospective offsets")
            cal, fit = reconstruct_indices(old, d2["source_manifest"]["sha256"], d2["configuration"]["seed"])
            records.append({"increment": inc, "class_id": cid, "class_name": c["name"],
                            "source_training_rows": c["source_rows"], "prospective_fit_rows": c["fit_rows"],
                            "calibration_rows": len(offsets), "group_ids_equal_row_ids": groups_equal_rows,
                            "both_historical_index_hashes_verified": True,
                            **overlap_counts(offsets, cal, fit)})
    require({r["class_id"] for r in records} == set(historical) == set(native_classes)
            and len(records) == len(historical), "incomplete or duplicate class coverage")
    missing_fields = [x for x in ("Timestamp", "Flow ID", "Source IP", "Destination IP") if x not in data["features"]]
    return {"schema": "etg-input-feasibility-audit-v1", "audit_status": "COMPLETED",
            "real_execution_ready": False, "scientific_result_generated": False,
            "numpy_version": np.__version__, "scope": "Read-only row-index and metadata audit, no feature/model execution",
            "class_records": records,
            "prospective_calibration_ids_reconstructed": True,
            "P_O_calibration_metadata_bytes_equal": True,
            "total_calibration_rows": sum(r["calibration_rows"] for r in records),
            "total_overlaps_native_fit": sum(r["overlaps_native_fit"] for r in records),
            "total_matches_native_calibration": sum(r["matches_native_calibration"] for r in records),
            "total_outside_both": sum(r["outside_both"] for r in records),
            "group_modes": group_modes, "capture_independence_established": False,
            "registered_raw_file_names": list(data["raw_sha256"]),
            "absent_grouping_columns_in_saved_feature_schema": missing_fields,
            "source_file_group_partition_capacity": {"known_source_files": len(data["raw_sha256"]),
                "required_disjoint_roles": 4, "file_groups_alone_suffice": False,
                "claim": "File identifiers alone cannot fill four disjoint nonempty roles; no claim that days are independent."},
            "class_support_necessary_floor": {"scope": "Count-only lower bound, not a registered numerical policy",
                "minimum_fit_rows_per_seen_class": 1, "minimum_acceptance_groups_per_seen_class": 2,
                "minimum_disjoint_fit_and_acceptance_rows_per_seen_class": 3,
                "classes_below_floor": [r["class_name"] for r in records if r["calibration_rows"] < 3],
                "passing_count_floor_implies_feasibility": False},
            "gates": {"reuse_P_O_pool_with_native_D2_checkpoints": "BLOCKED_TRAINING_ROW_OVERLAP",
                "all_eight_class_strict_acceptance_from_existing_calibration": "BLOCKED_HEARTBLEED_SUPPORT",
                "independent_capture_groups": "NOT_ESTABLISHED",
                "native_four_partition_repeated_attributions": "NOT_VERIFIED"},
            "bound_source_sha256": bindings,
            "limits": ["No feature-array rebuild or model execution.",
                "Exact index overlap is bound to native D2 selection, not proof of each checkpoint's training implementation.",
                "Row-identity exclusion does not prove feature-duplicate or capture independence.",
                "No fresh remote or W&B verification.",
                "No data or numerical gate is changed by this audit."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", required=True)
    args = parser.parse_args()
    result = audit(args.workspace_root)
    print(json.dumps(result, indent=2, ensure_ascii=True))
