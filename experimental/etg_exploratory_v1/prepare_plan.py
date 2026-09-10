"""Read-only metadata planning. Does not load feature or model tensors."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from pilot_core import digest, partition_indices, require


def load_auditor():
    path = Path(__file__).resolve().parents[1] / "etg_input_audit" / "audit_inputs.py"
    require(hashlib.sha256(path.read_bytes()).hexdigest() ==
            "a93389778d57ed0abf9c80f65c3a1add014e879279502d90553d498f54b22a5b",
            "auditor dependency changed before import")
    spec = importlib.util.spec_from_file_location("etg_input_audit_bound", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(root):
    root = Path(root).resolve()
    policy_path = Path(__file__).with_name("policy.json")
    p = json.loads(policy_path.read_text(encoding="utf-8"))
    require(p["schema"] == "etg-early-descriptive-v1" and p["classes"] == [0, 1, 2, 3]
            and p["old_classes"] == [0, 1] and p["training_seed"] == 1
            and p["old_checkpoint"] == 0 and p["new_checkpoint"] == 1, "fixed pilot identity changed")
    a = load_auditor()
    native = root / "work/replayids_score_readiness_20260907"
    audit_path = native / "sampling_audit.json"
    require(a.sha(audit_path) == p["sampling_audit_sha256"], "sampling audit changed")
    require(a.sha(native / "streaming_manifest.json") == p["native_manifest_sha256"], "native manifest changed")
    source = root / "work/replayids_ofra_seed42_build/ofra_streaming_cache_v2/streaming_manifest.json"
    require(a.sha(source) == p["source_manifest_sha256"], "source manifest changed")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    require(audit["source_manifest"]["sha256"] == p["source_manifest_sha256"], "source mismatch")
    records = {c["id"]: c for c in audit["classes"]}
    rows = []
    for cid in p["classes"]:
        cal, fit = a.reconstruct_indices(records[cid], p["source_manifest_sha256"], 42)
        roles = partition_indices(cal, fit, cid, p["split_salt"], p["role_counts_per_class"])
        item = {"class_id": cid, "native_calibration_rows": len(cal), "roles": {}}
        for role, offsets in roles.items():
            ordinal = np.searchsorted(cal, offsets)
            require(np.array_equal(cal[ordinal], offsets), "calibration ordinal mismatch")
            item["roles"][role] = {"rows": len(offsets), "source_offsets_sha256": digest(offsets),
                                  "calibration_ordinals_sha256": digest(ordinal.tolist()),
                                  "native_class_batch512_context_sha256": digest([[int(i)//512, int(i)%512] for i in ordinal])}
        selected = set(i for part in roles.values() for i in part)
        require(len(selected) == sum(p["role_counts_per_class"].values()), "role overlap")
        item["unused_calibration_rows"] = len(cal) - len(selected)
        if cid in p["old_classes"]:
            bg = sorted(sorted(fit.tolist(), key=lambda i: (digest([p["split_salt"], "background", cid, i]), i))[:p["background_per_old_class"]])
            require(len(bg) == p["background_per_old_class"] and not selected.intersection(bg), "invalid background")
            item["background"] = {"rows": len(bg), "source_offsets_sha256": digest(bg), "origin": "native task-0 fitting pool"}
        rows.append(item)
    coverage = json.loads((native / "CHECKPOINT_COVERAGE.json").read_text(encoding="utf-8"))
    bound = []
    for cp, seen in [(0, [0, 1]), (1, [0, 1, 2, 3])]:
        expected = next(x for x in coverage["rows"] if x["seed"] == 1 and x["checkpoint"] == cp)
        cpdir = native / "seed_1" / f"checkpoint_{cp:03d}"
        manifest = json.loads((cpdir / "checkpoint_manifest.json").read_text(encoding="utf-8"))
        require(manifest["seed"] == 1 and manifest["checkpoint"] == cp and manifest["seen_classes"] == seen, "checkpoint identity mismatch")
        hashes = {}
        for name in ("checkpoint_manifest.json", "inference_state.npz"):
            actual = a.sha(cpdir / name)
            require(actual == expected["artifact_sha256"][name], "checkpoint file digest mismatch")
            hashes[name] = actual
        require(manifest["inference_state_sha256"] == hashes["inference_state.npz"], "state binding mismatch")
        bound.append({"seed": 1, "checkpoint": cp, "seen_classes": seen, "sha256": hashes})
    return {"schema": p["schema"], "status": "METADATA_PLAN_VERIFIED_NOT_EXPERIMENT_READY",
            "policy_sha256": a.sha(policy_path), "planner_sha256": a.sha(__file__),
            "decision_core_sha256": a.sha(Path(__file__).with_name("pilot_core.py")),
            "auditor_sha256": a.sha(Path(a.__file__)),
            "coverage_receipt_sha256": a.sha(native / "CHECKPOINT_COVERAGE.json"),
            "source_manifest_sha256": p["source_manifest_sha256"],
            "native_manifest_sha256": p["native_manifest_sha256"],
            "sampling_audit_sha256": p["sampling_audit_sha256"],
            "rows": rows, "checkpoints": bound,
            "total_role_rows": sum(p["role_counts_per_class"].values()) * 4,
            "attribution_target_rows": p["role_counts_per_class"]["trigger"] * 2,
            "whole_test_evaluated": False, "model_or_feature_tensors_loaded": False,
            "capture_independence_established": False, "real_execution_ready": False,
            "remaining_gates": p["required_before_execution"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.workspace_root), indent=2, allow_nan=False))
