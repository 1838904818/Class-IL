"""Export exactly the completed CIC17 Section39 assignment; no model execution.

The source root is explicit. Only the six assigned cell paths are opened, never
CSE18 or a recursive experiment search. Existing publication cells are immutable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_results import ROOT, SECTION, derived_copy, private_fields, read, sanitize, write_derived

LAUNCHER = "a4d55cd8e8d4087ea67f33a26535c721547483c3"
LAUNCHER_SHA = "b0a76a0dbb8ef7d2acd488e584eedd4219990b917b10bf576e4bb82bbdedded5"
SELECTION_SHA = "cd724fbb55c6f00678efc3dcf9d9df2122cd2fccae0290f6c5626b281a4d366e"
PACKAGE_SHA = "d7d9912bde5a8f30147b8873cd054484aca66caab57923998ae2975a656b5650"
TRAINING = "c97f83b1bfa14f4d2665ff33c45c897941e9ddb3"
REFERENCE = "270121272848087e4af85ec5290eaede0102d776"
WRAPPER = "045b86222c24d23389edae8ae75c0dd30fd80a42"
ASSIGNMENT = [("aacil_kd_wa", s) for s in (1, 2, 3, 4, 42)] + [("replay_bs", 42)]
TASKS = [[0, 1], [2, 3], [4, 5], [6, 7]]
DATA_SHA = {
    "x_train": "2d2c3943ec6898918dfa6bd83754107ad85214eec067270aa117edf7e271b0f5",
    "y_train": "fb3795e0b33c5d44bada0a8ed953fa56a4dd7a23ee30b0a8f80bd869ba82a6a8",
    "x_test": "07d826ed97e1dc167e94ed35c1109131c3417ab65e0c0db9637376cab2ab7ae2",
    "y_test": "871e4fd795d83839f2ae4d45be6c7e67a1efe17dbcb2ff5dd341be1d1e4eb0f0",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve()
    require(root.name == "section39_local_cic_v1", "Expected the dedicated Section39 formal root")
    exits = sorted((root / "attempts").glob("*/job_exit.json"))
    require(len(exits) == 1, "Review all physical attempts before exporting a resumed assignment")
    job, job_hash = read(exits[0])
    require(job["status"] == "complete" and job["current_cell"] is None
            and not job["smoke"] and not job["resume"], "Assignment is not a complete first formal attempt")
    require(job["launcher_commit"] == LAUNCHER and job["launcher_sha256"] == LAUNCHER_SHA
            and job["training_commit"] == TRAINING and job["package_sha256"] == PACKAGE_SHA
            and job["selection_sha256"] == SELECTION_SHA, "Assignment identity mismatch")
    require(job["planned_cells"] == [list(v) for v in ASSIGNMENT]
            and [(c["method"], c["seed"]) for c in job["cells"]] == ASSIGNMENT
            and all(c["exit_code"] == 0 for c in job["cells"]), "Six unique successful cells required")
    require(job["cache"]["dataset"] == "CIC-IDS-2017"
            and job["cache"]["train_rows"] == 2262203 and job["cache"]["test_rows"] == 565673,
            "Full CIC17 assignment required")
    label = lambda p: "section39_local_cic_v1/" + p.resolve().relative_to(root).as_posix()
    prepared = []
    for method, seed in ASSIGNMENT:
        folder = root / "cells" / method / f"seed_{seed}"
        result_path = folder / "raw/result.json"
        source, source_hash = read(result_path)
        wrapper_path = folder / ("section32_result.json" if method == "aacil_kd_wa" else "section31_result.json")
        wrapper, wrapper_hash = read(wrapper_path)
        invocation, invocation_hash = read(folder / "section39_invocation.json")
        require(invocation["launcher_commit"] == LAUNCHER and invocation["launcher_sha256"] == LAUNCHER_SHA
                and invocation["actual_entrypoint"] == "ncd/run_section39_local.py"
                and invocation["dataset"] == "CIC-IDS-2017" and invocation["method"] == method
                and invocation["seed"] == seed and invocation["smoke"] is False
                and invocation["selection_changed"] is False
                and invocation["CSE18_model_metrics_accessed"] is False
                and invocation["legacy_shift_cache_validation_only"] is True, "Invocation mismatch")
        ident, cfg = source["identity"], source["identity"]["effective_config"]
        require(source["completed"] is True and wrapper["completed"] is True
                and wrapper["raw_result_sha256"] == source_hash and wrapper["smoke"] is False,
                "Raw/wrapper completion or hash mismatch")
        for record in (ident, wrapper):
            require((record["dataset"], record["method"], record["seed"]) == ("CIC-IDS-2017", method, seed),
                    "Cell identity mismatch")
        require(ident["git_commit"] == (TRAINING if method == "aacil_kd_wa" else REFERENCE)
                and ident["config_sha256"] == "fd631316de03fa6723d6a585f23b827c8646a738e1119d9fe4b0caa29e87c3bb"
                and ident["source_hashes"] == wrapper["identity"]["source_hashes"], "Frozen source/config mismatch")
        require(source["tasks"] == TASKS and ident["tasks"] == TASKS
                and len(source["checkpoints"]) == 4 and cfg["smoke"] is False
                and not cfg.get("selection_only") and not source.get("selection_only"), "Formal trajectory required")
        for key, value in {"hidden": 128, "batch_size": 256, "epochs_per_task": 8,
                           "learning_rate": .001, "buffer_per_class": 200,
                           "min_steps_per_task": 2000, "normal_training_rows": 50000,
                           "cpu_threads": 2, "device": "cuda"}.items():
            require(cfg[key] == value, "Fixed configuration mismatch: " + key)
        for key, digest in DATA_SHA.items():
            binding = ident["data_binding"][key]
            require(binding["sha256"] == digest and binding["shape"][0] ==
                    (2262203 if "train" in key else 565673), "CIC data binding mismatch")
        if method == "aacil_kd_wa":
            selected = ident["frozen_selection"]
            require(cfg["lambda_g"] == .5 and cfg["derpp_alpha"] == cfg["derpp_beta"] == 1
                    and ident["wrapper_source_commit"] == WRAPPER
                    and selected["selection_sha256"] == SELECTION_SHA and selected["eligible"] is False
                    and selected["status"] == "forgetting_constraint_not_satisfied", "Frozen fallback changed")
        for index, cp in enumerate(source["checkpoints"]):
            require(cp["task_index"] == index and cp["seen_classes"] == list(range((index + 1) * 2))
                    and cp["primary_alpha"] == .02 and "confusion_matrix" in cp["operating_points"]["0.02"],
                    "Saved alpha2 trajectory missing")
        cost = next(c for c in job["cells"] if (c["method"], c["seed"]) == (method, seed))
        execution = {
            "invocation": sanitize(invocation), "invocation_sha256": invocation_hash,
            "wrapper_record": label(wrapper_path), "wrapper_record_sha256": wrapper_hash,
            "wrapper_identity": wrapper["identity"],
            "wrapper_actual_entrypoint": wrapper["actual_entrypoint"],
            "wrapper_cost_seconds": wrapper["cost_seconds"],
            "wrapper_adapter_cost_seconds": wrapper["adapter_cost_seconds"],
            "checkpoint_resume": wrapper["checkpoint_resume"],
            "assignment_exit_sha256": job_hash, "assignment_attempt": exits[0].parent.name,
            "assignment_started_at": job["started_at"], "assignment_ended_at": job["ended_at"],
            "cell_attempt": {k: v for k, v in cost.items() if k != "command"},
            "pre_training_failure": {
                "occurred_at": "2026-10-20T11:32:42+08:00",
                "phase": "cache receipt validation before child, attempt directory or GPU training",
                "cause": "Legacy CIC17 receipt has six arrays; original wrapper expected four",
                "gpu_training_seconds": 0, "wall_seconds": None,
                "wall_seconds_status": "not measured; retained stderr is not a duration measurement",
            },
            "cost_scope": "Original raw, wrapper and parent timings are nested, not additive. Device utilization includes other desktop activity; not an exclusive hardware benchmark.",
            "evidence_role": "CIC17 development-descriptive completion, not blind confirmation; CSE18 remains sealed.",
            "legacy_shift_role": job["legacy_shift_cache_role"],
        }
        destination = SECTION / "publication_v1/cic_ids_2017" / f"{method}_seed{seed}"
        exported, _ = derived_copy(source, source_hash, result_path, result_path, source_hash, label)
        exported["section39_execution"] = execution
        prepared.append((destination / "result.json", exported))
        for cp in source["checkpoints"]:
            cp_path = Path(cp["checkpoint_file"])
            require(cp_path.as_posix() == f"checkpoints/checkpoint_{cp['task_index']:03d}.pt",
                    "Unexpected checkpoint name")
            sidecar_path = result_path.parent / cp_path.with_suffix(".json")
            sidecar, sidecar_hash = read(sidecar_path)
            require(sidecar == cp, "Sidecar differs from saved raw checkpoint record")
            exported, _ = derived_copy(sidecar, sidecar_hash, sidecar_path, result_path, source_hash, label)
            prepared.append((destination / cp_path.with_suffix(".json"), exported))
    # Validate every new artifact before writing any; never overwrite a different
    # published cell and never enumerate or regenerate old result groups.
    for path, value in prepared:
        require(not list(private_fields(value)), "Private path in publication artifact")
        if path.exists():
            require(read(path)[0] == value, "Different publication already exists: " + path.name)
    written = sum(write_derived(path, value) for path, value in prepared)
    print(json.dumps({"completed_cells": 6, "small_json_files": len(prepared), "written": written,
                      "no_training_inference_or_checkpoint_reads": True}))


if __name__ == "__main__":
    main()
