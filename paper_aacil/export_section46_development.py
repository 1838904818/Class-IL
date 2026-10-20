"""Export one complete five-seed noGKD+WA development dataset, without inference.

Only the explicitly assigned dataset/root is read. Sealed CSE18 paths are never
enumerated. All saved operating points, source identities and costs are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_results import SECTION, derived_copy, private_fields, read, sanitize, write_derived

METHOD = "aacil_nogkd_wa"
SEEDS = (1, 2, 3, 4, 42)
COUNTS = {"UNSW-NB15": (120991, 30248), "NSL-KDD": (125973, 22544),
          "CIC-IDS-2017": (2262203, 565673)}
LOCAL_COMMIT = "30f28e88fe78f9e274f0d8999111c4b99c7ebf23"
LOCAL_SHA = "0be456e8c586cfdc938c8bfe4b76b781464c2f80d790d08f4aae43cdda775f90"
HPC_COMMIT = "69f53314fb0c5e6f3cde41bc90d343eed067e35c"
WRAPPER_COMMIT = "359102b09f389a8c1d59b60d973dd8129c1dd5a7"
PACKAGE_SHA = "3d5f190496f89dd07d10401f40a5e4ea2b0beaaa322e0f32dbc1f00c6dece0d1"
COMPACT_SHA = "36fe10ca4cac4a5b4fe54152cebe3d20419f7b4fcfc3f7a1443c0db07c3deb04"


def require(value, message):
    if not value:
        raise ValueError(message)


def read_shared(path):
    """Snapshot a live Windows status file without preventing atomic replacement."""
    import ctypes
    import msvcrt
    import os
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                                   ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    kernel.CreateFileW.restype = ctypes.c_void_p
    handle = kernel.CreateFileW(str(path), 0x80000000, 7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    with os.fdopen(msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY), "rb") as stream:
        content = stream.read()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=COUNTS, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--local-dataset-complete", action="store_true",
                        help="Export a complete dataset from the Section50 serial CIC17/NSL assignment")
    args = parser.parse_args()
    root = args.source_root.resolve()
    local = args.dataset == "UNSW-NB15" or args.local_dataset_complete
    slug = args.dataset.lower().replace("-", "_")
    exits = sorted((root / "attempts").glob(
        "*/status.json" if args.local_dataset_complete else "*/job_exit.json"))
    require(len(exits) == 1, "Inspect all attempt costs before exporting a resumed/multiple-attempt assignment")
    job, job_hash = (read_shared if args.local_dataset_complete else read)(exits[0])
    if args.local_dataset_complete:
        planned = [[d, s] for d in ("CIC-IDS-2017", "NSL-KDD") for s in SEEDS]
        require(args.dataset in ("CIC-IDS-2017", "NSL-KDD")
                and job["planned_cells"] == planned and job["status"] in ("running", "complete"),
                "Exact Section50 local assignment required")
        require([[c["dataset"], c["seed"]] for c in job["cells"]] == planned[:len(job["cells"])]
                and len(job["cells"]) <= len(planned)
                and all(c["method"] == METHOD and c["exit_code"] == 0 for c in job["cells"]),
                "Successful serial prefix required")
        require(job.get("current_cell") is None
                or job["current_cell"]["dataset"] != args.dataset, "Requested dataset still active")
        selected_cells = [c for c in job["cells"] if c["dataset"] == args.dataset]
    else:
        require(job["status"] == "complete" and job.get("current_cell") is None,
                "Complete first formal assignment required")
        selected_cells = job["cells"]
    require(
        job["smoke"] is False and job["resume"] is False, "First formal assignment required")
    require([(c["method"], c["seed"]) for c in selected_cells] == [(METHOD, s) for s in SEEDS]
            and all(c["exit_code"] == 0 for c in selected_cells), "Exactly five assigned successful cells required")
    if local:
        require(job["launcher_commit"] == LOCAL_COMMIT and job["launcher_sha256"] == LOCAL_SHA
                and job["package_sha256"] == PACKAGE_SHA
                and (args.local_dataset_complete
                     or job["planned_cells"] == [[args.dataset, s] for s in SEEDS]), "Local assignment identity differs")
        cells_root = root / slug
        commit = WRAPPER_COMMIT
    else:
        require(job["adapter_source_commit"] == HPC_COMMIT
                and job["methods"] == [METHOD] and job["seeds"] == list(SEEDS)
                and job["cache"]["dataset"] == args.dataset, "HPC assignment identity differs")
        cells_root, commit = root, HPC_COMMIT
    label = lambda p: f"section46_development/{slug}/" + p.resolve().relative_to(root).as_posix()
    prepared, common = [], None
    for seed in SEEDS:
        folder = cells_root / "cells" / METHOD / f"seed_{seed}"
        raw_path, wrapper_path = folder / "raw/result.json", folder / "section44_result.json"
        raw, raw_hash = read(raw_path)
        wrapper, wrapper_hash = read(wrapper_path)
        ident, cfg = raw["identity"], raw["identity"]["effective_config"]
        require(raw["completed"] is True and wrapper["completed"] is True and wrapper["smoke"] is False
                and wrapper["raw_result_sha256"] == raw_hash and wrapper["identity"] == ident,
                "Raw/wrapper completion or binding mismatch")
        require((ident["dataset"], ident["method"], ident["seed"]) == (args.dataset, METHOD, seed)
                and ident["git_commit"] == commit and raw["selection_only"] is False,
                "Wrong dataset, method, seed or source")
        require(ident["source_hashes"]["ncd/run_aacil_compact.py"] == COMPACT_SHA
                and cfg["smoke"] is False and cfg["selection_only"] is False
                and cfg["post_unsealing_design"] is True and cfg["independent_dataset_confirmation"] is False,
                "Wrong candidate or evidence role")
        for key, value in {"lambda_g": 0., "derpp_alpha": 1., "derpp_beta": 1.,
            "hidden": 128, "batch_size": 256, "epochs_per_task": 8, "learning_rate": .001,
            "buffer_per_class": 200, "min_steps_per_task": 2000,
            "normal_training_rows": 50000, "cpu_threads": 2, "device": "cuda"}.items():
            require(cfg[key] == value, "Fixed config mismatch: " + key)
        for split, count in zip(("train", "test"), COUNTS[args.dataset]):
            require(ident["data_binding"]["x_" + split]["shape"][0] == count
                    and ident["data_binding"]["y_" + split]["shape"][0] == count, "Full dataset required")
        shared = {k: ident[k] for k in ("source_hashes", "effective_config_sha256", "data_binding",
                                      "data_metadata_sha256", "tasks", "class_order")}
        if common is None:
            common = shared
        require(shared == common and raw["tasks"] == ident["tasks"]
                and len(raw["checkpoints"]) == len(raw["tasks"]), "Five-seed source/data/task mismatch")
        invocation = None
        if local:
            invocation, invocation_hash = read(folder / "section44_local_invocation.json")
            require(invocation["launcher_commit"] == LOCAL_COMMIT and invocation["launcher_sha256"] == LOCAL_SHA
                    and invocation["dataset"] == args.dataset and invocation["seed"] == seed
                    and invocation["smoke"] is False and invocation["selection_changed"] is False
                    and invocation["CSE18_model_metrics_accessed"] is False, "Local invocation mismatch")
        execution = {
            "assignment_exit_sha256": job_hash, "assignment_attempt": exits[0].parent.name,
            "assignment_started_at": job["started_at"], "assignment_ended_at": job.get("ended_at"),
            "job_id": job.get("job_id"), "launcher_commit": LOCAL_COMMIT if local else HPC_COMMIT,
            "cell_attempt": {k: v for k, v in next(c for c in selected_cells if c["seed"] == seed).items() if k != "command"},
            "wrapper_record": label(wrapper_path), "wrapper_sha256": wrapper_hash,
            "wrapper_actual_entrypoint": wrapper["actual_entrypoint"],
            "wrapper_command": sanitize({"command": wrapper["command"]})["command"],
            "wrapper_cost_seconds": wrapper["cost_seconds"],
            "adapter_cost_seconds": wrapper["adapter_cost_seconds"],
            "descriptive_development_dataset": True, "post_unsealing_design": True,
            "independent_dataset_confirmation": False, "selection_changed": False,
            "cost_scope": "Raw, wrapper and parent times are nested, not additive; local GPU utilization includes desktop activity.",
            "legacy_shift_role": "CIC17 loader validates stored shift arrays but does not train or evaluate them." if args.dataset == "CIC-IDS-2017" else None}
        if invocation is not None:
            execution.update(local_invocation=sanitize(invocation), local_invocation_sha256=invocation_hash)
        if args.local_dataset_complete:
            execution.pop("assignment_exit_sha256")
            execution.update(assignment_status_snapshot_sha256=job_hash,
                             assignment_status_at_export=job["status"],
                             dataset_completed_at=selected_cells[-1]["ended_at"],
                             complete_dataset_only=True,
                             other_dataset_metrics_read=False,
                             superseded_pending_hpc_job={"job_id": 489810 if args.dataset == "CIC-IDS-2017" else 489811,
                                 "state": "CANCELLED", "started_at": None,
                                 "ended_at": "2026-10-20T14:52:34+08:00",
                                 "elapsed_seconds": 0, "allocated_cpu_seconds": 0,
                                 "reason": "Section50 reassignment before resource allocation"})
        destination = SECTION / "publication_v1" / slug / f"{METHOD}_seed{seed}"
        exported, _ = derived_copy(raw, raw_hash, raw_path, raw_path, raw_hash, label)
        exported["section46_execution"] = execution
        prepared.append((destination / "result.json", exported))
        for index, cp in enumerate(raw["checkpoints"]):
            require(cp["task_index"] == index and cp["primary_alpha"] == .02
                    and all("confusion_matrix" in cp["operating_points"][a] for a in ("0.01", "0.02", "0.05")),
                    "Saved operating-point trajectory missing")
            require(cp["checkpoint_file"] == f"checkpoints/checkpoint_{index:03d}.pt", "Unexpected checkpoint path")
            sidecar_path = raw_path.parent / Path(cp["checkpoint_file"]).with_suffix(".json")
            sidecar, sidecar_hash = read(sidecar_path)
            require(sidecar == cp, "Sidecar differs from raw checkpoint")
            exported, _ = derived_copy(sidecar, sidecar_hash, sidecar_path, raw_path, raw_hash, label)
            prepared.append((destination / Path(cp["checkpoint_file"]).with_suffix(".json"), exported))
    for path, value in prepared:
        require(not list(private_fields(value)), "Private path in publication")
        require(not path.exists() or read(path)[0] == value, "Different publication already exists")
    count = sum(write_derived(p, value) for p, value in prepared)
    print(json.dumps({"dataset": args.dataset, "completed_cells": 5, "json_files": len(prepared),
                      "written": count, "training_inference_statistics_recomputed": False}))


if __name__ == "__main__":
    main()
