"""Export one completed Section55 old-seed exploratory cell, without inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_results import ROOT, derived_copy, private_fields, read, sanitize, write_derived
from export_section46_development import (
    COMPACT_SHA, METHOD, PACKAGE_SHA, SEEDS, WRAPPER_COMMIT, read_shared, require,
)

LAUNCHER = "a0744782a8ba1b282314363c9f6064330f2ea7ad"
LAUNCHER_SHA = "edade8e1300827e9ccd454418b1c372ba1eb76dc15dc6ae25c8d04ee63324dff"
DATASET = "CSE-CIC-IDS2018"
SLUG = "cse_cic_ids2018"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve()
    attempts = sorted((root / "attempts").glob("*/status.json"))
    require(len(attempts) == 1, "Multiple attempts require explicit cost reconciliation")
    job, snapshot_hash = read_shared(attempts[0])
    planned = [[DATASET, seed] for seed in SEEDS]
    require(job["planned_cells"] == planned and job["smoke"] is False
            and job["resume"] is False and job["status"] in ("running", "complete"),
            "First formal Section55 assignment required")
    require(job["launcher_commit"] == LAUNCHER and job["launcher_sha256"] == LAUNCHER_SHA
            and job["package_sha256"] == PACKAGE_SHA and job["wrapper_commit"] == WRAPPER_COMMIT,
            "Section55 assignment source mismatch")
    cells = job["cells"]
    require([[c["dataset"], c["seed"]] for c in cells] == planned[:len(cells)]
            and len(cells) <= len(planned)
            and all(c["method"] == METHOD and c["exit_code"] == 0 for c in cells),
            "Successful serial prefix required")
    selected = [c for c in cells if c["seed"] == args.seed]
    require(len(selected) == 1 and selected[0].get("ended_at")
            and (job.get("current_cell") is None or job["current_cell"]["seed"] != args.seed),
            "Requested cell is not complete")
    folder = root / SLUG / "cells" / METHOD / f"seed_{args.seed}"
    invocation, invocation_hash = read(folder / "section55_local_invocation.json")
    require(invocation["actual_entrypoint"] == "ncd/run_section55_local.py"
            and invocation["launcher_commit"] == LAUNCHER
            and invocation["launcher_sha256"] == LAUNCHER_SHA
            and invocation["package_sha256"] == PACKAGE_SHA
            and invocation["wrapper_commit"] == WRAPPER_COMMIT
            and (invocation["dataset"], invocation["method"], invocation["seed"])
            == (DATASET, METHOD, args.seed)
            and invocation["smoke"] is False and invocation["selection_changed"] is False
            and invocation["CSE18_released_seed_exploration"] is True
            and invocation["sealed_seeds_5_to_9_accessed"] is False,
            "Exploratory old-seed invocation required")
    raw_path = folder / "raw/result.json"
    raw, raw_hash = read(raw_path)
    wrapper_path = folder / "section44_result.json"
    wrapper, wrapper_hash = read(wrapper_path)
    ident = raw["identity"]
    cfg = ident["effective_config"]
    require(raw["completed"] is True and raw["selection_only"] is False
            and wrapper["completed"] is True and wrapper["smoke"] is False
            and wrapper["raw_result_sha256"] == raw_hash and wrapper["identity"] == ident,
            "Raw/wrapper completion binding mismatch")
    require((ident["dataset"], ident["method"], ident["seed"]) == (DATASET, METHOD, args.seed)
            and ident["git_commit"] == WRAPPER_COMMIT
            and ident["source_hashes"]["ncd/run_aacil_compact.py"] == COMPACT_SHA,
            "Training identity mismatch")
    fixed = {"lambda_g": 0., "derpp_alpha": 1., "derpp_beta": 1., "hidden": 128,
             "batch_size": 256, "epochs_per_task": 8, "learning_rate": .001,
             "buffer_per_class": 200, "min_steps_per_task": 2000,
             "normal_training_rows": 50000, "cpu_threads": 2, "device": "cuda",
             "section55": True, "exploratory": True, "previously_released_seeds": True,
             "release_policy": "section55_exploratory_old_seed_cells",
             "local_launcher_commit": LAUNCHER, "local_launcher_sha256": LAUNCHER_SHA,
             "smoke": False, "selection_only": False, "post_unsealing_design": True,
             "independent_dataset_confirmation": False}
    require(all(cfg.get(k) == v for k, v in fixed.items()), "Fixed exploratory config mismatch")
    for split, count in (("train", 12882081), ("test", 3255102)):
        require(ident["data_binding"]["x_" + split]["shape"] == [count, 78]
                and ident["data_binding"]["y_" + split]["shape"] == [count],
                "Full CSE18 data required")
    require(raw["tasks"] == ident["tasks"] and len(raw["checkpoints"]) == len(raw["tasks"]) == 4,
            "Complete four-stage trajectory required")
    label = lambda p: "section55_old_seed_exploration/" + p.resolve().relative_to(root).as_posix()
    execution = {
        "assignment_status_snapshot_sha256": snapshot_hash,
        "assignment_attempt": attempts[0].parent.name,
        "assignment_started_at": job["started_at"],
        "assignment_ended_at": job.get("ended_at"),
        "assignment_status_at_export": job["status"],
        "cell_attempt": {k: v for k, v in selected[0].items() if k != "command"},
        "local_invocation": sanitize(invocation), "local_invocation_sha256": invocation_hash,
        "wrapper_record": label(wrapper_path), "wrapper_sha256": wrapper_hash,
        "wrapper_actual_entrypoint": wrapper["actual_entrypoint"],
        "wrapper_command": sanitize({"command": wrapper["command"]})["command"],
        "wrapper_cost_seconds": wrapper["cost_seconds"],
        "adapter_cost_seconds": wrapper["adapter_cost_seconds"],
        "exploratory": True, "previously_released_seeds": True,
        "post_unsealing_design": True, "independent_dataset_confirmation": False,
        "selection_changed": False, "sealed_seeds_5_to_9_accessed": False,
        "complete_cell_only": True, "five_seed_aggregate": False,
        "cost_scope": "Raw, wrapper and parent times are nested, not additive; local GPU utilization includes desktop activity.",
    }
    destination = ROOT / "ncd/results/problem_scout/section55/exploratory_old_seeds_v1" / SLUG / f"{METHOD}_seed{args.seed}"
    exported, _ = derived_copy(raw, raw_hash, raw_path, raw_path, raw_hash, label)
    exported["section55_execution"] = execution
    prepared = [(destination / "result.json", exported)]
    for index, cp in enumerate(raw["checkpoints"]):
        require(cp["task_index"] == index and cp["primary_alpha"] == .02
                and all("confusion_matrix" in cp["operating_points"][a] for a in ("0.01", "0.02", "0.05"))
                and cp["checkpoint_file"] == f"checkpoints/checkpoint_{index:03d}.pt",
                "Saved operating-point trajectory mismatch")
        path = raw_path.parent / Path(cp["checkpoint_file"]).with_suffix(".json")
        sidecar, digest = read(path)
        require(sidecar == cp, "Sidecar mismatch")
        exported, _ = derived_copy(sidecar, digest, path, raw_path, raw_hash, label)
        prepared.append((destination / Path(cp["checkpoint_file"]).with_suffix(".json"), exported))
    require(not destination.exists(), "Cell already exported; never replace its original status snapshot")
    require(all(not list(private_fields(value)) for _, value in prepared), "Private publication metadata")
    for path, value in prepared:
        write_derived(path, value)
    print(json.dumps({"seed": args.seed, "json_files": len(prepared), "exploratory": True,
                      "training_inference_statistics_recomputed": False}))


if __name__ == "__main__":
    main()
