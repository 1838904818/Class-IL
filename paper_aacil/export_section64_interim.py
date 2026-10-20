"""Publish one complete local five-method decision set; never run final tests.

Only the Section64 and explicit Section66 local assignments are supported. HPC collection and the
1-percent external-baseline evaluation are separate, unfinished work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_results import ROOT, derived_copy, private_fields, read, write_derived
from export_section46_development import PACKAGE_SHA, WRAPPER_COMMIT, read_shared, require

DATASET = "CSE-CIC-IDS2018"
SLUG = "cse_cic_ids2018"
ORDER = ("aacil_nogkd_wa", "replay_bs", "aacil", "freq", "wacb",
         "aacil_kd_wa", "replay", "derpp")
DECISION = ORDER[:5]
LAUNCHER = "a900f810e4723125c60b882234a8cd87bb317474"
LAUNCHER_SHA = "3c44975f5a203838d0d876fb90ca007e1122310cec782151b6feea72db772055"
POLICY = "section64_interim_five_per_seed_final_25_decision_full_40"
SECTION66 = {
    "commit": "d7056fe1b405412d477761fe40f3afdfc603d604",
    "sha256": "d624528a15d77c1cf4ad44643f78666f2aa5df9da1c5f0eb1371b43b73c5061d",
}


def canonical_hash(value):
    """Match the frozen producer without importing any training runtime."""
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def complete_local_set(root, seed, section66=False):
    """Read structural status only; reject partial sets before opening results."""
    paths = sorted((root / "attempts").glob("*/status.json"))
    require(len(paths) == 1, "Multiple attempts need explicit cost reconciliation")
    job, digest = read_shared(paths[0])
    launcher = SECTION66["commit"] if section66 else LAUNCHER
    launcher_sha = SECTION66["sha256"] if section66 else LAUNCHER_SHA
    order = DECISION if section66 else ORDER
    require(job["launcher_commit"] == launcher
            and job["launcher_sha256"] == launcher_sha
            and job["package_sha256"] == PACKAGE_SHA
            and job["wrapper_commit"] == WRAPPER_COMMIT
            and job["smoke"] is False and job["resume"] is False
            and job["status"] in ("running", "complete"), "Assignment mismatch")
    planned = job["planned_cells"]
    seeds = list(dict.fromkeys(c[2] for c in planned))
    require(seeds in (([6],) if section66 else ([5], [6], [5, 6]))
            and (not section66 or seed == 6)
            and planned == [[DATASET, m, s] for s in seeds for m in order],
            "Unexpected serial assignment")
    cells = job["cells"]
    require([[c["dataset"], c["method"], c["seed"]] for c in cells]
            == planned[:len(cells)] and len(cells) <= len(planned)
            and all(c["exit_code"] == 0 and c.get("ended_at")
                    and c["elapsed_seconds"] >= 0 for c in cells),
            "Successful completed prefix required")
    selected = [c for c in cells if c["seed"] == seed and c["method"] in DECISION]
    require([c["method"] for c in selected] == list(DECISION),
            "Five decision cells are not complete; no model results opened")
    current = job.get("current_cell")
    require(not current or current["seed"] != seed or current["method"] not in DECISION,
            "Decision cell still active")
    return job, digest, paths[0], selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(5, 6), required=True)
    parser.add_argument("--section66", action="store_true",
                        help="Bind the published seed6 five-decision assignment")
    args = parser.parse_args()
    root = args.source_root.resolve()
    destination = ROOT / "ncd/results/problem_scout/section64/interim_local_v1" / f"seed_{args.seed}"
    require(not destination.exists(), "Already exported; preserve original snapshot")
    job, snapshot, status_path, cells = complete_local_set(root, args.seed, args.section66)
    section = "section66" if args.section66 else "section64"
    launcher = SECTION66["commit"] if args.section66 else LAUNCHER
    launcher_sha = SECTION66["sha256"] if args.section66 else LAUNCHER_SHA
    invocation_name = section + "_local_invocation.json"
    require(hashlib.sha256(args.package.read_bytes()).hexdigest() == PACKAGE_SHA,
            "Original frozen package required")
    with tarfile.open(args.package, "r") as archive:
        members = archive.getmembers()
        require(len(members) == 48 and all(m.isfile() for m in members), "Package inventory mismatch")
        hashes = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in members}
    prepared, rows, common = [], [], None
    label = lambda p: section + "_local/" + p.resolve().relative_to(root).as_posix()
    for cell in cells:
        method = cell["method"]
        folder = root / SLUG / "cells" / method / f"seed_{args.seed}"
        invocation, _ = read(folder / invocation_name)
        require(invocation["launcher_commit"] == launcher
                and invocation["launcher_sha256"] == launcher_sha
                and invocation["package_sha256"] == PACKAGE_SHA
                and invocation["wrapper_commit"] == WRAPPER_COMMIT
                and invocation["actual_entrypoint"] == f"ncd/run_{section}_local.py"
                and (invocation["dataset"], invocation["method"], invocation["seed"])
                == (DATASET, method, args.seed)
                and invocation["smoke"] is False and invocation["selection_changed"] is False,
                "Cell invocation mismatch")
        raw_path = folder / "raw/result.json"
        raw, raw_hash = read(raw_path)
        wrapper_path = folder / "section44_result.json"
        wrapper, wrapper_hash = read(wrapper_path)
        ident = wrapper["identity"]
        require(wrapper["completed"] is True and wrapper["smoke"] is False
                and wrapper["section44"] is True and raw["completed"] is True
                and raw["identity"] == ident and wrapper["raw_result_sha256"] == raw_hash
                and (wrapper["dataset"], wrapper["method"], wrapper["seed"])
                == (DATASET, method, args.seed)
                and (ident["dataset"], ident["method"], ident["seed"])
                == (DATASET, method, args.seed), "Completion/identity mismatch")
        require(ident["git_commit"] == WRAPPER_COMMIT
                and ident["source_hashes"]
                and all(hashes.get(p) == h for p, h in ident["source_hashes"].items()),
                "Frozen source mismatch")
        cfg = ident["effective_config"]
        require(ident["effective_config_sha256"] == canonical_hash(cfg)
                and ident["config_sha256"]
                == hashes["ncd/configs/problem_scout_section21.json"],
                "Configuration hash mismatch")
        fixed = {"section64": True, "smoke": False, "device": "cuda", "cpu_threads": 2,
                 "min_steps_per_task": 2000, "release_policy": POLICY,
                 "local_launcher_commit": launcher, "local_launcher_sha256": launcher_sha,
                 "original_selection_changed": False, "tuning_permitted": False}
        require(all(cfg.get(k) == v for k, v in fixed.items()), "Fixed execution mismatch")
        if args.section66:
            require(cfg.get("section66") is True
                    and cfg.get("assignment") == "seed6_decision_five",
                    "Section66 assignment configuration mismatch")
        for split, count in (("train", 12882081), ("test", 3255102)):
            require(ident["data_binding"]["x_" + split]["shape"] == [count, 78]
                    and ident["data_binding"]["y_" + split]["shape"] == [count],
                    "Full CSE18 data required")
        shared = {k: ident[k] for k in ("git_commit", "source_hashes", "config_sha256",
                  "data_binding", "data_metadata_sha256", "class_order", "tasks",
                  "original_train_counts", "section26_holdout")}
        require(common is None or shared == common, "Matched data/task/holdout mismatch")
        common = shared
        require(len(raw["checkpoints"]) == len(wrapper["stages"]) == len(ident["tasks"]) == 4,
                "Four completed stages required")
        sources = [raw_path, wrapper_path, folder / invocation_name]
        for index, cp in enumerate(raw["checkpoints"]):
            require(cp["task_index"] == index
                    and cp["checkpoint_file"] == f"checkpoints/checkpoint_{index:03d}.pt",
                    "Checkpoint sequence mismatch")
            sidepath = raw_path.parent / Path(cp["checkpoint_file"]).with_suffix(".json")
            sidecar, _ = read(sidepath)
            require(sidecar == cp, "Checkpoint sidecar mismatch")
            sources.append(sidepath)
            stage_path = folder / "calibrated_stages" / f"task_{index:03d}.json"
            stage, _ = read(stage_path)
            require(stage == wrapper["stages"][index] and stage["completed"] is True
                    and stage["binding"] == {
                        "checkpoint_sha256": cp["checkpoint_sha256"],
                        "identity_sha256": canonical_hash(ident), "alpha": .02},
                    "Saved stage binding mismatch")
            sources.append(stage_path)
        for source in sources:
            value, digest = read(source)
            value, _ = derived_copy(value, digest, source, raw_path, raw_hash, label)
            prepared.append((destination / method / source.relative_to(folder), value))
        rows.append({"method": method, "seed": args.seed, "alpha": .02,
                     "saved_metrics": wrapper["metrics"], "wrapper_sha256": wrapper_hash,
                     "cell_attempt": {k: v for k, v in cell.items() if k != "command"}})
    result = {
        "label": "interim, descriptive", "seed": args.seed, "physical_cells": 5,
        "rows": rows, "assignment_status_snapshot_sha256": snapshot,
        "assignment_attempt": status_path.parent.name, "assignment_status": job["status"],
        "assignment_started_at": job["started_at"], "assignment_ended_at": job.get("ended_at"),
        "alpha_0p01_status": "pending separate saved-point extraction or bound baseline evaluation",
        "final_tests_executed": False, "final_judgment_requires_all_25": True,
        "original_40_still_required": True, "early_stop": False, "selection_changed": False,
        "post_unsealing_design": True, "independent_dataset_confirmation": False,
        "training_or_inference_executed": False, "metrics_recomputed": False,
        "historical_failures_unchanged": True,
        "cost_scope": "Parent, raw and wrapper costs are nested, not additive.",
    }
    prepared.append((destination / "result.json", result))
    require(all(not list(private_fields(v)) for _, v in prepared), "Private metadata")
    # All five cells and all publication objects are validated before any write.
    for path, value in prepared:
        write_derived(path, value)
    print(json.dumps({"seed": args.seed, "files": len(prepared),
                      "label": "interim, descriptive", "final_tests_executed": False}))


if __name__ == "__main__":
    main()
