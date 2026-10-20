"""Publish only the closed all35 reference and all5 selected CSE18 releases.

No model calls, checkpoint loading, statistics or recursive experiment discovery.
The original protected files remain immutable. Publication copies retain all
scientific values and identities while rebinding sanitized raw-file hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_results import ROOT, derived_copy, private_fields, read, sanitize, write_derived

SEEDS = [1, 2, 3, 4, 42]
METHODS = ["replay", "derpp", "wacb", "freq", "replay_bs", "aacil", "aacil_kd"]
CSE = "CSE-CIC-IDS2018"
SELECTION_SHA = "cd724fbb55c6f00678efc3dcf9d9df2122cd2fccae0290f6c5626b281a4d366e"
REFERENCE = "270121272848087e4af85ec5290eaede0102d776"
TRAINING = "c97f83b1bfa14f4d2665ff33c45c897941e9ddb3"
WRAPPER = "045b86222c24d23389edae8ae75c0dd30fd80a42"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def payload(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def closed_release(root, methods, filename, summary_name):
    paths = [root / "cells" / m / f"seed_{s}" / filename for m in methods for s in SEEDS]
    require(all(p.is_file() for p in paths) and (root / summary_name).is_file(),
            "All physical completion files and original guarded release required")
    summary, summary_sha = read(root / summary_name)
    require(summary.get("completed") is True and summary.get("smoke") is False
            and summary.get("physical_cells") == len(paths)
            and summary.get("test_selected") is False, "Invalid original release")
    require(summary["source_results"] ==
            [{"path": p.relative_to(root).as_posix(), "sha256": digest(p)} for p in paths],
            "Original release source binding mismatch")
    return paths, summary, summary_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--selected-root", required=True, type=Path)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "ncd/results/problem_scout/section31/cse_release_v1")
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT)
    require(not output.exists(), "Publication already exists; do not repeat or overwrite")
    plans = [
        (args.reference_root.resolve(), METHODS, "section31_result.json", "reference_summary.json", "reference"),
        (args.selected_root.resolve(), ["aacil_kd_wa"], "section32_result.json", "selected_main_summary.json", "selected"),
    ]
    # Both releases are verified before opening any individual model record.
    closed = [closed_release(root, methods, filename, summary_name)
              for root, methods, filename, summary_name, role in plans]
    require(closed[0][1].get("primary_selected_candidate_included") is False,
            "Reference release must exclude selected candidate")
    prepared = []
    for (root, methods, filename, summary_name, role), (paths, summary, summary_sha) in zip(plans, closed):
        label = lambda p: role + "/" + p.resolve().relative_to(root).as_posix()
        attempts = {}
        for exit_path in sorted((root / "attempts").glob("*/job_exit.json")):
            job, job_sha = read(exit_path)
            require(job.get("status") == "complete" and job.get("smoke") is False
                    and job.get("resume") is False and job.get("current_cell") is None,
                    "Review failed/resumed/partial attempts before publishing")
            for cell in job["cells"]:
                key = cell["method"], cell["seed"]
                require(key not in attempts and cell["exit_code"] == 0, "Duplicate or failed physical cell")
                attempt = {k: v for k, v in job.items() if k.endswith(("_commit", "_sha256"))}
                attempt.update(attempt_id=exit_path.parent.name, job_exit_sha256=job_sha,
                               started_at=job["started_at"], ended_at=job["ended_at"],
                               job_id=job.get("job_id"), execution_host="HPC A100" if job.get("job_id") else "Local CUDA",
                               cell={k: v for k, v in cell.items() if k != "command"})
                attempt["cell"]["command"] = sanitize({"command": subprocess.list2cmdline(cell["command"])})["command"]
                attempt["cost_scope"] = "Parent, wrapper and raw timings are nested, not additive. Local CUDA and HPC A100 costs are not hardware-matched."
                attempts[key] = attempt
        require(set(attempts) == {(m, s) for m in methods for s in SEEDS}, "Closed physical assignment mismatch")
        bindings = []
        for path in paths:
            wrapper, wrapper_sha = read(path)
            method, seed = wrapper["method"], wrapper["seed"]
            require(wrapper["completed"] is True and wrapper["smoke"] is False
                    and wrapper["dataset"] == CSE
                    and path == root / "cells" / method / f"seed_{seed}" / filename,
                    "Invalid formal cell")
            raw_path = path.parent / wrapper["raw_result"]
            require(raw_path.resolve().is_relative_to(path.parent.resolve()), "Raw path escapes cell")
            raw, raw_sha = read(raw_path)
            require(wrapper["raw_result_sha256"] == raw_sha and raw["completed"] is True
                    and raw["identity"] == wrapper["identity"], "Raw/adapter identity or hash mismatch")
            identity = raw["identity"]
            cfg = identity["effective_config"]
            require((identity["dataset"], identity["method"], identity["seed"]) == (CSE, method, seed)
                    and identity["git_commit"] == (TRAINING if role == "selected" else REFERENCE)
                    and cfg["smoke"] is False and not cfg.get("selection_only"),
                    "Frozen formal training identity mismatch")
            require(identity["data_binding"]["x_train"]["shape"][0] == 12882081
                    and identity["data_binding"]["x_test"]["shape"][0] == 3255102,
                    "Uncapped full CSE18 required")
            if role == "selected":
                require(identity["wrapper_source_commit"] == WRAPPER
                        and identity["frozen_selection"]["selection_sha256"] == SELECTION_SHA
                        and identity["frozen_selection"]["eligible"] is False
                        and cfg["lambda_g"] == .5 and cfg["derpp_alpha"] == cfg["derpp_beta"] == 1,
                        "Frozen selected fallback changed")
            require(len(wrapper["stages"]) == len(raw["checkpoints"]) == len(raw["tasks"]),
                    "Incomplete trajectory")
            dest = output / role / path.relative_to(root)
            exported_raw, _ = derived_copy(raw, raw_sha, raw_path, raw_path, raw_sha, label)
            exported_raw["cse_execution"] = attempts[method, seed]
            prepared.append((dest.parent / wrapper["raw_result"], exported_raw))
            exported, _ = derived_copy(wrapper, wrapper_sha, path, path, wrapper_sha, label)
            exported["original_raw_result_sha256"] = raw_sha
            exported["raw_result_sha256"] = hashlib.sha256(payload(exported_raw)).hexdigest()
            exported["cse_execution"] = attempts[method, seed]
            exported["publication_derivation"]["changes"] += " raw_result_sha256 binds the sanitized copy; original_raw_result_sha256 preserves the original binding."
            prepared.append((dest, exported))
            bindings.append({"path": path.relative_to(root).as_posix(),
                             "sha256": hashlib.sha256(payload(exported)).hexdigest()})
            for cp in raw["checkpoints"]:
                cp_path = Path(cp["checkpoint_file"])
                require(cp_path.as_posix() == f"checkpoints/checkpoint_{cp['task_index']:03d}.pt",
                        "Unexpected checkpoint path")
                sidecar_path = raw_path.parent / cp_path.with_suffix(".json")
                sidecar, sidecar_sha = read(sidecar_path)
                require(sidecar == cp, "Saved sidecar differs from raw checkpoint record")
                copy, _ = derived_copy(sidecar, sidecar_sha, sidecar_path, raw_path, raw_sha, label)
                prepared.append((dest.parent / Path(wrapper["raw_result"]).parent / cp_path.with_suffix(".json"), copy))
        published_summary = sanitize(summary)
        published_summary["source_results"] = bindings
        published_summary["publication_derivation"] = {
            "original_release_sha256": summary_sha, "original_source_results": summary["source_results"],
            "changes": "Only source hashes rebound to path-sanitized publication copies. All released group metrics and costs unchanged.",
            "statistics_recomputed": False,
        }
        prepared.append((output / role / summary_name, published_summary))
    require(len({p for p, _ in prepared}) == len(prepared), "Duplicate output")
    for _, value in prepared:
        require(not list(private_fields(value)), "Private path remains")
    for path, value in prepared:
        write_derived(path, value)
    print(json.dumps({"physical_cells": 40, "small_json_files": len(prepared),
                      "statistics_training_inference_recomputed": False}))


if __name__ == "__main__":
    main()
