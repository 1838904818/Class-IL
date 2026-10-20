"""One fixed noGKD+WA development dataset per Slurm job, five serial seeds.

This adapter implements the Section46 priority without changing the already
submitted Section44 CSE18 packages or reading their sealed model measurements.
"""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ncd/hpc"))
import section44_job as wrapper

ref = wrapper.ref
ENTRY = "ncd/hpc/section46_development_job.py"
METHOD = "aacil_nogkd_wa"
SEEDS = (1, 2, 3, 4, 42)
DATASETS = ("NSL-KDD", "CIC-IDS-2017")
BASE_SOURCE = wrapper.source_binding
BASE_CONFIG = wrapper.configuration
BASE_CACHE = ref.check_cache


def source_binding():
    hashes = BASE_SOURCE()
    for path in (ENTRY, "ncd/hpc/run_section46_development.sbatch", "ncd/hpc/section44_job.py"):
        hashes[path] = ref.sha(ROOT / path)
    return hashes


def configuration(settings, method, smoke):
    config = BASE_CONFIG(settings, method, smoke)
    config.update(descriptive_development_dataset=True,
                  release_policy="complete_five_seed_development_group_only",
                  section46_priority=True, smoke_one_task_only=smoke)
    return config


def check_cache(root, dataset):
    if dataset != "CIC-IDS-2017":
        return BASE_CACHE(root, dataset)
    folder = root / "cic-ids-2017"
    receipt = folder / "data_quality.json"
    info = ref.read(receipt)
    counts = sum(info["train_counts"].values()), sum(info["id_counts"].values())
    files = info.get("cache_files", [])
    required = {f"{split}_{field}.npy" for split in ("train", "test", "shift") for field in ("x", "y")}
    if (info.get("status") != "complete" or info.get("dataset") != dataset
            or counts != ref.FULL_COUNTS["cic17"] or len(files) != 6
            or {r["path"] for r in files} != required):
        raise ValueError("Original complete CIC17 cache required")
    for row in files:
        if (folder / row["path"]).stat().st_size != row["bytes"]:
            raise ValueError("CIC17 cached array size differs")
    return {"dataset": dataset, "receipt_sha256": ref.sha(receipt),
            "train_rows": counts[0], "test_rows": counts[1],
            "shift_role": "Legacy cache validation only; no shift training, inference or metrics"}


def release_if_complete(output, smoke, methods):
    seeds = (1,) if smoke else SEEDS
    paths = [wrapper.completed_path(output, METHOD, seed) for seed in seeds]
    if not all(p.exists() for p in paths):
        return False
    bindings = [{"path": p.relative_to(output).as_posix(), "sha256": ref.sha(p)} for p in paths]
    target = output / "development_summary.json"
    if target.exists():
        previous = ref.read(target)
        if not previous.get("completed") or previous["source_results"] != bindings:
            raise ValueError("Development completion binding changed")
        return True
    rows = [ref.read(p) for p in paths]
    for seed, path, row in zip(seeds, paths, rows):
        if (not row["completed"] or row["smoke"] != smoke or row["seed"] != seed
                or row["method"] != METHOD or row["dataset"] != ("NSL-KDD" if smoke else ref.DATASET)
                or row["identity"]["effective_config"]["lambda_g"] != 0.
                or ref.sha(path.parent / row["raw_result"]) != row["raw_result_sha256"]):
            raise ValueError("Development cell identity differs")
    ref.write_json(target, {"completed": True, "smoke": smoke, "non_scientific": smoke,
        "dataset": rows[0]["dataset"], "method": METHOD, "seeds": list(seeds),
        "physical_cells": len(rows), "source_results": bindings,
        "aggregate_metrics_generated": False, "descriptive_development_dataset": True,
        "selection_changed": False, "CSE18_model_metrics_accessed": False})
    print(f"DEVELOPMENT_GROUP_COMPLETE cells={len(rows)} smoke={smoke}", flush=True)
    return True


def protect_owned(output, protected, attempt, methods, seeds, copied):
    wrapper.ORIGINAL_PROTECT(output, protected, attempt, methods, seeds, copied)
    summary = output / "development_summary.json"
    if summary.exists():
        target = protected / summary.name
        temporary = target.with_suffix(f".json.{os.getpid()}.tmp")
        shutil.copyfile(summary, temporary)
        temporary.replace(target)


def main():
    dataset = os.environ.get("NCD_DEVELOPMENT_DATASET", "")
    if dataset not in DATASETS:
        raise ValueError("Only explicitly disjoint NSL/CIC17 development assignment")
    smoke = "--local-smoke" in sys.argv
    if smoke:
        if "--smoke-dataset" not in sys.argv or sys.argv[sys.argv.index("--smoke-dataset") + 1] != "NSL-KDD":
            raise ValueError("One-task NSL CPU smoke only")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        from ncd import run_problem_scout as ps
        subset = ps.smoke_subset

        def one_task(data, x_shift, y_shift):
            small, x, y = subset(data, x_shift, y_shift)
            return replace(small, tasks=small.tasks[:1]), x, y

        ps.smoke_subset = one_task
        from ncd import run_aacil_compact as compact

        def one_task_summary(points, tasks):
            if len(points) != 1 or len(tasks) != 1:
                raise ValueError("Section46 smoke must remain exactly one task")
            return {"final_balanced_accuracy": points[0]["balanced_accuracy"],
                    "mean_incremental_balanced_accuracy": points[0]["balanced_accuracy"],
                    "forgetting": None, "old_class_ids": [], "per_class_forgetting": [],
                    "stage_balanced_accuracy": [points[0]["balanced_accuracy"]],
                    "definition": "NONSCIENTIFIC one-task smoke: no previous task, forgetting undefined"}

        # Original multi-task training and summaries are untouched in formal runs.
        compact.stagewise_metrics = one_task_summary
    wrapper.ENTRY = ENTRY
    wrapper.SEEDS = SEEDS
    wrapper.DISCLOSURE = "Section46 fixed noGKD+WA descriptive development; no selection, no CSE18 model access"
    wrapper.source_binding, wrapper.configuration = source_binding, configuration
    ref.ENTRY, ref.METHODS, ref.SEEDS, ref.DATASET = ENTRY, (METHOD,), SEEDS, dataset
    ref.DATASET_KEYS[dataset] = "nsl" if dataset == "NSL-KDD" else "cic17"
    ref.source_binding, ref.configuration, ref.check_cache = source_binding, configuration, check_cache
    ref.run_child, ref.completed_path = wrapper.run_child, wrapper.completed_path
    ref.release_if_complete, ref.protect_owned = release_if_complete, protect_owned
    source_binding()
    ref.main()


if __name__ == "__main__":
    main()
