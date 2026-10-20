"""Last post-unsealing CSE18 round: eight methods, five new seeds, one release.

This is a new wrapper, not a change to the original reference or selected-main
packages. It reuses their training, checkpoint, calibration and resource guard
implementations. New seeds on the same exposed dataset are not an independent
dataset confirmation. No tuning, scheduler submission or automatic retry occurs.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import section31_job as ref
import section32_job as selected

ENTRY = "ncd/hpc/section44_job.py"
METHODS = ("aacil_nogkd_wa", "aacil_kd_wa", "aacil", "replay_bs",
           "replay", "derpp", "wacb", "freq")
SEEDS = (5, 6, 7, 8, 9)
COMPACT = METHODS[:2]
CANDIDATE_SOURCE = "a0859e2d7f515c4dfaa96d25fea6cd5f3961d758"
COMPACT_SHA = "36fe10ca4cac4a5b4fe54152cebe3d20419f7b4fcfc3f7a1443c0db07c3deb04"
DISCLOSURE = (
    "Section44 last round, designed after original CSE18 unsealing; same dataset, "
    "new seeds, not independent dataset confirmation. All 40 physical cells "
    "must finish before aggregate release. No tuning or original-selection change."
)
ORIGINAL_SOURCE_BINDING = ref.source_binding
ORIGINAL_CONFIGURATION = ref.configuration
ORIGINAL_CHILD = ref.run_child
ORIGINAL_PROTECT = ref.protect_owned


def source_binding():
    hashes = ORIGINAL_SOURCE_BINDING()
    expected = selected.FROZEN_SOURCES | {"ncd/run_aacil_compact.py": COMPACT_SHA}
    for path, digest in expected.items():
        if ref.sha(ROOT / path) != digest:
            raise ValueError(f"Fixed training source changed: {path}")
        hashes[path] = digest
    for path in (ENTRY, "ncd/hpc/run_section44.sbatch", "ncd/hpc/section31_job.py",
                 "ncd/hpc/section32_job.py"):
        hashes[path] = ref.sha(ROOT / path)
    if ref.sha(ROOT / "ncd/configs/problem_scout_section21.json") != selected.CONFIG_SHA:
        raise ValueError("Fixed base configuration changed")
    return hashes


def configuration(settings, method, smoke):
    config = ORIGINAL_CONFIGURATION(settings, method, smoke)
    config.update(section44=True, section44_schema_version=1,
                  post_unsealing_design=True, independent_dataset_confirmation=False,
                  original_selection_changed=False, tuning_permitted=False,
                  release_policy="all_40_physical_cells_before_model_aggregate")
    if method in COMPACT:
        from ncd import run_aacil_compact as compact
        config.update(lambda_g=0. if method == COMPACT[0] else .5,
            derpp_alpha=1., derpp_beta=1., derpp_allocation_adaptation=compact.ADAPTATION,
            derpp_coefficient_source="Section44 fixed alpha=beta=1; no new tuning",
            ema_decay=.999, ema_consistency=1., cbf_epochs=1, cbf_lr_multiplier=.1,
            family_parameter_constraint="constant through all stages; train and inference <=2x full eventual original attack-head F",
            future_output_policy="unseen attack outputs masked; all parameter rows reserved at initialization",
            deadline_iso=None, selection_only=False,
            selection_aggregation="not applicable; Section44 fixed post-unsealing design",
            selection_validation_policy="permanent Normal holdout plus lawful retained attack validation memory; evaluation only")
    return config


def completed_path(output, method, seed):
    return ref.cell_root(output, method, seed) / "section44_result.json"


def run_child(args):
    method, seed = args.methods[0], args.seeds[0]
    if method not in METHODS or seed not in ((1,) if args.local_smoke else SEEDS):
        raise ValueError("Unregistered Section44 cell")
    if method not in COMPACT:
        ORIGINAL_CHILD(args)
        folder = ref.cell_root(args.output_root, method, seed)
        result = ref.read(folder / "section31_result.json")
        result.update(actual_entrypoint=ENTRY, disclosure=DISCLOSURE,
                      section44=True, post_unsealing_design=True,
                      independent_dataset_confirmation=False)
        target = completed_path(args.output_root, method, seed)
        if target.exists() and ref.read(target) != result:
            raise ValueError("Existing Section44 reference wrapper changed")
        if not target.exists():
            ref.write_json(target, result)
        return

    import numpy as np
    import sklearn
    import torch
    from ncd import run_aacil as aa, run_aacil_compact as compact, run_problem_scout as ps
    from ncd.prepare_problem_scout_data import prepare_setting
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    dataset = args.smoke_dataset if args.local_smoke else ref.DATASET
    folder = ref.cell_root(args.output_root, method, seed)
    folder.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "ncd/configs/problem_scout_section21.json"
    config = configuration(ref.read(config_path), method, args.local_smoke)
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Formal child requires CUDA")
    hashes = source_binding()
    started = time.monotonic()
    data, _, _, metadata = prepare_setting(dataset, config)
    original_counts = np.bincount(data.y_train, minlength=len(data.class_order))
    if args.local_smoke:
        data, _, _ = ps.smoke_subset(data, None, None)
    loading_seconds = time.monotonic() - started
    train, valid = ps.section26_pools(data, seed)
    identity = {
        "git_commit": os.environ["NCD_SOURCE_COMMIT"], "source_hashes": hashes,
        "adapter_entrypoint": ENTRY, "candidate_implementation_commit": CANDIDATE_SOURCE,
        "effective_config": config, "config_sha256": ref.sha(config_path),
        "effective_config_sha256": ps.canonical_hash(config),
        "dataset": dataset, "method": method, "seed": seed,
        "class_order": data.class_order, "tasks": data.tasks, "feature_names": data.feature_names,
        "data_binding": ps.binding(data, None, None), "data_metadata_sha256": ps.canonical_hash(metadata),
        "original_train_counts": original_counts.tolist(),
        "calibration_prior_role": "Original complete prepared TRAIN labels before holdout or smoke subset; no TEST role",
        "section27_access_policy": aa.POLICY, "section31_method": method,
        "section31_schema_version": 1, "effective_lambda_g": config["lambda_g"],
        "family_type": "aacil_kd_wa", "post_unsealing_design": True,
        "independent_dataset_confirmation": False,
        "section26_holdout": {str(c): {"train_count": len(train[c]), "validation_count": len(valid[c]),
            "validation_indices_sha256": hashlib.sha256(valid[c].tobytes()).hexdigest()} for c in valid},
        "runtime_versions": {"python": sys.version.split()[0], "torch": str(torch.__version__),
            "numpy": np.__version__, "sklearn": sklearn.__version__, "cuda": torch.version.cuda}}
    raw = folder / "raw"
    with redirect_stdout(ref.ProgressOnly(sys.stdout)):
        result = compact.run_cell(data, config, identity, raw, device)
    target = completed_path(args.output_root, method, seed)
    if target.exists():
        old = ref.read(target)
        if (not old.get("completed") or old.get("identity") != identity
                or old.get("raw_result_sha256") != ref.sha(raw / "result.json")):
            raise ValueError("Completed Section44 compact binding changed")
        for cp in result["checkpoints"]:
            path = raw / cp["checkpoint_file"]
            if ref.sha(path) != cp["checkpoint_sha256"] or ref.read(path.with_suffix(".json")) != cp:
                raise ValueError("Completed compact checkpoint changed")
        print(f"CELL_SKIP method={method} seed={seed} metrics=sealed", flush=True)
        return
    if not all("0.02" in cp.get("operating_points", {}) for cp in result["checkpoints"]):
        raise ValueError("Compact saved alpha2 matrices required, no fallback inference")
    stages, metrics, calibration_cost = ref.calibrated_stages(
        data, result, raw, config, device, original_counts)
    ref.write_json(target, {
        "completed": True, "completed_at": ref.utc(), "schema_version": 1,
        "dataset": dataset, "method": method, "seed": seed, "smoke": args.local_smoke,
        "identity": identity, "raw_result": "raw/result.json",
        "raw_result_sha256": ref.sha(raw / "result.json"),
        "actual_entrypoint": ENTRY, "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
        "metrics": metrics, "stages": stages, "architecture_capacity": result["architecture_capacity"],
        "cost_seconds": {k: v for k, v in result.items() if k.endswith("_seconds_total")},
        "adapter_cost_seconds": {"data_validation_loading_this_attempt": loading_seconds,
            "additional_calibration_evaluation_successful_stages": calibration_cost},
        "checkpoint_resume": "Identity-bound completed checkpoints reused; failed and unsaved task costs retained in attempts",
        "disclosure": DISCLOSURE, "section44": True, "post_unsealing_design": True,
        "independent_dataset_confirmation": False, "owner_deadline": None})
    print(f"CELL_COMPLETE method={method} seed={seed} checkpoints={len(stages)} metrics=sealed", flush=True)


def release_if_complete(output, smoke, methods):
    # Never use the current job's subset as the release boundary.
    seeds = (1,) if smoke else SEEDS
    paths = [completed_path(output, m, s) for m in METHODS for s in seeds]
    if not all(p.exists() for p in paths):
        return False
    target = output / "section44_summary.json"
    bindings = [{"path": p.relative_to(output).as_posix(), "sha256": ref.sha(p)} for p in paths]
    if target.exists():
        prior = ref.read(target)
        if not prior.get("completed") or prior.get("smoke") != smoke or prior.get("source_results") != bindings:
            raise ValueError("Released Section44 source changed")
        return True
    lock = output / ".section44_aggregate.lock"
    try:
        stream = lock.open("x", encoding="utf-8")
    except FileExistsError:
        return False
    try:
        with stream:
            json.dump({"pid": os.getpid(), "utc": ref.utc()}, stream)
        rows = [ref.read(p) for p in paths]
        common = rows[0]["identity"]
        common_keys = ("git_commit", "source_hashes", "config_sha256", "data_binding",
                       "data_metadata_sha256", "class_order", "tasks", "original_train_counts")
        # Complete identity/source validation precedes any aggregate calculation.
        for row, path in zip(rows, paths):
            if (row.get("completed") is not True or row.get("smoke") != smoke
                    or path != completed_path(output, row["method"], row["seed"])
                    or row["dataset"] != ("NSL-KDD" if smoke else ref.DATASET)
                    or not row.get("section44")):
                raise ValueError("Incomplete or incompatible Section44 cell")
            if any(row["identity"].get(k) != common.get(k) for k in common_keys):
                raise ValueError("Section44 mixes code, data or task identity")
            if ref.sha(path.parent / row["raw_result"]) != row["raw_result_sha256"]:
                raise ValueError("Section44 raw result changed")
            config = row["identity"]["effective_config"]
            if config != configuration(ref.read(ROOT / "ncd/configs/problem_scout_section21.json"), row["method"], smoke):
                raise ValueError("Section44 fixed effective configuration changed")
            same_seed = next(r for r in rows if r["seed"] == row["seed"])
            if row["identity"]["section26_holdout"] != same_seed["identity"]["section26_holdout"]:
                raise ValueError("Matched seed holdout differs across methods")
        keys = ("average_incremental_accuracy", "final_balanced_accuracy", "forgetting_all_old_classes",
                "final_attack_detection", "final_macro_f1", "final_normal_fpr")
        groups = []
        for method in METHODS:
            group = [r for r in rows if r["method"] == method]
            if sorted(r["seed"] for r in group) != list(seeds):
                raise ValueError("Section44 group seed mismatch")
            metrics = {}
            for key in keys:
                values = [r["metrics"][key] for r in group]
                defined = all(v is not None for v in values)
                metrics[key] = {"n": len(values), "mean": statistics.mean(values) if defined else None,
                    "sample_sd": statistics.stdev(values) if defined and len(values) > 1 else None}
            groups.append({"method": method, "seeds": list(seeds), "metrics": metrics,
                           "cost_seconds": [r["cost_seconds"] for r in group]})
        replay = next(g for g in groups if g["method"] == "replay")
        groups.append(replay | {"method": "replay_la", "derived_from": "replay",
            "cost_seconds": [], "training_cost_double_charged": False,
            "calibration_definition": "Same Normal-selected total offset; exact alpha2 prediction alias"})
        ref.write_json(target, {
            "completed": True, "released_at": ref.utc(), "smoke": smoke, "non_scientific": smoke,
            "physical_cells": len(rows), "groups": groups, "group_count": len(groups),
            "source_results": bindings, "release_rule": DISCLOSURE,
            "post_unsealing_design": True, "independent_dataset_confirmation": False,
            "test_selected": False, "original_selection_changed": False,
            "four_criteria_tests_executed": False, "last_round_regardless_outcome": True})
        print(f"SECTION44_AGGREGATE_RELEASED physical_cells={len(rows)} smoke={smoke}", flush=True)
        return True
    finally:
        lock.unlink(missing_ok=True)


def protect_owned(output, protected, attempt, methods, seeds, copied):
    ORIGINAL_PROTECT(output, protected, attempt, methods, seeds, copied)
    summary = output / "section44_summary.json"
    if summary.exists():
        target = protected / summary.name
        temporary = target.with_suffix(f".json.{os.getpid()}.tmp")
        shutil.copyfile(summary, temporary)
        temporary.replace(target)


def configure_supervisor():
    # Process-local hooks only. Old frozen files/packages remain untouched.
    ref.ENTRY, ref.METHODS, ref.SEEDS = ENTRY, METHODS, SEEDS
    ref.source_binding = source_binding
    ref.configuration = configuration
    ref.run_child, ref.completed_path = run_child, completed_path
    ref.release_if_complete, ref.protect_owned = release_if_complete, protect_owned


def main():
    if "--local-smoke" in sys.argv:
        if "--smoke-dataset" not in sys.argv or sys.argv[sys.argv.index("--smoke-dataset") + 1] != "NSL-KDD":
            raise ValueError("Section44 same-entry smoke is NSL-KDD only, never CSE18")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    configure_supervisor()
    if "--local-smoke" in sys.argv:
        # The inherited smoke supervisor assigns seed1 to its subprocess.
        ref.SEEDS = (1, *SEEDS)
    source_binding()
    ref.main()


if __name__ == "__main__":
    main()
