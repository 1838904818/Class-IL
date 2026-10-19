"""Guarded frozen KD+WA confirmation; no selection or scheduler submission.

The existing reference supervisor supplies UUID telemetry, serial children,
signals, disjoint cell locks, checkpoint-aware resume and protected small-file
copies. Its source is not modified. Model metrics stay sealed until all five
formal seeds finish; the one-seed CPU smoke is explicitly NONSCIENTIFIC.
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
import section31_job as supervisor

ENTRY = "ncd/hpc/section32_job.py"
METHOD = "aacil_kd_wa"
TRAINING_COMMIT = "c97f83b1bfa14f4d2665ff33c45c897941e9ddb3"
SELECTION_SHA = "cd724fbb55c6f00678efc3dcf9d9df2122cd2fccae0290f6c5626b281a4d366e"
CONFIG_SHA = "fd631316de03fa6723d6a585f23b827c8646a738e1119d9fe4b0caa29e87c3bb"
CANDIDATE = "aacil_kd_wa_lg0p5_a1_b1"
CONFIGURATION = dict(lambda_g=.5, der_alpha=1., der_beta=1., cbf_epochs=1,
                     cbf_lr_multiplier=.1, ema_decay=.999, ema_consistency=1.)
FROZEN_SOURCES = {
    "ncd/aacil_compact.py": "9aaa28d18859ed94cfddb404e62cc76f70ae12ec8d7083e89d83ad1764e0115e",
    "ncd/aacil_iso.py": "ac601910f90cee5dc9da1035231b843b19744eb8bd571ffbc46a184b389fde12",
    "ncd/aacil_retention.py": "a116dfd3ede1f6e3e0c416c14657c1a72e2e91074cf1826321fd8bfd2b9bd61c",
    "ncd/audit_quality_tabular.py": "4907f92e18b499742350a9be7c2dc9d6168fb071b28eaeb9c402d6bdf0aeffff",
    "ncd/crr_benchmarks.py": "8243d8723fa1d4de53296dd08c3a08faf3201502b189a6d9e0879427519ee923",
    "ncd/data.py": "793d577cae42bc6974c274ac8aafae814d5af3cca22a3085c3975d8a07c6e32b",
    "ncd/exp_h_data.py": "571e9335865e89511f55cdb751842de6cd5686349135e0773413757ba0544e6a",
    "ncd/groundtruth.py": "31fa9bba3ca0a9c5fdaac19460ab9bd4d1bd2cc83f984a186c179490b497b1a8",
    "ncd/model.py": "62ba71e74f3ee56d05b71cf67ab13f62d83d0e3940c5954a001013e13693a1f0",
    "ncd/prepare_problem_scout_data.py": "a772fdffb1071eca12cef278a5fdc6a502d5b4f65c243a5fb283ce2eedcbfc98",
    "ncd/run_aacil.py": "8e8264d56149012c454fcc4691c1159ccc3f4d69ffdbb57ecda6aaa77c88f8ee",
    "ncd/run_aacil_compact.py": "193cc12a52eb37a584be513e193ae1df671c85938c51ab538dfc3625c0fd0fff",
    "ncd/run_aacil_retention.py": "7947c110d5dc75dc7dcdba6a0a3d49026ca1595d41332da19191cff5b3cac2b8",
    "ncd/run_crr_real_sequence.py": "61842b40ed9e3662ec290deae0615e3692adae04e71d6e05db3f619d833e01eb",
    "ncd/run_problem_scout.py": "2d7f5cafe12469166424caf3c78a5c6784c6d6be379c93d59510721d04f4f0fa",
    "ncd/source_commit.py": "d2ec0d53353fc097c534a3dc6320ec9c1cf584c27f0b49f04a60301b41f3a010",
    "ncd/train_cil.py": "d2e88bf880897a74f2ac85d9564f638c6139a051b9f0e3f5175b576d805dda28",
    "src/methods/replay_dpmeans.py": "0165a79066cb5c3b0c0ecffa68dfe34b0d3be67026fd68e70713043af1243ecd",
}


def selected_configuration():
    if supervisor.sha(ROOT / "ncd/configs/problem_scout_section21.json") != CONFIG_SHA:
        raise ValueError("Frozen compact base configuration changed")
    path = Path(os.environ.get("NCD_SELECTION_FILE", ""))
    if not path.is_file() or supervisor.sha(path) != SELECTION_SHA:
        raise ValueError("Exact complete frozen development selection is required")
    selection = supervisor.read(path)
    selected = selection.get("selected_main", {})
    if (selection.get("completed") is not True or selection.get("selection_complete") is not True
            or selection.get("source_commit") != TRAINING_COMMIT
            or selection.get("test_metrics_used_for_selection") is not False
            or selection.get("selection_seeds") != [5, 6]
            or selection.get("selection_datasets") != ["UNSW-NB15", "NSL-KDD"]
            or selected.get("method") != METHOD or selected.get("selected_candidate_id") != CANDIDATE
            or selected.get("configuration") != CONFIGURATION):
        raise ValueError("Selected method/configuration or development-only provenance changed")
    for relative, expected in FROZEN_SOURCES.items():
        if supervisor.sha(ROOT / relative) != expected:
            raise ValueError(f"Frozen compact training source changed: {relative}")
    return {"selection_sha256": SELECTION_SHA, "candidate_id": CANDIDATE,
            "selection_completed_at": selection["completed_at"],
            "eligible": selected["eligible"], "status": selection["main_selection_status"],
            "selection_role": "Frozen development fallback; forgetting constraint not satisfied; no CSE18 selection"}


def completed_path(output, method, seed):
    return supervisor.cell_root(output, method, seed) / "section32_result.json"


def run_child(args):
    import numpy as np
    import sklearn
    import torch
    from ncd import run_aacil as aa, run_aacil_compact as compact, run_problem_scout as ps
    from ncd.prepare_problem_scout_data import prepare_setting
    selected = selected_configuration()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    dataset = args.smoke_dataset if args.local_smoke else supervisor.DATASET
    method, seed = args.methods[0], args.seeds[0]
    if method != METHOD or (args.local_smoke and dataset not in ("NSL-KDD", "UNSW-NB15")):
        raise ValueError("Only frozen KD+WA; smoke must not access CSE18")
    folder = supervisor.cell_root(args.output_root, method, seed)
    folder.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "ncd/configs/problem_scout_section21.json"
    config = supervisor.read(config_path)
    config.update(section27=True, section27_schema_version=1, section27_access_policy=aa.POLICY,
        section31=True, section31_schema_version=1, lambda_g=.5, derpp_alpha=1., derpp_beta=1.,
        derpp_allocation_adaptation=compact.ADAPTATION,
        derpp_coefficient_source="Frozen UNSW/NSL validation-only selection; no CSE18 tuning",
        ema_decay=.999, ema_consistency=1., cbf_epochs=1, cbf_lr_multiplier=.1,
        family_parameter_constraint="constant through all stages; train and inference <=2x full eventual original attack-head F",
        future_output_policy="unseen attack outputs masked; all parameter rows reserved at initialization",
        deadline_iso=None, gate_step_policy="matched_s2000",
        calibration_policy="permanent_normal_holdout_outside_m", normal_training_rows=50000,
        min_steps_per_task=0 if args.local_smoke else 2000, epochs_per_task=1 if args.local_smoke else 8,
        smoke=args.local_smoke, selection_only=False, selection_aggregation="not applicable; frozen selected main",
        selection_validation_policy="permanent Normal holdout plus lawful retained attack validation memory; evaluation only",
        cpu_threads=2, cpu_interop_threads=1, device="cpu" if args.local_smoke else "cuda")
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Formal child requires CUDA")
    wrapper_paths = (ENTRY, "ncd/hpc/run_section32.sbatch", "ncd/hpc/section31_job.py",
                     "ncd/hpc/problem_scout_job.py", "ncd/analyze_stagewise_forgetting.py")
    provenance = {"git_commit": TRAINING_COMMIT, "source_hashes": dict(FROZEN_SOURCES),
        "wrapper_source_commit": os.environ["NCD_SOURCE_COMMIT"],
        "wrapper_source_hashes": {p: supervisor.sha(ROOT / p) for p in wrapper_paths},
        "effective_config": config, "config_sha256": supervisor.sha(config_path),
        "effective_config_sha256": ps.canonical_hash(config), "frozen_selection": selected,
        "runtime_versions": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                             "numpy": np.__version__, "sklearn": sklearn.__version__, "cuda": torch.version.cuda}}
    start = time.monotonic()
    data, _, _, metadata = prepare_setting(dataset, config)
    original_train_counts = np.bincount(data.y_train, minlength=len(data.class_order))
    if args.local_smoke:
        data, _, _ = ps.smoke_subset(data, None, None)
    loading_seconds = time.monotonic() - start
    train, valid = ps.section26_pools(data, seed)
    identity = provenance | {"dataset": dataset, "method": method, "seed": seed,
        "class_order": data.class_order, "tasks": data.tasks, "feature_names": data.feature_names,
        "data_binding": ps.binding(data, None, None), "data_metadata_sha256": ps.canonical_hash(metadata),
        "section27_access_policy": aa.POLICY, "section31_method": method, "section31_schema_version": 1,
        "effective_lambda_g": .5, "family_type": method,
        "section26_holdout": {str(c): {"train_count": len(train[c]), "validation_count": len(valid[c]),
            "validation_indices_sha256": hashlib.sha256(valid[c].tobytes()).hexdigest()} for c in valid}}
    raw = folder / "raw"
    with redirect_stdout(supervisor.ProgressOnly(sys.stdout)):
        result = compact.run_cell(data, config, identity, raw, device)
    target = completed_path(args.output_root, method, seed)
    if target.exists():
        old = supervisor.read(target)
        if not old.get("completed") or old.get("identity") != identity or old.get("raw_result_sha256") != supervisor.sha(raw / "result.json"):
            raise ValueError("Completed selected-main cell binding changed")
        print(f"CELL_SKIP method={method} seed={seed} metrics=sealed", flush=True)
        return
    if not all("0.02" in cp.get("operating_points", {}) for cp in result["checkpoints"]):
        raise ValueError("Saved alpha2 confusion matrices required; no fallback inference")
    stages, metrics, calibration_cost = supervisor.calibrated_stages(data, result, raw, config, device, original_train_counts)
    output = {"completed": True, "completed_at": supervisor.utc(), "schema_version": 1,
        "dataset": dataset, "method": method, "seed": seed, "smoke": args.local_smoke,
        "identity": identity, "raw_result": "raw/result.json", "raw_result_sha256": supervisor.sha(raw / "result.json"),
        "actual_entrypoint": ENTRY, "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
        "metrics": metrics, "stages": stages, "architecture_capacity": result["architecture_capacity"],
        "cost_seconds": {k: v for k, v in result.items() if k.endswith("_seconds_total")},
        "adapter_cost_seconds": {"data_validation_loading_this_attempt": loading_seconds,
                                 "additional_calibration_evaluation_successful_stages": calibration_cost},
        "checkpoint_resume": "Frozen compact run_cell reuses identity-bound completed checkpoints; unsaved task work repeats, preserved attempt cost is not discarded",
        "owner_deadline": None, "disclosure": "All five formal seeds required before aggregate release; no CSE18 tuning"}
    supervisor.write_json(target, output)
    print(f"CELL_COMPLETE method={method} seed={seed} checkpoints={len(stages)} metrics=sealed", flush=True)


def release_if_complete(output, smoke, methods):
    seeds = (1,) if smoke else supervisor.SEEDS
    paths = [completed_path(output, METHOD, seed) for seed in seeds]
    if not all(p.exists() for p in paths):
        return False
    bindings = [{"path": p.relative_to(output).as_posix(), "sha256": supervisor.sha(p)} for p in paths]
    target = output / "selected_main_summary.json"
    if target.exists():
        old = supervisor.read(target)
        if not old.get("completed") or old.get("smoke") != smoke or old.get("source_results") != bindings:
            raise ValueError("Released selected-main aggregate source changed")
        return True
    lock = output / ".aggregate.lock"
    try:
        stream = lock.open("x", encoding="utf-8")
    except FileExistsError:
        return False
    try:
        with stream:
            json.dump({"pid": os.getpid(), "utc": supervisor.utc()}, stream)
        rows = [supervisor.read(p) for p in paths]
        common = rows[0]["identity"]
        for row, path, seed in zip(rows, paths, seeds):
            if not row.get("completed") or row.get("smoke") != smoke or row.get("method") != METHOD or row.get("seed") != seed:
                raise ValueError("Incomplete or incompatible selected-main cell")
            if not smoke and row["dataset"] != supervisor.DATASET:
                raise ValueError("Formal release must contain CSE18 only")
            if any(row["identity"].get(k) != common.get(k) for k in ("git_commit", "source_hashes", "wrapper_source_commit", "wrapper_source_hashes", "effective_config_sha256", "data_binding", "class_order", "tasks", "frozen_selection")):
                raise ValueError("Selected-main release mixes runtime, config, selection or data")
            if supervisor.sha(path.parent / row["raw_result"]) != row["raw_result_sha256"]:
                raise ValueError("Selected-main raw result changed")
        keys = ("average_incremental_accuracy", "final_balanced_accuracy", "forgetting_all_old_classes", "final_attack_detection", "final_macro_f1", "final_normal_fpr")
        metrics = {}
        for key in keys:
            values = [r["metrics"][key] for r in rows]
            defined = all(v is not None for v in values)
            metrics[key] = {"n": len(values), "mean": statistics.mean(values) if defined else None,
                            "sample_sd": statistics.stdev(values) if defined and len(values) > 1 else None}
        supervisor.write_json(target, {"completed": True, "released_at": supervisor.utc(), "smoke": smoke,
            "non_scientific": smoke, "method": METHOD, "seeds": list(seeds), "physical_cells": len(rows),
            "release_rule": "All five formal seeds complete before selected-main aggregate release",
            "source_results": bindings, "frozen_selection": common["frozen_selection"], "metrics": metrics,
            "architecture_capacity": [r["architecture_capacity"] for r in rows],
            "cost_seconds": [r["cost_seconds"] for r in rows], "test_selected": False})
        print(f"SELECTED_MAIN_AGGREGATE_RELEASED physical_cells={len(rows)} smoke={smoke}", flush=True)
        return True
    finally:
        lock.unlink(missing_ok=True)


ORIGINAL_PROTECT = supervisor.protect_owned


def protect_owned(output, protected, attempt, methods, seeds, copied):
    ORIGINAL_PROTECT(output, protected, attempt, methods, seeds, copied)
    summary = output / "selected_main_summary.json"
    if summary.exists():
        target = protected / summary.name
        temporary = target.with_suffix(f".json.{os.getpid()}.tmp")
        shutil.copyfile(summary, temporary)
        temporary.replace(target)


def main():
    selected_configuration()
    # The stable supervisor is parameterized in memory only; no source/import
    # file is edited and the reference package cannot be affected by this entry.
    supervisor.ENTRY = ENTRY
    supervisor.METHODS = (METHOD,)
    supervisor.DATASET_KEYS = {"NSL-KDD": "nsl", "UNSW-NB15": "unsw"}
    # Full-cache validation still needs the formal CSE18 key, while smoke choice
    # is checked before its data access in this entry.
    supervisor.DATASET_KEYS[supervisor.DATASET] = "cic18"
    if "--local-smoke" in sys.argv:
        smoke = sys.argv[sys.argv.index("--smoke-dataset") + 1] if "--smoke-dataset" in sys.argv else None
        if smoke not in ("NSL-KDD", "UNSW-NB15"):
            raise ValueError("Explicit NSL/UNSW CPU smoke dataset required; never smoke CSE18")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    supervisor.run_child = run_child
    supervisor.completed_path = completed_path
    supervisor.release_if_complete = release_if_complete
    supervisor.protect_owned = protect_owned
    supervisor.main()


if __name__ == "__main__":
    main()
