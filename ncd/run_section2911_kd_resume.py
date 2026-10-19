"""Resume the three remaining CIC17 KD seeds under the owner's Section29.11.

The frozen scientific entry and checkpoint identities remain unchanged. Only
its obsolete absolute-deadline execution check is explicitly replaced. Every
new matrix/cell result records that override and the separate wrapper source.
This entry accepts a fresh destination only; it never writes the old run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = ROOT / "ncd/results/problem_scout/section29/section299_cic_kd_local_v1"
CONFIG = ROOT / "ncd/configs/problem_scout_section21.json"
SOURCE = "7418b7d418f9974e88cba5dcc66c6126df117264"
OLD_DEADLINE = "2026-10-19T21:50:26+08:00"
DATASET, METHOD, SEEDS = "CIC-IDS-2017", "aacil_kd", [3, 4, 42]
SOURCE_PATHS = (
    "ncd/run_aacil.py", "ncd/run_problem_scout.py",
    "ncd/prepare_problem_scout_data.py", "ncd/crr_benchmarks.py",
    "ncd/run_crr_real_sequence.py", "ncd/model.py", "ncd/train_cil.py",
    "ncd/data.py", "ncd/source_commit.py", "ncd/groundtruth.py",
    "src/methods/replay_dpmeans.py", "ncd/exp_h_data.py",
    "ncd/audit_quality_tabular.py", "ncd/run_aacil_retention.py",
    "ncd/aacil_iso.py", "ncd/aacil_retention.py",
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def label(path):
    return path.resolve().relative_to(ROOT).as_posix()


def now():
    return datetime.now().astimezone().isoformat()


def validate_old_run():
    """Bounded input checks required to resume, not another scientific analysis."""
    matrix_path = OLD_ROOT / "result.json"
    matrix = read(matrix_path)
    expected = {"datasets": [DATASET], "methods": [METHOD], "seeds": [1, 2, 3, 4, 42]}
    current = matrix.get("current_cell", {})
    if (matrix.get("expected") != expected or matrix.get("execution_status") == "complete"
            or (current.get("dataset"), current.get("method"), current.get("seed")) != (DATASET, METHOD, 3)):
        raise ValueError("Old matrix is not the interrupted Section29.9 CIC KD run")
    provenance = matrix["provenance"]
    if provenance.get("git_commit") != SOURCE or set(provenance.get("source_hashes", {})) != set(SOURCE_PATHS):
        raise ValueError("Old scientific source or 16-file binding differs")
    for name in SOURCE_PATHS:
        if sha(ROOT / name) != provenance["source_hashes"][name]:
            raise ValueError(f"Frozen scientific source changed: {name}")
    if sha(CONFIG) != provenance["config_sha256"]:
        raise ValueError("Frozen configuration file changed")
    config = provenance["effective_config"]
    fixed = {"deadline_iso": OLD_DEADLINE, "lambda_g": .5,
        "derpp_alpha": 1., "derpp_beta": 1., "cbf_epochs": 1,
        "cbf_lr_multiplier": .1, "epochs_per_task": 8,
        "min_steps_per_task": 2000, "buffer_per_class": 200,
        "batch_size": 256, "normal_training_rows": 50000,
        "section298": True, "selection_only": False, "smoke": False,
        "device": "cuda", "gate_step_policy": "matched_s2000",
        "calibration_policy": "permanent_normal_holdout_outside_m"}
    if any(config.get(k) != v for k, v in fixed.items()):
        raise ValueError("Old KD scientific configuration differs from the frozen selection")
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest() != provenance["effective_config_sha256"]:
        raise ValueError("Old effective configuration binding is inconsistent")
    cells = matrix.get("cells", [])
    if len(cells) != 2 or sorted(c.get("seed") for c in cells) != [1, 2]:
        raise ValueError("The old matrix must contain exactly the two completed seeds 1 and 2")
    for item in cells:
        seed = item["seed"]
        result_path = OLD_ROOT / "cic_ids_2017" / f"{METHOD}_seed{seed}" / "result.json"
        result = read(result_path)
        identity = result.get("identity", {})
        if (not result.get("completed") or len(result.get("checkpoints", [])) != 4
                or (identity.get("dataset"), identity.get("method"), identity.get("seed")) != (DATASET, METHOD, seed)
                or any(identity.get(k) != v for k, v in provenance.items())
                or sha(result_path) != item.get("result_sha256")):
            raise ValueError(f"Old completed seed {seed} is not the bound completed result")
    seed3 = OLD_ROOT / "cic_ids_2017" / f"{METHOD}_seed3"
    if (seed3 / "result.json").exists():
        raise ValueError("Seed 3 already has a result; do not start another continuation")
    for seed in (4, 42):
        old_cell = OLD_ROOT / "cic_ids_2017" / f"{METHOD}_seed{seed}"
        if old_cell.exists() and any(old_cell.iterdir()):
            raise ValueError(f"Old seed {seed} has existing work; this fixed continuation is not applicable")
    directory = seed3 / "checkpoints"
    expected_files = {f"checkpoint_{task:03d}.{ext}" for task in (0, 1) for ext in ("pt", "json")}
    if not directory.is_dir() or {p.name for p in directory.iterdir()} != expected_files:
        raise ValueError("Seed 3 must have exactly checkpoints 000/001 and their JSON sidecars")
    return matrix, provenance, seed3, sha(matrix_path)


def saved_costs(output):
    """Sum disjoint saved stages; nested cost components are never totalled together."""
    inherited, new, records = {}, {}, []
    for seed in SEEDS:
        directory = output / "cic_ids_2017" / f"{METHOD}_seed{seed}" / "checkpoints"
        for path in sorted(directory.glob("checkpoint_*.json")):
            row = read(path)
            task = row["task_index"]
            if path.name != f"checkpoint_{task:03d}.json":
                raise ValueError("Saved-stage index differs from its sidecar filename")
            prior = seed == 3 and task in (0, 1)
            target = inherited if prior else new
            values = {k: v for k, v in row.items() if k.endswith("_seconds")
                and isinstance(v, (int, float)) and not isinstance(v, bool)}
            if any(not math.isfinite(v) or v < 0 for v in values.values()):
                raise ValueError("Saved stage has invalid cost values")
            for k, v in values.items():
                target[k] = target.get(k, 0.) + v
            records.append({"seed": seed, "task_index": task,
                "origin": "inherited_saved_stage" if prior else "new_saved_stage",
                "sidecar": label(path)})
    return {"inherited_saved_stage_cost_seconds": inherited,
        "new_saved_stage_cost_seconds": new,
        "all_saved_stage_cost_seconds": {k: inherited.get(k, 0.) + new.get(k, 0.) for k in sorted(set(inherited) | set(new))},
        "saved_stage_inventory": records,
        "cost_scope": "Each saved stage is counted once. training=gate+family; family includes base_family+CBF, so nested components are not additive again. Cumulative fields are excluded. Incomplete stages, failed work, loading and serialization are not in saved-stage totals.",
        "old_unsaved_task2_cost_seconds": None,
        "old_unsaved_task2_note": "The interrupted old task2 consumed work but has no saved checkpoint or exact recorded stage duration. Its cost is unknown, not zero; old logs remain preserved. Seeds1/2 are outside this continuation and must not be charged again as new execution."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
        help="A previously nonexistent directory under this repository's Section29 results")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    allowed = (ROOT / "ncd/results/problem_scout/section29").resolve()
    if not output.is_relative_to(allowed) or output == allowed or output == OLD_ROOT.resolve():
        parser.error("The new output must be a distinct Section29 subdirectory")
    if output.exists():
        parser.error("Continuation destination already exists; preserve it and choose a fresh directory")
    started = time.perf_counter()
    started_at = now()
    matrix, provenance, seed3, matrix_sha = validate_old_run()
    wrapper_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if len(wrapper_commit) != 40 or any(c not in "0123456789abcdef" for c in wrapper_commit):
        raise ValueError("Cannot identify the separate wrapper commit")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from ncd import run_aacil_retention as frozen
    if tuple(frozen.SOURCE_PATHS) != SOURCE_PATHS:
        raise ValueError("Imported frozen dependency list differs")
    source_files = []
    seed3_identity = None
    for task in (0, 1):
        checkpoint = seed3 / "checkpoints" / f"checkpoint_{task:03d}.pt"
        sidecar = checkpoint.with_suffix(".json")
        record = read(sidecar)
        state = frozen.torch.load(checkpoint, map_location="cpu", weights_only=True)
        identity = state.get("identity", {})
        if (any(identity.get(k) != v for k, v in provenance.items())
                or (identity.get("dataset"), identity.get("method"), identity.get("seed")) != (DATASET, METHOD, 3)
                or state.get("task_index") != task or record.get("task_index") != task
                or record.get("identity_hash") != frozen.canonical_hash(identity)
                or record.get("checkpoint_file") != f"checkpoints/checkpoint_{task:03d}.pt"
                or record.get("checkpoint_sha256") != sha(checkpoint)
                or state.get("training_seconds_cumulative") != record.get("training_seconds_cumulative")
                or state.get("seen_classes") != [c for part in identity["tasks"][:task + 1] for c in part]
                or (seed3_identity is not None and seed3_identity != identity)):
            raise ValueError(f"Seed3 checkpoint {task} cannot be resumed with its original identity")
        seed3_identity = identity
        for source in (checkpoint, sidecar):
            source_files.append({"source": label(source), "sha256": sha(source), "bytes": source.stat().st_size})
        del state
    invocation = ["python", "-I", "ncd/run_section2911_kd_resume.py", "--output-dir", label(output)]
    actual_command = subprocess.list2cmdline(invocation)
    forwarded = ["ncd/run_aacil_retention.py", "--datasets", DATASET, "--methods", METHOD,
        "--seeds", *map(str, SEEDS), "--lambda-g", "0.5", "--der-alpha", "1", "--der-beta", "1",
        "--cbf-epochs", "1", "--cbf-lr-multiplier", "0.1", "--source-commit", SOURCE,
        "--deadline-iso", OLD_DEADLINE, "--gate-step-policy", "matched_s2000",
        "--calibration-policy", "permanent_normal_holdout_outside_m", "--device", "cuda",
        "--output-dir", str(output)]
    metadata = {"section": "29.11", "schema_version": 1, "started_at": started_at,
        "scientific_source_commit": SOURCE, "wrapper_source_commit": wrapper_commit,
        "wrapper_source_sha256": sha(Path(__file__)), "actual_wrapper_command": actual_command,
        "command_representation": "Reproducible executed wrapper argv; executable is normalized to python and repository paths are relative, not private absolute paths.",
        "resume_source": label(OLD_ROOT), "resume_source_matrix_sha256": matrix_sha,
        "resume_seed": 3, "inherited_checkpoint_identity_sha256": frozen.canonical_hash(seed3_identity),
        "copied_checkpoint_sources": source_files, "new_seeds": [4, 42],
        "completed_prior_seeds_not_rerun": [1, 2],
        "deadline_override": {"authority": "docs/WORK_ORDER_2026-10-09.md Section29.11; explicit owner continuation without deadline",
            "old_deadline_disabled": True, "inherited_deadline_iso": OLD_DEADLINE,
            "active_deadline_iso": None, "historical_config_field_retained_for_exact_checkpoint_identity": True,
            "interpretation": "The inherited deadline field is historical metadata, not an active execution cutoff. Only deadline checking is replaced; scientific configuration and training functions are unchanged."}}
    output.mkdir(parents=True, exist_ok=False)
    target = output / "cic_ids_2017" / f"{METHOD}_seed3" / "checkpoints"
    target.mkdir(parents=True, exist_ok=False)
    for item in source_files:
        original = ROOT / item["source"]
        destination = target / original.name
        shutil.copy2(original, destination)
        if sha(destination) != item["sha256"]:
            raise ValueError("Copied checkpoint/sidecar bytes differ; preserve both directories")
    original_save, original_check, original_command = frozen.save_json, frozen.check_deadline, frozen.command
    original_argv, old_env = sys.argv, os.environ.get("NCD_SOURCE_COMMIT")

    def save_with_resume(path, value):
        if path.name == "result.json":
            payload = dict(value)
            execution = metadata | {"wrapper_elapsed_seconds": time.perf_counter() - started}
            if "identity" in value:
                seed = value["identity"]["seed"]
                costs = saved_costs(output)
                # A cell result reports its own stages; matrix metadata covers all three seeds.
                own_records = [r for r in costs["saved_stage_inventory"] if r["seed"] == seed]
                inherited_cost, new_cost = {}, {}
                for item in own_records:
                    row = read(ROOT / item["sidecar"])
                    dest = inherited_cost if item["origin"] == "inherited_saved_stage" else new_cost
                    for k, v in row.items():
                        if k.endswith("_seconds") and isinstance(v, (int, float)) and not isinstance(v, bool):
                            dest[k] = dest.get(k, 0.) + v
                costs.update(saved_stage_inventory=own_records,
                    inherited_saved_stage_cost_seconds=inherited_cost, new_saved_stage_cost_seconds=new_cost,
                    all_saved_stage_cost_seconds={k: inherited_cost.get(k, 0.) + new_cost.get(k, 0.) for k in sorted(set(inherited_cost) | set(new_cost))})
                execution["cost_accounting"] = costs
                execution["cell_resumed_from_old_checkpoints"] = seed == 3
            else:
                execution["cost_accounting"] = saved_costs(output)
            payload.update(section2911_resume=execution, command=actual_command)
            return original_save(path, payload)
        return original_save(path, value)  # Checkpoint sidecars retain the frozen schema and content.

    def deadline_released(config):
        if config != provenance["effective_config"]:
            raise ValueError("Section29.11 override applies only to the exact inherited scientific configuration")
        # The owner revoked this specific operational cutoff; no other control is bypassed.

    frozen.save_json, frozen.check_deadline, frozen.command = save_with_resume, deadline_released, lambda: actual_command
    try:
        save_with_resume(output / "result.json", {"schema_version": 1,
            "section29_schema_version": 1, "section298_schema_version": 1,
            "execution_status": "continuation_prepared", "provenance": provenance,
            "expected": {"datasets": [DATASET], "methods": [METHOD], "seeds": SEEDS},
            "expected_cells": 3, "cells": [], "current_cell": None, "smoke": False, "selection_only": False})
        with (output / "run.log").open("a", encoding="utf-8") as log:
            log.write("SECTION2911_DEADLINE_RELEASE " + json.dumps(metadata, sort_keys=True) + "\n")
        print("SECTION2911_CONTINUATION_START " + actual_command, flush=True)
        os.environ["NCD_SOURCE_COMMIT"] = SOURCE
        sys.argv = forwarded
        frozen.main()
    finally:
        frozen.save_json, frozen.check_deadline, frozen.command = original_save, original_check, original_command
        sys.argv = original_argv
        if old_env is None:
            os.environ.pop("NCD_SOURCE_COMMIT", None)
        else:
            os.environ["NCD_SOURCE_COMMIT"] = old_env


if __name__ == "__main__":
    main()
