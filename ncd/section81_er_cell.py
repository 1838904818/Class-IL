"""B1 ER cell orchestration; the executable supervisor supplies fixed exposures.

This module does not launch jobs. Exposure construction, source identity and the
global CUDA/resource guard remain the responsibility of the B1 entrypoint.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from ncd import run_aacil as aa
from ncd.run_aacil_retention import stagewise_metrics
from ncd.run_problem_scout import (
    canonical_hash, checkpoint_load, checkpoint_save, digest, require_finite,
    save_json,
)
from ncd.run_replay_bs import evaluate
from ncd.section81_er_training import train_er
from ncd.section81_memory import assemble_er_stage, capacities, retain_memory
from ncd.section81_tabtransformer import TabularClassifier, fit_feature_spec
from ncd.train_cil import seed_everything


def run_er_cell(data, config, identity, output, device, *, calibration_ids,
                original_counts, admission_training_counts, current_exposure,
                smoke_recovery=None):
    """Checkpoint each task, then calibrate/evaluate using the fixed BS rule.

    current_exposure(task_index) returns TRAIN-only slots for the current benign
    stream and newly admitted attacks. It must not supply old attack pools.
    Early-stopping rows and permanent calibration rows are distinct. No result
    is valid until the executable entrypoint binds that exposure policy.
    """
    if identity.get("method") != "er_bal_tabt" or identity.get("section") != 81:
        raise ValueError("B1 ER requires its own method and section identity")
    if identity.get("effective_config") != config:
        raise ValueError("B1 effective configuration is not bound to the cell")
    output.mkdir(parents=True, exist_ok=True)
    final = output / "result.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8"))
        if result.get("identity") != identity or not result.get("completed"):
            raise ValueError("Existing B1 cell differs or is incomplete")
        if len(result["checkpoints"]) != len(data.tasks):
            raise ValueError("Existing B1 cell lacks tasks")
        for record in result["checkpoints"]:
            checkpoint = output / record["checkpoint_file"]
            if (digest(checkpoint) != record["checkpoint_sha256"]
                    or json.loads(checkpoint.with_suffix(".json").read_text()) != record):
                raise ValueError("Existing B1 checkpoint or sidecar changed")
        return result

    seed = int(identity["seed"])
    normal = data.class_order.index("Normal")
    calibration_ids = np.asarray(calibration_ids)
    if (calibration_ids.ndim != 1 or calibration_ids.dtype.kind not in "iu"
            or not len(calibration_ids) or np.any(calibration_ids < 0)
            or np.any(calibration_ids >= len(data.y_train))):
        raise ValueError("Calibration row IDs must be in-range TRAIN integers")
    calibration_ids = calibration_ids.astype(np.int64, copy=False)
    if (len(np.unique(calibration_ids)) != len(calibration_ids)
            or not np.all(data.y_train[calibration_ids] == normal)):
        raise ValueError("Permanent calibration must contain unique Normal TRAIN rows")
    first = current_exposure(0)
    first_arrays = assemble_er_stage(first, {}, normal=normal, calibration_ids=calibration_ids)
    spec = fit_feature_spec(identity["dataset"], data.feature_names, first_arrays["train_x"])
    del first_arrays
    seed_everything(seed * 10000)
    model = TabularClassifier(spec, len(data.class_order)).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    memory, records = {}, []
    if smoke_recovery is not None:
        if not config["smoke"] or device.type != "cpu":
            raise ValueError("Checkpoint migration is restricted to diagnosed CPU smoke")
        old_identity = smoke_recovery["identity"]
        calibration_fields = {"calibration_prior_counts", "calibration_prior_scope"}
        ignored = {"source_hashes"} | calibration_fields
        if ({k: v for k, v in old_identity.items() if k not in ignored}
                != {k: v for k, v in identity.items() if k not in ignored}):
            raise ValueError("Smoke recovery changed data, configuration or identity")
        for key in calibration_fields:
            if key in old_identity and old_identity[key] != identity[key]:
                raise ValueError("Cannot replace an already bound calibration prior")
        if set(identity["source_hashes"]) != set(old_identity["source_hashes"]):
            raise ValueError("Smoke recovery changed the source manifest")
        changed = {k for k in identity["source_hashes"]
                   if identity["source_hashes"][k] != old_identity["source_hashes"].get(k)}
        if not changed.issubset({"ncd/run_section81_er.py", "ncd/section81_er_cell.py"}):
            raise ValueError("Smoke recovery changed a training or exposure primitive")
    for task_index, task in enumerate(data.tasks):
        seen = sorted({c for part in data.tasks[:task_index + 1] for c in part})
        checkpoint = output / "checkpoints" / f"checkpoint_{task_index:03d}.pt"
        sidecar = checkpoint.with_suffix(".json")
        if sidecar.exists() and not checkpoint.exists():
            raise ValueError("B1 metric sidecar has no checkpoint; diagnose before retry")
        if task_index == 0 and smoke_recovery is not None and not checkpoint.exists():
            source = smoke_recovery["checkpoint"]
            if digest(source) != smoke_recovery["checkpoint_sha256"]:
                raise ValueError("Original smoke checkpoint changed")
            state = checkpoint_load(source, smoke_recovery["identity"])
            if state["task_index"] != 0 or state["feature_spec"] != spec or state["seen_classes"] != seen:
                raise ValueError("Original smoke checkpoint schema differs")
            # Preserve the original checkpoint bytes and record the migration explicitly.
            lineage = state.get("recovered_from")
            state["recovered_from"] = {"checkpoint_sha256": digest(source),
                                       "identity": smoke_recovery["identity"],
                                       "prior_recovery": lineage,
                                       "calibration_only_metadata_added": sorted(
                                           calibration_fields - set(old_identity))}
            state["identity"] = identity
            checkpoint_save(checkpoint, state)
        if checkpoint.exists():
            state = checkpoint_load(checkpoint, identity)
            if (state["task_index"] != task_index or state["seen_classes"] != seen
                    or state["feature_spec"] != spec):
                raise ValueError("B1 checkpoint task or vocabulary changed")
            model.load_state_dict(state["model_state"])
            memory = aa.memory_from_tensors(state["memory"])
        else:
            current = first if task_index == 0 else current_exposure(task_index)
            expected = {normal} | (set(task) - {normal})
            if set(current) != expected:
                raise ValueError("Current exposure does not match registered task")
            arrays = assemble_er_stage(current, memory, normal=normal,
                                       calibration_ids=calibration_ids)
            task_seed = seed * 10000 + task_index
            seed_everything(task_seed)
            schedule = train_er(model, arrays["train_x"], arrays["train_y"],
                                arrays["validation_x"], arrays["validation_y"],
                                seen_classes=seen, seed=task_seed, device=device,
                                config=config, smoke=bool(config["smoke"]))
            started = time.perf_counter()
            targets = capacities(identity["memory_setting"], "er_bal_tabt", seen,
                                 normal, admission_training_counts)
            fresh = {c: slot for c, slot in current.items() if c not in memory}
            memory = retain_memory(memory, fresh, targets, seed=seed, task=task_index)
            selection_seconds = time.perf_counter() - started
            state = {"identity": identity, "task_index": task_index,
                     "seen_classes": seen, "feature_spec": spec,
                     "model_state": model.state_dict(),
                     "memory": aa.memory_to_tensors(memory),
                     "calibration_row_ids": torch.from_numpy(calibration_ids.copy()),
                     "optimization_row_ids": torch.from_numpy(arrays["train_ids"].copy()),
                     "stopping_validation_row_ids": torch.from_numpy(arrays["validation_ids"].copy()),
                     "schedule": schedule, "buffer_selection_seconds": selection_seconds,
                     "memory_targets": targets,
                     "memory_counts": {str(c): {s: len(slot[s + "_ids"])
                                                for s in ("train", "validation")}
                                       for c, slot in memory.items()},
                     "memory_bytes": sum(v.nbytes for slot in memory.values() for v in slot.values()),
                     "parameters": parameter_count}
            checkpoint_save(checkpoint, state)
            del arrays, current, fresh
        if task_index == 0:
            # Do not keep Task-0 feature pools alive as an unbounded replay store.
            del first
        checkpoint_sha = digest(checkpoint)
        if sidecar.exists():
            record = json.loads(sidecar.read_text(encoding="utf-8"))
            if (record["identity_hash"] != canonical_hash(identity)
                    or record["checkpoint_sha256"] != checkpoint_sha
                    or record["task_index"] != task_index
                    or record["seen_classes"] != seen
                    or record["checkpoint_file"] != checkpoint.relative_to(output).as_posix()):
                raise ValueError("B1 metric sidecar provenance differs")
        else:
            calibration, points, calibration_seconds, evaluation_seconds = evaluate(
                model, data, seen, calibration_ids,
                np.asarray([original_counts[c] for c in range(len(data.class_order))],
                           dtype=np.int64), config, device)
            record = {k: state[k] for k in ("task_index", "seen_classes", "schedule",
                      "buffer_selection_seconds", "memory_targets", "memory_counts",
                      "memory_bytes", "parameters")}
            record.update(identity_hash=canonical_hash(identity),
                          checkpoint_file=checkpoint.relative_to(output).as_posix(),
                          checkpoint_sha256=checkpoint_sha, calibration=calibration,
                          operating_points={k: points[k] for k in ("0.01", "0.02")},
                          calibration_seconds=calibration_seconds,
                          evaluation_seconds=evaluation_seconds)
            if "recovered_from" in state:
                record["recovered_from"] = state["recovered_from"]
            require_finite(record)
            save_json(sidecar, record)
        records.append(record)
    result = {"schema_version": 1, "identity": identity, "completed": True,
              "completed_at": aa.now(), "checkpoints": records,
              "feature_spec": spec, "parameters": parameter_count,
              "metrics": {key: stagewise_metrics([r["operating_points"][key] for r in records],
                                                 data.tasks) for key in ("0.01", "0.02")},
              "cost_scope": "Task schedules and buffer/calibration/evaluation costs; failed attempts are additional supervisor costs",
              "scope": "B1 adapted ER comparison; not upstream absolute-score reproduction or RQ1/RQ3 completion"}
    require_finite(result)
    save_json(final, result)
    return result
