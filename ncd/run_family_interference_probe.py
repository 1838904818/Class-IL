"""Instrument existing exploratory trajectories without changing real updates.

Five same-state one-step Adam interventions diagnose conditional component
effects. They are discarded, not new training methods or a hyperparameter
search. Per-example training trajectories are LOCAL ONLY, never test feedback.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from ncd import run_rare_boundary_pilot as pilot
from ncd.crr_benchmarks import _class_balanced_cross_entropy, training_epoch_batches
from ncd.model import MLP
from ncd.prepare_problem_scout_data import prepare_setting
from ncd.run_benign_coverage import SOURCE_PATHS as FROZEN_SOURCES, cell_lock
from ncd.run_problem_scout import (
    Tee, binding, canonical_hash, checkpoint_load, checkpoint_save, digest,
    evaluate, path_label, require_finite, save_json, seed_everything,
)
from ncd.source_commit import source_commit

ARMS = ("mir", "mir_rare_ce")
DATASETS = ("NSL-KDD", "CIC-IDS-2017")
CONDITIONS = ("full", "rare_only", "nonrare_only", "zero_gradient", "without_competitor")
REFERENCE_COMMIT = "fafcdd9f821c4d3460ceab340ba4b5ceae293263"
FAMILY = {"NSL-KDD": {"competitor": "R2L", "old_probe": "R2L"},
          "CIC-IDS-2017": {"competitor": "DoS", "old_probe": "WebAttack"}}
POLICIES = {
    "scope": "Exploratory one-step component intervention on eight existing MIR/MIR+rareCE paths. No novelty, final-improvement, historical-step recovery, RQ1/RQ3 or overall completion claim.",
    "real_path": "Original loss.backward and Adam step, task3 only, exactly 2000 real steps; common verified checkpoint002, original current/memory/candidate/class/index schedules. No optimizer-gradient replacement.",
    "access": "Only current task training rows and checkpoint002 resident memory. No old pool revisit, new memory, test retrieval, hyperparameter search, or test feedback.",
    "components": "Whole-minibatch CBCE weights and denominator are retained. Current rare rows contribute gR, all remaining rows gN; original extra rare CE belongs to gR. Fixed competitor component uses the same whole-batch weights.",
    "competitor": "NSL R2L and CIC DoS were fixed from previously inspected TRAIN traces. Exploratory choice, not held-out selection. A wrong predicted family alone does not identify its training gradient as the cause.",
    "intervention": "Pre-completed-step grid 0,50,...1950; next batch has five clones from identical model and Adam m/v/step. Explicit zero gradients advance Adam state. Each clone takes one step and evaluates identical fixed training/resident probes, then is discarded. Full clone is the compute-matched sham.",
    "estimand": "F(full)-F(rare_only) is the conditional one-step effect of including nonrare gradients on fixed rare TRAIN mean CE, holding prestate/batch fixed. F(full)-F(rare)-F(nonrare)+F(zero) records nonadditivity. Not an additive gradient attribution or a test generalization estimate.",
    "trace": "Step0 and every real post-update step record each current rare training row and fixed old rare resident row's prediction, true-minus-max-other margin and competitor ID. Prepared row IDs and resident-prefix indices have separate namespaces. New measurements, not recovered historical traces. Original every50-step aggregate trace is also retained.",
    "adam": "Raw gradient dot products do not guarantee Adam protection. Features/head block norms/dots and actual cloned displacements are diagnostic. Zero-gradient updates include existing moment and second-moment decay, not a no-update model.",
    "cost": "All five conditions share component-gradient construction and perform one Adam step plus identical probe forward rows. Real-path and added diagnostic costs are separate; different original arms are not claimed equal FLOPs, time or working memory.",
    "publication": "Per-row NPZ, endpoint PT and minimal prestate PT payloads are LOCAL ONLY. Publish only summarized JSON and sanitized source/config; fixed training row IDs are hashed in public metadata.",
    "resume": "Same-identity checkpoints every50 real steps contain Adam, audit/events and local raw traces. Completed durable steps are skipped; unsuccessful work after a durable boundary is failed-attempt cost, not a replicate.",
}


def scalar(value):
    return float(value.detach().double().item())


def exact_tree(left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                and left.dtype == right.dtype and left.shape == right.shape
                and torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, dict) or isinstance(right, dict):
        return isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys() and all(exact_tree(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(exact_tree(a, b) for a, b in zip(left, right))
    return left == right


def model_tensor_hash(state):
    h = __import__("hashlib").sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(np.asarray(value.shape, dtype="<i8").tobytes())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def read_reference(reference_root, dataset, seed, arm, data, hashes, prefix_ref, config):
    cell = reference_root / dataset.lower().replace("-", "_") / f"{arm}_seed{seed}"
    path = cell / "result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    ident = result["identity"]
    if not result.get("completed") or ident.get("smoke") or ident.get("steps") != 2000:
        raise ValueError("Reference must be a completed full 2000-step pilot")
    if (ident["dataset"], ident["seed"], ident["arm"], ident["git_commit"]) != (dataset, seed, arm, REFERENCE_COMMIT):
        raise ValueError("Reference cell/source differs from frozen pilot")
    if (ident["data_binding"] != hashes or ident["class_order"] != data.class_order
            or ident["tasks"] != data.tasks or ident["start_checkpoint"] != prefix_ref):
        raise ValueError("Reference data/prefix/class binding differs")
    for name, sha in ident["source_hashes"].items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"Historical source changed: {name}")
    if ident["config_sha256"] != digest(ROOT / "ncd/configs/problem_scout_section21.json"):
        raise ValueError("Historical raw config changed")
    # Smoke intentionally has two steps and capped current rows; other training
    # settings, including CPU/GPU numerics, must remain the original settings.
    for key in ("hidden", "learning_rate", "batch_size", "buffer_per_class", "cpu_threads", "device"):
        if ident["effective_config"][key] != config[key]:
            raise ValueError(f"Reference effective setting differs: {key}")
    payload = cell / result["final_checkpoint"]["path"]
    sha = digest(payload)
    if sha != result["final_checkpoint"]["sha256"]:
        raise ValueError("Reference actual final payload SHA differs")
    if json.loads(payload.with_suffix(".json").read_text(encoding="utf-8")) != result:
        raise ValueError("Reference checkpoint sidecar differs")
    state = checkpoint_load(payload, ident)
    if state["task_index"] != 3 or state["book"]["chains"] != result["audit"]["chains"]:
        raise ValueError("Reference checkpoint task/audit differs")
    return result, state, {"result_path": path_label(path), "result_sha256": digest(path),
                           "checkpoint_path": path_label(payload), "checkpoint_sha256": sha,
                           "identity_hash": canonical_hash(ident), "training_source_commit": ident["git_commit"],
                           "model_tensor_sha256": model_tensor_hash(state["model_state"]),
                           "actual_payload_verified": True}


def component_gradients(logits, labels, rare_mask, extra, competitor_id, parameters):
    classes, counts = torch.unique(labels, return_counts=True)
    weights = logits.new_zeros(logits.shape[1])
    weights[classes] = labels.numel() / (len(classes) * counts.to(logits.dtype))
    weighted_rows = F.cross_entropy(logits, labels, weight=weights, reduction="none")
    denominator = weights[labels].sum()
    rare_loss = weighted_rows[rare_mask].sum() / denominator + extra
    nonrare_loss = weighted_rows[~rare_mask].sum() / denominator
    competitor_loss = weighted_rows[labels == competitor_id].sum() / denominator
    gradients = {}
    for name, loss in (("rare", rare_loss), ("nonrare", nonrare_loss), ("competitor", competitor_loss)):
        gradients[name] = tuple(g.detach().clone() for g in torch.autograd.grad(loss, parameters, retain_graph=True))
    return gradients, {"rare_component_loss": scalar(rare_loss), "nonrare_component_loss": scalar(nonrare_loss),
                       "competitor_component_loss": scalar(competitor_loss), "weighted_denominator": scalar(denominator),
                       "minibatch_classes": classes.cpu().tolist(), "minibatch_counts": counts.cpu().tolist()}


def vector(values, indices):
    return torch.cat([values[i].detach().reshape(-1).double() for i in indices])


def gradient_blocks(names, gradients, full):
    result = {}
    for block in ("features", "head"):
        ids = [i for i, name in enumerate(names) if name.startswith(block + ".")]
        r, n, c, f = [vector(v, ids) for v in (gradients["rare"], gradients["nonrare"], gradients["competitor"], full)]
        rn, nn = scalar(r.norm()), scalar(n.norm())
        dot = scalar(torch.dot(r, n))
        result[block] = {"rare_l2": rn, "nonrare_l2": nn, "competitor_l2": scalar(c.norm()),
                         "full_l2": scalar(f.norm()), "rare_dot_nonrare": dot,
                         "rare_nonrare_cosine": dot / (rn * nn) if rn * nn else None,
                         "decomposition_residual_l2": scalar((f - r - n).norm()),
                         "decomposition_residual_max_abs": scalar((f - r - n).abs().max())}
    return result


def prediction_values(logits, labels):
    rows = torch.arange(len(labels), device=logits.device)
    other = logits.clone()
    other[rows, labels] = -torch.inf
    rival_logits, rival = other.max(dim=1)
    return logits.argmax(dim=1), logits[rows, labels] - rival_logits, rival


def evaluate_probes(model, groups, nclasses):
    result = {}
    model.eval()
    with torch.no_grad():
        for name, (x, y) in groups.items():
            logits = model(x)
            pred, margin, rival = prediction_values(logits, y)
            result[name] = {"rows": len(y), "mean_ce": scalar(F.cross_entropy(logits, y)),
                            "recall": scalar((pred == y).float().mean()),
                            "true_minus_max_other_mean": scalar(margin.mean()),
                            "true_minus_max_other_p05": scalar(torch.quantile(margin, .05)),
                            "prediction_counts": torch.bincount(pred, minlength=nclasses).cpu().tolist(),
                            "max_other_class_counts": torch.bincount(rival, minlength=nclasses).cpu().tolist()}
    model.train()
    return result


def one_step_clones(model, optimizer, full, gradients, names, groups, config, device, audit):
    """No clone can mutate the real optimizer, parameters, gradients, or RNG."""
    pilot.sync(device)
    began = time.perf_counter()
    before = tuple(p.detach().clone() for p in model.parameters())
    conditions = {"full": full, "rare_only": gradients["rare"], "nonrare_only": gradients["nonrare"],
                  "zero_gradient": tuple(torch.zeros_like(g) for g in full),
                  "without_competitor": tuple(g - c for g, c in zip(full, gradients["competitor"]))}
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
    pre_began = time.perf_counter()
    pre_probes = evaluate_probes(model, groups, len(model.head.bias))
    pilot.sync(device)
    pre_seconds = time.perf_counter() - pre_began
    audit["costs"]["prestate_probe_seconds"] += pre_seconds
    audit["costs"]["prestate_probe_forward_rows"] += sum(len(v[1]) for v in groups.values())
    results, deltas, full_next_optimizer = {}, {}, None
    for name in CONDITIONS:
        clone = copy.deepcopy(model)
        clone_optimizer = torch.optim.Adam(clone.parameters(), lr=config["learning_rate"])
        clone_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        for parameter, gradient in zip(clone.parameters(), conditions[name]):
            parameter.grad = gradient.detach().clone()  # Explicit zeros, never None.
        start = time.perf_counter()
        clone_optimizer.step()
        pilot.sync(device)
        adam_seconds = time.perf_counter() - start
        if name == "full":
            full_next_optimizer = copy.deepcopy(clone_optimizer.state_dict())
        delta = tuple(p.detach() - base for p, base in zip(clone.parameters(), before))
        deltas[name] = delta
        start = time.perf_counter()
        metrics = evaluate_probes(clone, groups, len(model.head.bias))
        pilot.sync(device)
        probe_seconds = time.perf_counter() - start
        displacement = {}
        for block in ("features", "head"):
            ids = [i for i, key in enumerate(names) if key.startswith(block + ".")]
            d, r, n = [vector(v, ids) for v in (delta, gradients["rare"], gradients["nonrare"])]
            displacement[block] = {"l2": scalar(d.norm()), "rare_gradient_dot_delta": scalar(torch.dot(r, d)),
                                   "nonrare_gradient_dot_delta": scalar(torch.dot(n, d))}
        results[name] = {"probes": metrics, "displacement": displacement,
                         "adam_seconds": adam_seconds, "probe_seconds": probe_seconds,
                         "virtual_optimizer_steps": 1, "probe_forward_rows": sum(len(v[1]) for v in groups.values())}
        audit["costs"]["clone_optimizer_steps"] += 1
        audit["costs"]["clone_probe_forward_rows"] += results[name]["probe_forward_rows"]
        del clone_optimizer, clone
    # Deep copies and deterministic forwards must not perturb the real RNG.
    if not torch.equal(torch.get_rng_state(), cpu_rng):
        raise ValueError("Diagnostic clones changed CPU RNG")
    if device.type == "cuda" and any(not torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), cuda_rng)):
        raise ValueError("Diagnostic clones changed CUDA RNG")
    conditional_delta = {}
    for block in ("features", "head"):
        ids = [i for i, key in enumerate(names) if key.startswith(block + ".")]
        delta = vector(deltas["full"], ids) - vector(deltas["rare_only"], ids)
        conditional_delta[block] = {"l2": scalar(delta.norm()),
                                    "rare_gradient_dot_full_minus_rare_delta": scalar(torch.dot(vector(gradients["rare"], ids), delta))}
    effects = {}
    for group in groups:
        f = {name: results[name]["probes"][group]["mean_ce"] for name in CONDITIONS}
        effects[group] = {"full_minus_rare_ce": f["full"] - f["rare_only"],
                          "nonrare_minus_zero_ce": f["nonrare_only"] - f["zero_gradient"],
                          "full_minus_nonrare_ce": f["full"] - f["nonrare_only"],
                          "rare_minus_zero_ce": f["rare_only"] - f["zero_gradient"],
                          "interaction_ce": f["full"] - f["rare_only"] - f["nonrare_only"] + f["zero_gradient"],
                          "full_minus_without_competitor_ce": f["full"] - f["without_competitor"],
                          "condition_minus_pre_ce": {name: value - pre_probes[group]["mean_ce"] for name, value in f.items()}}
    pilot.sync(device)
    audit["costs"]["clone_total_seconds"] += time.perf_counter() - began - pre_seconds
    return {"pre_probes": pre_probes, "pre_probe_seconds": pre_seconds, "conditions": results,
            "conditional_displacement": conditional_delta, "effects": effects}, deltas["full"], full_next_optimizer


def append_training_trace(model, combined_probe, raw, device, audit):
    pilot.sync(device)
    began = time.perf_counter()
    x, y = combined_probe
    model.eval()
    with torch.no_grad():
        pred, margin, rival = prediction_values(model(x), y)
    model.train()
    raw["predictions"].append(pred.cpu().to(torch.int16))
    raw["margins"].append(margin.cpu())
    raw["competitors"].append(rival.cpu().to(torch.int16))
    pilot.sync(device)
    audit["costs"]["per_step_trace_seconds"] += time.perf_counter() - began
    audit["costs"]["per_step_trace_forward_rows"] += len(y)


def raw_tensors(raw):
    return {key: torch.stack(values) for key, values in raw.items()}


def trace_summary(raw, truth, nclasses, start, end):
    predictions = torch.stack(raw["predictions"]).numpy()[:, start:end]
    margins = torch.stack(raw["margins"]).numpy()[:, start:end]
    correct = predictions == truth[None, :]
    final = correct[-1]
    return {"rows": len(truth), "steps_including_zero": len(predictions),
            "ever_correct_rows": int(correct.any(axis=0).sum()),
            "never_correct_rows": int((~correct.any(axis=0)).sum()),
            "final_correct_rows": int(final.sum()),
            "ever_correct_but_final_wrong_rows": int((correct.any(axis=0) & ~final).sum()),
            "correct_to_wrong_events": int((correct[:-1] & ~correct[1:]).sum()),
            "wrong_to_correct_events": int((~correct[:-1] & correct[1:]).sum()),
            "prediction_change_events": int((predictions[1:] != predictions[:-1]).sum()),
            "final_prediction_counts": np.bincount(predictions[-1], minlength=nclasses).tolist(),
            "final_margin_mean": float(margins[-1].mean()),
            "limit": "Exact newly measured per-update training trajectory; not historical optimization timing or test performance."}


def save_state(path, model, optimizer, identity, prefix, book, schedule, audit, raw):
    checkpoint_save(path, {"identity": identity, "task_index": 3,
                          "seen_classes": list(range(len(identity["class_order"]))),
                          "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                          "optimizer_state": optimizer.state_dict(), "buffer_x": prefix["buffer_x"], "buffer_y": prefix["buffer_y"],
                          "book": book, "training_schedule": schedule, "interference_audit": audit,
                          "rare_trace_local_only": raw_tensors(raw), "torch_rng_state": torch.get_rng_state(),
                          "cuda_rng_states": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else []})


def save_prestate(output, identity, pre_step, selected, model, optimizer, audit):
    began = time.perf_counter()
    path = output / "prestates" / f"step{pre_step:04d}.pt"
    state = {"identity": identity, "pre_step": pre_step,
             "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
             "optimizer_state": optimizer.state_dict(), "selected_pool_indices": torch.from_numpy(selected.copy())}
    if path.exists():
        prior = checkpoint_load(path, identity)
        if not exact_tree(prior, state):
            raise ValueError("Existing local prestate differs on resumed deterministic path")
    else:
        checkpoint_save(path, state)
    metadata = {"path": path.relative_to(output).as_posix(), "sha256": digest(path), "publish": False,
                "pre_step": pre_step, "selected_pool_indices_sha256": pilot.array_hash(selected)}
    audit["costs"]["prestate_serialization_seconds"] += time.perf_counter() - began
    return metadata


def exact_comparison(model, optimizer, prefix, book, schedule, reference, reference_state, metrics, smoke):
    if smoke:
        return {"required": False, "reason": "Two-step capped smoke is non-scientific; full reference payload still verified."}
    actual = model.state_dict()
    reference_model = reference_state["model_state"]
    parameter_equal = set(actual) == set(reference_model) and all(torch.equal(actual[k].detach().cpu(), reference_model[k]) for k in actual)
    checks = {"model_tensors_equal": parameter_equal,
              "final_confusion_equal": metrics["confusion_matrix"] == reference["id"]["confusion_matrix"],
              "all_six_audit_chains_equal": book["chains"] == reference["audit"]["chains"],
              "class_exposures_equal": book["class_exposures"] == reference["audit"]["class_exposures"],
              "training_schedule_equal": schedule == reference["training_schedule"],
              "optimizer_full_state_equal": exact_tree(optimizer.state_dict(), reference_state["optimizer_state"]),
              "resident_memory_equal": exact_tree(prefix["buffer_x"], reference_state["buffer_x"]) and exact_tree(prefix["buffer_y"], reference_state["buffer_y"]),
              "current_rows_equal": book["current_rows"] == reference["audit"]["current_rows"],
              "replay_rows_equal": book["replay_rows"] == reference["audit"]["replay_rows"]}
    if not all(checks.values()):
        raise ValueError(f"Instrumented path changed original endpoint/exposures: {checks}")
    return {"required": True, **checks, "model_tensor_sha256": model_tensor_hash(actual),
            "reference_model_tensor_sha256": model_tensor_hash(reference_model)}


def run_cell(data, x, y, ncurrent, original_rows, prefix, identity, output, config, device, reference, reference_state):
    result_path, progress = output / "result.json", output / "checkpoints/progress.pt"
    final_path, raw_path = output / "checkpoints/checkpoint_003.pt", output / "training_trajectory.local_only.npz"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("identity") != identity or not result.get("completed"):
            raise ValueError("Existing result identity/completeness differs")
        if digest(final_path) != result["final_checkpoint"]["sha256"] or digest(raw_path) != result["local_only_trace"]["sha256"]:
            raise ValueError("Completed payload or local raw trace changed")
        if json.loads(final_path.with_suffix(".json").read_text(encoding="utf-8")) != result:
            raise ValueError("Completed sidecar differs")
        for metadata in result["local_only_prestates"]:
            if digest(output / metadata["path"]) != metadata["sha256"]:
                raise ValueError("Completed local prestate changed")
        return result
    seed, nclasses = identity["seed"], len(data.class_order)
    task_seed = seed * 10000 + 3
    seed_everything(task_seed)
    model = MLP(x.shape[1], config["hidden"], nclasses).to(device)
    model.load_state_dict(prefix["model_state"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    names, parameters = zip(*model.named_parameters())
    book, schedule = pilot.fresh_book(nclasses), {}
    audit = {"events": [], "costs": {"component_seconds": 0.0, "component_backward_calls": 0,
             "component_backward_rows": 0, "clone_total_seconds": 0.0, "clone_optimizer_steps": 0,
             "clone_probe_forward_rows": 0, "per_step_trace_seconds": 0.0, "per_step_trace_forward_rows": 0,
             "prestate_probe_seconds": 0.0, "prestate_probe_forward_rows": 0, "prestate_serialization_seconds": 0.0,
             "instrumented_real_optimization_seconds": 0.0, "raw_serialization_seconds": 0.0},
             "full_clone_actual_update_exact_matches": 0, "full_clone_actual_optimizer_exact_matches": 0}
    raw = {key: [] for key in ("predictions", "margins", "competitors")}
    restoring = final_path if final_path.exists() else progress if progress.exists() else None
    if restoring:
        prior = checkpoint_load(restoring, identity)
        model.load_state_dict(prior["model_state"])
        optimizer.load_state_dict(prior["optimizer_state"])
        book, audit = prior["book"], prior["interference_audit"]
        raw = {key: list(tensor.unbind()) for key, tensor in prior["rare_trace_local_only"].items()}
        torch.set_rng_state(prior["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(prior["cuda_rng_states"])
        print(f"PROGRESS_RESUME step={book['step']} {identity['arm']}", flush=True)
    rare = identity["rare_classes"]
    normal = data.class_order.index("Normal")
    rare_ids = np.flatnonzero((np.arange(len(y)) < ncurrent) & np.isin(y, rare))
    old_ids = np.flatnonzero((np.arange(len(y)) >= ncurrent) & (y == identity["old_probe_class"]))
    if len(old_ids) != 200 or len(rare_ids) == 0:
        raise ValueError("Fixed training/resident probe membership differs")
    groups = {name: (torch.from_numpy(x[ids]).to(device), torch.from_numpy(y[ids]).to(device))
              for name, ids in (("current_rare_train", rare_ids), ("old_rare_resident", old_ids))}
    trajectory_ids = np.concatenate([rare_ids, old_ids])
    combined_probe = tuple(torch.cat([value[i] for value in groups.values()]) for i in (0, 1))
    if not raw["predictions"]:
        append_training_trace(model, combined_probe, raw, device, audit)
    if len(raw["predictions"]) != book["step"] + 1:
        raise ValueError("Resumed raw trace length differs from durable step")
    if not book["traces"]:
        trace, elapsed, rows = pilot.training_trace(model, x, y, ncurrent, rare, normal, 0, device)
        book["traces"].append(trace)
        book["costs"]["trace_seconds"] += elapsed
        book["costs"]["trace_forward_rows"] += rows
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    old = {c: np.flatnonzero((np.arange(len(y)) >= ncurrent) & (y == c)) for c in np.unique(y[ncurrent:])}
    index_x = np.arange(len(y), dtype=np.int64).reshape(-1, 1)
    expected = {key: pilot.EMPTY_HASH for key in ("baseline_indices", "current_indices", "candidate_indices", "class_counts")}
    resume_step, step = book["step"], 0
    for _, batches in training_epoch_batches(index_x, y, 256, task_seed, config["epochs_per_task"], config["min_steps_per_task"], schedule):
        for tensor_indices, _ in batches:
            step += 1
            original = tensor_indices.numpy().reshape(-1)
            began = time.perf_counter()
            candidates = pilot.candidates_for_step(original, y, ncurrent, old, seed, step)
            candidate_time = time.perf_counter() - began
            current = original[original < ncurrent]
            counts = np.bincount(y[original], minlength=nclasses)
            values = {"baseline_indices": original, "current_indices": current, "candidate_indices": candidates, "class_counts": counts}
            for key, value in values.items():
                expected[key] = pilot.advance(expected[key], value)
            if step <= resume_step:
                if step == resume_step and any(expected[k] != book["chains"][k] for k in expected):
                    raise ValueError("Resumed deterministic schedule changed")
                continue
            book["costs"]["candidate_build_seconds"] += candidate_time
            selected = pilot.retrieve(model, optimizer, x, y, original, candidates, ncurrent, device, config, book["costs"])
            actual_counts = np.bincount(y[selected], minlength=nclasses)
            if (not np.array_equal(selected[selected < ncurrent], current) or not np.array_equal(counts, actual_counts)
                    or len(np.unique(selected)) != len(selected)):
                raise ValueError("MIR changed current exposure/class quotas or duplicated minibatch rows")
            pilot.sync(device)
            began = time.perf_counter()
            bx, by = torch.from_numpy(x[selected]).to(device), torch.from_numpy(y[selected]).to(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx)
            base = _class_balanced_cross_entropy(logits, by)
            rare_mask = torch.from_numpy((selected < ncurrent) & np.isin(y[selected], rare)).to(device)
            ce, margin, proxy = pilot.extra_losses(logits, by, rare_mask, normal)
            extra = ce if identity["arm"] == "mir_rare_ce" else logits.sum() * 0.0
            loss = base + extra
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite original objective")
            measure = (step - 1) % 50 == 0
            component_seconds = 0.0
            if measure:
                pilot.sync(device)
                start_component = time.perf_counter()
                gradients, components = component_gradients(logits, by, rare_mask, extra, identity["competitor_class"], parameters)
                if any(p.grad is not None for p in parameters):
                    raise ValueError("Auxiliary autograd.grad polluted actual parameter.grad")
                pilot.sync(device)
                component_seconds = time.perf_counter() - start_component
                audit["costs"]["component_seconds"] += component_seconds
                audit["costs"]["component_backward_calls"] += 3
                audit["costs"]["component_backward_rows"] += 3 * len(selected)
            # Preserve the original backward, including its floating-point order.
            loss.backward()
            pilot.sync(device)
            before_clone_time = time.perf_counter()
            clone_seconds = 0.0
            if measure:
                full = tuple(p.grad.detach().clone() for p in parameters)
                event = {"pre_step": step - 1, "update_step": step, "batch_rows": len(selected),
                         "rare_rows": int(rare_mask.sum()), "competitor_rows": int((by == identity["competitor_class"]).sum()),
                         "selected_indices_sha256": pilot.array_hash(selected), "components": components,
                         "gradient_blocks": gradient_blocks(names, gradients, full)}
                event["components"].update(original_base_ce=scalar(base), original_total_loss=scalar(loss),
                                            component_sum_residual=scalar(loss) - components["rare_component_loss"] - components["nonrare_component_loss"])
                event["local_only_prestate"] = save_prestate(output, identity, step - 1, selected, model, optimizer, audit)
                actual_before = tuple(p.detach().clone() for p in parameters)
                counterfactual, full_delta, full_next_optimizer = one_step_clones(model, optimizer, full, gradients, names, groups, config, device, audit)
                event.update(counterfactual)
                audit["events"].append(event)
                clone_seconds = time.perf_counter() - before_clone_time
            optimizer.step()
            pilot.sync(device)
            elapsed = time.perf_counter() - began
            audit["costs"]["instrumented_real_optimization_seconds"] += elapsed
            book["costs"]["real_optimization_seconds"] += max(0.0, elapsed - component_seconds - clone_seconds)
            if measure:
                if not all(torch.equal(p.detach() - old_value, delta) for p, old_value, delta in zip(parameters, actual_before, full_delta)):
                    raise ValueError("Full clone displacement differs from actual Adam update")
                audit["full_clone_actual_update_exact_matches"] += 1
                if not exact_tree(optimizer.state_dict(), full_next_optimizer):
                    raise ValueError("Full clone optimizer nextstate differs from real Adam")
                audit["full_clone_actual_optimizer_exact_matches"] += 1
                del gradients, full, full_delta, full_next_optimizer, actual_before
            book["costs"]["real_forward_rows"] += len(selected)
            book["costs"]["real_backward_rows"] += len(selected)
            book["chains"].update(expected)
            book["chains"]["actual_indices"] = pilot.advance(book["chains"]["actual_indices"], selected)
            book["chains"]["actual_class_counts"] = pilot.advance(book["chains"]["actual_class_counts"], actual_counts)
            book["class_exposures"] = (np.asarray(book["class_exposures"]) + counts).tolist()
            book["current_rows"] += len(current)
            book["replay_rows"] += len(selected) - len(current)
            book["rare_rows"] += proxy["rare_rows"]
            book["rare_batches"] += int(proxy["rare_rows"] > 0)
            book["loss_sums"]["base_ce"] += float(base.detach())
            for key in ("ce", "margin", "ce_logit_gradient_l2", "margin_logit_gradient_l2"):
                book["loss_sums"]["extra_" + key] += proxy[key]
            book["step"] = step
            append_training_trace(model, combined_probe, raw, device, audit)
            if device.type == "cuda":
                book["peak_cuda_allocated_bytes"] = max(book["peak_cuda_allocated_bytes"], torch.cuda.max_memory_allocated(device))
                book["peak_cuda_reserved_bytes"] = max(book["peak_cuda_reserved_bytes"], torch.cuda.max_memory_reserved(device))
            if step % 50 == 0 or step == identity["steps"]:
                trace, elapsed_trace, rows = pilot.training_trace(model, x, y, ncurrent, rare, normal, step, device)
                trace["latest_batch_loss_and_proxy"] = {"base_ce": float(base.detach()), **proxy}
                book["traces"].append(trace)
                book["costs"]["trace_seconds"] += elapsed_trace
                book["costs"]["trace_forward_rows"] += rows
                save_began = time.perf_counter()
                save_state(progress, model, optimizer, identity, prefix, book, schedule, audit, raw)
                book["costs"]["serialization_seconds"] += time.perf_counter() - save_began
                print(f"STEP_COMPLETE arm={identity['arm']} seed={seed} step={step}/{identity['steps']}", flush=True)
    expected_events = list(range(0, identity["steps"], 50))
    if (step != identity["steps"] or schedule["optimizer_steps"] != identity["steps"] or book["step"] != step
            or [e["pre_step"] for e in audit["events"]] != expected_events
            or audit["full_clone_actual_update_exact_matches"] != len(expected_events)
            or audit["full_clone_actual_optimizer_exact_matches"] != len(expected_events)
            or len(raw["predictions"]) != step + 1):
        raise ValueError("Real-step/clone-grid/raw-trace completeness mismatch")
    if book["chains"]["class_counts"] != book["chains"]["actual_class_counts"]:
        raise ValueError("Actual class-count schedule changed")
    for event in audit["events"]:
        metadata = event["local_only_prestate"]
        if digest(output / metadata["path"]) != metadata["sha256"]:
            raise ValueError("Durable prestate payload changed or is missing")
    pilot.sync(device)
    eval_began = time.perf_counter()
    evaluation_rows = np.arange(len(data.y_test))
    if identity["smoke"]:
        evaluation_rows = np.concatenate([np.flatnonzero(data.y_test == c)[:64] for c in range(nclasses)])
    metrics = evaluate(model, data.x_test[evaluation_rows], data.y_test[evaluation_rows], tuple(range(nclasses)), data, config, device)
    pilot.sync(device)
    evaluation_seconds = time.perf_counter() - eval_began
    exact = exact_comparison(model, optimizer, prefix, book, schedule, reference, reference_state, metrics, identity["smoke"])
    # Recheck immutable originals before producing a completed result.
    if digest(ROOT / identity["reference"]["result_path"]) != identity["reference"]["result_sha256"] or digest(ROOT / identity["reference"]["checkpoint_path"]) != identity["reference"]["checkpoint_sha256"]:
        raise ValueError("Reference originals changed during instrumented run")
    trace_meta = {"current_rare_train": trace_summary(raw, y[rare_ids], nclasses, 0, len(rare_ids)),
                  "old_rare_resident": trace_summary(raw, y[old_ids], nclasses, len(rare_ids), len(trajectory_ids))}
    raw_began = time.perf_counter()
    arrays = {**{k: v.numpy() for k, v in raw_tensors(raw).items()},
              "current_prepared_row_ids": original_rows[rare_ids], "resident_prefix_indices": old_ids - ncurrent,
              "group_offsets": np.asarray([0, len(rare_ids), len(trajectory_ids)], dtype=np.int64),
              "true_class_ids": y[trajectory_ids], "steps": np.arange(step + 1, dtype=np.int64)}
    reused_final = final_path.exists()
    if reused_final and raw_path.exists():
        with np.load(raw_path, allow_pickle=False) as archived:
            if set(archived.files) != set(arrays) or any(not np.array_equal(archived[key], value) for key, value in arrays.items()):
                raise ValueError("Existing local NPZ differs from durable final checkpoint trace")
    else:
        temporary = raw_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(raw_path)
    raw_finalization_seconds = time.perf_counter() - raw_began
    # On final-PT recovery, its audit is already durable: re-verifying or
    # reconstructing local NPZ must not silently mutate the result's audit.
    if not reused_final:
        audit["costs"]["raw_serialization_seconds"] += raw_finalization_seconds
    final_serialization_seconds = 0.0
    if not final_path.exists():
        final_began = time.perf_counter()
        save_state(final_path, model, optimizer, identity, prefix, book, schedule, audit, raw)
        final_serialization_seconds = time.perf_counter() - final_began
    result = {"schema_version": 1, "completed": True, "identity": identity, "task_index": 3,
              "class_order": data.class_order, "tasks": data.tasks, "finished_utc": pilot.now(),
              "command": identity["command"], "log_path": identity["log_path"],
              "final_checkpoint": {"path": final_path.relative_to(output).as_posix(), "sha256": digest(final_path)},
              "reference_exact_reproduction": exact, "reference_actual_payload_verified": True,
              "training_schedule": schedule, "audit": book, "interference_audit": audit,
              "local_only_prestates": [event["local_only_prestate"] for event in audit["events"]],
              "training_trace_summaries": trace_meta,
              "local_only_trace": {"path": raw_path.name, "sha256": digest(raw_path), "publish": False,
                                   "current_prepared_row_ids_sha256": pilot.array_hash(original_rows[rare_ids]),
                                   "resident_prefix_indices_sha256": pilot.array_hash(old_ids - ncurrent),
                                   "group_order": ["current_rare_train", "old_rare_resident"],
                                   "group_offsets": [0, len(rare_ids), len(trajectory_ids)],
                                   "array_shape": [step + 1, len(trajectory_ids)]},
              "id": metrics, "evaluation_seconds": evaluation_seconds,
              "finalization_cost": {"final_checkpoint_reused": reused_final,
                                    "raw_write_or_verification_seconds_this_attempt": raw_finalization_seconds,
                                    "final_checkpoint_serialization_seconds_this_attempt": final_serialization_seconds,
                                    "note": "Raw time is already included in audit.raw_serialization_seconds on initial completion; do not add it twice. Resumed finalization overhead is this attempt only."},
              "memory_rows": len(y) - ncurrent, "memory_unchanged": True, "old_pool_revisits": 0,
              "evaluation_role": "smoke_non_scientific" if identity["smoke"] else "already_exposed_ID_test_exploratory_only",
              "policies": POLICIES}
    require_finite(result)
    save_json(final_path.with_suffix(".json"), result)
    save_json(result_path, result)
    print(f"CELL_COMPLETE {identity['dataset']} {identity['arm']} seed={seed}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=(1, 2), default=[1, 2])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--prefix-root", type=Path, default=ROOT / "ncd/results/problem_scout/section22/steps_2000_v1")
    parser.add_argument("--reference-root", type=Path, default=ROOT / "ncd/results/problem_scout/rare_boundary/pilot_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--smoke", action="store_true", help="Non-scientific two-step capped-current/test smoke, one pre-step0 clone event")
    args = parser.parse_args()
    if any(len(v) != len(set(v)) for v in (args.datasets, args.seeds, args.arms)):
        parser.error("Duplicate cells are forbidden")
    output, prefix_root, reference_root = args.output_dir.resolve(), args.prefix_root.resolve(), args.reference_root.resolve()
    if output == reference_root or output == prefix_root or output.is_relative_to(reference_root) or output.is_relative_to(prefix_root):
        raise ValueError("New output must not overwrite or live inside historical reference/prefix outputs")
    config_path = ROOT / "ncd/configs/problem_scout_section21.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(epochs_per_task=0 if args.smoke else 8, min_steps_per_task=2 if args.smoke else 2000, batch_size=256, smoke=args.smoke)
    if args.device:
        config["device"] = args.device
    if args.data_root:
        config["problem_scout_data_root"] = path_label(args.data_root)
    if not os.environ.get("NCD_SOURCE_COMMIT"):
        raise ValueError("Explicit NCD_SOURCE_COMMIT is required")
    commit, device = source_commit(ROOT), torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(config["cpu_threads"])
    command = subprocess.list2cmdline(["python", "-I", "ncd/run_family_interference_probe.py", *sys.argv[1:]])
    for path in (args.data_root, args.output_dir, args.prefix_root, args.reference_root, ROOT):
        if path is not None and path.is_absolute():
            command = command.replace(str(path), path_label(path)).replace(path.as_posix(), path_label(path))
    sources = {p: digest(ROOT / p) for p in [*FROZEN_SOURCES, "ncd/run_rare_boundary_pilot.py", "ncd/run_family_interference_probe.py"]}
    matrix_identity = {"git_commit": commit, "source_hashes": sources, "config_sha256": digest(config_path),
                       "effective_config": config, "datasets": args.datasets, "seeds": args.seeds, "arms": args.arms,
                       "smoke": args.smoke, "policies": POLICIES, "command": command, "log_path": path_label(output / "run.log")}
    output.mkdir(parents=True, exist_ok=True)
    cells, summary_path = [], output / "result.json"
    with cell_lock(output / ".execution.lock"), (output / "run.log").open("a", encoding="utf-8") as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)), pilot.record_matrix_failure(summary_path, matrix_identity, cells):
        if summary_path.exists() and json.loads(summary_path.read_text(encoding="utf-8"))["identity"] != matrix_identity:
            raise ValueError("Existing output identity differs; never overwrite a different run")
        pilot.matrix_summary(summary_path, matrix_identity, cells, "running")
        for dataset in args.datasets:
            data, xshift, yshift, _ = prepare_setting(dataset, config)
            hashes = binding(data, xshift, yshift)
            del xshift, yshift
            if len(data.tasks) != 4:
                raise ValueError("Exactly four-task datasets required")
            original_rows = np.flatnonzero(np.isin(data.y_train, data.tasks[3]))
            counts = {c: int(np.sum(data.y_train[original_rows] == c)) for c in data.tasks[3]}
            rare = [c for c, count in counts.items() if count < 200 and data.class_order[c] != "Normal"]
            if len(rare) != 1:
                raise ValueError("This frozen diagnostic requires one current rare class")
            if args.smoke:
                original_rows = np.concatenate([original_rows[data.y_train[original_rows] == c][:64] for c in data.tasks[3]])
                original_rows.sort()
            cx, cy = data.x_train[original_rows], data.y_train[original_rows]
            for seed in args.seeds:
                prefix, mx, my, prefix_ref = pilot.load_start(prefix_root, dataset, seed, hashes, data)
                x, y = np.concatenate([cx, mx]), np.concatenate([cy, my])
                target = max(config["epochs_per_task"] * ((len(y) + 255) // 256), config["min_steps_per_task"])
                if target != (2 if args.smoke else 2000):
                    raise ValueError("Exact real-step budget mismatch")
                for arm in args.arms:
                    reference, reference_state, reference_ref = read_reference(reference_root, dataset, seed, arm, data, hashes, prefix_ref, config)
                    cell_dir = output / dataset.lower().replace("-", "_") / f"{arm}_seed{seed}"
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    identity = {"dataset": dataset, "seed": seed, "arm": arm, "smoke": args.smoke,
                                "git_commit": commit, "source_hashes": sources, "config_sha256": digest(config_path),
                                "effective_config": config, "class_order": data.class_order, "tasks": data.tasks,
                                "data_binding": hashes, "start_checkpoint": prefix_ref, "reference": reference_ref,
                                "rare_classes": rare, "competitor_class": data.class_order.index(FAMILY[dataset]["competitor"]),
                                "old_probe_class": data.class_order.index(FAMILY[dataset]["old_probe"]),
                                "current_original_train_counts": {str(c): count for c, count in counts.items()},
                                "current_original_row_indices_sha256": pilot.array_hash(original_rows),
                                "allowed_pool_x_sha256": pilot.array_hash(x), "allowed_pool_y_sha256": pilot.array_hash(y),
                                "steps": target, "pre_step_grid": list(range(0, target, 50)), "conditions": list(CONDITIONS),
                                "candidate_per_class": 100, "extra_weight": 1.0, "policies": POLICIES,
                                "command": command, "log_path": path_label(cell_dir / "run.log"),
                                "runtime_versions": {"python": sys.version.split()[0], "torch": str(torch.__version__), "numpy": np.__version__, "cuda": torch.version.cuda},
                                "evaluation_role": "smoke_non_scientific" if args.smoke else "already_exposed_ID_test_exploratory_only"}
                    event = {"event": "started", "utc": pilot.now(), "pid": os.getpid(), "dataset": dataset, "seed": seed, "arm": arm}
                    pilot.matrix_summary(summary_path, matrix_identity, cells, "running", {**event, "log_path": path_label(cell_dir / "run.log")})
                    pilot.append_event(output / "driver_attempts.jsonl", event)
                    began = time.perf_counter()
                    try:
                        with cell_lock(cell_dir / ".execution.lock"), (cell_dir / "run.log").open("a", encoding="utf-8") as cell_log, redirect_stdout(Tee(sys.stdout, cell_log)), redirect_stderr(Tee(sys.stderr, cell_log)):
                            result = run_cell(data, x, y, len(cy), original_rows, prefix, identity, cell_dir, config, device, reference, reference_state)
                        pilot.append_event(output / "driver_attempts.jsonl", {**event, "event": "completed", "utc": pilot.now(), "exit_code": 0, "wall_seconds": time.perf_counter() - began})
                    except Exception:
                        pilot.append_event(output / "driver_attempts.jsonl", {**event, "event": "failed", "utc": pilot.now(), "exit_code": 1, "wall_seconds": time.perf_counter() - began})
                        traceback.print_exc()
                        raise
                    cells.append({"dataset": dataset, "seed": seed, "arm": arm, "result": (cell_dir / "result.json").relative_to(output).as_posix(), "result_sha256": digest(cell_dir / "result.json")})
                    pilot.matrix_summary(summary_path, matrix_identity, cells, "complete" if len(cells) == len(args.datasets) * len(args.seeds) * len(args.arms) else "running")
                    del reference, reference_state
                del prefix, mx, my, x, y
            del data, cx, cy
        print(f"MATRIX_COMPLETE cells={len(cells)} smoke={args.smoke}", flush=True)


if __name__ == "__main__":
    main()
