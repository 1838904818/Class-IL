"""Exploratory last-task fork: quota-matched Replay/MIR and rare-new protection.

Historical training sources and checkpoints are read-only. This is an Adam,
class-quota-matched MIR adaptation, not a reproduction of the original SGD
MIR. Seeds 1/2 and their ID tests have already been exposed. There is no tuning,
early stopping, test-driven selection, or independent-confirmation claim.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from ncd.crr_benchmarks import _class_balanced_cross_entropy, training_epoch_batches
from ncd.model import MLP
from ncd.prepare_problem_scout_data import prepare_setting
from ncd.run_benign_coverage import SOURCE_PATHS as FROZEN_SOURCES, cell_lock
from ncd.run_problem_scout import (
    Tee, binding, canonical_hash, checkpoint_load, checkpoint_save, digest,
    evaluate, path_label, require_finite, save_json, seed_everything,
)
from ncd.source_commit import source_commit

ARMS = ("random", "mir", "mir_rare_ce", "mir_rare_margin")
DATASETS = ("NSL-KDD", "CIC-IDS-2017")
EMPTY_HASH = hashlib.sha256(b"").hexdigest()
POLICIES = {
    "scope": "Exploratory task3 new-class learning diagnostic; not old-class forgetting, novelty, or completion of RQ1/RQ3.",
    "start": "One verified historical S2000 Replay checkpoint002 per dataset/seed; model and resident memory identical across arms. Original prefix provenance is retained, not relabelled as new-source training.",
    "access": "Only task3 current training rows and checkpoint002 resident memory are optimizer/retrieval inputs. No old training pool revisit, new memory selection, hidden probe buffer, or holdout/test retrieval.",
    "schedule": "Historical current+memory shuffled index schedule, batch256, eight epochs with minimum2000 steps; full pilot requires exactly2000 steps. Current IDs and per-step per-class quotas are preserved in every arm.",
    "mir": "Classwise candidate set contains baseline random replay IDs plus random other resident IDs; size=min(class memory,max(100,class quota)). Rank raw per-row CE increase after a copied current-only minibatch-class-balanced-CE Adam step with copied optimizer state. No real current-only step. This is quota-matched Adam-lookahead MIR, not original SGD MIR.",
    "rare": "Current-task non-Normal classes with original prepared training count<200. Same current minibatch rare rows; no oversampling. Extra weight1, no search. Empty rare minibatches add zero.",
    "arms": "random: original random Replay; mir: protection-off ablation; mir_rare_ce: MIR+mean rare-row CE; mir_rare_margin: MIR+mean softplus(zNormal-zTrue). Base loss is unchanged minibatch-class-balanced CE.",
    "scale": "Equal coefficient does not imply equal loss/parameter-gradient scale. Analytic extra-loss logit-gradient L2 proxies are recorded; they are not parameter gradients or a causal scale-matching guarantee.",
    "trace": "Step0 and every50 steps: fixed current rare rows and fixed resident memory only; diagnostic forwards never affect selection. No test feedback. Test inference occurs only at the final checkpoint, with an explicit exploratory label.",
    "cost": "Equal real steps, batch/class counts, current exposure and memory capacity, NOT equal FLOPs or total forward/backward sample touches. Virtual optimization, candidate scoring, real optimization, trace, serialization and final evaluation are reported separately.",
    "resume": "Same-identity progress checkpoints every50 steps include real Adam state and audit chains. Resume replays only the deterministic index schedule to the saved boundary; successful real steps are not repeated. Work after the last durable boundary is failed-attempt overhead, not a new scientific replicate.",
    "reference": "https://github.com/optimass/Maximally_Interfered_Retrieval/blob/35eda78bcdd35025b16b0b1039a926d34aad4851/mir.py",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def array_hash(array):
    value = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(np.asarray(value.shape, dtype="<i8").tobytes())
    h.update(value.tobytes())
    return h.hexdigest()


def advance(chain, values):
    values = np.asarray(values, dtype="<i8").reshape(-1)
    return hashlib.sha256(bytes.fromhex(chain) + len(values).to_bytes(8, "little") + values.tobytes()).hexdigest()


def append_event(path, event):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def matrix_summary(path, identity, cells, status, current_cell=None, error=None):
    save_json(path, {"schema_version": 1, "identity": identity, "status": status,
                     "expected_cells": len(identity["datasets"]) * len(identity["seeds"]) * len(identity["arms"]),
                     "completed_cells": len(cells), "cells": cells, "current_cell": current_cell,
                     "updated_utc": now(), "error": error})


@contextmanager
def record_matrix_failure(path, identity, cells):
    try:
        yield
    except Exception as error:
        prior = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        # An incompatible prior result must not be overwritten on a bad resume.
        if prior is None or prior.get("identity") == identity:
            matrix_summary(path, identity, cells, "failed", prior.get("current_cell") if prior else None,
                           {"type": type(error).__name__, "detail": "See retained local run.log and driver_attempts.jsonl; no automatic retry."})
        traceback.print_exc()
        raise


def load_start(prefix_root, dataset, seed, data_hashes, data):
    cell = prefix_root / dataset.lower().replace("-", "_") / f"replay_seed{seed}"
    result_path = cell / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    ident = result["identity"]
    if not result.get("completed") or len(result["checkpoints"]) != 4:
        raise ValueError("Prefix source cell is not a complete four-task result")
    if (ident["dataset"], ident["method"], ident["seed"]) != (dataset, "replay", seed):
        raise ValueError("Wrong historical Replay source cell")
    if ident["data_binding"] != data_hashes or ident["class_order"] != data.class_order or ident["tasks"] != data.tasks:
        raise ValueError("Prepared data/class schedule differs from historical prefix")
    expected = {"batch_size": 256, "hidden": 128, "learning_rate": .001,
                "buffer_per_class": 200, "epochs_per_task": 8,
                "min_steps_per_task": 2000, "smoke": False}
    if any(ident["effective_config"].get(k) != v for k, v in expected.items()):
        raise ValueError("Prefix is not the historical S2000 Replay configuration")
    cp = result["checkpoints"][2]
    path = cell / cp["checkpoint_file"]
    if path.name != "checkpoint_002.pt" or digest(path) != cp["checkpoint_sha256"]:
        raise ValueError("Prefix checkpoint SHA differs from result")
    if json.loads(path.with_suffix(".json").read_text(encoding="utf-8")) != cp:
        raise ValueError("Prefix checkpoint sidecar differs from result")
    if cp["identity_hash"] != canonical_hash(ident):
        raise ValueError("Prefix identity hash differs")
    state = checkpoint_load(path, ident)
    old = [c for task in data.tasks[:3] for c in task]
    if state["task_index"] != 2 or state["seen_classes"] != old:
        raise ValueError("Prefix is not immediately before final new classes")
    mx = np.concatenate([v.numpy() for v in state["buffer_x"]])
    my = np.concatenate([v.numpy() for v in state["buffer_y"]])
    if sorted(np.unique(my).tolist()) != old or any(int(np.sum(my == c)) != 200 for c in old):
        raise ValueError("Unexpected stored-memory class coverage/capacity")
    ref = {"path": path_label(path), "sha256": digest(path),
           "result_path": path_label(result_path), "result_sha256": digest(result_path),
           "source_commit": ident["git_commit"], "identity_hash": canonical_hash(ident),
           "source_hashes": ident["source_hashes"],
           "memory_x_sha256": array_hash(mx), "memory_y_sha256": array_hash(my),
           "memory_rows": len(my), "memory_bytes": mx.nbytes + my.nbytes}
    return state, mx, my, ref


def candidates_for_step(original, labels, ncurrent, old_by_class, seed, step):
    parts = []
    for c in sorted(old_by_class):
        chosen = original[(original >= ncurrent) & (labels[original] == c)]
        if len(chosen) == 0:
            continue
        eligible = old_by_class[c]
        size = min(len(eligible), max(100, len(chosen)))
        others = eligible[~np.isin(eligible, chosen)]
        rng = np.random.default_rng(np.random.SeedSequence([seed, 3, step, c, 27031]))
        added = rng.choice(others, size=size - len(chosen), replace=False)
        parts.append(np.sort(np.concatenate([chosen, added])).astype(np.int64))
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def retrieve(model, optimizer, x, y, original, candidates, ncurrent, device, config, costs):
    current = original[original < ncurrent]
    if len(current) == 0 or len(candidates) == 0:
        return original.copy()
    sync(device)
    began = time.perf_counter()
    future = copy.deepcopy(model)
    future_optimizer = torch.optim.Adam(future.parameters(), lr=config["learning_rate"])
    future_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    future_optimizer.zero_grad(set_to_none=True)
    cx = torch.from_numpy(x[current]).to(device)
    cy = torch.from_numpy(y[current]).to(device)
    _class_balanced_cross_entropy(future(cx), cy).backward()
    future_optimizer.step()
    sync(device)
    costs["virtual_adam_seconds"] += time.perf_counter() - began
    costs["virtual_steps"] += 1
    costs["virtual_forward_rows"] += len(current)
    costs["virtual_backward_rows"] += len(current)
    began = time.perf_counter()
    with torch.no_grad():
        qx = torch.from_numpy(x[candidates]).to(device)
        qy = torch.from_numpy(y[candidates]).to(device)
        before = F.cross_entropy(model(qx), qy, reduction="none")
        after = F.cross_entropy(future(qx), qy, reduction="none")
        scores = (after - before).cpu().numpy()
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite MIR scores")
    selected = original.copy()
    for c in np.unique(y[original[original >= ncurrent]]):
        slots = np.flatnonzero((original >= ncurrent) & (y[original] == c))
        available = np.flatnonzero(y[candidates] == c)
        rank = np.argsort(-scores[available], kind="stable")[:len(slots)]
        selected[slots] = candidates[available[rank]]
    sync(device)
    costs["candidate_scoring_seconds"] += time.perf_counter() - began
    costs["candidate_forward_rows"] += 2 * len(candidates)
    del future_optimizer, future
    return selected


def extra_losses(logits, labels, rare_mask, normal):
    if not torch.any(rare_mask):
        zero = logits.sum() * 0.0
        return zero, zero, {"rare_rows": 0, "ce": 0.0, "margin": 0.0,
                           "ce_logit_gradient_l2": 0.0, "margin_logit_gradient_l2": 0.0}
    z, target = logits[rare_mask], labels[rare_mask]
    rows = torch.arange(len(target), device=z.device)
    delta = z[:, normal] - z[rows, target]
    ce, margin = F.cross_entropy(z, target), F.softplus(delta).mean()
    with torch.no_grad():
        gradient = z.softmax(dim=1)
        gradient[rows, target] -= 1.0
        ce_norm = (gradient / len(target)).norm().item()
        margin_norm = (2.0 * (delta.sigmoid() / len(target)).square().sum()).sqrt().item()
    return ce, margin, {"rare_rows": len(target), "ce": float(ce.detach()), "margin": float(margin.detach()),
                       "ce_logit_gradient_l2": ce_norm, "margin_logit_gradient_l2": margin_norm}


def training_trace(model, x, y, ncurrent, rare_classes, normal, step, device):
    selected = np.flatnonzero((np.arange(len(y)) >= ncurrent) | np.isin(y, rare_classes))
    sync(device)
    began = time.perf_counter()
    values = []
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(selected), 2048):
            values.append(model(torch.from_numpy(x[selected[begin:begin + 2048]]).to(device)).cpu())
    z = torch.cat(values)
    truth = torch.from_numpy(y[selected])
    pred = z.argmax(dim=1)
    per_class = []
    for c in sorted(np.unique(y[selected]).tolist()):
        mask = truth == c
        zz = z[mask]
        delta = zz[:, c] - zz[:, normal]
        row = {"class_id": c, "role": "current_rare_train" if c in rare_classes else "resident_memory",
               "rows": int(mask.sum()), "recall": float((pred[mask] == c).float().mean()),
               "predicted_normal_fraction": float((pred[mask] == normal).float().mean()),
               "true_minus_normal_mean": float(delta.mean()), "true_minus_normal_p05": float(torch.quantile(delta, .05)),
               "prediction_counts": torch.bincount(pred[mask], minlength=z.shape[1]).tolist()}
        if c == normal:
            attack_columns = [i for i in range(z.shape[1]) if i != normal]
            row["normal_minus_max_attack_mean"] = float((zz[:, normal] - zz[:, attack_columns].max(dim=1).values).mean())
        per_class.append(row)
    model.train()
    sync(device)
    return {"step": step, "role": "training_and_resident_memory_only", "per_class": per_class}, time.perf_counter() - began, len(selected)


def fresh_book(nclasses):
    return {"step": 0, "started_utc": now(), "traces": [],
            "chains": {k: EMPTY_HASH for k in ("baseline_indices", "current_indices", "candidate_indices", "class_counts", "actual_indices", "actual_class_counts")},
            "class_exposures": [0] * nclasses, "current_rows": 0, "replay_rows": 0,
            "rare_rows": 0, "rare_batches": 0,
            "loss_sums": {k: 0.0 for k in ("base_ce", "extra_ce", "extra_margin", "extra_ce_logit_gradient_l2", "extra_margin_logit_gradient_l2")},
            "costs": {k: 0.0 for k in ("candidate_build_seconds", "virtual_adam_seconds", "candidate_scoring_seconds", "real_optimization_seconds", "trace_seconds", "serialization_seconds")}
            | {k: 0 for k in ("virtual_steps", "virtual_forward_rows", "virtual_backward_rows", "candidate_forward_rows", "real_forward_rows", "real_backward_rows", "trace_forward_rows")},
            "peak_cuda_allocated_bytes": 0, "peak_cuda_reserved_bytes": 0}


def save_progress(path, model, optimizer, identity, prefix_state, book, schedule):
    state = {"identity": identity, "task_index": 3,
             "seen_classes": list(range(len(identity["class_order"]))),
             "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
             "optimizer_state": optimizer.state_dict(),
             "buffer_x": prefix_state["buffer_x"], "buffer_y": prefix_state["buffer_y"],
             "book": book, "training_schedule": schedule,
             "torch_rng_state": torch.get_rng_state(),
             "cuda_rng_states": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else []}
    checkpoint_save(path, state)


def run_arm(data, x, y, ncurrent, original_rows, prefix_state, identity, output, config, device):
    result_path = output / "result.json"
    progress = output / "checkpoints/progress.pt"
    final_path = output / "checkpoints/checkpoint_003.pt"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("identity") != identity or not result.get("completed"):
            raise ValueError("Completed pilot identity differs")
        if digest(output / result["final_checkpoint"]["path"]) != result["final_checkpoint"]["sha256"]:
            raise ValueError("Pilot final payload hash changed")
        if json.loads(final_path.with_suffix(".json").read_text(encoding="utf-8")) != result:
            raise ValueError("Pilot sidecar differs")
        print(f"CELL_RESUME {identity['dataset']} {identity['arm']} seed={identity['seed']}", flush=True)
        return result
    nclasses, seed = len(data.class_order), identity["seed"]
    task_seed = seed * 10000 + 3
    seed_everything(task_seed)
    model = MLP(x.shape[1], config["hidden"], nclasses).to(device)
    model.load_state_dict(prefix_state["model_state"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    book = fresh_book(nclasses)
    schedule = {}
    restoring = final_path if final_path.exists() else progress if progress.exists() else None
    if restoring:
        prior = checkpoint_load(restoring, identity)
        model.load_state_dict(prior["model_state"])
        optimizer.load_state_dict(prior["optimizer_state"])
        book = prior["book"]
        torch.set_rng_state(prior["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(prior["cuda_rng_states"])
        print(f"PROGRESS_RESUME step={book['step']} {identity['arm']}", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rare = identity["rare_classes"]
    normal = data.class_order.index("Normal")
    if not book["traces"]:
        trace, elapsed, rows = training_trace(model, x, y, ncurrent, rare, normal, 0, device)
        book["traces"].append(trace)
        book["costs"]["trace_seconds"] += elapsed
        book["costs"]["trace_forward_rows"] += rows
    old = {c: np.flatnonzero((np.arange(len(y)) >= ncurrent) & (y == c)) for c in np.unique(y[ncurrent:])}
    # The original loader's shuffle depends only on row count/seed, not features.
    index_x = np.arange(len(y), dtype=np.int64).reshape(-1, 1)
    expected = {k: EMPTY_HASH for k in ("baseline_indices", "current_indices", "candidate_indices", "class_counts")}
    resume_step = book["step"]
    step = 0
    for _, batches in training_epoch_batches(index_x, y, 256, task_seed, config["epochs_per_task"], config["min_steps_per_task"], schedule):
        for tensor_indices, _ in batches:
            step += 1
            original = tensor_indices.numpy().reshape(-1)
            began = time.perf_counter()
            candidates = candidates_for_step(original, y, ncurrent, old, seed, step)
            candidate_time = time.perf_counter() - began
            current = original[original < ncurrent]
            counts = np.bincount(y[original], minlength=nclasses)
            values = {"baseline_indices": original, "current_indices": current,
                      "candidate_indices": candidates, "class_counts": counts}
            for key, value in values.items():
                expected[key] = advance(expected[key], value)
            if step <= resume_step:
                if step == resume_step and any(expected[k] != book["chains"][k] for k in expected):
                    raise ValueError("Resumed index/candidate/class schedule changed")
                continue
            book["costs"]["candidate_build_seconds"] += candidate_time
            selected = original.copy() if identity["arm"] == "random" else retrieve(model, optimizer, x, y, original, candidates, ncurrent, device, config, book["costs"])
            if not np.array_equal(selected[selected < ncurrent], current):
                raise ValueError("Current sample exposure changed")
            actual_counts = np.bincount(y[selected], minlength=nclasses)
            if not np.array_equal(counts, actual_counts):
                raise ValueError("Replay retrieval changed minibatch class quotas/CE weights")
            if len(np.unique(selected)) != len(selected):
                raise ValueError("Retrieval duplicated a row within a minibatch")
            sync(device)
            began = time.perf_counter()
            bx = torch.from_numpy(x[selected]).to(device)
            by = torch.from_numpy(y[selected]).to(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx)
            base = _class_balanced_cross_entropy(logits, by)
            rare_mask = torch.from_numpy((selected < ncurrent) & np.isin(y[selected], rare)).to(device)
            ce, margin, proxy = extra_losses(logits, by, rare_mask, normal)
            extra = ce if identity["arm"] == "mir_rare_ce" else margin if identity["arm"] == "mir_rare_margin" else logits.sum() * 0.0
            loss = base + extra
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite pilot objective")
            loss.backward()
            optimizer.step()
            sync(device)
            book["costs"]["real_optimization_seconds"] += time.perf_counter() - began
            book["costs"]["real_forward_rows"] += len(selected)
            book["costs"]["real_backward_rows"] += len(selected)
            book["chains"].update(expected)
            book["chains"]["actual_indices"] = advance(book["chains"]["actual_indices"], selected)
            book["chains"]["actual_class_counts"] = advance(book["chains"]["actual_class_counts"], actual_counts)
            book["class_exposures"] = (np.asarray(book["class_exposures"]) + counts).tolist()
            book["current_rows"] += len(current)
            book["replay_rows"] += len(selected) - len(current)
            book["rare_rows"] += proxy["rare_rows"]
            book["rare_batches"] += int(proxy["rare_rows"] > 0)
            book["loss_sums"]["base_ce"] += float(base.detach())
            for key in ("ce", "margin", "ce_logit_gradient_l2", "margin_logit_gradient_l2"):
                book["loss_sums"]["extra_" + key] += proxy[key]
            book["step"] = step
            if device.type == "cuda":
                book["peak_cuda_allocated_bytes"] = max(book["peak_cuda_allocated_bytes"], torch.cuda.max_memory_allocated(device))
                book["peak_cuda_reserved_bytes"] = max(book["peak_cuda_reserved_bytes"], torch.cuda.max_memory_reserved(device))
            if step % 50 == 0 or step == identity["steps"]:
                trace, elapsed, rows = training_trace(model, x, y, ncurrent, rare, normal, step, device)
                trace["latest_batch_loss_and_proxy"] = {"base_ce": float(base.detach()), **proxy}
                book["traces"].append(trace)
                book["costs"]["trace_seconds"] += elapsed
                book["costs"]["trace_forward_rows"] += rows
                started_save = time.perf_counter()
                save_progress(progress, model, optimizer, identity, prefix_state, book, schedule)
                book["costs"]["serialization_seconds"] += time.perf_counter() - started_save
                print(f"STEP_COMPLETE arm={identity['arm']} seed={seed} step={step}/{identity['steps']} rare_rows={book['rare_rows']}", flush=True)
    if step != identity["steps"] or schedule["optimizer_steps"] != identity["steps"] or book["step"] != step:
        raise ValueError("Pilot optimizer-step budget mismatch")
    if book["chains"]["class_counts"] != book["chains"]["actual_class_counts"]:
        raise ValueError("Actual class schedule hash differs from baseline")
    if identity["arm"] == "random" and book["chains"]["actual_indices"] != book["chains"]["baseline_indices"]:
        raise ValueError("Random replay did not preserve the historical index schedule")
    if not final_path.exists():
        save_progress(final_path, model, optimizer, identity, prefix_state, book, schedule)
    sync(device)
    began = time.perf_counter()
    evaluation_rows = np.arange(len(data.y_test))
    if identity["smoke"]:
        evaluation_rows = np.concatenate([np.flatnonzero(data.y_test == c)[:64] for c in range(nclasses)])
    final_metrics = evaluate(model, data.x_test[evaluation_rows], data.y_test[evaluation_rows], tuple(range(nclasses)), data, config, device)
    sync(device)
    evaluation_seconds = time.perf_counter() - began
    if final_metrics["evaluated_rows"] != len(evaluation_rows):
        raise ValueError("Final test support mismatch")
    result = {"schema_version": 1, "completed": True, "identity": identity,
              "class_order": data.class_order, "tasks": data.tasks,
              "command": identity["command"], "log_path": identity["log_path"],
              "finished_utc": now(), "task_index": 3,
              "final_checkpoint": {"path": final_path.relative_to(output).as_posix(), "sha256": digest(final_path)},
              "training_schedule": schedule, "audit": book,
              "current_original_row_indices_sha256": array_hash(original_rows),
              "memory_rows": len(y) - ncurrent, "memory_unchanged": True, "old_pool_revisits": 0,
              "id": final_metrics, "evaluation_seconds": evaluation_seconds,
              "evaluation_role": "smoke_non_scientific" if identity["smoke"] else "already_exposed_ID_test_exploratory_only",
              "policies": POLICIES}
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--smoke", action="store_true", help="Non-scientific two-step check; current/test rows capped64/class, original prefix memory retained")
    args = parser.parse_args()
    if any(len(v) != len(set(v)) for v in (args.datasets, args.seeds, args.arms)):
        parser.error("Duplicate datasets/seeds/arms are forbidden")
    config_path = ROOT / "ncd/configs/problem_scout_section21.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(epochs_per_task=0 if args.smoke else 8, min_steps_per_task=2 if args.smoke else 2000, batch_size=256, smoke=args.smoke)
    if args.device:
        config["device"] = args.device
    if args.data_root:
        config["problem_scout_data_root"] = path_label(args.data_root)
    if not os.environ.get("NCD_SOURCE_COMMIT"):
        raise ValueError("Explicit NCD_SOURCE_COMMIT is required for the new entry; never reuse historical training commit implicitly")
    commit = source_commit(ROOT)
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(config["cpu_threads"])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    command = subprocess.list2cmdline(["python", "-I", "ncd/run_rare_boundary_pilot.py", *sys.argv[1:]])
    for path in (args.data_root, args.output_dir, args.prefix_root, ROOT):
        if path is not None and path.is_absolute():
            command = command.replace(str(path), path_label(path)).replace(path.as_posix(), path_label(path))
    sources = {p: digest(ROOT / p) for p in [*FROZEN_SOURCES, "ncd/run_rare_boundary_pilot.py"]}
    matrix_identity = {"git_commit": commit, "source_hashes": sources, "config_sha256": digest(config_path),
                       "effective_config": config, "datasets": args.datasets, "seeds": args.seeds,
                       "arms": args.arms, "smoke": args.smoke, "policies": POLICIES,
                       "command": command, "log_path": path_label(output / "run.log")}
    summary_path = output / "result.json"
    cells = []
    with cell_lock(output / ".execution.lock"), (output / "run.log").open("a", encoding="utf-8") as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)), record_matrix_failure(summary_path, matrix_identity, cells):
        if summary_path.exists() and json.loads(summary_path.read_text(encoding="utf-8"))["identity"] != matrix_identity:
            raise ValueError("Matrix source/config/grid differs; preserve output")
        matrix_summary(summary_path, matrix_identity, cells, "running")
        for dataset in args.datasets:
            data, xshift, yshift, _ = prepare_setting(dataset, config)
            hashes = binding(data, xshift, yshift)
            del xshift, yshift
            if len(data.tasks) != 4:
                raise ValueError("Pilot is restricted to four-task datasets")
            original_rows = np.flatnonzero(np.isin(data.y_train, data.tasks[3]))
            counts = {c: int(np.sum(data.y_train[original_rows] == c)) for c in data.tasks[3]}
            rare = [c for c, n in counts.items() if n < 200 and data.class_order[c] != "Normal"]
            if not rare:
                raise ValueError("No current rare attack class under the fixed count rule")
            if args.smoke:
                original_rows = np.concatenate([original_rows[data.y_train[original_rows] == c][:64] for c in data.tasks[3]])
                original_rows.sort()
            cx, cy = data.x_train[original_rows], data.y_train[original_rows]
            for seed in args.seeds:
                start, mx, my, ref = load_start(args.prefix_root.resolve(), dataset, seed, hashes, data)
                x, y = np.concatenate([cx, mx]), np.concatenate([cy, my])
                target = max(config["epochs_per_task"] * ((len(y) + 255) // 256), config["min_steps_per_task"])
                if target != (2 if args.smoke else 2000):
                    raise ValueError("The specified task does not meet the exact fixed step budget")
                expected_schedule = None
                for arm in args.arms:
                    cell_dir = output / dataset.lower().replace("-", "_") / f"{arm}_seed{seed}"
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    identity = {"dataset": dataset, "seed": seed, "arm": arm, "smoke": args.smoke,
                                "git_commit": commit, "source_hashes": sources, "config_sha256": digest(config_path),
                                "effective_config": config, "class_order": data.class_order, "tasks": data.tasks,
                                "data_binding": hashes, "start_checkpoint": ref,
                                "rare_classes": rare, "current_original_train_counts": {str(c): n for c, n in counts.items()},
                                "current_original_row_indices_sha256": array_hash(original_rows),
                                "allowed_pool_x_sha256": array_hash(x), "allowed_pool_y_sha256": array_hash(y),
                                "steps": target, "candidate_per_class": 100, "extra_weight": 1.0,
                                "command": command, "log_path": path_label(cell_dir / "run.log"),
                                "runtime_versions": {"python": sys.version.split()[0], "torch": str(torch.__version__), "numpy": np.__version__, "cuda": torch.version.cuda},
                                "evaluation_role": "already_exposed_ID_test_exploratory_only", "policies": POLICIES}
                    event = {"event": "started", "utc": now(), "pid": os.getpid(), "dataset": dataset, "seed": seed, "arm": arm}
                    matrix_summary(summary_path, matrix_identity, cells, "running",
                                   {"dataset": dataset, "seed": seed, "arm": arm,
                                    "pid": os.getpid(), "log_path": path_label(cell_dir / "run.log"), "started_utc": event["utc"]})
                    append_event(output / "driver_attempts.jsonl", event)
                    started = time.perf_counter()
                    try:
                        with cell_lock(cell_dir / ".execution.lock"), (cell_dir / "run.log").open("a", encoding="utf-8") as cell_log, redirect_stdout(Tee(sys.stdout, cell_log)), redirect_stderr(Tee(sys.stderr, cell_log)):
                            result = run_arm(data, x, y, len(cy), original_rows, start, identity, cell_dir, config, device)
                        shared = {k: result["audit"]["chains"][k] for k in ("baseline_indices", "current_indices", "candidate_indices", "class_counts", "actual_class_counts")}
                        if expected_schedule is None:
                            expected_schedule = shared
                        elif shared != expected_schedule:
                            raise ValueError("Cross-arm current/candidate/class schedule hashes differ")
                        append_event(output / "driver_attempts.jsonl", event | {"event": "completed", "utc": now(), "exit_code": 0, "wall_seconds": time.perf_counter() - started})
                    except Exception:
                        append_event(output / "driver_attempts.jsonl", event | {"event": "failed", "utc": now(), "exit_code": 1, "wall_seconds": time.perf_counter() - started})
                        traceback.print_exc()
                        raise
                    cells.append({"dataset": dataset, "seed": seed, "arm": arm,
                                  "result": (cell_dir / "result.json").relative_to(output).as_posix(),
                                  "result_sha256": digest(cell_dir / "result.json")})
                    matrix_summary(summary_path, matrix_identity, cells,
                                   "complete" if len(cells) == len(args.datasets) * len(args.seeds) * len(args.arms) else "running")
                del start, mx, my, x, y
            del data, cx, cy
        print(f"MATRIX_COMPLETE cells={len(cells)} smoke={args.smoke}", flush=True)


if __name__ == "__main__":
    main()
