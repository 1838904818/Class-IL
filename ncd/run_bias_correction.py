"""One Section23 control cell. Historical Section21/22 outputs stay read-only.

Development selection uses only a training holdout and seeds 5/6. Each task
checkpoint includes the completed correction, so resumption cannot apply a
second WA/BiC/CBF pass. Corrections are adaptations, not original-paper replicas.
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
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from dataclasses import replace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from ncd.model import MLP
from ncd.crr_benchmarks import _train
from ncd.prepare_problem_scout_data import prepare_setting
from ncd.run_problem_scout import (
    ClassReservoir, Tee, binding, canonical_hash, checkpoint_load,
    checkpoint_save, digest, evaluate, path_label, require_finite, save_json,
    seed_everything, smoke_subset, train_derpp, update_memory,
)
from ncd.source_commit import source_commit
from ncd.section23_bias import apply_wa, fit_bic, balanced_finetune, train_ssil

SOURCE_PATHS = [
    "ncd/run_bias_correction.py", "ncd/section23_bias.py", "ncd/run_bias_control.py",
    "ncd/run_problem_scout.py", "ncd/crr_benchmarks.py", "ncd/model.py",
    "ncd/prepare_problem_scout_data.py", "ncd/train_cil.py", "ncd/data.py",
    "ncd/source_commit.py", "ncd/run_crr_real_sequence.py",
]


@contextmanager
def cell_lock(path):
    """Kernel-owned lock survives an orphan driver, and releases on child exit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("A process already owns this cell") from exc
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def validate_request(spec, dataset, base, correction, candidate, seed, phase):
    if dataset not in spec["datasets"] or base not in {"replay", spec["best_methods"][dataset]}:
        raise ValueError("Dataset/base is outside Section23")
    if correction not in spec["candidates"] or candidate not in range(len(spec["candidates"][correction])):
        raise ValueError("Correction/candidate outside fixed grid")
    allowed = spec["development_seeds"] if phase == "development" else spec["reporting_seeds"]
    if phase not in {"development", "formal"} or seed not in allowed:
        raise ValueError("Development/reporting seed roles cannot be mixed")
    if (spec["development_seeds"] != [5, 6] or spec["reporting_seeds"] != [1, 2, 3, 4, 42]
            or spec["epochs_per_task"] != 8 or spec["min_steps_per_task"] != 2000
            or spec["batch_size"] != 256):
        raise ValueError("Section23 seed/budget freeze differs")


def configuration_input_hashes(spec):
    base = json.loads((ROOT / spec["base_config"]).read_text(encoding="utf-8"))
    paths = {spec["base_config"], *base["lambda_files"].values()}
    return {p: digest(ROOT / p) for p in sorted(paths)}


def verify_formal_selection(path, spec, dataset, base, correction, candidate, config_path):
    if path is None:
        raise ValueError("Formal runs require the completed seed5/6 selection")
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("selection_seeds") != [5, 6]:
        raise ValueError("Selection used wrong seeds")
    provenance = selection["provenance"]
    if provenance["config_sha256"] != digest(config_path) or provenance["git_commit"] != source_commit(ROOT):
        raise ValueError("Selection source/config differs from formal training")
    if provenance.get("config_input_hashes") != configuration_input_hashes(spec):
        raise ValueError("Selected base configuration/lambda inputs changed")
    if provenance["source_hashes"] != {p: digest(ROOT / p) for p in SOURCE_PATHS}:
        raise ValueError("Selected training source was modified")
    groups = selection["selections"]
    expected = {(d, b, c) for d in spec["datasets"] for b in ("replay", spec["best_methods"][d]) for c in spec["candidates"]}
    keys = [(g["dataset"], g["base"], g["correction"]) for g in groups]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Selection lacks a complete development grid")
    chosen = groups[keys.index((dataset, base, correction))]
    if chosen["selected_candidate"] != candidate:
        raise ValueError("Formal candidate differs from development-selected candidate")
    candidates = chosen["candidates"]
    if [c["candidate"] for c in candidates] != list(range(len(spec["candidates"][correction]))):
        raise ValueError("Selection omitted a candidate")
    for row in candidates:
        proofs = row["development_results"]
        if [r["seed"] for r in proofs] != [5, 6]:
            raise ValueError("Candidate selection has wrong evidence seeds")
        values = []
        for proof in proofs:
            source = (path.parent / proof["result"]).resolve()
            if path.parent.resolve() not in source.parents or digest(source) != proof["result_sha256"]:
                raise ValueError("Development evidence changed or escaped root")
            cell = json.loads(source.read_text(encoding="utf-8"))
            ident = cell["identity"]
            if (not cell.get("completed") or ident["smoke"] or ident["phase"] != "development"
                    or ident["seed"] != proof["seed"] or ident["candidate"] != row["candidate"]
                    or ident["dataset"] != dataset or ident["base_method"] != base or ident["correction"] != correction
                    or ident["evaluation_role"] != "training_holdout_not_official_ID_test"):
                raise ValueError("Selection evidence roles differ")
            values.append(cell["checkpoints"][-1]["id"]["macro_f1"])
        if abs(sum(values) / 2 - row["mean_validation_macro_f1"]) > 1e-12:
            raise ValueError("Candidate mean differs from seed5/6 results")
    winner = max(candidates, key=lambda c: (c["mean_validation_macro_f1"], -c["candidate"]))
    if winner["candidate"] != candidate:
        raise ValueError("Selection does not maximize development metric")
    return {"path": path_label(path), "sha256": digest(path), "selected_candidate": candidate,
            "selection_seeds": [5, 6], "metric": selection["selection_metric"]}


def training_holdout(data, seed, fraction):
    """Remove validation from training, without ever consulting official test."""
    if seed not in {5, 6}:
        raise ValueError("Training holdout is for development seeds only")
    train, valid = [], []
    for c in range(len(data.class_order)):
        rows = np.flatnonzero(data.y_train == c)
        if len(rows) < 2:
            raise ValueError("Need at least two training rows per class for development")
        shuffled = np.random.default_rng(seed * 17011 + c).permutation(rows)
        count = min(len(rows) - 1, max(1, int(len(rows) * fraction)))
        valid.extend(shuffled[:count])
        train.extend(shuffled[count:])
    train, valid = np.asarray(sorted(train)), np.asarray(sorted(valid))
    split = {"role": "training_holdout_not_official_ID_test", "seed": seed,
             "fraction": fraction, "training_rows": len(train), "validation_rows": len(valid),
             "train_indices_sha256": hashlib.sha256(train.astype('<i8').tobytes()).hexdigest(),
             "validation_indices_sha256": hashlib.sha256(valid.astype('<i8').tobytes()).hexdigest(),
             "disjoint_indices": not bool(np.intersect1d(train, valid).size)}
    if not split["disjoint_indices"]:
        raise ValueError("Development split overlap")
    return replace(data, x_train=data.x_train[train], y_train=data.y_train[train],
                   x_test=data.x_train[valid], y_test=data.y_train[valid]), split


def reserve_calibration(x, y, seed, maximum, fraction):
    """Persistent BiC holdout is selected at class introduction, before training."""
    keep, held_x, held_y = [], [], []
    for c in np.unique(y):
        rows = np.flatnonzero(y == c)
        if len(rows) < 2:
            raise ValueError("BiC requires training and heldout rows for every class")
        n = min(maximum, len(rows) - 1, max(1, int(len(rows) * fraction)))
        shuffled = np.random.default_rng(seed * 13007 + int(c)).permutation(rows)
        held_x.append(x[shuffled[:n]].copy())
        held_y.append(y[shuffled[:n]].copy())
        keep.extend(shuffled[n:])
    keep = np.asarray(sorted(keep))
    return x[keep], y[keep], held_x, held_y


def balanced_calibration(xs, ys, seed):
    if not xs or len(xs) != len(ys):
        raise ValueError("Missing calibration classes")
    count = min(len(y) for y in ys)
    if count < 1:
        raise ValueError("Empty balanced calibration")
    bx, by = [], []
    for x, y in zip(xs, ys):
        idx = np.random.default_rng(seed + int(y[0]) * 73).choice(len(y), count, replace=False)
        bx.append(x[idx]); by.append(y[idx])
    return np.concatenate(bx), np.concatenate(by)


def old_memory_arrays(memory, bx, by, old_classes, input_dim):
    if memory is None:
        return (np.concatenate(bx), np.concatenate(by)) if bx else (
            np.empty((0, input_dim), np.float32), np.empty(0, np.int64))
    xx, yy = [], []
    for c in old_classes:
        count = min(memory.per_class, memory.observations[c])
        if count:
            xx.append(memory.x[c, :count].numpy().copy())
            yy.append(np.full(count, c, dtype=np.int64))
    return (np.concatenate(xx), np.concatenate(yy)) if xx else (
        np.empty((0, input_dim), np.float32), np.empty(0, np.int64))


def train_base(model, current_x, current_y, bx, by, memory, teacher, old, seen,
               method, config, regularization, task_seed, device, stats):
    if method == "derpp":
        train_derpp(model, current_x, current_y, memory, config, task_seed, device,
                    training_stats=stats)
        return len(current_y)
    x = np.concatenate([current_x, *bx]) if bx else current_x
    y = np.concatenate([current_y, *by]) if by else current_y
    _train(model, x, y, method=method, old_model=teacher,
           old_classes=old, seen_classes=seen, epochs=config["epochs_per_task"],
           batch_size=config["batch_size"], learning_rate=config["learning_rate"],
           seed=task_seed, regularization=regularization, temperature=config["temperature"],
           rcl_features=config["rcl_features"], rcl_temperature=config["rcl_temperature"],
           rcl_beta=config["rcl_beta"], rcl_prediction_weight=config["rcl_prediction_weight"],
           min_steps=config["min_steps_per_task"], training_stats=stats)
    return len(y)


def run_cell(data, metadata, identity, output, config, spec, params, device):
    final_path = output / "result.json"
    if final_path.exists():
        value = json.loads(final_path.read_text(encoding="utf-8"))
        if value.get("completed") is not True or value["identity"] != identity:
            raise ValueError("Completed cell identity differs")
        if len(value["checkpoints"]) != len(data.tasks):
            raise ValueError("Completed cell has missing tasks")
        for cp in value["checkpoints"]:
            if digest(output / cp["checkpoint_file"]) != cp["checkpoint_sha256"]:
                raise ValueError("Completed checkpoint changed")
            saved = (output / cp["checkpoint_file"]).with_suffix(".json")
            if json.loads(saved.read_text(encoding="utf-8")) != cp:
                raise ValueError("Completed metrics changed")
        print("CELL_RESUME", identity["dataset"], identity["method"], identity["seed"], flush=True)
        return value
    seed, base, correction = identity["seed"], identity["base_method"], identity["correction"]
    seed_everything(seed * 10000)
    model = MLP(data.x_train.shape[1], config["hidden"], len(data.class_order)).to(device)
    bx, by, hx, hy, records = [], [], [], [], []
    memory = ClassReservoir(config["buffer_per_class"], data.x_train.shape[1], len(data.class_order)) if base == "derpp" else None
    cumulative = extra_cumulative = 0.0
    old = ()
    for t, task in enumerate(data.tasks):
        seen = tuple(c for group in data.tasks[:t + 1] for c in group)
        task_seed = seed * 10000 + t
        seed_everything(task_seed)
        cp_path = output / "checkpoints" / f"checkpoint_{t:03d}.pt"
        metric_path = cp_path.with_suffix(".json")
        if cp_path.exists():
            state = checkpoint_load(cp_path, identity)
            if state["task_index"] != t or state["seen_classes"] != list(seen) or not state["correction_complete"]:
                raise ValueError("Checkpoint correction/task mismatch")
            model.load_state_dict(state["model_state"])
            bx = [a.numpy().copy() for a in state["buffer_x"]]
            by = [a.numpy().copy() for a in state["buffer_y"]]
            hx = [a.numpy().copy() for a in state["holdout_x"]]
            hy = [a.numpy().copy() for a in state["holdout_y"]]
            if memory is not None:
                memory = ClassReservoir(config["buffer_per_class"], data.x_train.shape[1], len(data.class_order), state["derpp_memory"])
            cumulative, extra_cumulative = state["training_seconds_cumulative"], state["correction_seconds_cumulative"]
            print(f"CHECKPOINT_RESUME task={t}", flush=True)
        else:
            if t == 0:
                # Match Section22's task-local fresh initialization.
                model = MLP(data.x_train.shape[1], config["hidden"], len(data.class_order)).to(device)
            mask = np.isin(data.y_train, task)
            cx, cy = data.x_train[mask], data.y_train[mask]
            original_current_rows = len(cy)
            if correction == "bic":
                cx, cy, nx, ny = reserve_calibration(cx, cy, task_seed,
                    spec["bic_holdout_per_class_max"], spec["bic_holdout_fraction"])
                hx.extend(nx); hy.extend(ny)
            teacher = copy.deepcopy(model).eval() if t else None
            if teacher is not None:
                for p in teacher.parameters():
                    p.requires_grad_(False)
            ox, oy = old_memory_arrays(memory, bx, by, old, data.x_train.shape[1])
            stats = {}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            print(f"TASK_TRAIN task={t} current_rows={len(cy)} old_memory_rows={len(oy)}", flush=True)
            if correction == "ssil":
                train_ssil(model=model, current_x=cx, current_y=cy, buffer_x=bx, buffer_y=by,
                           memory=memory, base_method=base, teacher=teacher, old_classes=old,
                           seen_classes=seen, tasks=data.tasks[:t + 1], config=config,
                           regularization=identity["regularization"], task_seed=task_seed,
                           device=device, training_stats=stats, tkd_weight=params["tkd_weight"])
                train_rows = len(cy) if base == "derpp" else len(cy) + len(oy)
            else:
                train_rows = train_base(model, cx, cy, bx, by, memory,
                    teacher if base in {"replay_lwf", "rcl_tab"} else None,
                    old, seen, base, config, identity["regularization"], task_seed, device, stats)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            cumulative += elapsed
            target = max(config["epochs_per_task"] * ((train_rows + config["batch_size"] - 1) // config["batch_size"]), config["min_steps_per_task"])
            if stats.get("optimizer_steps") != target or stats.get("target_steps") != target:
                raise ValueError("Base optimizer budget mismatch")
            del teacher
            started = time.perf_counter()
            correction_stats = {"applied": False, "optimizer_steps": 0, "reason": "initial task"}
            if t and correction == "wa":
                correction_stats = apply_wa(model, old, tuple(task))
            elif t and correction == "bic":
                vx, vy = balanced_calibration(hx, hy, task_seed + 919)
                correction_stats = fit_bic(model, vx, vy, seen, tuple(task),
                    lr=params["learning_rate"], steps=params["steps"], batch_size=config["batch_size"],
                    device=device, seed=task_seed + 1919)
                del vx, vy
            elif t and correction == "cbf":
                correction_stats = balanced_finetune(model, cx, cy, ox, oy, data.tasks[:t + 1],
                    lr=params["learning_rate"], epochs=params["epochs"],
                    batch_size=config["batch_size"], seed=task_seed + 2919, device=device)
            elif correction == "ssil":
                correction_stats = {"applied": True, "optimizer_steps": 0,
                                    "reason": "SS-IL modifies base optimization; no post-hoc steps"}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            extra_elapsed = time.perf_counter() - started
            extra_cumulative += extra_elapsed
            peak_alloc = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
            started = time.perf_counter()
            if memory is None:
                bx, by = update_memory(model, cx, cy, bx, by, base, config, task_seed, device)
            selection_seconds = time.perf_counter() - started
            held_counts = {int(y[0]): len(y) for y in hy}
            stored_counts = ({c: min(memory.per_class, n) for c, n in enumerate(memory.observations)}
                             if memory is not None else {int(y[0]): len(y) for y in by})
            if any(stored_counts.get(c, 0) + held_counts.get(c, 0) > 200 for c in seen):
                raise ValueError("Training memory plus BiC holdout exceeds200/class")
            state = {"identity": identity, "task_index": t, "seen_classes": list(seen),
                     "new_classes": list(task), "correction_complete": True,
                     "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                     "buffer_x": [torch.from_numpy(a.copy()) for a in bx],
                     "buffer_y": [torch.from_numpy(a.copy()) for a in by],
                     "holdout_x": [torch.from_numpy(a.copy()) for a in hx],
                     "holdout_y": [torch.from_numpy(a.copy()) for a in hy],
                     "derpp_memory": memory.state() if memory is not None else None,
                     "training_schedule": stats, "correction": correction_stats,
                     "training_seconds": elapsed, "training_seconds_cumulative": cumulative,
                     "correction_seconds": extra_elapsed, "correction_seconds_cumulative": extra_cumulative,
                     "buffer_selection_seconds": selection_seconds, "train_rows": train_rows,
                     "new_training_rows": len(cy), "new_rows_before_internal_holdout": original_current_rows,
                     "training_memory_counts": {str(c): n for c, n in stored_counts.items()},
                     "calibration_holdout_counts": {str(c): n for c, n in held_counts.items()},
                     "peak_cuda_allocated_bytes": peak_alloc, "peak_cuda_reserved_bytes": peak_reserved}
            checkpoint_save(cp_path, state)
            del cx, cy, ox, oy
            print(f"TASK_TRAINED task={t} base_seconds={elapsed:.3f} correction_seconds={extra_elapsed:.3f}", flush=True)
        cp_hash = digest(cp_path)
        if metric_path.exists():
            record = json.loads(metric_path.read_text(encoding="utf-8"))
            if record["identity_hash"] != canonical_hash(identity) or record["checkpoint_sha256"] != cp_hash:
                raise ValueError("Metric checkpoint identity differs")
        else:
            started = time.perf_counter()
            excluded = {"identity", "model_state", "buffer_x", "buffer_y", "holdout_x", "holdout_y", "derpp_memory"}
            record = {k: v for k, v in state.items() if k not in excluded}
            record.update(checkpoint_file=cp_path.relative_to(output).as_posix(), checkpoint_sha256=cp_hash,
                          identity_hash=canonical_hash(identity),
                          id=evaluate(model, data.x_test, data.y_test, seen, data, config, device),
                          evaluation_role=identity["evaluation_role"], shift=None)
            record["evaluation_seconds"] = time.perf_counter() - started
            save_json(metric_path, record)
        records.append(record)
        old = seen
        print(f"TASK_COMPLETE task={t} macro_f1={record['id']['macro_f1']:.6f}", flush=True)
    result = {"schema_version": 1, "completed": True, "identity": identity,
              "class_order": data.class_order, "tasks": data.tasks,
              "train_counts": {str(c): int(np.sum(data.y_train == c)) for c in range(len(data.class_order))},
              "checkpoints": records, "training_seconds_total": cumulative,
              "correction_seconds_total": extra_cumulative,
              "setting_metadata": metadata, "adaptations": spec["adaptations"], "sources": spec["sources"],
              "command": identity["command"], "log_path": path_label(output / "run.log")}
    require_finite(result)
    save_json(final_path, result)
    print("CELL_COMPLETE", identity["dataset"], identity["method"], seed, flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "ncd/configs/bias_correction_section23.json")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--correction", required=True)
    p.add_argument("--candidate", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--phase", choices=("development", "formal"), required=True)
    p.add_argument("--selection", type=Path)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", choices=("cpu", "cuda"))
    args = p.parse_args()
    spec = json.loads(args.config.read_text(encoding="utf-8"))
    validate_request(spec, args.dataset, args.base, args.correction, args.candidate, args.seed, args.phase)
    selected = verify_formal_selection(args.selection, spec, args.dataset, args.base, args.correction,
                                      args.candidate, args.config) if args.phase == "formal" else None
    if args.phase == "development" and args.selection is not None:
        raise ValueError("Development cannot consume formal selection")
    config = json.loads((ROOT / spec["base_config"]).read_text(encoding="utf-8"))
    config.update(min_steps_per_task=spec["min_steps_per_task"], epochs_per_task=spec["epochs_per_task"],
                  batch_size=spec["batch_size"], ssil_old_replay_batch_size=spec["ssil_old_replay_batch_size"],
                  smoke=args.smoke)
    if args.device:
        config["device"] = args.device
    if args.correction == "bic":
        config["buffer_per_class"] = spec["bic_training_memory_per_class"]
    if args.smoke:
        if args.phase != "development":
            raise ValueError("Smoke cannot produce formal evidence")
        config.update(epochs_per_task=1, min_steps_per_task=0, buffer_per_class=16)
    params = dict(spec["candidates"][args.correction][args.candidate])
    if args.smoke:
        if "epochs" in params:
            params["epochs"] = 1
        if "steps" in params:
            params["steps"] = 2
    torch.set_num_threads(config["cpu_threads"])
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with cell_lock(output / ".execution.lock"), (output / "run.log").open("a", encoding="utf-8") as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        data, _, _, metadata = prepare_setting(args.dataset, config)
        if args.smoke:
            data, _, _ = smoke_subset(data, None, None)
        split = {"role": "official_ID_test_not_used_for_selection"}
        if args.phase == "development":
            data, split = training_holdout(data, args.seed, spec["development_validation_fraction"])
        selection_path = ROOT / config["lambda_files"].get(args.dataset, config["lambda_files"]["default"])
        historical = json.loads(selection_path.read_text(encoding="utf-8"))
        if historical.get("selection_seeds") != [5, 6]:
            raise ValueError("Historical base regularization must use development seeds")
        command = subprocess.list2cmdline(["python", "-I", "ncd/run_bias_correction.py", *sys.argv[1:]])
        # CLI uses relative repo paths in durable/public provenance.
        command = command.replace(str(ROOT), ".").replace(ROOT.as_posix(), ".")
        identity = {"dataset": args.dataset, "base_method": args.base, "correction": args.correction,
                    "method": args.base + "+" + args.correction, "candidate": args.candidate,
                    "seed": args.seed, "phase": args.phase, "smoke": args.smoke,
                    "git_commit": source_commit(ROOT), "effective_config": config,
                    "hyperparameters": params, "config_sha256": digest(args.config),
                    "config_input_hashes": configuration_input_hashes(spec),
                    "source_hashes": {s: digest(ROOT / s) for s in SOURCE_PATHS},
                    "class_order": data.class_order, "tasks": data.tasks,
                    "data_binding": binding(data, None, None), "development_split": split,
                    "evaluation_role": split["role"], "selection_seeds": [5, 6],
                    "frozen_selection": selected,
                    "regularization": historical["lambdas"].get(args.base, 0.0),
                    "lambda_file": path_label(selection_path), "lambda_file_sha256": digest(selection_path),
                    "runtime_versions": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                                         "numpy": np.__version__, "cuda": torch.version.cuda},
                    "command": command, "log_path": path_label(output / "run.log")}
        run_cell(data, metadata, identity, output, config, spec, params, device)


if __name__ == "__main__":
    main()
