"""Exp H: unchanged Section 15 learners, saved checkpoints, and shifted recalls.

CRR remains a rejected comparison method.  No learner, regularization strength,
or training configuration is selected using the shifted test sets.  The S3
checkpoint reconstruction is checked against every previously registered NSL
task-history metric because the original Section 15 runner saved no weights.

Both PI endpoints use identical ID rows and feature permutations before/after.
Self-margin means score(c) - max(score(k): k in classes seen BEFORE transition,
k != c); fixing that contrast set avoids attributing head expansion to drift.
Prediction recalls still use all classes seen at each respective checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import f1_score

from ncd.crr_benchmarks import METHODS, _train
from ncd.data import PreparedData
from ncd.model import DecisionScore, MLP
from ncd.run_crr_real_sequence import (
    SEEDS, _recalls_and_predictions, _score, _scorer, _select_buffer, _write_csv,
    spearman,
)
from ncd.source_commit import source_commit
from ncd.train_cil import Tee, seed_everything


TRAINING_KEYS = (
    "hidden", "batch_size", "epochs_per_task", "learning_rate",
    "buffer_per_class", "temperature", "rcl_features", "rcl_temperature",
    "rcl_beta", "rcl_prediction_weight",
)
SELF_MARGIN_DEFINITION = (
    "score_c - max(score_k for k in classes seen before transition, k != c); "
    "the same old contrast set is used at both checkpoints"
)


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _command() -> str:
    return subprocess.list2cmdline(['python', '-I', 'ncd/run_exp_h.py', *sys.argv[1:]])


def _save_json(path: Path, value: dict) -> None:
    """Atomic completion avoids treating an interrupted JSON write as a result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _take_per_class(x: np.ndarray, y: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.concatenate([np.flatnonzero(y == c)[:limit] for c in np.unique(y)])
    return x[indices], y[indices]


def _smoke_data(data: PreparedData, x_shift: np.ndarray, y_shift: np.ndarray):
    fields = {}
    for split in ("train", "test"):
        fields[f"x_{split}"], fields[f"y_{split}"] = _take_per_class(
            getattr(data, f"x_{split}"), getattr(data, f"y_{split}"), 64
        )
    return replace(data, **fields), *_take_per_class(x_shift, y_shift, 64)


def _rng_state() -> dict:
    return {
        "cpu": torch.get_rng_state().cpu(),
        "cuda": [value.cpu() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def _restore_rng(state: dict) -> None:
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA checkpoint cannot resume training without CUDA")
        torch.cuda.set_rng_state_all(state["cuda"])


def _load_checkpoint(path: Path, identity: dict) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["identity"] != identity:
        raise ValueError(f"checkpoint configuration/source mismatch: {path}")
    return checkpoint


def _checkpoint_scorer(checkpoint: dict, device: torch.device) -> DecisionScore:
    model = MLP(checkpoint["input_dim"], checkpoint["hidden"], checkpoint["n_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    prototypes = {int(k): value.to(device) for k, value in checkpoint["prototypes"].items()}
    return DecisionScore(model, checkpoint["inference"], checkpoint["n_classes"], prototypes).to(device).eval()


def _verify_reconstruction(history: list[dict], reference: dict, config: dict, regularization: float) -> dict:
    """Check every original task recall and Macro-F1, without comparing runtime."""
    reference_config = reference["effective_config"]
    for key in TRAINING_KEYS:
        if config[key] != reference_config[key]:
            raise ValueError(f"S3 reconstruction changes registered training parameter {key}")
    expected = reference["rows"][0]
    if expected["regularization"] != regularization:
        raise ValueError("S3 reconstruction changes registered regularization")
    if len(history) != len(expected["task_history"]):
        raise ValueError("S3 reconstruction has a different number of checkpoints")
    differences = []
    checks = 0
    for observed, prior in zip(history, expected["task_history"]):
        if observed["task_index"] != prior["task_index"] or observed["seen_classes"] != prior["seen_classes"]:
            raise ValueError("S3 task order differs from registered sequence")
        if set(observed["recalls"]) != set(prior["recalls"]):
            raise ValueError("S3 class recall keys differ from registered sequence")
        pairs = [("macro_f1", observed["macro_f1"], prior["macro_f1"])]
        pairs.extend((f"recall_{c}", observed["recalls"][c], prior["recalls"][c]) for c in prior["recalls"])
        for metric, value, old in pairs:
            delta = abs(float(value) - float(old))
            checks += 1
            if delta > 1e-10:
                differences.append({"task_index": observed["task_index"], "metric": metric, "observed": value, "registered": old, "absolute_difference": delta})
    return {
        "matched": not differences,
        "metric_count": checks,
        "absolute_tolerance": 1e-10,
        "reference_git_commit": reference["git_commit"],
        "differences": differences,
    }


def _verify_pi_reconstruction(rows: list[dict], reference: dict) -> dict:
    """The class-level mean must recover the already published pair PI rows."""
    grouped: dict[tuple[int, int], list[float]] = {}
    for row in reference["pi_drift_rows"]:
        key = (int(row["transition"]), int(row["target_class"]))
        grouped.setdefault(key, []).append(float(row["old_pair_pi_drift"]))
    differences = []
    checks = 0
    for row in rows:
        key = (int(row["transition"]), int(row["class_id"]))
        if key not in grouped:
            raise ValueError(f"registered S3 PI rows lack class/transition {key}")
        expected = float(np.mean(grouped[key]))
        delta = abs(row["old_pair_pi_drift"] - expected)
        checks += 1
        if delta > 1e-8:
            differences.append({"transition": key[0], "class_id": key[1], "observed": row["old_pair_pi_drift"], "registered": expected, "absolute_difference": delta})
    return {"matched": not differences, "metric_count": checks, "absolute_tolerance": 1e-8, "differences": differences}


def _train_checkpoints(data: PreparedData, config: dict, identity: dict, output: Path,
                       device: torch.device, evaluate_only: bool) -> tuple[list[Path], list[dict]]:
    """Exactly reproduce run_crr_real_sequence training, saving at task boundaries."""
    method, seed, regularization = identity["method"], identity["seed"], identity["regularization"]
    seed_everything(seed)
    model = MLP(data.x_train.shape[1], int(config["hidden"]), len(data.class_order)).to(device)
    buffer_x: list[np.ndarray] = []
    buffer_y: list[np.ndarray] = []
    previous_seen: tuple[int, ...] = ()
    paths, history = [], []
    training_seconds = 0.0
    batch_size = int(config["batch_size"])
    for task_index, task in enumerate(data.tasks):
        seen = tuple(int(c) for values in data.tasks[:task_index + 1] for c in values)
        path = output / "checkpoints" / f"checkpoint_{task_index:03d}.pt"
        if path.exists():
            checkpoint = _load_checkpoint(path, identity)
            if checkpoint["seen_classes"] != list(seen):
                raise ValueError(f"checkpoint task order mismatch: {path}")
            model.load_state_dict(checkpoint["model_state"])
            buffer_x = [value.numpy().copy() for value in checkpoint["buffer_x"]]
            buffer_y = [value.numpy().copy() for value in checkpoint["buffer_y"]]
            history.append(checkpoint["task_history_row"])
            training_seconds = checkpoint["training_seconds"]
            _restore_rng(checkpoint["rng_state"])
            print(f"CHECKPOINT_RESUME {identity['setting']} {method} seed={seed} task={task_index}", flush=True)
        else:
            if evaluate_only:
                raise FileNotFoundError(f"--evaluate-only requires saved checkpoint: {path}")
            current_mask = np.isin(data.y_train, tuple(int(c) for c in task))
            train_x, train_y = data.x_train[current_mask], data.y_train[current_mask]
            if buffer_x:
                train_x = np.concatenate([train_x, *buffer_x], axis=0)
                train_y = np.concatenate([train_y, *buffer_y], axis=0)
            teacher = copy.deepcopy(model).to(device).eval() if task_index else None
            training_seconds += _train(
                model, train_x, train_y, method=method, old_model=teacher,
                old_classes=previous_seen, seen_classes=seen,
                epochs=int(config["epochs_per_task"]), batch_size=batch_size,
                learning_rate=float(config["learning_rate"]), seed=seed * 10000 + task_index,
                regularization=regularization, temperature=float(config["temperature"]),
                rcl_features=int(config["rcl_features"]), rcl_temperature=float(config["rcl_temperature"]),
                rcl_beta=float(config["rcl_beta"]), rcl_prediction_weight=float(config["rcl_prediction_weight"]),
            )
            buffer_x, buffer_y = _select_buffer(
                model, data, seen, method, int(config["buffer_per_class"]),
                seed * 100 + task_index, device, batch_size,
            )
            scorer = _scorer(model, method, buffer_x, buffer_y, len(data.class_order), device, batch_size)
            recalls, pred, labels = _recalls_and_predictions(scorer, data, seen, device, batch_size)
            row = {
                "task_index": task_index, "seen_classes": list(seen), "recalls": recalls,
                "macro_f1": float(f1_score(labels, pred, labels=list(seen), average="macro", zero_division=0)),
            }
            history.append(row)
            prototypes = {
                int(class_id): scorer.prototype_matrix[index].detach().cpu().clone()
                for index, class_id in enumerate(scorer.seen_ids.tolist())
            }
            checkpoint = {
                "identity": identity, "task_index": task_index, "seen_classes": list(seen),
                "model_state": {k: value.detach().cpu().clone() for k, value in model.state_dict().items()},
                "buffer_x": [torch.from_numpy(value.copy()) for value in buffer_x],
                "buffer_y": [torch.from_numpy(value.copy()) for value in buffer_y],
                "prototypes": prototypes, "inference": "nme" if method == "icarl" else "logit",
                "input_dim": data.x_train.shape[1], "hidden": int(config["hidden"]),
                "n_classes": len(data.class_order), "class_order": data.class_order,
                "task_history_row": row, "training_seconds": training_seconds, "rng_state": _rng_state(),
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".pt.tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(path)
            print(f"CHECKPOINT_SAVED {identity['setting']} {method} seed={seed} task={task_index} macro_f1={row['macro_f1']:.6f}", flush=True)
        previous_seen = seen
        paths.append(path)
    return paths, history


def _shift_recalls(scorer: DecisionScore, x: np.ndarray, y: np.ndarray,
                   seen: tuple[int, ...], device: torch.device, batch_size: int) -> dict[str, float]:
    mask = np.isin(y, seen)
    if not np.any(mask):
        return {}
    selected_y = y[mask]
    scores = _score(scorer, x[mask], device, batch_size)
    predictions = np.asarray(seen, dtype=np.int64)[np.argmax(scores[:, list(seen)], axis=1)]
    return {str(c): float(np.mean(predictions[selected_y == c] == c)) for c in seen if np.any(selected_y == c)}


def _dependency_vectors(scorer: DecisionScore, data: PreparedData, old: tuple[int, ...],
                        seed: int, device: torch.device, batch_size: int) -> tuple[dict, dict]:
    """Same signed, class-conditional PI/permutation as Section 15; shared scoring."""
    pairs, self_vectors = {}, {}
    for c in old:
        others = tuple(k for k in old if k != c)
        x = data.x_test[data.y_test == c]
        if not len(x) or not others:
            continue
        clean = _score(scorer, x, device, batch_size)
        pair_mean = {k: float(np.mean(clean[:, c] - clean[:, k])) for k in others}
        self_mean = float(np.mean(clean[:, c] - np.max(clean[:, list(others)], axis=1)))
        for k in others:
            pairs[(c, k)] = np.zeros(x.shape[1], dtype=np.float64)
        self_vectors[c] = np.zeros(x.shape[1], dtype=np.float64)
        for feature in range(x.shape[1]):
            changed = x.copy()
            rng = np.random.default_rng(seed * 1000003 + c * 1009 + feature)
            changed[:, feature] = changed[rng.permutation(len(changed)), feature]
            scores = _score(scorer, changed, device, batch_size)
            for k in others:
                pairs[(c, k)][feature] = pair_mean[k] - float(np.mean(scores[:, c] - scores[:, k]))
            self_vectors[c][feature] = self_mean - float(np.mean(scores[:, c] - np.max(scores[:, list(others)], axis=1)))
    return pairs, self_vectors


def _evaluate(data: PreparedData, x_shift: np.ndarray, y_shift: np.ndarray, config: dict,
              identity: dict, paths: list[Path], history: list[dict], output: Path,
              device: torch.device) -> tuple[list[dict], list[dict], list[dict]]:
    rows, shift_history = [], []
    batch_size = int(config["batch_size"])
    # Evaluate each actual prediction adapter, including saved NME prototypes.
    for path in paths:
        checkpoint = _load_checkpoint(path, identity)
        scorer = _checkpoint_scorer(checkpoint, device)
        recalls = _shift_recalls(scorer, x_shift, y_shift, tuple(checkpoint["seen_classes"]), device, batch_size)
        shift_history.append({"task_index": checkpoint["task_index"], "recalls": recalls})
        del scorer
    for after_task in range(1, len(paths)):
        old = tuple(history[after_task - 1]["seen_classes"])
        if len(old) < 2:
            continue
        prior = _checkpoint_scorer(_load_checkpoint(paths[after_task - 1], identity), device)
        following = _checkpoint_scorer(_load_checkpoint(paths[after_task], identity), device)
        before_pair, before_self = _dependency_vectors(prior, data, old, identity["seed"] * 100 + after_task, device, batch_size)
        after_pair, after_self = _dependency_vectors(following, data, old, identity["seed"] * 100 + after_task, device, batch_size)
        vectors = {}
        for (c, k), value in before_pair.items():
            vectors[f"before_pair_c{c}_k{k}"] = value
            vectors[f"after_pair_c{c}_k{k}"] = after_pair[(c, k)]
        for c, value in before_self.items():
            vectors[f"before_self_c{c}"] = value
            vectors[f"after_self_c{c}"] = after_self[c]
        vector_path = output / f"pi_vectors_transition_{after_task:03d}.npz"
        np.savez_compressed(vector_path, **vectors)
        for c in old:
            key = str(c)
            if key not in shift_history[after_task - 1]["recalls"]:
                continue  # Not a common class: no shifted-recall observation exists.
            pair_keys = [(c, k) for k in old if k != c]
            row = {
                "setting": identity["setting"], "method": identity["method"], "seed": identity["seed"],
                "transition": after_task, "task_before": after_task - 1, "task_after": after_task,
                "class_id": c, "class_name": data.class_order[c],
                "old_pair_pi_drift": float(np.mean([1.0 - spearman(before_pair[p], after_pair[p]) for p in pair_keys])),
                "self_margin_pi_drift": 1.0 - spearman(before_self[c], after_self[c]),
                "shift_recall_before": shift_history[after_task - 1]["recalls"][key],
                "shift_recall_after": shift_history[after_task]["recalls"][key],
                "id_recall_before": history[after_task - 1]["recalls"][key],
                "id_recall_after": history[after_task]["recalls"][key],
                "old_contrast_count": len(pair_keys), "pi_probe_rows": int(np.sum(data.y_test == c)),
                "shift_rows": int(np.sum(y_shift == c)), "feature_count": data.x_test.shape[1],
                "pi_vectors_file": vector_path.name,
            }
            rows.append(row)
        print(f"TRANSITION_COMPLETE {identity['setting']} {identity['method']} seed={identity['seed']} transition={after_task} rows={len(rows)}", flush=True)
        del prior, following
    retention_rows = []
    for c, name in enumerate(data.class_order):
        available = [item["recalls"][str(c)] for item in shift_history if str(c) in item["recalls"]]
        if available:
            retention_rows.append({
                "setting": identity["setting"], "method": identity["method"], "seed": identity["seed"],
                "class_id": c, "class_name": name, "shift_recall_final": available[-1],
                "shift_recall_best": max(available), "checkpoint_count": len(available),
                "shift_rows": int(np.sum(y_shift == c)),
            })
    return rows, retention_rows, shift_history


def run_cell(setting: str, method: str, seed: int, data: PreparedData, x_shift: np.ndarray,
             y_shift: np.ndarray, metadata: dict, config: dict, selection: dict, output: Path,
             commit: str, reference_dir: Path, log_path: Path, evaluate_only: bool = False,
             smoke: bool = False) -> dict:
    identity = {
        "setting": setting, "method": method, "seed": seed, "git_commit": commit,
        "effective_config": config, "regularization": 0.0 if method == "replay" else float(selection["lambdas"][method]),
        "lambda_source_commit": selection["git_commit"], "smoke": smoke,
        "class_order": data.class_order, "tasks": data.tasks, "feature_names": data.feature_names,
        "train_rows": len(data.x_train), "id_rows": len(data.x_test), "shift_rows": len(x_shift),
    }
    result_path = output / "exp_h_result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["identity"] != identity:
            raise ValueError(f"existing result configuration/source mismatch: {result_path}")
        print(f"CELL_RESUME {setting} {method} seed={seed}", flush=True)
        return result
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    started = time.perf_counter()
    print(f"CELL_START {setting} {method} seed={seed}", flush=True)
    paths, history = _train_checkpoints(data, config, identity, output, device, evaluate_only)
    reconstruction = None
    if setting == "S3" and not smoke:
        reference_path = reference_dir / "shards" / f"{method}_seed{seed}" / "real_result.json"
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        reconstruction = _verify_reconstruction(history, reference, config, identity["regularization"])
        if not reconstruction["matched"]:
            _save_json(output / "reconstruction_comparison.json", reconstruction)
            raise RuntimeError(f"S3 reconstruction differs from registered metrics: {output}; {reconstruction['differences'][:3]}")
        print(f"RECONSTRUCTION_MATCH {method} seed={seed} metrics={reconstruction['metric_count']}", flush=True)
    rows, retention_rows, shift_history = _evaluate(data, x_shift, y_shift, config, identity, paths, history, output, device)
    if reconstruction is not None:
        reconstruction["old_pair_pi_comparison"] = _verify_pi_reconstruction(rows, reference)
        if not reconstruction["old_pair_pi_comparison"]["matched"]:
            _save_json(output / "reconstruction_comparison.json", reconstruction)
            raise RuntimeError(f"S3 reconstructed old-pair PI differs from registered pair rows: {output}")
    result = {
        "schema_version": 1, "identity": identity, "git_commit": commit,
        "command": _command(), "log_path": _portable_path(log_path),
        "effective_config": config, "lambda_selection": selection,
        "rows": rows, "retention_rows": retention_rows, "task_history": history,
        "shift_history": shift_history, "setting_metadata": metadata,
        "reconstruction_comparison": reconstruction, "self_margin_definition": SELF_MARGIN_DEFINITION,
        "checkpoints": [str(path.relative_to(output)) for path in paths],
        "elapsed_seconds": time.perf_counter() - started,
        "scope": "WORK_ORDER Section 17 drift-harm measurement; frozen Section 15 methods; no CRR modification",
    }
    _save_json(result_path, result)
    _write_csv(output / "exp_h_rows.csv", rows)
    _write_csv(output / "exp_h_retention_rows.csv", retention_rows)
    print(f"CELL_COMPLETE {setting} {method} seed={seed} units={len(rows)}", flush=True)
    return result


def run(args: argparse.Namespace) -> None:
    from ncd.exp_h_data import prepare_setting

    torch.set_num_threads(2)
    config = json.loads((ROOT / "ncd/configs/crr_section15.json").read_text(encoding="utf-8"))
    config.update(json.loads(args.config.read_text(encoding="utf-8")))
    if args.device:
        config["device"] = args.device
    if args.smoke:
        config.update(epochs_per_task=1, smoke_rows_per_class=64, buffer_per_class=16)
    config["exp_h_smoke"] = args.smoke
    selection_path = args.lambda_file or ROOT / "ncd/results/crr/section15/registered/nsl-real/lambda_selections.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["dataset"] != "NSL-KDD" or set(selection["lambdas"]) != set(METHODS):
        raise ValueError("Exp H requires the frozen Section 15 NSL 8-method lambda selection")
    if selection["selection_seeds"] != [5, 6]:
        raise ValueError("unexpected calibration seeds")
    commit = source_commit(ROOT)
    os.environ["NCD_SOURCE_COMMIT"] = commit
    seeds = (1,) if args.smoke else SEEDS
    combined_rows, combined_retention, cell_summaries, metadata = [], [], [], {}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def snapshot() -> Path:
        result = {
            "schema_version": 1, "git_commit": commit, "effective_config": config,
            "command": _command(), "log_path": _portable_path(args.log_path),
            "settings": args.settings, "methods": list(METHODS), "seeds": list(seeds), "smoke": args.smoke,
            "lambda_selection": selection, "setting_metadata": metadata, "rows": combined_rows,
            "retention_rows": combined_retention, "cells": cell_summaries,
            "self_margin_definition": SELF_MARGIN_DEFINITION,
        }
        path = args.output_dir / "exp_h_result.json"
        _save_json(path, result)
        _write_csv(args.output_dir / "exp_h_rows.csv", combined_rows)
        _write_csv(args.output_dir / "exp_h_retention_rows.csv", combined_retention)
        return path

    def analyze(path: Path) -> None:
        # The same entry goes through the final analysis/report step, including smoke.
        analysis_command = [
            sys.executable, "-I", str(ROOT / "ncd/analyze_exp_h.py"),
            "--input", str(path), "--output-json", str(args.output_dir / "exp_h_analysis.json"),
            "--report", str(args.output_dir / "EXP_H_REPORT.md"),
        ]
        with subprocess.Popen(analysis_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace") as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
            return_code = process.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, analysis_command)

    for setting in args.settings:
        print(f"SETTING_START {setting}", flush=True)
        data, x_shift, y_shift, setting_metadata = prepare_setting(setting, config)
        if args.smoke:
            data, x_shift, y_shift = _smoke_data(data, x_shift, y_shift)
        if len(x_shift) != len(y_shift) or x_shift.shape[1] != data.x_train.shape[1]:
            raise ValueError(f"{setting} shifted features or labels are not aligned")
        if not np.isfinite(x_shift).all() or not np.isfinite(data.x_train).all() or not np.isfinite(data.x_test).all():
            raise ValueError(f"{setting} data contain non-finite transformed values")
        if "feature_alignment" not in setting_metadata:
            raise ValueError("prepare_setting must report feature_alignment")
        metadata[setting] = dict(setting_metadata)
        metadata[setting].update(
            status="in_progress", smoke=args.smoke,
            expected_units_per_cell=int(sum(
                sum(np.any(y_shift == c) for task in data.tasks[:after] for c in task)
                for after in range(1, len(data.tasks))
                if sum(len(task) for task in data.tasks[:after]) >= 2
            )),
        )
        print(f"SETTING_DATA {setting} train={data.x_train.shape} id={data.x_test.shape} shift={x_shift.shape}", flush=True)
        for method in METHODS:
            for seed in seeds:
                cell_output = args.output_dir / setting / "shards" / f"{method}_seed{seed}"
                result = run_cell(setting, method, seed, data, x_shift, y_shift, setting_metadata,
                                  config, selection, cell_output, commit, args.reference_dir,
                                  args.log_path, args.evaluate_only, args.smoke)
                combined_rows.extend(result["rows"])
                combined_retention.extend(result["retention_rows"])
                cell_summaries.append({
                    "setting": setting, "method": method, "seed": seed,
                    "result": str((cell_output / "exp_h_result.json").relative_to(args.output_dir)),
                    "reconstruction_comparison": result["reconstruction_comparison"],
                    "elapsed_seconds": result["elapsed_seconds"],
                })
                snapshot()
        metadata[setting]["status"] = "complete"
        analyze(snapshot())
        print(f"SETTING_COMPLETE {setting}", flush=True)
    path = snapshot()
    print(f"MATRIX_COMPLETE cells={len(cell_summaries)} units={len(combined_rows)} output={path}", flush=True)
    print(f"ANALYSIS_COMPLETE report={args.output_dir / 'EXP_H_REPORT.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", nargs="+", choices=("S3", "S2", "S1"), default=["S3", "S2", "S1"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--lambda-file", type=Path)
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "ncd/results/crr/section15/registered/nsl-real")
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.config = (ROOT / args.config).resolve()
    args.output_dir = (ROOT / args.output_dir).resolve()
    args.reference_dir = (ROOT / args.reference_dir).resolve()
    if args.lambda_file:
        args.lambda_file = (ROOT / args.lambda_file).resolve()
    args.log_path = (ROOT / args.log_path if args.log_path else args.output_dir / "run.log").resolve()
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    with args.log_path.open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                run(args)
            except Exception:
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
