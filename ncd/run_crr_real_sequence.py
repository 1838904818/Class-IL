"""Run one registered Section 15 real-sequence CRR cell.

The runner deliberately accepts exactly one dataset, method, and seed.  Slurm
can therefore checkpoint each of the 40 registered cells independently.  The
old-pair PI drift is measured on the prediction score actually used by each
method (NME for iCaRL, logits otherwise).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from sklearn.metrics import f1_score

from ncd.crr_benchmarks import METHODS, _train
from ncd.data import PreparedData, prepare_dataset
from ncd.model import DecisionScore, MLP
from ncd.source_commit import source_commit
from ncd.train_cil import Tee, extract_features, herding_select, prototypes_from_buffer, seed_everything


SEEDS = (1, 2, 3, 4, 42)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    scalar = [
        {
            key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
            for key, value in row.items()
        }
        for row in rows
    ]
    fields = list(dict.fromkeys(key for row in scalar for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scalar)


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks with deterministic tie handling, equivalent to scipy.rankdata."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank, right_rank = rankdata(left), rankdata(right)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= 1e-12:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.clip(np.dot(left_centered, right_centered) / denominator, -1.0, 1.0))


def _score(scorer: DecisionScore, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    scorer.eval()
    pieces: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            pieces.append(scorer(batch).cpu().numpy())
    return np.concatenate(pieces, axis=0)


def _scorer(
    model: MLP,
    method: str,
    buffer_x: list[np.ndarray],
    buffer_y: list[np.ndarray],
    class_count: int,
    device: torch.device,
    batch_size: int,
) -> DecisionScore:
    prototypes = None
    inference = "logit"
    if method == "icarl":
        prototypes = prototypes_from_buffer(model, buffer_x, buffer_y, device, batch_size)
        prototypes = {key: value.to(device) for key, value in prototypes.items()}
        inference = "nme"
    return DecisionScore(model, inference, class_count, prototypes).to(device)


def _select_buffer(
    model: MLP,
    data: PreparedData,
    seen: tuple[int, ...],
    method: str,
    count: int,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for class_id in seen:
        values = data.x_train[data.y_train == class_id]
        keep = min(count, len(values))
        if method == "icarl":
            features = extract_features(model, values, device, batch_size)
            chosen = herding_select(features, keep)
        else:
            chosen = np.sort(
                np.random.default_rng(seed * 10007 + class_id).choice(
                    len(values), keep, replace=False
                )
            )
        xs.append(values[chosen].copy())
        ys.append(np.full(keep, class_id, dtype=np.int64))
    return xs, ys


def _recalls_and_predictions(
    scorer: DecisionScore,
    data: PreparedData,
    seen: tuple[int, ...],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    mask = np.isin(data.y_test, seen)
    y = data.y_test[mask]
    scores = _score(scorer, data.x_test[mask], device, batch_size)
    local = np.argmax(scores[:, list(seen)], axis=1)
    pred = np.asarray(seen, dtype=np.int64)[local]
    recalls = {
        str(class_id): float(np.mean(pred[y == class_id] == class_id))
        for class_id in seen
    }
    return recalls, pred, y


def _pi_vectors(
    scorer: DecisionScore,
    data: PreparedData,
    old_classes: tuple[int, ...],
    seed: int,
    device: torch.device,
    batch_size: int,
) -> dict[tuple[int, int], np.ndarray]:
    result: dict[tuple[int, int], np.ndarray] = {
        (c, k): np.zeros(data.x_test.shape[1], dtype=np.float64)
        for c in old_classes
        for k in old_classes
        if c != k
    }
    for class_id in old_classes:
        x = data.x_test[data.y_test == class_id]
        clean = _score(scorer, x, device, batch_size)
        clean_margin = {
            other: float(np.mean(clean[:, class_id] - clean[:, other]))
            for other in old_classes
            if other != class_id
        }
        for feature in range(x.shape[1]):
            changed = x.copy()
            rng = np.random.default_rng(seed * 1000003 + class_id * 1009 + feature)
            changed[:, feature] = changed[rng.permutation(len(changed)), feature]
            changed_score = _score(scorer, changed, device, batch_size)
            for other, baseline in clean_margin.items():
                permuted = float(
                    np.mean(changed_score[:, class_id] - changed_score[:, other])
                )
                result[(class_id, other)][feature] = baseline - permuted
    return result


def run_cell(config: dict, dataset_name: str, method: str, seed: int, regularization: float) -> dict:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    if seed not in SEEDS:
        raise ValueError(f"unregistered seed: {seed}")
    device = torch.device(str(config.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    data = prepare_dataset(dataset_name, (REPO_ROOT / config["data_root"]).resolve())
    seed_everything(seed)
    model = MLP(data.x_train.shape[1], int(config.get("hidden", 128)), len(data.class_order)).to(device)
    buffer_x: list[np.ndarray] = []
    buffer_y: list[np.ndarray] = []
    history: list[dict] = []
    pi_rows: list[dict] = []
    training_seconds = 0.0
    pi_seconds = 0.0
    previous_model: MLP | None = None
    previous_buffer_x: list[np.ndarray] = []
    previous_buffer_y: list[np.ndarray] = []
    previous_seen: tuple[int, ...] = ()

    for task_index, task in enumerate(data.tasks):
        current = tuple(int(value) for value in task)
        seen = tuple(int(value) for values in data.tasks[: task_index + 1] for value in values)
        current_mask = np.isin(data.y_train, current)
        train_x = data.x_train[current_mask]
        train_y = data.y_train[current_mask]
        if buffer_x:
            train_x = np.concatenate([train_x, *buffer_x], axis=0)
            train_y = np.concatenate([train_y, *buffer_y], axis=0)
        teacher = copy.deepcopy(model).to(device).eval() if task_index else None
        training_seconds += _train(
            model,
            train_x,
            train_y,
            method=method,
            old_model=teacher,
            old_classes=previous_seen,
            seen_classes=seen,
            epochs=int(config.get("epochs_per_task", 8)),
            batch_size=int(config.get("batch_size", 256)),
            learning_rate=float(config.get("learning_rate", 1e-3)),
            seed=seed * 10000 + task_index,
            regularization=0.0 if method == "replay" else regularization,
            temperature=float(config.get("temperature", 2.0)),
            rcl_features=int(config.get("rcl_features", 8)),
            rcl_temperature=float(config.get("rcl_temperature", 0.5)),
            rcl_beta=float(config.get("rcl_beta", 1.0)),
            rcl_prediction_weight=float(config.get("rcl_prediction_weight", 0.5)),
        )
        new_buffer_x, new_buffer_y = _select_buffer(
            model,
            data,
            seen,
            method,
            int(config.get("buffer_per_class", 200)),
            seed * 100 + task_index,
            device,
            int(config.get("batch_size", 256)),
        )
        scorer = _scorer(
            model,
            method,
            new_buffer_x,
            new_buffer_y,
            len(data.class_order),
            device,
            int(config.get("batch_size", 256)),
        )
        recalls, pred, y = _recalls_and_predictions(
            scorer, data, seen, device, int(config.get("batch_size", 256))
        )
        history.append(
            {
                "task_index": task_index,
                "seen_classes": list(seen),
                "recalls": recalls,
                "macro_f1": float(
                    f1_score(y, pred, labels=list(seen), average="macro", zero_division=0)
                ),
            }
        )
        if previous_model is not None and len(previous_seen) >= 2:
            started = time.perf_counter()
            prior_scorer = _scorer(
                previous_model,
                method,
                previous_buffer_x,
                previous_buffer_y,
                len(data.class_order),
                device,
                int(config.get("batch_size", 256)),
            )
            before = _pi_vectors(
                prior_scorer,
                data,
                previous_seen,
                seed * 100 + task_index,
                device,
                int(config.get("batch_size", 256)),
            )
            after = _pi_vectors(
                scorer,
                data,
                previous_seen,
                seed * 100 + task_index,
                device,
                int(config.get("batch_size", 256)),
            )
            for (class_id, contrast), prior_vector in before.items():
                rho = spearman(prior_vector, after[(class_id, contrast)])
                pi_rows.append(
                    {
                        "dataset": dataset_name,
                        "seed": seed,
                        "method": method,
                        "transition": task_index,
                        "target_class": class_id,
                        "contrast_class": contrast,
                        "target_name": data.class_order[class_id],
                        "contrast_name": data.class_order[contrast],
                        "spearman": rho,
                        "old_pair_pi_drift": 1.0 - rho,
                        "feature_count": int(data.x_test.shape[1]),
                        "target_test_rows": int(np.sum(data.y_test == class_id)),
                    }
                )
            pi_seconds += time.perf_counter() - started
        previous_model = copy.deepcopy(model).to(device).eval()
        previous_buffer_x = [value.copy() for value in new_buffer_x]
        previous_buffer_y = [value.copy() for value in new_buffer_y]
        previous_seen = seen
        buffer_x, buffer_y = new_buffer_x, new_buffer_y

    final_recalls = history[-1]["recalls"]
    forgetting_values: list[float] = []
    for class_id in range(len(data.class_order)):
        available = [
            float(item["recalls"][str(class_id)])
            for item in history
            if str(class_id) in item["recalls"]
        ]
        if available:
            forgetting_values.append(max(available) - float(final_recalls[str(class_id)]))
    mean_drift = float(np.mean([row["old_pair_pi_drift"] for row in pi_rows])) if pi_rows else 0.0
    row = {
        "dataset": dataset_name,
        "seed": seed,
        "method": method,
        "regularization": float(0.0 if method == "replay" else regularization),
        "macro_f1": float(history[-1]["macro_f1"]),
        "forgetting": float(np.mean(forgetting_values)),
        "old_pair_pi_drift": mean_drift,
        "training_seconds": float(training_seconds),
        "pi_evaluation_seconds": float(pi_seconds),
        "cost_vs_replay": None,
        "class_balanced_ce": True,
        "inference": "nme" if method == "icarl" else "logit",
        "final_recalls": final_recalls,
        "task_history": history,
    }
    return {
        "schema_version": 1,
        "git_commit": source_commit(REPO_ROOT),
        "effective_config": config,
        "rows": [row],
        "pi_drift_rows": pi_rows,
        "scope": "Section 15 real-sequence cell; cross-method adjudication pending",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True, choices=("NSL-KDD", "UNSW-NB15"))
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--regularization", required=True, type=float)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    config = json.loads(Path(args.config).resolve().read_text(encoding="utf-8"))
    if args.device:
        config["device"] = args.device
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        try:
            result = run_cell(config, args.dataset, args.method, args.seed, args.regularization)
            result_path = output / "real_result.json"
            result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
            _write_csv(output / "real_rows.csv", result["rows"])
            _write_csv(output / "real_pi_drift_rows.csv", result["pi_drift_rows"])
            (output / "COMPLETED.json").write_text(
                json.dumps(
                    {
                        "git_commit": result["git_commit"],
                        "dataset": args.dataset,
                        "method": args.method,
                        "seed": args.seed,
                        "row_count": 1,
                        "pi_row_count": len(result["pi_drift_rows"]),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            print(json.dumps({"complete": str(result_path), "rows": 1}))
        except Exception:
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
