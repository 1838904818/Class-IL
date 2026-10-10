"""Train one Exp 0 continual-learning run and save every task checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

# PyTorch requires this before the first CUDA context is created when
# deterministic algorithms are enabled.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# ``python -I ncd/train_cil.py`` intentionally ignores the current directory.
# Add only this checked-out repository so the announced isolated command works.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ncd.data import PreparedData, prepare_dataset
from ncd.model import DecisionScore, MLP
from ncd.source_commit import source_commit
from src.methods.replay_dpmeans import _dpmeans_select


SUPPORTED_METHODS = {
    "naive",
    "ewc",
    "lwf",
    "replay",
    "replay-herding",
    "replay-dpmeans",
    "icarl",
}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def loader(x: np.ndarray, y: np.ndarray, batch_size: int, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


def train_epoch(
    model: MLP,
    x: np.ndarray,
    y: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    lr: float,
    epochs: int,
    seed: int,
    old_model: MLP | None = None,
    old_classes: list[int] | None = None,
    temperature: float = 2.0,
    distillation_alpha: float = 1.0,
    ewc_snapshots: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]] | None = None,
    ewc_lambda: float = 0.0,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    old_classes = old_classes or []
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader(x, y, batch_size, seed + epoch):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            if old_model is not None and old_classes:
                with torch.no_grad():
                    targets = F.softmax(old_model(batch_x)[:, old_classes] / temperature, dim=1)
                current = F.log_softmax(logits[:, old_classes] / temperature, dim=1)
                loss = loss + (
                    distillation_alpha
                    * F.kl_div(current, targets, reduction="batchmean")
                    * temperature**2
                )
            if ewc_snapshots:
                penalty = torch.zeros((), device=device)
                for previous_parameters, previous_fisher in ewc_snapshots:
                    for name, parameter in model.named_parameters():
                        penalty = penalty + (
                            previous_fisher[name]
                            * (parameter - previous_parameters[name]).square()
                        ).sum()
                loss = loss + ewc_lambda * penalty
            loss.backward()
            optimizer.step()


def compute_fisher(
    model: MLP,
    x: np.ndarray,
    y: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    sample_size: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Estimate the Phase I empirical diagonal Fisher on a fixed sample."""
    rng = np.random.default_rng(seed)
    chosen = np.arange(len(x))
    if len(chosen) > sample_size:
        chosen = np.sort(rng.choice(chosen, sample_size, replace=False))
    dataset = TensorDataset(torch.from_numpy(x[chosen]), torch.from_numpy(y[chosen]))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    fisher = {
        name: torch.zeros_like(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    samples = 0
    model.eval()
    for batch_x, batch_y in data_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        model.zero_grad(set_to_none=True)
        loss = F.nll_loss(F.log_softmax(model(batch_x), dim=1), batch_y)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                fisher[name] += parameter.grad.detach().square() * batch_x.shape[0]
        samples += batch_x.shape[0]
    for name in fisher:
        fisher[name] /= max(samples, 1)
    model.zero_grad(set_to_none=True)
    return fisher


def extract_features(model: MLP, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            parts.append(model.features(batch).cpu().numpy())
    return np.concatenate(parts, axis=0)


def herding_select(features: np.ndarray, count: int) -> np.ndarray:
    count = min(count, len(features))
    if count == len(features):
        return np.arange(len(features), dtype=np.int64)
    target_mean = features.mean(axis=0)
    feature_norm = np.einsum("ij,ij->i", features, features)
    selected: list[int] = []
    available = np.ones(len(features), dtype=bool)
    running = np.zeros(features.shape[1], dtype=features.dtype)
    for step in range(count):
        target = (step + 1) * target_mean - running
        distance = feature_norm - 2.0 * (features @ target)
        distance[~available] = np.inf
        index = int(np.argmin(distance))
        selected.append(index)
        available[index] = False
        running += features[index]
    return np.asarray(selected, dtype=np.int64)


def prototypes_from_buffer(
    model: MLP,
    buffer_x: list[np.ndarray],
    buffer_y: list[np.ndarray],
    device: torch.device,
    batch_size: int,
) -> dict[int, torch.Tensor]:
    all_x = np.concatenate(buffer_x, axis=0)
    all_y = np.concatenate(buffer_y, axis=0)
    features = extract_features(model, all_x, device, batch_size)
    result: dict[int, torch.Tensor] = {}
    for class_id in sorted(set(all_y.tolist())):
        mean = features[all_y == class_id].mean(axis=0)
        mean /= np.linalg.norm(mean) + 1e-12
        result[int(class_id)] = torch.from_numpy(mean.astype(np.float32))
    return result


def evaluate_recalls(
    scorer: DecisionScore,
    data: PreparedData,
    seen_classes: list[int],
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    scorer.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(data.x_test), batch_size):
            batch = torch.from_numpy(data.x_test[start : start + batch_size]).to(device)
            predictions.append(scorer(batch).argmax(dim=1).cpu().numpy())
    predicted = np.concatenate(predictions)
    recalls: dict[str, float] = {}
    for class_id in seen_classes:
        mask = data.y_test == class_id
        recalls[str(class_id)] = float(np.mean(predicted[mask] == class_id)) if np.any(mask) else float("nan")
    return recalls


def save_probe_background(data: PreparedData, output: Path, seed: int) -> Path:
    rng = np.random.default_rng(seed + 991)
    probe_x: list[np.ndarray] = []
    probe_y: list[np.ndarray] = []
    probe_original_index: list[np.ndarray] = []
    for class_id in range(len(data.class_order)):
        candidates = np.flatnonzero(data.y_test == class_id)
        chosen = candidates if len(candidates) <= 200 else np.sort(rng.choice(candidates, 200, replace=False))
        probe_x.append(data.x_test[chosen])
        probe_y.append(np.full(len(chosen), class_id, dtype=np.int64))
        probe_original_index.append(chosen.astype(np.int64))
    task0_rows = np.flatnonzero(np.isin(data.y_train, data.tasks[0]))
    background_rows = np.sort(rng.choice(task0_rows, min(100, len(task0_rows)), replace=False))
    path = output / "probe_background.npz"
    np.savez_compressed(
        path,
        probe_x=np.concatenate(probe_x, axis=0),
        probe_y=np.concatenate(probe_y, axis=0),
        probe_original_index=np.concatenate(probe_original_index, axis=0),
        background_x=data.x_train[background_rows],
        background_y=data.y_train[background_rows],
        background_train_index=background_rows,
        feature_names=np.asarray(data.feature_names, dtype=str),
        class_names=np.asarray(data.class_order, dtype=str),
    )
    return path


def run(config: dict, config_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output = (repo_root / config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    seed_everything(seed)
    device_name = str(config.get("device", "cuda"))
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    data_root = (repo_root / config["data_root"]).resolve()
    method = str(config["method"]).lower().replace("_", "-")
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"unsupported method: {method}; expected one of {sorted(SUPPORTED_METHODS)}")
    print(f"dataset={config['dataset']} method={method} seed={seed} device={device}")
    print(f"data_root={data_root}")
    data = prepare_dataset(str(config["dataset"]), data_root)
    print(f"train={data.x_train.shape} test={data.x_test.shape} tasks={data.tasks}")
    probe_path = save_probe_background(data, output, seed)

    hidden = int(config.get("hidden", 128))
    batch_size = int(config.get("batch_size", 256))
    epochs = int(config.get("epochs_per_task", 8))
    lr = float(config.get("learning_rate", 1e-3))
    buffer_per_class = int(config.get("buffer_per_class", 200))
    ewc_lambda = float(config.get("ewc_lambda", 5000.0))
    fisher_sample_size = int(config.get("fisher_sample_size", 2000))
    distillation_temperature = float(config.get("distillation_temperature", 2.0))
    distillation_alpha = float(config.get("distillation_alpha", 1.0))
    n_classes = len(data.class_order)
    model = MLP(data.x_train.shape[1], hidden, n_classes).to(device)
    buffer_x: list[np.ndarray] = []
    buffer_y: list[np.ndarray] = []
    old_model: MLP | None = None
    ewc_snapshots: list[
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
    ] = []
    seen_classes: list[int] = []
    checkpoint_records: list[dict] = []
    started = time.perf_counter()

    for checkpoint, task in enumerate(data.tasks):
        mask = np.isin(data.y_train, task)
        task_x = data.x_train[mask]
        task_y = data.y_train[mask]
        replay_method = method in {"replay", "replay-herding", "replay-dpmeans", "icarl"}
        if buffer_x and replay_method:
            train_x = np.concatenate([task_x, *buffer_x], axis=0)
            train_y = np.concatenate([task_y, *buffer_y], axis=0)
        else:
            train_x, train_y = task_x, task_y
        print(f"checkpoint={checkpoint} task={task} train_rows={len(train_x)}")
        train_epoch(
            model,
            train_x,
            train_y,
            device=device,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            seed=seed + checkpoint * 1000,
            old_model=old_model if method in {"lwf", "icarl"} else None,
            old_classes=list(seen_classes),
            temperature=distillation_temperature,
            distillation_alpha=distillation_alpha,
            ewc_snapshots=ewc_snapshots if method == "ewc" else None,
            ewc_lambda=ewc_lambda,
        )
        if method in {"lwf", "icarl"}:
            old_model = copy.deepcopy(model).eval()
            for parameter in old_model.parameters():
                parameter.requires_grad_(False)
        if method == "ewc":
            fisher = compute_fisher(
                model,
                task_x,
                task_y,
                device=device,
                batch_size=batch_size,
                sample_size=fisher_sample_size,
                seed=seed + checkpoint * 1000 + 701,
            )
            parameters = {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            ewc_snapshots.append((parameters, fisher))
        if replay_method:
            features = (
                extract_features(model, task_x, device, batch_size)
                if method in {"replay-herding", "replay-dpmeans", "icarl"}
                else None
            )
            for class_id in task:
                local = np.flatnonzero(task_y == class_id)
                if method == "replay":
                    rng = np.random.default_rng(seed * 10000 + checkpoint * 100 + class_id)
                    selected = rng.choice(local, min(buffer_per_class, len(local)), replace=False)
                elif method in {"replay-herding", "icarl"}:
                    selected = local[herding_select(features[local], buffer_per_class)]
                else:
                    np.random.seed(seed * 10000 + checkpoint * 100 + class_id)
                    selected = local[_dpmeans_select(features[local], buffer_per_class)]
                buffer_x.append(task_x[selected])
                buffer_y.append(task_y[selected])
        seen_classes.extend(task)
        prototypes = (
            prototypes_from_buffer(model, buffer_x, buffer_y, device, batch_size)
            if method == "icarl"
            else {}
        )
        inference = "nme" if method == "icarl" else "logit"
        scorer = DecisionScore(model, inference, n_classes, {k: v.to(device) for k, v in prototypes.items()}).to(device)
        recalls = evaluate_recalls(scorer, data, seen_classes, device, batch_size)
        checkpoint_path = output / f"checkpoint_{checkpoint:03d}.pt"
        torch.save(
            {
                "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "prototypes": prototypes,
                "checkpoint": checkpoint,
                "seen_classes": list(seen_classes),
                "task": list(task),
                "inference": inference,
                "input_dim": data.x_train.shape[1],
                "hidden": hidden,
                "n_classes": n_classes,
                "class_order": data.class_order,
            },
            checkpoint_path,
        )
        checkpoint_records.append(
            {
                "checkpoint": checkpoint,
                "path": checkpoint_path.name,
                "seen_classes": list(seen_classes),
                "per_class_recall": recalls,
            }
        )
        print(f"checkpoint={checkpoint} recalls={json.dumps(recalls, sort_keys=True)}")

    result = {
        "dataset": config["dataset"],
        "method": method,
        "seed": seed,
        "git_commit": source_commit(repo_root),
        "config": str(config_path.relative_to(repo_root)).replace("\\", "/"),
        "command": " ".join(sys.argv),
        "log_path": config["log_path"],
        "data_sources": data.source_files,
        "task0_only_preprocessing": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "class_order": data.class_order,
        "tasks": data.tasks,
        "feature_count": len(data.feature_names),
        "train_rows": int(len(data.x_train)),
        "test_rows": int(len(data.x_test)),
        "method_parameters": {
            "buffer_per_class": buffer_per_class if replay_method else None,
            "distillation_temperature": (
                distillation_temperature if method in {"lwf", "icarl"} else None
            ),
            "distillation_alpha": (
                distillation_alpha if method in {"lwf", "icarl"} else None
            ),
            "ewc_lambda": ewc_lambda if method == "ewc" else None,
            "fisher_sample_size": fisher_sample_size if method == "ewc" else None,
        },
        "probe_background": probe_path.name,
        "checkpoints": checkpoint_records,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "training_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved={output / 'training_result.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = (repo_root / args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    log_path = (repo_root / config["log_path"]).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        with redirect_stdout(Tee(sys.__stdout__, log_file)), redirect_stderr(Tee(sys.__stderr__, log_file)):
            try:
                run(config, config_path)
            except Exception:
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
