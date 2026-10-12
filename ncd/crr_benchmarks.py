"""Ground-truth benchmarks and matched learners for CRR (work order section 12).

The module deliberately keeps benchmark construction separate from reporting:
``hijack`` injects a label-correlated marker into an old class, while
``necessity`` creates a new class that shares the old class' strongest feature
and differs on its second-ranked permutation-importance feature.  The latter
forces a legitimate change in the old-vs-new decision evidence.

All methods start from the same pre-boundary checkpoint and receive the same
number of replay examples.  Reliance losses use a stochastic, one-feature-per-
minibatch marginal-resampling estimator, matching Section 12.3's random j and
keeping training cost bounded.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from ncd.data import PreparedData
from ncd.groundtruth import append_markers, correlated_markers, pure_noise_markers
from ncd.model import MLP
from ncd.train_cil import extract_features, herding_select, loader, seed_everything


METHODS = (
    "replay",
    "icarl",
    "replay_lwf",
    "kd_perturbed",
    "rcl_tab",
    "global_ir",
    "oldpair_kd",
    "crr",
)


@dataclass(frozen=True)
class Boundary:
    old_classes: tuple[int, ...]
    new_classes: tuple[int, ...]
    target_class: int
    contrast_class: int
    synthetic_class: int


@dataclass
class Benchmark:
    name: str
    strength_name: str
    strength: float
    train_x: np.ndarray
    train_y: np.ndarray
    eval_x: np.ndarray
    eval_y: np.ndarray
    metadata: dict


def boundary_for(data: PreparedData) -> Boundary:
    """Return the earliest boundary with at least two old classes.

    NSL-KDD has one class per task, so the 1->2 boundary is required. UNSW has
    two classes per task and can use its 0->1 boundary.
    """
    old_task_count = 2 if len(data.tasks[0]) == 1 else 1
    old_classes = tuple(c for task in data.tasks[:old_task_count] for c in task)
    new_classes = tuple(data.tasks[old_task_count])
    if len(old_classes) < 2 or not new_classes:
        raise ValueError("CRR requires at least two old classes and one new class")
    return Boundary(
        old_classes=old_classes,
        new_classes=new_classes,
        target_class=old_classes[-1],
        contrast_class=old_classes[0],
        synthetic_class=new_classes[0],
    )


def _cap_per_class(
    x: np.ndarray, y: np.ndarray, classes: Iterable[int], cap: int | None, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isin(y, list(classes))
    x, y = x[mask], y[mask]
    if cap is None:
        return x, y
    chosen: list[np.ndarray] = []
    for class_id in sorted(set(y.tolist())):
        local = np.flatnonzero(y == class_id)
        rng = np.random.default_rng(seed * 1009 + class_id)
        if len(local) > cap:
            local = np.sort(rng.choice(local, cap, replace=False))
        chosen.append(local)
    rows = np.concatenate(chosen)
    return x[rows], y[rows]


def add_noise_markers(x: np.ndarray, columns: int, seed: int) -> np.ndarray:
    return append_markers(x, pure_noise_markers(len(x), columns, seed))


def train_base_checkpoint(
    data: PreparedData,
    boundary: Boundary,
    *,
    seed: int,
    marker_columns: int,
    hidden: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    rows_per_class: int | None = None,
) -> tuple[MLP, np.ndarray, np.ndarray]:
    """Train the common old-class checkpoint used by every compared method."""
    seed_everything(seed)
    x, y = _cap_per_class(
        data.x_train, data.y_train, boundary.old_classes, rows_per_class, seed
    )
    augmented = add_noise_markers(x, marker_columns, seed + 101)
    model = MLP(augmented.shape[1], hidden, len(data.class_order)).to(device)
    _train(
        model,
        augmented,
        y,
        method="replay",
        old_model=None,
        old_classes=(),
        seen_classes=boundary.old_classes,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        regularization=0.0,
        temperature=2.0,
    )
    return model, augmented, y


def replay_buffer(
    model: MLP,
    x: np.ndarray,
    y: np.ndarray,
    old_classes: tuple[int, ...],
    *,
    per_class: int,
    seed: int,
    method: str,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected: list[np.ndarray] = []
    embeddings = None
    if method == "icarl":
        embeddings = extract_features(model, x, device, batch_size)
    for class_id in old_classes:
        local = np.flatnonzero(y == class_id)
        count = min(per_class, len(local))
        if method == "icarl":
            assert embeddings is not None
            chosen = local[herding_select(embeddings[local], count)]
        else:
            rng = np.random.default_rng(seed * 10007 + class_id)
            chosen = np.sort(rng.choice(local, count, replace=False))
        selected.append(chosen)
    rows = np.concatenate(selected)
    return x[rows], y[rows]


def build_hijack(
    data: PreparedData,
    boundary: Boundary,
    buffer_x: np.ndarray,
    buffer_y: np.ndarray,
    *,
    rho: float,
    marker_columns: int,
    seed: int,
    rows_per_class: int | None = None,
) -> Benchmark:
    current_x, current_y = _cap_per_class(
        data.x_train, data.y_train, boundary.new_classes, rows_per_class, seed + 5
    )
    current_x = add_noise_markers(current_x, marker_columns, seed + 201)
    clean_replay = buffer_x.copy()
    replay_marker = correlated_markers(
        buffer_y,
        target_class=boundary.target_class,
        rho=rho,
        columns=marker_columns,
        seed=seed + 202,
    )
    clean_replay[:, -marker_columns:] = replay_marker
    train_x = np.concatenate([current_x, clean_replay])
    train_y = np.concatenate([current_y, buffer_y])

    seen = boundary.old_classes + boundary.new_classes
    eval_x, eval_y = _cap_per_class(
        data.x_test, data.y_test, seen, rows_per_class, seed + 6
    )
    test_marker = correlated_markers(
        eval_y,
        target_class=boundary.target_class,
        rho=rho,
        columns=marker_columns,
        seed=seed + 203,
    )
    eval_x = append_markers(eval_x, test_marker)
    return Benchmark(
        name="hijack",
        strength_name="rho",
        strength=float(rho),
        train_x=train_x,
        train_y=train_y,
        eval_x=eval_x,
        eval_y=eval_y,
        metadata={"marker_columns": marker_columns},
    )


def _predict_logits(
    model: MLP, x: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            parts.append(model(batch).cpu().numpy())
    return np.concatenate(parts)


def permutation_importance_pair(
    model: MLP,
    x: np.ndarray,
    y: np.ndarray,
    *,
    target: int,
    contrast: int,
    real_features: int,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> tuple[int, int, np.ndarray]:
    """Rank features by target-vs-contrast balanced-accuracy permutation loss."""
    pair_mask = np.isin(y, [target, contrast])
    pair_x, pair_y = x[pair_mask], y[pair_mask]
    base = _predict_logits(model, pair_x, device, batch_size)
    base_pred = np.where(base[:, target] >= base[:, contrast], target, contrast)

    def balanced_accuracy(pred: np.ndarray) -> float:
        return float(
            np.mean(
                [np.mean(pred[pair_y == c] == c) for c in (target, contrast)]
            )
        )

    base_score = balanced_accuracy(base_pred)
    drops = np.zeros(real_features, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for feature in range(real_features):
        changed = pair_x.copy()
        changed[:, feature] = changed[rng.permutation(len(changed)), feature]
        logits = _predict_logits(model, changed, device, batch_size)
        pred = np.where(logits[:, target] >= logits[:, contrast], target, contrast)
        drops[feature] = base_score - balanced_accuracy(pred)
    order = np.argsort(-drops, kind="stable")
    return int(order[0]), int(order[1]), drops


def build_necessity(
    data: PreparedData,
    boundary: Boundary,
    buffer_x: np.ndarray,
    buffer_y: np.ndarray,
    base_model: MLP,
    *,
    k: float,
    marker_columns: int,
    seed: int,
    device: torch.device,
    batch_size: int,
    rows_per_class: int | None = None,
    noise_scale: float = 0.02,
) -> Benchmark:
    old_eval_x, old_eval_y = _cap_per_class(
        data.x_test, data.y_test, boundary.old_classes, rows_per_class, seed + 11
    )
    old_eval_x = add_noise_markers(old_eval_x, marker_columns, seed + 211)
    f1, f2, drops = permutation_importance_pair(
        base_model,
        old_eval_x,
        old_eval_y,
        target=boundary.target_class,
        contrast=boundary.contrast_class,
        real_features=data.x_train.shape[1],
        seed=seed + 212,
        device=device,
        batch_size=batch_size,
    )

    train_source = data.x_train[data.y_train == boundary.target_class]
    test_source = data.x_test[data.y_test == boundary.target_class]
    if rows_per_class is not None:
        train_source = train_source[:rows_per_class]
        test_source = test_source[:rows_per_class]
    scale = float(np.std(train_source[:, f2]))
    if scale < 1e-6:
        scale = 1.0

    def synthesize(source: np.ndarray, item_seed: int) -> np.ndarray:
        rng = np.random.default_rng(item_seed)
        result = source.astype(np.float32, copy=True)
        feature_scale = np.std(train_source, axis=0).astype(np.float32)
        feature_scale[feature_scale < 1e-6] = 1.0
        noise = rng.normal(0.0, noise_scale, size=result.shape).astype(np.float32)
        result += noise * feature_scale
        result[:, f1] = source[:, f1]  # registered invariant
        result[:, f2] = source[:, f2] + float(k) * scale
        return result

    synthetic_train = add_noise_markers(
        synthesize(train_source, seed + 213), marker_columns, seed + 214
    )
    synthetic_test = add_noise_markers(
        synthesize(test_source, seed + 215), marker_columns, seed + 216
    )
    synthetic_train_y = np.full(
        len(synthetic_train), boundary.synthetic_class, dtype=np.int64
    )
    synthetic_test_y = np.full(
        len(synthetic_test), boundary.synthetic_class, dtype=np.int64
    )

    target_test = data.x_test[data.y_test == boundary.target_class]
    if rows_per_class is not None:
        target_test = target_test[:rows_per_class]
    target_test = add_noise_markers(target_test, marker_columns, seed + 217)
    target_y = np.full(len(target_test), boundary.target_class, dtype=np.int64)
    eval_x = np.concatenate([target_test, synthetic_test])
    eval_y = np.concatenate([target_y, synthetic_test_y])
    return Benchmark(
        name="necessity",
        strength_name="k",
        strength=float(k),
        train_x=np.concatenate([synthetic_train, buffer_x]),
        train_y=np.concatenate([synthetic_train_y, buffer_y]),
        eval_x=eval_x,
        eval_y=eval_y,
        metadata={
            "f1": f1,
            "f2": f2,
            "f2_train_std": scale,
            "permutation_importance_drops": drops.tolist(),
            "noise_scale": noise_scale,
            "f1_max_abs_change_train": float(
                np.max(np.abs(synthetic_train[:, f1] - train_source[:, f1]))
            ),
        },
    )


def _pair_effects(
    clean: torch.Tensor,
    changed: torch.Tensor,
    labels: torch.Tensor,
    pair_classes: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[torch.Tensor] = []
    owner: list[torch.Tensor] = []
    for class_id in pair_classes:
        mask = labels == class_id
        if not torch.any(mask):
            continue
        others = [other for other in pair_classes if other != class_id]
        if not others:
            continue
        clean_margin = clean[mask, class_id, None] - clean[mask][:, others]
        changed_margin = changed[mask, class_id, None] - changed[mask][:, others]
        rows.append(clean_margin - changed_margin)
        owner.append(torch.full((int(mask.sum()),), class_id, device=labels.device))
    if not rows:
        return clean.new_zeros((0, 0)), labels.new_zeros((0,))
    return torch.cat(rows), torch.cat(owner)


def _js_from_effects(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    p = F.softmax(torch.abs(teacher), dim=1)
    q = F.softmax(torch.abs(student), dim=1)
    m = 0.5 * (p + q)
    return 0.5 * (
        F.kl_div(torch.log(m + 1e-8), p, reduction="batchmean")
        + F.kl_div(torch.log(m + 1e-8), q, reduction="batchmean")
    )


def _pair_margin_mse(
    student: torch.Tensor,
    teacher: torch.Tensor,
    labels: torch.Tensor,
    old_classes: tuple[int, ...],
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for class_id in old_classes:
        mask = labels == class_id
        others = [other for other in old_classes if other != class_id]
        if torch.any(mask) and others:
            s = student[mask, class_id, None] - student[mask][:, others]
            t = teacher[mask, class_id, None] - teacher[mask][:, others]
            scale = t.detach().abs().mean().clamp_min(1e-3)
            losses.append(((s - t) / scale).square().mean())
    return torch.stack(losses).mean() if losses else student.new_zeros(())


def _train(
    model: MLP,
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
    old_model: MLP | None,
    old_classes: tuple[int, ...],
    seen_classes: tuple[int, ...],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    regularization: float,
    temperature: float,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = next(model.parameters()).device
    rng = np.random.default_rng(seed + 319)
    started = time.perf_counter()
    if old_model is not None:
        old_model.eval()
        for parameter in old_model.parameters():
            parameter.requires_grad_(False)
    for epoch in range(epochs):
        model.train()
        for bx, by in loader(x, y, batch_size, seed + epoch):
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = F.cross_entropy(logits, by)
            if old_model is not None and method != "replay":
                feature = int(rng.integers(0, bx.shape[1]))
                changed = bx.clone()
                order = torch.randperm(len(bx), device=device)
                changed[:, feature] = changed[order, feature]
                with torch.no_grad():
                    teacher = old_model(bx)
                    teacher_changed = old_model(changed)

                if method in {"icarl", "replay_lwf"}:
                    target = F.softmax(
                        teacher[:, list(old_classes)] / temperature, dim=1
                    )
                    current = F.log_softmax(
                        logits[:, list(old_classes)] / temperature, dim=1
                    )
                    extra = F.kl_div(current, target, reduction="batchmean") * temperature**2
                elif method == "kd_perturbed":
                    student_changed = model(changed)
                    target = F.softmax(
                        teacher_changed[:, list(old_classes)] / temperature, dim=1
                    )
                    current = F.log_softmax(
                        student_changed[:, list(old_classes)] / temperature, dim=1
                    )
                    extra = F.kl_div(current, target, reduction="batchmean") * temperature**2
                elif method == "oldpair_kd":
                    extra = _pair_margin_mse(logits, teacher, by, old_classes)
                else:
                    student_changed = model(changed)
                    pair_classes = old_classes if method == "crr" else seen_classes
                    student_effect, _ = _pair_effects(
                        logits, student_changed, by, pair_classes
                    )
                    teacher_effect, _ = _pair_effects(
                        teacher, teacher_changed, by, pair_classes
                    )
                    if student_effect.numel() == 0:
                        extra = logits.new_zeros(())
                    elif method == "rcl_tab":
                        extra = _js_from_effects(student_effect, teacher_effect)
                    else:
                        scale = teacher_effect.detach().abs().mean().clamp_min(1e-3)
                        extra = ((student_effect - teacher_effect) / scale).square().mean()
                loss = loss + regularization * extra
            loss.backward()
            optimizer.step()
    return time.perf_counter() - started


def _nme_predictions(
    model: MLP,
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    seen_classes: tuple[int, ...],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    train_features = extract_features(model, train_x, device, batch_size)
    eval_features = extract_features(model, eval_x, device, batch_size)
    prototypes = []
    valid = []
    for class_id in seen_classes:
        mask = train_y == class_id
        if np.any(mask):
            mean = train_features[mask].mean(axis=0)
            mean /= np.linalg.norm(mean) + 1e-12
            prototypes.append(mean)
            valid.append(class_id)
    ef = eval_features / np.maximum(np.linalg.norm(eval_features, axis=1, keepdims=True), 1e-12)
    distance = ((ef[:, None, :] - np.asarray(prototypes)[None, :, :]) ** 2).sum(axis=2)
    return np.asarray(valid, dtype=np.int64)[np.argmin(distance, axis=1)]


def _logit_predictions(
    model: MLP,
    x: np.ndarray,
    seen_classes: tuple[int, ...],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    logits = _predict_logits(model, x, device, batch_size)
    local = np.argmax(logits[:, list(seen_classes)], axis=1)
    return np.asarray(seen_classes, dtype=np.int64)[local]


def _recall(pred: np.ndarray, y: np.ndarray, class_id: int) -> float:
    mask = y == class_id
    return float(np.mean(pred[mask] == class_id)) if np.any(mask) else math.nan


def _margin_dependency(
    model: MLP,
    x: np.ndarray,
    *,
    target: int,
    contrast: int,
    feature: int,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> float:
    logits = _predict_logits(model, x, device, batch_size)
    changed = x.copy()
    rng = np.random.default_rng(seed)
    changed[:, feature] = changed[rng.permutation(len(changed)), feature]
    changed_logits = _predict_logits(model, changed, device, batch_size)
    margin = logits[:, target] - logits[:, contrast]
    changed_margin = changed_logits[:, target] - changed_logits[:, contrast]
    return float(np.mean(np.abs(margin - changed_margin)))


def run_method(
    base_model: MLP,
    benchmark: Benchmark,
    boundary: Boundary,
    *,
    method: str,
    regularization: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    temperature: float,
    seed: int,
    device: torch.device,
) -> tuple[MLP, dict]:
    if method not in METHODS:
        raise ValueError(f"unknown CRR method: {method}")
    model = copy.deepcopy(base_model).to(device)
    teacher = copy.deepcopy(base_model).to(device).eval()
    seen = tuple(sorted(set(benchmark.train_y.tolist())))
    seconds = _train(
        model,
        benchmark.train_x,
        benchmark.train_y,
        method=method,
        old_model=teacher,
        old_classes=boundary.old_classes,
        seen_classes=seen,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        regularization=regularization,
        temperature=temperature,
    )
    if method == "icarl":
        pred = _nme_predictions(
            model,
            benchmark.train_x,
            benchmark.train_y,
            benchmark.eval_x,
            seen,
            device,
            batch_size,
        )
    else:
        pred = _logit_predictions(model, benchmark.eval_x, seen, device, batch_size)
    metric_labels = sorted(set(int(value) for value in benchmark.eval_y.tolist()))
    metrics = {
        "training_seconds": seconds,
        "macro_f1": float(
            f1_score(
                benchmark.eval_y,
                pred,
                labels=metric_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "recalls": {str(c): _recall(pred, benchmark.eval_y, c) for c in metric_labels},
    }
    if benchmark.name == "hijack":
        marker_columns = int(benchmark.metadata["marker_columns"])
        changed = benchmark.eval_x.copy()
        rng = np.random.default_rng(seed + 717)
        changed[:, -marker_columns:] = changed[
            rng.permutation(len(changed)), -marker_columns:
        ]
        decorrelated = benchmark.eval_x.copy()
        decorrelated[:, -marker_columns:] = pure_noise_markers(
            len(decorrelated), marker_columns, seed + 718
        )
        predict = _nme_predictions if method == "icarl" else None
        if predict is not None:
            changed_pred = predict(model, benchmark.train_x, benchmark.train_y, changed, seen, device, batch_size)
            decorrelated_pred = predict(model, benchmark.train_x, benchmark.train_y, decorrelated, seen, device, batch_size)
        else:
            changed_pred = _logit_predictions(model, changed, seen, device, batch_size)
            decorrelated_pred = _logit_predictions(model, decorrelated, seen, device, batch_size)
        target = boundary.target_class
        metrics.update(
            {
                "target_recall_correlated": _recall(pred, benchmark.eval_y, target),
                "target_recall_marker_permuted": _recall(changed_pred, benchmark.eval_y, target),
                "marker_recall_drop": _recall(pred, benchmark.eval_y, target)
                - _recall(changed_pred, benchmark.eval_y, target),
                "target_recall_decorrelated": _recall(decorrelated_pred, benchmark.eval_y, target),
            }
        )
    else:
        c, d = boundary.target_class, boundary.synthetic_class
        c_mask, d_mask = benchmark.eval_y == c, benchmark.eval_y == d
        f2 = int(benchmark.metadata["f2"])
        base_dependency = _margin_dependency(
            base_model,
            benchmark.eval_x[c_mask],
            target=c,
            contrast=d,
            feature=f2,
            seed=seed + 719,
            device=device,
            batch_size=batch_size,
        )
        learned_dependency = _margin_dependency(
            model,
            benchmark.eval_x[c_mask],
            target=c,
            contrast=d,
            feature=f2,
            seed=seed + 719,
            device=device,
            batch_size=batch_size,
        )
        metrics.update(
            {
                "c_recall": _recall(pred, benchmark.eval_y, c),
                "d_recall": _recall(pred, benchmark.eval_y, d),
                "c_to_d": float(np.mean(pred[c_mask] == d)),
                "d_to_c": float(np.mean(pred[d_mask] == c)),
                "pair_confusion_rate": float(
                    0.5 * (np.mean(pred[c_mask] == d) + np.mean(pred[d_mask] == c))
                ),
                "base_c_f2_dependency": base_dependency,
                "c_f2_dependency": learned_dependency,
                "c_f2_dependency_change": learned_dependency - base_dependency,
            }
        )
    return model, metrics
