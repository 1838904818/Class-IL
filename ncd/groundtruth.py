"""Run the first real-data Exp 2 intervention-ground-truth slice.

The implementation follows WORK_ORDER_2026-10-09 section 4.  It trains a
Replay model on the first NSL-KDD task, injects a synthetic shortcut while the
next class is learned, measures dependence by permuting the marker at test
time, and compares B1/B2/B4 with a permutation-calibrated NCD statistic.  N1
and a function-preserving N2 reparameterization are evaluated alongside P1.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

# PyTorch requires this before the first CUDA context is created when
# deterministic algorithms are enabled.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from scipy.stats import kendalltau, rankdata
from sklearn.metrics import roc_auc_score

from ncd.data import PreparedData, prepare_dataset
from ncd.explain import (
    EXPLAINERS,
    bootstrap_estimates,
    expected_gradients,
    feature_ablation,
    grad_x_input,
)
from ncd.model import DecisionScore, MLP
from ncd.train_cil import Tee, seed_everything, train_epoch


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return vector / (vector.sum() + 1e-12)


def topk(vector: np.ndarray, k: int) -> set[int]:
    return set(np.argsort(-vector, kind="stable")[: min(k, len(vector))].tolist())


def jaccard(left: np.ndarray, right: np.ndarray, k: int) -> float:
    a, b = topk(left, k), topk(right, k)
    return len(a & b) / len(a | b)


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    similarity = np.dot(left, right) / max(float(denom), 1e-12)
    return float(1.0 - np.clip(similarity, -1.0, 1.0))


def kendall_distance(left: np.ndarray, right: np.ndarray) -> float:
    value = kendalltau(left, right, variant="b").statistic
    return float(1.0 - value) if np.isfinite(value) else float("nan")


def clex_explanation_drift(left: np.ndarray, right: np.ndarray) -> float:
    """CLEX ED from Cossu et al., Neurocomputing 597 (2024) 127960.

    Equation 6 compares the total positive contribution of two explanations
    and divides their squared difference by the feature count.
    """
    left_mass = float(np.maximum(left, 0.0).sum())
    right_mass = float(np.maximum(right, 0.0).sum())
    return (left_mass - right_mass) ** 2 / len(left)


def rank_unit_vectors(estimates: np.ndarray) -> np.ndarray:
    ranks = rankdata(estimates, axis=1, method="average").astype(np.float64)
    ranks -= ranks.mean(axis=1, keepdims=True)
    return ranks / np.maximum(np.linalg.norm(ranks, axis=1, keepdims=True), 1e-12)


def ncd_permutation_test(
    left: np.ndarray,
    right: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> dict[str, float]:
    """One-sided permutation test for an increase in full-ranking distance."""
    if left.shape != right.shape:
        raise ValueError(f"estimate shape mismatch: {left.shape} != {right.shape}")
    joined = np.concatenate([rank_unit_vectors(left), rank_unit_vectors(right)], axis=0)
    distance = 1.0 - np.clip(joined @ joined.T, -1.0, 1.0)
    group_size = len(left)

    def statistic(first: np.ndarray, second: np.ndarray) -> tuple[float, float, float]:
        cross = distance[np.ix_(first, second)].reshape(-1)
        first_within = distance[np.ix_(first, first)][np.triu_indices(len(first), 1)]
        second_within = distance[np.ix_(second, second)][np.triu_indices(len(second), 1)]
        within = np.concatenate([first_within, second_within])
        between_median = float(np.median(cross))
        within_median = float(np.median(within))
        return between_median - within_median, between_median, within_median

    original_first = np.arange(group_size)
    original_second = np.arange(group_size, group_size * 2)
    observed, between_median, within_median = statistic(original_first, original_second)
    rng = np.random.default_rng(seed)
    exceed = 0
    universe = np.arange(group_size * 2)
    for _ in range(permutations):
        shuffled = rng.permutation(universe)
        permuted, _, _ = statistic(shuffled[:group_size], shuffled[group_size:])
        exceed += int(permuted >= observed)
    return {
        "ncd_effect": observed,
        "between_median_1_minus_spearman": between_median,
        "within_median_1_minus_spearman": within_median,
        "p_value": (exceed + 1) / (permutations + 1),
    }


def benjamini_hochberg(values: list[float]) -> list[float]:
    count = len(values)
    order = np.argsort(values)
    adjusted = np.empty(count, dtype=np.float64)
    running = 1.0
    for reverse_rank in range(count - 1, -1, -1):
        index = int(order[reverse_rank])
        rank = reverse_rank + 1
        running = min(running, values[index] * count / rank)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def pure_noise_markers(rows: int, columns: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(rows, columns), dtype=np.int8).astype(np.float32)


def correlated_markers(
    labels: np.ndarray,
    *,
    target_class: int,
    rho: float,
    columns: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    active = (labels == target_class) & (rng.random(len(labels)) < rho)
    markers = np.zeros((len(labels), columns), dtype=np.float32)
    markers[active] = 1.0
    return markers


def append_markers(features: np.ndarray, markers: np.ndarray) -> np.ndarray:
    return np.concatenate([features.astype(np.float32, copy=False), markers], axis=1)


def replay_indices(
    features: np.ndarray,
    labels: np.ndarray,
    classes: list[int],
    *,
    per_class: int,
    seed: int,
) -> np.ndarray:
    selected: list[np.ndarray] = []
    for class_id in classes:
        local = np.flatnonzero(labels == class_id)
        rng = np.random.default_rng(seed * 10000 + class_id)
        selected.append(rng.choice(local, min(per_class, len(local)), replace=False))
    return np.concatenate(selected)


def scorer_for(model: MLP, n_classes: int, device: torch.device) -> DecisionScore:
    return DecisionScore(model, "logit", n_classes).to(device).eval()


def predictions(
    scorer: DecisionScore, features: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            parts.append(scorer(batch).argmax(dim=1).cpu().numpy())
    return np.concatenate(parts)


def class_recall(predicted: np.ndarray, labels: np.ndarray, target: int) -> float:
    mask = labels == target
    return float(np.mean(predicted[mask] == target))


def marker_effect(
    scorer: DecisionScore,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    target: int,
    marker_columns: int,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> dict[str, float]:
    original = class_recall(predictions(scorer, features, device, batch_size), labels, target)
    permuted = features.copy()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(permuted))
    permuted[:, -marker_columns:] = permuted[order, -marker_columns:]
    shuffled = class_recall(predictions(scorer, permuted, device, batch_size), labels, target)
    return {
        "recall_original": original,
        "recall_marker_permuted": shuffled,
        "recall_drop": original - shuffled,
    }


def feature_mask_effect(
    scorer: DecisionScore,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    target: int,
    feature_indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    original = class_recall(predictions(scorer, features, device, batch_size), labels, target)
    masked = features.copy()
    masked[:, feature_indices] = 0.0
    intervened = class_recall(predictions(scorer, masked, device, batch_size), labels, target)
    return {
        "recall_original": original,
        "recall_feature_masked": intervened,
        "recall_drop": original - intervened,
    }


def function_preserving_rescaling(model: MLP, seed: int) -> MLP:
    """Apply a non-identity, function-preserving ReLU reparameterization.

    Positive rescaling is preferable to a hidden-unit permutation here.  A
    permutation changes the floating-point accumulation order in the next
    GEMM and can exceed the work order's strict 1e-5 logit tolerance despite
    being algebraically identical.  Powers of two preserve the accumulation
    order and are exactly representable in float32.
    """
    transformed = copy.deepcopy(model).eval()
    first = transformed.features[0]
    second = transformed.features[2]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    selected = torch.randperm(first.out_features, generator=generator)[: first.out_features // 2]
    with torch.no_grad():
        first.weight[selected] = first.weight[selected] * 2.0
        first.bias[selected] = first.bias[selected] * 2.0
        second.weight[:, selected] = second.weight[:, selected] * 0.5
    return transformed


def maximum_logit_error(
    left: DecisionScore,
    right: DecisionScore,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    maximum = 0.0
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            maximum = max(maximum, float(torch.max(torch.abs(left(batch) - right(batch))).item()))
    return maximum


def explain_estimates(
    scorer: DecisionScore,
    *,
    target: int,
    probe: np.ndarray,
    background: np.ndarray,
    repeats: int,
    bootstraps: int,
    eg_nsamples: int,
    seed: int,
    device: torch.device,
) -> dict[str, dict[str, np.ndarray]]:
    probe_tensor = torch.from_numpy(probe).to(device)
    background_tensor = torch.from_numpy(background).to(device)
    baseline = background_tensor.mean(dim=0)
    result: dict[str, dict[str, np.ndarray]] = {}
    for explainer_index, explainer in enumerate(EXPLAINERS):
        item_seed = seed + explainer_index * 10000
        if explainer == "expected_gradients":
            attributes = expected_gradients(
                scorer,
                target,
                probe_tensor,
                background_tensor,
                repeats,
                eg_nsamples,
                item_seed,
            )
        elif explainer == "feature_ablation":
            single = feature_ablation(scorer, target, probe_tensor, baseline)
            attributes = np.repeat(single[None, :, :], repeats, axis=0)
        else:
            single = grad_x_input(scorer, target, probe_tensor, baseline)
            attributes = np.repeat(single[None, :, :], repeats, axis=0)
        estimates, repeat_ids, bootstrap_ids = bootstrap_estimates(
            attributes, bootstraps, item_seed + 57
        )
        repeat_estimates = np.abs(attributes).mean(axis=1)
        repeat_estimates /= repeat_estimates.sum(axis=1, keepdims=True) + 1e-12
        result[explainer] = {
            "estimates": estimates,
            "repeat_estimates": repeat_estimates.astype(np.float32),
            "single_vector": normalize(np.abs(attributes[0]).mean(axis=0)).astype(np.float32),
            "repeat_mean_vector": normalize(np.abs(attributes).mean(axis=(0, 1))).astype(np.float32),
            "signed_mean_vector": attributes.mean(axis=(0, 1)).astype(np.float32),
            "repeat_ids": repeat_ids,
            "bootstrap_ids": bootstrap_ids,
        }
    return result


def comparison_rows(
    condition: str,
    strength: float,
    left: dict[str, dict[str, np.ndarray]],
    right: dict[str, dict[str, np.ndarray]],
    *,
    truth: dict,
    config: dict,
    seed: int,
) -> list[dict]:
    rows: list[dict] = []
    for explainer_index, explainer in enumerate(EXPLAINERS):
        left_item, right_item = left[explainer], right[explainer]
        test = ncd_permutation_test(
            left_item["estimates"],
            right_item["estimates"],
            permutations=int(config["permutations"]),
            seed=seed + explainer_index * 1000,
        )
        no_bootstrap_test = ncd_permutation_test(
            left_item["repeat_estimates"],
            right_item["repeat_estimates"],
            permutations=int(config["permutations"]),
            seed=seed + 10000 + explainer_index * 1000,
        )
        left_first_repeat = left_item["estimates"][left_item["repeat_ids"] == 0]
        right_first_repeat = right_item["estimates"][right_item["repeat_ids"] == 0]
        no_repeats_test = ncd_permutation_test(
            left_first_repeat,
            right_first_repeat,
            permutations=int(config["permutations"]),
            seed=seed + 20000 + explainer_index * 1000,
        )
        b1 = jaccard(
            left_item["single_vector"], right_item["single_vector"], int(config["top_k"])
        )
        b2 = jaccard(
            left_item["repeat_mean_vector"],
            right_item["repeat_mean_vector"],
            int(config["top_k"]),
        )
        rows.append(
            {
                "condition": condition,
                "strength": strength,
                "explainer": explainer,
                "truth_positive": bool(truth["truth_positive"]),
                "silent_eligible": bool(truth["silent_eligible"]),
                "intervention_kind": truth["intervention_kind"],
                "checkpoint0_intervention_recall_drop": truth[
                    "checkpoint0_intervention_recall_drop"
                ],
                "checkpoint1_intervention_recall_drop": truth[
                    "checkpoint1_intervention_recall_drop"
                ],
                "intervention_effect_change": truth["intervention_effect_change"],
                "recall_delta": truth["recall_delta"],
                "b1_single_jaccard": b1,
                "b1_fixed_drift": b1 < float(config["fixed_jaccard_threshold"]),
                "b2_repeat_mean_jaccard": b2,
                "b2_fixed_drift": b2 < float(config["fixed_jaccard_threshold"]),
                "b3_clex_ed": clex_explanation_drift(
                    left_item["signed_mean_vector"], right_item["signed_mean_vector"]
                ),
                "b4_cosine_distance": cosine_distance(
                    left_item["repeat_mean_vector"], right_item["repeat_mean_vector"]
                ),
                "b4_kendall_distance": kendall_distance(
                    left_item["repeat_mean_vector"], right_item["repeat_mean_vector"]
                ),
                "ncd_no_bootstrap_effect": no_bootstrap_test["ncd_effect"],
                "ncd_no_bootstrap_p_value": no_bootstrap_test["p_value"],
                "ncd_no_repeats_effect": no_repeats_test["ncd_effect"],
                "ncd_no_repeats_p_value": no_repeats_test["p_value"],
                **test,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def discrimination_metrics(rows: list[dict]) -> list[dict]:
    """Compute per-explainer AUROC and TPR at a negative-control FPR <= 5%."""
    metric_getters = {
        "B1_single_Jaccard": lambda row: 1.0 - float(row["b1_single_jaccard"]),
        "B2_repeat_mean_Jaccard": lambda row: 1.0
        - float(row["b2_repeat_mean_jaccard"]),
        "B3_CLEX_ED": lambda row: float(row["b3_clex_ed"]),
        "B4_cosine": lambda row: float(row["b4_cosine_distance"]),
        "B4_Kendall": lambda row: float(row["b4_kendall_distance"]),
        "NCD": lambda row: float(row["ncd_effect"]),
    }
    output: list[dict] = []
    for explainer in EXPLAINERS:
        explainer_rows = [row for row in rows if row["explainer"] == explainer]
        negatives = [row for row in explainer_rows if row["condition"].startswith("N")]
        positives = [
            row
            for row in explainer_rows
            if row["condition"].startswith("P")
            and row["truth_positive"]
            and row["silent_eligible"]
        ]
        for metric, getter in metric_getters.items():
            negative_scores = np.asarray([getter(row) for row in negatives], dtype=np.float64)
            positive_scores = np.asarray([getter(row) for row in positives], dtype=np.float64)
            if not len(negative_scores) or not len(positive_scores):
                continue
            labels = np.concatenate(
                [np.zeros(len(negative_scores)), np.ones(len(positive_scores))]
            )
            scores = np.concatenate([negative_scores, positive_scores])
            threshold = float(np.quantile(negative_scores, 0.95, method="higher"))
            output.append(
                {
                    "explainer": explainer,
                    "metric": metric,
                    "negative_count": len(negative_scores),
                    "positive_count": len(positive_scores),
                    "auroc": float(roc_auc_score(labels, scores)),
                    "matched_threshold": threshold,
                    "matched_fpr": float(np.mean(negative_scores > threshold)),
                    "matched_tpr": float(np.mean(positive_scores > threshold)),
                }
            )
    return output


def effective_config(config: dict, smoke: bool) -> dict:
    result = copy.deepcopy(config)
    if smoke:
        result.update(
            {
                "epochs_per_task": 1,
                "rho_values": [0.9],
                "p2_mask_fractions": [1.0],
                "probe_per_class": 24,
                "background_rows": 16,
                "repeats": 2,
                "bootstraps": 2,
                "expected_gradients_nsamples": 2,
                "permutations": 20,
                "output_dir": "ncd/results/exp2/smoke",
                "log_path": "ncd/logs/exp2_groundtruth_smoke.log",
            }
        )
    return result


def run(config: dict, config_path: Path) -> None:
    if config["dataset"] != "NSL-KDD" or config["method"] != "replay":
        raise ValueError("the first Exp 2 slice is fixed to NSL-KDD Replay")
    if int(config["transition_before"]) != 0:
        raise ValueError("the first Exp 2 slice requires transition 0 -> 1")
    seed = int(config["seed"])
    seed_everything(seed)
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    data_root = (REPO_ROOT / config["data_root"]).resolve()
    data: PreparedData = prepare_dataset(config["dataset"], data_root)
    target = int(config["target_class"])
    marker_columns = int(config["marker_columns"])
    batch_size = int(config["batch_size"])
    hidden = int(config["hidden"])
    n_classes = len(data.class_order)
    output_dir = (REPO_ROOT / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"effective_config": config, "git_commit": git_commit()}, sort_keys=True))
    print(f"data train={data.x_train.shape} test={data.x_test.shape} tasks={data.tasks}")
    started = time.perf_counter()

    task0_classes = data.tasks[0]
    task1_classes = data.tasks[1]
    task0_mask = np.isin(data.y_train, task0_classes)
    task0_x = data.x_train[task0_mask]
    task0_y = data.y_train[task0_mask]
    task0_marker = pure_noise_markers(len(task0_x), marker_columns, seed + 100)
    task0_augmented = append_markers(task0_x, task0_marker)
    model0 = MLP(task0_augmented.shape[1], hidden, n_classes).to(device)
    train_epoch(
        model0,
        task0_augmented,
        task0_y,
        device=device,
        batch_size=batch_size,
        lr=float(config["learning_rate"]),
        epochs=int(config["epochs_per_task"]),
        seed=seed,
    )
    checkpoint0_path = output_dir / "checkpoint_000.pt"
    torch.save({"model_state": model0.state_dict(), "marker_columns": marker_columns}, checkpoint0_path)
    selected_local = replay_indices(
        task0_x,
        task0_y,
        task0_classes,
        per_class=int(config["buffer_per_class"]),
        seed=seed,
    )
    buffer_x = task0_x[selected_local]
    buffer_y = task0_y[selected_local]

    probe_candidates = np.flatnonzero(data.y_test == target)
    rng = np.random.default_rng(seed + 300)
    probe_indices = (
        probe_candidates
        if len(probe_candidates) <= int(config["probe_per_class"])
        else np.sort(rng.choice(probe_candidates, int(config["probe_per_class"]), replace=False))
    )
    background_indices = rng.choice(
        np.arange(len(task0_augmented)),
        min(int(config["background_rows"]), len(task0_augmented)),
        replace=False,
    )
    background = task0_augmented[np.sort(background_indices)]
    all_rows: list[dict] = []
    scenario_records: list[dict] = []
    compact: dict[str, np.ndarray] = {}

    task1_mask = np.isin(data.y_train, task1_classes)
    current_x = data.x_train[task1_mask]
    current_y = data.y_train[task1_mask]
    seen0_test = np.isin(data.y_test, task0_classes)
    seen1_test = np.isin(data.y_test, task0_classes + task1_classes)

    for rho_index, rho_value in enumerate(config["rho_values"]):
        rho = float(rho_value)
        scenario_seed = seed * 1_000_000 + rho_index * 100_000
        replay_x = np.concatenate([current_x, buffer_x], axis=0)
        replay_y = np.concatenate([current_y, buffer_y], axis=0)
        train_marker = correlated_markers(
            replay_y,
            target_class=target,
            rho=rho,
            columns=marker_columns,
            seed=scenario_seed + 1,
        )
        model1 = copy.deepcopy(model0).to(device)
        train_epoch(
            model1,
            append_markers(replay_x, train_marker),
            replay_y,
            device=device,
            batch_size=batch_size,
            lr=float(config["learning_rate"]),
            epochs=int(config["epochs_per_task"]),
            seed=seed + 1000,
        )
        checkpoint1_path = output_dir / f"checkpoint_001_rho_{rho:.1f}.pt"
        torch.save({"model_state": model1.state_dict(), "rho": rho}, checkpoint1_path)
        scorer0 = scorer_for(model0, n_classes, device)
        scorer1 = scorer_for(model1, n_classes, device)

        test_marker = correlated_markers(
            data.y_test,
            target_class=target,
            rho=rho,
            columns=marker_columns,
            seed=scenario_seed + 2,
        )
        test_augmented = append_markers(data.x_test, test_marker)
        effect0 = marker_effect(
            scorer0,
            test_augmented[seen0_test],
            data.y_test[seen0_test],
            target=target,
            marker_columns=marker_columns,
            device=device,
            batch_size=batch_size,
            seed=scenario_seed + 3,
        )
        effect1 = marker_effect(
            scorer1,
            test_augmented[seen1_test],
            data.y_test[seen1_test],
            target=target,
            marker_columns=marker_columns,
            device=device,
            batch_size=batch_size,
            seed=scenario_seed + 3,
        )
        recall_delta = effect1["recall_original"] - effect0["recall_original"]
        silent = recall_delta > -float(config["silent_recall_drop"])
        truth_positive = (
            effect1["recall_drop"] >= float(config["truth_recall_drop"])
            and abs(effect0["recall_drop"])
            <= float(config["checkpoint0_max_marker_effect"])
        )
        probe = test_augmented[probe_indices]
        explanations0 = explain_estimates(
            scorer0,
            target=target,
            probe=probe,
            background=background,
            repeats=int(config["repeats"]),
            bootstraps=int(config["bootstraps"]),
            eg_nsamples=int(config["expected_gradients_nsamples"]),
            seed=scenario_seed + 10,
            device=device,
        )
        explanations1 = explain_estimates(
            scorer1,
            target=target,
            probe=probe,
            background=background,
            repeats=int(config["repeats"]),
            bootstraps=int(config["bootstraps"]),
            eg_nsamples=int(config["expected_gradients_nsamples"]),
            seed=scenario_seed + 10,
            device=device,
        )
        explanations1_independent = explain_estimates(
            scorer1,
            target=target,
            probe=probe,
            background=background,
            repeats=int(config["repeats"]),
            bootstraps=int(config["bootstraps"]),
            eg_nsamples=int(config["expected_gradients_nsamples"]),
            seed=scenario_seed + 50000,
            device=device,
        )
        reparameterized_model = function_preserving_rescaling(model1, scenario_seed + 7).to(device)
        reparameterized_scorer = scorer_for(reparameterized_model, n_classes, device)
        max_error = maximum_logit_error(
            scorer1,
            reparameterized_scorer,
            test_augmented[seen1_test],
            device,
            batch_size,
        )
        if max_error >= 1e-5:
            raise RuntimeError(f"N2 function preservation failed: max logit error {max_error}")
        explanations1_reparameterized = explain_estimates(
            reparameterized_scorer,
            target=target,
            probe=probe,
            background=background,
            repeats=int(config["repeats"]),
            bootstraps=int(config["bootstraps"]),
            eg_nsamples=int(config["expected_gradients_nsamples"]),
            seed=scenario_seed + 10,
            device=device,
        )
        truth = {
            "truth_positive": truth_positive,
            "silent_eligible": silent,
            "intervention_kind": "P1_marker_permutation",
            "checkpoint0_intervention_recall_drop": effect0["recall_drop"],
            "checkpoint1_intervention_recall_drop": effect1["recall_drop"],
            "intervention_effect_change": abs(effect1["recall_drop"] - effect0["recall_drop"]),
            "recall_delta": recall_delta,
        }
        all_rows.extend(
            comparison_rows(
                "N1_same_checkpoint",
                rho,
                explanations1,
                explanations1_independent,
                truth={**truth, "truth_positive": False},
                config=config,
                seed=scenario_seed + 60000,
            )
        )
        all_rows.extend(
            comparison_rows(
                "N2_function_preserving",
                rho,
                explanations1,
                explanations1_reparameterized,
                truth={**truth, "truth_positive": False},
                config=config,
                seed=scenario_seed + 70000,
            )
        )
        all_rows.extend(
            comparison_rows(
                "P1_shortcut",
                rho,
                explanations0,
                explanations1,
                truth=truth,
                config=config,
                seed=scenario_seed + 80000,
            )
        )
        for label, collection in (
            ("checkpoint0", explanations0),
            ("checkpoint1", explanations1),
            ("n1_independent", explanations1_independent),
            ("n2_reparameterized", explanations1_reparameterized),
        ):
            for explainer in EXPLAINERS:
                prefix = f"rho_{rho_index}_{label}_{explainer}"
                compact[f"{prefix}_estimates"] = collection[explainer]["estimates"].astype(
                    np.float16
                )
                compact[f"{prefix}_repeat_estimates"] = collection[explainer][
                    "repeat_estimates"
                ].astype(np.float16)
                compact[f"{prefix}_single"] = collection[explainer]["single_vector"].astype(
                    np.float16
                )
                compact[f"{prefix}_repeat_mean"] = collection[explainer][
                    "repeat_mean_vector"
                ].astype(np.float16)
                compact[f"{prefix}_signed_mean"] = collection[explainer][
                    "signed_mean_vector"
                ].astype(np.float16)
        scenario = {
            "condition": "P1_shortcut",
            "rho": rho,
            "checkpoint0": effect0,
            "checkpoint1": effect1,
            "recall_delta": recall_delta,
            "silent_eligible": silent,
            "truth_positive": truth_positive,
            "n2_max_abs_logit_error": max_error,
        }
        scenario_records.append(scenario)
        print(json.dumps(scenario, sort_keys=True), flush=True)
        del model1, scorer0, scorer1, reparameterized_model, reparameterized_scorer
        torch.cuda.empty_cache()

    # P2: force the checkpoint-1 replay examples for the target class to stop
    # carrying checkpoint-0's three most important real input features.  The
    # attribution only selects the candidate features; the truth label comes
    # from the independently measured recall intervention below.
    p2_target = int(config.get("p2_target_class", target))
    p2_test_marker = pure_noise_markers(len(data.x_test), marker_columns, seed + 900)
    p2_test_augmented = append_markers(data.x_test, p2_test_marker)
    p2_probe_candidates = np.flatnonzero(data.y_test == p2_target)
    p2_probe_rng = np.random.default_rng(seed + 1300 + p2_target)
    p2_probe_indices = (
        p2_probe_candidates
        if len(p2_probe_candidates) <= int(config["probe_per_class"])
        else np.sort(
            p2_probe_rng.choice(
                p2_probe_candidates, int(config["probe_per_class"]), replace=False
            )
        )
    )
    p2_probe = p2_test_augmented[p2_probe_indices]
    p2_scorer0 = scorer_for(model0, n_classes, device)
    real_feature_count = data.x_train.shape[1]
    p2_top_k = int(config["p2_top_k"])
    if "p2_feature_indices" in config:
        p2_top_features = np.asarray(config["p2_feature_indices"], dtype=np.int64)
        if len(p2_top_features) != p2_top_k:
            raise ValueError("p2_feature_indices length must equal p2_top_k")
        if np.any(p2_top_features < 0) or np.any(p2_top_features >= real_feature_count):
            raise ValueError("p2_feature_indices contains an out-of-range feature")
    else:
        p2_selection_explanations0 = explain_estimates(
            p2_scorer0,
            target=p2_target,
            probe=p2_probe,
            background=background,
            repeats=int(config["repeats"]),
            bootstraps=int(config["bootstraps"]),
            eg_nsamples=int(config["expected_gradients_nsamples"]),
            seed=seed * 1_000_000 + 410_000,
            device=device,
        )
        p2_top_features = np.argsort(
            -p2_selection_explanations0["feature_ablation"]["repeat_mean_vector"][
                :real_feature_count
            ],
            kind="stable",
        )[:p2_top_k]
    p2_feature_names = [str(data.feature_names[index]) for index in p2_top_features]
    p2_effect0 = feature_mask_effect(
        p2_scorer0,
        p2_test_augmented[seen0_test],
        data.y_test[seen0_test],
        target=p2_target,
        feature_indices=p2_top_features,
        device=device,
        batch_size=batch_size,
    )
    replay_x_base = np.concatenate([current_x, buffer_x], axis=0)
    replay_y_base = np.concatenate([current_y, buffer_y], axis=0)
    for fraction_index, fraction_value in enumerate(config["p2_mask_fractions"]):
        fraction = float(fraction_value)
        scenario_seed = seed * 1_000_000 + 500_000 + fraction_index * 100_000
        p2_train_x = replay_x_base.copy()
        p2_train_y = replay_y_base.copy()
        target_rows = np.flatnonzero(replay_y_base == p2_target)
        fraction_rng = np.random.default_rng(scenario_seed + 1)
        selected_target_rows = target_rows[
            fraction_rng.random(len(target_rows)) < fraction
        ]
        p2_training_mode = str(
            config.get("p2_training_mode", "replace_target_buffer_rows")
        )
        added_masked_rows = 0
        if p2_training_mode == "replace_target_buffer_rows":
            p2_train_x[np.ix_(selected_target_rows, p2_top_features)] = 0.0
        elif p2_training_mode == "append_masked_replay_copies":
            multiplier = int(config.get("p2_masked_replica_multiplier", 1))
            if multiplier < 1:
                raise ValueError("p2_masked_replica_multiplier must be positive")
            masked_copies = replay_x_base[selected_target_rows].copy()
            masked_copies[:, p2_top_features] = 0.0
            masked_copies = np.tile(masked_copies, (multiplier, 1))
            masked_labels = np.tile(replay_y_base[selected_target_rows], multiplier)
            p2_train_x = np.concatenate([p2_train_x, masked_copies], axis=0)
            p2_train_y = np.concatenate([p2_train_y, masked_labels], axis=0)
            added_masked_rows = len(masked_copies)
        else:
            raise ValueError(f"unsupported p2_training_mode: {p2_training_mode}")
        p2_marker_seed = int(config.get("p2_marker_seed", scenario_seed + 2))
        p2_train_seed = int(config.get("p2_train_seed", seed + 2000))
        p2_train_marker = pure_noise_markers(
            len(p2_train_x), marker_columns, p2_marker_seed
        )
        p2_model1 = copy.deepcopy(model0).to(device)
        train_epoch(
            p2_model1,
            append_markers(p2_train_x, p2_train_marker),
            p2_train_y,
            device=device,
            batch_size=batch_size,
            lr=float(config["learning_rate"]),
            epochs=int(config["epochs_per_task"]),
            seed=p2_train_seed,
        )
        p2_checkpoint_path = output_dir / f"checkpoint_001_p2_mask_{fraction:.2f}.pt"
        torch.save(
            {
                "model_state": p2_model1.state_dict(),
                "mask_fraction": fraction,
                "feature_indices": p2_top_features.tolist(),
            },
            p2_checkpoint_path,
        )
        p2_scorer1 = scorer_for(p2_model1, n_classes, device)
        p2_effect1 = feature_mask_effect(
            p2_scorer1,
            p2_test_augmented[seen1_test],
            data.y_test[seen1_test],
            target=p2_target,
            feature_indices=p2_top_features,
            device=device,
            batch_size=batch_size,
        )
        p2_recall_delta = p2_effect1["recall_original"] - p2_effect0["recall_original"]
        p2_effect_change = abs(p2_effect1["recall_drop"] - p2_effect0["recall_drop"])
        p2_silent = p2_recall_delta > -float(config["silent_recall_drop"])
        p2_truth_positive = p2_effect_change >= float(config["truth_recall_drop"])
        p2_explanations0 = explain_estimates(
            p2_scorer0,
            target=p2_target,
            probe=p2_probe,
            background=background,
            repeats=int(config["repeats"]),
            bootstraps=int(config["bootstraps"]),
            eg_nsamples=int(config["expected_gradients_nsamples"]),
            seed=scenario_seed + 10,
            device=device,
        )
        p2_explanations1 = explain_estimates(
            p2_scorer1,
            target=p2_target,
            probe=p2_probe,
            background=background,
            repeats=int(config["repeats"]),
            bootstraps=int(config["bootstraps"]),
            eg_nsamples=int(config["expected_gradients_nsamples"]),
            seed=scenario_seed + 10,
            device=device,
        )
        p2_explanations1_independent = explain_estimates(
            p2_scorer1,
            target=p2_target,
            probe=p2_probe,
            background=background,
            repeats=int(config["repeats"]),
            bootstraps=int(config["bootstraps"]),
            eg_nsamples=int(config["expected_gradients_nsamples"]),
            seed=scenario_seed + 50000,
            device=device,
        )
        p2_reparameterized_model = function_preserving_rescaling(
            p2_model1, scenario_seed + 7
        ).to(device)
        p2_reparameterized_scorer = scorer_for(
            p2_reparameterized_model, n_classes, device
        )
        p2_max_error = maximum_logit_error(
            p2_scorer1,
            p2_reparameterized_scorer,
            p2_test_augmented[seen1_test],
            device,
            batch_size,
        )
        if p2_max_error >= 1e-5:
            raise RuntimeError(
                f"P2 N2 function preservation failed: max logit error {p2_max_error}"
            )
        p2_explanations1_reparameterized = explain_estimates(
            p2_reparameterized_scorer,
            target=p2_target,
            probe=p2_probe,
            background=background,
            repeats=int(config["repeats"]),
            bootstraps=int(config["bootstraps"]),
            eg_nsamples=int(config["expected_gradients_nsamples"]),
            seed=scenario_seed + 10,
            device=device,
        )
        p2_truth = {
            "truth_positive": p2_truth_positive,
            "silent_eligible": p2_silent,
            "intervention_kind": f"P2_top_feature_zero_{p2_training_mode}",
            "checkpoint0_intervention_recall_drop": p2_effect0["recall_drop"],
            "checkpoint1_intervention_recall_drop": p2_effect1["recall_drop"],
            "intervention_effect_change": p2_effect_change,
            "recall_delta": p2_recall_delta,
        }
        all_rows.extend(
            comparison_rows(
                "N1_same_checkpoint_P2",
                fraction,
                p2_explanations1,
                p2_explanations1_independent,
                truth={**p2_truth, "truth_positive": False},
                config=config,
                seed=scenario_seed + 60000,
            )
        )
        all_rows.extend(
            comparison_rows(
                "N2_function_preserving_P2",
                fraction,
                p2_explanations1,
                p2_explanations1_reparameterized,
                truth={**p2_truth, "truth_positive": False},
                config=config,
                seed=scenario_seed + 70000,
            )
        )
        all_rows.extend(
            comparison_rows(
                "P2_feature_mask",
                fraction,
                p2_explanations0,
                p2_explanations1,
                truth=p2_truth,
                config=config,
                seed=scenario_seed + 80000,
            )
        )
        for label, collection in (
            ("checkpoint0", p2_explanations0),
            ("checkpoint1", p2_explanations1),
            ("n1_independent", p2_explanations1_independent),
            ("n2_reparameterized", p2_explanations1_reparameterized),
        ):
            for explainer in EXPLAINERS:
                prefix = f"p2_{fraction_index}_{label}_{explainer}"
                for key in (
                    "estimates",
                    "repeat_estimates",
                    "single_vector",
                    "repeat_mean_vector",
                    "signed_mean_vector",
                ):
                    compact[f"{prefix}_{key}"] = collection[explainer][key].astype(
                        np.float16
                    )
        p2_scenario = {
            "condition": "P2_feature_mask",
            "mask_fraction": fraction,
            "target_class": p2_target,
            "top_feature_indices": p2_top_features.tolist(),
            "top_feature_names": p2_feature_names,
            "masked_target_buffer_rows": len(selected_target_rows),
            "training_mode": p2_training_mode,
            "added_masked_rows": added_masked_rows,
            "marker_seed": p2_marker_seed,
            "train_seed": p2_train_seed,
            "checkpoint0": p2_effect0,
            "checkpoint1": p2_effect1,
            "intervention_effect_change": p2_effect_change,
            "recall_delta": p2_recall_delta,
            "silent_eligible": p2_silent,
            "truth_positive": p2_truth_positive,
            "n2_max_abs_logit_error": p2_max_error,
        }
        scenario_records.append(p2_scenario)
        print(json.dumps(p2_scenario, sort_keys=True), flush=True)
        del (
            p2_model1,
            p2_scorer1,
            p2_reparameterized_model,
            p2_reparameterized_scorer,
        )
        torch.cuda.empty_cache()
    del p2_scorer0

    for explainer in EXPLAINERS:
        indices = [index for index, row in enumerate(all_rows) if row["explainer"] == explainer]
        for prefix, p_field, effect_field in (
            ("ncd", "p_value", "ncd_effect"),
            (
                "ncd_no_bootstrap",
                "ncd_no_bootstrap_p_value",
                "ncd_no_bootstrap_effect",
            ),
            ("ncd_no_repeats", "ncd_no_repeats_p_value", "ncd_no_repeats_effect"),
        ):
            adjusted = benjamini_hochberg(
                [float(all_rows[index][p_field]) for index in indices]
            )
            for index, q_value in zip(indices, adjusted):
                all_rows[index][f"{prefix}_q_value"] = q_value
                all_rows[index][f"{prefix}_drift"] = (
                    q_value < 0.05 and all_rows[index][effect_field] > 0
                )
        for index in indices:
            all_rows[index]["ncd_no_fdr_drift"] = (
                all_rows[index]["p_value"] < 0.05
                and all_rows[index]["ncd_effect"] > 0
            )

    consensus_rows: list[dict] = []
    scenario_keys = list(
        dict.fromkeys((row["condition"], row["strength"]) for row in all_rows)
    )
    for condition, strength in scenario_keys:
        subset = [
            row
            for row in all_rows
            if row["condition"] == condition and row["strength"] == strength
        ]
        consensus_rows.append(
            {
                "condition": condition,
                "strength": strength,
                "truth_positive": subset[0]["truth_positive"],
                "silent_eligible": subset[0]["silent_eligible"],
                "b1_consensus_drift": sum(row["b1_fixed_drift"] for row in subset) >= 2,
                "b2_consensus_drift": sum(row["b2_fixed_drift"] for row in subset) >= 2,
                "ncd_consensus_drift": sum(row["ncd_drift"] for row in subset) >= 2,
                "ncd_explainer_votes": sum(row["ncd_drift"] for row in subset),
                "ncd_no_bootstrap_consensus_drift": sum(
                    row["ncd_no_bootstrap_drift"] for row in subset
                )
                >= 2,
                "ncd_no_repeats_consensus_drift": sum(
                    row["ncd_no_repeats_drift"] for row in subset
                )
                >= 2,
                "ncd_no_fdr_consensus_drift": sum(
                    row["ncd_no_fdr_drift"] for row in subset
                )
                >= 2,
            }
        )

    negatives = [row for row in consensus_rows if row["condition"].startswith("N")]
    strong_positives = [
        row
        for row in consensus_rows
        if row["condition"] == "P1_shortcut"
        and row["strength"] >= 0.9
        and row["truth_positive"]
        and row["silent_eligible"]
    ]
    p2_positives = [
        row
        for row in consensus_rows
        if row["condition"] == "P2_feature_mask"
        and row["truth_positive"]
        and row["silent_eligible"]
    ]
    discrimination = discrimination_metrics(all_rows)
    compact_path = output_dir / "exp2_groundtruth_compact.npz"
    np.savez_compressed(compact_path, **compact)
    if compact_path.stat().st_size >= 5 * 1024 * 1024:
        raise RuntimeError("compact Exp 2 artifact exceeds 5 MiB")
    table_path = output_dir / "exp2_groundtruth_table.csv"
    consensus_path = output_dir / "exp2_groundtruth_consensus.csv"
    discrimination_path = output_dir / "exp2_groundtruth_discrimination.csv"
    write_csv(table_path, all_rows)
    write_csv(consensus_path, consensus_rows)
    write_csv(discrimination_path, discrimination)

    def rate(rows: list[dict], field: str) -> float | None:
        return float(np.mean([row[field] for row in rows])) if rows else None

    result = {
        "analysis": "Exp 2 intervention-ground-truth slice with P1/P2 and ablations",
        "scope": (
            f"{config['dataset']} {config['method']} seed {config['seed']}; "
            f"P1 target class {config['target_class']}; "
            f"P2 target class {config['p2_target_class']}; "
            f"transition {config['transition_before']}->{config['transition_before'] + 1}; "
            "N1/N2/P1/P2"
        ),
        "git_commit": git_commit(),
        "config": str(config_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "effective_config": config,
        "command": " ".join(sys.argv),
        "log_path": config["log_path"],
        "scenario_records": scenario_records,
        "consensus": consensus_rows,
        "discrimination": discrimination,
        "b3_reference": {
            "name": "CLEX Explanation Drift (Equation 6)",
            "doi": "10.1016/j.neucom.2024.127960",
            "adaptation": "same positive-attribution-mass distance applied to the paired checkpoint explanations",
        },
        "pilot_summary": {
            "negative_control_count": len(negatives),
            "ncd_negative_fpr": rate(negatives, "ncd_consensus_drift"),
            "b2_negative_fpr": rate(negatives, "b2_consensus_drift"),
            "strong_truth_positive_count": len(strong_positives),
            "ncd_strong_positive_tpr": rate(strong_positives, "ncd_consensus_drift"),
            "b2_strong_positive_tpr": rate(strong_positives, "b2_consensus_drift"),
            "p2_silent_truth_positive_count": len(p2_positives),
            "ncd_p2_tpr": rate(p2_positives, "ncd_consensus_drift"),
            "b2_p2_tpr": rate(p2_positives, "b2_consensus_drift"),
            "ablations": {
                "full": {
                    "negative_fpr": rate(negatives, "ncd_consensus_drift"),
                    "strong_p1_tpr": rate(strong_positives, "ncd_consensus_drift"),
                },
                "no_probe_bootstrap": {
                    "negative_fpr": rate(
                        negatives, "ncd_no_bootstrap_consensus_drift"
                    ),
                    "strong_p1_tpr": rate(
                        strong_positives, "ncd_no_bootstrap_consensus_drift"
                    ),
                },
                "no_explainer_repeats": {
                    "negative_fpr": rate(
                        negatives, "ncd_no_repeats_consensus_drift"
                    ),
                    "strong_p1_tpr": rate(
                        strong_positives, "ncd_no_repeats_consensus_drift"
                    ),
                },
                "no_fdr": {
                    "negative_fpr": rate(negatives, "ncd_no_fdr_consensus_drift"),
                    "strong_p1_tpr": rate(
                        strong_positives, "ncd_no_fdr_consensus_drift"
                    ),
                },
            },
        },
        "artifacts": {
            "table": str(table_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "consensus": str(consensus_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "discrimination": str(discrimination_path.relative_to(REPO_ROOT)).replace(
                "\\", "/"
            ),
            "compact": str(compact_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "compact_bytes": compact_path.stat().st_size,
            "local_checkpoints": str(output_dir),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "This is a one-dataset, one-method, one-seed, one-transition pilot.",
            "Matched-FPR thresholds are calibrated and evaluated on the same small negative-control slice.",
            "Additional targets, datasets, methods, seeds, and an external calibration split remain pending.",
            "No publication-level TPR/FPR claim is made from this pilot alone.",
        ],
    }
    result_path = output_dir / "exp2_groundtruth_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config_path = (REPO_ROOT / args.config).resolve()
    config = effective_config(
        json.loads(config_path.read_text(encoding="utf-8")), args.smoke
    )
    log_path = (REPO_ROOT / config["log_path"]).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        with redirect_stdout(Tee(sys.__stdout__, log_file)), redirect_stderr(
            Tee(sys.__stderr__, log_file)
        ):
            try:
                run(config, config_path)
            except Exception:
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
