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
    return float(1.0 - np.dot(left, right) / max(float(denom), 1e-12))


def kendall_distance(left: np.ndarray, right: np.ndarray) -> float:
    value = kendalltau(left, right, variant="b").statistic
    return float(1.0 - value) if np.isfinite(value) else float("nan")


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
        result[explainer] = {
            "estimates": estimates,
            "single_vector": normalize(np.abs(attributes[0]).mean(axis=0)).astype(np.float32),
            "repeat_mean_vector": normalize(np.abs(attributes).mean(axis=(0, 1))).astype(np.float32),
            "repeat_ids": repeat_ids,
            "bootstrap_ids": bootstrap_ids,
        }
    return result


def comparison_rows(
    condition: str,
    rho: float,
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
                "rho": rho,
                "explainer": explainer,
                "truth_positive": bool(truth["truth_positive"]),
                "silent_eligible": bool(truth["silent_eligible"]),
                "checkpoint0_marker_recall_drop": truth["checkpoint0_marker_recall_drop"],
                "checkpoint1_marker_recall_drop": truth["checkpoint1_marker_recall_drop"],
                "recall_delta": truth["recall_delta"],
                "b1_single_jaccard": b1,
                "b1_fixed_drift": b1 < float(config["fixed_jaccard_threshold"]),
                "b2_repeat_mean_jaccard": b2,
                "b2_fixed_drift": b2 < float(config["fixed_jaccard_threshold"]),
                "b4_cosine_distance": cosine_distance(
                    left_item["repeat_mean_vector"], right_item["repeat_mean_vector"]
                ),
                "b4_kendall_distance": kendall_distance(
                    left_item["repeat_mean_vector"], right_item["repeat_mean_vector"]
                ),
                **test,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def effective_config(config: dict, smoke: bool) -> dict:
    result = copy.deepcopy(config)
    if smoke:
        result.update(
            {
                "epochs_per_task": 1,
                "rho_values": [0.9],
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
            "checkpoint0_marker_recall_drop": effect0["recall_drop"],
            "checkpoint1_marker_recall_drop": effect1["recall_drop"],
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
                compact[f"{prefix}_single"] = collection[explainer]["single_vector"].astype(
                    np.float16
                )
                compact[f"{prefix}_repeat_mean"] = collection[explainer][
                    "repeat_mean_vector"
                ].astype(np.float16)
        scenario = {
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

    for explainer in EXPLAINERS:
        indices = [index for index, row in enumerate(all_rows) if row["explainer"] == explainer]
        adjusted = benjamini_hochberg([float(all_rows[index]["p_value"]) for index in indices])
        for index, q_value in zip(indices, adjusted):
            all_rows[index]["q_value"] = q_value
            all_rows[index]["ncd_drift"] = q_value < 0.05 and all_rows[index]["ncd_effect"] > 0

    consensus_rows: list[dict] = []
    for condition in ("N1_same_checkpoint", "N2_function_preserving", "P1_shortcut"):
        for rho in [float(value) for value in config["rho_values"]]:
            subset = [row for row in all_rows if row["condition"] == condition and row["rho"] == rho]
            consensus_rows.append(
                {
                    "condition": condition,
                    "rho": rho,
                    "truth_positive": subset[0]["truth_positive"],
                    "silent_eligible": subset[0]["silent_eligible"],
                    "b1_consensus_drift": sum(row["b1_fixed_drift"] for row in subset) >= 2,
                    "b2_consensus_drift": sum(row["b2_fixed_drift"] for row in subset) >= 2,
                    "ncd_consensus_drift": sum(row["ncd_drift"] for row in subset) >= 2,
                    "ncd_explainer_votes": sum(row["ncd_drift"] for row in subset),
                }
            )

    negatives = [row for row in consensus_rows if row["condition"].startswith("N")]
    strong_positives = [
        row
        for row in consensus_rows
        if row["condition"] == "P1_shortcut"
        and row["rho"] >= 0.9
        and row["truth_positive"]
        and row["silent_eligible"]
    ]
    compact_path = output_dir / "exp2_groundtruth_compact.npz"
    np.savez_compressed(compact_path, **compact)
    if compact_path.stat().st_size >= 5 * 1024 * 1024:
        raise RuntimeError("compact Exp 2 artifact exceeds 5 MiB")
    table_path = output_dir / "exp2_groundtruth_table.csv"
    consensus_path = output_dir / "exp2_groundtruth_consensus.csv"
    write_csv(table_path, all_rows)
    write_csv(consensus_path, consensus_rows)
    result = {
        "analysis": "Exp 2 first intervention-ground-truth slice",
        "scope": "NSL-KDD Replay seed 1; target Normal; transition 0->1; N1/N2/P1",
        "git_commit": git_commit(),
        "config": str(config_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "effective_config": config,
        "command": " ".join(sys.argv),
        "log_path": config["log_path"],
        "scenario_records": scenario_records,
        "consensus": consensus_rows,
        "pilot_summary": {
            "ncd_negative_fpr": float(np.mean([row["ncd_consensus_drift"] for row in negatives])),
            "b2_negative_fpr": float(np.mean([row["b2_consensus_drift"] for row in negatives])),
            "strong_truth_positive_count": len(strong_positives),
            "ncd_strong_positive_tpr": (
                float(np.mean([row["ncd_consensus_drift"] for row in strong_positives]))
                if strong_positives
                else None
            ),
            "b2_strong_positive_tpr": (
                float(np.mean([row["b2_consensus_drift"] for row in strong_positives]))
                if strong_positives
                else None
            ),
        },
        "artifacts": {
            "table": str(table_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "consensus": str(consensus_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "compact": str(compact_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "compact_bytes": compact_path.stat().st_size,
            "local_checkpoints": str(output_dir),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "This is a one-dataset, one-method, one-seed, one-transition pilot.",
            "P2, CLEX, additional targets, and matched-FPR threshold calibration remain pending.",
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
