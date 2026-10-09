"""Screen P2 constructions without using the held-out evaluation split for selection.

The work-order P2 intervention asks whether checkpoint t+1 changes its
dependence on the target class's top-three real features while target recall
remains approximately unchanged.  Candidate construction is selected only on
a deterministic calibration half of the target-class test rows.  The selected
candidate is then reported once on the untouched evaluation half.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from ncd.data import prepare_dataset
from ncd.groundtruth import (
    append_markers,
    feature_mask_effect,
    pure_noise_markers,
    replay_indices,
    scorer_for,
)
from ncd.model import MLP
from ncd.train_cil import Tee, seed_everything, train_epoch


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("P2 screen produced no candidate rows")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cpu_state_dict(model: MLP) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def target_split(labels: np.ndarray, target: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.flatnonzero(labels == target)
    rng = np.random.default_rng(seed)
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    midpoint = len(shuffled) // 2
    if midpoint == 0 or midpoint == len(shuffled):
        raise RuntimeError("target-class test split is empty")
    return np.sort(shuffled[:midpoint]), np.sort(shuffled[midpoint:])


def effect_on_indices(
    scorer,
    augmented_test: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    target: int,
    top_features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    return feature_mask_effect(
        scorer,
        augmented_test[indices],
        labels[indices],
        target=target,
        feature_indices=top_features,
        device=device,
        batch_size=batch_size,
    )


def run(config: dict, config_path: Path) -> None:
    if config["dataset"] not in {"NSL-KDD", "UNSW-NB15"} or config["method"] != "replay":
        raise ValueError("the P2 screen currently supports NSL-KDD/UNSW-NB15 Replay")
    seed = int(config["seed"])
    seed_everything(seed)
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    started = time.perf_counter()
    data = prepare_dataset(config["dataset"], (REPO_ROOT / config["data_root"]).resolve())
    target = int(config["target_class"])
    marker_columns = int(config["marker_columns"])
    batch_size = int(config["batch_size"])
    n_classes = len(data.class_order)
    real_feature_count = data.x_train.shape[1]
    checkpoint_path = (REPO_ROOT / config["base_checkpoint"]).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model0 = MLP(real_feature_count + marker_columns, int(config["hidden"]), n_classes)
    model0.load_state_dict(checkpoint["model_state"])
    model0 = model0.to(device).eval()
    scorer0 = scorer_for(model0, n_classes, device)

    test_markers = pure_noise_markers(len(data.x_test), marker_columns, seed + 900)
    augmented_test = append_markers(data.x_test, test_markers)
    calibration_indices, evaluation_indices = target_split(data.y_test, target, seed + 910)

    feature_rows: list[dict] = []
    for feature_index in range(real_feature_count):
        effect = effect_on_indices(
            scorer0,
            augmented_test,
            data.y_test,
            calibration_indices,
            target,
            np.asarray([feature_index]),
            device,
            batch_size,
        )
        feature_rows.append(
            {
                "feature_index": feature_index,
                "feature_name": str(data.feature_names[feature_index]),
                "calibration_recall_drop": effect["recall_drop"],
            }
        )
    ranked = sorted(
        feature_rows,
        key=lambda row: (-float(row["calibration_recall_drop"]), int(row["feature_index"])),
    )
    selection_mode = str(config.get("feature_selection", "individual_top3"))
    selection_trace: list[dict] = []
    if selection_mode == "individual_top3":
        top_features = np.asarray(
            [int(row["feature_index"]) for row in ranked[: int(config["p2_top_k"])]],
            dtype=np.int64,
        )
    elif selection_mode == "greedy_joint_recall_drop":
        selected_features: list[int] = []
        remaining = set(range(real_feature_count))
        for step in range(int(config["p2_top_k"])):
            candidates: list[dict] = []
            for feature_index in sorted(remaining):
                joint = np.asarray(selected_features + [feature_index], dtype=np.int64)
                effect = effect_on_indices(
                    scorer0,
                    augmented_test,
                    data.y_test,
                    calibration_indices,
                    target,
                    joint,
                    device,
                    batch_size,
                )
                candidates.append(
                    {
                        "step": step + 1,
                        "candidate_feature_index": feature_index,
                        "candidate_feature_name": str(data.feature_names[feature_index]),
                        "joint_recall_drop": effect["recall_drop"],
                    }
                )
            best = sorted(
                candidates,
                key=lambda row: (
                    -float(row["joint_recall_drop"]),
                    int(row["candidate_feature_index"]),
                ),
            )[0]
            selected = int(best["candidate_feature_index"])
            selected_features.append(selected)
            remaining.remove(selected)
            selection_trace.append(best)
        top_features = np.asarray(selected_features, dtype=np.int64)
    else:
        raise ValueError(f"unsupported feature_selection: {selection_mode}")
    top_feature_names = [str(data.feature_names[index]) for index in top_features]
    checkpoint0_calibration = effect_on_indices(
        scorer0,
        augmented_test,
        data.y_test,
        calibration_indices,
        target,
        top_features,
        device,
        batch_size,
    )
    checkpoint0_evaluation = effect_on_indices(
        scorer0,
        augmented_test,
        data.y_test,
        evaluation_indices,
        target,
        top_features,
        device,
        batch_size,
    )

    task0_classes = data.tasks[0]
    task1_classes = data.tasks[1]
    task0_mask = np.isin(data.y_train, task0_classes)
    task0_x = data.x_train[task0_mask]
    task0_y = data.y_train[task0_mask]
    selected_local = replay_indices(
        task0_x,
        task0_y,
        task0_classes,
        per_class=int(config["buffer_per_class"]),
        seed=seed,
    )
    buffer_x = task0_x[selected_local]
    buffer_y = task0_y[selected_local]
    task1_mask = np.isin(data.y_train, task1_classes)
    current_x = data.x_train[task1_mask]
    current_y = data.y_train[task1_mask]
    base_x = np.concatenate([current_x, buffer_x], axis=0)
    base_y = np.concatenate([current_y, buffer_y], axis=0)
    target_buffer_x = buffer_x[buffer_y == target]
    target_buffer_y = buffer_y[buffer_y == target]

    candidate_rows: list[dict] = []
    candidate_states: dict[tuple[float, int], dict[str, torch.Tensor]] = {}
    candidate_index = 0
    for fraction_index, fraction_value in enumerate(config["mask_fractions"]):
        fraction = float(fraction_value)
        selection_seed = (
            int(config.get("selection_seed_base", seed * 1_000_000 + 600_000))
            + fraction_index * 100_000
            + 1
        )
        selection_rng = np.random.default_rng(selection_seed)
        selected_count = max(1, int(round(len(target_buffer_x) * fraction)))
        selected = np.sort(
            selection_rng.choice(len(target_buffer_x), selected_count, replace=False)
        )
        corrupted = target_buffer_x[selected].copy()
        corrupted[:, top_features] = 0.0
        corrupted_y = target_buffer_y[selected]
        for multiplier_value in config["masked_replica_multipliers"]:
            multiplier = int(multiplier_value)
            scenario_seed = seed * 1_000_000 + 700_000 + candidate_index * 1000
            extra_x = np.tile(corrupted, (multiplier, 1))
            extra_y = np.tile(corrupted_y, multiplier)
            train_x = np.concatenate([base_x, extra_x], axis=0)
            train_y = np.concatenate([base_y, extra_y], axis=0)
            marker_seed = int(config.get("fixed_marker_seed", scenario_seed + 1))
            train_seed = int(config.get("fixed_train_seed", seed + 3000))
            train_markers = pure_noise_markers(len(train_x), marker_columns, marker_seed)
            model1 = copy.deepcopy(model0).to(device)
            train_epoch(
                model1,
                append_markers(train_x, train_markers),
                train_y,
                device=device,
                batch_size=batch_size,
                lr=float(config["learning_rate"]),
                epochs=int(config["epochs_per_task"]),
                seed=train_seed,
            )
            scorer1 = scorer_for(model1, n_classes, device)
            calibration = effect_on_indices(
                scorer1,
                augmented_test,
                data.y_test,
                calibration_indices,
                target,
                top_features,
                device,
                batch_size,
            )
            evaluation = effect_on_indices(
                scorer1,
                augmented_test,
                data.y_test,
                evaluation_indices,
                target,
                top_features,
                device,
                batch_size,
            )
            calibration_effect_change = abs(
                calibration["recall_drop"] - checkpoint0_calibration["recall_drop"]
            )
            evaluation_effect_change = abs(
                evaluation["recall_drop"] - checkpoint0_evaluation["recall_drop"]
            )
            calibration_recall_delta = (
                calibration["recall_original"] - checkpoint0_calibration["recall_original"]
            )
            evaluation_recall_delta = (
                evaluation["recall_original"] - checkpoint0_evaluation["recall_original"]
            )
            calibration_truth = calibration_effect_change >= float(config["truth_recall_drop"])
            calibration_silent = calibration_recall_delta > -float(config["silent_recall_drop"])
            row = {
                "candidate_index": candidate_index,
                "mask_fraction": fraction,
                "masked_replica_multiplier": multiplier,
                "selected_target_rows": selected_count,
                "added_masked_rows": len(extra_x),
                "selection_seed": selection_seed,
                "marker_seed": marker_seed,
                "train_seed": train_seed,
                "calibration_checkpoint0_recall": checkpoint0_calibration["recall_original"],
                "calibration_checkpoint1_recall": calibration["recall_original"],
                "calibration_recall_delta": calibration_recall_delta,
                "calibration_checkpoint0_mask_drop": checkpoint0_calibration["recall_drop"],
                "calibration_checkpoint1_mask_drop": calibration["recall_drop"],
                "calibration_effect_change": calibration_effect_change,
                "calibration_truth_positive": calibration_truth,
                "calibration_silent_eligible": calibration_silent,
                "evaluation_checkpoint0_recall": checkpoint0_evaluation["recall_original"],
                "evaluation_checkpoint1_recall": evaluation["recall_original"],
                "evaluation_recall_delta": evaluation_recall_delta,
                "evaluation_checkpoint0_mask_drop": checkpoint0_evaluation["recall_drop"],
                "evaluation_checkpoint1_mask_drop": evaluation["recall_drop"],
                "evaluation_effect_change": evaluation_effect_change,
                "evaluation_truth_positive": evaluation_effect_change
                >= float(config["truth_recall_drop"]),
                "evaluation_silent_eligible": evaluation_recall_delta
                > -float(config["silent_recall_drop"]),
            }
            candidate_rows.append(row)
            if calibration_truth and calibration_silent:
                candidate_states[(fraction, multiplier)] = cpu_state_dict(model1)
            print(json.dumps(row, sort_keys=True), flush=True)
            del model1, scorer1
            torch.cuda.empty_cache()
            candidate_index += 1

    eligible = [
        row
        for row in candidate_rows
        if row["calibration_truth_positive"] and row["calibration_silent_eligible"]
    ]
    selected_row = None
    if eligible:
        candidate_selection = str(config.get("candidate_selection", "largest_effect"))
        if candidate_selection == "largest_effect":
            selected_row = sorted(
                eligible,
                key=lambda row: (
                    -float(row["calibration_effect_change"]),
                    -float(row["calibration_recall_delta"]),
                    int(row["added_masked_rows"]),
                ),
            )[0]
        elif candidate_selection == "robust_margin":
            truth_threshold = float(config["truth_recall_drop"])
            silent_threshold = float(config["silent_recall_drop"])
            selected_row = sorted(
                eligible,
                key=lambda row: (
                    -min(
                        float(row["calibration_effect_change"]) - truth_threshold,
                        float(row["calibration_recall_delta"]) + silent_threshold,
                    ),
                    -float(row["calibration_effect_change"]),
                    int(row["added_masked_rows"]),
                ),
            )[0]
        else:
            raise ValueError(f"unsupported candidate_selection: {candidate_selection}")

    output_dir = (REPO_ROOT / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "p2_feature_ranking.csv"
    candidates_path = output_dir / "p2_candidate_screen.csv"
    result_path = output_dir / "p2_screen_result.json"
    write_csv(features_path, feature_rows)
    write_csv(candidates_path, candidate_rows)
    selected_checkpoint = None
    if selected_row is not None:
        key = (
            float(selected_row["mask_fraction"]),
            int(selected_row["masked_replica_multiplier"]),
        )
        selected_checkpoint = output_dir / "selected_p2_checkpoint.pt"
        torch.save(
            {
                "model_state": candidate_states[key],
                "top_feature_indices": top_features.tolist(),
                "top_feature_names": top_feature_names,
                "selection": selected_row,
            },
            selected_checkpoint,
        )
    result = {
        "analysis": "P2 calibration screen with held-out evaluation",
        "git_commit": git_commit(),
        "config": str(config_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "effective_config": config,
        "command": f"python -I -B ncd/p2_screen.py --config {config_path.relative_to(REPO_ROOT).as_posix()}",
        "top_feature_indices": top_features.tolist(),
        "top_feature_names": top_feature_names,
        "feature_selection": selection_mode,
        "candidate_selection": str(config.get("candidate_selection", "largest_effect")),
        "feature_selection_trace": selection_trace,
        "checkpoint0_calibration": checkpoint0_calibration,
        "checkpoint0_evaluation": checkpoint0_evaluation,
        "calibration_rows": len(calibration_indices),
        "evaluation_rows": len(evaluation_indices),
        "candidate_count": len(candidate_rows),
        "calibration_eligible_count": len(eligible),
        "selected_candidate": selected_row,
        "selected_evaluation_pass": bool(
            selected_row is not None
            and selected_row["evaluation_truth_positive"]
            and selected_row["evaluation_silent_eligible"]
        ),
        "artifacts": {
            "feature_ranking": str(features_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "candidate_screen": str(candidates_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "selected_checkpoint": None
            if selected_checkpoint is None
            else str(selected_checkpoint),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "Candidate selection used only the calibration half; the evaluation half was not used for ranking.",
            "This screen selects a P2 construction but does not evaluate explanation-drift detectors.",
            "The selected construction still requires a full N1/N2/P2 detector run and additional seeds.",
        ],
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = (REPO_ROOT / args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
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
