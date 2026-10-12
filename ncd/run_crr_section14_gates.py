"""Run the Section 14 Oracle, Replay necessity, and Replay hijack gates."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from ncd.crr_benchmarks import (
    boundary_for,
    build_hijack,
    build_necessity,
    replay_buffer,
    run_method,
    train_base_checkpoint,
)
from ncd.data import prepare_dataset
from ncd.model import MLP
from ncd.source_commit import source_commit
from ncd.train_cil import seed_everything


def _write_csv(path: Path, rows: list[dict]) -> None:
    scalar = [
        {
            key: json.dumps(value, sort_keys=True)
            if isinstance(value, (dict, list))
            else value
            for key, value in row.items()
        }
        for row in rows
    ]
    fields = list(dict.fromkeys(key for row in scalar for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scalar)


def _oracle_initial(base_model: MLP, input_dim: int, seed: int, device: torch.device) -> MLP:
    seed_everything(seed)
    return MLP(
        input_dim, base_model.head.in_features, base_model.head.out_features
    ).to(device)


def _gate_verdict(rows: list[dict]) -> dict:
    datasets = ("NSL-KDD", "UNSW-NB15")
    seeds = (1, 2)
    expected_necessity = {
        (dataset, seed, method, k)
        for dataset in datasets
        for seed in seeds
        for method in ("oracle", "replay")
        for k in (1.0, 2.0, 3.0)
    }
    observed_necessity = {
        (str(row["dataset"]), int(row["seed"]), str(row["method"]), float(row["k"]))
        for row in rows
        if row["benchmark"] == "necessity"
    }
    expected_hijack = {
        (dataset, seed, step, rho)
        for dataset in datasets
        for seed in seeds
        for step in (1, 2, 3)
        for rho in (0.9, 1.0)
    }
    observed_hijack = {
        (
            str(row["dataset"]),
            int(row["seed"]),
            int(row["hijack_construction_step"]),
            float(row["rho"]),
        )
        for row in rows
        if row["benchmark"] == "hijack"
    }
    complete = (
        len(rows) == 48
        and observed_necessity == expected_necessity
        and observed_hijack == expected_hijack
    )
    necessity_oracle = [
        row
        for row in rows
        if row["benchmark"] == "necessity"
        and row["method"] == "oracle"
        and float(row["k"]) >= 2.0
    ]
    necessity_replay = [
        row
        for row in rows
        if row["benchmark"] == "necessity"
        and row["method"] == "replay"
        and float(row["k"]) >= 2.0
    ]
    oracle_pass = bool(necessity_oracle) and all(
        float(row["c_recall"]) >= 0.7 and float(row["d_recall"]) >= 0.7
        for row in necessity_oracle
    )
    replay_necessity_pass = bool(necessity_replay) and all(
        float(row["c_recall"]) >= 0.5
        and float(row["d_recall"]) >= 0.5
        and float(row["c_f2_dependency_change"]) > 0.0
        for row in necessity_replay
    )

    hijack_rows = [row for row in rows if row["benchmark"] == "hijack"]
    construction_steps = sorted(
        {int(row["hijack_construction_step"]) for row in hijack_rows}
    )
    step_checks = []
    selected_step = None
    for step in construction_steps:
        items = [
            row
            for row in hijack_rows
            if int(row["hijack_construction_step"]) == step
        ]
        passed = bool(items) and all(
            float(row["marker_recall_drop"]) >= 0.15
            and float(row["marker_decorrelation_recall_drop"]) >= 0.15
            for row in items
        )
        step_checks.append(
            {"construction_step": step, "passed": passed, "rows": len(items)}
        )
        if passed and selected_step is None:
            selected_step = step

    return {
        "schema_version": 1,
        "gate": "section14.2-14.3",
        "complete": complete,
        "criteria": {
            "oracle_necessity_k_ge_2": "c_recall >= 0.7 and d_recall >= 0.7",
            "replay_necessity_k_ge_2": (
                "c_recall >= 0.5, d_recall >= 0.5, c_f2_dependency_change > 0"
            ),
            "replay_hijack_rho_0_9_1_0": (
                "marker_recall_drop >= 0.15 and marker_decorrelation_recall_drop >= 0.15"
            ),
        },
        "oracle_necessity_passed": oracle_pass,
        "replay_necessity_passed": replay_necessity_pass,
        "hijack_step_checks": step_checks,
        "selected_hijack_construction_step": selected_step,
        "necessity_gate_passed": complete and oracle_pass and replay_necessity_pass,
        "hijack_gate_passed": complete and selected_step is not None,
        "passed": (
            complete
            and oracle_pass
            and replay_necessity_pass
            and selected_step is not None
        ),
        "decision": (
            "SUBMIT_FIVE_SEED_MATRIX"
            if complete
            and oracle_pass
            and replay_necessity_pass
            and selected_step is not None
            else "STOP_AND_REPAIR_BENCHMARK"
        ),
    }


def run(config: dict) -> dict:
    device = torch.device(str(config.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seeds = [int(value) for value in config.get("seeds", [1, 2])]
    if seeds != [1, 2]:
        raise ValueError("Section 14 gate must run exactly seeds 1 and 2")
    datasets = [str(value) for value in config["datasets"]]
    if sorted(datasets) != ["NSL-KDD", "UNSW-NB15"]:
        raise ValueError("Section 14 gate must run NSL-KDD and UNSW-NB15")
    batch_size = int(config.get("batch_size", 256))
    marker_columns = int(config.get("marker_columns", 4))
    data_root = (REPO_ROOT / config["data_root"]).resolve()
    rows: list[dict] = []
    metadata: list[dict] = []
    started = time.perf_counter()

    for dataset_index, dataset_name in enumerate(datasets):
        data = prepare_dataset(dataset_name, data_root)
        boundary = boundary_for(data)
        for seed in seeds:
            base_model, base_x, base_y = train_base_checkpoint(
                data,
                boundary,
                seed=seed,
                marker_columns=marker_columns,
                hidden=int(config.get("hidden", 128)),
                epochs=int(config.get("epochs_base", 8)),
                batch_size=batch_size,
                learning_rate=float(config.get("learning_rate", 1e-3)),
                device=device,
                rows_per_class=config.get("rows_per_class"),
            )
            buffer_x, buffer_y = replay_buffer(
                base_model,
                base_x,
                base_y,
                boundary.old_classes,
                per_class=int(config.get("buffer_per_class", 200)),
                seed=seed,
                method="replay",
                device=device,
                batch_size=batch_size,
            )

            for k_index, k in enumerate(config.get("k_values", [1.0, 2.0, 3.0])):
                common_seed = seed * 100000 + dataset_index * 10000 + k_index * 100
                replay_benchmark = build_necessity(
                    data,
                    boundary,
                    buffer_x,
                    buffer_y,
                    base_model,
                    k=float(k),
                    marker_columns=marker_columns,
                    seed=common_seed,
                    device=device,
                    batch_size=batch_size,
                    rows_per_class=config.get("rows_per_class"),
                    noise_scale=float(config.get("necessity_noise_scale", 0.02)),
                )
                _, replay_metrics = run_method(
                    base_model,
                    replay_benchmark,
                    boundary,
                    method="replay",
                    regularization=0.0,
                    epochs=int(config.get("epochs_transition", 8)),
                    batch_size=batch_size,
                    learning_rate=float(config.get("learning_rate", 1e-3)),
                    temperature=float(config.get("temperature", 2.0)),
                    seed=common_seed + 1,
                    device=device,
                )
                replay_row = {
                    "dataset": dataset_name,
                    "seed": seed,
                    "method": "replay",
                    "benchmark": "necessity",
                    "k": float(k),
                    **replay_metrics,
                }
                rows.append(replay_row)
                print(json.dumps(replay_row, sort_keys=True), flush=True)

                oracle_benchmark = build_necessity(
                    data,
                    boundary,
                    buffer_x,
                    buffer_y,
                    base_model,
                    k=float(k),
                    marker_columns=marker_columns,
                    seed=common_seed,
                    device=device,
                    batch_size=batch_size,
                    rows_per_class=config.get("rows_per_class"),
                    noise_scale=float(config.get("necessity_noise_scale", 0.02)),
                    full_joint=True,
                )
                oracle_initial = _oracle_initial(
                    base_model,
                    oracle_benchmark.train_x.shape[1],
                    common_seed + 2,
                    device,
                )
                _, oracle_metrics = run_method(
                    oracle_initial,
                    oracle_benchmark,
                    boundary,
                    method="replay",
                    regularization=0.0,
                    epochs=int(config.get("epochs_transition", 8)),
                    batch_size=batch_size,
                    learning_rate=float(config.get("learning_rate", 1e-3)),
                    temperature=float(config.get("temperature", 2.0)),
                    seed=common_seed + 3,
                    device=device,
                )
                oracle_row = {
                    "dataset": dataset_name,
                    "seed": seed,
                    "method": "oracle",
                    "benchmark": "necessity",
                    "k": float(k),
                    **oracle_metrics,
                }
                rows.append(oracle_row)
                print(json.dumps(oracle_row, sort_keys=True), flush=True)
                metadata.append(
                    {
                        "dataset": dataset_name,
                        "seed": seed,
                        "benchmark": "necessity",
                        "k": float(k),
                        **oracle_benchmark.metadata,
                    }
                )

            for step in (1, 2, 3):
                for rho_index, rho in enumerate((0.9, 1.0)):
                    item_seed = (
                        seed * 1000000
                        + dataset_index * 100000
                        + step * 10000
                        + rho_index * 100
                    )
                    benchmark = build_hijack(
                        data,
                        boundary,
                        buffer_x,
                        buffer_y,
                        rho=rho,
                        marker_columns=marker_columns,
                        seed=item_seed,
                        rows_per_class=config.get("rows_per_class"),
                        construction_step=step,
                    )
                    _, metrics = run_method(
                        base_model,
                        benchmark,
                        boundary,
                        method="replay",
                        regularization=0.0,
                        epochs=int(config.get("epochs_transition", 8)),
                        batch_size=batch_size,
                        learning_rate=float(config.get("learning_rate", 1e-3)),
                        temperature=float(config.get("temperature", 2.0)),
                        seed=item_seed + 1,
                        device=device,
                    )
                    row = {
                        "dataset": dataset_name,
                        "seed": seed,
                        "method": "replay",
                        "benchmark": "hijack",
                        "rho": rho,
                        **metrics,
                    }
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)
                    metadata.append(
                        {
                            "dataset": dataset_name,
                            "seed": seed,
                            "benchmark": "hijack",
                            "rho": rho,
                            **benchmark.metadata,
                        }
                    )

            del base_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    verdict = _gate_verdict(rows)
    return {
        "schema_version": 1,
        "git_commit": source_commit(REPO_ROOT),
        "effective_config": config,
        "rows": rows,
        "benchmark_metadata": metadata,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
        "scope": "Section 14 gate evidence only; not a method-comparison verdict",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config.update(
        {
            "seeds": [1, 2],
            "rho_values": [0.9, 1.0],
            "k_values": [1.0, 2.0, 3.0],
            "registered_matrix": False,
        }
    )
    if args.device:
        config["device"] = args.device
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = run(config)
    (output / "gate_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(output / "gate_rows.csv", result["rows"])
    (output / "GATE_VERDICT.json").write_text(
        json.dumps(result["verdict"], indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "COMPLETED.json").write_text(
        json.dumps(
            {
                "git_commit": result["git_commit"],
                "row_count": len(result["rows"]),
                "passed": result["verdict"]["passed"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": result["verdict"]["decision"],
                "rows": len(result["rows"]),
                "selected_hijack_step": result["verdict"][
                    "selected_hijack_construction_step"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
