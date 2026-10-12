"""Run the Section 12 CRR benchmark matrix on NSL-KDD / UNSW-NB15."""

from __future__ import annotations

import argparse
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

import torch

from ncd.crr_benchmarks import (
    METHODS,
    boundary_for,
    build_hijack,
    build_necessity,
    replay_buffer,
    run_method,
    train_base_checkpoint,
)
from ncd.data import prepare_dataset
from ncd.source_commit import source_commit
from ncd.train_cil import Tee


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    scalar_rows: list[dict] = []
    for row in rows:
        scalar_rows.append(
            {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
        )
    fieldnames = list(dict.fromkeys(key for row in scalar_rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scalar_rows)


def _effective(config: dict, smoke: bool) -> dict:
    result = json.loads(json.dumps(config))
    if smoke:
        result.update(
            {
                "seeds": [1],
                "rho_values": [0.9],
                "k_values": [2.0],
                "epochs_base": 1,
                "epochs_transition": 1,
                "rows_per_class": 512,
                "buffer_per_class": 64,
                "output_dir": "ncd/results/crr/smoke_seed1",
                "log_path": "ncd/logs/crr_smoke_seed1.log",
            }
        )
    return result


def run(config: dict) -> dict:
    device = torch.device(str(config.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    datasets = [str(value) for value in config["datasets"]]
    invalid = set(datasets) - {"NSL-KDD", "UNSW-NB15"}
    if invalid:
        raise ValueError(f"unsupported datasets: {sorted(invalid)}")
    methods = tuple(str(value) for value in config.get("methods", METHODS))
    if config.get("registered_matrix", True):
        if set(methods) != set(METHODS) or len(methods) != len(METHODS):
            raise ValueError(f"the registered comparison requires exactly {METHODS}")
    elif not methods or not set(methods).issubset(METHODS):
        raise ValueError(f"calibration methods must be a nonempty subset of {METHODS}")

    rows: list[dict] = []
    benchmark_metadata: list[dict] = []
    data_root = (REPO_ROOT / config["data_root"]).resolve()
    batch_size = int(config.get("batch_size", 256))
    marker_columns = int(config.get("marker_columns", 4))
    regularization = float(config.get("regularization", 1.0))
    started = time.perf_counter()
    for dataset_index, dataset_name in enumerate(datasets):
        data = prepare_dataset(dataset_name, data_root)
        boundary = boundary_for(data)
        for seed in [int(value) for value in config["seeds"]]:
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
            buffers: dict[str, tuple] = {}
            for selection in ("random", "icarl"):
                buffers[selection] = replay_buffer(
                    base_model,
                    base_x,
                    base_y,
                    boundary.old_classes,
                    per_class=int(config.get("buffer_per_class", 200)),
                    seed=seed,
                    method="icarl" if selection == "icarl" else "replay",
                    device=device,
                    batch_size=batch_size,
                )

            hijack_cache: dict[tuple[str, int], object] = {}
            necessity_cache: dict[tuple[str, int], object] = {}
            for selection, (buffer_x, buffer_y) in buffers.items():
                for rho_index, rho in enumerate(config["rho_values"]):
                    hijack_cache[(selection, rho_index)] = build_hijack(
                        data,
                        boundary,
                        buffer_x,
                        buffer_y,
                        rho=float(rho),
                        marker_columns=marker_columns,
                        seed=seed * 100000 + rho_index * 1000 + dataset_index * 100,
                        rows_per_class=config.get("rows_per_class"),
                    )
                for k_index, k in enumerate(config["k_values"]):
                    necessity_cache[(selection, k_index)] = build_necessity(
                        data,
                        boundary,
                        buffer_x,
                        buffer_y,
                        base_model,
                        k=float(k),
                        marker_columns=marker_columns,
                        seed=seed * 100000 + k_index * 1000 + dataset_index * 100 + 50,
                        device=device,
                        batch_size=batch_size,
                        rows_per_class=config.get("rows_per_class"),
                        noise_scale=float(config.get("necessity_noise_scale", 0.02)),
                    )

            for method_index, method in enumerate(methods):
                selection = "icarl" if method == "icarl" else "random"
                for rho_index, rho in enumerate(config["rho_values"]):
                    benchmark = hijack_cache[(selection, rho_index)]
                    _, metrics = run_method(
                        base_model,
                        benchmark,
                        boundary,
                        method=method,
                        regularization=0.0 if method == "replay" else regularization,
                        epochs=int(config.get("epochs_transition", 8)),
                        batch_size=batch_size,
                        learning_rate=float(config.get("learning_rate", 1e-3)),
                        temperature=float(config.get("temperature", 2.0)),
                        seed=seed * 1000000 + method_index * 10000 + rho_index,
                        device=device,
                    )
                    row = {
                        "dataset": dataset_name,
                        "seed": seed,
                        "method": method,
                        "benchmark": benchmark.name,
                        benchmark.strength_name: benchmark.strength,
                        **metrics,
                    }
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)

                for k_index, k in enumerate(config["k_values"]):
                    benchmark = necessity_cache[(selection, k_index)]
                    if method in {"replay", "icarl"}:
                        benchmark_metadata.append(
                            {
                                "dataset": dataset_name,
                                "seed": seed,
                                "selection": "icarl" if method == "icarl" else "random",
                                "benchmark": benchmark.name,
                                "k": benchmark.strength,
                                "old_classes": list(boundary.old_classes),
                                "target_class": boundary.target_class,
                                "contrast_class": boundary.contrast_class,
                                "synthetic_class": boundary.synthetic_class,
                                **benchmark.metadata,
                            }
                        )
                    _, metrics = run_method(
                        base_model,
                        benchmark,
                        boundary,
                        method=method,
                        regularization=0.0 if method == "replay" else regularization,
                        epochs=int(config.get("epochs_transition", 8)),
                        batch_size=batch_size,
                        learning_rate=float(config.get("learning_rate", 1e-3)),
                        temperature=float(config.get("temperature", 2.0)),
                        seed=seed * 1000000 + method_index * 10000 + 500 + k_index,
                        device=device,
                    )
                    row = {
                        "dataset": dataset_name,
                        "seed": seed,
                        "method": method,
                        "benchmark": benchmark.name,
                        benchmark.strength_name: benchmark.strength,
                        **metrics,
                    }
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)
            del base_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    replay_cost = {
        (row["dataset"], row["seed"], row["benchmark"], row.get("rho"), row.get("k")): row[
            "training_seconds"
        ]
        for row in rows
        if row["method"] == "replay"
    }
    for row in rows:
        key = (row["dataset"], row["seed"], row["benchmark"], row.get("rho"), row.get("k"))
        row["cost_vs_replay"] = row["training_seconds"] / max(replay_cost[key], 1e-9)
    return {
        "schema_version": 1,
        "git_commit": source_commit(REPO_ROOT),
        "effective_config": config,
        "methods": list(methods),
        "rows": rows,
        "benchmark_metadata": benchmark_metadata,
        "elapsed_seconds": time.perf_counter() - started,
        "scope": "Section 12 ground-truth smoke/full matrix; not a novelty or publication verdict",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--dataset", choices=("NSL-KDD", "UNSW-NB15"))
    parser.add_argument("--output-dir")
    parser.add_argument("--log-path")
    parser.add_argument("--seeds", help="comma-separated override; calibration only")
    parser.add_argument("--methods", help="comma-separated override; calibration only")
    parser.add_argument("--regularization", type=float)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = _effective(json.loads(config_path.read_text(encoding="utf-8")), args.smoke)
    if args.device:
        config["device"] = args.device
    if args.dataset:
        config["datasets"] = [args.dataset]
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.log_path:
        config["log_path"] = args.log_path
    if args.seeds:
        config["seeds"] = [int(value) for value in args.seeds.split(",")]
        config["registered_matrix"] = False
    if args.methods:
        config["methods"] = [value.strip() for value in args.methods.split(",") if value.strip()]
        config["registered_matrix"] = False
    if args.regularization is not None:
        config["regularization"] = args.regularization
    output = (REPO_ROOT / config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_path = (REPO_ROOT / config["log_path"]).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(
        Tee(sys.stderr, log)
    ):
        try:
            print(json.dumps({"config": config, "git_commit": source_commit(REPO_ROOT)}, sort_keys=True))
            result = run(config)
            result_path = output / "crr_result.json"
            result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
            _write_csv(output / "crr_rows.csv", result["rows"])
            (output / "COMPLETED.json").write_text(
                json.dumps(
                    {
                        "git_commit": result["git_commit"],
                        "datasets": config["datasets"],
                        "seeds": config["seeds"],
                        "row_count": len(result["rows"]),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            print(f"CRR_COMPLETE result={result_path} rows={len(result['rows'])}")
        except Exception:
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
