"""Run the fixed 45-cell Section 448 local confirmation in resumable batches."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASETS = ("UNSW-NB15", "CIC-IDS-2017", "CSE-CIC-IDS2018")
DATASET_DIRS = {
    "UNSW-NB15": "unsw-nb15",
    "CIC-IDS-2017": "cic-ids-2017",
    "CSE-CIC-IDS2018": "cic-ids-2018",
}
ARMS = ("AR-Stateless", "V1a", "V1b")
SEEDS = (15, 16, 17, 18, 19)
THREADS = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def planned_cells(dataset: str) -> tuple[tuple[str, int], ...]:
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported confirmation dataset: {dataset}")
    return tuple((arm, seed) for arm in ARMS for seed in SEEDS)


def result_directory(output_root: Path, dataset: str, arm: str, seed: int) -> Path:
    if dataset not in DATASET_DIRS or arm not in ARMS or seed not in SEEDS:
        raise ValueError("Cell is outside the fixed Section 448 confirmation scope")
    return (output_root / "confirm" / DATASET_DIRS[dataset] / arm
            / f"seed{seed}")


def source_paths() -> tuple[str, ...]:
    return (
        "ncd/run_section448_confirm.py",
        "ncd/section138_cell.py",
        "ncd/section448_variants.py",
        "ncd/section156_models.py",
        "ncd/section156_config.py",
        "ncd/section81_exposure.py",
        "ncd/run_aacil.py",
        "ncd/prepare_problem_scout_data.py",
        "ncd/run_problem_scout.py",
        "ncd/default_decision.py",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True,
                        help="Pinned commit containing this entry point; keep constant across dataset batches")
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(THREADS)
    os.environ["MKL_NUM_THREADS"] = str(THREADS)
    os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS)

    import numpy as np
    import sklearn
    import torch
    from ncd.prepare_problem_scout_data import prepare_setting
    from ncd.run_problem_scout import binding, save_json
    from ncd.section138_cell import run_cell
    from ncd.section156_config import configurations
    from ncd.section156_models import build_models, train_gate

    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    data_root = args.data_root.resolve(strict=True)
    output_root = args.output_root.resolve()
    config = json.loads((ROOT / "ncd/configs/problem_scout_section21.json")
                        .read_text(encoding="utf-8"))
    config.update(
        problem_scout_data_root=str(data_root), device="cpu", cpu_threads=THREADS,
        smoke=False, lambda_g=0.0, epochs_per_task=8, min_steps_per_task=2000,
        batch_size=256, eval_batch_size=4096, learning_rate=0.001,
        derpp_alpha=1.0, derpp_beta=1.0, deadline_iso=None,
        normal_training_rows=50000, ema_decay=0.999, ema_consistency=1.0, hidden=128,
        section156={
            "gate": "histgb", "family": "histgb", "control": True,
            "normal_training_rows": 50000, "attack_memory_per_class": 200,
            "gate_retention": False, "gate_schedule": "section123_es",
            "last_layer_cbf_epochs": 0, "store_logits": False,
            "weight_alignment": False, "benign_hard": False,
        },
    )
    data, _, _, _ = prepare_setting(args.dataset, config)
    configuration = next(item for item in configurations() if item.identifier == "C1")

    paths = source_paths()
    hashes = {name: sha256(ROOT / name) for name in paths}

    def trainer(model, x, y, *, schedule, **kwargs):
        return train_gate(model, x, y, kind="histgb", **kwargs)

    for arm, seed in planned_cells(args.dataset):
        cell_dir = result_directory(output_root, args.dataset, arm, seed)
        cell_dir.mkdir(parents=True, exist_ok=True)
        identity = {
            "git_commit": args.source_commit,
            "dataset": args.dataset,
            "seed": seed,
            "method": arm,
            "reference_configuration": "C1" if arm == "AR-Stateless" else "C1+Section448",
            "effective_config": {**config, "problem_scout_data_root": "<DATA_ROOT>"},
            "engineering_defaults": {
                "kmeans_n_init": "auto",
                "kmeans_n_init_scope": "engineering_default_not_method_hyperparameter",
            },
            "data_binding": binding(data, None, None),
            "source_sha256": hashes,
            "command": [
                "run_section448_confirm.py", "--dataset", args.dataset,
                "--seed", str(seed), "--arm", arm, "--threads", str(THREADS),
                "--data-root", "<DATA_ROOT>", "--output-root", "<OUTPUT_ROOT>",
                "--source-commit", args.source_commit,
            ],
            "output_role": "section448_confirmation",
            "smoke": False,
            "protocol": {
                "scope": "fixed follow-up to development screening; not registered S1",
                "arms": list(ARMS), "seeds": list(SEEDS), "dataset_order": list(DATASETS),
                "threads": THREADS,
                "normal_training_rows": 50000,
                "threshold_source": "10% Normal TRAIN holdout; full TRAIN prior",
                "memory_rule": "attack-only bounded memory; stateless HistGB retrained each task",
                "variant_quota": "aa.attack_targets('aacil', seen, normal); capacity includes unchanged train/validation slots",
                "V1_cap": "new-class optimization rows capped at class capacity for V1a and 4x capacity for V1b; validation slots unchanged",
                "kmeans_n_init": "auto",
                "selection_basis": "V1a and V1b advanced by the fixed Section 448 screening rule; AR-Stateless retained as comparator",
            },
        }

        result_path = cell_dir / "cell" / "result.json"
        was_complete = result_path.exists()
        started = time.perf_counter()
        result = run_cell(
            data, config, identity, cell_dir / "cell", configuration,
            model_builder=build_models, gate_trainer=trainer,
            variant=None if arm == "AR-Stateless" else arm,
        )
        elapsed = time.perf_counter() - started

        if not was_complete:
            runtime = {
                "elapsed_seconds": elapsed,
                "platform": platform.platform(),
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
                "cpu_threads": THREADS,
            }
            result["runtime"] = runtime
            save_json(result_path, result)
            save_json(cell_dir / "summary.json", {
                "completed": True, "scientific_result": True,
                "source_commit": identity["git_commit"], "dataset": args.dataset,
                "seed": seed, "method": arm,
                "checkpoints": len(result["checkpoints"]), "runtime": runtime,
            })

        print(json.dumps({
            "completed": True, "scientific_result": True,
            "dataset": args.dataset, "seed": seed, "arm": arm,
            "resumed_existing_result": was_complete,
            "elapsed_seconds_this_attempt": elapsed,
        }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
