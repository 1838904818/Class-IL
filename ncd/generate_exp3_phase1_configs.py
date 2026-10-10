"""Generate the frozen Exp 3 configs for the five remaining Phase I methods."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "ncd" / "configs"
DATASETS = {"NSL-KDD": "nsl-kdd", "UNSW-NB15": "unsw-nb15"}
METHODS = ("naive", "ewc", "lwf", "replay-herding", "replay-dpmeans")
SEEDS = (1, 2, 3, 4, 42)


def config_for(dataset: str, dataset_slug: str, method: str, seed: int) -> dict:
    method_token = method.replace("-", "_")
    short_dataset = "nsl" if dataset == "NSL-KDD" else "unsw"
    config = {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "data_root": "../kaggle_leon_release/datasets",
        "device": "cuda",
        "hidden": 128,
        "batch_size": 256,
        "epochs_per_task": 8,
        "learning_rate": 0.001,
        "output_dir": (
            f"ncd/results/exp3/runs/{dataset_slug}/{method}/seed{seed}_deterministic"
        ),
        "log_path": (
            f"ncd/logs/exp3_{short_dataset}_{method_token}_seed{seed}_train_deterministic.log"
        ),
    }
    if method in {"replay-herding", "replay-dpmeans"}:
        config["buffer_per_class"] = 200
    if method == "ewc":
        config["ewc_lambda"] = 5000.0
        config["fisher_sample_size"] = 2000
    if method == "lwf":
        config["distillation_temperature"] = 2.0
        config["distillation_alpha"] = 1.0
    return config


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for dataset, dataset_slug in DATASETS.items():
        short_dataset = "nsl" if dataset == "NSL-KDD" else "unsw"
        for method in METHODS:
            method_token = method.replace("-", "_")
            for seed in SEEDS:
                path = CONFIG_DIR / f"exp3_{short_dataset}_{method_token}_seed{seed}.json"
                text = json.dumps(
                    config_for(dataset, dataset_slug, method, seed), indent=2
                ) + "\n"
                if path.exists() and path.read_text(encoding="utf-8") != text:
                    raise RuntimeError(f"refusing to replace mismatched config: {path}")
                path.write_text(text, encoding="utf-8")
                written.append(path.relative_to(REPO_ROOT).as_posix())
    print(json.dumps({"written": written, "count": len(written)}, indent=2))


if __name__ == "__main__":
    main()
