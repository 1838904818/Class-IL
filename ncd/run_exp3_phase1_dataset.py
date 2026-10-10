"""Run and resume the five remaining Exp 3 Phase I methods for one dataset."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ncd.source_commit import source_commit


REMAINING_METHODS = ("naive", "ewc", "lwf", "replay-herding", "replay-dpmeans")
ALL_PHASE_I_METHODS = (
    "naive", "ewc", "lwf", "replay", "replay-herding", "replay-dpmeans", "icarl",
)
SEEDS = (1, 2, 3, 4, 42)
DATASETS = {
    "nsl": ("NSL-KDD", "nsl-kdd", REMAINING_METHODS),
    "unsw": ("UNSW-NB15", "unsw-nb15", REMAINING_METHODS),
    "cic17": ("CIC-IDS-2017", "cic-ids-2017", ALL_PHASE_I_METHODS),
    "cic18": ("CSE-CIC-IDS2018", "cic-ids-2018", ALL_PHASE_I_METHODS),
}
EXPLAINERS = {"expected_gradients", "feature_ablation", "grad_x_input"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def training_complete(
    result_path: Path,
    *,
    dataset: str,
    method: str,
    seed: int,
    commit: str,
) -> bool:
    if not result_path.is_file():
        return False
    result = read_json(result_path)
    if (
        result.get("dataset") != dataset
        or result.get("method") != method
        or result.get("seed") != seed
        or result.get("git_commit") != commit
        or result.get("deterministic_algorithms_enabled") is not True
    ):
        return False
    run_dir = result_path.parent
    checkpoints = result.get("checkpoints", [])
    return bool(checkpoints) and all(
        (run_dir / checkpoint["path"]).is_file() for checkpoint in checkpoints
    ) and (run_dir / result["probe_background"]).is_file()


def explanation_complete(
    result_path: Path,
    *,
    training_path: Path,
    dataset: str,
    method: str,
    seed: int,
    commit: str,
) -> bool:
    if not result_path.is_file():
        return False
    result = read_json(result_path)
    if (
        result.get("dataset") != dataset
        or result.get("method") != method
        or result.get("seed") != seed
        or result.get("git_commit") != commit
        or result.get("training_git_commit") != commit
        or set(result.get("explainers", [])) != EXPLAINERS
        or result.get("repeats") != 10
        or result.get("probe_bootstraps") != 10
        or result.get("expected_gradients_nsamples") != 32
        or result.get("deterministic_algorithms_enabled") is not True
    ):
        return False
    training = read_json(training_path)
    expected = sum(len(item["seen_classes"]) for item in training["checkpoints"]) * len(EXPLAINERS)
    records = result.get("records", [])
    return len(records) == expected and all(
        record.get("estimate_count") == 100
        and (result_path.parent / record["path"]).is_file()
        for record in records
    )


def run_command(command: list[str]) -> None:
    print("COMMAND " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--state-path", required=True)
    args = parser.parse_args()

    commit = source_commit(REPO_ROOT)
    environment_commit = os.environ.get("NCD_SOURCE_COMMIT", "").strip().lower()
    if environment_commit and environment_commit != commit:
        raise RuntimeError("packaged source commit does not match NCD_SOURCE_COMMIT")
    dataset, dataset_slug, methods = DATASETS[args.dataset]
    state_path = (REPO_ROOT / args.state_path).resolve()
    started = time.time()
    completed: list[dict] = []

    for method in methods:
        method_token = method.replace("-", "_")
        for seed in SEEDS:
            config_path = Path("ncd/configs") / f"exp3_{args.dataset}_{method_token}_seed{seed}.json"
            config = read_json(REPO_ROOT / config_path)
            if (
                config.get("dataset") != dataset
                or config.get("method") != method
                or config.get("seed") != seed
                or config.get("device") != "cuda"
                or config.get("epochs_per_task") != 8
            ):
                raise RuntimeError(f"unexpected frozen config: {config_path}")
            run_dir = REPO_ROOT / config["output_dir"]
            training_path = run_dir / "training_result.json"
            explanation_path = run_dir / "explanation_result.json"
            if not training_complete(
                training_path,
                dataset=dataset,
                method=method,
                seed=seed,
                commit=commit,
            ):
                run_command(
                    [sys.executable, "-I", "ncd/train_cil.py", "--config", config_path.as_posix()]
                )
            if not training_complete(
                training_path,
                dataset=dataset,
                method=method,
                seed=seed,
                commit=commit,
            ):
                raise RuntimeError(f"training output validation failed: {training_path}")

            explanation_log = (
                f"ncd/logs/exp3_{args.dataset}_{method_token}_seed{seed}_explain_deterministic.log"
            )
            if not explanation_complete(
                explanation_path,
                training_path=training_path,
                dataset=dataset,
                method=method,
                seed=seed,
                commit=commit,
            ):
                run_command(
                    [
                        sys.executable,
                        "-I",
                        "ncd/explain.py",
                        "--training-result",
                        training_path.relative_to(REPO_ROOT).as_posix(),
                        "--repeats",
                        "10",
                        "--bootstraps",
                        "10",
                        "--eg-nsamples",
                        "32",
                        "--device",
                        "cuda",
                        "--log-path",
                        explanation_log,
                    ]
                )
            if not explanation_complete(
                explanation_path,
                training_path=training_path,
                dataset=dataset,
                method=method,
                seed=seed,
                commit=commit,
            ):
                raise RuntimeError(f"explanation output validation failed: {explanation_path}")
            completed.append(
                {
                    "method": method,
                    "seed": seed,
                    "training_result": training_path.relative_to(REPO_ROOT).as_posix(),
                    "explanation_result": explanation_path.relative_to(REPO_ROOT).as_posix(),
                }
            )
            write_json(
                state_path,
                {
                    "dataset": dataset,
                    "dataset_slug": dataset_slug,
                    "source_commit": commit,
                    "completed_runs": completed,
                    "expected_runs": len(methods) * len(SEEDS),
                    "complete": False,
                    "updated_unix": time.time(),
                },
            )
            print(f"RUN_COMPLETE method={method} seed={seed}", flush=True)

    write_json(
        state_path,
        {
            "dataset": dataset,
            "dataset_slug": dataset_slug,
            "source_commit": commit,
            "completed_runs": completed,
            "expected_runs": len(methods) * len(SEEDS),
            "complete": True,
            "started_unix": started,
            "completed_unix": time.time(),
            "elapsed_seconds": time.time() - started,
        },
    )
    print(f"DATASET_COMPLETE dataset={dataset} runs={len(completed)}", flush=True)


if __name__ == "__main__":
    main()
