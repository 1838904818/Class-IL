from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ncd.data import PreparedData, prepare_dataset
from ncd.explain import load_scorer
from ncd.source_commit import source_commit


def metrics_from_predictions(
    y_true: np.ndarray,
    predicted: np.ndarray,
    seen_classes: list[int],
    n_classes: int,
) -> dict[str, object]:
    mask = np.isin(y_true, seen_classes)
    truth = y_true[mask]
    guesses = predicted[mask]
    if len(truth) == 0:
        raise RuntimeError("seen-class evaluation population is empty")
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (truth, guesses), 1)
    recalls: dict[str, float] = {}
    f1_values: list[float] = []
    for class_id in seen_classes:
        true_positive = int(confusion[class_id, class_id])
        support = int(confusion[class_id].sum())
        predicted_positive = int(confusion[:, class_id].sum())
        recall = true_positive / support if support else float("nan")
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls[str(class_id)] = float(recall)
        f1_values.append(float(f1))
    return {
        "seen_test_rows": int(len(truth)),
        "accuracy_seen": float(np.mean(guesses == truth)),
        "balanced_accuracy_seen": float(np.mean(list(recalls.values()))),
        "macro_f1_seen": float(np.mean(f1_values)),
        "per_class_recall": recalls,
        "confusion_matrix": confusion.tolist(),
    }


def predict(
    checkpoint_path: Path,
    data: PreparedData,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict]:
    scorer, payload = load_scorer(checkpoint_path, device)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(data.x_test), batch_size):
            batch = torch.from_numpy(data.x_test[start : start + batch_size]).to(device)
            outputs.append(scorer(batch).argmax(dim=1).cpu().numpy())
    return np.concatenate(outputs), payload


def summarize_forgetting(checkpoints: list[dict[str, object]]) -> dict[str, object]:
    final = checkpoints[-1]
    final_recalls = {int(key): float(value) for key, value in final["per_class_recall"].items()}
    forgetting: dict[str, float] = {}
    for class_id, final_recall in final_recalls.items():
        history = [
            float(item["per_class_recall"][str(class_id)])
            for item in checkpoints
            if str(class_id) in item["per_class_recall"]
        ]
        if len(history) >= 2:
            forgetting[str(class_id)] = float(max(history[:-1]) - final_recall)
    return {
        "average_forgetting_prior_classes": (
            float(np.mean(list(forgetting.values()))) if forgetting else None
        ),
        "forgetting_by_class": forgetting,
        "final_accuracy_seen": final["accuracy_seen"],
        "final_balanced_accuracy_seen": final["balanced_accuracy_seen"],
        "final_macro_f1_seen": final["macro_f1_seen"],
        "final_normal_recall": final_recalls.get(0),
        "final_confusion_matrix": final["confusion_matrix"],
    }


def collect_training_runs(
    run_repo: Path,
    state_path: Path | None,
    direct_training_results: list[Path],
) -> tuple[str, list[dict[str, object]], dict[str, object]]:
    """Resolve one dataset's completed training results without retraining.

    Dataset-runner state remains the normal HPC input.  Direct training results
    support the already-completed Replay/iCaRL runs whose original runner did
    not emit a dataset state file.
    """
    run_specs: list[dict[str, object]] = []
    provenance: dict[str, object] = {
        "upstream_state_path": None,
        "upstream_source_commit": None,
        "direct_training_results": [],
    }
    dataset: str | None = None

    if state_path is not None:
        resolved_state = state_path if state_path.is_absolute() else run_repo / state_path
        resolved_state = resolved_state.resolve(strict=True)
        state = json.loads(resolved_state.read_text(encoding="utf-8"))
        if not state.get("complete") or len(state.get("completed_runs", [])) != state.get(
            "expected_runs"
        ):
            raise RuntimeError("upstream dataset run is incomplete")
        dataset = str(state["dataset"])
        provenance["upstream_state_path"] = str(resolved_state)
        provenance["upstream_source_commit"] = state.get("source_commit")
        for run in state["completed_runs"]:
            training_path = (run_repo / run["training_result"]).resolve(strict=True)
            run_specs.append(
                {
                    "method": str(run["method"]),
                    "seed": int(run["seed"]),
                    "training_path": training_path,
                    "training_result": str(run["training_result"]),
                }
            )

    for supplied_path in direct_training_results:
        supplied_label = supplied_path.as_posix()
        training_path = (
            supplied_path if supplied_path.is_absolute() else Path.cwd() / supplied_path
        ).resolve(strict=True)
        training = json.loads(training_path.read_text(encoding="utf-8"))
        training_dataset = str(training["dataset"])
        if dataset is None:
            dataset = training_dataset
        elif training_dataset != dataset:
            raise ValueError(
                f"mixed datasets are not allowed: expected {dataset}, got {training_dataset}"
            )
        run_specs.append(
            {
                "method": str(training["method"]),
                "seed": int(training["seed"]),
                "training_path": training_path,
                "training_result": supplied_label,
            }
        )
        provenance["direct_training_results"].append(supplied_label)

    if dataset is None or not run_specs:
        raise ValueError("provide --state-path and/or at least one --training-result")
    identities = [(str(run["method"]), int(run["seed"])) for run in run_specs]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate method/seed training result")
    return dataset, run_specs, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Exp 3 checkpoints on the frozen test split")
    parser.add_argument("--run-repo", type=Path, default=Path("."))
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--training-result", type=Path, action="append", default=[])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    run_repo = args.run_repo.resolve(strict=True)
    dataset, run_specs, provenance = collect_training_runs(
        run_repo, args.state_path, args.training_result
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    data = prepare_dataset(dataset, args.data_root)

    runs: list[dict[str, object]] = []
    for run in run_specs:
        training_path = Path(run["training_path"])
        training = json.loads(training_path.read_text(encoding="utf-8"))
        checkpoint_metrics: list[dict[str, object]] = []
        for checkpoint in training["checkpoints"]:
            checkpoint_id = int(checkpoint["checkpoint"])
            predicted, payload = predict(
                training_path.parent / checkpoint["path"], data, device, args.batch_size
            )
            seen_classes = [int(value) for value in payload["seen_classes"]]
            metrics = metrics_from_predictions(
                data.y_test, predicted, seen_classes, len(data.class_order)
            )
            recorded = {str(key): float(value) for key, value in checkpoint["per_class_recall"].items()}
            max_error = max(
                abs(recorded[key] - float(metrics["per_class_recall"][key]))
                for key in recorded
            )
            if max_error > 1.0e-8:
                raise RuntimeError(
                    f"checkpoint recall mismatch for {run['method']} seed {run['seed']}: {max_error}"
                )
            checkpoint_metrics.append(
                {
                    "checkpoint": checkpoint_id,
                    "seen_classes": seen_classes,
                    "recorded_recall_max_abs_error": max_error,
                    **metrics,
                }
            )
        summary = summarize_forgetting(checkpoint_metrics)
        runs.append(
            {
                "method": run["method"],
                "seed": int(run["seed"]),
                "training_result": run["training_result"],
                "checkpoints": checkpoint_metrics,
                **summary,
            }
        )
        print(
            "PERFORMANCE_RUN_OK "
            f"dataset={dataset} method={run['method']} seed={run['seed']} "
            f"balanced_accuracy={summary['final_balanced_accuracy_seen']:.6f} "
            f"forgetting={summary['average_forgetting_prior_classes']:.6f}",
            flush=True,
        )

    payload = {
        "analysis": "Exp 3 frozen-test performance and forgetting",
        "analysis_source_commit": source_commit(REPO_ROOT),
        **provenance,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "dataset": dataset,
        "class_order": data.class_order,
        "test_rows": int(len(data.y_test)),
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "runs": runs,
        "limitations": [
            "The frozen test split is used only for post-training evaluation, not model selection.",
            "Accuracy and macro F1 use the currently seen-class test population; predictions into unseen output nodes count as errors.",
            "Average forgetting is the mean max-past-minus-final recall over classes observed before the final checkpoint.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"PERFORMANCE_COMPLETE output={args.output} runs={len(runs)}", flush=True)


if __name__ == "__main__":
    main()
