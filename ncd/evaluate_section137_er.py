"""Inference-only default decisions from a completed ER trajectory; no retraining."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "2")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--dependency-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    a = p.parse_args()
    if a.output.exists():
        p.error("Output already exists; never overwrite an evaluation")
    sys.path.insert(1, str(a.dependency_root.resolve(strict=True)))
    import numpy as np
    import torch
    from ncd.default_decision import predict_single, metrics_from_predictions, summarize
    from ncd.prepare_problem_scout_data import prepare_setting
    from ncd.run_problem_scout import checkpoint_load, digest, binding
    from ncd.section81_tabtransformer import TabularClassifier
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    cell = a.source_root.resolve(strict=True) / "run/cell"
    raw = json.loads((cell / "result.json").read_text())
    identity = raw["identity"]
    if not raw["completed"] or identity["method"] != "er_bal_tabt":
        raise ValueError("Completed ER trajectory required")
    if identity["effective_config"]["smoke"]:
        raise ValueError("Formal checkpoints required")
    data, _, _, _ = prepare_setting(identity["dataset"], identity["effective_config"])
    if binding(data, None, None) != identity["data_binding"]:
        raise ValueError("Data binding changed")
    records = []
    for cp in raw["checkpoints"]:
        path = cell / cp["checkpoint_file"]
        if digest(path) != cp["checkpoint_sha256"]:
            raise ValueError("Checkpoint hash mismatch")
        state = checkpoint_load(path, identity)
        if state["seen_classes"] != cp["seen_classes"] or state["task_index"] != cp["task_index"]:
            raise ValueError("Checkpoint task mismatch")
        model = TabularClassifier(state["feature_spec"], len(data.class_order)).to(a.device)
        model.load_state_dict(state["model_state"], strict=True)
        model.eval()
        rows = np.flatnonzero(np.isin(data.y_test, cp["seen_classes"]))
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(rows), 256):
                ids = rows[start:start + 256]
                x = torch.as_tensor(np.array(data.x_test[ids], copy=True), device=a.device)
                predictions.append(predict_single(model(x), cp["seen_classes"]).cpu().numpy())
        metrics = metrics_from_predictions(data.y_test[rows], np.concatenate(predictions),
                                           cp["seen_classes"], data.class_order)
        records.append(dict(task_index=cp["task_index"], checkpoint_sha256=cp["checkpoint_sha256"],
                            default_decision=metrics))
        print(f"DEFAULT_STAGE_COMPLETE task={cp['task_index']}", flush=True)
        del model
    result = dict(completed=True, inference_only=True, retraining=False,
                  source_result_sha256=digest(cell / "result.json"),
                  training_source_commit=identity["git_commit"], dataset=identity["dataset"],
                  seed=identity["seed"], device=a.device, checkpoints=records,
                  default_summary=summarize([r["default_decision"] for r in records], data.tasks),
                  elapsed_seconds=time.perf_counter() - started)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
