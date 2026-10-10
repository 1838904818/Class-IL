"""Generate repeated, probe-bootstrapped explanations for one Exp 0 run."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

# SHAP gradient computations can initialize CUDA; configure deterministic
# CuBLAS before importing PyTorch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch import nn

from ncd.model import DecisionScore, MLP
from ncd.source_commit import source_commit
from ncd.train_cil import Tee, seed_everything


EXPLAINERS = ("expected_gradients", "feature_ablation", "grad_x_input")


class TargetScore(nn.Module):
    def __init__(self, scorer: DecisionScore, target: int):
        super().__init__()
        self.scorer = scorer
        self.target = target

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.scorer(inputs)[:, self.target : self.target + 1]


def as_feature_matrix(values: object, n_rows: int, n_features: int) -> np.ndarray:
    if isinstance(values, list):
        if len(values) != 1:
            raise ValueError(f"unexpected SHAP output list length: {len(values)}")
        values = values[0]
    array = np.asarray(values, dtype=np.float32)
    if array.shape == (n_rows, n_features, 1):
        array = array[:, :, 0]
    elif array.shape == (1, n_rows, n_features):
        array = array[0]
    if array.shape != (n_rows, n_features):
        raise ValueError(f"unexpected SHAP output shape: {array.shape}")
    return array


def expected_gradients(
    scorer: DecisionScore,
    target: int,
    probe: torch.Tensor,
    background: torch.Tensor,
    repeats: int,
    nsamples: int,
    seed: int,
) -> np.ndarray:
    import shap

    wrapped = TargetScore(scorer, target).eval()
    explainer = shap.GradientExplainer(wrapped, background)
    output: list[np.ndarray] = []
    for repeat in range(repeats):
        values = explainer.shap_values(probe, nsamples=nsamples, rseed=seed + repeat)
        output.append(as_feature_matrix(values, len(probe), probe.shape[1]))
    return np.stack(output, axis=0)


def feature_ablation(
    scorer: DecisionScore,
    target: int,
    probe: torch.Tensor,
    baseline: torch.Tensor,
) -> np.ndarray:
    scorer.eval()
    with torch.no_grad():
        reference = scorer(probe)[:, target]
        parts: list[torch.Tensor] = []
        for feature in range(probe.shape[1]):
            ablated = probe.clone()
            ablated[:, feature] = baseline[feature]
            parts.append(reference - scorer(ablated)[:, target])
    return torch.stack(parts, dim=1).cpu().numpy().astype(np.float32)


def grad_x_input(
    scorer: DecisionScore,
    target: int,
    probe: torch.Tensor,
    baseline: torch.Tensor,
) -> np.ndarray:
    scorer.eval()
    inputs = probe.detach().clone().requires_grad_(True)
    score = scorer(inputs)[:, target].sum()
    gradient = torch.autograd.grad(score, inputs, create_graph=False)[0]
    return (gradient * (inputs - baseline)).detach().cpu().numpy().astype(np.float32)


def bootstrap_estimates(
    attributions: np.ndarray,
    bootstraps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized mean-|attribution| estimates and repeat/bootstrap ids."""
    repeats, n_rows, n_features = attributions.shape
    estimates: list[np.ndarray] = []
    repeat_ids: list[int] = []
    bootstrap_ids: list[int] = []
    rng = np.random.default_rng(seed)
    for repeat in range(repeats):
        absolute = np.abs(attributions[repeat])
        for bootstrap in range(bootstraps):
            chosen = rng.integers(0, n_rows, size=n_rows)
            vector = absolute[chosen].mean(axis=0)
            vector /= vector.sum() + 1e-12
            estimates.append(vector.astype(np.float32))
            repeat_ids.append(repeat)
            bootstrap_ids.append(bootstrap)
    return (
        np.stack(estimates, axis=0),
        np.asarray(repeat_ids, dtype=np.int16),
        np.asarray(bootstrap_ids, dtype=np.int16),
    )


def load_scorer(checkpoint_path: Path, device: torch.device) -> tuple[DecisionScore, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MLP(payload["input_dim"], payload["hidden"], payload["n_classes"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    prototypes = {int(k): v.to(device) for k, v in payload.get("prototypes", {}).items()}
    scorer = DecisionScore(model, payload["inference"], payload["n_classes"], prototypes).to(device).eval()
    return scorer, payload


def run(args: argparse.Namespace) -> None:
    result_path = (REPO_ROOT / args.training_result).resolve()
    training = json.loads(result_path.read_text(encoding="utf-8"))
    run_dir = result_path.parent
    output_dir = run_dir / "explanations"
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_data = np.load(run_dir / training["probe_background"], allow_pickle=False)
    probe_x = probe_data["probe_x"].astype(np.float32)
    probe_y = probe_data["probe_y"].astype(np.int64)
    background_x = probe_data["background_x"].astype(np.float32)
    feature_names = probe_data["feature_names"]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seed_everything(int(training["seed"]))
    started = time.perf_counter()
    records: list[dict] = []
    print(
        f"dataset={training['dataset']} method={training['method']} seed={training['seed']} "
        f"repeats={args.repeats} bootstraps={args.bootstraps} eg_nsamples={args.eg_nsamples}"
    )
    for checkpoint_record in training["checkpoints"]:
        checkpoint = int(checkpoint_record["checkpoint"])
        scorer, payload = load_scorer(run_dir / checkpoint_record["path"], device)
        background = torch.from_numpy(background_x).to(device)
        baseline = background.mean(dim=0)
        for class_id in payload["seen_classes"]:
            class_id = int(class_id)
            class_probe = probe_x[probe_y == class_id]
            probe = torch.from_numpy(class_probe).to(device)
            print(f"checkpoint={checkpoint} class={class_id} probe_rows={len(class_probe)}")
            for explainer in EXPLAINERS:
                item_started = time.perf_counter()
                item_seed = int(training["seed"]) * 1_000_000 + checkpoint * 10_000 + class_id * 100
                if explainer == "expected_gradients":
                    attrs = expected_gradients(
                        scorer,
                        class_id,
                        probe,
                        background,
                        args.repeats,
                        args.eg_nsamples,
                        item_seed,
                    )
                elif explainer == "feature_ablation":
                    single = feature_ablation(scorer, class_id, probe, baseline)
                    attrs = np.repeat(single[None, :, :], args.repeats, axis=0)
                else:
                    single = grad_x_input(scorer, class_id, probe, baseline)
                    attrs = np.repeat(single[None, :, :], args.repeats, axis=0)
                estimates, repeat_ids, bootstrap_ids = bootstrap_estimates(
                    attrs,
                    args.bootstraps,
                    item_seed + 57,
                )
                artifact = output_dir / f"checkpoint_{checkpoint:03d}_class_{class_id:02d}_{explainer}.npz"
                np.savez_compressed(
                    artifact,
                    attributions=attrs,
                    estimates=estimates,
                    repeat_ids=repeat_ids,
                    bootstrap_ids=bootstrap_ids,
                    probe_row_count=np.asarray([len(class_probe)], dtype=np.int32),
                    feature_names=feature_names,
                )
                elapsed = time.perf_counter() - item_started
                records.append(
                    {
                        "checkpoint": checkpoint,
                        "class_id": class_id,
                        "explainer": explainer,
                        "path": str(artifact.relative_to(run_dir)).replace("\\", "/"),
                        "probe_rows": len(class_probe),
                        "estimate_count": len(estimates),
                        "elapsed_seconds": elapsed,
                    }
                )
                print(f"  explainer={explainer} estimates={len(estimates)} elapsed={elapsed:.2f}s")
        del scorer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {
        "dataset": training["dataset"],
        "method": training["method"],
        "seed": training["seed"],
        "git_commit": source_commit(REPO_ROOT),
        "training_result": str(result_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "training_git_commit": training["git_commit"],
        "command": " ".join(sys.argv),
        "log_path": args.log_path,
        "decision_score": "actual NME negative squared distance" if training["method"] == "icarl" else "actual classifier logit",
        "explainers": list(EXPLAINERS),
        "repeats": args.repeats,
        "probe_bootstraps": args.bootstraps,
        "expected_gradients_nsamples": args.eg_nsamples,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "records": records,
        "elapsed_seconds": time.perf_counter() - started,
    }
    result_file = run_dir / "explanation_result.json"
    result_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved={result_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-result", required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--bootstraps", type=int, default=10)
    parser.add_argument("--eg-nsamples", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-path", required=True)
    args = parser.parse_args()
    log_path = (REPO_ROOT / args.log_path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        with redirect_stdout(Tee(sys.__stdout__, log_file)), redirect_stderr(Tee(sys.__stderr__, log_file)):
            try:
                run(args)
            except Exception:
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
