"""Exploratory fixed-probe RQ1 observation of the rare-boundary pilot.

This is an offline measurement, not a training entry point or a silent-drift
decision rule. All four completed arms are required. Attribution NPZ files are
LOCAL ONLY; the small JSON contains row identifiers and descriptive metrics.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ncd.explain import expected_gradients
from ncd.model import DecisionScore, MLP
from ncd.prepare_problem_scout_data import prepare_setting
from ncd.run_problem_scout import (
    Tee, binding, canonical_hash, digest, path_label, require_finite, save_json,
)
from ncd.source_commit import source_commit


ARMS = ("random", "mir", "mir_rare_ce", "mir_rare_margin")
TARGET_NAMES = {"NSL-KDD": ("Normal", "R2L"),
                "CIC-IDS-2017": ("Normal", "WebAttack")}
REPEATS, NSAMPLES, TOP_K = 2, 64, 8
PROBES_PER_CLASS, BACKGROUND_ROWS = 8, 32


def resolve(path: str | Path, base: Path = ROOT) -> Path:
    path = Path(path)
    return (path if path.is_absolute() else base / path).resolve()


def array_hash(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def load_checkpoint(path: Path, expected_sha: str, identity: dict | None = None) -> dict:
    if digest(path) != expected_sha:
        raise ValueError(f"Checkpoint SHA differs: {path_label(path)}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    require_finite(state)
    if identity is not None and state.get("identity") != identity:
        raise ValueError("Checkpoint identity differs from its completed result")
    return state


def scorer_from_state(state: dict) -> DecisionScore:
    weights = state["model_state"]
    hidden, inputs = weights["features.0.weight"].shape
    classes = weights["head.weight"].shape[0]
    model = MLP(int(inputs), int(hidden), int(classes))
    model.load_state_dict(weights, strict=True)
    # Explain the true old-class logit, never the predicted class or a margin.
    return DecisionScore(model, "logit", int(classes)).cpu().eval()


def fixed_probes(labels: np.ndarray, classes: list[int], seed: int) -> np.ndarray:
    """Sample by original prepared-test row index, independently of predictions."""
    selected = []
    for class_id in classes:
        candidates = np.flatnonzero(labels == class_id)
        if not len(candidates):
            raise ValueError(f"No test probes available for class {class_id}")
        rng = np.random.default_rng(np.random.SeedSequence([seed, class_id]))
        chosen = rng.choice(candidates, size=min(PROBES_PER_CLASS, len(candidates)),
                            replace=False)
        selected.extend(sorted(map(int, chosen)))
    return np.asarray(selected, dtype=np.int64)


def fixed_background(state: dict, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Round-robin class-balanced sample from saved checkpoint002 memory only."""
    xs = [value.detach().cpu().numpy() for value in state["buffer_x"]]
    ys = [value.detach().cpu().numpy() for value in state["buffer_y"]]
    if not xs or len(xs) != len(ys):
        raise ValueError("Starting checkpoint has no valid saved replay memory")
    x, y = np.concatenate(xs), np.concatenate(ys)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(x) < BACKGROUND_ROWS:
        raise ValueError("Starting saved memory must contain at least 32 valid rows")
    classes = sorted(map(int, np.unique(y)))
    if not set(classes).issubset(set(map(int, state["seen_classes"]))):
        raise ValueError("Starting memory contains a not-yet-seen class")
    queues = {}
    for class_id in classes:
        indices = np.flatnonzero(y == class_id)
        rng = np.random.default_rng(np.random.SeedSequence([seed, class_id]))
        queues[class_id] = list(map(int, rng.permutation(indices)))
    offsets = {class_id: 0 for class_id in classes}
    selected = []
    while len(selected) < BACKGROUND_ROWS:
        for class_id in classes:
            offset = offsets[class_id]
            if offset < len(queues[class_id]):
                selected.append(queues[class_id][offset])
                offsets[class_id] += 1
                if len(selected) == BACKGROUND_ROWS:
                    break
    indices = np.asarray(selected, dtype=np.int64)
    return np.asarray(x[indices], dtype=np.float32), y[indices].astype(np.int64), indices


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Ascending ranks with average ranks for exact ties; no arbitrary tie jitter."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    first = 0
    while first < len(order):
        end = first + 1
        while end < len(order) and values[order[end]] == values[order[first]]:
            end += 1
        ranks[order[first:end]] = (first + end - 1) / 2.0 + 1.0
        first = end
    return ranks


def similarity(left: np.ndarray, right: np.ndarray) -> dict:
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.ndim != 1 or right.shape != left.shape or len(left) < TOP_K:
        raise ValueError("Attribution vectors must share at least eight features")
    require_finite([left, right])
    abs_left, abs_right = np.abs(left), np.abs(right)
    zero_left, zero_right = not np.any(abs_left), not np.any(abs_right)
    lrank, rrank = average_ranks(abs_left), average_ranks(abs_right)
    lrank, rrank = lrank - lrank.mean(), rrank - rrank.mean()
    rank_denom = np.linalg.norm(lrank) * np.linalg.norm(rrank)
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    li = np.lexsort((np.arange(len(left)), -abs_left))[:TOP_K]
    ri = np.lexsort((np.arange(len(right)), -abs_right))[:TOP_K]
    intersection = len(set(map(int, li)) & set(map(int, ri)))
    return {
        "top8_jaccard": None if zero_left or zero_right else intersection / (2 * TOP_K - intersection),
        "cosine_signed": None if denom == 0 else float(np.clip(left @ right / denom, -1, 1)),
        "spearman_abs_importance": None if rank_denom == 0 else float(np.clip(lrank @ rrank / rank_denom, -1, 1)),
        "zero_attribution_left": bool(zero_left), "zero_attribution_right": bool(zero_right),
        "left_cutoff_tied_features": int(np.sum(abs_left == abs_left[li[-1]])),
        "right_cutoff_tied_features": int(np.sum(abs_right == abs_right[ri[-1]])),
    }


def summarize(rows: list[dict]) -> dict:
    result = {"rows": len(rows)}
    for metric in ("top8_jaccard", "cosine_signed", "spearman_abs_importance"):
        values = [row[metric] for row in rows if row[metric] is not None]
        result[metric] = {"mean": float(np.mean(values)) if values else None,
                          "valid_rows": len(values), "undefined_rows": len(rows) - len(values)}
    return result


def read_inputs(pilot_root: Path, dataset: str, seed: int) -> tuple[dict, list[dict], dict]:
    slug = dataset.lower().replace("-", "_")
    records = []
    common = None
    for arm in ARMS:
        result_path = pilot_root / slug / f"{arm}_seed{seed}" / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("completed") is not True:
            raise ValueError(f"Required arm is not complete: {arm}")
        identity = result["identity"]
        if identity.get("smoke") is not False:
            raise ValueError("This exploratory observation requires completed non-smoke arms")
        if (identity["dataset"], int(identity["seed"]), identity["arm"]) != (dataset, seed, arm):
            raise ValueError("Pilot dataset/seed/arm identity differs")
        comparison = {key: identity[key] for key in
                      ("dataset", "seed", "class_order", "tasks", "data_binding", "start_checkpoint")}
        if common is None:
            common = comparison
        elif comparison != common:
            raise ValueError("Four arms must have identical data and a common starting checkpoint")
        records.append({"arm": arm, "result_path": result_path, "result": result})
    start = common["start_checkpoint"]
    start_result_path = resolve(start["result_path"])
    if digest(start_result_path) != start["result_sha256"]:
        raise ValueError("Starting training-result SHA differs")
    start_result = json.loads(start_result_path.read_text(encoding="utf-8"))
    state = load_checkpoint(resolve(start["path"]), start["sha256"], start_result["identity"])
    if int(state["task_index"]) != 2:
        raise ValueError("Common starting state must be checkpoint002")
    for key in ("dataset", "seed", "class_order", "tasks", "data_binding"):
        if state["identity"][key] != common[key]:
            raise ValueError(f"Starting checkpoint differs in {key}")
    return state, records, common


def run(args: argparse.Namespace, output: Path) -> None:
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    state, records, common = read_inputs(resolve(args.pilot_root), args.dataset, args.seed)
    identity = records[0]["result"]["identity"]
    config = identity["effective_config"]
    # This existing loader only verifies and opens frozen prepared arrays;
    # it never prepares new data or fits a transform.
    data, shift_x, shift_y, metadata = prepare_setting(args.dataset, config)
    if binding(data, shift_x, shift_y) != common["data_binding"]:
        raise ValueError("Prepared arrays differ from the training data binding")
    if data.class_order != common["class_order"] or data.tasks != common["tasks"]:
        raise ValueError("Prepared class/task order differs")
    classes = [data.class_order.index(name) for name in TARGET_NAMES[args.dataset]]
    seen_start = list(map(int, state["seen_classes"]))
    expected_start = [int(c) for task in data.tasks[:3] for c in task]
    if seen_start != expected_start or not set(classes).issubset(seen_start):
        raise ValueError("Probe classes must be seen at the common checkpoint002")
    if classes[-1] != expected_start[-1]:
        raise ValueError("Named rare class is not the final old class")
    probe_indices = fixed_probes(data.y_test, classes, args.probe_seed)
    probe_x = np.asarray(data.x_test[probe_indices], dtype=np.float32)
    probe_y = np.asarray(data.y_test[probe_indices], dtype=np.int64)
    background_x, background_y, background_indices = fixed_background(state, args.background_seed)
    background = torch.from_numpy(background_x)
    all_attrs, all_logits, all_predictions = [], [], []
    checkpoint_records, noise_records = [], []
    entries = [("start", state, common["start_checkpoint"]["path"], common["start_checkpoint"]["sha256"])]
    for record in records:
        result, cell_path = record["result"], record["result_path"].parent
        ref = result["final_checkpoint"]
        path = resolve(ref["path"], cell_path)
        final = load_checkpoint(path, ref["sha256"], result["identity"])
        if int(final["task_index"]) != 3:
            raise ValueError("Pilot endpoints must be task3/checkpoint003")
        entries.append((record["arm"], final, path_label(path), ref["sha256"]))
    for checkpoint_name, checkpoint, checkpoint_path, checkpoint_sha in entries:
        checkpoint_started = time.perf_counter()
        scorer = scorer_from_state(checkpoint)
        expected_seen = [int(c) for task in data.tasks[:int(checkpoint["task_index"]) + 1] for c in task]
        seen = list(map(int, checkpoint.get("seen_classes", expected_seen)))
        if seen != expected_seen:
            raise ValueError("Checkpoint seen-class order differs")
        with torch.no_grad():
            logits = scorer(torch.from_numpy(probe_x)).numpy()
        require_finite(logits)
        predictions = np.asarray(seen, dtype=np.int64)[logits[:, seen].argmax(axis=1)]
        attrs = np.empty((REPEATS, len(probe_indices), probe_x.shape[1]), dtype=np.float32)
        for class_id in classes:
            positions = np.flatnonzero(probe_y == class_id)
            # Crucially no checkpoint/arm offset: Monte Carlo draws are paired.
            rseed = args.explainer_seed + int(class_id) * 100
            attrs[:, positions, :] = expected_gradients(
                scorer, int(class_id), torch.from_numpy(probe_x[positions]),
                background, REPEATS, NSAMPLES, rseed,
            )
        require_finite(attrs)
        noise_rows = []
        for pos, test_index in enumerate(probe_indices):
            noise_rows.append({"test_row_index": int(test_index), "class_id": int(probe_y[pos]),
                               **similarity(attrs[0, pos], attrs[1, pos])})
        noise_records.append({"checkpoint": checkpoint_name, "repeat_pair": [0, 1],
                              "rows": noise_rows, "summary": summarize(noise_rows)})
        checkpoint_records.append({"checkpoint": checkpoint_name, "task_index": int(checkpoint["task_index"]),
                                   "path": checkpoint_path, "sha256": checkpoint_sha,
                                   "seen_classes": seen, "elapsed_seconds": time.perf_counter() - checkpoint_started})
        all_attrs.append(attrs)
        all_logits.append(logits)
        all_predictions.append(predictions)
        print(f"EXPLAINED checkpoint={checkpoint_name} probes={len(probe_indices)} repeats={REPEATS}", flush=True)
        del scorer
    attrs = np.stack(all_attrs)
    logits, predictions = np.stack(all_logits), np.stack(all_predictions)
    comparisons = []
    for end, arm in enumerate(ARMS, start=1):
        rows = []
        for pos, test_index in enumerate(probe_indices):
            class_id = int(probe_y[pos])
            per_repeat = [dict(repeat=repeat, **similarity(attrs[0, repeat, pos], attrs[end, repeat, pos]))
                          for repeat in range(REPEATS)]
            rows.append({"test_row_index": int(test_index), "class_id": class_id,
                         "class_name": data.class_order[class_id],
                         "start_prediction": int(predictions[0, pos]), "end_prediction": int(predictions[end, pos]),
                         "prediction_stable": bool(predictions[0, pos] == predictions[end, pos]),
                         "start_correct": bool(predictions[0, pos] == class_id),
                         "end_correct": bool(predictions[end, pos] == class_id),
                         "start_target_logit": float(logits[0, pos, class_id]),
                         "end_target_logit": float(logits[end, pos, class_id]),
                         "paired_repeat_metrics": per_repeat,
                         **similarity(attrs[0, :, pos].mean(axis=0), attrs[end, :, pos].mean(axis=0))})
        comparisons.append({"arm": arm, "rows": rows, "summary_mean_attribution": summarize(rows),
                            "prediction_stable_rows": sum(row["prediction_stable"] for row in rows),
                            "class_summaries": {data.class_order[c]: summarize([row for row in rows if row["class_id"] == c]) for c in classes}})
    raw_path = output / "attributions_local_only.npz"
    np.savez_compressed(raw_path, attributions=attrs, logits=logits, predictions=predictions,
                        checkpoint_names=np.asarray([entry[0] for entry in entries]),
                        test_row_indices=probe_indices, target_class_ids=probe_y,
                        background_memory_indices=background_indices, background_labels=background_y,
                        feature_names=np.asarray(data.feature_names))
    import shap
    portable_command = (
        f"python -I ncd/explain_rare_boundary_pilot.py --pilot-root {path_label(resolve(args.pilot_root))} "
        f"--dataset {args.dataset} --seed {args.seed} --output-dir {path_label(output)} "
        f"--probe-seed {args.probe_seed} --background-seed {args.background_seed} "
        f"--explainer-seed {args.explainer_seed}"
    )
    result = {
        "schema_version": 1, "completed": True, "dataset": args.dataset, "seed": args.seed,
        "git_commit": source_commit(ROOT), "script_sha256": digest(Path(__file__)),
        "command": portable_command, "command_note": "Equivalent repo-relative command; local command invocation is recorded by the runner.",
        "log_path": path_label(output / "run.log"),
        "scope": "Exploratory small fixed-probe RQ1 observation; no silent-drift threshold, hypothesis test, causal claim, or RQ1/overall completion claim.",
        "config": {"probe_seed": args.probe_seed, "background_seed": args.background_seed,
                   "explainer_seed": args.explainer_seed, "probes_per_class_max": PROBES_PER_CLASS,
                   "background_rows": BACKGROUND_ROWS, "repeats": REPEATS, "nsamples": NSAMPLES,
                   "top_k": TOP_K, "device": "cpu", "torch_threads": 2, "torch_interop_threads": 2,
                   "target_classes": list(TARGET_NAMES[args.dataset])},
        "versions": {"torch": str(torch.__version__), "numpy": np.__version__, "shap": shap.__version__},
        "sources": [{"arm": r["arm"], "result_path": path_label(r["result_path"]),
                     "result_sha256": digest(r["result_path"]), "training_git_commit": r["result"]["identity"].get("git_commit")} for r in records],
        "data_binding": common["data_binding"], "data_metadata_sha256": canonical_hash(metadata),
        "class_order": data.class_order, "feature_names": data.feature_names,
        "probe": {"population": "Original prepared test rows of Normal and the named final old class; never prediction-filtered.",
                  "test_row_indices": probe_indices.tolist(), "class_ids": probe_y.tolist(),
                  "x_sha256": array_hash(probe_x), "y_sha256": array_hash(probe_y)},
        "background": {"population": "Only the common checkpoint002 saved replay memory; class round-robin stratified 32 rows without replacement.",
                       "memory_row_indices": background_indices.tolist(), "class_ids": background_y.tolist(),
                       "x_sha256": array_hash(background_x), "y_sha256": array_hash(background_y)},
        "measurement": {"explainer": "ncd.explain.expected_gradients / shap.GradientExplainer",
                        "decision_score": "Actual true-old-class classifier logit, identical target across all checkpoints.",
                        "prediction": "Argmax over each checkpoint's seen classes, as in the base evaluation.",
                        "paired_randomness": "Per-class rseed=explainer_seed+100*class_id; repeats use rseed+0/+1 at every checkpoint.",
                        "metrics": "Top8 Jaccard and Spearman use absolute attribution importance; cosine uses signed attributions. Exact rank ties use average ranks; top8 ties use feature index. Zero-norm cosine/all-zero top8 and constant-rank Spearman are null, not evidence of stability.",
                        "summary": "Unweighted row means of metrics from repeat-mean attributions; paired-repeat metrics and same-checkpoint repeat noise are separately retained.",
                        "limitations": "At most 8 probes/class and 32 background rows; two Monte Carlo repeats characterize only a small noise sample. Stable predictions may be stably wrong. Attribution changes are not automatically silent explanation drift or a causal mechanism."},
        "checkpoints": checkpoint_records, "comparisons": comparisons,
        "within_checkpoint_repeat_noise": noise_records,
        "raw_attributions": {"path": raw_path.name, "sha256": digest(raw_path), "local_only": True,
                             "shape": list(attrs.shape), "axes": ["checkpoint", "repeat", "probe", "feature"]},
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(output / "result.json", result)
    print(f"COMPLETE result={path_label(output / 'result.json')} local_only_npz={raw_path.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", required=True)
    parser.add_argument("--dataset", required=True, choices=list(TARGET_NAMES))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--probe-seed", type=int, default=20261019)
    parser.add_argument("--background-seed", type=int, default=20261020)
    parser.add_argument("--explainer-seed", type=int, default=20261021)
    args = parser.parse_args()
    output = resolve(args.output_dir)
    # No overwrite/resume shortcut: preserve earlier raw attributions and logs.
    output.mkdir(parents=True, exist_ok=False)
    with (output / "run.log").open("x", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                run(args, output)
            except Exception:
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
