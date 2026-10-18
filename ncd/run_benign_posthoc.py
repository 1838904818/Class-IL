"""Section 25 D4/D5: fixed test-time diagnostics on saved final checkpoints.

No fitting, threshold search, or training is performed. D5 uses the exact saved
Replay exemplars, raw penultimate embeddings, mean Euclidean distance to k=10
neighbors, and a within-class leave-one-out 95th percentile (linear quantile).
It is an alert-suppressing diagnostic under RQ2, not RQ3 governance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ncd.model import MLP
from ncd.prepare_problem_scout_data import prepare_setting
from ncd.run_problem_scout import (
    Tee, binding, canonical_hash, digest, metrics_from_confusion, path_label,
    require_finite,
)
from ncd.source_commit import source_commit

SEEDS = {"CIC-IDS-2017": [1, 2, 3], "NSL-KDD": [1, 2, 3, 4, 42]}
SOURCE_PATHS = ["ncd/run_benign_posthoc.py", "ncd/model.py",
                "ncd/prepare_problem_scout_data.py", "ncd/run_problem_scout.py",
                "ncd/source_commit.py", "ncd/data.py", "ncd/train_cil.py"]
POLICIES = {
    "scope": "Final checkpoint only; full prepared ID test split; no shift test or retraining.",
    "D4": "logit + log(original prepared training count / total training rows); tau=1; no search.",
    "D5_reference": "Exact Replay final checkpoint buffer, selected randomly within each class at its introduction; no old full training pool is added.",
    "D5_embedding": "Raw 128-dimensional final MLP penultimate ReLU activations, no normalization.",
    "D5_distance": "Arithmetic mean of the 10 smallest Euclidean distances in float64.",
    "D5_threshold": "Per-class 95th percentile of reference-to-reference mean 10-NN distances, diagonal self row excluded; linear quantile; duplicates remain legitimate other rows.",
    "D5_decision": "Original Normal predictions unchanged. Original attack prediction retained only when its mean 10-NN distance is strictly below that class threshold, else Normal. D4 and D5 are separate treatments.",
    "D5_leakage": "No test labels/features used in reference or threshold construction; no hyperparameter search.",
    "scientific_limit": "D5 deliberately suppresses attack predictions as a diagnostic; it is not a non-suppressive RQ3 mechanism.",
}


def write_json(path, value):
    require_finite(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    for attempt in range(40):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.05)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def array_hash(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def load_reference(root, dataset, method, seed):
    path = root / dataset.lower().replace("-", "_") / f"{method}_seed{seed}" / "result.json"
    value = read_json(path)
    identity = value["identity"]
    if not value.get("completed") or identity.get("effective_config", {}).get("smoke"):
        raise ValueError("Reference is not a completed scientific cell")
    if (identity["dataset"], identity["method"], identity["seed"]) != (dataset, method, seed):
        raise ValueError("Reference cell identity differs")
    if len(value["checkpoints"]) != len(value["tasks"]):
        raise ValueError("Reference cell is incomplete")
    return path, value


def find_checkpoint(result_path, result, roots):
    cp = result["checkpoints"][-1]
    relative = Path(cp["checkpoint_file"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid reference checkpoint path")
    wanted = cp["checkpoint_sha256"]
    candidates = [result_path.parent / relative]
    dataset_slug, cell_name = result_path.parent.parent.name, result_path.parent.name
    for root in roots:
        candidates.append(root / dataset_slug / cell_name / relative)
        if root.exists():
            candidates.extend(root.glob(f"**/{dataset_slug}/{cell_name}/{relative.as_posix()}"))
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        if digest(candidate) == wanted:
            return candidate
    raise FileNotFoundError(f"Missing matching final checkpoint: {dataset_slug}/{cell_name}/{relative.as_posix()} SHA256={wanted}")


def validate_saved_state(state, result, data):
    identity = state["identity"]
    expected = result["identity"]
    # Merged Section21 JSONs sanitize storage paths. Verify scientific fields,
    # actual payload hash, and the original checkpoint identity hash separately.
    for key in ("dataset", "method", "seed", "git_commit", "class_order", "tasks",
                "feature_names", "data_binding", "regularization", "source_hashes"):
        if identity[key] != expected[key]:
            raise ValueError(f"Checkpoint reference identity differs: {key}")
    cp = result["checkpoints"][-1]
    if canonical_hash(identity) != cp["identity_hash"]:
        raise ValueError("Actual checkpoint identity hash differs")
    if state["task_index"] != len(data.tasks) - 1 or state["seen_classes"] != list(range(len(data.class_order))):
        raise ValueError("Not the complete final checkpoint")
    if identity["class_order"] != data.class_order or identity["tasks"] != data.tasks:
        raise ValueError("Prepared data class/task order differs")


def make_support(model, state, data, seed):
    started = time.perf_counter()
    refs, records = {}, []
    normal = data.class_order.index("Normal")
    intro = {c: t for t, task in enumerate(data.tasks) for c in task}
    seen = set()
    for x, y in zip(state["buffer_x"], state["buffer_y"]):
        if len(y) == 0 or len(torch.unique(y)) != 1:
            raise ValueError("Invalid saved class memory")
        c = int(y[0])
        if c in seen:
            raise ValueError("Duplicate saved memory class")
        seen.add(c)
        training_rows = np.flatnonzero(data.y_train == c)
        count = min(200, len(training_rows))
        task_seed = seed * 10000 + intro[c]
        selected = np.sort(np.random.default_rng(task_seed * 10007 + c).choice(len(training_rows), count, replace=False))
        indices = training_rows[selected]
        if not np.array_equal(x.numpy(), data.x_train[indices]) or len(y) != count:
            raise ValueError("Saved memory does not match original training-only Replay selection")
        if c == normal:
            continue
        if count <= 10:
            raise ValueError("D5 requires at least 11 reference examples for leave-one-out k=10")
        with torch.no_grad():
            embedding = model.features(x).to(torch.float64)
            distances = torch.cdist(embedding, embedding, p=2, compute_mode="use_mm_for_euclid_dist")
            distances.fill_diagonal_(float("inf"))
            loo = distances.topk(10, largest=False).values.mean(dim=1).numpy()
        threshold = float(np.quantile(loo, 0.95, method="linear"))
        refs[c] = (embedding, threshold)
        records.append({"class_id": c, "class_name": data.class_order[c], "reference_count": count,
                        "reference_training_indices": indices.tolist(),
                        "reference_features_sha256": array_hash(x.numpy()),
                        "embedding_sha256": array_hash(embedding.numpy()),
                        "threshold": threshold, "leave_one_out_distances": loo.tolist()})
    if seen != set(range(len(data.class_order))):
        raise ValueError("Saved memory is missing a class")
    return refs, records, time.perf_counter() - started


def score(model, data, priors, refs, batch_size, smoke):
    n = len(data.class_order)
    normal = data.class_order.index("Normal")
    cms = {name: np.zeros((n, n), dtype=np.int64) for name in (["baseline", "D4", "D5"] if refs is not None else ["baseline", "D4"])}
    rejected = np.zeros((n, n), dtype=np.int64)
    counts = np.bincount(data.y_train, minlength=n)
    selected = (np.concatenate([np.flatnonzero(data.y_test == c)[:64] for c in range(n)])
                if smoke else np.arange(len(data.y_test)))
    timings = {"forward_seconds": 0.0, "D4_seconds": 0.0, "D5_seconds": 0.0}
    adjustment = torch.from_numpy(np.log(priors).astype(np.float32))
    model.eval()
    with torch.no_grad():
        for start in range(0, len(selected), batch_size):
            idx = selected[start:start + batch_size]
            x = torch.from_numpy(np.asarray(data.x_test[idx]))
            t = time.perf_counter()
            embedding = model.features(x)
            logits = model.head(embedding)
            if not torch.isfinite(logits).all() or not torch.isfinite(embedding).all():
                raise ValueError("Nonfinite inference")
            pred = logits.argmax(dim=1)
            timings["forward_seconds"] += time.perf_counter() - t
            t = time.perf_counter()
            adjusted = (logits + adjustment).argmax(dim=1)
            timings["D4_seconds"] += time.perf_counter() - t
            predictions = {"baseline": pred.numpy(), "D4": adjusted.numpy()}
            if refs is not None:
                t = time.perf_counter()
                gated = pred.clone()
                for c, (reference, threshold) in refs.items():
                    rows = torch.nonzero(pred == c).flatten()
                    if len(rows):
                        distances = torch.cdist(embedding[rows].to(torch.float64), reference,
                                                p=2, compute_mode="use_mm_for_euclid_dist")
                        keep = distances.topk(10, largest=False).values.mean(dim=1) < threshold
                        gated[rows[~keep]] = normal
                timings["D5_seconds"] += time.perf_counter() - t
                predictions["D5"] = gated.numpy()
                changed = pred.numpy() != predictions["D5"]
                rejected += np.bincount(data.y_test[idx][changed] * n + pred.numpy()[changed], minlength=n * n).reshape(n, n)
            for name, prediction in predictions.items():
                cms[name] += np.bincount(data.y_test[idx] * n + prediction, minlength=n * n).reshape(n, n)
    metrics = {}
    for name, cm in cms.items():
        value = metrics_from_confusion(cm, list(range(n)), data.class_order, data.tasks, counts)
        value["normal_recall"] = 1.0 - value["normal_fpr"]
        new, old = data.tasks[-1], [c for task in data.tasks[:-1] for c in task]
        value["new_macro_recall"] = float(np.mean([value["per_class"][c]["recall"] for c in new]))
        value["old_macro_recall"] = float(np.mean([value["per_class"][c]["recall"] for c in old]))
        metrics[name] = value
    return metrics, timings, rejected.tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay-root", type=Path, default=Path("ncd/results/problem_scout/section22/steps_2000_v1"))
    p.add_argument("--joint-root", type=Path, default=Path("ncd/results/problem_scout/section21/merged_v1"))
    p.add_argument("--joint-weights-root", type=Path, action="append", default=[Path("ncd/results/problem_scout/section21/formal_v1")])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--data-root")
    p.add_argument("--datasets", nargs="+", choices=list(SEEDS), default=list(SEEDS))
    p.add_argument("--methods", nargs="+", choices=["replay", "joint"], default=["replay", "joint"])
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.cpu_threads < 1 or args.batch_size < 1:
        p.error("positive CPU threads and batch size required")
    if not args.smoke and not os.environ.get("NCD_SOURCE_COMMIT"):
        p.error("Formal posthoc execution requires explicit NCD_SOURCE_COMMIT")
    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True)
    out = args.output_dir.resolve()
    roots = [args.replay_root.resolve(), args.joint_root.resolve()]
    if any(out == r or r in out.parents or out in r.parents for r in roots):
        p.error("Output cannot overlap original raw reference directories")
    out.mkdir(parents=True, exist_ok=True)
    config = {"datasets": args.datasets, "methods": args.methods,
              "seeds": {d: ([1] if args.smoke else (args.seeds or SEEDS[d])) for d in args.datasets},
              "cpu_threads": args.cpu_threads, "batch_size": args.batch_size, "smoke": args.smoke,
              "D4_tau": 1.0, "D5_k": 10, "D5_quantile": 0.95, "policies": POLICIES}
    for d, seeds in config["seeds"].items():
        if len(set(seeds)) != len(seeds) or any(s not in SEEDS[d] for s in seeds):
            p.error("Seeds must be an unrepeated subset of the Section25 fixed dataset seeds")
    provenance = {"git_commit": source_commit(ROOT), "config": config,
                  "source_hashes": {v: digest(ROOT / v) for v in SOURCE_PATHS},
                  "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": str(torch.__version__)},
                  "command": ["python", "-I", "ncd/run_benign_posthoc.py", *sys.argv[1:]],
                  "log": "run.log", "device": "cpu"}
    previous = out / "result.json"
    if previous.exists() and read_json(previous)["provenance"] != provenance:
        raise ValueError("Resume source/config/command changed; preserve existing output and use a new directory")
    rows, missing = [], []
    expected = sum(len(config["seeds"][d]) * len(args.methods) for d in args.datasets)
    with (out / "run.log").open("a", encoding="utf-8") as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        def snapshot(status):
            write_json(previous, {"schema_version": 1, "status": status, "smoke": args.smoke,
                                 "expected_cells": expected, "completed_cells": len(rows), "cells": rows,
                                 "missing": missing, "provenance": provenance,
                                 "updated_utc": datetime.now(timezone.utc).isoformat()})
        try:
            for dataset in args.datasets:
                first_path, first = load_reference(args.replay_root, dataset, "replay", config["seeds"][dataset][0])
                data_config = dict(first["identity"]["effective_config"])
                if args.data_root:
                    data_config["problem_scout_data_root"] = args.data_root
                t = time.perf_counter()
                data, xs, ys, metadata = prepare_setting(dataset, data_config)
                actual_binding = binding(data, xs, ys)
                if actual_binding != first["identity"]["data_binding"]:
                    raise ValueError("Prepared full dataset differs from reference")
                load_seconds = time.perf_counter() - t
                counts = np.bincount(data.y_train, minlength=len(data.class_order))
                priors = counts / counts.sum()
                for method in args.methods:
                    reference_root = args.replay_root if method == "replay" else args.joint_root
                    for seed in config["seeds"][dataset]:
                        start = time.perf_counter()
                        rp, reference = load_reference(reference_root, dataset, method, seed)
                        if reference["identity"]["data_binding"] != actual_binding:
                            raise ValueError("Reference methods do not use the same prepared data")
                        try:
                            payload = find_checkpoint(rp, reference, args.joint_weights_root if method == "joint" else [])
                        except FileNotFoundError as error:
                            missing.append({"dataset": dataset, "method": method, "seed": seed, "reason": str(error)})
                            print(f"CELL_MISSING {dataset} {method} seed={seed}: {error}", flush=True)
                            snapshot("partial")
                            continue
                        output = out / dataset.lower().replace("-", "_") / f"{method}_seed{seed}" / "result.json"
                        identity = {"provenance": provenance, "dataset": dataset, "method": method, "seed": seed,
                                    "reference_result_sha256": digest(rp),
                                    "checkpoint_sha256": reference["checkpoints"][-1]["checkpoint_sha256"],
                                    "data_binding": actual_binding}
                        if output.exists():
                            result = read_json(output)
                            if result["identity"] != identity or not result.get("completed"):
                                raise ValueError("Existing cell differs from requested frozen diagnostic")
                            print(f"CELL_SKIPPED {dataset} {method} seed={seed}", flush=True)
                        else:
                            print(f"CELL_START {dataset} {method} seed={seed}", flush=True)
                            state = torch.load(payload, map_location="cpu", weights_only=True)
                            validate_saved_state(state, reference, data)
                            model = MLP(data.x_train.shape[1], reference["identity"]["effective_config"]["hidden"], len(data.class_order))
                            model.load_state_dict(state["model_state"])
                            model.eval()
                            refs, supports, support_seconds = (None, [], 0.0)
                            if method == "replay":
                                refs, supports, support_seconds = make_support(model, state, data, seed)
                            metrics, timings, rejected = score(model, data, priors, refs, args.batch_size, args.smoke)
                            baseline_same = metrics["baseline"]["confusion_matrix"] == reference["checkpoints"][-1]["id"]["confusion_matrix"]
                            if not args.smoke and not baseline_same:
                                raise ValueError("CPU baseline inference does not reproduce the saved final confusion matrix exactly")
                            result = {"completed": True, "smoke": args.smoke, "identity": identity,
                                      "class_order": data.class_order, "tasks": data.tasks,
                                      "reference_result": path_label(rp), "checkpoint": path_label(payload),
                                      "original_training_counts": counts.tolist(), "original_training_priors": priors.tolist(),
                                      "baseline_confusion_exact_match": baseline_same, "metrics": metrics,
                                      "D5_support": supports, "D5_rejected_true_by_original_prediction": rejected,
                                      "timing": timings | {"support_seconds": support_seconds,
                                                           "dataset_load_and_verify_seconds_shared": load_seconds,
                                                           "cell_total_seconds": time.perf_counter() - start},
                                      "completed_utc": datetime.now(timezone.utc).isoformat()}
                            write_json(output, result)
                            print(f"CELL_DONE {dataset} {method} seed={seed} seconds={result['timing']['cell_total_seconds']:.3f}", flush=True)
                        rows.append({"dataset": dataset, "method": method, "seed": seed,
                                     "result": output.relative_to(out).as_posix(), "result_sha256": digest(output)})
                        snapshot("running")
            snapshot("complete" if len(rows) == expected else "partial")
            print(f"MATRIX_DONE completed={len(rows)} expected={expected} missing={len(missing)}", flush=True)
        except BaseException:
            snapshot("failed")
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
