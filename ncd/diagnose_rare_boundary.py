"""CPU-only task-boundary diagnostics of frozen Replay and Section25 D2.

All prepared test rows of attack classes with fewer than 2000 original training
rows are fixed diagnostic probes, never fitting/selection data. Existing saved
task-end checkpoints cannot resolve within-task acquisition/forgetting. Full
per-probe logits live only in compressed local NPZ files, never model features.
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

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ncd.model import MLP
from ncd.prepare_problem_scout_data import prepare_setting
from ncd.run_problem_scout import (
    Tee, binding, canonical_hash, metrics_from_confusion, path_label, require_finite,
)
from ncd.source_commit import source_commit

SEEDS = {"CIC-IDS-2017": [1, 2, 3], "NSL-KDD": [1, 2, 3, 4, 42]}
RARE_CLASSES = {
    "CIC-IDS-2017": ["WebAttack", "Botnet", "Infiltration"],
    "NSL-KDD": ["R2L", "U2R"],
}
SOURCE_PATHS = [
    "ncd/diagnose_rare_boundary.py", "ncd/model.py",
    "ncd/prepare_problem_scout_data.py", "ncd/run_problem_scout.py",
    "ncd/data.py", "ncd/source_commit.py",
]
POLICIES = {
    "scope": "Existing task-end checkpoints only; CPU inference; no training or parameter search.",
    "source_selection": "Replay and D2 paths/hashes are taken from the frozen Section25 analysis; no source substitution by score.",
    "probe_selection": "Every prepared ID test row in an attack class with original training count <2000, in ascending prepared-test index order. This diagnostic roster is not a mechanism hyperparameter.",
    "test_role": "Previously exposed test data are descriptive probes only, not tuning, model selection, independent confirmation or causal evidence.",
    "prediction": "argmax(logits[:, seen_classes]) mapped back through the saved ordered seen_classes, exactly as the source evaluator. No task ID input and no new gating.",
    "unseen": "Future true classes are excluded from original checkpoint evaluation. Their extra probe predictions are labelled unseen_probe_only; no family/binary recall or valid true-class margin is claimed before introduction.",
    "margin": "Raw logits, not probabilities: true-Normal; true-max(other seen); max(seen attacks)-Normal. True-class decision margins are null until the class is seen.",
    "granularity": "One saved state after each complete task. No initial/epoch/optimizer-step weights are available; no within-task flip time, cause or learning-then-forgetting claim is identifiable.",
    "transitions": "Only transitions whose true class was already seen at the previous checkpoint count as post-introduction changes; an introduction miss is not post-introduction forgetting.",
    "verification": "Reproduce every saved seen-class confusion matrix over its full original ID evaluation population; hash actual checkpoint payloads before CPU loading.",
    "local_trace": "Compressed NPZ contains only prepared-test row indices, labels, logits, masks, margins and predictions. LOCAL ONLY; no feature vectors or raw traffic rows are saved.",
    "json_samples": "All six Infiltration probe indices; first three ascending prepared-test indices for each other rare class, fixed independently of prediction outcomes.",
    "research_limit": "Bounded descriptive RQ2 diagnostic; not explanation drift, a new mechanism, or completion of RQ1/RQ3/overall research.",
}


def shared_open(path):
    """Do not impede Windows atomic replacement while reading status/results."""
    if os.name != "nt":
        return Path(path).open("rb")
    import ctypes
    import msvcrt
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                                  ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                                  ctypes.c_void_p]
    kernel.CreateFileW.restype = ctypes.c_void_p
    handle = kernel.CreateFileW(str(Path(path).resolve()), 0x80000000, 7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    return os.fdopen(msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY), "rb")


def read_json(path):
    with shared_open(path) as stream:
        return json.load(stream)


def digest(path):
    h = hashlib.sha256()
    with shared_open(path) as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest()


def replace_file(temp, path):
    for attempt in range(40):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.05)


def write_json(path, value):
    require_finite(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    replace_file(temp, path)


def resolve_repo_path(label):
    path = Path(label)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def one(rows, label):
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one frozen reference: {label}; got {len(rows)}")
    return rows[0]


def select_sources(analysis, dataset, seed):
    replay = one([r for r in analysis["references"]
                  if (r["dataset"], r["seed"], r["method"]) == (dataset, seed, "replay")],
                 f"{dataset}/Replay/{seed}")
    d2 = one([r for r in analysis["cells"]
              if (r["dataset"], r["seed"], r["treatment"]) == (dataset, seed, "D2")],
             f"{dataset}/D2/{seed}")
    return [("Replay", resolve_repo_path(replay["result"]), replay["sha256"]),
            ("D2", resolve_repo_path(d2["evidence"]["result"]), d2["evidence"]["result_sha256"])]


def inspect_source(method, path, wanted, dataset, seed):
    if digest(path) != wanted:
        raise ValueError(f"Frozen analysis/result SHA differs: {path_label(path)}")
    result = read_json(path)
    ident = result["identity"]
    if not result.get("completed") or ident["effective_config"].get("smoke"):
        raise ValueError("Reference is not a completed full-data scientific cell")
    if (ident["dataset"], ident["seed"], ident["method"]) != (dataset, seed, "replay"):
        raise ValueError("Reference dataset/seed/base method differs")
    if method == "D2" and (ident.get("treatment"), ident.get("d2_budget_mode")) != ("D2", "fixed_final"):
        raise ValueError("Expected the selected fixed_final D2")
    if method == "Replay" and ident.get("treatment") is not None:
        raise ValueError("Replay reference unexpectedly carries a correction")
    tasks, classes = result["tasks"], result["class_order"]
    if ident["tasks"] != tasks or ident["class_order"] != classes:
        raise ValueError("Identity and result class/task order differ")
    if len(tasks) != 4 or len(result["checkpoints"]) != len(tasks):
        raise ValueError("Expected all four saved task-end checkpoints")
    if sorted(c for task in tasks for c in task) != list(range(len(classes))):
        raise ValueError("Tasks do not partition class IDs exactly once")
    payloads = []
    for t, cp in enumerate(result["checkpoints"]):
        seen = [c for task in tasks[:t + 1] for c in task]
        if cp["task_index"] != t or cp["seen_classes"] != seen or cp["new_classes"] != tasks[t]:
            raise ValueError("Saved checkpoint task/seen order differs")
        rel = Path(cp["checkpoint_file"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe checkpoint-relative path")
        payload = path.parent / rel
        if digest(payload) != cp["checkpoint_sha256"]:
            raise ValueError(f"Actual payload SHA differs: {path_label(payload)}")
        if read_json(payload.with_suffix(".json")) != cp:
            raise ValueError("Sidecar differs from frozen result checkpoint record")
        payloads.append(payload)
    return result, payloads


def validate_state(state, result, cp, data):
    expected, actual = result["identity"], state["identity"]
    keys = ["dataset", "method", "seed", "git_commit", "class_order", "tasks",
            "data_binding", "source_hashes", "treatment", "d2_budget_mode"]
    for key in keys:
        if actual.get(key) != expected.get(key):
            raise ValueError(f"Actual checkpoint scientific identity differs: {key}")
    # Published HPC copies redact storage paths, not numerical configuration.
    for key in set(actual["effective_config"]) | set(expected["effective_config"]):
        if key != "problem_scout_data_root" and actual["effective_config"].get(key) != expected["effective_config"].get(key):
            raise ValueError(f"Checkpoint configuration differs: {key}")
    if canonical_hash(actual) != cp["identity_hash"]:
        raise ValueError("Actual checkpoint identity hash differs from recorded identity")
    for key in ("task_index", "seen_classes", "training_schedule"):
        if state.get(key) != cp.get(key):
            raise ValueError(f"Checkpoint payload/record differs: {key}")
    if actual["class_order"] != data.class_order or actual["tasks"] != data.tasks:
        raise ValueError("Prepared class/task order differs")
    if actual["source_hashes"]["ncd/model.py"] != digest(ROOT / "ncd/model.py"):
        raise ValueError("Current inference architecture source differs from saved source")


def infer_checkpoint(model, data, seen, probe_indices, batch_size):
    """Use original full-seen-test batching; unseen probes are a separate view."""
    n = len(data.class_order)
    selected = np.flatnonzero(np.isin(data.y_test, seen))
    cm = np.zeros((n, n), dtype=np.int64)
    logits = np.empty((len(probe_indices), n), dtype=np.float32)
    recorded = np.zeros(len(probe_indices), dtype=bool)
    locator = np.full(len(data.y_test), -1, dtype=np.int64)
    locator[probe_indices] = np.arange(len(probe_indices))
    seen_array = np.asarray(seen, dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(selected), batch_size):
            indices = selected[start:start + batch_size]
            scores = model(torch.from_numpy(np.asarray(data.x_test[indices])))
            if not torch.isfinite(scores).all():
                raise ValueError("Nonfinite CPU inference logits")
            pred = seen_array[scores[:, seen].argmax(dim=1).numpy()]
            cm += np.bincount(data.y_test[indices] * n + pred, minlength=n * n).reshape(n, n)
            positions = locator[indices]
            present = positions >= 0
            logits[positions[present]] = scores.numpy()[present]
            recorded[positions[present]] = True
        unseen_positions = np.flatnonzero(~recorded)
        for start in range(0, len(unseen_positions), batch_size):
            pos = unseen_positions[start:start + batch_size]
            scores = model(torch.from_numpy(np.asarray(data.x_test[probe_indices[pos]])))
            if not torch.isfinite(scores).all():
                raise ValueError("Nonfinite unseen-only diagnostic logits")
            logits[pos] = scores.numpy()
    pred = seen_array[np.argmax(logits[:, seen], axis=1)]
    return logits, pred, cm


def stats(values):
    values = np.asarray(values)
    return {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max())}


def probe_summary(logits, pred, labels, indices, seen, tasks, t, classes, train_counts):
    normal = classes.index("Normal")
    attacks = [c for c in seen if c != normal]
    margins = {"best_seen_attack_minus_normal": logits[:, attacks].max(axis=1) - logits[:, normal]}
    margins["true_minus_normal_raw"] = logits[np.arange(len(labels)), labels] - logits[:, normal]
    true_seen = np.isin(labels, seen)
    # The zero fill in this local array is invalid wherever true_seen is false.
    true_other = np.zeros(len(labels), dtype=np.float32)
    for c in np.unique(labels):
        if c in seen:
            pos = labels == c
            other = [s for s in seen if s != c]
            true_other[pos] = logits[pos, c] - logits[pos][:, other].max(axis=1)
    margins["true_minus_best_other_seen"] = true_other
    rows = []
    for c in np.unique(labels):
        positions = np.flatnonzero(labels == c)
        intro = next(i for i, task in enumerate(tasks) if int(c) in task)
        phase = "unseen_probe_only" if t < intro else "first_introduction" if t == intro else "post_introduction"
        valid = c in seen
        sample_pos = positions if classes[c] == "Infiltration" else positions[:3]
        samples = []
        for pos in sample_pos:
            samples.append({"prepared_test_index": int(indices[pos]), "true_class": int(c),
                           "prediction": int(pred[pos]), "raw_logits": logits[pos].tolist(),
                           "true_class_seen": bool(valid),
                           "true_minus_normal": float(margins["true_minus_normal_raw"][pos]) if valid else None,
                           "true_minus_best_other_seen": float(true_other[pos]) if valid else None,
                           "best_seen_attack_minus_normal": float(margins["best_seen_attack_minus_normal"][pos])})
        histogram = np.bincount(pred[positions], minlength=len(classes))
        rows.append({"class_id": int(c), "class_name": classes[c], "introduced_task_index": intro,
                     "phase": phase, "original_train_rows": int(train_counts[c]), "probe_rows": len(positions),
                     "prediction_counts": histogram.tolist(), "prediction_counts_include_unseen_only_diagnostics": not valid,
                     "family_recall": float(histogram[c] / len(positions)) if valid else None,
                     "binary_attack_recall": float(1 - histogram[normal] / len(positions)) if valid else None,
                     "true_minus_normal": stats(margins["true_minus_normal_raw"][positions]) if valid else None,
                     "true_minus_best_other_seen": stats(true_other[positions]) if valid else None,
                     "best_seen_attack_minus_normal": stats(margins["best_seen_attack_minus_normal"][positions]),
                     "fixed_samples": samples})
    return rows, margins, true_seen


def transitions(predictions, labels, tasks, classes):
    normal, n = classes.index("Normal"), len(classes)
    result = []
    for c in np.unique(labels):
        positions = labels == c
        intro = next(t for t, task in enumerate(tasks) if int(c) in task)
        first = predictions[intro, positions]
        updates = []
        for t in range(intro + 1, len(tasks)):
            before, after = predictions[t - 1, positions], predictions[t, positions]
            matrix = np.bincount(before * n + after, minlength=n * n).reshape(n, n)
            updates.append({"from_task_index": t - 1, "to_task_index": t,
                            "update_new_classes": tasks[t], "prediction_transition_matrix": matrix.tolist(),
                            "attack_to_normal": int(np.sum((before != normal) & (after == normal))),
                            "normal_to_attack": int(np.sum((before == normal) & (after != normal))),
                            "correct_family_to_normal": int(np.sum((before == c) & (after == normal))),
                            "other_attack_to_normal": int(np.sum((before != c) & (before != normal) & (after == normal))),
                            "correct_family_to_wrong": int(np.sum((before == c) & (after != c))),
                            "wrong_to_correct_family": int(np.sum((before != c) & (after == c)))})
        result.append({"class_id": int(c), "class_name": classes[c], "introduced_task_index": intro,
                       "introduction_correct_family": int(np.sum(first == c)),
                       "introduction_normal_misses": int(np.sum(first == normal)),
                       "post_introduction_updates_available": len(updates), "updates": updates,
                       "limit": "No subsequent task-end checkpoint; cannot identify later forgetting" if not updates else "Task-end differences only, not within-update flip times"})
    return result


def run_cell(method, reference_path, reference, payloads, data, probes, train_counts, identity, output):
    target = output / "result.json"
    trace_path = output / "probe_trace.local_only.npz"
    if target.exists():
        previous = read_json(target)
        if previous.get("identity") != identity or not previous.get("completed"):
            raise ValueError("Existing diagnostic source/config differs; preserve it and use a new output directory")
        if digest(trace_path) != previous["local_trace"]["sha256"]:
            raise ValueError("Existing local-only trace missing or changed")
        print(f"CELL_SKIPPED {identity['dataset']} {method} seed={identity['seed']}", flush=True)
        return previous
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(data.y_test[probes])
    all_logits, all_predictions, all_masks, all_margins, records = [], [], [], [], []
    normal = data.class_order.index("Normal")
    for cp, payload in zip(reference["checkpoints"], payloads):
        t = cp["task_index"]
        state = torch.load(payload, map_location="cpu", weights_only=True)
        validate_state(state, reference, cp, data)
        model = MLP(data.x_test.shape[1], reference["identity"]["effective_config"]["hidden"], len(data.class_order))
        model.load_state_dict(state["model_state"])
        config = reference["identity"]["effective_config"]
        step_start = time.perf_counter()
        logits, pred, cm = infer_checkpoint(model, data, cp["seen_classes"], probes, int(config["eval_batch_size"]))
        expected_cm = np.asarray(cp["id"]["confusion_matrix"], dtype=np.int64)
        if not np.array_equal(cm, expected_cm):
            raise ValueError(f"Full seen-test CM does not exactly reproduce saved task={t}: L1={int(np.abs(cm-expected_cm).sum())}")
        summaries, margins, true_seen = probe_summary(logits, pred, labels, probes, cp["seen_classes"],
                                                     data.tasks, t, data.class_order, train_counts)
        # Row counts for every introduced probe must also agree with original CM.
        for row in summaries:
            if row["phase"] != "unseen_probe_only" and row["prediction_counts"] != cm[row["class_id"]].tolist():
                raise ValueError("Probe predictions do not reproduce the full-test class CM row")
        metrics = metrics_from_confusion(cm, cp["seen_classes"], data.class_order, data.tasks, train_counts)
        seen_attack = [c for c in cp["seen_classes"] if c != normal]
        records.append({"task_index": t, "new_classes": cp["new_classes"], "seen_classes": cp["seen_classes"],
                        "checkpoint_file": path_label(payload), "checkpoint_sha256": cp["checkpoint_sha256"],
                        "payload_verified": True, "identity_hash": cp["identity_hash"],
                        "optimizer_steps_this_task": cp["training_schedule"]["optimizer_steps"],
                        "training_schedule": cp["training_schedule"],
                        "full_seen_confusion_exact_match": True, "full_seen_confusion_sha256": canonical_hash(cm.tolist()),
                        "evaluated_rows": int(cm.sum()), "excluded_unseen_rows": len(data.y_test) - int(cm.sum()),
                        "normal_false_positive_count": int(cm[normal].sum() - cm[normal, normal]),
                        "attack_to_normal_count": int(cm[seen_attack, normal].sum()),
                        "normal_fpr": metrics["normal_fpr"], "binary_attack_recall": metrics["attack_recall"],
                        "attack_family_macro_recall": metrics["attack_family_macro_recall"],
                        "macro_f1": metrics["macro_f1"], "balanced_accuracy": metrics["balanced_accuracy"],
                        "rare_classes": summaries, "inference_seconds": time.perf_counter() - step_start})
        all_logits.append(logits); all_predictions.append(pred); all_masks.append(true_seen); all_margins.append(margins)
        print(f"CHECKPOINT_DONE {identity['dataset']} {method} seed={identity['seed']} task={t} full_cm_exact=True", flush=True)
        del state, model
    predictions = np.stack(all_predictions)
    temp = trace_path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, prepared_test_indices=probes, true_labels=labels,
                        raw_logits=np.stack(all_logits), masked_predictions=predictions,
                        task_indices=np.arange(len(records), dtype=np.int64),
                        true_class_seen=np.stack(all_masks),
                        seen_classes_mask=np.asarray([[c in r["seen_classes"] for c in range(len(data.class_order))] for r in records]),
                        true_minus_normal_raw=np.stack([m["true_minus_normal_raw"] for m in all_margins]),
                        true_minus_best_other_seen=np.stack([m["true_minus_best_other_seen"] for m in all_margins]),
                        best_seen_attack_minus_normal=np.stack([m["best_seen_attack_minus_normal"] for m in all_margins]))
    replace_file(temp, trace_path)
    result = {"schema_version": 1, "completed": True, "smoke": identity["provenance"]["config"]["smoke"],
              "identity": identity, "reference_result": path_label(reference_path),
              "original_training_source_commit": reference["identity"]["git_commit"],
              "class_order": data.class_order, "tasks": data.tasks, "probe_rows": len(probes),
              "probe_indices_sha256": array_hash(probes), "probe_labels_sha256": array_hash(labels),
              "checkpoints": records, "post_introduction_transitions": transitions(predictions, labels, data.tasks, data.class_order),
              "local_trace": {"file": trace_path.name, "sha256": digest(trace_path), "bytes": trace_path.stat().st_size,
                              "publication": "LOCAL ONLY; do not publish this payload", "contains_features": False,
                              "true_margin_validity": "Use true_class_seen. true_minus_best_other_seen is zero-filled before introduction; raw true-minus-Normal is non-decision diagnostic before introduction."},
              "elapsed_seconds": time.perf_counter() - started, "completed_utc": datetime.now(timezone.utc).isoformat()}
    write_json(target, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=Path("ncd/results/problem_scout/section25/analysis_v1/result.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=list(SEEDS), default=list(SEEDS))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--data-root")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--smoke", action="store_true", help="Non-scientific real-data entry validation; defaults to seed1, still verifies complete saved test CMs.")
    args = parser.parse_args()
    if args.cpu_threads < 1 or len(set(args.datasets)) != len(args.datasets):
        parser.error("Positive CPU threads and unique datasets required")
    if not args.smoke and not os.environ.get("NCD_SOURCE_COMMIT"):
        parser.error("Formal diagnostics require explicit NCD_SOURCE_COMMIT; smoke records actual working-source hashes")
    seeds = {d: (args.seeds or ([1] if args.smoke else SEEDS[d])) for d in args.datasets}
    for d, values in seeds.items():
        if len(set(values)) != len(values) or any(s not in SEEDS[d] for s in values):
            parser.error("Seeds must be a unique subset of the fixed Section25 dataset seeds")
    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True)
    analysis_path = resolve_repo_path(args.analysis)
    analysis = read_json(analysis_path)
    if not analysis.get("complete") or analysis.get("missing"):
        raise ValueError("Source Section25 analysis is not complete")
    sources = {d: [(s, *spec) for s in seeds[d] for spec in select_sources(analysis, d, s)] for d in args.datasets}
    out = resolve_repo_path(args.output_dir)
    protected = [analysis_path.parent] + [spec[2].parent for rows in sources.values() for spec in rows]
    if any(out == p or p in out.parents or out in p.parents for p in protected):
        parser.error("Diagnostic output cannot overlap original result/analysis directories")
    config = {"datasets": args.datasets, "seeds": seeds, "cpu_threads": args.cpu_threads, "smoke": args.smoke,
              "rare_original_training_count_exclusive_upper_bound": 2000, "rare_classes": RARE_CLASSES,
              "batching": "Original eval_batch_size and original seen-test row order per checkpoint", "policies": POLICIES}
    provenance = {"git_commit": source_commit(ROOT), "source_hashes": {p: digest(ROOT / p) for p in SOURCE_PATHS},
                  "source_analysis": path_label(analysis_path), "source_analysis_sha256": digest(analysis_path),
                  "config": config, "command": ["python", "-I", "ncd/diagnose_rare_boundary.py", *sys.argv[1:]],
                  "device": "cpu", "log": "run.log", "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": str(torch.__version__)}}
    summary_path = out / "result.json"
    if summary_path.exists() and read_json(summary_path)["provenance"] != provenance:
        raise ValueError("Output source/config changed; preserve old diagnostic and choose a new directory")
    out.mkdir(parents=True, exist_ok=True)
    rows, dataset_records = [], []
    expected = sum(2 * len(v) for v in seeds.values())
    with (out / "run.log").open("a", encoding="utf-8") as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        def snapshot(status):
            write_json(summary_path, {"schema_version": 1, "status": status, "complete": status == "complete",
                                      "smoke": args.smoke, "provenance": provenance, "datasets": dataset_records,
                                      "expected_cells": expected, "completed_cells": len(rows), "cells": rows,
                                      "updated_utc": datetime.now(timezone.utc).isoformat()})
        try:
            snapshot("running")
            for dataset in args.datasets:
                first_seed, first_method, first_path, first_hash = sources[dataset][0]
                first, _ = inspect_source(first_method, first_path, first_hash, dataset, first_seed)
                data_config = dict(first["identity"]["effective_config"])
                if args.data_root:
                    data_config["problem_scout_data_root"] = args.data_root
                started = time.perf_counter()
                data, x_shift, y_shift, _ = prepare_setting(dataset, data_config)
                actual_binding = binding(data, x_shift, y_shift)
                train_counts = np.bincount(data.y_train, minlength=len(data.class_order))
                rare = [c for c, count in enumerate(train_counts) if data.class_order[c] != "Normal" and count < 2000]
                if [data.class_order[c] for c in rare] != RARE_CLASSES[dataset]:
                    raise ValueError("Fixed rare-class roster differs from original training counts")
                probes = np.flatnonzero(np.isin(data.y_test, rare)).astype(np.int64)
                dataset_records.append({"dataset": dataset, "class_order": data.class_order, "tasks": data.tasks,
                                        "data_binding": actual_binding, "probe_rows": len(probes), "probe_indices_sha256": array_hash(probes),
                                        "rare_classes": [{"class_name": data.class_order[c], "class_id": c, "train_rows": int(train_counts[c]),
                                                          "test_rows": int(np.sum(data.y_test == c)),
                                                          "introduced_task_index": next(t for t, task in enumerate(data.tasks) if c in task)} for c in rare],
                                        "dataset_load_and_verify_seconds_shared": time.perf_counter() - started})
                for seed, method, path, wanted in sources[dataset]:
                    reference, payloads = inspect_source(method, path, wanted, dataset, seed)
                    for key, expected_binding in reference["identity"]["data_binding"].items():
                        if actual_binding.get(key) != expected_binding:
                            raise ValueError(f"Actual data differ from saved {method} {dataset}: {key}")
                    identity = {"provenance": provenance, "dataset": dataset, "seed": seed, "method": method,
                                "reference_result_sha256": wanted, "checkpoint_sha256": [c["checkpoint_sha256"] for c in reference["checkpoints"]],
                                "probe_indices_sha256": array_hash(probes)}
                    cell_out = out / dataset.lower().replace("-", "_") / f"{method.lower()}_seed{seed}"
                    print(f"CELL_START {dataset} {method} seed={seed}", flush=True)
                    value = run_cell(method, path, reference, payloads, data, probes, train_counts, identity, cell_out)
                    rows.append({"dataset": dataset, "method": method, "seed": seed,
                                 "result": (cell_out / "result.json").relative_to(out).as_posix(),
                                 "result_sha256": digest(cell_out / "result.json"), "checkpoints": len(value["checkpoints"]),
                                 "full_seen_confusion_exact_matches": sum(c["full_seen_confusion_exact_match"] for c in value["checkpoints"]),
                                 "post_introduction_transitions": value["post_introduction_transitions"],
                                 "elapsed_seconds": value["elapsed_seconds"]})
                    snapshot("running")
                del data, x_shift, y_shift
            snapshot("complete")
            print(f"DIAGNOSTIC_COMPLETE cells={len(rows)}/{expected}; no training; CPU only", flush=True)
        except BaseException:
            snapshot("failed")
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
