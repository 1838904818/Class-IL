"""Section 19 S4/S5 confirmation, reusing the frozen Section 17 learners/PI.

Only cleaned, mapping-bound NetFlow input is accepted. No data preparation,
label mapping, deduplication, tuning, or historical-result writes happen here.
The same entry supports a one-seed, one-epoch, 64-row/class smoke run; smoke
completion is an implementation check and never a scientific confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ncd.analyze_exp_h import PRIMARY, analyze
from ncd.crr_benchmarks import METHODS
from ncd.run_crr_real_sequence import SEEDS, _write_csv
from ncd.run_exp_h import (
    SELF_MARGIN_DEFINITION, TRAINING_KEYS, _evaluate, _load_checkpoint, _save_json,
    _smoke_data, _train_checkpoints,
)
from ncd.source_commit import source_commit
from ncd.train_cil import Tee

SETTINGS = ("S4", "S5")
DEFAULT_OUTPUT = ROOT / "ncd/results/exp_h/section19/local_v1"
DEFAULT_CONFIG = ROOT / "ncd/configs/exp_h_section19.json"
LAMBDA_FILE = ROOT / "ncd/results/crr/section15/registered/nsl-real/lambda_selections.json"
COST_DEFINITIONS = {
    "training_seconds_total": (
        "Cumulative successful training time from the final checkpoint; restored task prefixes "
        "count once. Excludes buffer selection, evaluation, checkpoint I/O, and work lost before "
        "a checkpoint, so this is not total resource consumption across failed attempts."
    ),
    "elapsed_seconds_this_attempt": (
        "Cell wall time through training/checkpoint restoration and PI evaluation in the attempt "
        "that produced this completed cell. Not cumulative across retries; unchanged on cell reuse."
    ),
    "elapsed_seconds": "Legacy alias of elapsed_seconds_this_attempt, not cumulative training cost.",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode("utf-8")).hexdigest()


def path_label(path: Path) -> str:
    # Results and full logs are local. Prefer reproducible relative paths even
    # for sibling data directories; an absolute path on another drive is local.
    try:
        return Path(os.path.relpath(path.resolve(), ROOT)).as_posix()
    except ValueError:
        return str(path.resolve())


def require_finite(value, label="result") -> None:
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{label} contains a nonfinite tensor")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a nonfinite value")
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, f"{label}/{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite(item, f"{label}/{index}")


def array_binding(data, x_shift, y_shift) -> dict:
    output = {}
    for name, value in [("x_train", data.x_train), ("y_train", data.y_train),
                        ("x_test", data.x_test), ("y_test", data.y_test),
                        ("x_shift", x_shift), ("y_shift", y_shift)]:
        value = np.ascontiguousarray(value)
        output[name] = {"shape": list(value.shape), "dtype": value.dtype.str,
                        "sha256": hashlib.sha256(memoryview(value).cast("B")).hexdigest()}
    return output


def validate_data(data, x_shift, y_shift, metadata) -> tuple[set, set]:
    flat = [int(c) for task in data.tasks for c in task]
    if len(data.tasks) < 2 or len(data.tasks[0]) < 2 or flat != list(range(len(data.class_order))):
        raise ValueError("Frozen tasks must cover every class once in order, starting with >=2 classes")
    if not metadata.get("feature_alignment"):
        raise ValueError("Cleaned input must document its exact common feature schema")
    nfeatures = len(data.feature_names)
    for name, x, y in [("train", data.x_train, data.y_train),
                       ("id", data.x_test, data.y_test), ("shift", x_shift, y_shift)]:
        if x.ndim != 2 or x.shape[1] != nfeatures or y.ndim != 1 or len(x) != len(y) or not len(y):
            raise ValueError(f"{name} array shape is invalid")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"{name} contains nonfinite values")
        if y.dtype.kind not in "iu" or not set(np.unique(y)).issubset(flat):
            raise ValueError(f"{name} contains unmapped class IDs")
    if set(np.unique(data.y_train)) != set(flat) or set(np.unique(data.y_test)) != set(flat):
        raise ValueError("Every training class requires source train and ID probes")
    shifted = set(map(int, np.unique(y_shift)))
    units = {(after, c) for after in range(1, len(data.tasks))
             for task in data.tasks[:after] for c in task if c in shifted}
    if not units:
        raise ValueError("Frozen sequence has no observable old-class transitions")
    return units, shifted


def validate_cell(result: dict, identity: dict, output: Path, units: set, classes: set) -> None:
    if result.get("identity") != identity or result.get("git_commit") != identity["git_commit"]:
        raise ValueError(f"Saved cell source/config/data mismatch: {output}")
    rows, retention = result.get("rows", []), result.get("retention_rows", [])
    keys = [(r.get("transition"), r.get("class_id")) for r in rows]
    if len(keys) != len(units) or set(keys) != units:
        raise ValueError(f"Missing/duplicate transition rows: {output}")
    keep = [r.get("class_id") for r in retention]
    if len(keep) != len(classes) or set(keep) != classes:
        raise ValueError(f"Missing/duplicate retention rows: {output}")
    for row in rows + retention:
        if any(row.get(k) != identity[k] for k in ("setting", "method", "seed")):
            raise ValueError(f"Cell row identity mismatch: {output}")
    for row in rows:
        for key in ("old_pair_pi_drift", "self_margin_pi_drift", "shift_recall_before",
                    "shift_recall_after", "id_recall_before", "id_recall_after"):
            if not isinstance(row.get(key), (float, int)) or not math.isfinite(row[key]):
                raise ValueError(f"Missing/nonfinite {key}: {output}")
        if any(not 0 <= row[k] <= 1 for k in ("shift_recall_before", "shift_recall_after",
                                             "id_recall_before", "id_recall_after")):
            raise ValueError(f"Recall outside [0,1]: {output}")
        if not (output / row["pi_vectors_file"]).is_file():
            raise FileNotFoundError(f"Missing saved PI vectors: {output}")
    for name in {row["pi_vectors_file"] for row in rows}:
        with np.load(output / name, allow_pickle=False) as vectors:
            if not vectors.files or any(vectors[k].shape != (len(identity["feature_names"]),)
                                        or not np.isfinite(vectors[k]).all() for k in vectors.files):
                raise ValueError(f"Invalid/nonfinite PI vectors: {output / name}")
    for row in retention:
        if any(not isinstance(row.get(k), (float, int)) or not 0 <= row[k] <= 1
               for k in ("shift_recall_final", "shift_recall_best")):
            raise ValueError(f"Invalid retention recall: {output}")
    n = len(identity["tasks"])
    if any(len(result.get(k, [])) != n for k in ("checkpoints", "task_history", "shift_history")):
        raise ValueError(f"Incomplete task history/checkpoints: {output}")
    previous_training_seconds = 0.0
    for checkpoint in result["checkpoints"]:
        if not (output / checkpoint).is_file():
            raise FileNotFoundError(f"Missing checkpoint: {output / checkpoint}")
        saved = _load_checkpoint(output / checkpoint, identity)
        if not saved.get("model_state"):
            raise ValueError(f"Checkpoint has no model state: {output / checkpoint}")
        require_finite(saved, "checkpoint")
        training_seconds = saved.get("training_seconds")
        if type(training_seconds) not in (float, int) or training_seconds < previous_training_seconds:
            raise ValueError(f"Invalid/nonmonotonic cumulative checkpoint training time: {output / checkpoint}")
        previous_training_seconds = training_seconds
    for key in ("training_seconds_total", "elapsed_seconds_this_attempt", "elapsed_seconds"):
        if type(result.get(key)) not in (float, int) or not math.isfinite(result[key]) or result[key] < 0:
            raise ValueError(f"Missing/invalid cell cost field {key}: {output}")
    if result["training_seconds_total"] != previous_training_seconds:
        raise ValueError(f"Cell training cost differs from final checkpoint: {output}")
    if result["elapsed_seconds"] != result["elapsed_seconds_this_attempt"]:
        raise ValueError(f"Legacy elapsed time is not its labelled attempt-time alias: {output}")
    if result.get("cost_definitions") != COST_DEFINITIONS:
        raise ValueError(f"Cell cost definitions are missing or changed: {output}")
    require_finite(result)


def load_config(args) -> tuple[dict, dict, dict]:
    registered = json.loads((ROOT / "ncd/configs/exp_h_section17.json").read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key in (*TRAINING_KEYS, "floor_recall", "bootstrap_replicates", "bootstrap_seed"):
        if config.get(key) != registered[key]:
            raise ValueError(f"Section 19 changes frozen Section 17 parameter: {key}")
    if config.get("settings") != list(SETTINGS):
        raise ValueError("Confirmation requires both S4 and S5 in the registered order")
    root = args.cleaned_data_root or os.environ.get("NCD_CLEANED_DATA_ROOT") or config.get("cleaned_data_root")
    if not root:
        raise ValueError("Supply --cleaned-data-root or NCD_CLEANED_DATA_ROOT after data preparation")
    cleaned = (ROOT / root).resolve()
    config["cleaned_data_root"] = path_label(cleaned)
    mapping = config.get("frozen_mapping_sha256")
    if not isinstance(mapping, str) or re.fullmatch(r"[0-9a-f]{64}", mapping) is None:
        raise ValueError("The class/feature mapping must be frozen in config before any model run")
    preparation_path = cleaned / "preparation.json"
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    if preparation.get("frozen_mapping_sha256") != mapping:
        raise ValueError("Cleaned input mapping differs from the frozen configuration")
    if args.device:
        config["device"] = args.device
    if args.smoke:
        config.update(epochs_per_task=1, buffer_per_class=16, smoke_rows_per_class=64)
    config["exp_h_smoke"] = args.smoke
    selection = json.loads(LAMBDA_FILE.read_text(encoding="utf-8"))
    if selection.get("dataset") != "NSL-KDD" or set(selection.get("lambdas", {})) != set(METHODS) or selection.get("selection_seeds") != [5, 6]:
        raise ValueError("Frozen Section 15 NSL eight-method calibration is required")
    receipts = {"config_file_sha256": digest(args.config), "effective_config_sha256": canonical_hash(config),
                "preparation_sha256": digest(preparation_path), "lambda_file_sha256": digest(LAMBDA_FILE),
                "frozen_mapping_sha256": mapping}
    return config, selection, receipts


def run_cell(setting, method, seed, data, x_shift, y_shift, metadata, config,
             selection, output, provenance, bindings, units, classes, evaluate_only=False):
    identity = {"setting": setting, "method": method, "seed": seed,
                "git_commit": provenance["git_commit"], "effective_config": config,
                "regularization": float(selection["lambdas"][method]),
                "lambda_source_commit": selection["git_commit"], "smoke": config["exp_h_smoke"],
                "class_order": data.class_order, "tasks": data.tasks,
                "feature_names": data.feature_names, "data_binding": bindings,
                "source_hashes": provenance["source_hashes"], "receipts": provenance["receipts"]}
    path = output / "exp_h_result.json"
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        validate_cell(result, identity, output, units, classes)
        print(f"CELL_RESUME {setting} {method} seed={seed}", flush=True)
        return result
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    print(f"CELL_START {setting} {method} seed={seed}", flush=True)
    checkpoints, history = _train_checkpoints(data, config, identity, output, device, evaluate_only)
    training_seconds_total = _load_checkpoint(checkpoints[-1], identity)["training_seconds"]
    rows, retention, shifted = _evaluate(data, x_shift, y_shift, config, identity,
                                          checkpoints, history, output, device)
    elapsed_seconds_this_attempt = time.perf_counter() - started
    result = {"schema_version": 1, "identity": identity, **provenance,
              "effective_config": config, "lambda_selection": selection,
              "rows": rows, "retention_rows": retention, "task_history": history,
              "shift_history": shifted, "setting_metadata": metadata,
              "self_margin_definition": SELF_MARGIN_DEFINITION,
              "checkpoints": [str(p.relative_to(output)) for p in checkpoints],
              "training_seconds_total": training_seconds_total,
              "elapsed_seconds_this_attempt": elapsed_seconds_this_attempt,
              "elapsed_seconds": elapsed_seconds_this_attempt, "cost_definitions": COST_DEFINITIONS,
              "scope": "WORK_ORDER Section 19 confirmation; unchanged Section 17 learners and PI"}
    validate_cell(result, identity, output, units, classes)
    _write_csv(output / "exp_h_rows.csv", rows)
    _write_csv(output / "exp_h_retention_rows.csv", retention)
    _save_json(path, result)
    print(f"CELL_COMPLETE {setting} {method} seed={seed} units={len(rows)}", flush=True)
    return result


def _s1_condition(s1: dict | None) -> bool | None:
    """Verify the numeric S1 evidence rather than trusting its convenience flag."""
    if s1 is None:
        return None
    primary = s1.get("primary_adjusted", {})
    flag = s1.get("s1_adjusted_condition_met")
    if primary.get("status") != "complete":
        if flag is not None:
            raise ValueError("S1 condition flag is present without complete adjusted numeric evidence")
        return None
    rho, ci = primary.get("rho"), primary.get("ci95")
    if (type(rho) not in (float, int) or not math.isfinite(rho) or not -1 <= rho <= 1
            or not isinstance(ci, (list, tuple)) or len(ci) != 2
            or any(type(v) not in (float, int) or not math.isfinite(v) or not -1 <= v <= 1 for v in ci)
            or ci[0] > ci[1]):
        raise ValueError("S1 adjusted rho or 95% CI is missing, nonfinite, or invalid")
    computed = rho >= 0.2 and ci[0] > 0
    if type(flag) is not bool or flag is not computed:
        raise ValueError("S1 adjusted condition flag disagrees with its rho and 95% CI")
    return computed


def confirmation_verdict(statistics: dict, s1: dict | None, smoke=False) -> dict:
    base = {"criterion": "Section 19.3: controlled S1 plus at least one S4/S5 positive rho>=0.2, CI excludes zero"}
    if smoke:
        return {**base, "decision": "pending", "reason": "Smoke output is not scientific confirmation"}
    condition = _s1_condition(s1)
    passes = {}
    for name in SETTINGS:
        detail = statistics.get("settings", {}).get(name, {})
        primary = detail.get("unit_level", {}).get(PRIMARY, {})
        rho, ci = primary.get("rho"), primary.get("ci95")
        if detail.get("status") != "complete" or primary.get("status") != "complete" or rho is None or not ci or any(v is None for v in ci):
            return {**base, "decision": "pending", "reason": f"{name} is incomplete or its primary interval is unidentified"}
        passes[name] = rho >= 0.2 and ci[0] > 0
    base["setting_conditions"] = passes
    if not any(passes.values()):
        return {**base, "decision": "harm_not_confirmed", "reason": "Neither S4 nor S5 meets the registered condition"}
    if condition is None:
        return {**base, "decision": "pending", "reason": "Controlled S1 robustness result is missing or unidentified"}
    if condition is not True:
        return {**base, "decision": "does_not_meet_confirmation_gate", "reason": "A confirmation setting meets its condition but controlled S1 does not"}
    return {**base, "decision": "harm_confirmed", "reason": "Controlled S1 and at least one confirmation setting meet their registered conditions"}


def aggregate(payload: dict, input_path: Path, output: Path, s1_path: Path, smoke: bool) -> dict:
    statistics = analyze(payload, n_bootstrap=2000, random_seed=1705)
    # Section 17.6 is not the confirmation decision rule; do not publish it here.
    statistics.pop("verdict", None)
    statistics["work_order"] = "Section 19.3; frozen Section 17.5 estimator"
    s1 = json.loads(s1_path.read_text(encoding="utf-8")) if s1_path.is_file() else None
    statistics["verdict"] = confirmation_verdict(statistics, s1, smoke)
    statistics["input_sha256"] = digest(input_path)
    statistics["s1_robustness"] = {"path": path_label(s1_path), "sha256": digest(s1_path) if s1 else None,
                                   "s1_adjusted_condition_met": s1.get("s1_adjusted_condition_met") if s1 else None}
    statistics["execution_status"] = payload["execution_status"]
    require_finite(statistics)
    _save_json(output / "exp_h_analysis.json", statistics)
    print(f"ANALYSIS_SAVED report_status={statistics['report_status']} decision={statistics['verdict']['decision']}", flush=True)
    return statistics


def run(args) -> None:
    from ncd.exp_h_netflow import prepare_setting

    config, selection, receipts = load_config(args)
    torch.set_num_threads(2)
    paths = [Path(__file__), ROOT / "ncd/exp_h_netflow.py", ROOT / "ncd/run_exp_h.py",
             ROOT / "ncd/crr_benchmarks.py", ROOT / "ncd/run_crr_real_sequence.py",
             ROOT / "ncd/analyze_exp_h.py", ROOT / "ncd/model.py", ROOT / "ncd/data.py"]
    provenance = {"git_commit": source_commit(ROOT), "receipts": receipts,
                  "source_hashes": {path_label(p): digest(p) for p in paths},
                  "command": subprocess.list2cmdline(["python", "-I", "ncd/run_exp_h_confirmation.py", *sys.argv[1:]]),
                  "log_path": path_label(args.log_path)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = args.output_dir / "exp_h_result.json"
    if combined_path.exists():
        prior = json.loads(combined_path.read_text(encoding="utf-8"))
        for key in ("git_commit", "receipts", "source_hashes"):
            if prior.get(key) != provenance[key]:
                raise ValueError(f"Existing output has a different {key}; preserve it and use a new output directory")
    seeds = (1,) if args.smoke else SEEDS
    rows, retention, cells, metadata = [], [], [], {}

    def snapshot(status="in_progress"):
        result = {"schema_version": 1, **provenance, "effective_config": config,
                  "settings": list(SETTINGS), "methods": list(METHODS), "seeds": list(seeds),
                  "smoke": args.smoke, "lambda_selection": selection, "setting_metadata": metadata,
                  "rows": rows, "retention_rows": retention, "cells": cells,
                  "self_margin_definition": SELF_MARGIN_DEFINITION, "execution_status": status,
                  "training_seconds_total": sum(c["training_seconds_total"] for c in cells),
                  "training_cost_scope": "Sum of each completed cell's final checkpoint once; incomplete cells are excluded",
                  "cost_definitions": COST_DEFINITIONS,
                  "config": {"expected_settings": list(SETTINGS)}}
        _write_csv(args.output_dir / "exp_h_rows.csv", rows)
        _write_csv(args.output_dir / "exp_h_retention_rows.csv", retention)
        _save_json(combined_path, result)
        return result

    for setting in SETTINGS:
        print(f"SETTING_START {setting}", flush=True)
        data, x_shift, y_shift, detail = prepare_setting(setting, config)
        if args.smoke:
            data, x_shift, y_shift = _smoke_data(data, x_shift, y_shift)
        units, classes = validate_data(data, x_shift, y_shift, detail)
        bindings = array_binding(data, x_shift, y_shift)
        metadata[setting] = dict(detail, status="in_progress", smoke=args.smoke,
                                 expected_units_per_cell=len(units), array_binding=bindings)
        print(f"SETTING_DATA {setting} train={data.x_train.shape} id={data.x_test.shape} shift={x_shift.shape}", flush=True)
        for method in METHODS:
            for seed in seeds:
                output = args.output_dir / setting / "shards" / f"{method}_seed{seed}"
                result = run_cell(setting, method, seed, data, x_shift, y_shift, detail, config,
                                  selection, output, provenance, bindings, units, classes, args.evaluate_only)
                rows.extend(result["rows"])
                retention.extend(result["retention_rows"])
                cells.append({"setting": setting, "method": method, "seed": seed,
                              "result": str((output / "exp_h_result.json").relative_to(args.output_dir)),
                              "training_seconds_total": result["training_seconds_total"],
                              "elapsed_seconds_this_attempt": result["elapsed_seconds_this_attempt"],
                              "elapsed_seconds": result["elapsed_seconds"]})
                snapshot()
        metadata[setting]["status"] = "complete"
        aggregate(snapshot(), combined_path, args.output_dir, args.s1_robustness, args.smoke)
        print(f"SETTING_COMPLETE {setting}", flush=True)
        del data, x_shift, y_shift
    expected = {(s, m, seed) for s in SETTINGS for m in METHODS for seed in seeds}
    observed = [(c["setting"], c["method"], c["seed"]) for c in cells]
    if len(observed) != len(expected) or set(observed) != expected:
        raise RuntimeError("Refusing completion for a partial or duplicated matrix")
    payload = snapshot("smoke_complete" if args.smoke else "complete")
    aggregate(payload, combined_path, args.output_dir, args.s1_robustness, args.smoke)
    print(f"MATRIX_EXECUTION_COMPLETE smoke={args.smoke} cells={len(cells)} units={len(rows)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cleaned-data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--s1-robustness", type=Path, default=Path("ncd/results/exp_h/section19/s1_robustness/s1_robustness.json"))
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--smoke", action="store_true", default=os.environ.get("NCD_LOCAL_SMOKE") == "1")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    args.config = (ROOT / args.config).resolve()
    args.output_dir = (ROOT / (args.output_dir or (DEFAULT_OUTPUT.with_name("smoke_v1") if args.smoke else DEFAULT_OUTPUT))).resolve()
    for historical in (ROOT / "ncd/results/crr", ROOT / "ncd/results/exp_h/section17"):
        if args.output_dir.is_relative_to(historical.resolve()):
            parser.error("Historical CRR/Section17 output directories cannot be used")
    args.s1_robustness = (ROOT / args.s1_robustness).resolve()
    args.log_path = (ROOT / args.log_path if args.log_path else args.output_dir / "run.log").resolve()
    for historical in (ROOT / "ncd/results/crr", ROOT / "ncd/results/exp_h/section17"):
        if args.log_path.is_relative_to(historical.resolve()):
            parser.error("Historical CRR/Section17 logs cannot be used")
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    with args.log_path.open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                run(args)
            except Exception:
                traceback.print_exc()
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
