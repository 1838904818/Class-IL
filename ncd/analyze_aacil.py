"""Section27.8/27.9 audit of saved, Normal-holdout-calibrated AA-CIL results.

No training, forward pass, threshold search, or test-FPR interpolation occurs
here. Only completed cells are admitted; absent five-seed groups stay pending.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("NSL-KDD", "UNSW-NB15", "CIC-IDS-2017")
SEEDS = (1, 2, 3, 4, 42)
ALPHAS = (.01, .02, .05)
NEW = ("aacil", "ts_mem", "aa_noattmem")
BASE = ("replay", "derpp", "wacb", "freq")
STRONGEST = (*BASE, "replay_la", "ts_mem", "aa_noattmem")
DISPLAY = (*BASE, "replay_la", *NEW, "farq", "joint")
METRICS = ("normal_fpr", "attack_detection", "macro_f1", "balanced_accuracy",
           "normal_recall", "attack_family_macro_recall", "attack_guard_recall")
MATCH_CONFIG = ("hidden", "learning_rate", "epochs_per_task", "min_steps_per_task",
                "batch_size", "buffer_per_class")
LABELS = {"aacil": "AA-CIL", "ts_mem": "TS-mem", "aa_noattmem": "AA-noAttMem",
          "replay": "Replay", "derpp": "DER++", "wacb": "WACB", "freq": "Freq",
          "replay_la": "Replay+LA", "farq": "FARQ (supplementary)", "joint": "Joint (offline)"}
CALIBRATION_PROTOCOL = {
    "section": "27.9", "targets": list(ALPHAS), "total_tau_range": [-1., 3.],
    "pool": "Same seed-specific permanent Normal TRAIN holdout for every method; never optimized",
    "rule": "First adjacent tau grid bracket; 50 bisections using only Normal holdout FP counts; retain feasible upper endpoint",
    "finite_sample_limit": "FP <= floor(alpha * calibration_count); no extrapolation outside grid FPR range",
    "test_role": "Evaluate selected total tau only; test metrics never select tau",
    "replay_la": "Same Replay weights and total tau; baseline tau=1 plus delta, no extra training or inference",
    "limitation": "The permanent pool was available to allocation validation; it is not a fresh independent calibration sample",
}
EXTENDED_CALIBRATION_PROTOCOL = {
    "section": "27.9", "version": 2, "targets": list(ALPHAS), "initial_tau_range": [-1., 3.],
    "tau_range_policy": "normal_holdout_only_bidirectional_bracket_expansion",
    "initial_expansion_step": 4., "expansion_multiplier": 2., "maximum_expansion_steps": 32,
    "bisection_iterations": 50,
    "pool": "Same seed-specific permanent Normal TRAIN holdout for every method; never optimized",
    "rule": "First adjacent initial-grid bracket when present; otherwise expand toward the missing infeasible lower or feasible upper endpoint; retain feasible upper endpoint after 50 bisections",
    "finite_sample_limit": "FP <= floor(alpha * calibration_count); strict Normal prior maximum and nonincreasing observed FP required",
    "test_role": "Evaluate selected total tau only; test metrics never select tau or the expansion range",
    "replay_la": "Same Replay weights and total tau; baseline tau=1 plus delta, no extra training or inference",
    "section26_10_unchanged": True,
    "limitation": "The permanent pool was available to allocation validation; it is not a fresh independent calibration sample",
}


def read(path):
    # Atomic writers may replace JSON while monitoring; never deny replacement.
    if os.name == "nt":
        import msvcrt
        create = ctypes.windll.kernel32.CreateFileW
        create.argtypes = (ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                           ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p)
        create.restype = ctypes.c_void_p
        handle = create(str(path.resolve()), 0x80000000, 7, None, 3, 128, None)
        if handle == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_last_error(), "Cannot read result", str(path))
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY)
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def label(path):
    p = path.resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.name


def stamp(value):
    d = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if d.tzinfo is None:
        raise ValueError("Completion timestamp has no timezone")
    return d


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return a is None and b is None or a is not None and b is not None and abs(a-b) <= 1e-12


def cm_metrics(value, names, seen=None):
    """Independent integer arithmetic; small attack classes remain in MF1/BAcc."""
    cm = np.asarray(value)
    n, normal = len(names), names.index("Normal")
    seen = list(range(n)) if seen is None else list(seen)
    require(cm.shape == (n, n) and np.issubdtype(cm.dtype, np.integer)
            and bool(np.all(cm >= 0)), "Invalid integer confusion matrix")
    support, predicted, tp = cm.sum(1), cm.sum(0), cm.diagonal()
    require(all(support[c] > 0 for c in seen), "A required test class is absent")
    unseen = sorted(set(range(n))-set(seen))
    require(not unseen or (cm[unseen].sum() == 0 and cm[:, unseen].sum() == 0),
            "Unseen classes entered evaluation")
    attack = [c for c in seen if c != normal]
    guard = [c for c in attack if support[c] >= 20]
    recall = np.divide(tp, support, out=np.zeros(n, float), where=support > 0)
    f1 = np.divide(2*tp, support+predicted, out=np.zeros(n, float), where=(support+predicted) > 0)
    return {"confusion_matrix": cm.tolist(), "support": support.tolist(),
            "normal_fpr": float((support[normal]-tp[normal])/support[normal]),
            "attack_detection": float(cm[np.ix_(attack, attack)].sum()/support[attack].sum()),
            "macro_f1": float(f1[seen].mean()), "balanced_accuracy": float(recall[seen].mean()),
            "normal_recall": float(recall[normal]),
            "attack_family_macro_recall": float(recall[attack].mean()),
            "attack_guard_recall": float(recall[guard].mean()) if guard else None,
            "guard_excluded_classes": [c for c in attack if support[c] < 20],
            "per_class": [{"class_id": c, "class_name": names[c], "support": int(support[c]),
                           "precision": float(tp[c]/predicted[c]) if predicted[c] else None,
                           "recall": float(recall[c]), "f1": float(f1[c])} for c in seen]}


def check_metrics(raw, calculated):
    aliases = {"attack_recall": "attack_detection",
               "attack_family_macro_recall_support_ge20": "attack_guard_recall"}
    for key in METRICS:
        if key in raw:
            require(close(raw[key], calculated[key]), f"Stored {key} differs from CM")
    for key, target in aliases.items():
        if key in raw:
            require(close(raw[key], calculated[target]), f"Stored {key} differs from CM")
    if "support" in raw:
        require(raw["support"] == calculated["support"], "Stored class supports differ from CM")
    for row in raw.get("per_class", []):
        expected = next(r for r in calculated["per_class"] if r["class_id"] == row["class_id"])
        require(row["support"] == expected["support"], "Per-class test support differs")
        for key in ("precision", "recall", "f1"):
            require(close(row[key], expected[key]), f"Per-class {key} differs from CM")


def stats(values):
    values = list(values)
    if not values or any(x is None for x in values):
        return {"n": len(values), "mean": None, "sample_sd": None,
                "reason": "An undefined value is retained, not replaced with zero"}
    return {"n": len(values), "mean": float(np.mean(values)),
            "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}


class Audit:
    def __init__(self, args):
        self.args, self.matches, self.data, self.sources = args, {}, {}, set()
        self.supports = {}
        self.payloads, self.sidecars = set(), set()
        self.torch = None
        if args.weights == "verify":
            import torch
            torch.set_num_threads(2)
            torch.set_num_interop_threads(1)
            self.torch = torch

    def identity(self, ident, new):
        cfg = ident["effective_config"]
        require(bool(ident.get("git_commit")) and canonical(cfg) == ident["effective_config_sha256"],
                "Missing source commit or changed effective config")
        require(bool(cfg.get("smoke")) == self.args.smoke, "Do not mix smoke and scientific cells")
        fixed = {"batch_size": 256, "buffer_per_class": 200,
                 "epochs_per_task": 1 if self.args.smoke else 8,
                 "min_steps_per_task": 0 if self.args.smoke else 2000}
        require(all(cfg.get(k) == v for k, v in fixed.items()), "Training schedule is not prescribed")
        if new:
            require(cfg.get("section27_schema_version") == 1
                    and ident.get("section27_access_policy") == "benign_refresh_shared_attack_memory_only"
                    and cfg.get("gate_step_policy") == "matched_s2000"
                    and cfg.get("calibration_policy") == "permanent_normal_holdout_outside_m",
                    "AA-CIL protocol or calibration policy differs")
            require(cfg.get("normal_training_rows") == 50000, "Extra benign-stream size differs")
        else:
            require(cfg.get("section26_schema_version") == 2
                    and ident.get("section26_access_policy") == "benign_refresh_attack_memory_only",
                    "Superseded Section26 policy is not a matched baseline")
        match = {k: ident[k] for k in ("data_binding", "data_metadata_sha256", "class_order",
                                      "tasks", "feature_names", "section26_holdout", "config_sha256")}
        match["training_config"] = {k: cfg[k] for k in MATCH_CONFIG}
        key = (ident["dataset"], ident["seed"])
        require(key not in self.matches or self.matches[key] == match,
                "Compared methods have different data, task order, holdout or training protocol")
        self.matches[key] = match
        if self.args.weights == "verify":
            for name, expected in ident["source_hashes"].items():
                if (name, expected) not in self.sources:
                    require(digest(ROOT/name) == expected, f"Bound source bytes differ: {name}")
                    self.sources.add((name, expected))
            require(digest(ROOT/"ncd/configs/problem_scout_section21.json") == ident["config_sha256"],
                    "Original configuration bytes differ")

    def dataset(self, ident):
        if self.args.weights != "verify":
            return None
        key = canonical([ident["data_binding"],ident["data_metadata_sha256"]])
        if key not in self.data:
            folder = self.args.data_root or Path(ident["effective_config"]["problem_scout_data_root"])
            folder = (folder if folder.is_absolute() else ROOT/folder)/ident["dataset"].lower()
            receipt = read(folder/"data_quality.json")
            require(canonical(receipt)==ident["data_metadata_sha256"] and receipt.get("status")=="complete"
                    and all(receipt[k]==ident[k] for k in ("class_order","tasks","feature_names")),
                    "Prepared data receipt/mapping differs")
            arrays = {}
            for name, filename in (("x_train", "train_x.npy"), ("y_train", "train_y.npy"),
                                   ("x_test", "test_x.npy"), ("y_test", "test_y.npy")):
                arrays[name] = np.load(folder/filename, mmap_mode="r")
            if self.args.smoke:
                for split in ("train", "test"):
                    y = arrays["y_"+split]
                    ids = np.concatenate([np.flatnonzero(y == c)[:64] for c in np.unique(y)])
                    for kind in ("x_", "y_"):
                        arrays[kind+split] = arrays[kind+split][ids]
            for name, a in arrays.items():
                bound = ident["data_binding"][name]
                require(list(a.shape) == bound["shape"] and a.dtype.str == bound["dtype"]
                        and hashlib.sha256(memoryview(a).cast("B")).hexdigest() == bound["sha256"],
                        f"Actual prepared {name} differs")
                for start in range(0,len(a),16384):
                    require(bool(np.isfinite(a[start:start+16384]).all()), "Nonfinite prepared array")
            for split in ("train","test"):
                y = arrays["y_"+split]
                require(y.dtype==np.int64 and len(y)==len(arrays["x_"+split])
                        and not np.any((y<0)|(y>=len(ident["class_order"]))), "Prepared labels/rows differ")
            self.data[key] = arrays
        return self.data[key]

    def pools(self, ident):
        data = self.dataset(ident)
        if data is None:
            return None
        train, valid = {}, {}
        for c in range(len(ident["class_order"])):
            rows = np.flatnonzero(data["y_train"] == c)
            count = max(5, math.ceil(.1*len(rows)))
            perm = np.random.default_rng(ident["seed"]*100003+c*1009+17).permutation(rows)
            train[c], valid[c] = np.sort(perm[count:]), perm[:count]
            expected = {"train_count": len(train[c]), "validation_count": len(valid[c]),
                        "validation_indices_sha256": hashlib.sha256(valid[c].tobytes()).hexdigest()}
            require(ident["section26_holdout"][str(c)] == expected, "Permanent holdout reconstruction differs")
        return train, valid, data

    def test_support(self, ident, measured, seen):
        key = (ident["dataset"],ident["seed"],tuple(seen))
        require(key not in self.supports or self.supports[key]==measured["support"], "Compared methods have different test supports")
        self.supports[key] = measured["support"]
        data = self.dataset(ident)
        if data is not None:
            expected = np.bincount(data["y_test"], minlength=len(ident["class_order"]))
            expected[[c for c in range(len(expected)) if c not in seen]] = 0
            require(measured["support"] == expected.tolist(), "CM did not evaluate the exact prescribed test population")

    def checkpoint(self, path, cp, ident):
        payload = (path.parent/cp["checkpoint_file"]).resolve()
        require(payload.is_relative_to(path.parent.resolve()), "Checkpoint path escapes cell")
        require(cp["identity_hash"] == canonical(ident), "Checkpoint identity hash differs")
        sidecar = payload.with_suffix(".json")
        require(read(sidecar) == cp, "Original checkpoint sidecar differs from result")
        self.sidecars.add(str(sidecar))
        if self.args.weights != "verify":
            return None
        require(digest(payload) == cp["checkpoint_sha256"], "Actual checkpoint SHA differs")
        state = self.torch.load(payload, map_location="cpu", weights_only=True)
        require(state["identity"] == ident and state["task_index"] == cp["task_index"]
                and state["seen_classes"] == cp["seen_classes"] and state["allocation"] == cp["allocation"],
                "Actual checkpoint identity/task/allocation differs")
        for key in ("gate_model_state", "family_model_state", "model_state"):
            if key in state:
                require(all(bool(self.torch.isfinite(v).all()) for v in state[key].values()), "Nonfinite saved model tensor")
        self.payloads.add(str(payload))
        return state


def schedule(record, rows, cfg):
    expected = max(cfg["min_steps_per_task"], cfg["epochs_per_task"]*math.ceil(rows/cfg["batch_size"]))
    require(record["rows_per_epoch"] == rows and record["target_steps"] == expected
            and record["optimizer_steps"] == expected, "Actual optimizer schedule differs")
    return expected


def check_storage(item, rows, features):
    expected = {"rows": rows, "feature_bytes": rows*features*4,
                "logical_label_bytes": rows*8, "index_bytes": rows*8,
                "logical_feature_label_index_bytes": rows*(features*4+16),
                "feature_and_index_array_bytes": rows*(features*4+8), "retained_dense_label_array_bytes": 0}
    require(all(item.get(k) == v for k, v in expected.items()), "Declared sample storage arithmetic differs")


def new_cell(path, result, audit):
    ident, checkpoints = result["identity"], result["checkpoints"]
    audit.identity(ident, True)
    pools = audit.pools(ident)
    cfg, names, method = ident["effective_config"], ident["class_order"], ident["method"]
    normal, features = names.index("Normal"), ident["data_binding"]["x_train"]["shape"][1]
    holdout = ident["section26_holdout"][str(normal)]
    require(len(checkpoints) == len(ident["tasks"]), "Completed AA cell has missing tasks")
    history, previous, final = [], None, {}
    costs = {k: 0. for k in ("gate_training_seconds", "family_training_seconds", "training_seconds",
                             "buffer_selection_seconds", "calibration_seconds", "evaluation_seconds")}
    gate_steps, family_steps = 0, 0
    for t, cp in enumerate(checkpoints):
        seen = [c for task in ident["tasks"][:t+1] for c in task]
        require(cp["task_index"] == t and cp["seen_classes"] == seen, "AA task order differs")
        state = audit.checkpoint(path, cp, ident)
        a = cp["allocation"]
        attacks = sorted(set(seen)-{normal})
        M = 200*len(seen)
        base, remainder = divmod(M, len(attacks))
        targets = {str(c): base+int(j < remainder) if method == "aacil" else 200 for j, c in enumerate(attacks)}
        require(a["attack_target_counts"] == targets and a["capacity_total"] == M
                and not a["old_attack_training_pool_access"] and not a["old_attack_validation_pool_access"]
                and a["attack_memory_shared_between_gate_and_family"] and a["attack_memory_counted_copies"] == 1,
                "AA memory allocation/access policy differs")
        ntotal = sum(a["actual_attack_training_counts"].values())+sum(a["actual_attack_validation_counts"].values())
        ntotal += a["normal_replay_training_count"]+a["normal_replay_validation_count"]
        require(a["actual_total"] == ntotal <= M and a["unfilled_capacity"] == M-ntotal, "Actual replay occupancy differs")
        require(a["normal_replay_target"] == (200 if method == "ts_mem" else 0)
                and a["normal_training_outside_replay_memory"] == (method != "ts_mem")
                and a["normal_refreshed"] and a["normal_refresh_rng_seed"] == (ident["seed"]*10000+t)*10007+normal,
                "Normal stream/memory policy differs")
        require(a["normal_gate_training_count"] == min(holdout["train_count"], 180 if method == "ts_mem" else 50000)
                and a["normal_replay_training_count"] == (a["normal_gate_training_count"] if method == "ts_mem" else 0)
                and a["normal_replay_validation_count"] == (min(20, holdout["validation_count"]) if method == "ts_mem" else 0)
                and a["normal_share_not_transferred_to_attacks"] == (method == "aa_noattmem"),
                "TS-mem or extra benign-row budget differs")
        check_storage(a["replay_memory_storage"], ntotal, features)
        check_storage(a["normal_gate_training_storage"], a["normal_gate_training_count"], features)
        check_storage(cp["calibration_pool_storage"], holdout["validation_count"], features)
        require(cp["calibration_pool_storage"]["outside_replay_M"]
                and cp["calibration_pool_storage"]["never_optimized"], "Calibration budget/training role differs")
        require(cp["calibration_additional_unique_rows_outside_M"] == holdout["validation_count"]-a["normal_replay_validation_count"]
                and cp["calibration_overlap_with_retained_normal_validation"] == a["normal_replay_validation_count"],
                "Calibration/replay overlap storage was double-counted")
        for c in attacks:
            key = str(c)
            ti, vi = a["training_row_ids"][key], a["validation_row_ids"][key]
            nv = max(5, math.ceil(targets[key]*.1))
            require(len(ti) == a["actual_attack_training_counts"][key]
                    and len(vi) == a["actual_attack_validation_counts"][key]
                    and len(ti) <= targets[key]-nv and len(vi) <= nv
                    and len(ti) == len(set(ti)) and len(vi) == len(set(vi))
                    and not set(ti).intersection(vi), "Attack memory count/role differs")
            if c not in ident["tasks"][t]:
                require(set(ti).issubset(previous["training_row_ids"][key])
                        and set(vi).issubset(previous["validation_row_ids"][key]), "Old attacks were replenished")
        gs, fs = cp["gate_training_schedule"], cp["family_training_schedule"]
        attack_rows = sum(cp["optimization_attack_counts"].values())
        gate_steps += schedule(gs, attack_rows+a["normal_gate_training_count"], cfg)
        family_steps += schedule(fs, attack_rows, cfg)
        require(close(cp["training_seconds"], cp["gate_training_seconds"]+cp["family_training_seconds"]),
                "G+F training time does not reconcile")
        if state is not None:
            train, valid, data = pools
            calibration_ids = state["calibration_row_ids"].numpy()
            require(np.array_equal(calibration_ids, valid[normal]), "Checkpoint calibration IDs differ")
            benign = state["normal_training_row_ids"].numpy()
            limit = 180 if method == "ts_mem" else 50000
            expected = np.sort(train[normal][np.random.default_rng(a["normal_refresh_rng_seed"]).permutation(len(train[normal]))[:limit]])
            require(np.array_equal(benign, expected) and not np.intersect1d(benign, calibration_ids).size
                    and hashlib.sha256(benign.tobytes()).hexdigest() == a["normal_training_row_ids_sha256"],
                    "Normal refresh or exclusion differs")
            normal_validation = state["normal_replay_validation_row_ids"].numpy()
            expected_validation = np.sort(valid[normal][np.random.default_rng(a["normal_refresh_rng_seed"]+53).permutation(len(valid[normal]))[:20]]) if method == "ts_mem" else np.empty(0, np.int64)
            require(np.array_equal(normal_validation, expected_validation)
                    and normal_validation.tolist() == a["normal_replay_validation_row_ids"], "TS-mem validation refresh differs")
            if method == "ts_mem":
                require(np.array_equal(state["normal_replay_train_x"].numpy(), data["x_train"][benign])
                        and np.array_equal(state["normal_replay_validation_x"].numpy(), data["x_train"][normal_validation]),
                        "TS-mem retained Normal payload differs")
            for c in attacks:
                key = str(c)
                mem = state["attack_memory"][key]
                for role, pool in (("train", train[c]), ("validation", valid[c])):
                    ids = mem[f"{role}_ids"].numpy()
                    saved_key = "training_row_ids" if role == "train" else "validation_row_ids"
                    require(ids.tolist() == a[saved_key][key] and np.all(np.isin(ids, pool))
                            and np.array_equal(mem[f"{role}_x"].numpy(), data["x_train"][ids]), "Stored attack payload/roles differ")
                used = state["optimization_attack_row_ids"][key].numpy()
                expected_used = train[c] if c in ident["tasks"][t] else np.asarray(previous["training_row_ids"][key])
                require(np.array_equal(used, expected_used) and len(used) == cp["optimization_attack_counts"][key],
                        "Attack optimization rows/counts differ")
        require(set(cp["operating_points"]) == set(cp["calibration"]) == {str(x) for x in ALPHAS}, "Missing alpha point")
        current = {}
        for alpha in ALPHAS:
            k, cal = str(alpha), cp["calibration"][str(alpha)]
            n, limit = holdout["validation_count"], int(Fraction(str(alpha))*holdout["validation_count"])
            require(cal["target_fpr"] == alpha and cal["calibration_rows"] == n
                    and cal["row_ids_sha256"] == holdout["validation_indices_sha256"]
                    and cal["allowed_false_alarms"] == limit and 0 <= cal["actual_false_alarms"] <= limit
                    and close(cal["actual_fpr"], cal["actual_false_alarms"]/n)
                    and cal["order_statistic_zero_based"] == n-limit-1 and math.isfinite(cal["threshold"]),
                    "G threshold calibration arithmetic/provenance differs")
            measured = cm_metrics(cp["operating_points"][k]["confusion_matrix"], names, seen)
            check_metrics(cp["operating_points"][k], measured)
            audit.test_support(ident, measured, seen)
            current[k] = {"status": "available", "calibration_fpr": cal["actual_fpr"],
                          "threshold": cal["threshold"], "metrics": measured}
        require(cp["id"] == cp["operating_points"]["0.02"], "Primary operating point differs")
        history.append({"task": t, "gate_steps": gs["optimizer_steps"], "family_steps": fs["optimizer_steps"],
                        "capacity": M, "actual_total": ntotal, "attack_targets": targets,
                        "actual_attack_training": a["actual_attack_training_counts"],
                        "actual_attack_validation": a["actual_attack_validation_counts"],
                        "normal_training_storage": a["normal_gate_training_storage"],
                        "replay_storage": a["replay_memory_storage"], "calibration_storage": cp["calibration_pool_storage"],
                        "checkpoint_sha256": cp["checkpoint_sha256"]})
        for key in costs:
            require(math.isfinite(cp[key]) and cp[key] >= 0, "Invalid recorded cost")
            costs[key] += cp[key]
        previous, final = a, current
    for key, value in costs.items():
        require(close(result[key+"_total"], value), "Cell cost sum differs")
    costs.update(gate_optimizer_steps=gate_steps, family_optimizer_steps=family_steps,
                 optimizer_steps=gate_steps+family_steps)
    return {"dataset": ident["dataset"], "method": method, "seed": ident["seed"],
            "completed_at": result["completed_at"], "source_result": label(path), "source_sha256": digest(path),
            "identity_sha256": canonical(ident), "points": final, "costs": costs, "trajectory": history,
            "physical_method": method, "training_source_commit": ident["git_commit"]}


def source_for_curve(curve, roots):
    b = curve["binding"]
    p = (ROOT/b["source_result"]).resolve()
    candidates = [p] + [root/curve["dataset"].lower().replace("-", "_")/f"{curve['method']}_seed{curve['seed']}"/"result.json" for root in roots]
    for candidate in candidates:
        if candidate.exists() and digest(candidate) == b["source_result_sha256"]:
            return candidate, read(candidate)
    raise ValueError("Curve's exact original completed source result is unavailable")


def original_curve_for_extension(curve, audit):
    """Bind the new calibration to its immutable, already-paid curve attempt."""
    b = curve["binding"]
    path = (ROOT/b["input_curve_result"]).resolve()
    require(path.exists() and digest(path) == b["input_curve_sha256"],
            "Extended calibration's original curve is unavailable or changed")
    original = read(path)
    require(curve.get("schema_version") == 2
            and curve.get("analysis_role") == "section27_9_extended_normal_holdout_calibration"
            and curve.get("excluded_from_section26_10_judgement") is True
            and original.get("completed") and original["verification"]["all_requested_checks_passed"]
            and canonical(original["binding"]) == b["input_curve_binding_sha256"],
            "Extended calibration's original curve binding differs")
    for key in ("source_result_sha256", "identity_sha256", "checkpoint_sha256", "checkpoint_file",
                "training_source_commit", "source_hashes", "config_sha256", "data_binding",
                "receipt_sha256", "original_train_counts", "prior", "log_prior_float32"):
        require(b[key] == original["binding"][key], "Extended calibration changed its trained source")
    require(curve["existing_curve_cost_seconds"] == original["cost_seconds"]
            and curve["existing_normal_holdout_calibration_cost_seconds"] ==
            original["normal_holdout_calibration"]["cost_seconds"], "Historical curve cost was changed")
    require(curve["normal_holdout_calibration"]["pool"] == original["normal_holdout_calibration"]["pool"]
            and curve["normal_holdout_calibration"]["grid"] == original["normal_holdout_calibration"]["grid"],
            "Extended calibration changed the original holdout or 41-point grid")
    if audit.args.weights == "verify":
        require(b["analysis_script_sha256"] == digest(ROOT/"ncd/calibrate_aacil_baselines.py")
                and b["utility_source_sha256"] == digest(ROOT/"ncd/analyze_farq_operating_curves.py")
                and b["utility_source_sha256"] == original["binding"]["analysis_script_sha256"],
                "Calibration utility source bytes differ")
    return original


def extended_query_trace(cal, count, prior, normal):
    queries = cal["search_evaluations"]
    lookup = {p["tau"]: p["false_positives"] for p in queries}
    require(len(lookup) == len(queries) and list(lookup) == sorted(lookup)
            and all(math.isfinite(t) and isinstance(v, int) and 0 <= v <= count for t, v in lookup.items())
            and all(a >= b for a, b in zip(lookup.values(), list(lookup.values())[1:])),
            "Extended calibration Normal-only query trace differs")
    require(all(p["normal_count"] == count and close(p["normal_fpr"], p["false_positives"]/count)
                and lookup[p["tau"]] == p["false_positives"] for p in [*queries, *cal["grid"]]),
            "Extended calibration query count arithmetic differs")
    gaps = np.asarray(prior, dtype=np.float32)[normal] - np.asarray(
        [v for c, v in enumerate(prior) if c != normal], dtype=np.float32)
    require(bool(np.all(gaps > 0)) and cal["monotonicity"]["normal_prior_strictly_largest"]
            and cal["monotonicity"]["observed_queries_nondecreasing_tau_nonincreasing_fp"]
            and close(float(gaps.min()), cal["monotonicity"]["minimum_normal_log_prior_gap"]),
            "Extended calibration monotonic-prior condition differs")
    return lookup


def extended_endpoint(op, grid, queries, alpha, count):
    """Replay the recorded Normal-only search arithmetic without model inference."""
    limit = int(Fraction(str(alpha))*count)
    feasible = [i for i, p in enumerate(grid) if p["false_positives"] <= limit]
    expansion, direction, initial = [], "none", None
    if feasible and feasible[0] > 0:
        j = feasible[0]
        lo, hi = grid[j-1]["tau"], grid[j]["tau"]
        initial = [lo, hi]
    else:
        step = 4.
        if feasible:
            direction, hi = "negative", -1.
            lo = hi
            for _ in range(32):
                lo -= step
                value = queries[lo]
                expansion.append({"tau": lo, "false_positives": value})
                if value > limit:
                    break
                hi, step = lo, step*2.
        else:
            direction, lo = "positive", 3.
            hi = lo
            for _ in range(32):
                hi += step
                value = queries[hi]
                expansion.append({"tau": hi, "false_positives": value})
                if value <= limit:
                    break
                lo, step = hi, step*2.
    require(queries[lo] > limit >= queries[hi]
            and op["starting_interval"] == [-1., 3.] and op["initial_grid_bracket"] == initial
            and op["expansion_direction"] == direction and op["expansion_steps"] == expansion
            and op["bracket_before_bisection"] == [lo, hi], "Extended calibration bracket search differs")
    for _ in range(50):
        mid = (lo+hi)/2.
        if queries[mid] <= limit:
            hi = mid
        else:
            lo = mid
    require(op["status"] == "available" and op["total_tau"] == hi
            and op["final_bisection_interval"] == [lo, hi] and op["bisection_iterations"] == 50
            and queries[lo] > limit >= queries[hi]
            and op["calibration_lower_false_positives"] == queries[lo]
            and op["calibration_false_positives"] == queries[hi] and op["calibration_count"] == count
            and close(op["calibration_fpr"], queries[hi]/count), "Extended calibration bisection differs")


def baseline_cell(path, curve, audit):
    require(curve["completed"] and curve["verification"]["all_requested_checks_passed"], "Curve is not verified complete")
    require("normal_holdout_calibration" in curve, "Test-FPR interpolation is not a calibrated baseline")
    extended = curve["normal_holdout_calibration"]["protocol"].get("version") == 2
    original_curve = original_curve_for_extension(curve, audit) if extended else curve
    source, result = source_for_curve(curve, audit.args.baseline_input)
    ident, b = result["identity"], curve["binding"]
    audit.identity(ident, False)
    pools = audit.pools(ident)
    require(result["completed"] and ident["dataset"] == curve["dataset"] and ident["method"] == curve["method"]
            and ident["seed"] == curve["seed"] and canonical(ident) == b["identity_sha256"]
            and b["source_completed_at"] == result["completed_at"]
            and ident["data_binding"] == b["data_binding"] and ident["git_commit"] == b["training_source_commit"]
            and ident["config_sha256"] == b["config_sha256"] and ident["source_hashes"] == b["source_hashes"]
            and result.get("section26_schema_version") == 2
            and result.get("section26_policy") == "benign_refresh_attack_memory_only"
            and b["parameters"]["calibrate_normal_holdout"], "Calibrated result source identity differs")
    normal, names = ident["class_order"].index("Normal"), ident["class_order"]
    counts = np.asarray([ident["section26_holdout"][str(c)]["train_count"]+ident["section26_holdout"][str(c)]["validation_count"] for c in range(len(names))])
    require(b["original_train_counts"]==counts.tolist() and result["train_counts"]=={str(c):int(v) for c,v in enumerate(counts)}
            and b["prior"]==(counts/counts.sum()).tolist()
            and b["log_prior_float32"]==np.log(counts/counts.sum()).astype(np.float32).tolist(),
            "Calibrated prior does not match original TRAIN frequencies")
    if curve["method"] == "joint":
        require(ident.get("offline_joint_reference") is True, "Joint must disclose offline access exception")
    require(len(result["checkpoints"]) == len(ident["tasks"]), "Baseline trajectory incomplete")
    costs = {"training_seconds": 0., "buffer_selection_seconds": 0., "evaluation_seconds": 0., "optimizer_steps": 0}
    history, previous = [], None
    for t, cp in enumerate(result["checkpoints"]):
        seen = [c for task in ident["tasks"][:t+1] for c in task]
        require(cp["task_index"] == t and cp["seen_classes"] == seen, "Baseline task order differs")
        state = audit.checkpoint(source, cp, ident)
        costs["optimizer_steps"] += schedule(cp["training_schedule"], cp["train_rows"], ident["effective_config"])
        measured = cm_metrics(cp["id"]["confusion_matrix"], names, seen)
        check_metrics(cp["id"], measured)
        audit.test_support(ident, measured, seen)
        a = cp["allocation"]
        require(a["capacity_total"] == 200*len(seen) and a["actual_total"] ==
                sum(a["actual_training_counts"].values())+sum(a["actual_validation_counts"].values())
                and a["actual_total"] <= a["capacity_total"] and not a["old_attack_training_pool_access"]
                and not a["old_attack_validation_pool_access"], "Baseline allocation arithmetic/access differs")
        require(curve["method"] != "joint" or a["reference_not_replay_budget"], "Offline Joint is not a replay-budget control")
        if curve["method"] == "joint":
            expected_counts = {str(c):ident["section26_holdout"][str(c)]["train_count"] for c in seen}
            require(cp["train_class_counts"]==expected_counts and cp["train_rows"]==sum(expected_counts.values()),
                    "Offline Joint must exclude the permanent holdout from optimization")
        for c in seen:
            key = str(c)
            ti, vi = a["training_row_ids"][key], a["validation_row_ids"][key]
            require(len(ti) == a["actual_training_counts"][key] and len(vi) == a["actual_validation_counts"][key]
                    and not set(ti).intersection(vi), "Baseline retained data roles differ")
            if c != normal and c not in ident["tasks"][t]:
                require(set(ti).issubset(previous["training_row_ids"][key])
                        and set(vi).issubset(previous["validation_row_ids"][key]), "Baseline replenished old attack memory")
            if state is not None:
                train, valid, _ = pools
                require(np.all(np.isin(ti, train[c])) and np.all(np.isin(vi, valid[c]))
                        and state["validation_row_ids"][key].tolist() == vi, "Baseline actual validation IDs differ")
        history.append({"task":t,"optimizer_steps":cp["training_schedule"]["optimizer_steps"],
                        "capacity":a["capacity_total"],"actual_total":a["actual_total"],
                        "target_counts":a["target_counts"],"actual_training_counts":a["actual_training_counts"],
                        "actual_validation_counts":a["actual_validation_counts"],
                        "offline_reference_not_replay_budget":curve["method"]=="joint",
                        "checkpoint_sha256":cp["checkpoint_sha256"]})
        previous = a
        for key in ("training_seconds", "buffer_selection_seconds", "evaluation_seconds"):
            require(math.isfinite(cp[key]) and cp[key] >= 0, "Invalid baseline cost")
            costs[key] += cp[key]
    require(close(result["training_seconds_total"],costs["training_seconds"]), "Baseline training-cost sum differs")
    final = result["checkpoints"][-1]
    require(b["checkpoint_sha256"] == final["checkpoint_sha256"] and b["checkpoint_file"] == final["checkpoint_file"],
            "Curve used a different final checkpoint")
    if extended:
        require(all(curve["verification"].get(k) is True for k in
                    ("normal_holdout_identity", "normal_prior_strict_max", "normal_only_selection",
                     "extended_calibration_verified", "tau0_exact_cm"))
                and curve["reproduction_checks"]["tau0"]["confusion_matrix"] == final["id"]["confusion_matrix"],
                "Extended calibration reproduction/selection checks are incomplete")
        if ident["method"] == "replay":
            require(curve["verification"].get("replay_tau1_exact_cm") is True
                    and curve["reproduction_checks"]["tau1"]["confusion_matrix"] == final["la_id"]["confusion_matrix"],
                    "Extended Replay tau1 reproduction differs")
    cal = curve["normal_holdout_calibration"]
    holdout = ident["section26_holdout"][str(normal)]
    pool, grid = cal["pool"], cal["grid"]
    count, features = holdout["validation_count"], ident["data_binding"]["x_train"]["shape"][1]
    expected_protocol = EXTENDED_CALIBRATION_PROTOCOL if extended else CALIBRATION_PROTOCOL
    require(cal["protocol"] == expected_protocol and b["calibration_protocol_sha256"] == canonical(expected_protocol)
            and pool["count"] == count
            and pool["indices_sha256"] == holdout["validation_indices_sha256"]
            and pool["outside_attack_replay_memory_budget"] and pool["feature_bytes"] == count*features*4
            and pool["label_bytes"] == pool["index_bytes"] == count*8
            and pool["feature_and_label_storage_bytes"] == count*(features*4+8), "Baseline calibration pool/storage differs")
    taus = np.linspace(-1., 3., 41).tolist()
    require([p["tau"] for p in grid] == taus, "Calibration tau grid differs")
    fp = [p["false_positives"] for p in grid]
    require(all(isinstance(x, int) and 0 <= x <= count for x in fp)
            and all(a >= b for a, b in zip(fp, fp[1:]))
            and all(p["normal_count"]==count and close(p["normal_fpr"],p["false_positives"]/count) for p in grid),
            "Calibration grid is invalid/nonmonotone")
    require(set(cal["operating_points"]) == {str(a) for a in ALPHAS}, "Calibration alpha grid differs")
    queries = extended_query_trace(cal, count, b["log_prior_float32"], normal) if extended else None
    points = {}
    for alpha in ALPHAS:
        k, op = str(alpha), cal["operating_points"][str(alpha)]
        limit = int(Fraction(str(alpha))*count)
        require(op["target_fpr"] == alpha and op["maximum_false_positives"] == limit, "Calibration target differs")
        if extended:
            extended_endpoint(op, grid, queries, alpha, count)
        elif not fp[-1]/count <= alpha <= fp[0]/count:
            require(op["status"] == "unavailable", "Calibration extrapolated outside fixed grid")
            points[k] = {"status": "unavailable", "reason": op["reason"]}
            continue
        else:
            j = next(i for i, value in enumerate(fp) if value <= limit)
            lo, hi = op["final_bisection_interval"]
            require(op["status"] == "available" and op["initial_grid_bracket"] == [taus[max(0,j-1)], taus[j]]
                    and taus[max(0,j-1)] <= lo <= hi == op["total_tau"] <= taus[j]
                    and op["bisection_iterations"] == (50 if j else 0)
                    and isinstance(op["calibration_false_positives"],int)
                    and 0 <= op["calibration_false_positives"] <= limit and op["calibration_count"] == count
                    and close(op["calibration_fpr"], op["calibration_false_positives"]/count), "Calibration selection arithmetic differs")
        measured = cm_metrics(op["test"]["confusion_matrix"], names)
        check_metrics(op["test"], measured)
        require(measured["support"] == cm_metrics(final["id"]["confusion_matrix"], names)["support"], "Calibrated test population differs")
        points[k] = {"status": "available", "calibration_fpr": op["calibration_fpr"],
                     "total_tau": op["total_tau"], "metrics": measured}
    if ident["method"] == "replay":
        alias = cal["replay_la_alias"]
        expected = {k:dict(v,delta_tau=v["total_tau"]-1.) if v["status"]=="available" else dict(v)
                    for k,v in cal["operating_points"].items()}
        require(alias["same_checkpoint_and_predictions"] and alias["initial_tau"]==1.
                and alias["extra_training_seconds"]==alias["extra_inference_seconds"]==0.
                and alias["operating_points"]==expected, "Replay+LA calibrated alias differs")
    costs["holdout_calibration_seconds"] = original_curve["normal_holdout_calibration"]["cost_seconds"]["calibration_total"]
    costs["curve_analysis_elapsed_seconds"] = original_curve["cost_seconds"]["cell_attempt_elapsed"]
    if extended:
        require(close(cal["cost_seconds"]["calibration_total"],
                      cal["cost_seconds"]["normal_holdout_logit_inference"] +
                      cal["cost_seconds"]["normal_only_search_seconds"]),
                "Extended Normal-only calibration cost sum differs")
        costs["extended_holdout_calibration_seconds"] = cal["cost_seconds"]["calibration_total"]
        costs["extended_calibration_analysis_elapsed_seconds"] = curve["cost_seconds"]["cell_attempt_elapsed"]
        for key, value in curve["cost_seconds"].items():
            require(math.isfinite(value) and value >= 0, "Invalid extended calibration cost")
            if key != "cell_attempt_elapsed":
                costs["extended_"+key] = value
    return {"dataset": ident["dataset"], "method": ident["method"], "seed": ident["seed"],
            "completed_at": result["completed_at"], "source_result": label(source), "source_sha256": digest(source),
            "curve_result": label(path), "curve_sha256": digest(path), "identity_sha256": canonical(ident),
            "points": points, "costs": costs, "calibration_storage": pool, "trajectory":history,
            "physical_method": ident["method"], "training_source_commit": ident["git_commit"],
            "calibration_protocol_version": 2 if extended else 1,
            "original_curve_result": b["input_curve_result"] if extended else label(path),
            "original_curve_sha256": digest((ROOT/b["input_curve_result"]).resolve()) if extended else digest(path),
            "offline_reference": ident["method"] == "joint",
            "cost_note": "Training counted once. Original curve and extended-calibration elapsed are separate physical attempts. Component timers are nested, except separately identified shared data verification; driver attempt ledgers retain source/data/resume overhead. Do not add component, cell and driver elapsed totals together."}


def analysis_attempt_ledgers(roots):
    """Retain shared verification/resume costs without duplicating cell timers."""
    allowed = ("started_at", "finished_at", "status", "elapsed_seconds", "elapsed_includes",
               "new_inference_cells_attempted", "new_inference_cells_completed", "reused_cells",
               "resume_validation_seconds")
    ledgers = []
    for root in sorted({p.resolve() for p in roots}):
        path = root/"summary.json"
        if not path.exists():
            continue
        raw = read(path)
        attempts = raw.get("execution", {}).get("attempts", [])
        for attempt in attempts:
            for key in ("elapsed_seconds", "resume_validation_seconds"):
                if key in attempt:
                    require(math.isfinite(attempt[key]) and attempt[key] >= 0,
                            "Invalid driver attempt cost")
        ledgers.append({"source_summary":label(path), "summary_content_sha256":canonical(raw),
                        "attempts":[{k:a[k] for k in allowed if k in a} for a in attempts],
                        "role":"Physical analysis attempt costs including source/data/resume work; overlaps cell timers, not additive"})
    return ledgers


def paired(a, b, metric, alternative):
    if a is None or b is None or len(a) != 5 or len(b) != 5:
        return {"status": "pending", "reason": "Requires two complete matched five-seed groups"}
    if any(a[s][metric] is None or b[s][metric] is None for s in SEEDS):
        return {"status": "pending", "reason": "Undefined endpoint retained"}
    delta = np.asarray([a[s][metric]-b[s][metric] for s in SEEDS], float)*100
    if not np.isfinite(delta).all():
        return {"status": "pending", "reason": "Undefined endpoint"}
    mean, sd = float(delta.mean()), float(delta.std(ddof=1))
    out = {"seeds": list(SEEDS), "delta_pp": delta.tolist(), "delta_pp_mean": mean,
           "delta_pp_sample_sd": sd, "alternative": alternative, "paired_effect_dz": mean/sd if sd else None}
    if sd == 0:
        out.update(status="not_estimable", t_statistic=None, p_one_sided=None,
                   reason="Zero paired variance: no finite t test; no significance claim")
    else:
        test = scipy_stats.ttest_1samp(delta, 0., alternative=alternative)
        out.update(status="available", t_statistic=float(test.statistic), p_one_sided=float(test.pvalue),
                   significant_at_0_05=bool(test.pvalue < .05))
    return out


def descriptive_pair(a, b, metric):
    if a is None or b is None or len(a)!=5 or len(b)!=5 or any(a[s][metric] is None or b[s][metric] is None for s in SEEDS):
        return {"status":"pending", "reason":"Two complete calibrated five-seed endpoints required"}
    values = [(a[s][metric]-b[s][metric])*100 for s in SEEDS]
    return {"status":"available", "seeds":list(SEEDS), "direction":"AA-CIL minus offline Joint",
            "delta_pp":values, "delta_pp_mean":float(np.mean(values)), "delta_pp_sample_sd":float(np.std(values,ddof=1))}


def summarize(cells, args):
    index = {(r["dataset"], r["method"], r["seed"]): r for r in cells}
    groups, complete = [], {}
    for dataset in args.datasets:
        for method in DISPLAY:
            rows = [index.get((dataset, method, seed)) for seed in SEEDS]
            points = {}
            for alpha in ALPHAS:
                key = str(alpha)
                selected = [r["points"].get(key) if r else None for r in rows]
                ok = all(p and p["status"] == "available" for p in selected) and not args.smoke
                if not ok:
                    points[key] = {"status": "pending", "available_seeds": [s for s,p in zip(SEEDS,selected) if p and p["status"] == "available"],
                                   "reason": "Five calibrated formal seeds required; no extrapolation or missing-seed substitution"}
                    continue
                metrics = {s:p["metrics"] for s,p in zip(SEEDS,selected)}
                complete[(dataset, method, key)] = metrics
                n = len(selected[0]["metrics"]["per_class"])
                points[key] = {"status": "complete", "metrics": {m: stats([p["metrics"][m] for p in selected]) for m in METRICS},
                               "calibration_fpr": stats(p["calibration_fpr"] for p in selected),
                               "per_class": [{"class_id": c, "class_name": selected[0]["metrics"]["per_class"][c]["class_name"],
                                              **{m:stats(p["metrics"]["per_class"][c][m] for p in selected) for m in ("precision","recall","f1")}}
                                             for c in range(n)]}
            present = [r for r in rows if r]
            keys = sorted(set().union(*(r["costs"] for r in present))) if present else []
            cost_summary = {k:stats(r["costs"][k] for r in present if k in r["costs"])
                            | {"sum":sum(r["costs"][k] for r in present if k in r["costs"])} for k in keys}
            storage = [{"seed":r["seed"],"calibration":r.get("calibration_storage", r["trajectory"][-1].get("calibration_storage")),
                        "tasks":[{k:v for k,v in t.items() if k in ("task","capacity","actual_total","normal_training_storage","replay_storage")}
                                 for t in r["trajectory"]]} for r in present]
            groups.append({"dataset": dataset, "method": method, "completed_cells": len(present), "points": points,
                           "costs":cost_summary, "storage":storage,
                           "cost_role":"alias only; no additional physical cost" if method=="replay_la" else "physical execution",
                           "cost_complete":len(present)==5 and not args.smoke})
    comparisons = {}
    for dataset in args.datasets:
        d = {}
        aa = complete.get((dataset,"aacil","0.02"))
        for metric, alternative in (("attack_detection","greater"),("macro_f1","less")):
            candidates = {m:complete.get((dataset,m,"0.02")) for m in STRONGEST}
            if any(v is None for v in candidates.values()):
                d[metric] = {"status":"pending", "missing_baselines":[m for m,v in candidates.items() if v is None]}
                continue
            means = {m:float(np.mean([v[s][metric] for s in SEEDS])) for m,v in candidates.items()}
            winner = max(STRONGEST,key=lambda m:means[m])
            d[metric] = paired(aa,candidates[winner],metric,alternative) | {"strongest_baseline":winner,
                "candidate_means":means, "tie_rule":"fixed listed baseline order; never per-seed winner selection"}
        d["attack_guard_recall"] = paired(aa,complete.get((dataset,"replay","0.02")),"attack_guard_recall","less")
        d["joint_descriptive"] = {m:descriptive_pair(aa,complete.get((dataset,"joint","0.02")),m) for m in METRICS}
        comparisons[dataset] = d
    def decision(items, at_least=None):
        """Resolve known outcomes without pretending missing endpoints were tested."""
        available = [x for x in items if x.get("status") == "available"]
        significant = sum(x["significant_at_0_05"] for x in available)
        unknown = len(DATASETS) - len(available)
        complete = unknown == 0 and not args.smoke
        if args.smoke:
            status = "pending"
        elif at_least is not None:
            status = ("pass" if significant >= at_least else
                      "fail" if significant + unknown < at_least else "pending")
        else:
            status = "fail" if significant else "pass" if complete else "pending"
        return {"status": status, "evidence_complete": complete,
                "available_dataset_tests": len(available), "unavailable_dataset_tests": unknown,
                "significant_dataset_tests": significant}
    primary = [comparisons[d]["attack_detection"] for d in args.datasets]
    mf1 = [comparisons[d]["macro_f1"] for d in args.datasets]
    guard = [comparisons[d]["attack_guard_recall"] for d in args.datasets]
    ablation = paired(complete.get(("CIC-IDS-2017","aacil","0.02")),complete.get(("CIC-IDS-2017","ts_mem","0.02")),"macro_f1","greater")
    criteria = [
        {"criterion":1,"description":"AA-CIL calibrated-alpha2% ADR higher than metric-wise strongest baseline on at least two datasets",
         **decision(primary, at_least=2)},
        {"criterion":2,"description":"No significant MF1 decline versus metric-wise strongest baseline on any dataset",
         **decision(mf1),
         "interpretation":"No detected significant decline under the work-order rule is NOT demonstrated noninferiority or equivalence"},
        {"criterion":3,"description":"No significant attack-family guard decline versus Replay",
         **decision(guard),
         "interpretation":"Support below20 is excluded only from this guard, not other metrics; nonsignificance is not noninferiority"},
        {"criterion":4,"description":"CIC AA-CIL MF1 significantly higher than TS-mem","test":ablation,
         "status":"pending" if ablation.get("status")!="available" or args.smoke else "pass" if ablation["significant_at_0_05"] else "fail",
         "evidence_complete":ablation.get("status")=="available" and not args.smoke},
        {"criterion":5,"description":"Calibrated Joint gaps and separate G/F/summed costs reported descriptively",
         "status":"reported" if len(args.datasets)==3 and all((d,m,"0.02") in complete for d in args.datasets for m in (*NEW,"joint")) else "pending"}]
    return groups, comparisons, criteria


def markdown(result):
    def pct(v):
        return "—" if v["mean"] is None else f"{v['mean']*100:.3f}±{v['sample_sd']*100:.3f}"
    heading = "# NONSCIENTIFIC SMOKE — Section27 schema audit" if result["smoke"] else "# Section27 calibrated operating-point analysis"
    lines = [heading, "",
             "NONSCIENTIFIC: smoke rows/checkpoints only; no efficacy or significance conclusion." if result["smoke"] else
             "A logically determined pass/fail is kept separate from evidence completeness; missing endpoints remain explicitly missing.", "",
             "α is selected on the common permanent Normal holdout; actual test FPR is reported, not forced to α. Values are five-seed mean±sample SD (%). Incomplete groups remain pending.", "",
             "| Dataset | Method | n/5 | Calibration FPR | Test FPR | ADR | Macro-F1 | Attack guard |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for g in result["groups"]:
        if g["method"]=="farq" and not g["completed_cells"]:
            continue
        p = g["points"]["0.02"]
        vals = ["pending"]*5 if p["status"]!="complete" else [pct(p["calibration_fpr"]),*[pct(p["metrics"][m]) for m in ("normal_fpr","attack_detection","macro_f1","attack_guard_recall")]]
        count = 5 if p["status"]=="complete" else len(p["available_seeds"])
        lines.append(f"| {g['dataset']} | {LABELS[g['method']]} | {count}/5 | " + " | ".join(vals) + " |")
    lines += ["", "| Section27.8 item | Status |", "|---|---|"]
    lines += [f"| {c['criterion']}. {c['description']} | {c['status']} |" for c in result["criteria"]]
    for g in result["groups"]:
        if g["method"] in NEW and g["completed_cells"]:
            c = g["costs"]
            lines += ["", f"{g['dataset']} {LABELS[g['method']]} ({g['completed_cells']}/5): recorded G/F/sum training "
                      f"{c['gate_training_seconds']['sum']:.3f}/{c['family_training_seconds']['sum']:.3f}/{c['training_seconds']['sum']:.3f} s; "
                      f"G/F steps {int(c['gate_optimizer_steps']['sum'])}/{int(c['family_optimizer_steps']['sum'])}. "
                      "Per-task actual memory, benign stream and separate calibration rows/bytes are in result.json; capacity is an upper limit, not actual occupancy."]
    lines += ["", "Strongest baselines use one fixed five-seed mean winner per dataset/metric among the seven Section27.4 controls; FARQ is supplementary only. Replay+LA shares Replay weights and calibrated total τ, so its physical cost is not charged twice.",
              "No detected significant harm is not proof of noninferiority/equivalence. These paired tests are unadjusted, use five seeds, and retain selection/multiplicity limitations. Joint is an offline access exception, not a guaranteed attainable upper bound. This RQ2 sub-study does not close RQ1, RQ3, or substantive innovation.",
              f"Payload mode: {result['verification']['weights_mode']}; actual payload hashes checked: {result['verification']['actual_payloads_verified']}. No new inference or threshold fitting was performed by this analyzer."]
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", default=[], help="Completed run_aacil matrix roots")
    parser.add_argument("--curves", type=Path, action="append", default=[], help="Roots containing Normal-holdout-calibrated curve cells")
    parser.add_argument("--calibrations", type=Path, action="append", default=[],
                        help="Separate Section27.9 extended-calibration roots; preferred only for the same exact trained source")
    parser.add_argument("--baseline-input", type=Path, action="append", default=[], help="Optional exact raw-source relocation roots")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", choices=DATASETS, nargs="+", default=list(DATASETS))
    parser.add_argument("--weights", choices=("verify","portable"), default="verify")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Non-scientific schema audit only; every scientific criterion stays pending")
    args = parser.parse_args()
    require(len(args.datasets)==len(set(args.datasets)), "Duplicate requested dataset")
    audit, cells, duplicates, missing = Audit(args), [], [], []
    expected_seeds = (1,) if args.smoke else SEEDS
    for dataset in args.datasets:
        slug = dataset.lower().replace("-","_")
        for method in (*NEW,*BASE,"farq","joint"):
            roots = args.input if method in NEW else [*args.curves, *args.calibrations]
            for seed in expected_seeds:
                candidates = []
                for root in dict.fromkeys(p.resolve() for p in roots):
                    path = root.resolve()/slug/f"{method}_seed{seed}"/"result.json"
                    if not path.exists():
                        continue
                    raw = read(path)
                    if not raw.get("completed"):
                        continue
                    row = new_cell(path,raw,audit) if method in NEW else baseline_cell(path,raw,audit)
                    require((row["dataset"],row["method"],row["seed"])==(dataset,method,seed), "Cell path/identity differs")
                    candidates.append(row)
                by_source = {}
                for row in candidates:
                    by_source.setdefault(row["source_sha256"], []).append(row)
                candidates = []
                for variants in by_source.values():
                    variants.sort(key=lambda r:(-r.get("calibration_protocol_version", 0),
                                                r.get("curve_sha256", r["source_sha256"])))
                    selected_variant = variants[0]
                    for other in variants[1:]:
                        if other.get("calibration_protocol_version") == selected_variant.get("calibration_protocol_version"):
                            require(other["points"] == selected_variant["points"],
                                    "Same-source same-protocol calibration outputs conflict")
                    selected_variant["alternative_calibration_inputs"] = [
                        {"path":v.get("curve_result",v["source_result"]),
                         "sha256":v.get("curve_sha256",v["source_sha256"]),
                         "protocol_version":v.get("calibration_protocol_version"), "costs":v["costs"],
                         "role":"same trained source; protocol-priority selection, not an additional training run"}
                        for v in variants[1:]]
                    candidates.append(selected_variant)
                candidates.sort(key=lambda r:(stamp(r["completed_at"]),r["source_sha256"]))
                if not candidates:
                    missing.append({"dataset":dataset,"method":method,"seed":seed,
                                    "role":"optional supplementary" if method=="farq" else "offline reference" if method=="joint" else "required main comparison"})
                    continue
                selected = candidates[0]
                cells.append(selected)
                duplicates.extend({"dataset":dataset,"method":method,"seed":seed,"source_result":r["source_result"],
                                   "source_sha256":r["source_sha256"],"costs":r["costs"],"role":"duplicate execution; excluded from efficacy"}
                                  for r in candidates[1:] if r["source_sha256"]!=selected["source_sha256"])
                if method=="replay":
                    cells.append(selected | {"method":"replay_la","alias_of":"replay",
                                             "extra_training_seconds":0.,"extra_inference_seconds":0.})
    groups, comparisons, criteria = summarize(cells,args)
    required = [c["status"] for c in criteria[:4]]
    overall = "fail" if "fail" in required else "pending" if "pending" in required or criteria[4]["status"]=="pending" else "pass"
    analysis_roots = [*args.curves, *args.calibrations,
                      *((ROOT/r["original_curve_result"]).resolve().parents[2]
                        for r in cells if "original_curve_result" in r)]
    result = {"schema_version":1,"generated_at":datetime.now(timezone.utc).isoformat(),
              "smoke":args.smoke,"primary_alpha":.02,"alphas":list(ALPHAS),"formal_seeds":list(SEEDS),
              "strongest_baseline_candidates":list(STRONGEST),"farq_role":"supplementary; excluded from primary strongest-baseline choice",
              "overall_registered_decision":overall,
              "evidence_completeness":{"all_scientific_criterion_tests_available":all(c["evidence_complete"] for c in criteria[:4]),
                                       "joint_and_cost_reporting_complete":criteria[4]["status"]=="reported",
                                       "decision_does_not_imply_all_evidence_complete":True},
              "counts":{"expected_new_physical_cells":len(args.datasets)*len(NEW)*len(expected_seeds),
                        "completed_new_physical_cells":sum(r["method"] in NEW for r in cells),
                        "completed_baseline_physical_cells":sum(r["method"] in BASE for r in cells),
                        "completed_joint_reference_cells":sum(r["method"]=="joint" for r in cells),
                        "completed_farq_supplementary_cells":sum(r["method"]=="farq" for r in cells),
                        "derived_replay_la_cells":sum(r["method"]=="replay_la" for r in cells)},
              "groups":groups,"comparisons":comparisons,"criteria":criteria,"cells":cells,"duplicates":duplicates,"missing":missing,
              "physical_analysis_attempt_ledgers":analysis_attempt_ledgers(analysis_roots),
              "verification":{"weights_mode":args.weights,"actual_payloads_verified":len(audit.payloads),
                              "sidecars_checked":len(audit.sidecars),"raw_source_files_checked":len(audit.sources),
                              "prepared_data_bindings_read":len(audit.data),
                              "payload_claim":"Actual SHA, CPU checkpoint identity and available saved data roles checked" if args.weights=="verify" else "Metadata/sidecar consistency only; payloads and datasets NOT read or verified",
                              "calibration_limit":"Recorded pool identities/count arithmetic checked; logits/threshold selection not re-inferred"},
              "scope_limits":["Normal calibration pool was used for allocation validation; not a fresh untouched validation sample.",
                              "α controls calibration FPR, not exact equality of test FPR.",
                              "One-sided paired t tests with n=5, unadjusted across endpoints/datasets; strongest baseline selected by complete group mean, not by seed.",
                              "Nonsignificance does not establish noninferiority or equivalence.",
                              "Two models and extra benign exposure are not compute-matched to one-stage methods; compare actual G/F/summed costs.",
                              "Stored sample-array bytes are not serialized PT file size; implicit labels and logical dense labels are distinguished.",
                              "Joint revisits old attacks offline; its gap is descriptive and it is not guaranteed to dominate.",
                              "No Section26.10 test-set-interpolated operating point enters this analysis.",
                              "Only bounded RQ2 evidence; RQ1/RQ3 and the locked substantive innovation requirement remain open."],
              "analysis_source_sha256":digest(Path(__file__))}
    output = args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=True)
    (output/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False),encoding="utf-8")
    (output/"summary.md").write_text(markdown(result),encoding="utf-8")
    print(json.dumps({"physical_selected_cells":sum(r["method"]!="replay_la" for r in cells),
                      "scientific_groups_complete":sum(g["points"]["0.02"]["status"]=="complete" for g in groups),
                      "actual_payloads_verified":len(audit.payloads),"criteria":criteria},allow_nan=False))


if __name__ == "__main__":
    main()
