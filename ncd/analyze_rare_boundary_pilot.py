"""Independently summarize the complete, already-exposed 16-cell task3 pilot.

No training, inference, data preparation, arm selection, or significance test.
Verify mode checks actual local checkpoint payloads; portable mode does not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("NSL-KDD", "CIC-IDS-2017")
SEEDS = (1, 2)
ARMS = ("random", "mir", "mir_rare_ce", "mir_rare_margin")
COMMON_CHAINS = ("baseline_indices", "current_indices", "candidate_indices",
                 "class_counts", "actual_class_counts")
METRICS = ("macro_f1", "balanced_accuracy", "normal_fpr",
           "attack_family_macro_recall", "binary_attack_micro_recall",
           "binary_attack_macro_recall", "old_class_macro_recall", "new_class_macro_recall")


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def local(base, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(), "Expected relative source path")
    path = (base / relative).resolve()
    require(path.is_relative_to(base.resolve()), "Source escapes its declared root")
    return path


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def equal_number(actual, expected, label):
    if expected is None:
        require(actual is None, f"{label}: expected undefined/null")
    else:
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
                f"{label}: recorded value differs from independent calculation")


def matrix(value, n):
    require(isinstance(value, list) and len(value) == n
            and all(isinstance(row, list) and len(row) == n for row in value), "Invalid CM shape")
    require(all(integer(x) for row in value for x in row), "CM must contain nonnegative JSON integers")
    require(all(sum(row) > 0 for row in value), "Full test must contain every class")
    return value


def metrics(record, names, tasks, expected_support):
    """Rows=true class, columns=predicted class; no training metric helper used."""
    n, normal = len(names), names.index("Normal")
    cm = matrix(record["confusion_matrix"], n)
    support = [sum(row) for row in cm]
    require(support == expected_support, "CM support differs from frozen full ID test")
    require(record["confusion_labels"] == list(range(n)), "CM label order mismatch")
    require(record["evaluated_rows"] == record["total_available_rows"] == sum(support)
            and record["excluded_unseen_rows"] == 0, "Partial or excluded test rows")
    intro = {c: t for t, task in enumerate(tasks) for c in task}
    rows = []
    for c, name in enumerate(names):
        predicted, tp = sum(row[c] for row in cm), cm[c][c]
        rows.append({"class_id": c, "class_name": name, "introduced_task": intro[c],
                     "support": support[c], "predicted": predicted, "true_positive": tp,
                     "predicted_normal_count": cm[c][normal],
                     "other_attack_count": sum(cm[c][k] for k in range(n) if k != normal and k != c),
                     "precision": tp / predicted if predicted else None,
                     "recall": tp / support[c], "f1": 2 * tp / (support[c] + predicted),
                     "binary_attack_recall": (support[c] - cm[c][normal]) / support[c] if c != normal else None,
                     "small_support_caveat": f"Only {support[c]} test rows; one row changes recall by {100 / support[c]:.6g} pp."
                     if support[c] <= 67 else None})
    attack = [c for c in range(n) if c != normal]
    old, new = [c for task in tasks[:3] for c in task], tasks[3]
    out = {"macro_f1": statistics.mean(r["f1"] for r in rows),
           "balanced_accuracy": statistics.mean(r["recall"] for r in rows),
           "normal_fpr": 1 - rows[normal]["recall"],
           "attack_family_macro_recall": statistics.mean(rows[c]["recall"] for c in attack),
           "binary_attack_micro_recall": sum(support[c] - cm[c][normal] for c in attack) / sum(support[c] for c in attack),
           "binary_attack_macro_recall": statistics.mean(rows[c]["binary_attack_recall"] for c in attack),
           "old_class_macro_recall": statistics.mean(rows[c]["recall"] for c in old),
           "new_class_macro_recall": statistics.mean(rows[c]["recall"] for c in new)}
    for key in ("macro_f1", "balanced_accuracy", "normal_fpr", "attack_family_macro_recall"):
        equal_number(record[key], out[key], key)
    equal_number(record["attack_recall"], out["binary_attack_micro_recall"], "attack_recall")
    require(record["normal_support"] == support[normal]
            and record["attack_support"] == sum(support[c] for c in attack), "Binary support mismatch")
    recorded = record["per_class"]
    require(len(recorded) == n and [r["class_id"] for r in recorded] == list(range(n)), "Per-class IDs mismatch")
    for actual, calc in zip(recorded, rows):
        require(actual["class_name"] == calc["class_name"] and actual["support"] == calc["support"], "Per-class support/name mismatch")
        for key in ("precision", "recall", "f1"):
            equal_number(actual[key], calc[key], f"{calc['class_name']} {key}")
    return out, rows


def memory_signature(state):
    import numpy as np
    def array_hash(value):
        value = np.ascontiguousarray(value)
        h = hashlib.sha256()
        h.update(str(value.dtype).encode())
        h.update(np.asarray(value.shape, dtype="<i8").tobytes())
        h.update(value.tobytes())
        return h.hexdigest()
    x = np.concatenate([v.numpy() for v in state["buffer_x"]])
    y = np.concatenate([v.numpy() for v in state["buffer_y"]])
    require(np.isfinite(x).all(), "Nonfinite resident memory")
    return {"memory_x_sha256": array_hash(x), "memory_y_sha256": array_hash(y),
            "memory_rows": len(y), "memory_bytes": x.nbytes + y.nbytes}, y


def load_payload(path, expected_hash, identity):
    import torch
    require(sha(path) == expected_hash, "Actual checkpoint SHA mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    require(state["identity"] == identity, "Checkpoint identity mismatch")
    require(all(torch.isfinite(v).all().item() for v in state["model_state"].values()), "Nonfinite model weights")
    return state


def prefix(identity, verify):
    ref = identity["start_checkpoint"]
    rp = local(ROOT, ref["result_path"])
    require(sha(rp) == ref["result_sha256"], "Historical result SHA mismatch")
    prior = read(rp)
    pi, names, tasks = prior["identity"], identity["class_order"], identity["tasks"]
    require(prior["completed"] is True and len(prior["checkpoints"]) == 4, "Incomplete historical source")
    require((pi["dataset"], pi["seed"], pi["method"]) == (identity["dataset"], identity["seed"], "replay"), "Wrong prefix source")
    require(pi["class_order"] == names and pi["tasks"] == tasks
            and pi["data_binding"] == identity["data_binding"], "Historical data/task binding mismatch")
    require(all(pi["effective_config"].get(k) == v for k, v in {
        "batch_size": 256, "hidden": 128, "learning_rate": .001, "buffer_per_class": 200,
        "epochs_per_task": 8, "min_steps_per_task": 2000, "smoke": False}.items()),
        "Prefix is not historical S2000 Replay")
    require(canonical(pi) == ref["identity_hash"] and pi["git_commit"] == ref["source_commit"]
            and pi["source_hashes"] == ref["source_hashes"], "Prefix provenance mismatch")
    cp = prior["checkpoints"][2]
    path = local(ROOT, ref["path"])
    require(path == local(rp.parent, cp["checkpoint_file"]) and path.name == "checkpoint_002.pt"
            and cp["checkpoint_sha256"] == ref["sha256"] and cp["identity_hash"] == ref["identity_hash"], "Wrong prefix checkpoint")
    require(read(path.with_suffix(".json")) == cp, "Prefix sidecar mismatch")
    old = [c for task in tasks[:3] for c in task]
    require(ref["memory_rows"] == 200 * len(old), "Prefix resident capacity differs")
    if verify:
        state = load_payload(path, ref["sha256"], pi)
        require(state["task_index"] == 2 and state["seen_classes"] == old, "Prefix task mismatch")
        sig, labels = memory_signature(state)
        require(all(sig[k] == ref[k] for k in sig), "Prefix resident-memory hash mismatch")
        require(set(labels.tolist()) == set(old) and all(int((labels == c).sum()) == 200 for c in old), "Prefix class capacity mismatch")
    cm = matrix(prior["checkpoints"][3]["id"]["confusion_matrix"], len(names))
    support = [sum(row) for row in cm]
    require(sum(support) == identity["data_binding"]["y_test"]["shape"][0], "Full test binding size mismatch")
    return support


def audit_cell(result, verify, cell_path):
    ident, book, schedule = result["identity"], result["audit"], result["training_schedule"]
    require(result["completed"] is True and result["task_index"] == 3 and ident["smoke"] is False, "Incomplete/smoke cell")
    require(result["class_order"] == ident["class_order"] and result["tasks"] == ident["tasks"], "Cell class/task disagreement")
    require(result["evaluation_role"] == ident["evaluation_role"] == "already_exposed_ID_test_exploratory_only", "Wrong evaluation role")
    require(ident["steps"] == book["step"] == schedule["optimizer_steps"] == schedule["target_steps"] == 2000, "Step budget mismatch")
    require(result["memory_unchanged"] is True and result["old_pool_revisits"] == 0
            and result["memory_rows"] == ident["start_checkpoint"]["memory_rows"], "Memory/access boundary mismatch")
    require(result["current_original_row_indices_sha256"] == ident["current_original_row_indices_sha256"], "Current row hash mismatch")
    chains = book["chains"]
    require(all(isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v) for v in chains.values()), "Invalid schedule hash")
    require(chains["actual_class_counts"] == chains["class_counts"], "Changed class quotas")
    if ident["arm"] == "random":
        require(chains["actual_indices"] == chains["baseline_indices"], "Random arm changed baseline indices")
    exposures, costs = book["class_exposures"], book["costs"]
    require(len(exposures) == len(ident["class_order"]) and all(integer(v) for v in exposures), "Invalid class exposures")
    touches = book["current_rows"] + book["replay_rows"]
    require(sum(exposures) == touches == schedule["examples_presented"] == costs["real_forward_rows"] == costs["real_backward_rows"], "Real sample-touch mismatch")
    current = ident["tasks"][3]
    require(sum(exposures[c] for c in current) == book["current_rows"]
            and sum(exposures[c] for c in ident["rare_classes"]) == book["rare_rows"], "Current/rare exposures mismatch")
    require(schedule["rows_per_epoch"] == sum(ident["current_original_train_counts"].values()) + result["memory_rows"], "Allowed pool size mismatch")
    for key, value in costs.items():
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0, f"Invalid cost: {key}")
        if key.endswith("_rows") or key == "virtual_steps":
            require(integer(value), f"Noninteger sample-touch cost: {key}")
    if ident["arm"] == "random":
        require(all(costs[k] == 0 for k in ("virtual_steps", "virtual_forward_rows", "virtual_backward_rows", "candidate_forward_rows")), "Random has virtual/retrieval cost")
    else:
        require(costs["virtual_steps"] <= 2000 and costs["virtual_forward_rows"] == costs["virtual_backward_rows"]
                and costs["virtual_forward_rows"] <= book["current_rows"], "Invalid virtual-update counters")
    final = result["final_checkpoint"]
    require(final["path"] == "checkpoints/checkpoint_003.pt", "Unexpected final checkpoint location")
    path = local(cell_path.parent, final["path"])
    sidecar = path.with_suffix(".json")
    require(read(sidecar) == result, "Final sidecar differs from result")
    if verify:
        state = load_payload(path, final["sha256"], ident)
        require(state["task_index"] == 3 and state["seen_classes"] == list(range(len(ident["class_order"]))), "Final task/seen classes mismatch")
        require(state["book"] == book and state["training_schedule"] == schedule, "Final audit/schedule differs from payload")
        sig, _ = memory_signature(state)
        require(all(sig[k] == ident["start_checkpoint"][k] for k in sig), "Final resident memory changed")
    return sha(sidecar)


def stats(values):
    finite = [v for v in values if v is not None]
    return {"n": len(finite), "mean": statistics.mean(finite) if finite else None,
            "sample_sd": statistics.stdev(finite) if len(finite) > 1 else None}


def difference(a, b):
    return 100 * (a - b) if a is not None and b is not None else None


def analyze(pilot, verify):
    mp = pilot / "result.json"
    matrix_result = read(mp)
    mi = matrix_result["identity"]
    require(matrix_result["status"] == "complete" and matrix_result["completed_cells"] == matrix_result["expected_cells"] == 16
            and mi["smoke"] is False, "Require the complete nonsmoke 16-cell matrix")
    require(sorted(mi["datasets"]) == sorted(DATASETS) and sorted(mi["seeds"]) == list(SEEDS)
            and sorted(mi["arms"]) == sorted(ARMS), "Unexpected pilot grid")
    expected = {(d, s, a) for d in DATASETS for s in SEEDS for a in ARMS}
    actual = [(c["dataset"], c["seed"], c["arm"]) for c in matrix_result["cells"]]
    require(len(actual) == 16 and set(actual) == expected, "Duplicate/missing/unexpected cells")
    cells, common, supports, sources = {}, {}, {}, []
    for entry in matrix_result["cells"]:
        key = (entry["dataset"], entry["seed"], entry["arm"])
        rp = local(pilot, entry["result"])
        require(sha(rp) == entry["result_sha256"], "Cell result SHA differs from matrix")
        result = read(rp)
        ident = result["identity"]
        require((ident["dataset"], ident["seed"], ident["arm"]) == key, "Cell identity/grid mismatch")
        require(all(ident[k] == mi[k] for k in ("git_commit", "source_hashes", "config_sha256", "effective_config", "policies")), "Matrix/cell source or configuration mismatch")
        names, tasks = ident["class_order"], ident["tasks"]
        require(len(set(names)) == len(names) and names.count("Normal") == 1 and len(tasks) == 4
                and [c for task in tasks for c in task] == list(range(len(names))), "Invalid class/task ordering")
        config = ident["effective_config"]
        require(all(config.get(k) == v for k, v in {"batch_size": 256, "hidden": 128, "learning_rate": .001, "buffer_per_class": 200,
                    "epochs_per_task": 8, "min_steps_per_task": 2000, "smoke": False}.items())
                and ident["candidate_per_class"] == 100 and ident["extra_weight"] == 1.0, "Pilot fixed configuration changed")
        require(ident["rare_classes"] == [c for c in tasks[3] if names[c] != "Normal" and ident["current_original_train_counts"][str(c)] < 200], "Rare-class definition differs")
        require(set(ident["current_original_train_counts"]) == {str(c) for c in tasks[3]}
                and all(integer(v) and v > 0 for v in ident["current_original_train_counts"].values()), "Invalid current class counts")
        sidecar_hash = audit_cell(result, verify, rp)
        group = key[:2]
        invariant = {"identity": {k: v for k, v in ident.items() if k not in ("arm", "command", "log_path")},
                     "training_schedule": result["training_schedule"],
                     "chains": {k: result["audit"]["chains"][k] for k in COMMON_CHAINS},
                     "exposure": {k: result["audit"][k] for k in ("class_exposures", "current_rows", "replay_rows", "rare_rows", "rare_batches")}}
        if group not in common:
            common[group] = invariant
            supports[group] = prefix(ident, verify)
        else:
            require(invariant == common[group], "Cross-arm prefix/current/candidate/class schedule or exposure differs")
        calculated, per_class = metrics(result["id"], names, tasks, supports[group])
        for cls in per_class:
            cls["rare_current_class"] = cls["class_id"] in ident["rare_classes"]
        cost = dict(result["audit"]["costs"])
        cost.update(evaluation_seconds=result["evaluation_seconds"], steps=result["audit"]["step"],
                    memory_rows=result["memory_rows"], memory_bytes=ident["start_checkpoint"]["memory_bytes"],
                    current_rows=result["audit"]["current_rows"], replay_rows=result["audit"]["replay_rows"],
                    rare_rows=result["audit"]["rare_rows"], rare_batches=result["audit"]["rare_batches"],
                    peak_cuda_allocated_bytes=result["audit"]["peak_cuda_allocated_bytes"],
                    peak_cuda_reserved_bytes=result["audit"]["peak_cuda_reserved_bytes"],
                    evaluation_forward_rows=sum(supports[group]))
        cost["recorded_component_seconds"] = sum(v for k, v in cost.items() if k.endswith("_seconds"))
        cost["total_forward_sample_touches"] = sum(cost[k] for k in ("real_forward_rows", "virtual_forward_rows", "candidate_forward_rows", "trace_forward_rows", "evaluation_forward_rows"))
        cost["total_backward_sample_touches"] = cost["real_backward_rows"] + cost["virtual_backward_rows"]
        cells[key] = {"dataset": key[0], "seed": key[1], "arm": key[2], "metrics": calculated, "per_class": per_class,
                      "cost": cost, "class_exposures": result["audit"]["class_exposures"],
                      "loss_sums_and_logit_gradient_proxies": result["audit"]["loss_sums"],
                      "started_utc": result["audit"]["started_utc"], "finished_utc": result["finished_utc"]}
        sources.append({"dataset": key[0], "seed": key[1], "arm": key[2], "result": entry["result"],
                        "result_sha256": entry["result_sha256"], "sidecar_sha256": sidecar_hash,
                        "final_checkpoint_sha256": result["final_checkpoint"]["sha256"],
                        "confusion_matrix_sha256": canonical(result["id"]["confusion_matrix"]),
                        "audit_sha256": canonical(result["audit"]), "identity_sha256": canonical(ident)})
    summaries, paired = [], []
    for dataset in DATASETS:
        require(supports[(dataset, 1)] == supports[(dataset, 2)], "Seed test populations differ")
        for arm in ARMS:
            rows = [cells[(dataset, seed, arm)] for seed in SEEDS]
            reference = [cells[(dataset, seed, "random")] for seed in SEEDS]
            deltas = [{k: difference(row["metrics"][k], base["metrics"][k]) for k in METRICS} for row, base in zip(rows, reference)]
            per_class, paired_classes = [], []
            for c, cls in enumerate(rows[0]["per_class"]):
                per_class.append({"class_id": c, "class_name": cls["class_name"], "support_per_seed": cls["support"],
                                  "small_support_caveat": cls["small_support_caveat"],
                                  "rare_current_class": cls["rare_current_class"],
                                  "predicted_normal_count": stats([row["per_class"][c]["predicted_normal_count"] for row in rows]),
                                  "other_attack_count": stats([row["per_class"][c]["other_attack_count"] for row in rows]),
                                  **{k: stats([row["per_class"][c][k] for row in rows]) for k in ("precision", "recall", "f1", "binary_attack_recall")}})
                pd = [{k: difference(row["per_class"][c][k], base["per_class"][c][k]) for k in ("precision", "recall", "f1", "binary_attack_recall")}
                      for row, base in zip(rows, reference)]
                paired_classes.append({"class_id": c, "class_name": cls["class_name"], "per_seed_pp": [{"seed": s, **d} for s, d in zip(SEEDS, pd)],
                                       **{k: stats([d[k] for d in pd]) for k in pd[0]}})
            summaries.append({"dataset": dataset, "arm": arm, "seeds": list(SEEDS),
                              "metrics": {k: stats([row["metrics"][k] for row in rows]) for k in METRICS},
                              "per_class": per_class, "cost": {k: stats([row["cost"][k] for row in rows]) for k in rows[0]["cost"]}})
            paired.append({"dataset": dataset, "arm": arm, "reference": "random", "unit": "percentage_points",
                           "per_seed": [{"seed": s, **d} for s, d in zip(SEEDS, deltas)],
                           "metrics": {k: stats([d[k] for d in deltas]) for k in METRICS}, "per_class": paired_classes})
    return {"schema_version": 1, "completed": True, "created_utc": datetime.now(timezone.utc).isoformat(),
            "analyzer_sha256": sha(Path(__file__)), "matrix_result_sha256": sha(mp),
            "training_git_commit": mi["git_commit"], "training_source_hashes": mi["source_hashes"],
            "training_config_sha256": mi["config_sha256"], "effective_config_sha256": canonical(mi["effective_config"]),
            "weights_mode": "verify" if verify else "portable",
            "payload_verification": "Actual 16 final and 4 common-prefix local PT hashes, identities, final audits/schedules and resident memory checked. No inference rerun."
            if verify else "JSON hashes, sidecars, metadata and invariants checked only; actual PT bytes/identities/memory are NOT verified.",
            "scope": "All 16 cells, two already-exposed seeds per dataset, all four arms; no test-driven selection, formal confirmation, causal mechanism, novelty, RQ1/RQ3 or overall closure claim.",
            "conclusion_boundary": "All four prespecified arms are reported, not a post-hoc test-selected mechanism proof. This small exposed-test pilot cannot establish substantive innovation or an independent confirmatory effect.",
            "checks": {"complete_exact_grid": True, "common_prefix_and_resident_memory_metadata": True,
                       "historical_base_parameters_and_data_binding": True, "full_test_integer_cm_and_frozen_support": True,
                       "current_candidate_class_quota_schedule_hashes": True, "same_real_steps_and_class_exposures": True,
                       "actual_checkpoint_payloads": verify},
            "definitions": {"rates": "Fractions; summaries use per-seed mean and sample SD (ddof=1), n=2; no confidence or significance inference.",
                            "paired": "100*(arm-random), same dataset/seed; lower Normal FPR is better, higher other rates is better. Undefined precision remains null, never zero-filled.",
                            "attack_family_macro_recall": "Mean exact-class recall over attack classes.",
                            "binary_attack_micro_recall": "All true attacks predicted as any attack / all true attacks.",
                            "binary_attack_macro_recall": "Mean across true attack classes of P(predicted non-Normal | that true class).",
                            "confusion_counts": "predicted_normal_count=CM[c,Normal]; other_attack_count=sum CM[c,k] for k neither Normal nor c. For rare attacks these partition errors; correct-family count is true_positive.",
                            "old_new": "Unweighted exact-class recall means over tasks0-2 and task3 respectively; not a forgetting score.",
                            "cost": "Real steps, memory and exposures matched, NOT equal FLOPs. Row touches are not FLOPs. Recorded components exclude data loading, some final serialization, and unrecorded failed-attempt work; no complete wall-cost claim.",
                            "gradient_proxy": "Extra CE/margin logit-gradient L2 sums are not parameter-gradient norms or matched objective scale."},
            "common_groups": [{"dataset": d, "seed": s, "invariants_sha256": canonical(common[(d, s)]),
                               "prefix": {k: v for k, v in common[(d, s)]["identity"]["start_checkpoint"].items() if k not in ("path", "result_path")},
                               "schedule_hashes": common[(d, s)]["chains"], "full_test_support": supports[(d, s)]} for d in DATASETS for s in SEEDS],
            "sources": sources, "cells": [cells[(d, s, a)] for d in DATASETS for s in SEEDS for a in ARMS],
            "summaries": summaries, "paired_vs_random": paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", choices=("verify", "portable"), default="verify")
    args = parser.parse_args()
    pilot, output = args.pilot_root.resolve(), args.output_dir.resolve()
    require(output != pilot and not pilot.is_relative_to(output) and not output.is_relative_to(pilot),
            "Analysis output must not overlap the original pilot tree")
    require(pilot.is_relative_to(ROOT) and output.is_relative_to(ROOT), "Use repository-local input/output paths")
    commit = os.environ.get("NCD_SOURCE_COMMIT", "")
    require(len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), "Explicit NCD_SOURCE_COMMIT is required for analyzer provenance")
    command = subprocess.list2cmdline(["python", "-I", "ncd/analyze_rare_boundary_pilot.py",
                                      "--pilot-root", pilot.relative_to(ROOT).as_posix(),
                                      "--output-dir", output.relative_to(ROOT).as_posix(), "--weights", args.weights])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.log").open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                print(f"START analyzer_commit={commit} command={command}", flush=True)
                if args.weights == "verify":
                    import torch
                    torch.set_num_threads(2)
                result = analyze(pilot, args.weights == "verify")
                result.update(analyzer_git_commit=commit, command=command,
                              command_note="Equivalent repo-relative command; local command invocation is recorded by the runner.",
                              log_path=(output / "run.log").relative_to(ROOT).as_posix())
                temporary = output / "result.json.tmp"
                temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
                temporary.replace(output / "result.json")
                print(f"COMPLETE cells=16 dataset_arm_groups=8 weights={args.weights}; no training/inference performed", flush=True)
            except Exception:
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
