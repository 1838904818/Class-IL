"""Recompute Section26 operational metrics and decisions from matched integer CMs.

Historical Section21/22/25 runs are not admitted to the matched main table.
No training, tuning, data preparation, or score-based duplicate selection occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("CIC-IDS-2017", "UNSW-NB15", "NSL-KDD")
SEEDS = (1, 2, 3, 4, 42)
METHODS = ("replay", "fixed50", "replay_la", "derpp", "farq", "farq_all")
NAMES = dict(zip(METHODS, ("Replay", "Fixed-50", "Replay + LA", "DER++", "FARQ", "FARQ-all")))
SCALARS = ("normal_fpr", "attack_detection", "macro_f1", "balanced_accuracy",
           "normal_recall", "attack_family_macro_recall", "attack_guard_recall")
MATCHED_CONFIG = ("hidden", "batch_size", "learning_rate", "epochs_per_task",
                  "min_steps_per_task", "buffer_per_class", "section26")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def label(path):
    return path.resolve().relative_to(ROOT).as_posix()


def safe_text(value):
    return str(value).replace(str(ROOT), "<repository>").replace(ROOT.as_posix(), "<repository>")


def stats(values):
    return {"mean": float(np.mean(values)),
            "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}


def metrics(value, names, seen=None):
    """Independent integer-CM arithmetic; do not import runner metric helpers."""
    cm = np.asarray(value["confusion_matrix"])
    n = len(names)
    if (cm.shape != (n, n) or not np.issubdtype(cm.dtype, np.integer)
            or np.any(cm < 0) or value["confusion_labels"] != list(range(n))):
        raise ValueError("Invalid integer confusion matrix or label order")
    seen = list(range(n)) if seen is None else list(seen)
    support, predicted, tp = cm.sum(1), cm.sum(0), cm.diagonal()
    if np.any(support[seen] == 0):
        raise ValueError("An evaluated class has no test support")
    normal = names.index("Normal")
    attack = [c for c in seen if c != normal]
    guard = [c for c in attack if support[c] >= 20]
    recall = np.divide(tp, support, out=np.zeros(n), where=support != 0)
    precision = np.divide(tp, predicted, out=np.zeros(n), where=predicted != 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(n), where=(support + predicted) != 0)
    out = {"normal_fpr": float(1 - recall[normal]),
           "attack_detection": float(1 - cm[attack, normal].sum() / support[attack].sum()),
           "macro_f1": float(f1[seen].mean()), "balanced_accuracy": float(recall[seen].mean()),
           "normal_recall": float(recall[normal]),
           "attack_family_macro_recall": float(recall[attack].mean()),
           "attack_guard_recall": float(recall[guard].mean()) if guard else None,
           "guard_excluded_classes": [names[c] for c in attack if c not in guard],
           "support": support.tolist(), "confusion_matrix": cm.tolist(),
           "per_class": [{"class_id": c, "class_name": names[c], "support": int(support[c]),
                          "precision": float(precision[c]) if predicted[c] else None,
                          "recall": float(recall[c]), "f1": float(f1[c])} for c in seen]}
    for stored, computed in (("normal_fpr", "normal_fpr"), ("attack_recall", "attack_detection"),
                             ("macro_f1", "macro_f1"), ("balanced_accuracy", "balanced_accuracy"),
                             ("attack_family_macro_recall", "attack_family_macro_recall")):
        if stored in value and not np.isclose(value[stored], out[computed], rtol=0, atol=1e-12):
            raise ValueError(f"Stored {stored} differs from integer-CM arithmetic")
    return out


def validate_allocation(allocation, identity, seen):
    """Recompute the prescribed quota from recorded integer validation errors."""
    normal = identity["class_order"].index("Normal")
    attacks = sorted(c for c in seen if c != normal)
    total = 200 * len(seen)
    error = {c: allocation["validation_error_counts"][str(c)] / allocation["validation_eval_counts"][str(c)] for c in seen}
    f, m = error[normal], sum(error[c] for c in attacks) / len(attacks)
    if any(not np.isclose(allocation[k], v, rtol=0, atol=1e-12) for k, v in (("f", f), ("m", m))):
        raise ValueError("Stored f/m differs from integer validation errors")
    method = identity["method"]
    q = 1 / len(seen)
    caps = {c: 200 for c in seen}
    if method in ("fixed50", "farq"):
        q = .5 if method == "fixed50" else max(q, min(.6, f / (f + m) if f + m else q))
        if not np.isclose(allocation["q_requested"], q, rtol=0, atol=1e-12):
            raise ValueError("Requested FARQ q does not follow the fixed formula")
        nc = min(math.floor(allocation["q_requested"] * total), total - 20 * len(attacks))
        rest, rem = divmod(total - nc, len(attacks))
        caps = {normal: nc, **{c: rest + int(i < rem) for i, c in enumerate(attacks)}}
    elif method == "farq_all" and sum(error.values()):
        raw = {c: (total - 20 * len(seen)) * error[c] / sum(error.values()) for c in seen}
        caps = {c: 20 + math.floor(raw[c]) for c in seen}
        order = sorted(seen, key=lambda c: (-(raw[c] - math.floor(raw[c])), c))
        for c in order[:total - sum(caps.values())]:
            caps[c] += 1
        q = caps[normal] / total
    if allocation["target_counts"] != {str(c): caps[c] for c in seen} or not np.isclose(allocation["q_requested"], q, rtol=0, atol=1e-12):
        raise ValueError("Recorded class quotas differ from formula/apportionment")
    if any(caps[c] < 20 for c in attacks):
        raise ValueError("An attack class target falls below20")
    heldout = identity["section26_holdout"]
    if allocation["heldout_pool_counts"] != {str(c): heldout[str(c)]["validation_count"] for c in seen}:
        raise ValueError("Recorded heldout-pool counts differ from identity")
    training, valid = allocation["actual_training_counts"], allocation["actual_validation_counts"]
    actual = sum(training.values()) + sum(valid.values())
    if actual != allocation["actual_total"]:
        raise ValueError("Retained training plus validation memory does not reconcile")
    normal_n = training[str(normal)] + valid[str(normal)]
    for key, value in (("normal_actual_fraction_of_capacity", normal_n / total),
                       ("normal_actual_fraction_of_used", normal_n / actual)):
        if not np.isclose(allocation[key], value, rtol=0, atol=1e-12):
            raise ValueError("Actual Normal quota fraction does not reconcile")


def load_cell(path, weights):
    value = read(path)
    identity = value["identity"]
    config = identity["effective_config"]
    if not value.get("completed") or config.get("smoke") or not config.get("section26"):
        raise ValueError("Not a completed formal Section26 matched-holdout cell")
    if (config.get("epochs_per_task") != 8 or config.get("min_steps_per_task") != 2000
            or config.get("buffer_per_class") != 200 or config.get("batch_size") != 256):
        raise ValueError("Not the prescribed full S2000/8-epoch/200-per-class/batch256 setting")
    completed = datetime.fromisoformat(value["completed_at"].replace("Z", "+00:00"))
    if completed.tzinfo is None:
        raise ValueError("Completion timestamp has no timezone")
    names, tasks = identity["class_order"], identity["tasks"]
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if len(value["checkpoints"]) != len(tasks):
        raise ValueError("Missing checkpoint")
    trajectory, checkpoints = [], []
    for t, cp in enumerate(value["checkpoints"]):
        seen = [c for task in tasks[:t + 1] for c in task]
        if cp["task_index"] != t or cp["seen_classes"] != seen:
            raise ValueError("Checkpoint task or class order differs")
        if cp["identity_hash"] != identity_hash:
            raise ValueError("Checkpoint identity hash differs")
        schedule = cp.get("training_schedule", {})
        if schedule.get("optimizer_steps", 0) < 2000 or schedule.get("optimizer_steps") != schedule.get("target_steps"):
            raise ValueError("Actual optimizer updates do not satisfy the recorded S2000 schedule")
        measured = metrics(cp["id"], names, seen)
        payload = path.parent / cp["checkpoint_file"]
        sidecar = payload.with_suffix(".json")
        if read(sidecar) != cp:
            raise ValueError("Checkpoint sidecar differs from result")
        verified = weights == "verify"
        if verified and digest(payload) != cp["checkpoint_sha256"]:
            raise ValueError("Actual checkpoint SHA differs")
        allocation = cp.get("allocation")
        if not allocation:
            raise ValueError("Missing matched-holdout allocation evidence")
        validate_allocation(allocation, identity, seen)
        if identity["method"] != "joint":
            if allocation["capacity_total"] != 200 * len(seen):
                raise ValueError("Replay capacity differs from 200 times seen classes")
            if allocation["actual_total"] > allocation["capacity_total"]:
                raise ValueError("Actual memory exceeds matched capacity")
        checkpoints.append({"file": cp["checkpoint_file"], "sha256": cp["checkpoint_sha256"],
                            "payload_verified_this_analysis": verified})
        row = {"task_index": t, "metrics": measured, "allocation": allocation}
        if "la_id" in cp:
            row["la_metrics"] = metrics(cp["la_id"], names, seen)
        trajectory.append(row)
    cost = {"training_seconds": float(value["training_seconds_total"]),
            "selection_seconds": float(sum(c["buffer_selection_seconds"] for c in value["checkpoints"])),
            "evaluation_seconds": float(sum(c["evaluation_seconds"] for c in value["checkpoints"]))}
    out = {"dataset": identity["dataset"], "method": identity["method"], "seed": identity["seed"],
           "completed_at": completed.isoformat(), "metrics": trajectory[-1]["metrics"],
           "trajectory": trajectory, "cost": cost,
           "source": {"result": label(path), "result_sha256": digest(path),
                      "git_commit": identity["git_commit"], "checkpoints": checkpoints},
           "comparison_identity": {"data_binding": identity["data_binding"], "class_order": names,
                                   "tasks": tasks, "section26_holdout": identity["section26_holdout"],
                                   "config": {k: config.get(k) for k in MATCHED_CONFIG}}}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", choices=("verify", "portable"), default="portable")
    args = parser.parse_args()
    selected, duplicates, copies, invalid = {}, [], [], []
    for dataset in DATASETS:
        slug = dataset.lower().replace("-", "_")
        for method in ("replay", "fixed50", "derpp", "farq", "farq_all", "joint"):
            for seed in SEEDS:
                candidates = []
                paths = {r.resolve() / slug / f"{method}_seed{seed}" / "result.json" for r in args.input}
                for path in sorted(paths):
                    if not path.is_file():
                        continue
                    try:
                        row = load_cell(path, args.weights)
                        if (row["dataset"], row["method"], row["seed"]) != (dataset, method, seed):
                            raise ValueError("Path and recorded cell identity differ")
                        candidates.append(row)
                    except (KeyError, ValueError, OSError) as error:
                        invalid.append({"result": label(path), "reason": safe_text(error)})
                candidates.sort(key=lambda r: (datetime.fromisoformat(r["completed_at"]), r["source"]["result"]))
                if candidates:
                    distinct = []
                    executions = set()
                    for row in candidates:
                        execution = (row["completed_at"], row["source"]["git_commit"],
                                     tuple(c["sha256"] for c in row["source"]["checkpoints"]))
                        if execution in executions:
                            copies.append(row["source"])
                        else:
                            distinct.append(row)
                            executions.add(execution)
                    selected[dataset, method, seed] = distinct[0]
                    duplicates.extend(distinct[1:])
    # Reject accidental mixtures of splits, model budgets, and holdout protocols.
    for dataset in DATASETS:
        for seed in SEEDS:
            rows = [r for (d, _, s), r in selected.items() if d == dataset and s == seed]
            if any(r["comparison_identity"] != rows[0]["comparison_identity"] for r in rows[1:]):
                raise ValueError(f"Unmatched data/config at {dataset} seed {seed}")
            counts = [r["trajectory"][-1]["allocation"].get("heldout_pool_counts") for r in rows]
            if any(c != counts[0] for c in counts[1:]):
                raise ValueError(f"Unmatched permanent holdout counts at {dataset} seed {seed}")
            supports = [r["metrics"]["support"] for r in rows]
            if any(s != supports[0] for s in supports[1:]):
                raise ValueError(f"Unmatched full-test supports at {dataset} seed {seed}")
    for (dataset, method, seed), row in list(selected.items()):
        if method == "replay" and "la_metrics" in row["trajectory"][-1]:
            la = dict(row, method="replay_la", metrics=row["trajectory"][-1]["la_metrics"],
                      cost={"shares_replay_training": True},
                      trajectory=[dict(c, metrics=c["la_metrics"]) for c in row["trajectory"]])
            selected[dataset, "replay_la", seed] = la
    summaries, missing = [], []
    for dataset in DATASETS:
        for method in METHODS + ("joint",):
            rows = [selected[dataset, method, s] for s in SEEDS if (dataset, method, s) in selected]
            absent = [s for s in SEEDS if (dataset, method, s) not in selected]
            missing.extend({"dataset": dataset, "method": method, "seed": s} for s in absent)
            summary = {"dataset": dataset, "method": method, "n": len(rows), "expected_n": 5,
                       "complete": not absent, "missing_seeds": absent}
            if rows:
                summary["metrics"] = {k: stats([r["metrics"][k] for r in rows]) for k in SCALARS}
                summary["per_class"] = []
                for c, item in enumerate(rows[0]["metrics"]["per_class"]):
                    summary["per_class"].append({k: item[k] for k in ("class_id", "class_name", "support")} |
                        {k: (stats([r["metrics"]["per_class"][c][k] for r in rows])
                             if all(r["metrics"]["per_class"][c][k] is not None for r in rows) else None)
                         for k in ("precision", "recall", "f1")})
                paired, closures = [], []
                for row in rows:
                    base = selected.get((dataset, "replay", row["seed"]))
                    joint = selected.get((dataset, "joint", row["seed"]))
                    if base:
                        paired.append({"seed": row["seed"], **{k: row["metrics"][k] - base["metrics"][k] for k in SCALARS}})
                    if base and joint:
                        denom = base["metrics"]["normal_fpr"] - joint["metrics"]["normal_fpr"]
                        closures.append({"seed": row["seed"], "denominator": denom,
                                         "ratio": (base["metrics"]["normal_fpr"] - row["metrics"]["normal_fpr"]) / denom if denom else None})
                summary["paired_seed_changes"] = paired
                summary["paired_changes"] = {k: stats([r[k] for r in paired]) for k in SCALARS} if paired else {}
                summary["closure_seed_ratios"] = closures
                summary["closure"] = stats([r["ratio"] for r in closures]) if closures and all(r["ratio"] is not None for r in closures) else None
                summary["closure_complete"] = len(closures) == 5 and summary["closure"] is not None
                summary["nonpositive_denominator_seeds"] = [r["seed"] for r in closures if r["denominator"] <= 0]
                summary["near_zero_denominator_seeds_abs_below_0_1pp"] = [r["seed"] for r in closures if abs(r["denominator"]) < .001]
            summaries.append(summary)
    by = {(r["dataset"], r["method"]): r for r in summaries}
    verdicts = []

    def verdict(number, description, required, evaluate):
        ready = all(by[k]["complete"] for k in required)
        values, passed = evaluate() if ready else ({}, None)
        verdicts.append({"criterion": number, "description": description,
                         "status": "pending" if passed is None else "pass" if passed else "fail", "values": values})

    cic, unsw, nsl = DATASETS

    def first():
        row = by[cic, "farq"]
        delta = row["paired_changes"]
        values = {"macro_f1_delta_pp": 100 * delta["macro_f1"]["mean"],
                  "attack_guard_delta_pp": 100 * delta["attack_guard_recall"]["mean"],
                  "paired_mean_closure_percent": 100 * row["closure"]["mean"] if row["closure_complete"] else None}
        passed = (values["macro_f1_delta_pp"] >= 3 - 1e-10 and values["attack_guard_delta_pp"] >= -2 - 1e-10
                  and values["paired_mean_closure_percent"] >= 50 - 1e-10) if row["closure_complete"] else None
        return values, passed

    def compare(dataset, other, threshold, strict=False):
        delta = 100 * (by[dataset, "farq"]["metrics"]["macro_f1"]["mean"] - by[dataset, other]["metrics"]["macro_f1"]["mean"])
        values = {"farq_macro_f1_delta_pp": delta, "comparison": other, "threshold_pp": threshold}
        if dataset == unsw and by[dataset, "fixed50"]["complete"]:
            values["fixed50_delta_vs_replay_pp"] = 100 * by[dataset, "fixed50"]["paired_changes"]["macro_f1"]["mean"]
        return values, delta > threshold if strict else delta >= threshold - 1e-10

    verdict(1, "CIC17: MF1 gain >=3pp; mean paired FPR-gap closure >=50%; attack-family guard decline <=2pp",
            [(cic, x) for x in ("farq", "replay", "joint")], first)
    verdict(2, "CIC17: FARQ MF1 >= Fixed-50", [(cic, x) for x in ("farq", "fixed50")], lambda: compare(cic, "fixed50", 0))
    verdict(3, "UNSW: FARQ MF1 >= Replay minus 1pp; report Fixed-50 contrast", [(unsw, x) for x in ("farq", "replay")], lambda: compare(unsw, "replay", -1))
    verdict(4, "NSL: FARQ MF1 >= Replay minus 1pp", [(nsl, x) for x in ("farq", "replay")], lambda: compare(nsl, "replay", -1))
    verdict(5, "CIC17: FARQ MF1 strictly greater than FARQ-all", [(cic, x) for x in ("farq", "farq_all")], lambda: compare(cic, "farq_all", 0, True))
    result = {"schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
              "complete": not missing and not invalid, "weights_mode": args.weights,
              "weights_scope": "verify hashes each actual checkpoint payload; portable verifies metadata and sidecars only",
              "source_selection": "Earliest timezone-aware completed_at among valid complete matched cells; never score-based. All valid duplicates and their costs retained.",
              "definitions": {"units": "Proportions in JSON; percentages/percentage points in Markdown",
                              "attack_detection": "P(predicted any attack | true any attack), support-weighted binary detection",
                              "attack_guard": "Macro recall over attack families with test support >=20; excluded families remain in MF1/BAcc/per-class tables",
                              "closure": "Mean of all paired seed ratios (FPR_Replay-FPR_method)/(FPR_Replay-FPR_Joint); signed denominators retained; zero is undefined",
                              "memory_scope": "M=200*seen includes retained training replay and validation; matched permanent holdout, with explicitly permitted seen training-archive access; not strict no-history streaming",
                              "scope": "Section26 matched-holdout experiments only; not an OFRA repair or overall RQ1/RQ3/innovation closure"},
              "analysis_provenance": {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                                      "source_sha256": digest(Path(__file__)), "command": ["python", "-I", "ncd/analyze_farq.py", *map(safe_text, sys.argv[1:])]},
              "cells": list(selected.values()), "duplicates": duplicates, "identical_result_copies": copies,
              "total_cost_seconds": {scope: {key: sum(r["cost"][key] for r in rows) for key in
                                             ("training_seconds", "selection_seconds", "evaluation_seconds")}
                                     for scope, rows in (("selected_physical_runs_including_joint", [r for r in selected.values() if r["method"] != "replay_la"]),
                                                         ("duplicate_executions", duplicates))},
              "invalid_sources": invalid,
              "completed_main_outputs": sum(r["method"] != "joint" for r in selected.values()), "expected_main_outputs": 90,
              "completed_joint_references": sum(r["method"] == "joint" for r in selected.values()), "expected_joint_references": 15,
              "missing": missing, "summary": summaries, "criteria_26_5": verdicts,
              "all_criteria_evaluated": all(v["status"] != "pending" for v in verdicts),
              "registered_success": "fail" if any(v["status"] == "fail" for v in verdicts) else "pending" if any(v["status"] == "pending" for v in verdicts) else "pass"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    text = ["Section26 matched results; incomplete groups are descriptive only. Values: mean ± sample SD (%).", "",
            "| Dataset / method (n/5) | Benign FPR ↓ | Attack detection ↑ | Macro-F1 | BAcc | Normal recall | Attack-family guard recall |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        if row["method"] == "joint":
            continue
        values = []
        for k in ("normal_fpr", "attack_detection", "macro_f1", "balanced_accuracy", "normal_recall", "attack_guard_recall"):
            item = row.get("metrics", {}).get(k)
            values.append("未完成" if item is None else f"{100 * item['mean']:.2f}" + (f" ± {100 * item['sample_sd']:.2f}" if item["sample_sd"] is not None else ""))
        text.append(f"| {row['dataset']} / {NAMES[row['method']]} ({row['n']}/5){' 未完成' if not row['complete'] else ''} | " + " | ".join(values) + " |")
    text += ["", "| Dataset / FARQ task | n/5 | Requested Normal q (%) | Actual Normal / capacity (%) | Actual Normal / used (%) |",
             "|---|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        rows = [r for (d, m, _), r in selected.items() if d == dataset and m == "farq"]
        if not rows:
            text.append(f"| {dataset} / all tasks | 0/5 | 未完成 | 未完成 | 未完成 |")
        for t in range(len(rows[0]["trajectory"]) if rows else 0):
            allocations = [r["trajectory"][t]["allocation"] for r in rows]
            values = []
            for key in ("q_requested", "normal_actual_fraction_of_capacity", "normal_actual_fraction_of_used"):
                value = stats([a[key] for a in allocations])
                values.append(f"{100 * value['mean']:.2f}" + (f" ± {100 * value['sample_sd']:.2f}" if value["sample_sd"] is not None else ""))
            text.append(f"| {dataset} / {t} | {len(rows)}/5 | " + " | ".join(values) + " |")
    text += ["", "26.5: " + "; ".join(f"{v['criterion']}: {v['status']}" for v in verdicts) + ".",
             "Full per-class precision/recall, all signed closure ratios, task allocations, selected/duplicate costs and source hashes are in result.json.",
             "No claim of full research completion or mechanism novelty follows from these thresholds."]
    (args.output_dir / "tables.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(json.dumps({"complete": result["complete"], "completed_main_outputs": result["completed_main_outputs"],
                      "completed_joint_references": result["completed_joint_references"], "missing_outputs": len(missing),
                      "invalid_sources": invalid, "criteria_26_5": verdicts}, indent=2))


if __name__ == "__main__":
    main()
