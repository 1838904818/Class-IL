"""Independently recompute Section25 diagnostics from integer confusion matrices.

No model fitting, hyperparameter selection, or test-dependent threshold changes.
The declared closure is the mean of paired seed ratios, including signed gaps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = {"CIC-IDS-2017": [1, 2, 3], "NSL-KDD": [1, 2, 3, 4, 42]}


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


def metrics(value, names, tasks):
    cm = np.asarray(value["confusion_matrix"])
    n = len(names)
    if cm.shape != (n, n) or not np.issubdtype(cm.dtype, np.integer) or np.any(cm < 0):
        raise ValueError("Invalid complete confusion matrix")
    if value["confusion_labels"] != list(range(n)):
        raise ValueError("Confusion label order differs")
    support, predicted, tp = cm.sum(1), cm.sum(0), cm.diagonal()
    if np.any(support == 0):
        raise ValueError("A formal test class has zero support")
    recall = tp / support
    precision = np.divide(tp, predicted, out=np.zeros(n), where=predicted != 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(n), where=(support + predicted) != 0)
    normal = names.index("Normal")
    new = tasks[-1]
    old = [c for group in tasks[:-1] for c in group]
    result = {"normal_recall": float(recall[normal]), "macro_f1": float(f1.mean()),
              "balanced_accuracy": float(recall.mean()),
              "attack_macro_recall": float(np.delete(recall, normal).mean()),
              "new_macro_recall": float(recall[new].mean()),
              "old_macro_recall": float(recall[old].mean()),
              "support": support.tolist(), "confusion_matrix": cm.tolist(),
              "per_class": [{"class_name": name, "precision": float(precision[c]) if predicted[c] else None,
                             "recall": float(recall[c]), "f1": float(f1[c]),
                             "support": int(support[c])} for c, name in enumerate(names)]}
    for stored, computed in (("macro_f1", "macro_f1"), ("balanced_accuracy", "balanced_accuracy"),
                             ("attack_family_macro_recall", "attack_macro_recall")):
        if not np.isclose(value[stored], result[computed], rtol=0, atol=1e-12):
            raise ValueError(f"Stored {stored} does not reproduce")
    if not np.isclose(1 - value["normal_fpr"], result["normal_recall"], rtol=0, atol=1e-12):
        raise ValueError("Stored Normal recall does not reproduce")
    if len(value["per_class"]) != n:
        raise ValueError("Incomplete per-class metrics")
    for c, row in enumerate(value["per_class"]):
        if row["class_id"] != c or row["class_name"] != names[c]:
            raise ValueError("Per-class label differs")
        for key in ("precision", "recall", "f1"):
            if key == "precision" and predicted[c] == 0:
                if row[key] is not None:
                    raise ValueError("Zero-prediction precision must remain undefined")
                continue
            if not np.isclose(row[key], result["per_class"][c][key], rtol=0, atol=1e-12):
                raise ValueError(f"Stored class {key} does not reproduce")
    return result


def source(path, value, weights):
    evidence = {"result": label(path), "result_sha256": digest(path)}
    if "checkpoints" in value:
        evidence["git_commit"] = value["identity"]["git_commit"]
        evidence["checkpoints"] = []
        for cp in value["checkpoints"]:
            payload = path.parent / cp["checkpoint_file"]
            verified = False
            if weights == "verify":
                if digest(payload) != cp["checkpoint_sha256"]:
                    raise ValueError("Local payload hash differs")
                verified = True
            evidence["checkpoints"].append({"file": cp["checkpoint_file"],
                                            "sha256": cp["checkpoint_sha256"],
                                            "payload_verified_this_analysis": verified})
    else:
        evidence["git_commit"] = value["identity"]["provenance"]["git_commit"]
        evidence["checkpoint_sha256"] = value["identity"]["checkpoint_sha256"]
        evidence["baseline_confusion_exact_match"] = value["baseline_confusion_exact_match"]
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("ncd/results/problem_scout/section21/merged_v1"))
    parser.add_argument("--steps-2000", type=Path, default=Path("ncd/results/problem_scout/section22/steps_2000_v1"))
    parser.add_argument("--posthoc", type=Path, default=Path("ncd/results/problem_scout/section25/posthoc_v1"))
    parser.add_argument("--coverage", type=Path, action="append")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", choices=("portable", "verify"), default="portable")
    args = parser.parse_args()
    coverage = args.coverage or [Path("ncd/results/problem_scout/section25/coverage_v1")]
    cells, missing, references = [], [], []
    for dataset, seeds in SEEDS.items():
        slug = dataset.lower().replace("-", "_")
        for seed in seeds:
            refs, ref_identities = {}, {}
            for method, directory in (("replay", args.steps_2000), ("joint", args.baseline)):
                path = directory / slug / f"{method}_seed{seed}" / "result.json"
                value = read(path)
                identity = value["identity"]
                ref_identities[method] = identity
                if (identity["dataset"], identity["method"], identity["seed"]) != (dataset, method, seed):
                    raise ValueError("Reference pairing mismatch")
                if not value["completed"] or identity["effective_config"].get("smoke"):
                    raise ValueError("Reference is not formal and complete")
                names, tasks = identity["class_order"], identity["tasks"]
                refs[method] = metrics(value["checkpoints"][-1]["id"], names, tasks)
                references.append({"dataset": dataset, "seed": seed, "method": method,
                                   "metrics": refs[method], "result": label(path), "sha256": digest(path)})
            denom = refs["joint"]["normal_recall"] - refs["replay"]["normal_recall"]
            for treatment in ("D1", "D2", "D3", "D4", "D5", "Joint+D4"):
                if treatment in ("D1", "D2", "D3"):
                    candidates = [r / slug / f"{treatment}_seed{seed}" / "result.json" for r in coverage]
                    path = next((p for p in candidates if p.is_file()), candidates[0])
                else:
                    method = "joint" if treatment == "Joint+D4" else "replay"
                    path = args.posthoc / slug / f"{method}_seed{seed}" / "result.json"
                if not path.is_file():
                    missing.append({"dataset": dataset, "seed": seed, "treatment": treatment})
                    continue
                value = read(path)
                identity = value["identity"]
                if not value.get("completed") or value.get("smoke") or identity.get("smoke"):
                    raise ValueError("Treatment is not formal and complete")
                if identity["dataset"] != dataset or identity["seed"] != seed:
                    raise ValueError("Treatment pairing mismatch")
                if value["class_order"] != names or value["tasks"] != tasks:
                    raise ValueError("Treatment class/task order mismatch")
                if treatment in ("D1", "D2", "D3"):
                    if identity["treatment"] != treatment or identity["method"] != "replay":
                        raise ValueError("Treatment identity differs")
                    if len(value["checkpoints"]) != len(tasks):
                        raise ValueError("Coverage checkpoints are incomplete")
                    original = ref_identities["replay"]
                    for key in ("x_train", "y_train", "x_test", "y_test"):
                        if identity["data_binding"][key] != original["data_binding"][key]:
                            raise ValueError("Coverage prepared data differs from Replay")
                    for key in ("hidden", "batch_size", "learning_rate", "epochs_per_task", "min_steps_per_task", "temperature"):
                        if identity["effective_config"][key] != original["effective_config"][key]:
                            raise ValueError(f"Coverage training configuration differs: {key}")
                    measured = metrics(value["checkpoints"][-1]["id"], names, tasks)
                    cost = {k: value[k] for k in ("training_seconds_total", "selection_seconds_total", "evaluation_seconds_total")}
                    evidence = source(path, value, args.weights)
                    evidence["d2_budget_mode"] = identity["d2_budget_mode"]
                else:
                    if not value["baseline_confusion_exact_match"]:
                        raise ValueError("Posthoc original predictions did not reproduce")
                    ref = refs["joint" if treatment == "Joint+D4" else "replay"]
                    if value["metrics"]["baseline"]["confusion_matrix"] != ref["confusion_matrix"]:
                        raise ValueError("Posthoc reference confusion differs")
                    measured = metrics(value["metrics"]["D4" if treatment == "Joint+D4" else treatment], names, tasks)
                    cost, evidence = value["timing"], source(path, value, "portable")
                if measured["support"] != refs["replay"]["support"]:
                    raise ValueError("Full test supports differ")
                delta = {k: measured[k] - refs["replay"][k] for k in
                         ("normal_recall", "macro_f1", "attack_macro_recall", "balanced_accuracy", "new_macro_recall", "old_macro_recall")}
                cells.append({"dataset": dataset, "seed": seed, "treatment": treatment,
                              "metrics": measured, "delta_vs_replay": delta, "denominator": denom,
                              "closure": delta["normal_recall"] / denom if denom != 0 else None,
                              "normal_delta_vs_joint": measured["normal_recall"] - refs["joint"]["normal_recall"],
                              "cost": cost, "evidence": evidence})
    d2_modes = {r["evidence"]["d2_budget_mode"] for r in cells if r["treatment"] == "D2"}
    if len(d2_modes) > 1:
        raise ValueError("Cannot combine different D2 allocation interpretations")
    summary = []
    for dataset, seeds in SEEDS.items():
        for treatment in ("D1", "D2", "D3", "D4", "D5", "Joint+D4"):
            selected = [r for r in cells if r["dataset"] == dataset and r["treatment"] == treatment]
            row = {"dataset": dataset, "treatment": treatment, "n": len(selected),
                   "expected_n": len(seeds), "complete": len(selected) == len(seeds)}
            if selected:
                row["means"] = {key: float(np.mean([r["metrics"][key] for r in selected])) for key in
                                ("normal_recall", "macro_f1", "attack_macro_recall", "balanced_accuracy", "new_macro_recall", "old_macro_recall")}
                row["paired_changes"] = {key: float(np.mean([r["delta_vs_replay"][key] for r in selected])) for key in row["means"]}
                row["sample_sd"] = {key: float(np.std([r["metrics"][key] for r in selected], ddof=1))
                                    if len(selected) > 1 else None for key in row["means"]}
                row["closure"] = (float(np.mean([r["closure"] for r in selected]))
                                  if all(r["closure"] is not None for r in selected) else None)
                row["nonpositive_denominator_seeds"] = [r["seed"] for r in selected if r["denominator"] <= 0]
                row["near_zero_denominator_seeds_abs_below_0_1pp"] = [r["seed"] for r in selected if abs(r["denominator"]) < .001]
                row["guardrails_met"] = row["complete"] and row["paired_changes"]["macro_f1"] >= -1e-12 and row["paired_changes"]["attack_macro_recall"] >= -.02 - 1e-12
                row["normal_delta_vs_joint"] = float(np.mean([r["normal_delta_vs_joint"] for r in selected]))
            summary.append(row)
    result = {"schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(), "complete": not missing,
              "analysis_provenance": {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                                      "source_sha256": digest(Path(__file__)),
                                      "command": ["python", "-I", "ncd/analyze_benign_diagnostics.py", *sys.argv[1:]],
                                      "dataset_seeds": SEEDS},
              "weights_mode": args.weights,
              "weights_scope": "verify rehashes all available coverage checkpoints. Posthoc payloads were hash-checked by its inference entry; this analysis checks its exact baseline confusion reproduction and recorded source hash, not a second payload read.",
              "closure_definition": "Mean of all paired seed ratios, retaining signed denominators; zero denominator is undefined. No seed deletion or ratio-of-means substitution.",
              "guardrails_definition": "Dataset seed-mean Macro-F1 delta >=0 and attack-class macro recall delta >=-0.02; every seed change retained separately.",
              "coverage_priority": [label(r) for r in coverage],
              "hypothesis_limit": "Only complete same-treatment two-dataset results can meet the stated rule; D1 is increased-budget reference only. Positive diagnostics are not new mechanisms or overall RQ completion.",
              "references": references, "cells": cells, "summary": summary, "missing": missing}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"complete": result["complete"], "completed_cells": len(cells), "missing_cells": len(missing), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
