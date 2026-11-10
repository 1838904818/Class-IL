"""Check and export one completed §489 B1/B2 cell without exposing local paths."""
from __future__ import annotations

import argparse
import hashlib
import json
from math import isclose
from pathlib import Path
from statistics import mean


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def verify_matrix(view):
    labels = view["confusion_labels"]
    cm = view["confusion_matrix"]
    if len(labels) != len(cm) or len(labels) != len(set(labels)) or any(
        len(row) != len(labels) or any(type(v) is not int or v < 0 for v in row)
        for row in cm
    ):
        raise ValueError("Malformed operating-point confusion matrix")
    totals = [sum(row) for row in cm]
    recall = {label: cm[i][i] / n for i, (label, n) in enumerate(zip(labels, totals)) if n}
    if sum(totals) != view["evaluated_rows"] or not isclose(
        mean(recall.values()), view["balanced_accuracy"], abs_tol=1e-12, rel_tol=0
    ) or not isclose(1 - recall[0], view["normal_fpr"], abs_tol=1e-12, rel_tol=0):
        raise ValueError("Stored operating point disagrees with the matrix")


def export(cell, reference_path, output):
    result_bytes = (cell / "cell/result.json").read_bytes()
    summary_bytes = (cell / "summary.json").read_bytes()
    result, summary = json.loads(result_bytes), json.loads(summary_bytes)
    reference = json.loads(reference_path.read_bytes())
    identity = result["identity"]
    ref_id = reference["identity"]
    cfg = identity["effective_config"]
    dataset, method, seed = (identity[k] for k in ("dataset", "method", "seed"))
    requested = cfg["normal_training_rows"]
    b1 = method == "R3-A" and cfg.get("section489_benign_volume") is True
    b2 = method == "R3-H" and cfg.get("section165_benign_refresh") is True
    if not (b1 and requested in (5000, 10000, 20000) or b2 and requested == 50000):
        raise ValueError("Unexpected §489 arm")
    ref_method = "R3-A" if b1 else "R3-L"
    if seed not in range(10, 15) or not all(
        x["completed"] and x["scientific_result"] for x in (result, summary, reference)
    ) or cfg["smoke"] is not False or (
        summary["dataset"], summary["method"], summary["seed"], summary["source_commit"]
    ) != (dataset, method, seed, identity["git_commit"]) or (
        ref_id["dataset"], ref_id["method"], ref_id["seed"]
    ) != (dataset, ref_method, seed):
        raise ValueError("Cell, summary or reference identity mismatch")
    if any(result[key] != reference[key] for key in ("tasks", "class_order")) or (
        identity["data_binding"] != ref_id["data_binding"]
    ):
        raise ValueError("Reference data binding or task order differs")
    checkpoints = result["checkpoints"]
    if len(checkpoints) != len(result["tasks"]) or len(checkpoints) != summary["checkpoints"] or (
        len(checkpoints) != len(reference["checkpoints"])
    ):
        raise ValueError("Missing stage")
    for i, (cp, ref) in enumerate(zip(checkpoints, reference["checkpoints"])):
        if cp["task_index"] != i or cp["seen_classes"] != ref["seen_classes"]:
            raise ValueError("Unexpected stage or label support")
        checkpoint = cell / "cell" / cp["checkpoint_file"]
        if sha256(checkpoint.read_bytes()) != cp["checkpoint_sha256"] or (
            json.loads(checkpoint.with_suffix(".json").read_bytes()) != cp
        ):
            raise ValueError("Checkpoint or stage sidecar differs")
        benign = cp["benign_exposure"]
        n = benign["optimization_rows"] + benign["stopping_rows"]
        if benign["requested_rows"] != requested or not 0 < n <= requested or not (
            0 < benign["retained_normal_rows"] <= 200 if b2
            else benign["retained_normal_rows"] == 0
        ):
            raise ValueError("Normal role counts do not match the registered arm")
        for key in ("0.01", "0.02", "0.05"):
            cal, old_cal = cp["calibration"][key], ref["calibration"][key]
            if any(cal[k] != old_cal[k] for k in ("row_ids_sha256", "calibration_rows", "target_fpr")):
                raise ValueError("Permanent calibration reference differs")
            if not 0 <= cal["actual_false_alarms"] <= cal["allowed_false_alarms"]:
                raise ValueError("Calibration exceeds the saved budget")
            verify_matrix(cp["operating_points"][key])
    cfg["problem_scout_data_root"] = "<DATA_ROOT>"
    for flag, replacement in (("--data-root", "<DATA_ROOT>"), ("--output-root", "<OUTPUT_ROOT>")):
        cmd = identity["command"]
        if cmd.count(flag) != 1:
            raise ValueError(f"Missing or ambiguous command flag: {flag}")
        cmd[cmd.index(flag) + 1] = replacement
    result["publication"] = {
        "source_result_sha256": sha256(result_bytes),
        "redaction_scope": "Only local data-root and output-root strings were replaced.",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "cell").mkdir()
    with (output / "cell/result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    with (output / "summary.json").open("xb") as stream:
        stream.write(summary_bytes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.cell, args.reference, args.output)


if __name__ == "__main__":
    main()
