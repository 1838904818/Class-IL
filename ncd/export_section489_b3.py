"""Export the descriptive §489 B3 decomposition from existing §165 results.

The columns mirror the UNSW old-class table. Peak excludes the final stage;
error destinations are percentages of all TEST rows of the true class, not
percentages conditional on making an error. No model is loaded or rerun.
"""
from __future__ import annotations

import argparse
import csv
import json
from fractions import Fraction
from pathlib import Path
from statistics import mean


SCOPE = {
    "cic-ids-2017": ("CIC-IDS-2017", ("R3-A", "R3-L")),
    "cse-cic-ids2018": ("CSE-CIC-IDS2018", ("R3-A", "R3-L", "ar", "replay_bs")),
}
SEEDS = tuple(range(10, 15))
TARGET = "0.01"


def read_cell(root, directory, dataset, arm, seed):
    path = root / directory / f"{arm}_seed{seed}" / "result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    identity = result["identity"]
    method = "aacil_nogkd_wa" if arm == "ar" else arm
    scientific = result.get("scientific_result")
    if "scientific_result" not in result and arm in ("ar", "replay_bs"):
        # These older published §165 results have no scientific_result field.
        scientific = identity["effective_config"]["smoke"] is False
    if not (result["completed"] and scientific) or (
        identity["dataset"], identity["method"], identity["seed"]
    ) != (dataset, method, seed):
        raise ValueError(f"Wrong or incomplete cell: {path}")
    tasks, names, checkpoints = result["tasks"], result["class_order"], result["checkpoints"]
    intro = {label: i for i, task in enumerate(tasks) for label in task}
    if len(checkpoints) != len(tasks) or len(tasks) < 2 or names[0] != "Normal" or (
        set(intro) != set(range(len(names))) or sum(map(len, tasks)) != len(names)
    ):
        raise ValueError(f"Invalid trajectory: {path}")
    recalls = []
    for i, cp in enumerate(checkpoints):
        view = cp["operating_points"][TARGET]
        labels, cm = view["confusion_labels"], view["confusion_matrix"]
        if cp["task_index"] != i or len(labels) != len(set(labels)) or len(cm) != len(labels) or any(
            len(row) != len(labels) or any(type(x) is not int or x < 0 for x in row)
            for row in cm
        ):
            raise ValueError(f"Invalid stage matrix: {path} task {i}")
        counts = {label: dict(zip(labels, row)) for label, row in zip(labels, cm)}
        support = {label: sum(row.values()) for label, row in counts.items()}
        seen = {label for label, j in intro.items() if j <= i}
        if sum(support.values()) != view["evaluated_rows"] or seen != {
            label for label, n in support.items() if n
        }:
            raise ValueError(f"Invalid support: {path} task {i}")
        recalls.append({label: Fraction(counts[label][label], support[label]) for label in seen})
    out = {}
    for label in (c for task in tasks[:-1] for c in task):
        first = recalls[intro[label]][label]
        peak = max(stage[label] for stage in recalls[intro[label]:-1])
        final = recalls[-1][label]
        row, n = counts[label], support[label]
        to_normal = Fraction(row[0], n) if label != 0 else Fraction(0)
        to_newer = Fraction(sum(row[j] for j in intro if j != 0 and intro[j] > intro[label]), n)
        to_old = Fraction(sum(row[j] for j in intro
                              if j not in (0, label) and intro[j] <= intro[label]), n)
        if final + to_normal + to_newer + to_old != 1:
            raise ValueError(f"Destination partition is incomplete: {path} class {label}")
        out[label] = (first, peak, final, peak-final, to_normal, to_newer, to_old)
    return names, intro, out


def rows(root):
    fields = (
        "first_recall_pct", "peak_pre_final_recall_pct", "final_recall_pct",
        "signed_forgetting_pp", "final_to_normal_pct", "final_to_newer_pct",
        "final_to_old_or_same_stage_pct",
    )
    for directory, (dataset, methods) in SCOPE.items():
        reference = None
        for arm in methods:
            cells = [read_cell(root, directory, dataset, arm, seed) for seed in SEEDS]
            names, intro, initial = cells[0]
            if any((n, i, set(x)) != (names, intro, set(initial)) for n, i, x in cells):
                raise ValueError(f"Class/task mismatch across seeds: {dataset} {arm}")
            if reference is None:
                reference = (names, intro)
            elif reference != (names, intro):
                raise ValueError(f"Class/task mismatch across arms: {dataset}")
            for label in initial:
                values = zip(*(data[label] for _, _, data in cells))
                yield {
                    "dataset": dataset, "dataset_dir": directory, "arm": arm,
                    "scope": "posthoc_descriptive", "n": len(SEEDS),
                    "seeds": ";".join(map(str, SEEDS)), "holdout_target_fpr_pct": 1,
                    "class_id": label, "class": names[label], "introduction_task": intro[label],
                    **{field: float(100 * mean(col)) for field, col in zip(fields, values)},
                }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section165-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = list(rows(args.section165_root))
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
