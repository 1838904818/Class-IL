"""Export the registered §448 confirmation from all 45 published result files.

Input root is paper_aacil/results/section448/confirm. CSE's on-disk alias
cic-ids-2018 is CSE-CIC-IDS2018, external to the confirmation verdict.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from fractions import Fraction
from pathlib import Path

from scipy.stats import ttest_1samp


DATASETS = {
    "UNSW-NB15": "unsw-nb15",
    "CIC-IDS-2017": "cic-ids-2017",
    "CSE-CIC-IDS2018": "cic-ids-2018",
}
EXTERNAL = "CSE-CIC-IDS2018"
ARMS = ("AR-Stateless", "V1a", "V1b")
SEEDS = tuple(range(15, 20))
TARGET = "0.01"
ALPHA = 0.05
MARGIN_PP = 1


def cm_metrics(view):
    labels = tuple(view["confusion_labels"])
    cm = view["confusion_matrix"]
    if len(labels) != len(set(labels)) or len(cm) != len(labels) or any(
        len(row) != len(labels) or any(type(x) is not int or x < 0 for x in row)
        for row in cm
    ):
        raise ValueError("Invalid confusion matrix")
    support = {label: sum(row) for label, row in zip(labels, cm)}
    recall = {label: Fraction(cm[i][i], support[label])
              for i, label in enumerate(labels) if support[label]}
    bacc = statistics.mean(recall.values())
    fpr = 1 - recall[0]
    if sum(support.values()) != view["evaluated_rows"] or not all(
        math.isclose(float(actual), view[field], rel_tol=0, abs_tol=1e-12)
        for field, actual in (("balanced_accuracy", bacc), ("normal_fpr", fpr))
    ):
        raise ValueError("Stored operating metrics disagree with the matrix")
    return recall, support, bacc, fpr


def read_cell(root, dataset, arm, seed):
    path = root / DATASETS[dataset] / arm / f"seed{seed}" / "cell/result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    identity = result["identity"]
    if not (result["completed"] and result["scientific_result"]) or (
        identity["dataset"], identity["method"], identity["seed"]
    ) != (dataset, arm, seed):
        raise ValueError(f"Wrong or incomplete cell: {path}")
    tasks, checkpoints = result["tasks"], result["checkpoints"]
    if len(tasks) != len(checkpoints) or len(tasks) < 2:
        raise ValueError(f"Incomplete trajectory: {path}")
    recall, supports, calibration = [], [], []
    for i, cp in enumerate(checkpoints):
        cal = cp["calibration"][TARGET]
        if cp["task_index"] != i or cal["target_fpr"] != float(TARGET):
            raise ValueError(f"Wrong stage or calibration target: {path}")
        recalls, support, bacc, fpr = cm_metrics(cp["operating_points"][TARGET])
        if set(recalls) != set().union(*map(set, tasks[: i + 1])):
            raise ValueError(f"Missing class support: {path} task {i}")
        recall.append(recalls)
        supports.append(support)
        calibration.append((cal["row_ids_sha256"], cal["calibration_rows"]))
    forgetting = statistics.mean(
        max(stage[label] for stage in recall[i:-1]) - recall[-1][label]
        for i, task in enumerate(tasks[:-1]) for label in task
    )
    pairing = (identity["data_binding"], result["class_order"], tasks, supports,
               calibration)
    return (bacc * 100, forgetting * 100, fpr * 100), pairing


def upper_t(values):
    if all(x == values[0] for x in values):
        raise ValueError("Paired t statistic undefined: zero variance")
    return float(ttest_1samp(list(map(float, values)), 0, alternative="greater").pvalue)


def upper_sign_flip(values):
    observed = sum(values)
    signs = itertools.product((-1, 1), repeat=len(values))
    return sum(sum(s * v for s, v in zip(pattern, values)) >= observed
               for pattern in signs) / 2 ** len(values)


def holm(pvalues):
    out = [0.0] * len(pvalues)
    bound = 0.0
    for rank, idx in enumerate(sorted(range(len(pvalues)), key=pvalues.__getitem__)):
        bound = max(bound, min(1.0, (len(pvalues) - rank) * pvalues[idx]))
        out[idx] = bound
    return out


def summary(root):
    cells = {}
    pairing = {}
    for dataset, arm, seed in itertools.product(DATASETS, ARMS, SEEDS):
        metrics, match = read_cell(root, dataset, arm, seed)
        if pairing.setdefault((dataset, seed), match) != match:
            raise ValueError(f"Unmatched comparison: {(dataset, arm, seed)}")
        cells[dataset, arm, seed] = metrics
    rows = []
    tests = []
    for dataset, arm in itertools.product(DATASETS, ARMS):
        series = [cells[dataset, arm, seed] for seed in SEEDS]
        baseline = [cells[dataset, ARMS[0], seed] for seed in SEEDS]
        row = {
            "dataset": dataset, "dataset_dir": DATASETS[dataset],
            "scope": "external_descriptive" if dataset == EXTERNAL else "confirmation",
            "arm": arm, "n": len(SEEDS), "seeds": ";".join(map(str, SEEDS)),
            "holdout_target_fpr_pct": float(TARGET) * 100,
        }
        for key, values in zip(("final_bacc_pct", "signed_forgetting_pp", "test_fpr_pct"),
                               zip(*series)):
            row[f"mean_{key}"] = float(statistics.mean(values))
            row[f"sample_sd_{key}"] = statistics.stdev(map(float, values))
        row.update(dict.fromkeys((
            "F_mean_forgetting_reduction_pp", "F_p_raw", "F_p_holm_6",
            "F_p_sign_flip", "F_pass", "N_mean_final_bacc_delta_pp",
            "N_margin_pp", "N_p_raw", "N_pass", "variant_confirmed",
        )))
        if arm != ARMS[0]:
            f = [b[1] - a[1] for a, b in zip(series, baseline)]
            n = [a[0] - b[0] for a, b in zip(series, baseline)]
            row.update(F_mean_forgetting_reduction_pp=float(statistics.mean(f)),
                       F_p_raw=upper_t(f), F_p_sign_flip=upper_sign_flip(f),
                       N_mean_final_bacc_delta_pp=float(statistics.mean(n)),
                       N_margin_pp=MARGIN_PP,
                       N_p_raw=upper_t([x + MARGIN_PP for x in n]))
            tests.append(row)
        rows.append(row)
    for row, adjusted in zip(tests, holm([row["F_p_raw"] for row in tests])):
        row.update(F_p_holm_6=adjusted, F_pass=adjusted < ALPHA,
                   N_pass=row["N_p_raw"] < ALPHA)
    for arm in ARMS[1:]:
        registered = [r for r in tests if r["arm"] == arm and r["dataset"] != EXTERNAL]
        for row in registered:
            row["variant_confirmed"] = all(r["F_pass"] and r["N_pass"] for r in registered)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = summary(args.confirm_root)
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
