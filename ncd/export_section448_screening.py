"""Export the fixed-rule Section 448 development-screening table."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


DATASETS = ("UNSW-NB15", "CIC-IDS-2017")
DATASET_DIRS = {"UNSW-NB15": "unsw-nb15", "CIC-IDS-2017": "cic-ids-2017"}
ARMS = ("AR-Stateless", "V1a", "V1b", "V2", "V3")
SEEDS = (1, 2)
VARIANTS = ARMS[1:]
TARGET_FPR = "0.01"


def calibrated_forgetting(result: dict) -> float:
    """Match the project class-mean signed-forgetting definition at 1% FPR."""
    checkpoints = result["checkpoints"]
    tasks = result["tasks"]
    if len(checkpoints) != len(tasks) or len(tasks) < 2:
        raise ValueError("Incomplete task/checkpoint trajectory")
    recalls = []
    for checkpoint in checkpoints:
        per_class = checkpoint["operating_points"][TARGET_FPR]["per_class"]
        recalls.append({int(row["class_id"]): float(row["recall"]) for row in per_class})
    terms = []
    for introduction, task in enumerate(tasks[:-1]):
        for class_id in task:
            class_id = int(class_id)
            if any(class_id not in recalls[i] for i in range(introduction, len(tasks))):
                raise ValueError("Class missing from its calibrated trajectory")
            best_pre_final = max(recalls[i][class_id]
                                 for i in range(introduction, len(tasks) - 1))
            terms.append(best_pre_final - recalls[-1][class_id])
    if not terms:
        raise ValueError("No classes available for forgetting calculation")
    return statistics.mean(terms)


def cell_metrics(result: dict) -> dict:
    identity = result["identity"]
    if (not result.get("completed") or not result.get("scientific_result")
            or identity.get("engineering_defaults", {}).get("kmeans_n_init") != "auto"
            or identity.get("effective_config", {}).get("cpu_threads") != 2):
        raise ValueError("Cell is incomplete or does not match n_init=auto / 2-thread screening")
    final = result["checkpoints"][-1]["operating_points"][TARGET_FPR]
    return {
        "dataset": identity["dataset"],
        "seed": int(identity["seed"]),
        "arm": identity["method"],
        "source_commit": identity["git_commit"],
        "final_bacc_pp": float(final["balanced_accuracy"]) * 100,
        "forgetting_pp": calibrated_forgetting(result) * 100,
        "test_fpr_pp": float(final["normal_fpr"]) * 100,
    }


def aggregate_cells(cells: list[dict], expected: int = 2) -> dict:
    values = {key: [cell[key] for cell in cells]
              for key in ("final_bacc_pp", "forgetting_pp", "test_fpr_pp")}
    out = {"n_complete": len(cells), "n_expected": expected,
           "missing_seeds": ";".join(str(seed) for seed in SEEDS
                                     if seed not in {cell["seed"] for cell in cells})}
    for key, samples in values.items():
        out[f"mean_{key}"] = statistics.mean(samples) if samples else None
        out[f"sample_sd_{key}"] = statistics.stdev(samples) if len(samples) > 1 else None
    out["source_commit"] = ";".join(sorted({cell["source_commit"] for cell in cells}))
    return out


def fixed_rule(aggregates: dict) -> dict:
    baseline = {ds: aggregates[ds]["AR-Stateless"] for ds in DATASETS}
    if any(row["n_complete"] != 2 for row in baseline.values()):
        raise ValueError("Both AR-Stateless seeds are required on each dataset")
    eligible = []
    outcome = {}
    for arm in VARIANTS:
        if any(aggregates[ds][arm]["n_complete"] != 2 for ds in DATASETS):
            outcome[arm] = "INCOMPLETE"
            continue
        d_unsw_bacc = (aggregates["UNSW-NB15"][arm]["mean_final_bacc_pp"]
                       - baseline["UNSW-NB15"]["mean_final_bacc_pp"])
        d_cic_bacc = (aggregates["CIC-IDS-2017"][arm]["mean_final_bacc_pp"]
                      - baseline["CIC-IDS-2017"]["mean_final_bacc_pp"])
        d_unsw_forgetting = (aggregates["UNSW-NB15"][arm]["mean_forgetting_pp"]
                             - baseline["UNSW-NB15"]["mean_forgetting_pp"])
        if d_unsw_forgetting <= -5 and d_unsw_bacc >= -1 and d_cic_bacc >= -1:
            eligible.append((d_unsw_forgetting, arm))
        else:
            outcome[arm] = "FAIL_FIXED_RULE"
    eligible.sort(key=lambda item: (item[0], item[1]))
    for rank, (_, arm) in enumerate(eligible[:2], start=1):
        outcome[arm] = f"ADVANCE_{rank}"
    for _, arm in eligible[2:]:
        outcome[arm] = "ELIGIBLE_NOT_ADVANCED"
    return outcome


def build_rows(results_root: Path) -> list[dict]:
    cells = {ds: {arm: [] for arm in ARMS} for ds in DATASETS}
    source_commits = set()
    for dataset in DATASETS:
        for arm in ARMS:
            for seed in SEEDS:
                path = (results_root / "screen" / DATASET_DIRS[dataset] / arm
                        / f"seed{seed}" / "cell" / "result.json")
                if not path.exists():
                    continue
                result = json.loads(path.read_text(encoding="utf-8"))
                cell = cell_metrics(result)
                if (cell["dataset"], cell["arm"], cell["seed"]) != (dataset, arm, seed):
                    raise ValueError(f"Cell identity/path mismatch: {path}")
                cells[dataset][arm].append(cell)
                source_commits.add(cell["source_commit"])
    if len(source_commits) != 1:
        raise ValueError(f"Expected one pinned source commit, found {sorted(source_commits)}")
    aggregates = {ds: {arm: aggregate_cells(cells[ds][arm]) for arm in ARMS}
                  for ds in DATASETS}
    decisions = fixed_rule(aggregates)
    baseline = {ds: aggregates[ds]["AR-Stateless"] for ds in DATASETS}
    rows = []
    for ds in DATASETS:
        for arm in ARMS:
            row = {"dataset": ds, "arm": arm, **aggregates[ds][arm],
                   "delta_final_bacc_vs_ar_pp": None,
                   "delta_forgetting_vs_ar_pp": None,
                   "fixed_rule_outcome": decisions.get(arm, "BASELINE")}
            if arm != "AR-Stateless" and row["n_complete"] == 2:
                row["delta_final_bacc_vs_ar_pp"] = (
                    row["mean_final_bacc_pp"] - baseline[ds]["mean_final_bacc_pp"])
                row["delta_forgetting_vs_ar_pp"] = (
                    row["mean_forgetting_pp"] - baseline[ds]["mean_forgetting_pp"])
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("paper_aacil/results/section448"))
    parser.add_argument("--output", type=Path,
                        default=Path("paper_aacil/results/section448/screening_summary.csv"))
    args = parser.parse_args()
    rows = build_rows(args.results_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for arm, result in sorted({row["arm"]: row["fixed_rule_outcome"] for row in rows}.items()):
        print(f"{arm}: {result}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
