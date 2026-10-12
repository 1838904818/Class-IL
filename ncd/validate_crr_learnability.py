"""Apply the Section 13.3 Replay learnability gate before five-seed runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(result_path: Path) -> dict:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload["rows"]
        if row["method"] == "replay"
        and row["benchmark"] == "necessity"
        and float(row["k"]) >= 2.0
    ]
    if not rows:
        raise ValueError("no Replay necessity rows with k >= 2 were found")
    checks = []
    for row in rows:
        d_recall = float(row["d_recall"])
        c_recall = float(row["c_recall"])
        pair_confusion = float(row["pair_confusion_rate"])
        dependency_change = float(row["c_f2_dependency_change"])
        checks.append(
            {
                "dataset": row["dataset"],
                "seed": int(row["seed"]),
                "k": float(row["k"]),
                "d_recall_ge_0_5": d_recall >= 0.5,
                "pair_confusion_lt_0_5": pair_confusion < 0.5,
                "d_recall": d_recall,
                "c_recall": c_recall,
                "pair_confusion_rate": pair_confusion,
                "c_f2_dependency_change": dependency_change,
            }
        )
    datasets = sorted({str(row["dataset"]) for row in rows})
    expected = sorted(str(value) for value in payload["effective_config"]["datasets"])
    dependency_checks = []
    for dataset in datasets:
        dataset_items = [item for item in checks if item["dataset"] == dataset]
        max_k = max(float(item["k"]) for item in dataset_items)
        values = [
            float(item["c_f2_dependency_change"])
            for item in dataset_items
            if float(item["k"]) == max_k
        ]
        mean_change = sum(values) / len(values)
        dependency_checks.append(
            {
                "dataset": dataset,
                "k": max_k,
                "mean_c_f2_dependency_change_at_max_k": mean_change,
                "c_f2_dependency_increased": mean_change > 0.0,
            }
        )
    passed = datasets == expected and all(
        item[criterion]
        for item in checks
        for criterion in (
            "d_recall_ge_0_5",
            "pair_confusion_lt_0_5",
        )
    ) and all(item["c_f2_dependency_increased"] for item in dependency_checks)
    metadata = [
        item
        for item in payload.get("benchmark_metadata", [])
        if item.get("selection") == "random" and float(item.get("k", 0.0)) >= 2.0
    ]
    return {
        "schema_version": 1,
        "source_result": str(result_path),
        "git_commit": payload["git_commit"],
        "gate": "section13.3",
        "criteria": {
            "d_recall": ">= 0.5",
            "pair_confusion_rate": "< 0.5",
            "c_f2_dependency_change_at_max_k": "> 0 within each dataset",
            "scope": "Replay rows with k >= 2; d/confusion per row, dependency at the largest registered shift",
        },
        "datasets": datasets,
        "checks": checks,
        "dependency_checks": dependency_checks,
        "feature_distributions": metadata,
        "passed": passed,
        "decision": "CONTINUE" if passed else "STOP_AND_REPAIR_BENCHMARK",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    verdict = validate(args.result.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"decision": verdict["decision"], "passed": verdict["passed"]}))
    if not verdict["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
