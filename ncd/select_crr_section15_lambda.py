"""Select each Section 15 method's lambda on seeds 5 and 6 only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REGULARIZED_METHODS = (
    "icarl",
    "replay_lwf",
    "kd_perturbed",
    "rcl_tab",
    "global_ir",
    "oldpair_kd",
    "crr",
)


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("selection metric has no rows")
    return sum(values) / len(values)


def select(paths: list[Path], method: str) -> dict:
    if method not in REGULARIZED_METHODS:
        raise ValueError(f"unsupported regularized method: {method}")
    candidates: list[dict] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["effective_config"]
        if sorted(int(seed) for seed in config["seeds"]) != [5, 6]:
            raise ValueError(f"lambda selection may use only seeds 5 and 6: {path}")
        if set(config["methods"]) != {"replay", method}:
            raise ValueError(f"calibration must contain replay and {method}: {path}")
        rows = [
            row
            for row in payload["rows"]
            if row["method"] == method
            and row["benchmark"] == "necessity"
            and float(row["k"]) in {2.0, 3.0}
        ]
        if len(rows) != 4:
            raise ValueError(f"expected four {method} calibration rows: {path}")
        value = float(config["regularization"])
        cost = _mean([float(row["cost_vs_replay"]) for row in rows])
        candidates.append(
            {
                "lambda": value,
                "mean_cd_balanced_accuracy": _mean(
                    [float(row["cd_balanced_accuracy"]) for row in rows]
                ),
                "mean_macro_f1": _mean([float(row["macro_f1"]) for row in rows]),
                "mean_cost_vs_replay": cost,
                "cost_feasible": cost <= 3.0,
            }
        )
    if len({item["lambda"] for item in candidates}) != len(candidates):
        raise ValueError("duplicate lambda candidate")
    feasible = [item for item in candidates if item["cost_feasible"]]
    pool = feasible or candidates
    chosen = min(
        pool,
        key=lambda item: (
            -item["mean_cd_balanced_accuracy"],
            -item["mean_macro_f1"],
            item["lambda"],
        ),
    )
    return {
        "method": method,
        "selection_seeds": [5, 6],
        "selection_strengths": {"necessity_k": [2.0, 3.0]},
        "selection_rule": (
            "cost<=3x when feasible, then highest mean c/d balanced accuracy, "
            "then highest macro-F1, then smaller lambda"
        ),
        "selected_lambda": chosen["lambda"],
        "cost_feasible_candidate_found": bool(feasible),
        "candidates": sorted(candidates, key=lambda item: item["lambda"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--method", required=True, choices=REGULARIZED_METHODS)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = select(args.inputs, args.method)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"method": args.method, "selected_lambda": result["selected_lambda"]}))


if __name__ == "__main__":
    main()
