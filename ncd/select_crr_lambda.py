"""Select CRR lambda on the preregistered calibration seeds 5 and 6 only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _rank(values: dict[float, float]) -> dict[float, int]:
    ordered = sorted(values, key=lambda key: (values[key], key))
    return {key: index for index, key in enumerate(ordered)}


def select(paths: list[Path]) -> dict:
    candidates: dict[float, dict] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["effective_config"]
        if sorted(int(seed) for seed in config["seeds"]) != [5, 6]:
            raise ValueError(f"lambda selection may use only seeds 5 and 6: {path}")
        if set(config["methods"]) != {"replay", "crr"}:
            raise ValueError(f"calibration must contain only replay and CRR: {path}")
        value = float(config["regularization"])
        crr = [row for row in payload["rows"] if row["method"] == "crr"]
        replay = [row for row in payload["rows"] if row["method"] == "replay"]
        replay_index = {
            (row["dataset"], row["seed"], row["benchmark"], row.get("rho"), row.get("k")): row
            for row in replay
        }
        hijack = [
            max(0.0, float(row["marker_recall_drop"]))
            for row in crr
            if row["benchmark"] == "hijack" and float(row["rho"]) >= 0.7
        ]
        necessity = [
            float(row["pair_confusion_rate"])
            for row in crr
            if row["benchmark"] == "necessity" and float(row["k"]) >= 2.0
        ]
        recall_failures = 0
        for row in crr:
            if row["benchmark"] != "necessity" or float(row["k"]) < 2.0:
                continue
            key = (row["dataset"], row["seed"], row["benchmark"], row.get("rho"), row.get("k"))
            base = replay_index[key]
            recall_failures += int(float(row["c_recall"]) < float(base["c_recall"]))
            recall_failures += int(float(row["d_recall"]) < float(base["d_recall"]))
        cost = _mean([float(row["cost_vs_replay"]) for row in crr])
        candidates[value] = {
            "lambda": value,
            "mean_marker_recall_drop_rho_ge_0_7": _mean(hijack),
            "mean_pair_confusion_k_ge_2": _mean(necessity),
            "recall_failures_vs_replay": recall_failures,
            "mean_cost_vs_replay": cost,
            "feasible": recall_failures == 0 and cost <= 3.0,
        }

    marker_rank = _rank({key: item["mean_marker_recall_drop_rho_ge_0_7"] for key, item in candidates.items()})
    confusion_rank = _rank({key: item["mean_pair_confusion_k_ge_2"] for key, item in candidates.items()})
    for value, item in candidates.items():
        item["equal_weight_rank_sum"] = marker_rank[value] + confusion_rank[value]
    feasible = [item for item in candidates.values() if item["feasible"]]
    pool = feasible or list(candidates.values())
    chosen = min(
        pool,
        key=lambda item: (
            0 if item["feasible"] else 1,
            item["recall_failures_vs_replay"],
            max(0.0, item["mean_cost_vs_replay"] - 3.0),
            item["equal_weight_rank_sum"],
            item["lambda"],
        ),
    )
    return {
        "selection_seeds": [5, 6],
        "selection_rule": "feasible recall/cost first, then equal-weight hijack/confusion rank sum, then smaller lambda",
        "selected_lambda": chosen["lambda"],
        "feasible_candidate_found": bool(feasible),
        "candidates": [candidates[key] for key in sorted(candidates)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = select(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"selected_lambda": result["selected_lambda"]}))


if __name__ == "__main__":
    main()
