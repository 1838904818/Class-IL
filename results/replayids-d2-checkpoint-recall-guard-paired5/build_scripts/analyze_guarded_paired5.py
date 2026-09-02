from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

from scipy import stats


SEEDS = (1, 2, 3, 4, 42)
PRIMARY = ("final_macro_f1", "average_forgetting")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def result_path(root: Path, seed: int) -> Path:
    candidates = (
        root / f"seed_{seed}" / f"result_seed_{seed}.json",
        root / "guarded_checkpoint" / f"seed_{seed}" / f"result_seed_{seed}.json",
        root / "last_epoch" / f"seed_{seed}" / f"result_seed_{seed}.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing result for seed {seed} under {root}")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def final_view(result: dict) -> dict:
    return result["checkpoints"][-1]["views"]["official"]["arms"]["joint_cap3000"]


def metrics(result: dict) -> dict[str, float]:
    view = final_view(result)
    summary = result["summary"]["views"]["official"]["joint_cap3000"]
    binary = view["binary_detection"]
    return {
        "average_task_accuracy": float(summary["average_task_accuracy"]),
        "average_forgetting": float(summary["average_forgetting"]),
        "final_overall_accuracy": float(view["accuracy"]),
        "final_macro_f1": float(view["macro_f1"]),
        "final_balanced_accuracy": float(view["balanced_accuracy"]),
        "final_attack_detection_recall": float(binary["attack_detection_recall"]),
        "final_benign_false_positive_rate": float(binary["benign_false_positive_rate"]),
    }


def holm_two(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        current = min(1.0, (count - rank) * value)
        running = max(running, current)
        adjusted[name] = running
    return adjusted


def paired_summary(left: list[float], right: list[float]) -> dict:
    delta = [r - l for l, r in zip(left, right)]
    delta_sd = stdev(delta)
    sem = delta_sd / math.sqrt(len(delta))
    critical = stats.t.ppf(0.975, df=len(delta) - 1)
    t_result = stats.ttest_rel(right, left)
    try:
        w_result = stats.wilcoxon(right, left, alternative="two-sided", method="exact")
        w_stat, w_p = float(w_result.statistic), float(w_result.pvalue)
    except ValueError:
        w_stat, w_p = 0.0, 1.0
    return {
        "baseline_mean": mean(left),
        "baseline_sample_std": stdev(left),
        "candidate_mean": mean(right),
        "candidate_sample_std": stdev(right),
        "paired_delta_per_seed": delta,
        "paired_delta_mean": mean(delta),
        "paired_delta_sample_std": delta_sd,
        "paired_delta_95ci": [mean(delta) - critical * sem, mean(delta) + critical * sem],
        "paired_t_statistic": float(t_result.statistic),
        "paired_t_pvalue_raw": float(t_result.pvalue),
        "wilcoxon_statistic": w_stat,
        "wilcoxon_pvalue_raw": w_p,
        "cohen_dz": 0.0 if delta_sd == 0 else mean(delta) / delta_sd,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_results = []
    candidate_results = []
    files = []
    for seed in SEEDS:
        baseline_path = result_path(args.baseline_root, seed)
        candidate_path = result_path(args.candidate_root, seed)
        baseline = load(baseline_path)
        candidate = load(candidate_path)
        if int(baseline["seed"]) != seed or int(candidate["seed"]) != seed:
            raise ValueError(f"seed identity mismatch for {seed}")
        left_pc = final_view(baseline)["per_class"]
        right_pc = final_view(candidate)["per_class"]
        left_id = [(x["class_id"], x["class_name"], x["support"]) for x in left_pc]
        right_id = [(x["class_id"], x["class_name"], x["support"]) for x in right_pc]
        if left_id != right_id:
            raise ValueError(f"official-test class/support mismatch for seed {seed}")
        baseline_results.append(baseline)
        candidate_results.append(candidate)
        files.extend(
            [
                {"arm": "baseline", "seed": seed, "path": str(baseline_path), "sha256": sha256(baseline_path)},
                {"arm": "candidate", "seed": seed, "path": str(candidate_path), "sha256": sha256(candidate_path)},
            ]
        )

    left_metrics = [metrics(item) for item in baseline_results]
    right_metrics = [metrics(item) for item in candidate_results]
    aggregate = {}
    for name in left_metrics[0]:
        aggregate[name] = paired_summary(
            [row[name] for row in left_metrics], [row[name] for row in right_metrics]
        )
    adjusted = holm_two({name: aggregate[name]["paired_t_pvalue_raw"] for name in PRIMARY})
    for name, value in adjusted.items():
        aggregate[name]["paired_t_pvalue_holm_primary_two"] = value

    per_class = []
    identities = final_view(baseline_results[0])["per_class"]
    for index, identity in enumerate(identities):
        row = {"class_id": identity["class_id"], "class_name": identity["class_name"]}
        for field in ("precision", "recall", "f1"):
            left = [float(final_view(item)["per_class"][index][field]) for item in baseline_results]
            right = [float(final_view(item)["per_class"][index][field]) for item in candidate_results]
            row[field] = paired_summary(left, right)
        row["support_per_seed"] = [
            int(final_view(item)["per_class"][index]["support"]) for item in baseline_results
        ]
        per_class.append(row)

    output = {
        "schema_version": 1,
        "analysis_role": "independent_local_recomputation",
        "seeds": list(SEEDS),
        "route": "official/joint_cap3000",
        "primary_metrics": list(PRIMARY),
        "metrics": aggregate,
        "per_class": per_class,
        "input_files": files,
        "limitations": [
            "five paired seeds share one fixed data split",
            "n=5 gives limited exact Wilcoxon power and wide uncertainty intervals",
            "descriptive safety metrics are not additional confirmatory endpoints",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(f"WROTE {args.output}")
    print(f"SHA256 {sha256(args.output)}")


if __name__ == "__main__":
    main()
