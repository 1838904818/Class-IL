"""Section 20.2B: existing S1 within-transition, across-old-class Spearman.

No training/model imports or historical writes. Run from the repository root:
python -I ncd/results/dgc/day1/analyze_s1_within_transition.py
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SEEDS = (1, 2, 3, 4, 42)
METHODS = ("replay", "icarl", "replay_lwf", "kd_perturbed", "rcl_tab",
           "global_ir", "oldpair_kd", "crr")
DEFAULT_INPUT = ROOT / "ncd/results/exp_h/section17/registered/exp_h_result.json"


def ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        for index in order[start:stop]:
            result[index] = (start + stop + 1) / 2
        start = stop
    return result


def spearman(x, negative_y):
    """Undefined n<2/constant groups return None, never a numeric convention."""
    if len(x) < 2:
        return None, "fewer_than_two_classes"
    a, b = ranks(x), ranks(negative_y)
    a, b = a - a.mean(), b - b.mean()
    if not np.any(a) or not np.any(b):
        return None, "constant_rank_variable"
    return float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)), "identified"


def summarize(groups, replicates=2000, bootstrap_seed=1705):
    by_seed = {seed: [g for g in groups if g["seed"] == seed] for seed in SEEDS}
    identified = [g for g in groups if g["rho"] is not None]
    per_seed = []
    for seed, subset in by_seed.items():
        values = [g["rho"] for g in subset if g["rho"] is not None]
        per_seed.append({"seed": seed, "raw_groups": len(subset),
                         "identified_groups": len(values),
                         "undefined_groups": len(subset) - len(values),
                         "eligible_class_units": sum(g["eligible_class_count"] for g in subset),
                         "mean_rho": float(np.mean(values)) if values else None})
    rng = np.random.default_rng(bootstrap_seed)
    draws = []
    for _ in range(replicates):
        sampled = rng.choice(SEEDS, size=len(SEEDS), replace=True)
        # Duplicate sampled seeds duplicate their entire group collections. Each
        # identifiable group has weight one; this is not equal-weight seed means.
        values = [g["rho"] for seed in sampled for g in by_seed[int(seed)] if g["rho"] is not None]
        draws.append(float(np.mean(values)) if values else None)
    finite = [v for v in draws if v is not None]
    mean = float(np.mean([g["rho"] for g in identified])) if identified else None
    ci = [float(v) for v in np.quantile(finite, [.025, .975])] if finite else None
    if mean is None or len(finite) != replicates:
        verdict = "pending_unidentified"
    elif mean <= .1 or ci[0] <= 0 <= ci[1]:
        verdict = "veto"
    else:
        verdict = "passes_B_only"
    return {
        "raw_groups": len(groups), "identified_groups": len(identified),
        "undefined_reasons": dict(Counter(g["status"] for g in groups if g["rho"] is None)),
        "eligible_class_count_distribution": dict(sorted(Counter(g["eligible_class_count"] for g in groups).items())),
        "eligible_class_units": sum(g["eligible_class_count"] for g in groups),
        "class_units_in_identified_groups": sum(g["eligible_class_count"] for g in identified),
        "mean_rho": mean, "ci95": ci, "per_seed": per_seed,
        "bootstrap": {"replicates": replicates, "seed": bootstrap_seed,
                      "clusters": list(SEEDS), "valid_draws": len(finite),
                      "scheme": "resample five whole seed clusters; equally average identified groups"},
        "veto_mean_le_0_1": mean is not None and mean <= .1,
        "veto_ci_contains_zero": ci is not None and ci[0] <= 0 <= ci[1],
        "verdict": verdict,
    }, draws


def analyze(rows, floor_recall=.2):
    selected = [row for row in rows if row["setting"] == "S1"]
    if len(selected) != 440:
        raise ValueError("Expected all 440 registered S1 class-transition observations")
    grouped = defaultdict(list)
    seen = set()
    for row in selected:
        key = (row["method"], row["seed"], row["transition"], row["class_id"])
        if key in seen:
            raise ValueError(f"Duplicate S1 unit: {key}")
        seen.add(key)
        for name in ("old_pair_pi_drift", "shift_recall_before", "shift_recall_after"):
            if not np.isfinite(row[name]):
                raise ValueError(f"Nonfinite {name}: {key}")
        if any(not 0 <= row[name] <= 1 for name in ("shift_recall_before", "shift_recall_after")):
            raise ValueError(f"Recall outside [0,1]: {key}")
        grouped[key[:3]].append(row)
    expected = {(method, seed, transition) for method in METHODS for seed in SEEDS for transition in (1, 2, 3)}
    if set(grouped) != expected:
        raise ValueError("Missing or unexpected method/seed/transition groups")
    groups, class_rows = [], []
    for (method, seed, transition), raw in sorted(grouped.items()):
        raw = sorted(raw, key=lambda row: row["class_id"])
        if len(raw) != {1: 2, 2: 4, 3: 5}[transition]:
            raise ValueError("Unexpected raw class count for the registered transition")
        eligible = [row for row in raw if floor_recall is None or row["shift_recall_before"] >= floor_recall]
        x = [row["old_pair_pi_drift"] for row in eligible]
        drops = [row["shift_recall_before"] - row["shift_recall_after"] for row in eligible]
        rho, status = spearman(x, drops)
        groups.append({"method": method, "seed": seed, "transition": transition,
                       "raw_class_count": len(raw), "eligible_class_count": len(eligible),
                       "floor_excluded": len(raw) - len(eligible),
                       "raw_class_names": [row["class_name"] for row in raw],
                       "eligible_class_names": [row["class_name"] for row in eligible],
                       "x": x, "negative_y": drops, "rho": rho, "status": status})
        for row in raw:
            class_rows.append({"method": method, "seed": seed, "transition": transition,
                               "class_id": row["class_id"], "class_name": row["class_name"],
                               "x_old_pair_pi_drift": row["old_pair_pi_drift"],
                               "shift_recall_before": row["shift_recall_before"],
                               "shift_recall_after": row["shift_recall_after"],
                               "negative_y": row["shift_recall_before"] - row["shift_recall_after"],
                               "eligible": floor_recall is None or row["shift_recall_before"] >= floor_recall})
    summary, draws = summarize(groups)
    summary["raw_class_units"] = len(selected)
    summary["floor_excluded_class_units"] = sum(not row["eligible"] for row in class_rows)
    summary["eligible_class_composition"] = dict(sorted(Counter(row["class_name"] for row in class_rows if row["eligible"]).items()))
    return summary, groups, class_rows, draws


def self_checks():
    assert spearman([1], [2]) == (None, "fewer_than_two_classes")
    assert spearman([1, 1], [1, 2]) == (None, "constant_rank_variable")
    assert spearman([1, 2], [2, 2]) == (None, "constant_rank_variable")
    assert abs(spearman([1, 2, 3], [3, 2, 1])[0] + 1) < 1e-12
    assert abs(spearman([1, 1, 2], [2, 2, 3])[0] - 1) < 1e-12
    assert .2 >= .2 and not (.199999 >= .2)
    # Each group, not each class, gets equal weight; every seed stays a cluster.
    fake = [{"seed": s, "rho": 1.0 if s == 1 else -1.0,
             "eligible_class_count": 2, "status": "identified"} for s in SEEDS]
    fake.append({"seed": 1, "rho": 1.0, "eligible_class_count": 5, "status": "identified"})
    summary, draws = summarize(fake, replicates=20)
    assert abs(summary["mean_rho"] + 1/3) < 1e-12 and len(draws) == 20


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                             for key, value in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    self_checks()
    raw = args.input.read_bytes()
    payload = json.loads(raw)
    if payload.get("smoke") is not False:
        raise ValueError("Formal, not smoke, observations are required")
    summary, groups, class_rows, draws = analyze(payload["rows"])
    secondary, _, _, _ = analyze(payload["rows"], floor_recall=None)
    source_label = args.input.resolve().relative_to(ROOT).as_posix()
    output = {"question": "WORK_ORDER 20.2B within method/seed/transition across-old-class premise",
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_experiment_commit": payload["git_commit"], "source": source_label,
              "source_sha256": hashlib.sha256(raw).hexdigest(),
              "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "command": "python -I ncd/results/dgc/day1/analyze_s1_within_transition.py",
              "log_path": (args.output_dir.resolve() / "s1_within_transition.log").relative_to(ROOT).as_posix(),
              "numpy_version": np.__version__,
              "config": {"setting": "S1", "floor_exclude_before_lt": .2,
                         "x": "old_pair_pi_drift", "negative_y": "shift_recall_before - shift_recall_after",
                         "group_key": ["method", "seed", "transition"],
                         "average": "arithmetic mean of identified group Spearman correlations",
                         "bootstrap_replicates": 2000, "bootstrap_seed": 1705},
              "summary": summary, "groups": groups,
              "secondary_allrows": {
                  "interpretation": "Previously requested all-row sensitivity without the recall floor; does not replace the primary floor=0.2 verdict.",
                  "floor_exclude_before_lt": None, "summary": secondary},
              "caveats": ["Floor eligibility removes classes and can leave singleton groups.",
                          "Undefined group correlations are null, never 0 or 1.",
                          "Two-class nonconstant groups yield +1 or -1; report their prevalence.",
                          "Five seed clusters limit uncertainty precision.",
                          "Observed old-pair PI ranks are not a direct validation of the proposed online JS signal.",
                          "This is an existing-data premise test, not a causal mechanism test."]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "s1_within_transition.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "s1_within_transition_groups.csv", groups)
    write_csv(args.output_dir / "s1_within_transition_classes.csv", class_rows)
    write_csv(args.output_dir / "s1_within_transition_bootstrap.csv", [{"draw": i+1, "mean_rho": v} for i, v in enumerate(draws)])
    if args.input.read_bytes() != raw:
        raise RuntimeError("Historical input changed during this analysis")
    log = json.dumps({"primary_floor_0_2": summary, "secondary_allrows": secondary},
                     ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (args.output_dir / "s1_within_transition.log").write_text(log, encoding="utf-8")
    print(log, end="")


if __name__ == "__main__":
    main()
