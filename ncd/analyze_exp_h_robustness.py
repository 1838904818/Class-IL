"""Section 19.2: S1 robustness from immutable Section 17 rows; no training.

Average-rank X and -Y are separately residualized on an intercept, rank(Z),
and the specified additive categorical main effects. Every bootstrap draw
resamples five whole seed clusters, then recomputes ranks and dummy coding.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

METHODS = ("replay", "icarl", "replay_lwf", "kd_perturbed", "rcl_tab",
           "global_ir", "oldpair_kd", "crr")
SEEDS = (1, 2, 3, 4, 42)
METRIC = "old_pair_pi_drift"
FLOOR = 0.2
Z_ONLY = ()
ADJUSTED = ("method", "class_id")
NUMERIC = (METRIC, "shift_recall_before", "shift_recall_after",
           "id_recall_before", "id_recall_after")


def eligible_rows(rows):
    """Apply precisely the registered pre-transition recall < 0.2 exclusion."""
    for row in rows:
        if any(isinstance(row.get(k), bool) or not isinstance(row.get(k), (int, float)) or
               not math.isfinite(row[k]) for k in NUMERIC):
            raise ValueError("Missing or nonfinite statistic input")
        if any(not 0 <= row[k] <= 1 for k in NUMERIC[1:]):
            raise ValueError("Recall outside [0, 1]")
    return [r for r in rows if r["shift_recall_before"] >= FLOOR]


def partial_rank(rows, categorical=Z_ONLY):
    """Return a Pearson correlation of residual average ranks, or unidentified.

    Drop-first coding with an intercept spans the same additive-effects space
    as full dummies; absent levels are recomputed within each resampled set.
    """
    n = len(rows)
    if n < 3:
        return {"rho": None, "status": "unidentified", "row_count": n,
                "reason": "Fewer than three units"}
    values = np.asarray([[r[METRIC],
                          r["shift_recall_before"] - r["shift_recall_after"],
                          r["id_recall_after"] - r["id_recall_before"]]
                         for r in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite statistic input")
    ranks = np.column_stack([rankdata(values[:, i], method="average")
                             for i in range(3)])
    columns = [np.ones(n), ranks[:, 2]]
    names = ["intercept", "rank_Z"]
    references, levels_by_field = {}, {}
    for field in categorical:
        labels = np.asarray([str(r[field]) for r in rows])
        levels = sorted(set(labels))
        references[field] = levels[0]
        levels_by_field[field] = levels
        for level in levels[1:]:
            columns.append((labels == level).astype(float))
            names.append(f"{field}={level}")
    design = np.column_stack(columns)
    target = ranks[:, :2]
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    residuals = target - design @ coefficients
    residuals -= residuals.mean(axis=0)
    norms = np.linalg.norm(residuals, axis=0)
    scales = np.linalg.norm(target - target.mean(axis=0), axis=0)
    meta = {"row_count": n, "design_columns": names, "design_rank": int(rank),
            "dummy_reference_levels": references, "categorical_levels": levels_by_field}
    if np.any(norms <= 1e-10 * np.maximum(scales, 1.0)):
        return {**meta, "rho": None, "status": "unidentified",
                "reason": "Constant or numerically zero residual ranks"}
    rho = float((residuals[:, 0] @ residuals[:, 1]) / np.prod(norms))
    return {**meta, "rho": float(np.clip(rho, -1, 1)), "status": "complete",
            "reason": None}


def cluster_indices(rows, selected_seeds):
    """Keep all method/transition/class rows whenever a seed is sampled."""
    labels = np.asarray([r["seed"] for r in rows])
    return np.concatenate([np.flatnonzero(labels == seed) for seed in selected_seeds])


def bootstrap_variants(rows, draws=2000, random_seed=1705):
    if draws < 2000:
        raise ValueError("Section 17.5/19.2 requires at least 2000 draws")
    if set(r["seed"] for r in rows) != set(SEEDS):
        raise ValueError("All five registered seed clusters must remain")
    points = {"z_only": partial_rank(rows, Z_ONLY),
              "method_class_adjusted": partial_rank(rows, ADJUSTED)}
    generator = np.random.default_rng(random_seed)
    records = []
    samples = {name: [] for name in points}
    for index in range(draws):
        chosen = generator.choice(np.asarray(SEEDS), size=5, replace=True)
        ids = cluster_indices(rows, chosen)
        resampled = [rows[i] for i in ids]
        record = {"draw": index + 1, "sampled_rows": len(ids),
                  **{f"seed_draw_{i+1}": int(seed) for i, seed in enumerate(chosen)}}
        for name, controls in (("z_only", Z_ONLY), ("method_class_adjusted", ADJUSTED)):
            value = partial_rank(resampled, controls)["rho"]
            samples[name].append(value)
            record[name + "_rho"] = value
        records.append(record)
    for name, point in points.items():
        valid = [v for v in samples[name] if v is not None and math.isfinite(v)]
        complete = point["rho"] is not None and len(valid) == draws
        point.update({
            "bootstrap_requested": draws, "bootstrap_valid": len(valid),
            "bootstrap_undefined": draws - len(valid), "bootstrap_seed": random_seed,
            "seed_cluster_count": 5,
            "ci95": np.quantile(valid, [0.025, 0.975]).tolist() if complete else None,
            "status": "complete" if complete else "pending",
        })
        if not complete:
            point["reason"] = "At least one estimate is unidentified; no conditional CI is substituted"
    return points, records


def descriptive_subsets(all_rows, retained):
    methods = []
    for method in METHODS:
        raw = [r for r in all_rows if r["method"] == method]
        subset = [r for r in retained if r["method"] == method]
        z = partial_rank(subset, Z_ONLY)
        classes = partial_rank(subset, ("class_id",))
        methods.append({
            "method": method, "raw_units": len(raw), "included_units": len(subset),
            "floor_excluded_units": len(raw) - len(subset),
            "old_class_count": len({r["class_id"] for r in subset}),
            "seed_cluster_count": len({r["seed"] for r in subset}),
            "z_only_rho": z["rho"], "z_only_status": z["status"],
            "class_adjusted_rho": classes["rho"], "class_adjusted_status": classes["status"],
        })
    classes = sorted({r["class_id"] for r in all_rows})
    class_names = {r["class_id"]: r["class_name"] for r in all_rows}
    leave_one = []
    for class_id in classes:
        subset = [r for r in retained if r["class_id"] != class_id]
        raw = [r for r in all_rows if r["class_id"] != class_id]
        base = partial_rank(subset, Z_ONLY)
        adjusted = partial_rank(subset, ADJUSTED)
        leave_one.append({
            "omitted_class_id": class_id, "omitted_class_name": class_names[class_id],
            "raw_units": len(raw), "included_units": len(subset),
            "floor_excluded_units": len(raw) - len(subset),
            "omitted_included_units": len(retained) - len(subset),
            "z_only_rho": base["rho"], "z_only_status": base["status"],
            "method_class_adjusted_rho": adjusted["rho"],
            "method_class_adjusted_status": adjusted["status"],
        })
    return methods, leave_one


def analyze(payload, reference, draws=2000, random_seed=1705):
    all_rows = [r for r in payload["rows"] if r["setting"] == "S1"]
    if payload.get("smoke") or payload["setting_metadata"]["S1"]["status"] != "complete":
        raise ValueError("Require the complete registered S1 evaluation, not smoke")
    keys = [(r["method"], r["seed"], r["transition"], r["class_id"]) for r in all_rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate analysis units")
    cells = {(r["method"], r["seed"]) for r in all_rows}
    if cells != {(m, s) for m in METHODS for s in SEEDS}:
        raise ValueError("Require eight registered methods by five seeds")
    templates = [{(r["transition"], r["class_id"]) for r in all_rows
                  if (r["method"], r["seed"]) == (m, s)} for m, s in sorted(cells)]
    if any(t != templates[0] for t in templates):
        raise ValueError("Transition/class units differ across method/seed cells")
    if len(templates[0]) != payload["setting_metadata"]["S1"]["expected_units_per_cell"]:
        raise ValueError("Expected S1 transition/class units are missing")
    retained = eligible_rows(all_rows)
    variants, bootstrap_rows = bootstrap_variants(retained, draws, random_seed)
    original = reference["settings"]["S1"]["unit_level"][METRIC]
    base = variants["z_only"]
    if base["status"] != "complete" or not math.isclose(base["rho"], original["rho"], abs_tol=1e-12):
        raise ValueError("Recomputed base estimate differs from Section 17")
    # This exact comparison is meaningful when using the frozen original draws.
    same_bootstrap = (draws == original["bootstrap_requested"] and
                      random_seed == original["bootstrap_seed"])
    if same_bootstrap and not np.allclose(base["ci95"], original["ci95"], rtol=0, atol=1e-12):
        raise ValueError("Recomputed base bootstrap interval differs from Section 17")
    methods, leave_one = descriptive_subsets(all_rows, retained)
    primary = variants["method_class_adjusted"]
    condition = (primary["rho"] >= 0.2 and primary["ci95"][0] > 0
                 if primary["status"] == "complete" else None)
    method_summary = {}
    for name in ("z_only", "class_adjusted"):
        values = [r[name + "_rho"] for r in methods]
        method_summary[name] = {
            "positive_count": sum(v is not None and v > 0 for v in values),
            "identified_count": sum(v is not None for v in values), "method_count": 8,
        }
    minima = {}
    for name in ("z_only", "method_class_adjusted"):
        values = [r[name + "_rho"] for r in leave_one]
        minima[name] = min(values) if all(v is not None for v in values) else None
    by_class = []
    for class_id in sorted({r["class_id"] for r in all_rows}):
        raw = [r for r in all_rows if r["class_id"] == class_id]
        included = [r for r in retained if r["class_id"] == class_id]
        by_class.append({"class_id": class_id, "class_name": raw[0]["class_name"],
                         "raw_units": len(raw), "floor_excluded_units": len(raw)-len(included),
                         "included_units": len(included)})
    result = {
        "work_order": "19.2", "setting": "S1", "metric": METRIC,
        "primary_adjusted": primary, "baseline_z_only": base,
        "s1_adjusted_condition_met": condition,
        "section19_confirmation": "Not adjudicated here; requires S4/S5 results",
        "section17_verdict_preserved": reference["verdict"],
        "sample_size": {"raw_units": len(all_rows), "floor_excluded_units": len(all_rows)-len(retained),
                        "included_units": len(retained), "method_seed_cells": 40,
                        "included_by_seed": dict(Counter(r["seed"] for r in retained)),
                        "by_class": by_class},
        "per_method": methods, "per_method_positive_counts": method_summary,
        "leave_one_class_out": leave_one, "leave_one_class_out_minimum": minima,
        "analysis_config": {
            "floor_recall": FLOOR, "floor_rule": "Exclude before < 0.2; include exactly 0.2",
            "bootstrap_replicates": draws, "bootstrap_seed": random_seed,
            "response": "shift_recall_before - shift_recall_after (-Y)",
            "rank_control": "id_recall_after - id_recall_before (Z)",
            "primary_controls": ["intercept", "rank(Z)", "method dummies", "old-class dummies"],
            "categorical_model": "Additive main effects; no method-by-class interaction",
            "ranking": "Average ties; recompute ranks and dummy coding within each draw/subset",
            "bootstrap": "Five seed clusters with replacement; preserve all rows within sampled seed",
            "ci": "95% percentile; no row-wise bootstrap and no omitted undefined draws",
            "per_method_variants": ["rank(Z)", "rank(Z) + old-class dummies"],
            "leave_one_class_variants": ["rank(Z)", "rank(Z) + method + old-class dummies"],
            "descriptive_subset_ci": "Not requested; per-method and leave-one-class estimates are descriptive",
            "base_ci_matched_frozen_configuration": same_bootstrap,
        },
        "limitations": [
            "Section19.2 was requested after exploratory S1 discovery; it is not the original Section17 analysis.",
            "Additive method/class adjustment is not a method-by-class fixed-effect or causal model.",
            "Only five seed clusters; percentile CI stability and generalization are limited.",
            "Floor-eligible class composition differs across methods; counts are shown.",
            "Only classes with old-class transition units enter leave-one-class-out.",
            "This reuses pre-cleanup Section17 results; Section18 caveats are not waived.",
            "No retraining, cleanup, hyperparameter selection, or revision of CRR rejection.",
        ],
    }
    return result, bootstrap_rows


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--input-dir", type=Path,
                        default=root / "ncd/results/exp_h/section17/registered")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "ncd/results/exp_h/section19/s1_robustness")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=1705)
    args = parser.parse_args(argv)
    source = args.input_dir.resolve()
    destination = args.output_dir.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Cannot write inside historical input directory")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Output directory must be new or empty; historical outputs are immutable")
    destination.mkdir(parents=True, exist_ok=True)
    log_path = destination / "run.log"
    try:
        paths = [source / name for name in ("exp_h_result.json", "exp_h_analysis.json")]
        source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        payload, reference = [json.loads(p.read_bytes()) for p in paths]
        if source_hashes["exp_h_result.json"] != reference["input_sha256"]:
            raise ValueError("Section17 raw JSON does not match its analysis input hash")
        with threadpool_limits(limits=1):
            result, draws = analyze(payload, reference, args.bootstrap, args.bootstrap_seed)
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                             text=True).strip()
        safe_relative = lambda p: p.relative_to(root).as_posix() if p.is_relative_to(root) else p.name
        command = ["python", "ncd/analyze_exp_h_robustness.py",
                   "--input-dir", safe_relative(source), "--output-dir", safe_relative(destination),
                   "--bootstrap", str(args.bootstrap), "--bootstrap-seed", str(args.bootstrap_seed)]
        result.update({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit, "source_git_commit": payload["git_commit"],
            "source_directory": safe_relative(source), "source_sha256": source_hashes,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "command": " ".join(command), "log_path": safe_relative(log_path),
        })
        if source_hashes != {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}:
            raise RuntimeError("Historical source changed during analysis")
        (destination / "s1_robustness.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        write_csv(destination / "per_method.csv", result["per_method"])
        write_csv(destination / "leave_one_class_out.csv", result["leave_one_class_out"])
        write_csv(destination / "class_counts.csv", result["sample_size"]["by_class"])
        write_csv(destination / "bootstrap_draws.csv", draws)
        summary = {key: result[key] for key in ("git_commit", "source_git_commit", "command",
                                                "primary_adjusted", "baseline_z_only",
                                                "s1_adjusted_condition_met", "sample_size",
                                                "per_method_positive_counts",
                                                "leave_one_class_out_minimum")}
        text = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        log_path.write_text(text, encoding="utf-8")
        print(text, end="")
    except Exception:
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
