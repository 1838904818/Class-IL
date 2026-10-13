"""Section 17 Exp H analysis; no training or changes to registered CRR results.

Input JSON contains ``rows`` at setting/method/seed/transition/old-class grain,
``retention_rows`` at setting/method/seed/class grain, and optional ``settings``
and ``feature_alignment`` metadata. Recall values are fractions, not percents.
The CLI can write a partial report while experiments are running. Missing,
nonfinite, or unidentifiable primary data cannot produce a negative verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata


METHODS = (
    "replay", "icarl", "replay_lwf", "kd_perturbed", "rcl_tab", "global_ir",
    "oldpair_kd", "crr",
)
SEEDS = (1, 2, 3, 4, 42)
FLOOR = 0.2
MIN_BOOTSTRAPS = 2000
PRIMARY = "old_pair_pi_drift"
SECONDARY = "self_margin_pi_drift"
RECALL_FIELDS = (
    "shift_recall_before", "shift_recall_after", "id_recall_before",
    "id_recall_after",
)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool) and math.isfinite(float(value))


def correlation(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Pearson correlation; a constant vector is unidentified, never zero."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(a) != len(b) or len(a) < 3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    a, b = a - a.mean(), b - b.mean()
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return None
    return float(np.clip(np.dot(a, b) / norm_a / norm_b, -1.0, 1.0))


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or not all(finite_number(v) for v in list(x) + list(y)):
        return None
    return correlation(rankdata(x, method="average"), rankdata(y, method="average"))


def partial_spearman(
    x: Sequence[float], decline: Sequence[float], z: Sequence[float],
) -> float | None:
    """Correlate residual ranks of X and -Y after regressing each on rank(Z).

    Every regression includes an intercept. Average ranks handle ties. Constant
    Z reduces to an intercept-only adjustment; constant residual X or -Y has no
    estimable correlation. Ranks are recomputed in each bootstrap sample.
    """
    if len(x) != len(decline) or len(x) != len(z) or len(x) < 4:
        return None
    values = np.asarray([x, decline, z], dtype=float)
    if not np.isfinite(values).all():
        return None
    rx, ry, rz = (rankdata(a, method="average") for a in values)
    design = np.column_stack((np.ones(len(rz)), rz))
    residual_x = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    residual_y = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    return correlation(residual_x, residual_y)


def sample_cluster_indices(
    seeds: Sequence[int], rng: np.random.Generator,
) -> np.ndarray:
    """Draw seed IDs, then concatenate every row of each drawn seed cluster.

    Drawing a seed twice repeats its *entire* cluster twice. No methods,
    transitions, or classes are independently resampled.
    """
    seed_array = np.asarray(seeds)
    unique = np.unique(seed_array)
    if not len(unique):
        return np.asarray([], dtype=int)
    chosen = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([np.flatnonzero(seed_array == seed) for seed in chosen])


def cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]], metric: str = PRIMARY,
    n_bootstrap: int = MIN_BOOTSTRAPS, random_seed: int = 1705,
) -> dict[str, Any]:
    """Seed-cluster percentile CI for one setting and one drift endpoint."""
    if n_bootstrap < MIN_BOOTSTRAPS:
        raise ValueError("Section 17.5 requires at least 2000 seed-cluster bootstrap draws")
    seeds = np.asarray([int(row["seed"]) for row in rows], dtype=int)
    x = np.asarray([float(row[metric]) for row in rows])
    decline = np.asarray([float(row["shift_recall_before"]) - float(row["shift_recall_after"]) for row in rows])
    z = np.asarray([float(row["id_recall_after"]) - float(row["id_recall_before"]) for row in rows])
    estimate = partial_spearman(x, decline, z)
    result: dict[str, Any] = {
        "rho": estimate, "ci95": None, "bootstrap_requested": n_bootstrap,
        "bootstrap_valid": 0, "bootstrap_undefined": 0,
        "bootstrap_seed": random_seed, "cluster_count": len(np.unique(seeds)),
        "row_count": len(rows), "status": "pending", "reason": None,
    }
    if estimate is None:
        result["reason"] = "Partial correlation is unidentified (too few units or constant residual ranks)"
        return result
    if len(np.unique(seeds)) < 2:
        result["reason"] = "Fewer than two nonempty seed clusters"
        return result
    rng = np.random.default_rng(random_seed)
    boot = []
    for _ in range(n_bootstrap):
        indices = sample_cluster_indices(seeds, rng)
        value = partial_spearman(x[indices], decline[indices], z[indices])
        if value is not None:
            boot.append(value)
    result["bootstrap_valid"] = len(boot)
    result["bootstrap_undefined"] = n_bootstrap - len(boot)
    if len(boot) != n_bootstrap:
        # Dropping undefined draws silently would change the bootstrap's target.
        # Retain the counts and point estimate, but do not adjudicate from a
        # conditional interval or replace undefined correlations with zeros.
        result["reason"] = "Some seed-cluster draws have unidentified correlation; no formal CI"
        return result
    result["ci95"] = [float(v) for v in np.quantile(boot, [0.025, 0.975])]
    result["status"] = "complete"
    return result


def _unit_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (row["setting"], row["method"], int(row["seed"]), str(row["transition"]), str(row["class_id"]))


def _setting_metadata(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    data = payload.get("setting_metadata", payload.get("settings", []))
    if isinstance(data, dict):
        return {str(key): value if isinstance(value, dict) else {"status": value} for key, value in data.items()}
    return {str(row["setting"]): dict(row) if isinstance(row, dict) else {} for row in data if isinstance(row, dict) and "setting" in row}


def _row_is_valid(row: Mapping[str, Any], metric: str) -> bool:
    return finite_number(row.get(metric)) and all(
        finite_number(row.get(field)) and 0 <= float(row[field]) <= 1 for field in RECALL_FIELDS
    )


def _completeness(
    rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any],
    methods: Sequence[str], seeds: Sequence[int],
) -> list[str]:
    reasons = []
    expected_cells = {(method, seed) for method in methods for seed in seeds}
    cells = {(row["method"], int(row["seed"])) for row in rows}
    if cells != expected_cells:
        reasons.append(f"Expected {len(expected_cells)} method/seed cells; observed {len(cells)} (missing {len(expected_cells-cells)}, unexpected {len(cells-expected_cells)})")
    keys = [_unit_key(row) for row in rows]
    if len(set(keys)) != len(keys):
        reasons.append(f"Duplicate analysis-unit keys: {len(keys)-len(set(keys))}")
    units_by_cell: dict[tuple[str, int], set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        units_by_cell[(row["method"], int(row["seed"]))].add((str(row["transition"]), str(row["class_id"])))
    units = list(units_by_cell.values())
    if units and any(unit != units[0] for unit in units[1:]):
        reasons.append("Methods/seeds do not contain the same transition/class units")
    expected_count = metadata.get("expected_units_per_cell")
    if expected_count is not None and any(len(unit) != int(expected_count) for unit in units):
        reasons.append(f"Expected {int(expected_count)} transition/class units per method/seed")
    if metadata.get("expected_units") is not None and len(rows) != int(metadata["expected_units"]):
        reasons.append(f"Expected {int(metadata['expected_units'])} total units, found {len(rows)}")
    if metadata.get("status") != "complete":
        reasons.append("Setting producer has not marked its full evaluation complete")
    if metadata.get("smoke", False):
        reasons.append("Smoke runs are not formal evidence")
    return reasons


def _method_results(
    setting: str, rows: Sequence[Mapping[str, Any]],
    retention_rows: Sequence[Mapping[str, Any]], methods: Sequence[str], seeds: Sequence[int],
) -> dict[str, Any]:
    """Descriptive 8-method association, with explicit final/best endpoints.

    Mean drift uses floor-eligible units, matching the registered exclusion.
    Retention is first averaged over old classes within a method/seed and then
    over the five seeds. Repeated transitions never duplicate a class endpoint.
    Zero best recall has undefined final/best and is reported as missing.
    """
    entries, reasons = [], []
    lookup: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    seen = set()
    for row in retention_rows:
        if row.get("setting") != setting:
            continue
        try:
            key = (row["method"], int(row["seed"]), str(row["class_id"]))
        except (KeyError, TypeError, ValueError):
            reasons.append("Malformed retention row")
            continue
        if key in seen:
            reasons.append("Duplicate retention method/seed/class key")
        seen.add(key)
        lookup[key[:2]].append(row)
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        means = {metric: float(np.mean([row[metric] for row in selected])) if selected and all(finite_number(row.get(metric)) for row in selected) else None for metric in (PRIMARY, SECONDARY)}
        seed_retention, count = [], 0
        for seed in seeds:
            values = []
            observed_classes = set()
            expected_classes = {str(row["class_id"]) for row in selected if int(row["seed"]) == seed}
            for row in lookup.get((method, seed), []):
                class_id = str(row["class_id"])
                if class_id not in expected_classes:
                    continue
                final, best = row.get("shift_recall_final"), row.get("shift_recall_best")
                if not finite_number(final) or not finite_number(best) or not 0 <= float(final) <= float(best) <= 1 or float(best) <= 0:
                    reasons.append(f"{method}/seed {seed}: undefined or invalid final/best retention")
                    continue
                observed_classes.add(class_id)
                values.append(float(final) / float(best))
            if expected_classes != observed_classes or not expected_classes:
                reasons.append(f"{method}/seed {seed}: incomplete class retention endpoints")
            if values:
                seed_retention.append(float(np.mean(values)))
                count += len(values)
        entries.append({
            "method": method, "mean_old_pair_pi_drift": means[PRIMARY],
            "mean_self_margin_pi_drift": means[SECONDARY],
            "mean_shift_retention": float(np.mean(seed_retention)) if len(seed_retention) == len(seeds) else None,
            "retention_seeds": len(seed_retention), "retention_class_seed_units": count,
            "drift_units": len(selected),
        })
    estimates = {}
    for metric in (PRIMARY, SECONDARY):
        valid = [row for row in entries if row[f"mean_{metric}"] is not None and row["mean_shift_retention"] is not None]
        estimates[metric] = spearman([row[f"mean_{metric}"] for row in valid], [row["mean_shift_retention"] for row in valid]) if len(valid) == len(methods) else None
    return {
        "rows": entries, "rho_drift_vs_retention": estimates, "point_count": len(methods),
        "status": "complete" if not reasons and all(v is not None for v in estimates.values()) else "pending",
        "reasons": sorted(set(reasons)) or ([] if all(v is not None for v in estimates.values()) else ["Constant or missing method-level values"]),
        "interpretation": "Eight method means; descriptive association, not an independent confirmatory test",
    }


def adjudicate(settings: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Apply 17.6 using primary old-pair drift only, without changing cutoffs."""
    if len(settings) < 2 or any(item.get("status") != "complete" for item in settings.values()):
        return {"decision": "pending", "reason": "At least two complete settings and identifiable primary intervals are required", "harm_settings": []}
    primary = {name: item["unit_level"][PRIMARY] for name, item in settings.items()}
    if any(not finite_number(item.get("rho")) or not item.get("ci95") or not all(finite_number(v) for v in item["ci95"]) for item in primary.values()):
        return {"decision": "pending", "reason": "Missing or nonfinite primary estimate or confidence interval", "harm_settings": []}
    harm = [name for name, item in primary.items() if item["rho"] >= 0.2 and item["ci95"][0] > 0]
    if len(harm) >= 2:
        return {"decision": "harm_supported", "reason": "At least two settings have positive partial Spearman >= 0.2 and 95% CIs excluding zero", "harm_settings": harm}
    if all(item["rho"] < 0.1 for item in primary.values()) or all(item["ci95"][0] <= 0 <= item["ci95"][1] for item in primary.values()):
        return {"decision": "harm_not_supported", "reason": "All primary estimates are < 0.1, or every 95% CI contains zero", "harm_settings": harm}
    return {"decision": "intermediate", "reason": "Neither registered endpoint of Section 17.6 is met; project owner decides next step", "harm_settings": harm}


def analyze(payload: Mapping[str, Any], n_bootstrap: int = MIN_BOOTSTRAPS, random_seed: int = 1705) -> dict[str, Any]:
    if n_bootstrap < MIN_BOOTSTRAPS:
        raise ValueError("Section 17.5 requires at least 2000 bootstrap draws")
    config = payload.get("config", {})
    methods = tuple(config.get("expected_methods", METHODS))
    seeds = tuple(int(seed) for seed in config.get("expected_seeds", SEEDS))
    if set(methods) != set(METHODS) or len(methods) != 8 or set(seeds) != set(SEEDS) or len(seeds) != 5:
        raise ValueError("Formal Exp H requires the registered eight methods and seeds {1,2,3,4,42}")
    metadata = _setting_metadata(payload)
    rows = list(payload.get("rows", []))
    malformed = []
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        try:
            if any(row.get(field) is None for field in ("setting", "method", "seed", "transition", "class_id")):
                raise ValueError("Missing unit identity")
            key = _unit_key(row)
            if not isinstance(row["seed"], (int, np.integer)) or not all(key[pos] is not None for pos in range(len(key))):
                raise ValueError("Missing identity or noninteger seed")
            grouped[str(row["setting"])].append(row)
        except (AttributeError, KeyError, TypeError, ValueError):
            malformed.append(index)
    named_settings = payload.get("settings", [])
    named_settings = named_settings if isinstance(named_settings, list) and all(isinstance(value, str) for value in named_settings) else []
    names = sorted(set(grouped) | set(metadata) | set(config.get("expected_settings", [])) | set(named_settings))
    settings = {}
    for index, setting in enumerate(names):
        selected = grouped[setting]
        reasons = _completeness(selected, metadata.get(setting, {}), methods, seeds)
        bad = [row for row in selected if not _row_is_valid(row, PRIMARY)]
        secondary_bad = [row for row in selected if not _row_is_valid(row, SECONDARY)]
        floor = [row for row in selected if finite_number(row.get("shift_recall_before")) and 0 <= float(row["shift_recall_before"]) < FLOOR]
        valid = [row for row in selected if _row_is_valid(row, PRIMARY) and float(row["shift_recall_before"]) >= FLOOR]
        if bad:
            reasons.append(f"{len(bad)} rows contain missing, nonfinite, or invalid numeric values")
        if malformed:
            reasons.append(f"Input contains {len(malformed)} malformed analysis-unit identities")
        unit = {metric: cluster_bootstrap([row for row in valid if _row_is_valid(row, metric)], metric, n_bootstrap, random_seed + index) for metric in (PRIMARY, SECONDARY)}
        if secondary_bad:
            unit[SECONDARY].update(status="pending", reason="Missing, nonfinite, or invalid secondary endpoint values")
        if unit[PRIMARY]["status"] != "complete":
            reasons.append(unit[PRIMARY]["reason"])
        if set(int(row["seed"]) for row in valid) != set(seeds):
            reasons.append("Not all five seed clusters remain after the floor exclusion")
        method = _method_results(setting, valid, payload.get("retention_rows", []), methods, seeds)
        settings[setting] = {
            "status": "complete" if not reasons else "pending", "reasons": reasons,
            "report_status": "complete" if not reasons and method["status"] == "complete" and unit[SECONDARY]["status"] == "complete" else "partial",
            "metadata": metadata.get(setting, {}), "unit_level": unit, "method_level": method,
            "sample_size": {
                "raw_units": len(selected), "floor_excluded_units": len(floor),
                "invalid_numeric_units": len(bad), "included_units": len(valid),
                "secondary_invalid_numeric_units": len(secondary_bad),
                "expected_methods": len(methods), "observed_methods": len({row["method"] for row in selected}),
                "expected_seed_clusters": len(seeds), "included_seed_clusters": len({row["seed"] for row in valid}),
                "included_by_seed": dict(sorted(Counter(int(row["seed"]) for row in valid).items())),
                "floor_excluded_by_method": dict(sorted(Counter(row["method"] for row in floor).items())),
                "note": "Invalid and floor-excluded counts may overlap; neither is imputed",
            },
        }
    verdict = adjudicate(settings)
    feature_alignment = payload.get("feature_alignment", []) or [row for name in names for row in metadata.get(name, {}).get("feature_alignment", [])]
    provenance = payload.get("provenance", {}) or {key: payload[key] for key in ("git_commit", "effective_config", "command", "log_path", "lambda_selection", "self_margin_definition") if key in payload}
    return {
        "experiment": "Exp H", "work_order": "Section 17.5 / 17.6", "settings": settings,
        "verdict": verdict, "malformed_input_row_indices": malformed,
        "report_status": "complete" if settings and all(detail["report_status"] == "complete" for detail in settings.values()) else "partial",
        "feature_alignment": feature_alignment,
        "provenance": provenance,
        "analysis": {
            "primary_metric": PRIMARY, "secondary_metric": SECONDARY,
            "response": "shift_recall_before - shift_recall_after (-Y)",
            "control": "id_recall_after - id_recall_before (Z)",
            "estimator": "Pearson correlation of average-rank residuals, each regressed on an intercept and rank(Z)",
            "bootstrap": "Sample five seed clusters with replacement; retain all methods/transitions/classes in each sampled cluster; recompute ranks",
            "bootstrap_draws": n_bootstrap, "bootstrap_seed": random_seed,
            "ci": "95% percentile interval, unadjusted, separate per setting",
            "floor_rule": "Exclude shift_recall_before < 0.2; retain exactly 0.2",
            "method_level": "Mean floor-eligible X vs final/best shifted recall averaged across classes within seed then across seeds; eight method points; descriptive",
        },
        "scope": "RQ1/RO1 association under distribution shift; does not establish causation or substantive innovation. CRR rejection is retained without modification.",
    }


def _fmt(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}" if finite_number(value) else "pending"


def _text(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_scatter(payload: Mapping[str, Any], output: Path) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    settings = sorted({str(row.get("setting")) for row in payload.get("rows", []) if row.get("setting") is not None})
    if not settings:
        return False
    fig, axes = plt.subplots(1, len(settings), figsize=(5.2 * len(settings), 4.3), squeeze=False, sharey=True)
    colors = dict(zip(METHODS, plt.get_cmap("tab10").colors))
    for ax, setting in zip(axes[0], settings):
        for method in METHODS:
            rows = [row for row in payload["rows"] if str(row.get("setting")) == setting and row.get("method") == method and _row_is_valid(row, PRIMARY) and row["shift_recall_before"] >= FLOOR]
            ax.scatter([row[PRIMARY] for row in rows], [row["shift_recall_after"] - row["shift_recall_before"] for row in rows], s=15, alpha=.45, color=colors[method], label=method)
        ax.axhline(0, color="gray", linewidth=.8)
        ax.set_title(setting)
        ax.set_xlabel("Old-pair PI drift (X)")
        ax.grid(alpha=.15)
    axes[0, 0].set_ylabel("Shifted recall change (Y; after - before)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, .13, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def render_report(result: Mapping[str, Any], scatter_name: str | None = None) -> str:
    decisions = {"pending": "Pending", "harm_supported": "Harm supported by Section 17.6", "harm_not_supported": "Harm not supported by Section 17.6", "intermediate": "Intermediate; owner decision required"}
    lines = [
        "# Exp H: dependency drift and shifted recall", "",
        f"Decision: **{decisions[result['verdict']['decision']]}**. {result['verdict']['reason']}.", "",
        f"Report completeness: {result['report_status']}. The Section 17.6 decision uses the primary unit-level endpoint; an unidentified descriptive eight-point correlation does not add a new decision gate.", "",
        result["scope"], "", "## Feature alignment and evaluation scope", "",
        "| Setting | Training feature | Shift feature | Action | Notes |",
        "|---|---|---|---|---|",
    ]
    alignment = result.get("feature_alignment", [])
    if alignment:
        for row in alignment:
            lines.append("| " + " | ".join(_text(row.get(key, "")) for key in ("setting", "training_feature", "shift_feature", "action", "notes")) + " |")
    else:
        lines.append("| pending | | | | Feature alignment metadata not supplied |")
    for name, detail in result["settings"].items():
        metadata = detail["metadata"]
        description = metadata.get("description", metadata.get("notes", ""))
        if description:
            lines.extend(["", f"{name}: {_text(description)}"])
        for key in ("caveats", "limitations", "scope", "shift_description", "dataset_role", "id_shift_overlap"):
            if metadata.get(key):
                value = metadata[key]
                if isinstance(value, (list, tuple)):
                    value = "; ".join(str(part) for part in value)
                lines.extend(["", f"{name} {key.replace('_', ' ')}: {_text(value)}"])
    lines.extend([
        "", "## Table 1: unit-level partial Spearman", "",
        "X is PI drift; Y is shifted recall after minus before; Z is same-distribution recall after minus before. Average ranks are residualized on an intercept and rank(Z). The statistic correlates residual rank(X) with residual rank(-Y). The old-pair metric alone drives Section 17.6; self-margin is secondary.", "",
        "| Setting | Drift endpoint | Units | Partial rho (X, -Y given Z) | 95% seed-cluster CI | Valid bootstrap draws | Status |",
        "|---|---|---:|---:|---|---:|---|",
    ])
    for name, detail in result["settings"].items():
        for metric, unit in detail["unit_level"].items():
            interval = "[" + ", ".join(_fmt(v) for v in unit["ci95"]) + "]" if unit["ci95"] else "pending"
            lines.append(f"| {_text(name)} | {metric} | {unit['row_count']} | {_fmt(unit['rho'])} | {interval} | {unit['bootstrap_valid']}/{unit['bootstrap_requested']} | {detail['status']} / {unit['status']} |")
            if unit.get("reason"):
                lines.extend(["", f"{name} {metric}: {unit['reason']}.", ""])
        if detail["reasons"]:
            lines.extend(["", f"{name}: " + "; ".join(detail["reasons"]) + ".", ""])
    lines.extend([
        "", "Each draw resamples five entire seed clusters with replacement, keeping every method, transition and class belonging to each sampled seed together. Ranks are recomputed. Intervals are percentile intervals, unadjusted and separate by setting. Undefined draws are counted, not replaced by zero. Five seed clusters limit interval stability and generalization.", "",
        "## Table 2: method-level descriptive comparison", "",
        "Retention is shifted final recall divided by the best shifted recall seen for that class, averaged over old classes within seed and then over seeds. Only classes represented by floor-eligible drift units enter the matched retention summary. No denominator is stabilized or imputed.", "",
        "| Setting | Method | Mean old-pair drift | Mean self-margin drift | Mean shifted final/best recall | Retention seeds | Drift units |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for name, detail in result["settings"].items():
        for row in detail["method_level"]["rows"]:
            lines.append(f"| {_text(name)} | {row['method']} | {_fmt(row['mean_old_pair_pi_drift'])} | {_fmt(row['mean_self_margin_pi_drift'])} | {_fmt(row['mean_shift_retention'])} | {row['retention_seeds']}/5 | {row['drift_units']} |")
        estimates = detail["method_level"]["rho_drift_vs_retention"]
        lines.extend(["", f"{name}: eight-point Spearman (drift versus retention): old-pair {_fmt(estimates[PRIMARY])}; self-margin {_fmt(estimates[SECONDARY])}. Descriptive only; status {detail['method_level']['status']}.", ""])
        if detail["method_level"]["reasons"]:
            lines.extend(["Retention limitations: " + "; ".join(detail["method_level"]["reasons"]) + ".", ""])
    lines.extend([
        "", "## Table 3: sample sizes and exclusions", "",
        "| Setting | Raw units | Floor excluded (< 0.2) | Invalid numeric units | Included units | Methods | Nonempty seed clusters |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name, detail in result["settings"].items():
        n = detail["sample_size"]
        lines.append(f"| {_text(name)} | {n['raw_units']} | {n['floor_excluded_units']} | {n['invalid_numeric_units']} | {n['included_units']} | {n['observed_methods']}/{n['expected_methods']} | {n['included_seed_clusters']}/{n['expected_seed_clusters']} |")
        lines.extend(["", f"{name}: included by seed {n['included_by_seed']}; floor excluded by method {n['floor_excluded_by_method']}; invalid secondary values {n['secondary_invalid_numeric_units']}.", ""])
    lines.extend(["", "The floor rule uses recall at the pre-transition checkpoint only. Exactly 0.2 is retained. Invalid and floor-excluded counts can overlap; malformed identities are reported separately.", "", "## Scatter: X versus Y by shift setting", ""])
    if scatter_name:
        lines.append(f"![Old-pair drift versus shifted recall change by setting]({scatter_name})")
    else:
        lines.append("Pending: no valid input rows for the scatter.")
    lines.extend(["", "Points are old-class transition units after the floor exclusion, colored by method. The plotted response is Y, while the primary correlation uses -Y. This is an observational association across existing methods and cannot by itself identify a causal effect of drift.", "", "## Registered decision and provenance", "",
        "Harm supported: at least two settings have rho >= 0.2 and positive 95% CIs excluding zero. Harm not supported: all settings have rho < 0.1, or all CIs contain zero. Otherwise the outcome is intermediate. Incomplete or unidentified primary evidence remains pending. CRR's rejection is unchanged.", "",
        "```json", json.dumps({"verdict": result["verdict"], "provenance": result["provenance"], "analysis": result["analysis"], "malformed_input_row_indices": result["malformed_input_row_indices"]}, ensure_ascii=False, indent=2, allow_nan=False), "```", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--retention-input", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=MIN_BOOTSTRAPS)
    parser.add_argument("--bootstrap-seed", type=int, default=1705)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if args.retention_input:
        retention = json.loads(args.retention_input.read_text(encoding="utf-8"))
        payload["retention_rows"] = retention if isinstance(retention, list) else retention["retention_rows"]
    result = analyze(payload, args.bootstrap, args.bootstrap_seed)
    result["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    if args.retention_input:
        result["retention_input_sha256"] = hashlib.sha256(args.retention_input.read_bytes()).hexdigest()
    scatter = args.report.with_name(args.report.stem + "_scatter.png")
    plotted = write_scatter(payload, scatter)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(result, scatter.name if plotted else None), encoding="utf-8")
    print(json.dumps(result["verdict"], ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
