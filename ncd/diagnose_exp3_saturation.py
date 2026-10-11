"""Diagnose Exp 3 NCD saturation against the frozen Exp 2 controls.

This is an exploratory diagnostic.  It does not turn a post-hoc threshold into
confirmatory evidence.  The script uses cross-cell holdout evaluation to show
whether simple effect-size gates can retain the intervention benchmark while
reducing the near-universal Exp 3 significance calls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


EXP2_CELLS = (
    ("nsl-kdd", "replay", "NSL-KDD"),
    ("nsl-kdd", "icarl", "NSL-KDD"),
    ("unsw-nb15", "replay", "UNSW-NB15"),
    ("unsw-nb15", "icarl", "UNSW-NB15"),
)
SEEDS = (1, 2, 3, 4, 42)
EXPLAINERS = ("expected_gradients", "feature_ablation", "grad_x_input")
METRICS = ("absolute", "relative_within")
Q_THRESHOLD = 0.05


def parse_bool(value: object) -> bool:
    text = str(value).strip().lower()
    if text not in {"true", "false"}:
        raise ValueError(f"expected boolean text, got {value!r}")
    return text == "true"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_source(path: Path, root: Path) -> str:
    resolved = path.resolve(strict=True)
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return f"external/{sha256(resolved)[:12]}-{resolved.name}"


def nearest_rank(values: Iterable[float], quantile: float) -> float:
    """Return the deterministic nearest-rank quantile."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot calculate a quantile from no values")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile outside [0, 1]: {quantile}")
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def lower_order_quantile(values: Iterable[float], quantile: float) -> float:
    """Return a conservative lower order statistic for positive anchors."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot calculate a quantile from no values")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile outside [0, 1]: {quantile}")
    index = math.floor(quantile * (len(ordered) - 1))
    return ordered[index]


def metric_value(row: dict[str, object], metric: str) -> float:
    effect = float(row["ncd_effect"])
    if metric == "absolute":
        return effect
    if metric == "relative_within":
        within = float(row["within_median_1_minus_spearman"])
        return effect / max(within, 1e-12)
    raise ValueError(f"unsupported metric: {metric}")


def is_negative_control(row: dict[str, object]) -> bool:
    return str(row["condition"]).startswith(("N1", "N2"))


def is_eligible_positive(row: dict[str, object]) -> bool:
    condition = str(row["condition"])
    strength = float(row["strength"])
    selected = (condition == "P1_shortcut" and strength >= 0.9) or condition == "P2_feature_mask"
    return selected and parse_bool(row["truth_positive"]) and parse_bool(row["silent_eligible"])


def load_exp2_rows(exp2_root: Path) -> tuple[list[dict[str, object]], list[Path]]:
    rows: list[dict[str, object]] = []
    paths: list[Path] = []
    for dataset_dir, method, dataset in EXP2_CELLS:
        for seed in SEEDS:
            path = exp2_root / dataset_dir / method / f"seed{seed}" / "exp2_groundtruth_table.csv"
            if not path.is_file():
                raise FileNotFoundError(f"missing frozen Exp 2 table: {path}")
            paths.append(path)
            for raw in read_csv(path):
                row: dict[str, object] = dict(raw)
                row.update(dataset=dataset, method=method, seed=seed)
                rows.append(row)
    validate_exp2_rows(rows)
    return rows, paths


def exp2_scenario_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row["dataset"],
        row["method"],
        int(row["seed"]),
        row["condition"],
        float(row["strength"]),
    )


def group_rows(
    rows: Iterable[dict[str, object]],
    key_function,
) -> dict[tuple[object, ...], list[dict[str, object]]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[key_function(row)].append(row)
    return grouped


def validate_three_explainers(
    grouped: dict[tuple[object, ...], list[dict[str, object]]],
    label: str,
) -> None:
    for key, rows in grouped.items():
        explainers = {str(row["explainer"]) for row in rows}
        if len(rows) != 3 or explainers != set(EXPLAINERS):
            raise ValueError(f"incomplete {label} three-explainer unit {key}: {sorted(explainers)}")


def validate_exp2_rows(rows: list[dict[str, object]]) -> None:
    expected_files = len(EXP2_CELLS) * len(SEEDS)
    identities = {(row["dataset"], row["method"], int(row["seed"])) for row in rows}
    if len(identities) != expected_files:
        raise ValueError(f"incomplete Exp 2 identity coverage: {len(identities)} != {expected_files}")
    grouped = group_rows(rows, exp2_scenario_key)
    validate_three_explainers(grouped, "Exp 2")
    if not any(is_negative_control(row) for row in rows):
        raise ValueError("Exp 2 contains no N1/N2 negative controls")
    if not any(is_eligible_positive(row) for row in rows):
        raise ValueError("Exp 2 contains no eligible P1/P2 positives")


def noise_thresholds(rows: list[dict[str, object]], metric: str) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for explainer in EXPLAINERS:
        values = [
            metric_value(row, metric)
            for row in rows
            if row["explainer"] == explainer and is_negative_control(row)
        ]
        thresholds[explainer] = nearest_rank(values, 0.95)
    return thresholds


def positive_anchor_thresholds(rows: list[dict[str, object]], metric: str) -> dict[str, float]:
    noise = noise_thresholds(rows, metric)
    thresholds: dict[str, float] = {}
    for explainer in EXPLAINERS:
        values = [
            metric_value(row, metric)
            for row in rows
            if row["explainer"] == explainer and is_eligible_positive(row)
        ]
        thresholds[explainer] = max(noise[explainer], lower_order_quantile(values, 0.05))
    return thresholds


def consensus_vote(rows: list[dict[str, object]], predicate) -> bool:
    return sum(bool(predicate(row)) for row in rows) >= 2


def effect_gate_vote(
    rows: list[dict[str, object]],
    metric: str,
    thresholds: dict[str, float],
    inclusive: bool,
) -> bool:
    def selected(row: dict[str, object]) -> bool:
        value = metric_value(row, metric)
        threshold = thresholds[str(row["explainer"])]
        passes_effect = value >= threshold if inclusive else value > threshold
        return (
            float(row["ncd_q_value"]) < Q_THRESHOLD
            and float(row["ncd_effect"]) > 0.0
            and passes_effect
        )

    return consensus_vote(rows, selected)


def exp2_population(rows: list[dict[str, object]]) -> str | None:
    if is_negative_control(rows[0]):
        return "negative_control"
    if is_eligible_positive(rows[0]):
        return "eligible_positive"
    return None


def evaluate_exp2_rules(
    rows: list[dict[str, object]],
    thresholds: dict[tuple[str, str], dict[str, float]],
) -> list[dict[str, object]]:
    grouped = group_rows(rows, exp2_scenario_key)
    output: list[dict[str, object]] = []
    for key, unit_rows in grouped.items():
        population = exp2_population(unit_rows)
        if population is None:
            continue
        rules = {
            "B2_repeat_mean": consensus_vote(unit_rows, lambda row: parse_bool(row["b2_fixed_drift"])),
            "NCD_current": consensus_vote(unit_rows, lambda row: parse_bool(row["ncd_drift"])),
        }
        for calibration in ("noise95", "positive_anchor05"):
            for metric in METRICS:
                rules[f"{calibration}_{metric}"] = effect_gate_vote(
                    unit_rows,
                    metric,
                    thresholds[(calibration, metric)],
                    inclusive=calibration == "positive_anchor05",
                )
        for rule, detected in rules.items():
            output.append(
                {
                    "dataset": key[0],
                    "method": key[1],
                    "seed": key[2],
                    "condition": key[3],
                    "strength": key[4],
                    "population": population,
                    "rule": rule,
                    "detected": detected,
                }
            )
    return output


def cross_cell_evaluation(exp2_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for _, held_method, held_dataset in EXP2_CELLS:
        held_cell = (held_dataset, held_method)
        train = [
            row for row in exp2_rows if (row["dataset"], row["method"]) != held_cell
        ]
        test = [
            row for row in exp2_rows if (row["dataset"], row["method"]) == held_cell
        ]
        thresholds: dict[tuple[str, str], dict[str, float]] = {}
        for metric in METRICS:
            thresholds[("noise95", metric)] = noise_thresholds(train, metric)
            thresholds[("positive_anchor05", metric)] = positive_anchor_thresholds(train, metric)
        predictions = evaluate_exp2_rules(test, thresholds)
        grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
        for row in predictions:
            grouped[(str(row["rule"]), str(row["population"]))].append(row)
        for (rule, population), rows in sorted(grouped.items()):
            detected = sum(bool(row["detected"]) for row in rows)
            output.append(
                {
                    "held_out_dataset": held_dataset,
                    "held_out_method": held_method,
                    "rule": rule,
                    "population": population,
                    "detected": detected,
                    "total": len(rows),
                    "rate": detected / len(rows),
                }
            )
    return output


def exp3_unit_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row["dataset"],
        row["method"],
        int(row["seed"]),
        int(row["checkpoint_before"]),
        int(row["checkpoint_after"]),
        int(row["class_id"]),
    )


def load_exp3_rows(paths: list[Path]) -> list[dict[str, object]]:
    if len(paths) != 6:
        raise ValueError(f"expected six Exp 3 explainer tables, got {len(paths)}")
    rows: list[dict[str, object]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing Exp 3 explainer table: {path}")
        rows.extend(dict(row) for row in read_csv(path))
    grouped = group_rows(rows, exp3_unit_key)
    validate_three_explainers(grouped, "Exp 3")
    return rows


def exp3_predictions(
    exp3_rows: list[dict[str, object]],
    exp2_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    thresholds: dict[tuple[str, str], dict[str, float]] = {}
    threshold_rows: list[dict[str, object]] = []
    for metric in METRICS:
        calibrations = {
            "noise95": noise_thresholds(exp2_rows, metric),
            "positive_anchor05": positive_anchor_thresholds(exp2_rows, metric),
        }
        for calibration, values in calibrations.items():
            thresholds[(calibration, metric)] = values
            for explainer, value in values.items():
                threshold_rows.append(
                    {
                        "calibration": calibration,
                        "metric": metric,
                        "explainer": explainer,
                        "threshold": value,
                    }
                )

    silent_rows = [row for row in exp3_rows if parse_bool(row["silent_at_5pp"])]
    grouped = group_rows(silent_rows, exp3_unit_key)
    output: list[dict[str, object]] = []
    for key, unit_rows in grouped.items():
        rules = {
            "B2_repeat_mean": consensus_vote(unit_rows, lambda row: parse_bool(row["b2_fixed_drift"])),
            "NCD_current": consensus_vote(unit_rows, lambda row: parse_bool(row["ncd_drift"])),
        }
        for calibration in ("noise95", "positive_anchor05"):
            for metric in METRICS:
                converted = [dict(row, ncd_q_value=row["q_value"]) for row in unit_rows]
                rules[f"{calibration}_{metric}"] = effect_gate_vote(
                    converted,
                    metric,
                    thresholds[(calibration, metric)],
                    inclusive=calibration == "positive_anchor05",
                )
        for rule, detected in rules.items():
            output.append(
                {
                    "dataset": key[0],
                    "method": key[1],
                    "seed": key[2],
                    "checkpoint_before": key[3],
                    "checkpoint_after": key[4],
                    "class_id": key[5],
                    "rule": rule,
                    "detected": detected,
                }
            )
    return output, threshold_rows


def aggregate_counts(
    rows: list[dict[str, object]],
    dimensions: tuple[str, ...],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[name] for name in dimensions)].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        detected = sum(bool(row["detected"]) for row in values)
        record = {name: value for name, value in zip(dimensions, key)}
        record.update(detected=detected, total=len(values), rate=detected / len(values))
        output.append(record)
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summed_cross_cell(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["rule"]), str(row["population"]))].append(row)
    output = []
    for (rule, population), values in sorted(grouped.items()):
        detected = sum(int(row["detected"]) for row in values)
        total = sum(int(row["total"]) for row in values)
        output.append(
            {
                "rule": rule,
                "population": population,
                "detected": detected,
                "total": total,
                "rate": detected / total,
            }
        )
    return output


def render_report(
    cross_totals: list[dict[str, object]],
    exp3_totals: list[dict[str, object]],
    p_floor_rows: int,
    p_rows: int,
) -> str:
    cross = {(row["rule"], row["population"]): row for row in cross_totals}
    exp3 = {str(row["rule"]): row for row in exp3_totals}

    def fraction(row: dict[str, object]) -> str:
        return f"{row['detected']}/{row['total']} ({100 * float(row['rate']):.2f}%)"

    lines = [
        "# Exploratory Exp 3 saturation diagnostic",
        "",
        "This analysis is post hoc and does not establish a confirmatory detector or substantive innovation.",
        "Thresholds are learned only from the frozen Exp 2 controls, and cross-cell results hold out one",
        "dataset--learner cell at a time.",
        "",
        "| Rule | Exp 2 negative FP | Exp 2 eligible positive detection | Exp 3 silent drift |",
        "|---|---:|---:|---:|",
    ]
    order = (
        "B2_repeat_mean",
        "NCD_current",
        "noise95_absolute",
        "noise95_relative_within",
        "positive_anchor05_absolute",
        "positive_anchor05_relative_within",
    )
    for rule in order:
        lines.append(
            f"| {rule} | {fraction(cross[(rule, 'negative_control')])} | "
            f"{fraction(cross[(rule, 'eligible_positive')])} | {fraction(exp3[rule])} |"
        )
    lines.extend(
        [
            "",
            f"The minimum attainable permutation p-value occurs in {p_floor_rows}/{p_rows} "
            f"({100 * p_floor_rows / p_rows:.2f}%) Exp 3 explainer-level rows.",
            "",
            "## Interpretation",
            "",
            "The negative-control 95th-percentile effect gate preserves all held-out eligible positives,",
            "but it does not materially reduce Exp 3 saturation.  A stronger positive-anchored gate",
            "reduces the Exp 3 rate, but it misses held-out eligible positives and therefore fails the",
            "registered requirement of matching B2 sensitivity.  The current evidence does not support",
            "promoting either gate as a validated replacement.",
            "",
            "The fastest defensible next experiment is a fresh, frozen validation set containing subtle",
            "interventions near the decision boundary and additional independent negative controls.  No",
            "new full Exp 3 sweep is justified before that selective-validity gate is passed.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp2-root", type=Path, required=True)
    parser.add_argument("--exp3-explainer-csv", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    return parser


def main() -> None:
    args = build_parser().parse_args()
    exp2_rows, exp2_paths = load_exp2_rows(args.exp2_root)
    exp3_rows = load_exp3_rows(args.exp3_explainer_csv)

    cross_cell = cross_cell_evaluation(exp2_rows)
    cross_totals = summed_cross_cell(cross_cell)
    exp3_unit_rows, threshold_rows = exp3_predictions(exp3_rows, exp2_rows)
    exp3_dataset = aggregate_counts(exp3_unit_rows, ("dataset", "rule"))
    exp3_totals = aggregate_counts(exp3_unit_rows, ("rule",))

    minimum_p = 1.0 / 1001.0
    p_floor_rows = sum(abs(float(row["p_value"]) - minimum_p) < 1e-15 for row in exp3_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "cross_cell_evaluation.csv", cross_cell)
    write_csv(args.output_dir / "cross_cell_totals.csv", cross_totals)
    write_csv(args.output_dir / "exp3_dataset_summary.csv", exp3_dataset)
    write_csv(args.output_dir / "exp3_totals.csv", exp3_totals)
    write_csv(args.output_dir / "effect_thresholds.csv", threshold_rows)

    provenance = []
    for role, path in [
        *(("exp2", path) for path in exp2_paths),
        *(("exp3", path) for path in args.exp3_explainer_csv),
    ]:
        provenance.append(
            {
                "role": role,
                "source": portable_source(path, args.source_root),
                "sha256": sha256(path),
            }
        )

    result = {
        "analysis": "exploratory_exp3_saturation_diagnostic",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "exp2_files": len(exp2_paths),
            "exp2_explainer_rows": len(exp2_rows),
            "exp3_files": len(args.exp3_explainer_csv),
            "exp3_explainer_rows": len(exp3_rows),
            "exp3_silent_units": len(exp3_unit_rows) // 6,
            "permutations": 1000,
            "q_threshold": Q_THRESHOLD,
        },
        "inputs": provenance,
        "cross_cell_totals": cross_totals,
        "exp3_totals": exp3_totals,
        "exp3_dataset_summary": exp3_dataset,
        "effect_thresholds": threshold_rows,
        "minimum_p_rows": p_floor_rows,
        "minimum_p_total_rows": len(exp3_rows),
        "limitations": [
            "This is a post-hoc exploratory diagnostic, not a registered confirmatory analysis.",
            "Exp 2 positives are constructed interventions and do not establish external validity.",
            "The positive-anchor gate uses observed positive effects and requires fresh validation.",
        ],
    }
    (args.output_dir / "saturation_diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "SATURATION_DIAGNOSTIC.md").write_text(
        render_report(cross_totals, exp3_totals, p_floor_rows, len(exp3_rows)),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
