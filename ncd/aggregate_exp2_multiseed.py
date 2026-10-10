"""Aggregate Exp 2 intervention-grounded results across frozen random seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


SEEDS = (1, 2, 3, 4, 42)
METRICS = {
    "B1 single J@15": lambda row: 1.0 - float(row["b1_single_jaccard"]),
    "B2 repeat-mean J@15": lambda row: 1.0 - float(row["b2_repeat_mean_jaccard"]),
    "B3 CLEX ED": lambda row: float(row["b3_clex_ed"]),
    "B4 cosine": lambda row: float(row["b4_cosine_distance"]),
    "B4 Kendall": lambda row: float(row["b4_kendall_distance"]),
    "NCD": lambda row: float(row["ncd_effect"]),
}
COLORS = {
    "B1 single J@15": "#355C7D",
    "B2 repeat-mean J@15": "#D4A017",
    "B3 CLEX ED": "#C96A3D",
    "B4 cosine": "#7A8F3A",
    "B4 Kendall": "#B04A76",
    "NCD": "#2676A8",
}
DASHES = ("", "10 6", "10 4 2 4", "3 4", "12 4 3 4", "5 3")


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    half_width = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return centre - half_width, centre + half_width


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_roc(
    output: Path,
    curves: list[tuple[str, np.ndarray, np.ndarray, float]],
    negative_count: int,
    positive_count: int,
    dataset_label: str,
    method_label: str,
) -> None:
    width, height = 960, 640
    left, top, plot_width, plot_height = 92, 96, 560, 430

    def point(x_value: float, y_value: float) -> tuple[float, float]:
        return left + x_value * plot_width, top + (1.0 - y_value) * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#242A31}.tick{font-size:13px}.legend{font-size:13px}.note{font-size:14px}</style>',
        '<text x="92" y="38" font-size="23" font-weight="700">Exp 2 pooled ROC: intervention-grounded silent drift</text>',
        '<text x="92" y="65" font-size="14" fill="#5A6470">Scenario-level median across explainers · five seeds · descriptive same-slice analysis</text>',
    ]
    for value in np.linspace(0.0, 1.0, 6):
        x, y = point(float(value), float(value))
        svg.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" stroke="#D9DEE5"/>')
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#D9DEE5"/>')
        svg.append(f'<text class="tick" x="{x:.1f}" y="{top + plot_height + 24}" text-anchor="middle">{value:.1f}</text>')
        svg.append(f'<text class="tick" x="{left - 16}" y="{y + 5:.1f}" text-anchor="end">{value:.1f}</text>')
    svg.extend(
        [
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333A43" stroke-width="1.5"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333A43" stroke-width="1.5"/>',
            f'<text x="{left + plot_width / 2:.1f}" y="{top + plot_height + 58}" text-anchor="middle" font-size="16">False-positive rate</text>',
            f'<text x="28" y="{top + plot_height / 2:.1f}" text-anchor="middle" font-size="16" transform="rotate(-90 28 {top + plot_height / 2:.1f})">True-positive rate</text>',
        ]
    )
    x0, y0 = point(0.0, 0.0)
    x1, y1 = point(1.0, 1.0)
    svg.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#6B7280" stroke-width="1.5" stroke-dasharray="7 6"/>')
    for index, (label, fpr, tpr, auroc) in enumerate(curves):
        coordinates = [point(float(x), float(y)) for x, y in zip(fpr, tpr)]
        path = " ".join(("M" if offset == 0 else "L") + f" {x:.1f} {y:.1f}" for offset, (x, y) in enumerate(coordinates))
        dash = f' stroke-dasharray="{DASHES[index]}"' if DASHES[index] else ""
        svg.append(f'<path d="{path}" fill="none" stroke="{COLORS[label]}" stroke-width="2.5"{dash}/>')
        legend_y = 128 + index * 38
        svg.append(f'<line x1="690" y1="{legend_y}" x2="730" y2="{legend_y}" stroke="{COLORS[label]}" stroke-width="2.5"{dash}/>')
        svg.append(f'<text class="legend" x="742" y="{legend_y + 5}">{escape(label)} (AUC {auroc:.3f})</text>')
    svg.extend(
        [
            '<rect x="118" y="225" width="305" height="113" rx="8" fill="#FFFFFF" stroke="#BCC5CF"/>',
            '<text class="note" x="137" y="252" font-weight="700">Detector ranking is descriptive</text>',
            f'<text class="note" x="137" y="278">{negative_count} N1/N2 controls · {positive_count} eligible positives</text>',
            f'<text class="note" x="137" y="304">{escape(dataset_label)} · {escape(method_label)} · seeds 1/2/3/4/42</text>',
            '<text class="note" x="137" y="330">Internal held-out split; no external calibration</text>',
            '<text x="92" y="618" font-size="12.5" fill="#5A6470">Positive set: eligible intervention-positive and silent scenarios; true but non-silent constructions are excluded by definition.</text>',
            '</svg>',
        ]
    )
    output.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ncd/results/exp2_v2/nsl-kdd/replay")
    parser.add_argument("--output", default="ncd/results/exp2_v2/nsl-kdd/replay/multiseed")
    parser.add_argument(
        "--seed-dir",
        action="append",
        default=[],
        metavar="SEED=DIR",
        help="override a seed directory name, for example 1=seed1_matched",
    )
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    seed_dirs = {seed: f"seed{seed}" for seed in SEEDS}
    for mapping in args.seed_dir:
        seed_text, separator, directory = mapping.partition("=")
        if not separator or int(seed_text) not in seed_dirs or not directory:
            raise ValueError(f"invalid --seed-dir mapping: {mapping!r}")
        seed_dirs[int(seed_text)] = directory

    consensus_rows: list[dict[str, object]] = []
    measurement_rows: list[dict[str, str]] = []
    scenario_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    max_n2_error = 0.0
    dataset_labels: set[str] = set()
    method_labels: set[str] = set()
    scope_mismatches: list[int] = []
    p2_construction_statuses: set[str] = set()
    p2_screen_results: set[str] = set()

    for seed in SEEDS:
        seed_dir = root / seed_dirs[seed]
        result = json.loads((seed_dir / "exp2_groundtruth_result.json").read_text(encoding="utf-8"))
        dataset_labels.add(str(result["effective_config"]["dataset"]))
        method_labels.add(str(result["effective_config"]["method"]))
        p2_construction = result.get("p2_construction", {})
        p2_construction_statuses.add(str(p2_construction.get("status", "legacy_unspecified")))
        if p2_construction.get("screen_result"):
            p2_screen_results.add(str(p2_construction["screen_result"]))
        if f"seed {seed}" not in str(result.get("scope", "")):
            scope_mismatches.append(seed)
        with (seed_dir / "exp2_groundtruth_table.csv").open(newline="", encoding="utf-8") as handle:
            seed_measurements = list(csv.DictReader(handle))
        measurement_rows.extend({"seed": str(seed), **row} for row in seed_measurements)

        scenario_lookup: dict[tuple[str, float], dict[str, object]] = {}
        for record in result["scenario_records"]:
            strength = float(record.get("rho", record.get("mask_fraction", 1.0)))
            scenario_lookup[(str(record["condition"]), strength)] = record
            max_n2_error = max(max_n2_error, float(record["n2_max_abs_logit_error"]))

        seed_consensus = []
        for row in result["consensus"]:
            normalized = {"seed": seed, **row}
            seed_consensus.append(normalized)
            consensus_rows.append(normalized)
        negative = [row for row in seed_consensus if str(row["condition"]).startswith("N")]
        eligible_p1 = [
            row
            for row in seed_consensus
            if row["condition"] == "P1_shortcut"
            and float(row["strength"]) >= 0.9
            and parse_bool(row["truth_positive"])
            and parse_bool(row["silent_eligible"])
        ]
        eligible_p2 = [
            row
            for row in seed_consensus
            if row["condition"] == "P2_feature_mask"
            and parse_bool(row["truth_positive"])
            and parse_bool(row["silent_eligible"])
        ]
        seed_rows.append(
            {
                "seed": seed,
                "negative_controls": len(negative),
                "ncd_false_positives": sum(parse_bool(row["ncd_consensus_drift"]) for row in negative),
                "b2_false_positives": sum(parse_bool(row["b2_consensus_drift"]) for row in negative),
                "eligible_strong_p1": len(eligible_p1),
                "ncd_p1_detected": sum(parse_bool(row["ncd_consensus_drift"]) for row in eligible_p1),
                "b2_p1_detected": sum(parse_bool(row["b2_consensus_drift"]) for row in eligible_p1),
                "eligible_p2": len(eligible_p2),
                "ncd_p2_detected": sum(parse_bool(row["ncd_consensus_drift"]) for row in eligible_p2),
                "b2_p2_detected": sum(parse_bool(row["b2_consensus_drift"]) for row in eligible_p2),
            }
        )

        for row in seed_consensus:
            condition = str(row["condition"])
            strength = float(row["strength"])
            if condition.startswith("N"):
                population = "negative_control"
            elif parse_bool(row["truth_positive"]) and parse_bool(row["silent_eligible"]):
                population = "eligible_silent_positive"
            else:
                population = "excluded_non_silent_or_non_positive"
            scenario = scenario_lookup.get((condition, strength), {})
            scenario_rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "strength": strength,
                    "population": population,
                    "truth_positive": parse_bool(row["truth_positive"]),
                    "silent_eligible": parse_bool(row["silent_eligible"]),
                    "recall_delta": scenario.get("recall_delta", ""),
                    "intervention_effect_change": scenario.get(
                        "intervention_effect_change",
                        (
                            float(scenario.get("checkpoint1", {}).get("recall_drop", 0.0))
                            - float(scenario.get("checkpoint0", {}).get("recall_drop", 0.0))
                            if scenario
                            else ""
                        ),
                    ),
                    "b1_detected": parse_bool(row["b1_consensus_drift"]),
                    "b2_detected": parse_bool(row["b2_consensus_drift"]),
                    "ncd_detected": parse_bool(row["ncd_consensus_drift"]),
                    "ncd_votes": int(row["ncd_explainer_votes"]),
                }
            )

    eligible = [
        row
        for row in consensus_rows
        if str(row["condition"]).startswith("N")
        or (parse_bool(row["truth_positive"]) and parse_bool(row["silent_eligible"]))
    ]
    negatives = [row for row in eligible if str(row["condition"]).startswith("N")]
    positives = [row for row in eligible if str(row["condition"]).startswith("P")]
    positive_p1 = [row for row in positives if row["condition"] == "P1_shortcut" and float(row["strength"]) >= 0.9]
    positive_p2 = [row for row in positives if row["condition"] == "P2_feature_mask"]
    true_non_silent_p1 = [
        row
        for row in consensus_rows
        if row["condition"] == "P1_shortcut"
        and parse_bool(row["truth_positive"])
        and not parse_bool(row["silent_eligible"])
    ]
    excluded_p2 = [
        row
        for row in consensus_rows
        if row["condition"] == "P2_feature_mask"
        and not (parse_bool(row["truth_positive"]) and parse_bool(row["silent_eligible"]))
    ]
    if len(dataset_labels) != 1 or len(method_labels) != 1:
        raise RuntimeError(f"mixed dataset/method aggregation: {dataset_labels}, {method_labels}")
    dataset_label = next(iter(dataset_labels))
    method_label = next(iter(method_labels))

    detector_fields = {
        "B1 fixed J@15": "b1_consensus_drift",
        "B2 repeat-mean J@15": "b2_consensus_drift",
        "NCD": "ncd_consensus_drift",
        "NCD no bootstrap": "ncd_no_bootstrap_consensus_drift",
        "NCD no repeats": "ncd_no_repeats_consensus_drift",
        "NCD no FDR": "ncd_no_fdr_consensus_drift",
    }
    detector_rows: list[dict[str, object]] = []
    for detector, field in detector_fields.items():
        fp = sum(parse_bool(row[field]) for row in negatives)
        tp_p1 = sum(parse_bool(row[field]) for row in positive_p1)
        tp_p2 = sum(parse_bool(row[field]) for row in positive_p2)
        fpr_low, fpr_high = wilson_interval(fp, len(negatives))
        p1_low, p1_high = wilson_interval(tp_p1, len(positive_p1))
        p2_low, p2_high = wilson_interval(tp_p2, len(positive_p2))
        detector_rows.append(
            {
                "detector": detector,
                "negative_count": len(negatives),
                "false_positives": fp,
                "observed_fpr": fp / len(negatives),
                "fpr_wilson95_low": fpr_low,
                "fpr_wilson95_high": fpr_high,
                "eligible_strong_p1_count": len(positive_p1),
                "strong_p1_detected": tp_p1,
                "observed_strong_p1_tpr": tp_p1 / len(positive_p1),
                "strong_p1_tpr_wilson95_low": p1_low,
                "strong_p1_tpr_wilson95_high": p1_high,
                "eligible_p2_count": len(positive_p2),
                "p2_detected": tp_p2,
                "observed_p2_tpr": tp_p2 / len(positive_p2) if positive_p2 else None,
                "p2_tpr_wilson95_low": p2_low,
                "p2_tpr_wilson95_high": p2_high,
            }
        )

    grouped: dict[tuple[int, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in measurement_rows:
        is_negative = row["condition"].startswith("N")
        is_positive = row["condition"].startswith("P") and parse_bool(row["truth_positive"]) and parse_bool(row["silent_eligible"])
        if is_negative or is_positive:
            grouped[(int(row["seed"]), row["condition"], row["strength"])].append(row)
    labels = np.asarray([0 if key[1].startswith("N") else 1 for key in grouped], dtype=np.int64)
    if int((labels == 0).sum()) != len(negatives) or int((labels == 1).sum()) != len(positives):
        raise RuntimeError("consensus and measurement ROC populations disagree")

    roc_rows: list[dict[str, object]] = []
    curves = []
    for label, getter in METRICS.items():
        scores = np.asarray(
            [np.median([getter(row) for row in rows]) for rows in grouped.values()],
            dtype=np.float64,
        )
        if not np.isfinite(scores).all():
            raise RuntimeError(f"non-finite pooled scores for {label}")
        fpr, tpr, _ = roc_curve(labels, scores)
        auroc = float(roc_auc_score(labels, scores))
        roc_rows.append(
            {
                "metric": label,
                "scenario_negative_count": int((labels == 0).sum()),
                "scenario_positive_count": int((labels == 1).sum()),
                "auroc": auroc,
            }
        )
        curves.append((label, fpr, tpr, auroc))

    write_csv(output / "exp2_multiseed_per_seed.csv", seed_rows)
    write_csv(output / "exp2_multiseed_scenarios.csv", scenario_rows)
    write_csv(output / "exp2_multiseed_detector_summary.csv", detector_rows)
    write_csv(output / "exp2_multiseed_roc_summary.csv", roc_rows)
    draw_roc(
        output / "exp2_multiseed_roc.svg",
        curves,
        len(negatives),
        len(positives),
        dataset_label,
        method_label,
    )

    ncd = next(row for row in detector_rows if row["detector"] == "NCD")
    b1 = next(row for row in detector_rows if row["detector"] == "B1 fixed J@15")
    b2 = next(row for row in detector_rows if row["detector"] == "B2 repeat-mean J@15")
    no_bootstrap = next(row for row in detector_rows if row["detector"] == "NCD no bootstrap")
    no_repeats = next(row for row in detector_rows if row["detector"] == "NCD no repeats")
    no_fdr = next(row for row in detector_rows if row["detector"] == "NCD no FDR")
    p2_available = bool(positive_p2)

    def count_rate(row: dict[str, object], detected: str, total: str, rate: str) -> str:
        denominator = int(row[total])
        if denominator == 0:
            return "n/a"
        return f"{int(row[detected])}/{denominator} ({100 * float(row[rate]):.1f}%)"

    def count_only(row: dict[str, object], detected: str, total: str) -> str:
        denominator = int(row[total])
        return "n/a" if denominator == 0 else f"{int(row[detected])}/{denominator}"

    p2_scope_text = (
        "the screened P2 top-3 feature-mask construction"
        if p2_available
        else "no P2 detector case because the frozen P2 screen produced no eligible construction"
    )
    p2_screen_note = (
        "P2 candidate selection and held-out evaluation use disjoint row halves, but both come from the same official dataset and do not constitute external validation."
        if p2_available
        else f"Frozen P2 screen status: {', '.join(sorted(p2_construction_statuses))}; screen evidence: {', '.join(sorted(p2_screen_results)) or 'recorded in the raw results'}."
    )
    ablation_p2_sentence = (
        f" and P2 from {int(ncd['p2_detected'])}/{int(ncd['eligible_p2_count'])} to {int(no_bootstrap['p2_detected'])}/{int(no_bootstrap['eligible_p2_count'])}"
        if p2_available
        else ""
    )
    p2_population_note = (
        f"P2 cases failing the joint intervention-positive-and-silent eligibility rule: {len(excluded_p2)}. They were also excluded."
        if p2_available or excluded_p2
        else "P2 has no detector denominator because the frozen screen found no eligible construction."
    )
    point_gate_passes = (
        float(ncd["observed_fpr"]) <= 0.05
        and float(ncd["observed_strong_p1_tpr"]) >= float(b2["observed_strong_p1_tpr"])
    )
    gate_text = "passes" if point_gate_passes else "does not pass"
    available_evidence_text = (
        "This is bounded evidence that the available benchmark constructions and silent-drift measurement operate on this slice."
        if point_gate_passes
        else "This slice does not satisfy the frozen detector gate."
    )
    report = f"""# Exp 2 five-seed intervention-ground-truth result

## What ran

{dataset_label} {method_label}, seeds 1/2/3/4/42, transition 0->1, N1 same-checkpoint controls, N2 function-preserving controls, strong P1 shortcuts (rho >= 0.9), and {p2_scope_text}. Each detector uses three explainers. {p2_screen_note}

## Main numbers

| Quantity | NCD | B2 repeat mean |
|---|---:|---:|
| Negative controls | {int(ncd['false_positives'])}/{int(ncd['negative_count'])} false positives ({100 * float(ncd['observed_fpr']):.1f}%) | {int(b2['false_positives'])}/{int(b2['negative_count'])} ({100 * float(b2['observed_fpr']):.1f}%) |
| Eligible strong P1 | {int(ncd['strong_p1_detected'])}/{int(ncd['eligible_strong_p1_count'])} ({100 * float(ncd['observed_strong_p1_tpr']):.1f}%) | {int(b2['strong_p1_detected'])}/{int(b2['eligible_strong_p1_count'])} ({100 * float(b2['observed_strong_p1_tpr']):.1f}%) |
| Eligible P2 | {count_rate(ncd, 'p2_detected', 'eligible_p2_count', 'observed_p2_tpr')} | {count_rate(b2, 'p2_detected', 'eligible_p2_count', 'observed_p2_tpr')} |

Across the five seeds, {len(true_non_silent_p1)} intervention-positive P1 cases were non-silent (recall drop exceeded 5 percentage points), so they are excluded from the silent-positive TPR denominator rather than counted as detector misses. {p2_population_note} Maximum N2 logit error was {max_n2_error:.3g}.

| Ablation | False positives | Strong P1 detected | P2 detected |
|---|---:|---:|---:|
| Full NCD | {int(ncd['false_positives'])}/{int(ncd['negative_count'])} | {int(ncd['strong_p1_detected'])}/{int(ncd['eligible_strong_p1_count'])} | {count_only(ncd, 'p2_detected', 'eligible_p2_count')} |
| No probe bootstrap | {int(no_bootstrap['false_positives'])}/{int(no_bootstrap['negative_count'])} | {int(no_bootstrap['strong_p1_detected'])}/{int(no_bootstrap['eligible_strong_p1_count'])} | {count_only(no_bootstrap, 'p2_detected', 'eligible_p2_count')} |
| No explainer repeats | {int(no_repeats['false_positives'])}/{int(no_repeats['negative_count'])} | {int(no_repeats['strong_p1_detected'])}/{int(no_repeats['eligible_strong_p1_count'])} | {count_only(no_repeats, 'p2_detected', 'eligible_p2_count')} |
| No FDR | {int(no_fdr['false_positives'])}/{int(no_fdr['negative_count'])} | {int(no_fdr['strong_p1_detected'])}/{int(no_fdr['eligible_strong_p1_count'])} | {count_only(no_fdr, 'p2_detected', 'eligible_p2_count')} |

## Decision

The frozen point-estimate gate {gate_text}: NCD FPR is {100 * float(ncd['observed_fpr']):.1f}%, strong-P1 detection is {int(ncd['strong_p1_detected'])}/{int(ncd['eligible_strong_p1_count'])}, B2 is {int(b2['strong_p1_detected'])}/{int(b2['eligible_strong_p1_count'])}, and B1 is {int(b1['strong_p1_detected'])}/{int(b1['eligible_strong_p1_count'])}. All six continuous scores are reported as pooled descriptive AUROC values, not external validation. The 95% Wilson upper bound for the observed {int(ncd['false_positives'])}/{len(negatives)} FPR is {100 * float(ncd['fpr_wilson95_high']):.1f}%, so these data do not by themselves establish a population FPR below 5%. The pooled ROC is descriptive and not externally calibrated. Removing bootstrap changes P1 detection from {int(ncd['strong_p1_detected'])}/{int(ncd['eligible_strong_p1_count'])} to {int(no_bootstrap['strong_p1_detected'])}/{int(no_bootstrap['eligible_strong_p1_count'])}{ablation_p2_sentence}; the remaining ablation counts are reported above.

{available_evidence_text} It is not evidence of substantive innovation, NCD superiority, broad method generality, or publication readiness.

{f"Scope-label mismatches remain in raw completed seeds {scope_mismatches}; their effective configs and tables identify the actual seed, and the artifacts were not rewritten." if scope_mismatches else "All five raw result scope labels match their effective seed."}

## Next

Complete the remaining frozen dataset/method cells required by the work order and add external threshold calibration. If B2 remains tied, the defensible conclusion is that repeat averaging is sufficient and NCD should not be claimed as the novel mechanism.
"""
    (output / "EXP2_MULTI_SEED_REPORT.md").write_text(report, encoding="utf-8")

    print(
        json.dumps(
            {
                "seeds": list(SEEDS),
                "seed_dirs": seed_dirs,
                "dataset": dataset_label,
                "method": method_label,
                "negative_controls": len(negatives),
                "eligible_strong_p1": len(positive_p1),
                "eligible_p2": len(positive_p2),
                "max_n2_logit_error": max_n2_error,
                "detector_summary": detector_rows,
                "roc": roc_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
