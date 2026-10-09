"""Render the exploratory Exp 2 ROC figure from the inspectable result table."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


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

DASHES = ["", "10 6", "10 4 2 4", "3 4", "12 4 3 4", "5 3"]


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    table_path = Path(args.table)
    with table_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        is_negative = row["condition"].startswith("N")
        is_positive = (
            row["condition"].startswith("P")
            and parse_bool(row["truth_positive"])
            and parse_bool(row["silent_eligible"])
        )
        if is_negative or is_positive:
            grouped[(row["condition"], row["strength"])].append(row)

    labels = np.asarray(
        [0 if condition.startswith("N") else 1 for condition, _ in grouped],
        dtype=np.int64,
    )
    if int((labels == 0).sum()) != 10 or int((labels == 1).sum()) != 3:
        raise RuntimeError(
            f"unexpected ROC population: negatives={(labels == 0).sum()} positives={(labels == 1).sum()}"
        )

    summary_rows: list[dict[str, object]] = []
    curves: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    for index, (label, getter) in enumerate(METRICS.items()):
        scores = np.asarray(
            [np.median([getter(row) for row in scenario_rows]) for scenario_rows in grouped.values()],
            dtype=np.float64,
        )
        fpr, tpr, _ = roc_curve(labels, scores)
        auroc = float(roc_auc_score(labels, scores))
        summary_rows.append(
            {
                "metric": label,
                "scenario_negative_count": int((labels == 0).sum()),
                "scenario_positive_count": int((labels == 1).sum()),
                "auroc": auroc,
            }
        )
        curves.append((label, fpr, tpr, auroc))

    width, height = 960, 640
    left, top, plot_width, plot_height = 92, 96, 560, 430

    def point(x_value: float, y_value: float) -> tuple[float, float]:
        return (
            left + float(x_value) * plot_width,
            top + (1.0 - float(y_value)) * plot_height,
        )

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#242A31}.tick{font-size:13px}.legend{font-size:13px}.note{font-size:14px}</style>',
        '<text x="92" y="38" font-size="23" font-weight="700">Exp 2 exploratory ROC: intervention-grounded silent drift</text>',
        '<text x="92" y="65" font-size="14" fill="#5A6470">Scenario-level median across explainers · same-slice pilot · no external calibration</text>',
    ]
    for value in np.linspace(0.0, 1.0, 6):
        x, y = point(value, value)
        svg.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" stroke="#D9DEE5" stroke-width="1"/>'
        )
        svg.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#D9DEE5" stroke-width="1"/>'
        )
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
    svg.append(
        f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#6B7280" stroke-width="1.5" stroke-dasharray="7 6"/>'
    )
    for index, (label, fpr, tpr, auroc) in enumerate(curves):
        coordinates = [point(x_value, y_value) for x_value, y_value in zip(fpr, tpr)]
        path = " ".join(
            ("M" if offset == 0 else "L") + f" {x_value:.1f} {y_value:.1f}"
            for offset, (x_value, y_value) in enumerate(coordinates)
        )
        dash = f' stroke-dasharray="{DASHES[index]}"' if DASHES[index] else ""
        svg.append(
            f'<path d="{path}" fill="none" stroke="{COLORS[label]}" stroke-width="2.5"{dash}/>'
        )
        for x_value, y_value in coordinates:
            svg.append(
                f'<circle cx="{x_value:.1f}" cy="{y_value:.1f}" r="4" fill="#FFFFFF" stroke="{COLORS[label]}" stroke-width="1.8"/>'
            )
        legend_y = 128 + index * 38
        svg.append(
            f'<line x1="690" y1="{legend_y}" x2="730" y2="{legend_y}" stroke="{COLORS[label]}" stroke-width="2.5"{dash}/>'
        )
        svg.append(
            f'<text class="legend" x="742" y="{legend_y + 5}">{escape(label)} (AUC {auroc:.2f})</text>'
        )
    svg.extend(
        [
            '<rect x="118" y="225" width="290" height="93" rx="8" fill="#FFFFFF" stroke="#BCC5CF"/>',
            '<text class="note" x="137" y="252" font-weight="700">All six curves overlap exactly</text>',
            '<text class="note" x="137" y="278">10 N1/N2 controls · 3 silent positives</text>',
            '<text class="note" x="137" y="304">NSL-KDD · Replay · seed 1</text>',
            '<text x="92" y="618" font-size="12.5" fill="#5A6470">Positive set: P1 rho ≥ 0.9 plus the causally screened P2; exploratory selection and evaluation use the same dataset.</text>',
            "</svg>",
        ]
    )
    Path(args.output).write_text("\n".join(svg), encoding="utf-8")

    summary_path = Path(args.summary)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(
        {
            "output": str(Path(args.output)),
            "summary": str(summary_path),
            "negative_scenarios": int((labels == 0).sum()),
            "positive_scenarios": int((labels == 1).sum()),
            "aurocs": {row["metric"]: row["auroc"] for row in summary_rows},
        }
    )


if __name__ == "__main__":
    main()
