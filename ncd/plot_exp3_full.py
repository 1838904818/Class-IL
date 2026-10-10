"""Render the complete 140-run Exp. 3 confusion and sensitivity figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DATASETS = ("NSL-KDD", "UNSW-NB15", "CIC-IDS-2017", "CSE-CIC-IDS2018")
METHODS = (
    "naive",
    "ewc",
    "lwf",
    "replay",
    "replay-herding",
    "replay-dpmeans",
    "icarl",
)
SEEDS = (1, 2, 3, 4, 42)
COLORS = ((34, 92, 144), (213, 132, 43), (92, 120, 72), (168, 81, 118))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def merge_performance_payloads(
    payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge split dataset packages and require the registered 7 x 5 matrix."""
    merged: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        dataset = str(payload["dataset"])
        if dataset not in DATASETS:
            raise ValueError(f"unexpected dataset: {dataset}")
        if dataset not in merged:
            merged[dataset] = {
                "dataset": dataset,
                "class_order": list(payload["class_order"]),
                "runs": [],
            }
        elif list(payload["class_order"]) != merged[dataset]["class_order"]:
            raise ValueError(f"inconsistent class order for {dataset}")
        merged[dataset]["runs"].extend(payload["runs"])

    if set(merged) != set(DATASETS):
        raise ValueError(f"expected datasets {DATASETS}; got {sorted(merged)}")
    expected = {(method, seed) for method in METHODS for seed in SEEDS}
    for dataset, payload in merged.items():
        identities = [(str(run["method"]), int(run["seed"])) for run in payload["runs"]]
        if len(identities) != len(set(identities)):
            raise ValueError(f"duplicate performance identity in {dataset}")
        observed = set(identities)
        if observed != expected:
            raise ValueError(
                f"incomplete performance matrix for {dataset}: "
                f"missing={sorted(expected - observed)} extra={sorted(observed - expected)}"
            )
        payload["runs"].sort(key=lambda run: (METHODS.index(run["method"]), int(run["seed"])))
    return [merged[dataset] for dataset in DATASETS]


def pooled_dataset_confusion(
    payload: dict[str, Any],
) -> tuple[list[str], list[list[int]], list[list[float]]]:
    """Pool counts over all 35 method-seed runs, then normalize by true class."""
    labels = [str(label) for label in payload["class_order"]]
    size = len(labels)
    pooled = [[0 for _ in range(size)] for _ in range(size)]
    for run in payload["runs"]:
        matrix = run["final_confusion_matrix"]
        if len(matrix) != size or any(len(row) != size for row in matrix):
            raise ValueError(
                f"{payload['dataset']} {run['method']} seed {run['seed']} has invalid matrix shape"
            )
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                count = int(value)
                if count < 0 or count != value:
                    raise ValueError("confusion-matrix counts must be non-negative integers")
                pooled[row_index][column_index] += count

    normalized: list[list[float]] = []
    for row in pooled:
        denominator = sum(row)
        if denominator <= 0:
            raise ValueError(f"{payload['dataset']} contains an empty true-class row")
        normalized.append([100.0 * value / denominator for value in row])
    return labels, pooled, normalized


def pooled_dataset_curves(
    rows: list[dict[str, str]],
    threshold_field: str,
    drift_units_field: str,
    *,
    silence_tolerance: float = 0.05,
    top_k: int | None = None,
) -> dict[str, dict[float, float]]:
    """Pool registered unit counts by dataset rather than averaging rates."""
    totals: dict[tuple[str, float], list[int]] = {}
    for row in rows:
        if abs(float(row["silence_tolerance"]) - silence_tolerance) > 1e-12:
            continue
        if top_k is not None and int(row["top_k"]) != top_k:
            continue
        dataset = row["dataset"]
        if dataset not in DATASETS:
            raise ValueError(f"unexpected dataset in sensitivity rows: {dataset}")
        threshold = float(row[threshold_field])
        key = (dataset, threshold)
        totals.setdefault(key, [0, 0])
        totals[key][0] += int(row[drift_units_field])
        totals[key][1] += int(row["silent_units"])

    curves: dict[str, dict[float, float]] = {dataset: {} for dataset in DATASETS}
    for (dataset, threshold), (drift, silent) in totals.items():
        if silent <= 0:
            raise ValueError(f"empty denominator for {(dataset, threshold)}")
        curves[dataset][threshold] = 100.0 * drift / silent
    if any(not curve for curve in curves.values()):
        raise ValueError("missing one or more registered dataset curves")
    threshold_sets = {tuple(sorted(curve)) for curve in curves.values()}
    if len(threshold_sets) != 1:
        raise ValueError("inconsistent registered thresholds across datasets")
    return curves


def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    candidates = (
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _center(draw, xy: tuple[float, float], text: str, font, fill) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(
        (xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2),
        text,
        font=font,
        fill=fill,
    )


def _vertical_label(image, xy: tuple[int, int], text: str, font, fill) -> None:
    from PIL import Image, ImageDraw

    box = font.getbbox(text)
    width = box[2] - box[0] + 8
    height = box[3] - box[1] + 8
    label = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text((4 - box[0], 4 - box[1]), text, font=font, fill=fill)
    rotated = label.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    image.alpha_composite(
        rotated, (round(xy[0] - rotated.width / 2), xy[1] - rotated.height)
    )


def _blue(value: float) -> tuple[int, int, int]:
    proportion = max(0.0, min(1.0, value / 100.0)) ** 0.62
    low = (244, 248, 252)
    high = (25, 91, 145)
    return tuple(round(a + proportion * (b - a)) for a, b in zip(low, high))


def _draw_confusion_panel(image, draw, x: int, y: int, payload: dict[str, Any]) -> None:
    labels, _pooled, normalized = pooled_dataset_confusion(payload)
    size = len(labels)
    cell = min(76, 600 // size)
    grid = cell * size
    grid_x = x + 275
    grid_y = y + 165
    title_font = _font(28, bold=True)
    label_font = _font(18 if size <= 7 else 14)
    cell_font = _font(18 if size <= 7 else 14, bold=True)
    draw.text((x + 20, y + 10), payload["dataset"], font=title_font, fill=(31, 41, 55, 255))
    draw.text(
        (x + 20, y + 51),
        "Final checkpoint; pooled counts over 7 learners x 5 seeds",
        font=_font(17),
        fill=(82, 92, 104, 255),
    )

    for index, label in enumerate(labels):
        center = grid_y + index * cell + cell / 2
        box = draw.textbbox((0, 0), label, font=label_font)
        draw.text(
            (grid_x - 12 - (box[2] - box[0]), center - (box[3] - box[1]) / 2),
            label,
            font=label_font,
            fill=(44, 52, 64, 255),
        )
        _vertical_label(
            image,
            (round(grid_x + index * cell + cell / 2), grid_y - 9),
            label,
            label_font,
            (44, 52, 64, 255),
        )

    for row_index, row in enumerate(normalized):
        for column_index, value in enumerate(row):
            left = grid_x + column_index * cell
            top = grid_y + row_index * cell
            color = _blue(value)
            draw.rectangle(
                (left, top, left + cell, top + cell),
                fill=(*color, 255),
                outline=(255, 255, 255, 255),
                width=2,
            )
            text = "<1" if 0 < value < 1 else f"{value:.0f}"
            _center(
                draw,
                (left + cell / 2, top + cell / 2),
                text,
                cell_font,
                (255, 255, 255, 255) if value >= 45 else (28, 36, 48, 255),
            )

    axis_font = _font(18, bold=True)
    _center(
        draw,
        (grid_x + grid / 2, grid_y + grid + 29),
        "Predicted class",
        axis_font,
        (31, 41, 55, 255),
    )
    _vertical_label(
        image,
        (x + 32, round(grid_y + grid / 2 + 48)),
        "True class",
        axis_font,
        (31, 41, 55, 255),
    )


def render_confusion_figure(payloads: list[dict[str, Any]], output: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (2400, 1830), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text(
        (75, 34),
        "Full Exp. 3 final-checkpoint confusion matrices",
        font=_font(40, bold=True),
        fill=(25, 34, 47, 255),
    )
    draw.text(
        (75, 88),
        "Row-normalized percentages; each dataset pools all 35 registered method-seed runs",
        font=_font(23),
        fill=(80, 90, 104, 255),
    )
    for payload, (x, y) in zip(payloads, ((35, 145), (1210, 145), (35, 940), (1210, 940))):
        _draw_confusion_panel(image, draw, x, y, payload)
    draw.text(
        (75, 1785),
        "Cell labels are percentages of each true class after pooling raw counts. The figure summarizes class error structure, not method rankings.",
        font=_font(20),
        fill=(74, 84, 97),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output, format="PNG", dpi=(200, 200), optimize=True)


def _marker(draw, xy: tuple[float, float], index: int, color) -> None:
    x, y = xy
    radius = 9
    if index == 0:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(28, 34, 42), width=2)
    elif index == 1:
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(28, 34, 42), width=2)
    elif index == 2:
        draw.polygon(((x, y - 11), (x - 11, y + 9), (x + 11, y + 9)), fill=color, outline=(28, 34, 42))
    else:
        draw.polygon(((x, y - 11), (x - 11, y), (x, y + 11), (x + 11, y)), fill=color, outline=(28, 34, 42))


def _draw_sensitivity_panel(image, draw, x: int, y: int, title: str, x_title: str, curves) -> None:
    from PIL import Image, ImageDraw

    width, height = 700, 540
    left, right, top, bottom = x + 88, x + width - 25, y + 70, y + height - 72
    _center(draw, (x + width / 2, y + 25), title, _font(26, bold=True), (28, 37, 50))
    tick_font = _font(18)
    axis_font = _font(20, bold=True)
    for percentage in range(0, 101, 20):
        py = bottom - percentage / 100.0 * (bottom - top)
        draw.line((left, py, right, py), fill=(218, 224, 231), width=2)
        box = draw.textbbox((0, 0), str(percentage), font=tick_font)
        draw.text((left - 14 - (box[2] - box[0]), py - 10), str(percentage), font=tick_font, fill=(68, 77, 90))
    draw.line((left, top, left, bottom), fill=(65, 74, 87), width=3)
    draw.line((left, bottom, right, bottom), fill=(65, 74, 87), width=3)

    thresholds = sorted(next(iter(curves.values())))
    positions = [left + index * (right - left) / (len(thresholds) - 1) for index in range(len(thresholds))]
    for px, threshold in zip(positions, thresholds):
        _center(draw, (px, bottom + 28), f"{threshold:g}", tick_font, (68, 77, 90))
    for index, dataset in enumerate(DATASETS):
        values = [curves[dataset][threshold] for threshold in thresholds]
        points = [(px, bottom - value / 100.0 * (bottom - top)) for px, value in zip(positions, values)]
        draw.line(points, fill=COLORS[index], width=5, joint="curve")
        for point in points:
            _marker(draw, point, index, COLORS[index])
    _center(draw, ((left + right) / 2, y + height - 18), x_title, axis_font, (35, 44, 57))
    label = Image.new("RGBA", (220, 36), (255, 255, 255, 0))
    ImageDraw.Draw(label).text((0, 0), "Drift rate (%)", font=axis_font, fill=(35, 44, 57))
    rotated = label.rotate(90, expand=True)
    image.alpha_composite(rotated, (x + 8, round((top + bottom - rotated.height) / 2)))


def render_sensitivity_figure(
    ncd_rows: list[dict[str, str]],
    fixed_rows: list[dict[str, str]],
    output: Path,
) -> None:
    from PIL import Image, ImageDraw

    ncd = pooled_dataset_curves(ncd_rows, "q_threshold", "ncd_consensus_drift_units")
    b1 = pooled_dataset_curves(
        fixed_rows, "jaccard_threshold", "b1_consensus_drift_units", top_k=15
    )
    b2 = pooled_dataset_curves(
        fixed_rows, "jaccard_threshold", "b2_consensus_drift_units", top_k=15
    )
    image = Image.new("RGBA", (2300, 850), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((65, 28), "Full Exp. 3 threshold sensitivity", font=_font(38, bold=True), fill=(24, 33, 46))
    draw.text(
        (65, 78),
        "Pooled silent class-transition units at the 5-point recall tolerance; all 140 registered runs",
        font=_font(23),
        fill=(76, 86, 100),
    )
    for index, (dataset, color) in enumerate(zip(DATASETS, COLORS)):
        lx = 200 + index * 510
        ly = 145
        draw.line((lx, ly, lx + 48, ly), fill=color, width=5)
        _marker(draw, (lx + 24, ly), index, color)
        draw.text((lx + 64, ly - 12), dataset, font=_font(20), fill=(48, 57, 70))
    _draw_sensitivity_panel(image, draw, 35, 190, "NCD consensus", "FDR q threshold", ncd)
    _draw_sensitivity_panel(image, draw, 800, 190, "B1 fixed-rule consensus", "Jaccard threshold (k=15)", b1)
    _draw_sensitivity_panel(image, draw, 1565, 190, "B2 repeat-mean consensus", "Jaccard threshold (k=15)", b2)
    draw.text(
        (65, 816),
        "Rates pool counts over seven learners and five seeds per dataset. A flat saturated curve is not evidence of valid discrimination.",
        font=_font(20),
        fill=(74, 84, 97),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output, format="PNG", dpi=(200, 200), optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--performance", action="append", type=Path, required=True)
    parser.add_argument("--ncd-sensitivity", type=Path, required=True)
    parser.add_argument("--fixed-sensitivity", type=Path, required=True)
    parser.add_argument("--confusion-output", type=Path, required=True)
    parser.add_argument("--sensitivity-output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.performance]
    render_confusion_figure(merge_performance_payloads(payloads), args.confusion_output)
    render_sensitivity_figure(
        read_rows(args.ncd_sensitivity),
        read_rows(args.fixed_sensitivity),
        args.sensitivity_output,
    )
    print(f"confusion={args.confusion_output.as_posix()}")
    print(f"sensitivity={args.sensitivity_output.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
