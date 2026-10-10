from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


GROUPS = (
    ("NSL-KDD", "replay"),
    ("NSL-KDD", "icarl"),
    ("UNSW-NB15", "replay"),
    ("UNSW-NB15", "icarl"),
)
COLORS = ((34, 92, 144), (213, 132, 43), (92, 120, 72), (168, 81, 118))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def pooled_curves(
    rows: list[dict[str, str]],
    threshold_field: str,
    drift_units_field: str,
    *,
    silence_tolerance: float = 0.05,
    top_k: int | None = None,
) -> dict[tuple[str, str], dict[float, float]]:
    totals: dict[tuple[str, str, float], list[int]] = {}
    for row in rows:
        if abs(float(row["silence_tolerance"]) - silence_tolerance) > 1e-12:
            continue
        if top_k is not None and int(row["top_k"]) != top_k:
            continue
        group = (row["dataset"], row["method"])
        if group not in GROUPS:
            continue
        threshold = float(row[threshold_field])
        key = (*group, threshold)
        totals.setdefault(key, [0, 0])
        totals[key][0] += int(row[drift_units_field])
        totals[key][1] += int(row["silent_units"])

    curves: dict[tuple[str, str], dict[float, float]] = {group: {} for group in GROUPS}
    for (dataset, method, threshold), (drift, silent) in totals.items():
        if silent <= 0:
            raise ValueError(f"empty denominator for {(dataset, method, threshold)}")
        curves[(dataset, method)][threshold] = 100.0 * drift / silent
    if any(not curve for curve in curves.values()):
        raise ValueError("missing one or more registered dataset-method curves")
    return curves


def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    path = Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(path), size=size) if path.exists() else ImageFont.load_default()


def _center(draw, xy: tuple[float, float], text: str, font, fill) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(
        (xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2),
        text,
        font=font,
        fill=fill,
    )


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


def _draw_panel(image, draw, x: int, y: int, title: str, x_title: str, curves) -> None:
    from PIL import Image, ImageDraw

    width, height = 700, 540
    left, right, top, bottom = x + 88, x + width - 25, y + 70, y + height - 72
    title_font = _font(26, bold=True)
    tick_font = _font(18)
    axis_font = _font(20, bold=True)
    _center(draw, (x + width / 2, y + 25), title, title_font, (28, 37, 50))

    for percentage in range(0, 101, 20):
        py = bottom - percentage / 100.0 * (bottom - top)
        draw.line((left, py, right, py), fill=(218, 224, 231), width=2)
        box = draw.textbbox((0, 0), str(percentage), font=tick_font)
        draw.text((left - 14 - (box[2] - box[0]), py - 10), str(percentage), font=tick_font, fill=(68, 77, 90))
    draw.line((left, top, left, bottom), fill=(65, 74, 87), width=3)
    draw.line((left, bottom, right, bottom), fill=(65, 74, 87), width=3)

    thresholds = sorted(next(iter(curves.values())))
    x_positions = [left + index * (right - left) / (len(thresholds) - 1) for index in range(len(thresholds))]
    for px, threshold in zip(x_positions, thresholds):
        label = f"{threshold:g}"
        _center(draw, (px, bottom + 28), label, tick_font, (68, 77, 90))

    for index, group in enumerate(GROUPS):
        values = [curves[group][threshold] for threshold in thresholds]
        points = [
            (px, bottom - value / 100.0 * (bottom - top))
            for px, value in zip(x_positions, values)
        ]
        draw.line(points, fill=COLORS[index], width=5, joint="curve")
        for point in points:
            _marker(draw, point, index, COLORS[index])

    _center(draw, ((left + right) / 2, y + height - 18), x_title, axis_font, (35, 44, 57))
    label = Image.new("RGBA", (220, 36), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text((0, 0), "Drift rate (%)", font=axis_font, fill=(35, 44, 57))
    rotated = label.rotate(90, expand=True)
    image.alpha_composite(rotated, (x + 8, round((top + bottom - rotated.height) / 2)))


def render_figure(ncd_rows, fixed_rows, output: Path) -> None:
    from PIL import Image, ImageDraw

    ncd = pooled_curves(
        ncd_rows,
        "q_threshold",
        "ncd_consensus_drift_units",
    )
    b1 = pooled_curves(
        fixed_rows,
        "jaccard_threshold",
        "b1_consensus_drift_units",
        top_k=15,
    )
    b2 = pooled_curves(
        fixed_rows,
        "jaccard_threshold",
        "b2_consensus_drift_units",
        top_k=15,
    )

    image = Image.new("RGBA", (2300, 850), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((65, 28), "Interim Exp. 3 threshold sensitivity", font=_font(38, bold=True), fill=(24, 33, 46))
    draw.text(
        (65, 78),
        "Pooled silent class–transition units at the 5-point recall tolerance; 20/140 registered runs",
        font=_font(23),
        fill=(76, 86, 100),
    )

    legend_font = _font(20)
    for index, ((dataset, method), color) in enumerate(zip(GROUPS, COLORS)):
        lx = 215 + index * 505
        ly = 145
        draw.line((lx, ly, lx + 48, ly), fill=color, width=5)
        _marker(draw, (lx + 24, ly), index, color)
        draw.text((lx + 64, ly - 12), f"{dataset} / {method}", font=legend_font, fill=(48, 57, 70))

    _draw_panel(image, draw, 35, 190, "NCD consensus", "FDR q threshold", ncd)
    _draw_panel(image, draw, 800, 190, "B1 fixed-rule consensus", "Jaccard threshold (k=15)", b1)
    _draw_panel(image, draw, 1565, 190, "B2 repeat-mean consensus", "Jaccard threshold (k=15)", b2)
    draw.text(
        (65, 816),
        "Rates pool counts across seeds 1, 2, 3, 4, and 42. A flat saturated curve is not evidence of valid discrimination.",
        font=_font(20),
        fill=(74, 84, 97),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output, format="PNG", dpi=(200, 200), optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncd-sensitivity", type=Path, required=True)
    parser.add_argument("--fixed-sensitivity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render_figure(read_rows(args.ncd_sensitivity), read_rows(args.fixed_sensitivity), args.output)
    print(f"saved={args.output.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
