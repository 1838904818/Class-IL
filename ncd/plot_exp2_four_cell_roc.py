from __future__ import annotations

import argparse
import csv
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path


SERIES = (
    ("B1 single J@15", "#355C7D"),
    ("B2 repeat-mean J@15", "#D4A017"),
    ("B3 CLEX ED", "#C96A3D"),
    ("B4 cosine", "#7A8F3A"),
    ("B4 Kendall", "#B04A76"),
    ("NCD", "#2676A8"),
)
DASH_PATTERNS = ((100000, 0), (18, 9), (14, 6, 3, 6), (4, 7), (18, 6, 4, 6), (8, 5))
SOURCE_LEFT = 92.0
SOURCE_TOP = 96.0
SOURCE_WIDTH = 560.0
SOURCE_HEIGHT = 430.0


def parse_roc_svg(path: Path) -> dict[str, list[tuple[float, float]]]:
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    by_color = {color.lower(): label for label, color in SERIES}
    curves: dict[str, list[tuple[float, float]]] = {}
    for element in root.findall("svg:path", namespace):
        color = str(element.attrib.get("stroke", "")).lower()
        label = by_color.get(color)
        if not label or element.attrib.get("fill") != "none":
            continue
        coordinates = [
            (float(x_value), float(y_value))
            for x_value, y_value in re.findall(
                r"[ML]\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",
                element.attrib["d"],
            )
        ]
        normalized = [
            (
                (x_value - SOURCE_LEFT) / SOURCE_WIDTH,
                1.0 - (y_value - SOURCE_TOP) / SOURCE_HEIGHT,
            )
            for x_value, y_value in coordinates
        ]
        if len(normalized) < 2 or any(
            not (0.0 <= x_value <= 1.0 and 0.0 <= y_value <= 1.0)
            for x_value, y_value in normalized
        ):
            raise ValueError(f"invalid ROC path for {label} in {path}")
        curves[label] = normalized
    expected = {label for label, _ in SERIES}
    if set(curves) != expected:
        raise ValueError(f"missing ROC series in {path}: {sorted(expected - set(curves))}")
    return curves


def read_summary(path: Path) -> tuple[int, int, dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if {row["metric"] for row in rows} != {label for label, _ in SERIES}:
        raise ValueError(f"unexpected ROC summary metrics in {path}")
    negatives = {int(row["scenario_negative_count"]) for row in rows}
    positives = {int(row["scenario_positive_count"]) for row in rows}
    if len(negatives) != 1 or len(positives) != 1:
        raise ValueError(f"inconsistent ROC populations in {path}")
    aucs = {row["metric"]: float(row["auroc"]) for row in rows}
    return negatives.pop(), positives.pop(), aucs


def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    path = Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(path), size=size) if path.exists() else ImageFont.load_default()


def _draw_dashed_segment(draw, start, end, fill, width, pattern) -> None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return
    if pattern[1] == 0:
        draw.line((start, end), fill=fill, width=width)
        return
    position = 0.0
    pattern_index = 0
    draw_on = True
    while position < length:
        step = min(float(pattern[pattern_index]), length - position)
        if draw_on:
            first = position / length
            second = (position + step) / length
            draw.line(
                (
                    start[0] + dx * first,
                    start[1] + dy * first,
                    start[0] + dx * second,
                    start[1] + dy * second,
                ),
                fill=fill,
                width=width,
            )
        position += step
        pattern_index = (pattern_index + 1) % len(pattern)
        draw_on = not draw_on


def _draw_polyline(draw, points, fill, width, pattern) -> None:
    for start, end in zip(points, points[1:]):
        _draw_dashed_segment(draw, start, end, fill, width, pattern)


def _draw_panel(image, draw, x: int, y: int, title: str, source: Path, summary: Path) -> None:
    curves = parse_roc_svg(source)
    negatives, positives, aucs = read_summary(summary)
    if any(abs(value - 1.0) > 1e-12 for value in aucs.values()):
        raise ValueError(f"four-cell figure expects the frozen all-AUC=1 result: {summary}")

    width, height = 910, 565
    left, right, top, bottom = x + 92, x + 690, y + 84, y + 475
    draw.text((x + 20, y + 10), title, font=_font(27, bold=True), fill=(28, 37, 50))
    draw.text(
        (x + 20, y + 47),
        f"{negatives} negative controls · {positives} eligible silent positives",
        font=_font(18),
        fill=(78, 88, 101),
    )
    tick_font = _font(16)
    for index in range(6):
        value = index / 5
        px = left + value * (right - left)
        py = bottom - value * (bottom - top)
        draw.line((px, top, px, bottom), fill=(221, 226, 232), width=1)
        draw.line((left, py, right, py), fill=(221, 226, 232), width=1)
        label = f"{value:.1f}"
        draw.text((px - 12, bottom + 10), label, font=tick_font, fill=(68, 77, 90))
        draw.text((left - 42, py - 9), label, font=tick_font, fill=(68, 77, 90))
    draw.line((left, top, left, bottom), fill=(54, 63, 75), width=3)
    draw.line((left, bottom, right, bottom), fill=(54, 63, 75), width=3)
    _draw_dashed_segment(draw, (left, bottom), (right, top), (112, 120, 130), 2, (10, 8))

    for index, (label, color) in enumerate(SERIES):
        points = [
            (left + fpr * (right - left), bottom - tpr * (bottom - top))
            for fpr, tpr in curves[label]
        ]
        _draw_polyline(draw, points, color, 4, DASH_PATTERNS[index])
    draw.text((left + 190, bottom + 45), "False-positive rate", font=_font(19, bold=True), fill=(35, 44, 57))

    from PIL import Image, ImageDraw

    label_image = Image.new("RGBA", (190, 34), (255, 255, 255, 0))
    ImageDraw.Draw(label_image).text((0, 0), "True-positive rate", font=_font(19, bold=True), fill=(35, 44, 57))
    rotated = label_image.rotate(90, expand=True)
    image.alpha_composite(rotated, (x + 8, round((top + bottom - rotated.height) / 2)))


def render_figure(cells: list[tuple[str, Path, Path]], output: Path) -> None:
    from PIL import Image, ImageDraw

    if len(cells) != 4:
        raise ValueError("exactly four Exp 2 cells are required")
    image = Image.new("RGBA", (1960, 1430), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((65, 28), "Exp. 2 intervention-grounded ROC", font=_font(39, bold=True), fill=(24, 33, 46))
    draw.text(
        (65, 80),
        "Five-seed descriptive same-slice analysis; all six scores have AUROC 1.000 in every cell",
        font=_font(23),
        fill=(76, 86, 100),
    )

    legend_y = 145
    for index, (label, color) in enumerate(SERIES):
        column = index % 3
        row = index // 3
        lx = 130 + column * 620
        ly = legend_y + row * 38
        _draw_dashed_segment(draw, (lx, ly), (lx + 55, ly), color, 4, DASH_PATTERNS[index])
        draw.text((lx + 70, ly - 11), label, font=_font(19), fill=(48, 57, 70))

    positions = ((45, 230), (1005, 230), (45, 800), (1005, 800))
    for position, (title, source, summary) in zip(positions, cells):
        _draw_panel(image, draw, *position, title, source, summary)
    draw.text(
        (65, 1390),
        "Perfect same-slice separation validates the constructed contrast, not NCD superiority or external calibration.",
        font=_font(21),
        fill=(74, 84, 97),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output, format="PNG", dpi=(200, 200), optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", action="append", nargs=3, metavar=("TITLE", "SVG", "SUMMARY"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cells = [(title, Path(svg), Path(summary)) for title, svg, summary in args.cell]
    render_figure(cells, args.output)
    print(f"saved={args.output.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
