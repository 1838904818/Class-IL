from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METHODS = ("replay", "icarl")
EXPECTED_SEEDS = [1, 2, 3, 4, 42]


def pooled_row_normalized_confusion(
    payload: dict[str, Any], method: str
) -> tuple[list[str], list[int], list[list[int]], list[list[float]]]:
    """Pool final-checkpoint counts across seeds, then normalize each true class."""
    if method not in METHODS:
        raise ValueError(f"unsupported method: {method}")
    labels = [str(label) for label in payload["class_order"]]
    size = len(labels)
    runs = sorted(
        (run for run in payload["runs"] if run["method"] == method),
        key=lambda run: int(run["seed"]),
    )
    seeds = [int(run["seed"]) for run in runs]
    if seeds != EXPECTED_SEEDS:
        raise ValueError(
            f"{payload['dataset']} {method} requires seeds {EXPECTED_SEEDS}; got {seeds}"
        )

    pooled = [[0 for _ in range(size)] for _ in range(size)]
    for run in runs:
        matrix = run["final_confusion_matrix"]
        if len(matrix) != size or any(len(row) != size for row in matrix):
            raise ValueError(
                f"{payload['dataset']} {method} seed {run['seed']} has invalid matrix shape"
            )
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                count = int(value)
                if count < 0 or count != value:
                    raise ValueError("confusion-matrix counts must be non-negative integers")
                pooled[row_index][column_index] += count

    normalized = []
    for row in pooled:
        denominator = sum(row)
        if denominator <= 0:
            raise ValueError(f"{payload['dataset']} {method} contains an empty true-class row")
        normalized.append([100.0 * value / denominator for value in row])
    return labels, seeds, pooled, normalized


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


def _blue(value: float) -> tuple[int, int, int]:
    proportion = max(0.0, min(1.0, value / 100.0)) ** 0.62
    low = (244, 248, 252)
    high = (25, 91, 145)
    return tuple(round(a + proportion * (b - a)) for a, b in zip(low, high))


def _draw_centered(draw, xy: tuple[float, float], text: str, font, fill) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.text(
        (xy[0] - width / 2, xy[1] - height / 2), text, font=font, fill=fill
    )


def _draw_vertical_label(image, xy: tuple[int, int], text: str, font, fill) -> None:
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


def _draw_panel(
    image, draw, x: int, y: int, payload: dict[str, Any], method: str
) -> None:
    labels, seeds, _pooled, normalized = pooled_row_normalized_confusion(
        payload, method
    )
    size = len(labels)
    cell = min(72, 430 // size)
    grid = cell * size
    grid_x = x + 220
    grid_y = y + 145

    title_font = _font(28, bold=True)
    label_font = _font(20 if size <= 5 else 16)
    cell_font = _font(20 if size <= 5 else 15, bold=True)
    small_font = _font(18)
    draw.text(
        (x + 20, y + 12),
        f"{payload['dataset']} — {method.upper()}",
        font=title_font,
        fill=(31, 41, 55, 255),
    )
    draw.text(
        (x + 20, y + 51),
        f"Final checkpoint; pooled counts across seeds {seeds}",
        font=small_font,
        fill=(82, 92, 104, 255),
    )

    for index, label in enumerate(labels):
        center = grid_y + index * cell + cell / 2
        right = grid_x - 12
        box = draw.textbbox((0, 0), label, font=label_font)
        draw.text(
            (right - (box[2] - box[0]), center - (box[3] - box[1]) / 2),
            label,
            font=label_font,
            fill=(44, 52, 64, 255),
        )
        _draw_vertical_label(
            image,
            (round(grid_x + index * cell + cell / 2), grid_y - 10),
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
            text_fill = (
                (255, 255, 255, 255) if value >= 45 else (28, 36, 48, 255)
            )
            _draw_centered(
                draw,
                (left + cell / 2, top + cell / 2),
                f"{value:.0f}",
                cell_font,
                text_fill,
            )

    axis_font = _font(20, bold=True)
    _draw_centered(
        draw,
        (grid_x + grid / 2, grid_y + grid + 34),
        "Predicted class",
        axis_font,
        (31, 41, 55, 255),
    )
    _draw_vertical_label(
        image,
        (x + 42, round(grid_y + grid / 2 + 55)),
        "True class",
        axis_font,
        (31, 41, 55, 255),
    )


def render_figure(payloads: list[dict[str, Any]], output: Path) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:  # pragma: no cover - environment-specific dependency
        raise RuntimeError("Pillow is required to render the static figure") from error

    if len(payloads) != 2:
        raise ValueError("exactly two dataset payloads are required")
    image = Image.new("RGBA", (2360, 1580), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = _font(40, bold=True)
    subtitle_font = _font(24)
    draw.text(
        (80, 38),
        "Interim Exp. 3 final-checkpoint confusion matrices",
        font=title_font,
        fill=(25, 34, 47, 255),
    )
    draw.text(
        (80, 92),
        "Row-normalized percentages; Replay and iCaRL only; five local seeds per panel; not the 140-run full matrix",
        font=subtitle_font,
        fill=(80, 90, 104, 255),
    )

    panel_positions = ((40, 145), (1190, 145), (40, 855), (1190, 855))
    for payload, row in zip(payloads, (0, 1)):
        for method, column in zip(METHODS, (0, 1)):
            x, y = panel_positions[row * 2 + column]
            _draw_panel(image, draw, x, y, payload, method)

    note_font = _font(21)
    draw.text(
        (80, 1534),
        "Cell labels are percentages of each true class. Source: committed performance_metrics.json files.",
        font=note_font,
        fill=(74, 84, 97, 255),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(
        output, format="PNG", dpi=(200, 200), optimize=True
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    render_figure(payloads, args.output)
    print(f"saved={args.output.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
