from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METHOD_ORDER = (
    "naive",
    "ewc",
    "lwf",
    "replay-herding",
    "replay-dpmeans",
)
METHOD_LABELS = {
    "naive": "Naive",
    "ewc": "EWC",
    "lwf": "LwF",
    "replay-herding": "Replay-Herding",
    "replay-dpmeans": "Replay-DPMeans",
}
FIXED_COLOR = (205, 128, 45)
NCD_COLOR = (35, 91, 145)
TEXT = (31, 41, 55)
MUTED = (80, 90, 104)
GRID = (220, 225, 231)


def chart_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return reconciled method rows in a stable display order."""
    if payload.get("dataset") != "NSL-KDD":
        raise ValueError("this figure requires the NSL-KDD result")
    if int(payload["scope"]["runs"]) != 25:
        raise ValueError("this figure requires the completed 25-run slice")
    rows = {str(row["method"]): row for row in payload["method_summaries"]}
    if set(rows) != set(METHOD_ORDER):
        raise ValueError(f"method coverage mismatch: {sorted(rows)}")

    output = []
    totals = {"silent": 0, "b1": 0, "b2": 0, "ncd": 0}
    for method in METHOD_ORDER:
        row = rows[method]
        silent = int(row["silent_units"])
        b1 = int(row["b1_drift_units"])
        b2 = int(row["b2_drift_units"])
        ncd = int(row["ncd_drift_units"])
        if not (0 <= b1 <= silent and 0 <= b2 <= silent and 0 <= ncd <= silent):
            raise ValueError(f"invalid drift counts for {method}")
        if b1 != b2:
            raise ValueError(f"B1 and B2 are not identical for {method}")
        totals["silent"] += silent
        totals["b1"] += b1
        totals["b2"] += b2
        totals["ncd"] += ncd
        output.append(
            {
                "method": method,
                "label": METHOD_LABELS[method],
                "silent": silent,
                "fixed_units": b1,
                "fixed_rate": 100.0 * b1 / silent,
                "ncd_units": ncd,
                "ncd_rate": 100.0 * ncd / silent,
            }
        )

    pooled = payload["pooled_result"]
    expected = {
        "silent": int(pooled["silent_units"]),
        "b1": int(pooled["b1_drift_units"]),
        "b2": int(pooled["b2_drift_units"]),
        "ncd": int(pooled["ncd_drift_units"]),
    }
    if totals != expected:
        raise ValueError(f"method totals do not reconcile: {totals} != {expected}")
    return output


def q_curve(payload: dict[str, Any]) -> list[tuple[float, int, int, float]]:
    rows = sorted(payload["ncd_q_sensitivity"], key=lambda row: float(row["q"]))
    output = []
    for row in rows:
        q = float(row["q"])
        drift = int(row["drift_units"])
        silent = int(row["silent_units"])
        rate = 100.0 * drift / silent
        if abs(rate / 100.0 - float(row["drift_rate"])) > 1e-12:
            raise ValueError(f"q sensitivity rate mismatch at q={q}")
        output.append((q, drift, silent, rate))
    if [row[0] for row in output] != [0.01, 0.025, 0.05, 0.1]:
        raise ValueError("unexpected q sensitivity thresholds")
    return output


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


def _center(draw, xy: tuple[float, float], text: str, font, fill=TEXT) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(
        (xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2),
        text,
        font=font,
        fill=fill,
    )


def render_figure(payload: dict[str, Any], output: Path) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:  # pragma: no cover - environment-specific dependency
        raise RuntimeError("Pillow is required to render the static figure") from error

    methods = chart_rows(payload)
    q_rows = q_curve(payload)
    pooled = payload["pooled_result"]

    image = Image.new("RGB", (1900, 1050), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (70, 42),
        "NCD drift decisions saturate on the completed NSL-KDD slice",
        font=_font(42, bold=True),
        fill=TEXT,
    )
    draw.text(
        (70, 99),
        "Five methods × five seeds; 157 silent class–transition units; matched performance for these 25 runs is pending",
        font=_font(24),
        fill=MUTED,
    )

    # Panel A: primary registered decisions by method.
    draw.text((70, 168), "A  Registered primary decisions", font=_font(29, bold=True), fill=TEXT)
    left, right = 350, 1110
    top, bottom = 275, 815
    for tick in (0, 25, 50, 75, 100):
        x = left + tick / 100.0 * (right - left)
        draw.line((x, top, x, bottom), fill=GRID, width=2)
        _center(draw, (x, bottom + 26), str(tick), _font(18), MUTED)
    draw.line((left, top, left, bottom), fill=TEXT, width=3)
    draw.line((left, bottom, right, bottom), fill=TEXT, width=3)
    _center(draw, ((left + right) / 2, bottom + 66), "Drift rate (%)", _font(21, bold=True), TEXT)

    row_gap = 100
    for index, row in enumerate(methods):
        y = top + 36 + index * row_gap
        label_font = _font(21, bold=True)
        box = draw.textbbox((0, 0), row["label"], font=label_font)
        draw.text((left - 24 - (box[2] - box[0]), y + 18), row["label"], font=label_font, fill=TEXT)

        fixed_width = row["fixed_rate"] / 100.0 * (right - left)
        ncd_width = row["ncd_rate"] / 100.0 * (right - left)
        draw.rectangle((left, y, left + fixed_width, y + 27), fill=FIXED_COLOR)
        draw.rectangle((left, y + 38, left + ncd_width, y + 65), fill=NCD_COLOR)
        draw.text(
            (min(left + fixed_width + 10, right - 128), y + 1),
            f"{row['fixed_rate']:.1f}% ({row['fixed_units']}/{row['silent']})",
            font=_font(17, bold=True),
            fill=TEXT,
        )
        draw.text(
            (left + ncd_width - 128, y + 39),
            f"{row['ncd_rate']:.1f}% ({row['ncd_units']}/{row['silent']})",
            font=_font(17, bold=True),
            fill=(255, 255, 255),
        )

    draw.rectangle((410, 214, 438, 239), fill=FIXED_COLOR)
    draw.text((450, 212), "Fixed J@15 rule (B1 = B2 in this slice)", font=_font(19), fill=TEXT)
    draw.rectangle((795, 214, 823, 239), fill=NCD_COLOR)
    draw.text((835, 212), "NCD consensus", font=_font(19), fill=TEXT)

    # Panel B: NCD q sensitivity on a non-truncated 0–100% scale.
    panel_left, panel_right = 1270, 1820
    panel_top, panel_bottom = 275, 675
    draw.text((1220, 168), "B  NCD q sensitivity", font=_font(29, bold=True), fill=TEXT)
    for tick in (0, 25, 50, 75, 100):
        y = panel_bottom - tick / 100.0 * (panel_bottom - panel_top)
        draw.line((panel_left, y, panel_right, y), fill=GRID, width=2)
        box = draw.textbbox((0, 0), str(tick), font=_font(18))
        draw.text((panel_left - 35 - (box[2] - box[0]), y - 10), str(tick), font=_font(18), fill=MUTED)
    draw.line((panel_left, panel_top, panel_left, panel_bottom), fill=TEXT, width=3)
    draw.line((panel_left, panel_bottom, panel_right, panel_bottom), fill=TEXT, width=3)
    x_positions = [
        panel_left + index * (panel_right - panel_left) / (len(q_rows) - 1)
        for index in range(len(q_rows))
    ]
    points = []
    for x, (q, drift, silent, rate) in zip(x_positions, q_rows):
        y = panel_bottom - rate / 100.0 * (panel_bottom - panel_top)
        points.append((x, y))
        _center(draw, (x, panel_bottom + 27), f"{q:g}", _font(18), MUTED)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=NCD_COLOR, outline=TEXT, width=2)
        _center(draw, (x, y - 26), f"{rate:.1f}%", _font(17, bold=True), TEXT)
        _center(draw, (x, y + 25), f"{drift}/{silent}", _font(15), MUTED)
    draw.line(points, fill=NCD_COLOR, width=5)
    _center(draw, ((panel_left + panel_right) / 2, panel_bottom + 67), "BH-FDR q threshold", _font(21, bold=True), TEXT)
    draw.text((1165, 234), "Drift rate (%)", font=_font(19, bold=True), fill=TEXT)

    # Interpretation and provenance remain visible with the figure.
    draw.rounded_rectangle((1215, 770, 1825, 915), radius=12, fill=(245, 247, 250), outline=GRID, width=2)
    draw.text((1245, 792), "Interpretation boundary", font=_font(22, bold=True), fill=TEXT)
    draw.text((1245, 829), "• 68/68 future-observed units are NCD-positive.", font=_font(18), fill=TEXT)
    draw.text((1245, 859), "• No non-drift group exists for prediction claims.", font=_font(18), fill=TEXT)
    draw.text((1245, 889), "• NCD decision time is 4986.4× the fixed rule.", font=_font(18), fill=TEXT)

    draw.line((70, 945, 1830, 945), fill=GRID, width=2)
    draw.text(
        (70, 962),
        "Primary definition: Δrecall > −0.05; fixed J@15 < 0.70; NCD BH-FDR q < 0.05; ≥2 of 3 explainers.",
        font=_font(19),
        fill=MUTED,
    )
    draw.text(
        (70, 997),
        f"Source: DICC jobs 483664 and 483686; {int(pooled['ncd_drift_units'])}/{int(pooled['silent_units'])} pooled NCD-positive units. This is drift-only partial evidence, not detector validity or innovation.",
        font=_font(18),
        fill=MUTED,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", dpi=(200, 200), optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    render_figure(payload, args.output)
    print(f"saved={args.output.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
