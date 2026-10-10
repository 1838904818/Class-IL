from __future__ import annotations

import argparse
from pathlib import Path


NODES = {
    "checkpoints": (70, 250, 410, 190, "Adjacent CIL checkpoints", "M_t and M_{t+1}\nSame frozen learner run"),
    "audit_inputs": (70, 500, 410, 210, "Shared audit inputs", "Fixed class probes X_c\nFixed Task-0 background B_0\nPer-class recall R_{c,t}"),
    "explainers": (555, 300, 480, 330, "Repeated explanation ensemble", "Expected Gradients\nFeature Ablation\nGradient × Input\n10 repeats + 10 probe bootstraps"),
    "calibration": (1110, 300, 450, 330, "Noise-calibrated test", "Between-checkpoint distance D\nWithin-checkpoint noise W\nLabel permutation p values\nBH-FDR q within each run"),
    "decision": (1640, 300, 470, 330, "Class-transition decision", "NCD: at least 2 of 3 explainers\nSilent: ΔR_c > −0.05\nReport effect, q, agreement\nNo suppression or model update"),
    "rq1": (1640, 760, 470, 145, "RQ1 measurement", "Drift rate, agreement, sensitivity"),
    "rq2": (1640, 950, 470, 145, "RQ2 matched trade-offs", "Accuracy, forgetting, class safety, cost"),
    "rq3": (1640, 1140, 470, 145, "RQ3 governance", "Downstream audit only — not validated"),
    "baselines": (555, 760, 1005, 250, "Mandatory matched controls", "B1: single-explanation Jaccard @ k=15\nB2: repeat-mean Jaccard @ k=15\nSame probes, checkpoints, explainers, seeds, and silence rule"),
}

EDGES = (
    ("checkpoints", "explainers"),
    ("audit_inputs", "explainers"),
    ("explainers", "calibration"),
    ("calibration", "decision"),
    ("explainers", "baselines"),
    ("decision", "rq1"),
    ("decision", "rq2"),
    ("decision", "rq3"),
    ("baselines", "rq1"),
)


def validate_spec() -> None:
    if set(NODES) != {
        "checkpoints",
        "audit_inputs",
        "explainers",
        "calibration",
        "decision",
        "rq1",
        "rq2",
        "rq3",
        "baselines",
    }:
        raise ValueError("architecture node set drifted")
    for source, target in EDGES:
        if source not in NODES or target not in NODES:
            raise ValueError(f"invalid architecture edge: {(source, target)}")
    text = " ".join(value[4] + " " + value[5] for value in NODES.values())
    for required in ("BH-FDR", "2 of 3", "ΔR_c", "B1", "B2", "not validated"):
        if required not in text:
            raise ValueError(f"missing required architecture concept: {required}")


def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    path = Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(path), size=size) if path.exists() else ImageFont.load_default()


def _multiline_center(draw, box, text: str, font, fill, spacing: int = 8) -> None:
    x, y, width, height = box
    bounds = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    draw.multiline_text(
        (x + (width - text_width) / 2, y + (height - text_height) / 2),
        text,
        font=font,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def _draw_node(draw, key: str) -> None:
    x, y, width, height, title, body = NODES[key]
    palette = {
        "checkpoints": ((230, 239, 248), (35, 84, 125)),
        "audit_inputs": ((230, 239, 248), (35, 84, 125)),
        "explainers": ((241, 235, 219), (139, 94, 24)),
        "calibration": ((231, 240, 224), (73, 111, 60)),
        "decision": ((236, 229, 240), (121, 66, 102)),
        "baselines": ((245, 246, 248), (83, 91, 104)),
        "rq1": ((228, 238, 248), (35, 84, 125)),
        "rq2": ((232, 241, 227), (73, 111, 60)),
        "rq3": ((246, 232, 232), (145, 61, 61)),
    }
    fill, outline = palette[key]
    draw.rounded_rectangle(
        (x, y, x + width, y + height),
        radius=18,
        fill=fill,
        outline=outline,
        width=4,
    )
    draw.rounded_rectangle(
        (x, y, x + width, y + 55),
        radius=18,
        fill=outline,
        outline=outline,
        width=1,
    )
    draw.rectangle((x, y + 35, x + width, y + 56), fill=outline)
    _multiline_center(draw, (x + 12, y + 5, width - 24, 44), title, _font(23, bold=True), (255, 255, 255))
    _multiline_center(draw, (x + 18, y + 66, width - 36, height - 78), body, _font(21), (38, 47, 59), spacing=9)


def _anchor(key: str, side: str) -> tuple[float, float]:
    x, y, width, height, *_ = NODES[key]
    return {
        "left": (x, y + height / 2),
        "right": (x + width, y + height / 2),
        "top": (x + width / 2, y),
        "bottom": (x + width / 2, y + height),
    }[side]


def _arrow(draw, start, end, *, dashed: bool = False) -> None:
    color = (76, 86, 99)
    if dashed:
        position = 0
        length = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
        while position < length:
            first = position / length
            second = min(position + 14, length) / length
            draw.line(
                (
                    start[0] + (end[0] - start[0]) * first,
                    start[1] + (end[1] - start[1]) * first,
                    start[0] + (end[0] - start[0]) * second,
                    start[1] + (end[1] - start[1]) * second,
                ),
                fill=color,
                width=5,
            )
            position += 24
    else:
        draw.line((start, end), fill=color, width=5)
    import math

    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    wing = 18
    spread = 0.55
    points = [
        end,
        (end[0] - wing * math.cos(angle - spread), end[1] - wing * math.sin(angle - spread)),
        (end[0] - wing * math.cos(angle + spread), end[1] - wing * math.sin(angle + spread)),
    ]
    draw.polygon(points, fill=color)


def render(output: Path) -> None:
    from PIL import Image, ImageDraw

    validate_spec()
    image = Image.new("RGB", (2180, 1420), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((70, 38), "NCD measurement architecture and research boundaries", font=_font(40, bold=True), fill=(24, 33, 46))
    draw.text(
        (70, 92),
        "Task-free class-incremental NIDS · adjacent-checkpoint audit · matched controls",
        font=_font(24),
        fill=(76, 86, 100),
    )

    _arrow(draw, _anchor("checkpoints", "right"), _anchor("explainers", "left"))
    _arrow(draw, _anchor("audit_inputs", "right"), (555, 535))
    _arrow(draw, _anchor("explainers", "right"), _anchor("calibration", "left"))
    _arrow(draw, _anchor("calibration", "right"), _anchor("decision", "left"))
    _arrow(draw, _anchor("explainers", "bottom"), _anchor("baselines", "top"), dashed=True)
    _arrow(draw, _anchor("decision", "bottom"), _anchor("rq1", "top"))
    _arrow(draw, _anchor("rq1", "bottom"), _anchor("rq2", "top"))
    _arrow(draw, _anchor("rq2", "bottom"), _anchor("rq3", "top"), dashed=True)
    _arrow(draw, _anchor("baselines", "right"), _anchor("rq1", "left"))

    for key in NODES:
        _draw_node(draw, key)

    draw.rounded_rectangle((70, 1315, 2110, 1380), radius=14, fill=(246, 247, 249), outline=(153, 161, 171), width=2)
    _multiline_center(
        draw,
        (90, 1320, 2000, 54),
        "Research object: prediction, forgetting, silent explanation drift, and non-suppressive governance — not NCD, OFRA, BARE, SHAP, or a learner swap.",
        _font(21, bold=True),
        (52, 61, 73),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", dpi=(200, 200), optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.output)
    print(f"saved={args.output.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
