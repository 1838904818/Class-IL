"""Plot Section26.10 summary evidence without inference or training.

Only complete five-seed groups are drawn. Curves join means at the SAME tau;
fixed-FPR table entries are copied from the separately computed per-seed
interpolation summaries, never interpolated from the displayed mean curve.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("NSL-KDD", "UNSW-NB15", "CIC-IDS-2017")
METHODS = ("replay", "wacb", "freq", "farq", "derpp")
SEEDS = (1, 2, 3, 4, 42)
TAUS = tuple(-1.0 + i / 10.0 for i in range(41))
TARGETS = (0.02, 0.05, 0.10)
STYLES = {
    "replay": ("Replay", "#42464C", "-", "o"),
    "wacb": ("WACB", "#236B9B", "--", "s"),
    "freq": ("Freq", "#B88620", "-.", "^"),
    "farq": ("FARQ", "#A74478", "-", "D"),
    "derpp": ("DER++", "#637C32", ":", "x"),
}


def read_snapshot(path: Path) -> bytes:
    """Permit an atomic replacement of the source summary on Windows."""
    if os.name != "nt":
        return path.read_bytes()
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                       wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                       wintypes.HANDLE)
    create.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = create(str(path.resolve()), 0x80000000, 0x00000007,
                    None, 3, 0x00000080, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle(handle)
        raise
    with os.fdopen(fd, "rb") as stream:
        return stream.read()


def numeric(value, label, low=0.0, high=1.0):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not low <= value <= high):
        raise ValueError(f"Invalid {label}: expected finite value in [{low}, {high}]")
    return float(value)


def stat(value, label):
    return (numeric(value["mean"], label + ".mean"),
            numeric(value["sample_sd"], label + ".sample_sd"))


def validate(summary):
    protocol = summary["protocol"]
    if (summary.get("schema_version") != 1 or protocol.get("section") != "26.10"
            or protocol.get("secondary_after_primary_failure") is not True
            or protocol.get("primary_decision_unchanged") is not True):
        raise ValueError("Expected Section26.10 secondary-after-primary-failure evidence")
    taus = protocol["taus"]
    if len(taus) != 41 or any(abs(float(a) - b) > 1e-12 for a, b in zip(taus, TAUS)):
        raise ValueError("Expected fixed tau range [-1,3] with 41 points")
    if tuple(protocol["target_fprs"]) != TARGETS:
        raise ValueError("Expected test-interpolation targets 2%, 5%, 10%")
    groups = {}
    for group in summary["groups"]:
        key = (group["dataset"], group["method"])
        if key in groups or key[0] not in DATASETS or key[1] not in METHODS:
            raise ValueError(f"Unexpected or duplicate main group: {key}")
        seeds = group["completed_seeds"]
        if len(seeds) != len(set(seeds)) or not set(seeds).issubset(SEEDS):
            raise ValueError(f"Unexpected seed membership: {key}")
        complete = set(seeds) == set(SEEDS)
        if (group["status"] == "complete") != complete:
            raise ValueError(f"Completeness status disagrees with seed membership: {key}")
        if complete:
            curve = group["mean_tau_curve"]
            if len(curve) != 41:
                raise ValueError(f"Incomplete tau curve: {key}")
            for expected_tau, point in zip(TAUS, curve):
                if abs(float(point["tau"]) - expected_tau) > 1e-12:
                    raise ValueError(f"Tau order changed: {key}")
                for metric in ("normal_fpr", "attack_detection"):
                    stat(point[metric], f"{key}.{metric}")
            for target in TARGETS:
                op = group["operating_points"][str(target)]
                if op["status"] == "available":
                    for metric in ("attack_detection", "macro_f1"):
                        stat(op[metric], f"{key}.{target}.{metric}")
                elif op["status"] != "unavailable":
                    raise ValueError(f"Unexpected fixed-FPR status: {key}")
        groups[key] = group
    return groups


def table_cell(group, target):
    """Use the recorded per-seed interpolation aggregate, not curve geometry."""
    if not group or group["status"] != "complete":
        return "Incomplete"
    op = group["operating_points"][str(target)]
    if op["status"] != "available":
        return "Unavailable"
    adr, adr_sd = stat(op["attack_detection"], "operating ADR")
    mf1, mf1_sd = stat(op["macro_f1"], "operating Macro-F1")
    return f"{100*adr:.2f} +/- {100*adr_sd:.2f}\n{100*mf1:.2f} +/- {100*mf1_sd:.2f}"


def draw_dataset(summary, groups, dataset, source_sha, output, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelcolor": "#272B30", "text.color": "#272B30",
        "axes.edgecolor": "#666B72", "xtick.color": "#454A50",
        "ytick.color": "#454A50", "svg.fonttype": "none",
        "svg.hashsalt": "section26-operating-curves", "savefig.facecolor": "white",
    })
    figure = plt.figure(figsize=(10.4, 9.8), facecolor="white")
    figure.suptitle(f"{dataset}: mean-tau FPR / attack detection", x=.08,
                   y=.975, ha="left", fontsize=16, fontweight="bold")
    figure.text(.08, .940, "Section26.10 | SECONDARY AFTER PRIMARY FAILURE | Primary decision unchanged",
                fontsize=10, color="#6A4B23")
    figure.text(.08, .916, "Final test checkpoints; tau = -1 to 3, 41 fixed points; seeds 1, 2, 3, 4, 42.",
                fontsize=10)
    axis = figure.add_axes((.08, .455, .87, .405))
    missing, drawn = [], []
    max_fpr_percent = 0.0
    for method in METHODS:
        group = groups.get((dataset, method))
        label, color, line, marker = STYLES[method]
        count = len(group["completed_seeds"]) if group else 0
        if not group or group["status"] != "complete":
            missing.append(f"{label} {count}/5")
            continue
        curve = group["mean_tau_curve"]
        x = [100 * p["normal_fpr"]["mean"] for p in curve]
        y = [100 * p["attack_detection"]["mean"] for p in curve]
        max_fpr_percent = max(max_fpr_percent, max(x))
        axis.plot(x, y, label=f"{label} (5/5)", color=color, linestyle=line,
                  linewidth=1.8, marker=marker, markersize=4,
                  markerfacecolor="white", markeredgewidth=1.0, markevery=5)
        drawn.append(method)
    for target in TARGETS:
        axis.axvline(100 * target, color="#B5BAC0", linewidth=.7, linestyle=":", zorder=0)
    x_upper = max(15, 5 * math.ceil(max_fpr_percent / 5))
    x_step = 10 if x_upper > 50 else 5
    x_ticks = list(range(0, x_upper + 1, x_step))
    if x_ticks[-1] != x_upper:
        x_ticks.append(x_upper)
    axis.set(xlim=(0, x_upper), ylim=(0, 100),
             xlabel="Benign false-positive rate (%)",
             ylabel="Binary attack detection rate (%)")
    axis.set_xticks(x_ticks); axis.set_yticks(range(0, 101, 10))
    figure.text(.08, .885,
                f"X-axis: 0-{x_upper}% (ranges may differ by dataset); all sampled mean points included.",
                fontsize=9, color="#454A50")
    axis.grid(color="#E7E9EB", linewidth=.6)
    axis.spines[["top", "right"]].set_visible(False)
    if drawn:
        axis.legend(loc="lower right", fontsize=9, frameon=True,
                    edgecolor="#E0E3E6", framealpha=.95)
    else:
        axis.text(.5, .5, "No complete five-seed group; no curves drawn.",
                  ha="center", va="center", transform=axis.transAxes)
    figure.text(.08, .407, "Omitted incomplete groups: " + (", ".join(missing) if missing else "none"),
                fontsize=9)
    figure.text(.08, .382, "Fixed test-FPR summaries: per-seed interpolation, THEN five-seed aggregation",
                fontsize=10, fontweight="bold")
    table_axis = figure.add_axes((.08, .150, .87, .220))
    table_axis.set_axis_off()
    rows = [[STYLES[m][0], *[table_cell(groups.get((dataset, m)), t) for t in TARGETS]]
            for m in METHODS]
    table = table_axis.table(cellText=rows,
                            colLabels=["Method", "FPR 2%", "FPR 5%", "FPR 10%"],
                            colWidths=[.19, .27, .27, .27], cellLoc="center",
                            loc="center", bbox=(0, 0, 1, 1))
    table.auto_set_font_size(False); table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D9DDE1"); cell.set_linewidth(.45)
        cell.set_facecolor("#F0F2F4" if row == 0 else "white")
        if row == 0:
            cell.set_text_props(weight="bold")
        if row > 0 and col == 0:
            cell.set_text_props(color=STYLES[METHODS[row-1]][1], weight="bold")
    notes = (
        "Table cells: ADR mean +/- sample SD (top), Macro-F1 mean +/- sample SD (bottom), all in %. "
        "SD is seed spread, not a confidence interval. Unavailable means at least one seed cannot bracket the target.\n"
        "Same-tau means are NOT fixed-FPR per-seed interpolations. Curves show means without interval bands. "
        "Table entries come directly from summary operating_points; no mean-curve interpolation or extrapolation. "
        "These test-set curves are descriptive, not holdout-calibrated deployment guarantees."
    )
    wrapped = "\n".join(textwrap.fill(line, 140) for line in notes.splitlines())
    figure.text(.08, .123, wrapped, fontsize=8, va="top", linespacing=1.35)
    stamp = f"Summary snapshot: {summary['updated_at']} | SHA256 {source_sha[:16]}..."
    figure.text(.08, .018, stamp, fontsize=7.5, color="#626970")
    stem = dataset.lower().replace("-", "_") + "_mean_tau_fpr_adr"
    description = (f"Section26.10 secondary after primary failure. Summary SHA256 {source_sha}. "
                   f"Snapshot {summary['updated_at']}. Complete five-seed groups only. "
                   f"X-axis 0-{x_upper}% includes all sampled mean points; dataset ranges may differ. "
                   "Mean-tau curves differ from per-seed fixed-test-FPR interpolation. "
                   "No inference, training, extrapolation, or primary-decision change.")
    png, svg = output / (stem + ".png"), output / (stem + ".svg")
    figure.savefig(png, dpi=dpi, metadata={"Title": dataset + " Section26.10 operating curves",
                                        "Description": description, "Software": "plot_farq_operating_curves.py"})
    figure.savefig(svg, metadata={"Title": dataset + " Section26.10 operating curves",
                                "Description": description, "Creator": "plot_farq_operating_curves.py", "Date": None})
    # Strip exporter whitespace without changing SVG path geometry.
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(figure)
    return {"dataset": dataset, "drawn_complete_methods": drawn,
            "omitted_incomplete": missing, "x_axis_percent": [0, x_upper],
            "maximum_drawn_mean_fpr_percent": max_fpr_percent,
            "png": png.name, "svg": svg.name}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path,
                        default=ROOT / "ncd/results/problem_scout/section26/operating_curves_v1/summary.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace earlier figure exports")
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("dpi must be at least72")
    content = read_snapshot(args.summary)
    summary = json.loads(content.decode("utf-8-sig"))
    groups = validate(summary)
    output = args.output_dir or args.summary.parent / "figures"
    outputs = [output / (dataset.lower().replace("-", "_") + "_mean_tau_fpr_adr." + extension)
               for dataset in args.datasets for extension in ("png", "svg")]
    existing = [p.name for p in outputs if p.exists()]
    if existing and not args.overwrite:
        parser.error("Preserving existing figures; choose a new output directory or --overwrite: " + ", ".join(existing))
    output.mkdir(parents=True, exist_ok=True)
    source_sha = hashlib.sha256(content).hexdigest()
    rendered = [draw_dataset(summary, groups, dataset, source_sha, output, args.dpi) for dataset in args.datasets]
    print(json.dumps({"summary_sha256": source_sha, "summary_updated_at": summary["updated_at"],
                      "figure_exports": rendered, "inference_or_training": False}, indent=2))


if __name__ == "__main__":
    main()
