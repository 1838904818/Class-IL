from __future__ import annotations

"""Build the reviewed seed-42 Malaya validation comparison.

Chart contract
--------------
Question: How do model size and exact train/test duplicate exclusion affect the
final joint-uncapped multiclass Class-IL metrics?
Takeaway: The 128x2 encoder outperforms 256x4 under the fixed training budget;
duplicate exclusion mainly lowers overall accuracy and exposes weak minority
class performance.
Family: grouped vertical bars in small multiples, all absolute axes start at 0.
Evidence: two models x two test views, seed 42, final checkpoint, joint_uncapped.
Non-colour encoding: duplicate-excluded bars use hatching in addition to colour.
Outputs: one exact-value CSV and one static PNG, both derived only from validated
result JSON files.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
WORK_ROOT = PROJECT_ROOT.parent
RESULTS_ROOT = WORK_ROOT / "ofra_formal_v3_results_20260716"
OUTPUT_ROOT = RESULTS_ROOT / "figures"

RUNS = {
    "MLP 128x2": RESULTS_ROOT / "malaya_mlp_128x2" / "result_seed_42.json",
    "MLP 256x4": RESULTS_ROOT / "malaya_mlp_256x4" / "result_seed_42.json",
}
VIEWS = {
    "Official holdout (n=10,370)": "official",
    "Duplicate-excluded (n=6,590)": "duplicate_excluded",
}
METRICS = (
    ("final_overall_accuracy", "Overall accuracy", "higher is better"),
    ("final_macro_f1", "Macro-F1", "higher is better"),
    ("final_balanced_accuracy", "Balanced accuracy", "higher is better"),
    ("average_task_accuracy", "Average task accuracy", "higher is better"),
    ("average_forgetting", "Average forgetting", "lower is better"),
)
ARM = "joint_uncapped"


def _load_results() -> tuple[list[dict[str, object]], dict[str, dict]]:
    records: list[dict[str, object]] = []
    documents: dict[str, dict] = {}
    for model, path in RUNS.items():
        document = json.loads(path.read_text(encoding="utf-8"))
        documents[model] = document
        if (
            document.get("problem_type") != "application_classification"
            or document.get("metric_profile") != "generic_multiclass"
            or document.get("normal_class_id") is not None
        ):
            raise ValueError(f"invalid generic metric semantics in {path}")
        for view_label, view_name in VIEWS.items():
            summary = document["summary"]["views"][view_name][ARM]
            forbidden = {
                key
                for key in summary
                if "benign" in key or "attack_detection" in key
            }
            if forbidden:
                raise ValueError(f"binary NIDS metrics found in {path}: {forbidden}")
            record: dict[str, object] = {
                "model": model,
                "view": view_label,
                "view_key": view_name,
                "arm": ARM,
                "seed": int(document["seed"]),
                "protocol_sha256": document["protocol_sha256"],
                "deterministic_result_sha256": document[
                    "deterministic_result_sha256"
                ],
            }
            for metric, _, _ in METRICS:
                record[metric] = float(summary[metric])
            records.append(record)
    return records, documents


def _write_csv(records: list[dict[str, object]], path: Path) -> None:
    fields = list(records[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _plot(records: list[dict[str, object]], path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "#FCFCFD",
            "axes.edgecolor": "#374151",
            "text.color": "#1F2937",
            "axes.labelcolor": "#374151",
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=False)
    figure.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.13, wspace=0.25, hspace=0.48)

    model_order = list(RUNS)
    view_order = list(VIEWS)
    palette = {
        view_order[0]: "#315A7D",
        view_order[1]: "#C6922E",
    }
    hatches = {view_order[0]: "", view_order[1]: "///"}
    width = 0.34

    for axis, (metric, title, direction) in zip(axes.flat, METRICS):
        values = []
        for model_index, model in enumerate(model_order):
            for view_index, view in enumerate(view_order):
                record = next(
                    item
                    for item in records
                    if item["model"] == model and item["view"] == view
                )
                value = float(record[metric])
                values.append(value)
                x = model_index + (view_index - 0.5) * width
                bar = axis.bar(
                    x,
                    value,
                    width=width,
                    color=palette[view],
                    edgecolor="#263442",
                    linewidth=0.8,
                    hatch=hatches[view],
                    zorder=3,
                )[0]
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="semibold",
                )
        upper = max(0.1, max(values) * 1.24)
        axis.set_ylim(0, min(1.0, upper))
        axis.set_xticks(range(len(model_order)), model_order)
        axis.set_title(title, loc="left", fontweight="semibold", pad=25)
        axis.text(
            0.0,
            1.01,
            direction,
            transform=axis.transAxes,
            fontsize=8.5,
            color="#6B7280",
        )
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.8, zorder=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    note_axis = axes.flat[-1]
    note_axis.axis("off")
    note_axis.text(
        0,
        0.98,
        "Reading notes",
        va="top",
        fontsize=12,
        fontweight="semibold",
    )
    note_axis.text(
        0,
        0.82,
        "• Application/service labels; not benign vs attack.\n"
        "• One complete capture held out per class.\n"
        "• Duplicate-excluded removes 3,780 held-out rows\n"
        "  whose full 77-feature vector occurs in training.\n"
        "• Seed 42 is independently re-executed byte-for-byte.\n"
        "• Five-seed uncertainty is not shown in this figure.",
        va="top",
        linespacing=1.55,
        fontsize=10.5,
    )

    legend_handles = [
        Patch(
            facecolor=palette[view],
            edgecolor="#263442",
            hatch=hatches[view],
            label=view,
        )
        for view in view_order
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.065, 0.875),
        frameon=False,
        ncol=2,
    )
    figure.suptitle(
        "MalayaNetwork_GT seed-42 validation",
        x=0.065,
        y=0.965,
        ha="left",
        fontsize=20,
        fontweight="bold",
    )
    figure.text(
        0.065,
        0.915,
        "Final joint-uncapped metrics · 10 application classes · 77 flow features · fixed capture-level split",
        ha="left",
        fontsize=11,
        color="#4B5563",
    )
    figure.text(
        0.065,
        0.045,
        "Source: validated formal-v3 result JSON; Hugging Face revision 384a59278f98490ee6e93aae017e748078d29b6a.",
        ha="left",
        fontsize=9,
        color="#6B7280",
    )
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    records, _ = _load_results()
    csv_path = OUTPUT_ROOT / "malaya_seed42_final_metrics.csv"
    image_path = OUTPUT_ROOT / "malaya_seed42_joint_uncapped_comparison.png"
    _write_csv(records, csv_path)
    _plot(records, image_path)
    print(json.dumps({"csv": str(csv_path), "figure": str(image_path)}, indent=2))


if __name__ == "__main__":
    main()
