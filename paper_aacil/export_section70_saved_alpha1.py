"""Extract already published 1% operating points; no inference or final tests.

Freq and WACB require separate evaluation and are intentionally not substituted
with their 2% operating points. This is a partial evidence delivery, not H-low.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_results import ROOT, private_fields, read, write_derived
from export_section46_development import require

METHODS = ("aacil_nogkd_wa", "replay_bs", "aacil")
DECISION = (*METHODS, "freq", "wacb")
BASE = ROOT / "ncd/results/problem_scout/section64/interim_local_v1"
DEST = ROOT / "ncd/results/problem_scout/section70/saved_alpha1_v1"


def main():
    require(not DEST.exists(), "Saved-point delivery already exists; do not repeat")
    releases = {}
    # Verify all 25 completed decision cells before opening checkpoint metrics.
    for seed in range(5, 10):
        release, digest = read(BASE / f"seed_{seed}/result.json")
        require(release["seed"] == seed and release["physical_cells"] == 5,
                "Unexpected release identity")
        require([r["method"] for r in release["rows"]] == list(DECISION),
                "Decision set mismatch")
        for row in release["rows"]:
            cell = row["cell_attempt"]
            require(row["seed"] == seed and cell["seed"] == seed
                    and cell["method"] == row["method"]
                    and cell["exit_code"] == 0 and cell["ended_at"],
                    "Incomplete decision cell")
        releases[seed] = release, digest
    outputs = []
    for seed, (release, release_sha) in releases.items():
        for method in METHODS:
            folder = BASE / f"seed_{seed}" / method / "raw"
            raw, raw_sha = read(folder / "result.json")
            ident = raw["identity"]
            require(raw["completed"] is True and ident["seed"] == seed
                    and ident["method"] == method
                    and ident["dataset"] == "CSE-CIC-IDS2018", "Raw identity mismatch")
            paths = sorted((folder / "checkpoints").glob("checkpoint_*.json"))
            require(len(paths) == len(raw["checkpoints"]) == len(raw["tasks"]) == 4,
                    "Expected four complete stages")
            stages = []
            for task, path in enumerate(paths):
                cp, cp_sha = read(path)
                original = dict(cp)
                original.pop("publication_derivation")
                require(original == raw["checkpoints"][task], "Sidecar/raw mismatch")
                require(cp["task_index"] == task, "Stage order mismatch")
                calibration = (cp["normal_holdout_calibration"]["operating_points"]["0.01"]
                               if method == "replay_bs" else cp["calibration"]["0.01"])
                require(calibration["target_fpr"] == 0.01, "Wrong calibration target")
                stages.append({
                    "task_index": task, "seen_classes": cp["seen_classes"],
                    "checkpoint_sha256": cp["checkpoint_sha256"],
                    "source_sidecar": path.relative_to(ROOT).as_posix(),
                    "source_sidecar_sha256": cp_sha,
                    "calibration": calibration,
                    "calibration_context": cp.get("normal_holdout_calibration", {
                        "policy": cp.get("calibration_policy"),
                        "pool_storage": cp.get("calibration_pool_storage"),
                    }),
                    "saved_metrics": cp["operating_points"]["0.01"],
                })
            row = next(r for r in release["rows"] if r["method"] == method)
            value = {
                "dataset": ident["dataset"], "method": method, "seed": seed,
                "alpha": 0.01, "stages": stages, "tasks": raw["tasks"],
                "source_result": (folder / "result.json").relative_to(ROOT).as_posix(),
                "source_result_sha256": raw_sha, "source_identity": ident,
                "source_release_sha256": release_sha,
                "cell_attempt": row["cell_attempt"],
                "original_costs": {k: v for k, v in raw.items()
                                   if k.endswith("_seconds_total") or k == "cost_scope"},
                "metrics_recomputed": False, "training_or_inference_executed": False,
                "final_tests_executed": False, "h_low_complete": False,
                "missing_alpha1_methods": ["freq", "wacb"],
                "original_40_still_required": True, "selection_changed": False,
                "historical_failures_unchanged": True,
                "independent_dataset_confirmation": False,
            }
            require(not list(private_fields(value)), "Private metadata detected")
            outputs.append((DEST / f"{method}_seed{seed}/result.json", value))
    for path, value in outputs:
        write_derived(path, value)
    print("Extracted 15 cells / 60 saved 1% stages; Freq/WACB and final tests remain pending.")


if __name__ == "__main__":
    main()
