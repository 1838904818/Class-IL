"""Publish a complete external-baseline dataset without rerunning evaluation."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_results import read, private_fields, write_derived

METHODS = ("replay", "derpp", "wacb", "freq", "joint")
SEEDS = (1, 2, 3, 4, 42)
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("UNSW-NB15", "NSL-KDD", "CIC-IDS-2017"), required=True)
    args = parser.parse_args()
    slug = args.dataset.lower().replace("-", "_")
    source = args.source_root.resolve() / slug
    attempts = [read(p)[0] for p in sorted(source.glob("attempt_*.json"))]
    if not attempts or any(a["status"] not in ("failed", "complete") for a in attempts):
        raise ValueError("All evaluation attempts must be terminal")
    if not any(a["status"] == "complete" and len(a["completed_cells"]) == 25 for a in attempts):
        raise ValueError("No complete 25-cell evaluation assignment")
    output = ROOT / "ncd/results/problem_scout/section46/external_low_fpr_v1" / slug
    prepared = []
    for method in METHODS:
        for seed in SEEDS:
            folder = source / f"{method}_seed{seed}"
            result, digest = read(folder / "result.json")
            if not (result["completed"] and (result["dataset"], result["method"], result["seed"]) ==
                    (args.dataset, method, seed) and set(result["operating_points"]) == {"0.005", "0.01"}):
                raise ValueError("Incomplete or mismatched evaluated cell")
            for t in range(len(result["tasks"])):
                stage, _ = read(folder / f"task_{t:03d}.json")
                if not stage["completed"] or not stage["default_cm_reproduced"]:
                    raise ValueError("Stage reproduction incomplete")
                for alpha in ("0.005", "0.01"):
                    if stage["operating_points"][alpha]["metrics"] != result["operating_points"][alpha]["stages"][t]["metrics"]:
                        raise ValueError("Stage/result trajectory mismatch")
                prepared.append((output / folder.name / f"task_{t:03d}.json", stage))
            result["publication"] = {
                "original_result_sha256": digest,
                "evaluation_attempts": attempts,
                "attempt_cost_note": "Attempt elapsed times include reused-output validation and are not additive with nested stage costs. Failed attempts are preserved.",
                "training_repeated": False,
            }
            prepared.append((output / folder.name / "result.json", result))
    for path, value in prepared:
        if list(private_fields(value)):
            raise ValueError("Private content in evaluation publication")
        if path.exists() and read(path)[0] != value:
            raise ValueError("Different existing publication")
    for path, value in prepared:
        write_derived(path, value)
    print(f"Published {args.dataset}: 25 cells, {len(prepared)} JSON files; no inference or statistics rerun")


if __name__ == "__main__":
    main()
