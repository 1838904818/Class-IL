"""Run one §489 dataset locally, serially, using the frozen Section 165 entry point."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
DATASETS = ("UNSW-NB15", "CIC-IDS-2017", "CSE-CIC-IDS2018")
ARMS = (("R3-A-B1-5000", "R3-A", 5000), ("R3-A-B1-10000", "R3-A", 10000),
        ("R3-A-B1-20000", "R3-A", 20000), ("R3-H", "R3-H", None))
SEEDS = tuple(range(10, 15))
NON_EXPERIMENT_FILES = {"ncd/export_section489_cell.py", "ncd/run_section489_local.py"}


def check_source(commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Use the full frozen method commit")
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", commit, "--", "ncd"], cwd=REPO, text=True,
    ).splitlines()
    if set(changed) - NON_EXPERIMENT_FILES:
        raise RuntimeError(f"Experimental source differs from the pin: {changed}")


def check_completed(cell, dataset, method, seed, cap, commit):
    if not cell.exists():
        return False
    result_path, summary_path = cell / "cell/result.json", cell / "summary.json"
    if not result_path.is_file() or not summary_path.is_file():
        raise RuntimeError(f"Incomplete output; preserve and inspect before resuming: {cell}")
    result = json.loads(result_path.read_bytes())
    summary = json.loads(summary_path.read_bytes())
    identity = result["identity"]
    config = identity["effective_config"]
    if not all(x["completed"] and x["scientific_result"] for x in (result, summary)) or (
        identity["dataset"], identity["method"], identity["seed"], identity["git_commit"]
    ) != (dataset, method, seed, commit) or (
        summary["dataset"], summary["method"], summary["seed"], summary["source_commit"]
    ) != (dataset, method, seed, commit) or config["smoke"] is not False or (
        config["normal_training_rows"] != (cap if cap is not None else 50000)
        or config["cpu_threads"] != 2
        or (config.get("section489_benign_volume") is True) != (cap is not None)
        or (config.get("section165_benign_refresh") is True) != (method == "R3-H")
    ) or len(result["checkpoints"]) != len(result["tasks"]) or (
        len(result["checkpoints"]) != summary["checkpoints"]
    ):
        raise RuntimeError(f"Existing result does not match the planned cell: {cell}")
    return True


def run(args):
    dataset = args.dataset
    slug = dataset.lower()
    data, output, logs = (p.resolve(strict=True) for p in
                          (args.data_root, args.output_root, args.logs))
    env = dict(os.environ, NCD_SOURCE_COMMIT=args.source_commit, OPENBLAS_NUM_THREADS="2",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    n_done = 0
    for arm, method, cap in ARMS:
        for seed in SEEDS:
            check_source(args.source_commit)
            cell = output / "confirmation" / slug / arm / f"seed{seed}"
            if check_completed(cell, dataset, method, seed, cap, args.source_commit):
                print(f"CELL_REUSED {dataset} {arm} {seed}", flush=True)
                n_done += 1
                continue
            stem = f"section489-{slug}-{arm.lower()}-seed{seed}-{args.run_label}"
            stdout = logs / f"{stem}.stdout.log"
            stderr = logs / f"{stem}.stderr.log"
            if stdout.exists() or stderr.exists():
                raise RuntimeError(f"Prior logs exist; preserve and inspect: {stem}")
            cmd = [sys.executable, "-I", "-B", "-u", "ncd/run_section165.py",
                   "--dataset", dataset, "--method", method, "--seed", str(seed),
                   "--data-root", str(data), "--output-root", str(output), "--threads", "2"]
            if cap is not None:
                cmd.extend(["--normal-training-rows", str(cap)])
            print(f"CELL_START {dataset} {arm} {seed}", flush=True)
            with stdout.open("xb") as out, stderr.open("xb") as err:
                process = subprocess.run(cmd, cwd=REPO, env=env, stdout=out, stderr=err)
            if process.returncode != 0:
                raise RuntimeError(f"Cell failed with exit {process.returncode}: {stem}")
            if not check_completed(cell, dataset, method, seed, cap, args.source_commit):
                raise RuntimeError(f"Cell exited without a completed result: {stem}")
            print(f"CELL_COMPLETE {dataset} {arm} {seed} exit=0", flush=True)
            n_done += 1
    print(f"DATASET_COMPLETE {dataset} cells={n_done}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--logs", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-label", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9-]+", args.run_label):
        parser.error("run-label must be lowercase alphanumeric with optional hyphens")
    run(args)


if __name__ == "__main__":
    main()
