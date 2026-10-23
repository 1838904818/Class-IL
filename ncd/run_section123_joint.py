"""AR-Joint CPU entry using the registered fixed confirmation schedule."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import torch
from ncd.prepare_problem_scout_data import prepare_setting
from ncd.run_problem_scout import binding, digest, save_json, smoke_subset
from ncd.section123_joint_cell import run_joint_fixed
from ncd.source_commit import source_commit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("UNSW-NB15", "CIC-IDS-2017"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if not a.smoke:
        selection = json.loads((ROOT / "paper_aacil/results/section123_schedule_selection.json").read_text())
        if selection["selected_schedule"] != "original_fixed" or a.seed not in (10, 11, 12, 13, 14):
            p.error("Registered fixed schedule and fresh confirmation seed required")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    config = json.loads((ROOT / "ncd/configs/problem_scout_section21.json").read_text())
    config.update(problem_scout_data_root=str(a.data_root.resolve(strict=True)),
        hidden=128, device="cpu", cpu_threads=4, smoke=a.smoke, schedule="fixed",
        eval_batch_size=4096, deadline_iso=None, memory_setting="fixed200")
    output = a.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    data, _, _, _ = prepare_setting(a.dataset, config)
    if a.smoke:
        data, _, _ = smoke_subset(data, None, None, limit=64)
    sources = ("ncd/run_section123_joint.py", "ncd/section123_joint.py", "ncd/section123_joint_cell.py")
    identity = dict(git_commit=source_commit(ROOT), dataset=a.dataset, method="ar_joint",
        seed=a.seed, effective_config=config, command=sys.argv,
        source_hashes={s: digest(ROOT / s) for s in sources}, data_binding=binding(data, None, None))
    result = run_joint_fixed(data, config, identity, output / "cell")
    save_json(output / "summary.json", dict(completed=True, scientific_result=not a.smoke,
        smoke=a.smoke, dataset=a.dataset, method="ar_joint", seed=a.seed,
        checkpoint_count=len(result["checkpoints"]), parameters=result["parameters"],
        elapsed_seconds=time.perf_counter()-started, source_commit=identity["git_commit"]))


if __name__ == "__main__":
    main()
