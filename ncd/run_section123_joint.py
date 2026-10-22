"""Independent AR-Joint CPU entry; formal schedule remains selection-gated."""
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
        p.error("Formal execution remains disabled until TB schedule selection")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    config = json.loads((ROOT / "ncd/configs/problem_scout_section21.json").read_text())
    config.update(problem_scout_data_root=str(a.data_root.resolve(strict=True)),
        hidden=128, device="cpu", cpu_threads=4, smoke=True, schedule="fixed",
        eval_batch_size=4096, deadline_iso=None, memory_setting="fixed200")
    output = a.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    data, _, _, _ = prepare_setting(a.dataset, config)
    data, _, _ = smoke_subset(data, None, None, limit=64)
    sources = ("ncd/run_section123_joint.py", "ncd/section123_joint.py", "ncd/section123_joint_cell.py")
    identity = dict(git_commit=source_commit(ROOT), dataset=a.dataset, method="ar_joint",
        seed=a.seed, effective_config=config, command=sys.argv,
        source_hashes={s: digest(ROOT / s) for s in sources}, data_binding=binding(data, None, None))
    result = run_joint_fixed(data, config, identity, output / "cell")
    save_json(output / "summary.json", dict(completed=True, scientific_result=False,
        smoke=True, dataset=a.dataset, method="ar_joint", seed=a.seed,
        checkpoint_count=len(result["checkpoints"]), parameters=result["parameters"],
        elapsed_seconds=time.perf_counter()-started, source_commit=identity["git_commit"]))


if __name__ == "__main__":
    main()
