"""Independent CPU entrypoint for Section123 training-budget controls."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import torch
from ncd.prepare_problem_scout_data import prepare_setting
from ncd.run_problem_scout import binding, digest, save_json, smoke_subset
from ncd.section123_tb_cell import run_tb
from ncd.source_commit import source_commit
from ncd.train_cil import Tee


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("UNSW-NB15", "CIC-IDS-2017"), required=True)
    parser.add_argument("--method", choices=("ar_es", "replay_bs_es"), required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2, 3, 4, 42), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    config = json.loads((ROOT / "ncd/configs/problem_scout_section21.json").read_text())
    config.update(problem_scout_data_root=str(args.data_root.resolve(strict=True)),
                  hidden=128, device="cpu", cpu_threads=4, smoke=args.smoke,
                  eval_batch_size=4096, deadline_iso=None, lambda_g=0.,
                  derpp_alpha=1., derpp_beta=1., memory_setting="fixed200",
                  section123=True, schedule="ES", batch_size=256)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    with (output / "stdout.log").open("w", encoding="utf-8") as out, (output / "stderr.log").open("w", encoding="utf-8") as err:
        with redirect_stdout(Tee(sys.stdout, out)), redirect_stderr(Tee(sys.stderr, err)):
            try:
                data, _, _, _ = prepare_setting(args.dataset, config)
                counts = np.bincount(data.y_train, minlength=len(data.class_order))
                if args.smoke:
                    data, _, _ = smoke_subset(data, None, None, limit=64)
                sources = ("ncd/run_section123_tb.py", "ncd/section123_tb_cell.py", "ncd/section123_es_training.py")
                identity = dict(git_commit=source_commit(ROOT), dataset=args.dataset,
                    method=args.method, seed=args.seed, effective_config=config,
                    command=sys.argv, log_path=str(output / "stdout.log"),
                    source_hashes={p: digest(ROOT / p) for p in sources}, data_binding=binding(data, None, None))
                result = run_tb(data, counts, config, identity, output / "cell")
                save_json(output / "summary.json", dict(completed=True, smoke=args.smoke,
                    scientific_result=not args.smoke, method=args.method, dataset=args.dataset,
                    seed=args.seed, checkpoint_count=len(result["checkpoints"]),
                    elapsed_seconds=time.perf_counter() - started, source_commit=identity["git_commit"]))
            except BaseException:
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
