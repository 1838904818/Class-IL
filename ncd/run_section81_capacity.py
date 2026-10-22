"""Section 81 capacity AR entry: isolated three-hidden-layer factories."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch
from torch import nn
from ncd import run_aacil as aa, run_aacil_compact as ar
from ncd.aacil_compact import CompactFamily
from ncd.prepare_problem_scout_data import prepare_setting
from ncd.run_problem_scout import binding, canonical_hash, save_json, section26_pools, smoke_subset
from ncd.source_commit import source_commit
from ncd.train_cil import Tee


class CapacityMLP(nn.Module):
    def __init__(self, inputs, hidden, outputs):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, outputs)

    def forward(self, x):
        return self.head(self.features(x))


class CapacityFamily(CompactFamily):
    def __init__(self, inputs, hidden, outputs, method, ema_decay=.999):
        nn.Module.__init__(self)
        if method != "aacil_kd_wa":
            raise ValueError("Only fixed AR family is implemented")
        self.method, self.hidden, self.ema_decay = method, hidden, ema_decay
        self.student = CapacityMLP(inputs, hidden, outputs)
        self.reference_parameters = inputs * hidden + hidden + 2 * (hidden * hidden + hidden) + hidden * outputs + outputs
        self.register_buffer("seen_mask", torch.zeros(outputs, dtype=torch.bool))
        self.initial_parameter_count = sum(p.numel() for p in self.parameters())
        assert self.initial_parameter_count == self.reference_parameters
        self.assert_capacity()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("UNSW-NB15", "CIC-IDS-2017"), required=True)
    p.add_argument("--width", type=int, choices=(512, 1024), required=True)
    p.add_argument("--seed", type=int, choices=(1, 2, 3, 4, 42), required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if not args.smoke:
        p.error("Formal capacity matrix supervisor and resource measurements are not implemented yet")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    config = json.loads((ROOT / "ncd/configs/problem_scout_section21.json").read_text())
    config.update(problem_scout_data_root=str(args.data_root.resolve(strict=True)),
        hidden=args.width, hidden_layers=3, device="cpu", cpu_threads=4,
        section27=True, section31=True, section81=True, smoke=True,
        epochs_per_task=1, min_steps_per_task=0, normal_training_rows=50000,
        lambda_g=0., derpp_alpha=1., derpp_beta=1., learning_rate=.001,
        batch_size=256, ema_decay=.999, ema_consistency=1., cbf_epochs=1,
        cbf_lr_multiplier=.1, deadline_iso=None, gate_step_policy="matched_s2000",
        calibration_policy="permanent_normal_holdout_outside_m", selection_only=False,
        selection_validation_policy="Retained old-class validation only; no selection",
        derpp_allocation_adaptation=ar.ADAPTATION,
        derpp_coefficient_source="Section81 fixed original AR; no tuning")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    with (output / "stdout.log").open("w", encoding="utf-8") as out, (output / "stderr.log").open("w", encoding="utf-8") as err, redirect_stdout(Tee(sys.stdout, out)), redirect_stderr(Tee(sys.stderr, err)):
        data, _, _, metadata = prepare_setting(args.dataset, config)
        data, _, _ = smoke_subset(data, None, None, limit=64)
        train, valid = section26_pools(data, args.seed)
        identity = {"git_commit": source_commit(ROOT), "dataset": args.dataset,
            "method": ar.NOGKD_WA, "seed": args.seed, "effective_config": config,
            "command": sys.argv, "class_order": data.class_order, "tasks": data.tasks,
            "feature_names": data.feature_names, "data_binding": binding(data, None, None),
            "data_metadata_sha256": canonical_hash(metadata),
            "section26_holdout": {str(c): {"train_count": len(train[c]),
                "validation_count": len(valid[c]), "validation_indices_sha256": hashlib.sha256(valid[c].tobytes()).hexdigest()} for c in valid}}
        namespace = dict(ar.run_cell.__globals__)
        namespace.update(MLP=CapacityMLP, CompactFamily=CapacityFamily)
        runner = types.FunctionType(ar.run_cell.__code__, namespace, "capacity_ar", ar.run_cell.__defaults__)
        result = runner(data, config, identity, output / "cell", torch.device("cpu"))
        assert result["completed"] and len(result["checkpoints"]) == len(data.tasks)
        save_json(output / "summary.json", {"completed": True, "scientific_result": False,
            "smoke": True, "dataset": args.dataset, "width": args.width, "hidden_layers": 3,
            "seed": args.seed, "epochs_per_task": 1, "max_rows_per_class": 64,
            "cpu_threads": 4, "checkpoint_count": len(result["checkpoints"]),
            "elapsed_seconds": time.perf_counter() - start,
            "family_parameters": result["architecture_capacity"],
            "source_commit": identity["git_commit"],
            "entry_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limitations": "Smoke only. Not full-data runtime or resource sizing; formal matrix not enabled."})


if __name__ == "__main__":
    main()
