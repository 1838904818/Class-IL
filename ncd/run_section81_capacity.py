"""Section 81 capacity cell: isolated three-hidden-layer factories for four methods."""
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
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch
from torch import nn
from ncd import run_aacil as aa, run_aacil_compact as ar
from ncd import run_problem_scout as ps, run_replay_bs as bs
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
    p.add_argument("--method", choices=("ar", "replay_bs", "freq", "wacb"), default="ar")
    p.add_argument("--profile", action="store_true", help="Full-data, unchanged-training CPU timing cell")
    p.add_argument("--formal", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if sum((args.smoke, args.profile, args.formal)) != 1:
        p.error("Select exactly one execution mode")
    if args.formal and (not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") not in ("cpu-epyc-genoa", "cpu-epyc")):
        p.error("Formal capacity cells require the scheduled CPU matrix")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    config = json.loads((ROOT / "ncd/configs/problem_scout_section21.json").read_text())
    config.update(problem_scout_data_root=str(args.data_root.resolve(strict=True)),
        hidden=args.width, hidden_layers=3, device="cpu", cpu_threads=4,
        section27=True, section31=True, section81=True, smoke=args.smoke,
        epochs_per_task=1 if args.smoke else 8, min_steps_per_task=0 if args.smoke else 2000, normal_training_rows=50000,
        lambda_g=0., derpp_alpha=1., derpp_beta=1., learning_rate=.001,
        batch_size=256, ema_decay=.999, ema_consistency=1., cbf_epochs=1,
        cbf_lr_multiplier=.1, deadline_iso=None, gate_step_policy="matched_s2000",
        calibration_policy="permanent_normal_holdout_outside_m", selection_only=False,
        selection_validation_policy="Retained old-class validation only; no selection",
        derpp_allocation_adaptation=ar.ADAPTATION,
        derpp_coefficient_source="Section81 fixed original AR; no tuning")
    if args.method == "replay_bs":
        config.update(section28=True, section28_method="replay_bs", post_hoc=True,
            calibration_protocol=bs.CALIBRATION_PROTOCOL,
            deadline="9999-12-31T23:59:59+00:00")
    if args.method in ("freq", "wacb"):
        config.update(section26=True, section26_schema_version=2,
            section26_access_policy=ps.SECTION26_POLICY)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    start = time.perf_counter()
    with (output / "stdout.log").open("a", encoding="utf-8") as out, (output / "stderr.log").open("a", encoding="utf-8") as err, redirect_stdout(Tee(sys.stdout, out)), redirect_stderr(Tee(sys.stderr, err)):
        data, _, _, metadata = prepare_setting(args.dataset, config)
        counts = np.bincount(data.y_train, minlength=len(data.class_order))
        if args.smoke:
            data, _, _ = smoke_subset(data, None, None, limit=64)
        train, valid = section26_pools(data, args.seed)
        identity = {"git_commit": source_commit(ROOT), "dataset": args.dataset,
            "method": ar.NOGKD_WA if args.method == "ar" else args.method, "seed": args.seed, "effective_config": config,
            "command": [arg for arg in sys.argv if arg != "--resume"], "class_order": data.class_order, "tasks": data.tasks,
            "feature_names": data.feature_names, "data_binding": binding(data, None, None),
            "data_metadata_sha256": canonical_hash(metadata),
            "section26_holdout": {str(c): {"train_count": len(train[c]),
                "validation_count": len(valid[c]), "validation_indices_sha256": hashlib.sha256(valid[c].tobytes()).hexdigest()} for c in valid}}
        identity.update(original_train_counts=counts.tolist(), regularization=0.,
            section26_access_policy=ps.SECTION26_POLICY, section27_access_policy=aa.POLICY,
            section28_access_policy=aa.POLICY, offline_joint_reference=False,
            lambda_file=None, lambda_file_sha256=None, lambda_source_commit=None)
        function = ar.run_cell if args.method == "ar" else bs.run_cell if args.method == "replay_bs" else ps.run_cell
        namespace = dict(function.__globals__)
        namespace.update(MLP=CapacityMLP, CompactFamily=CapacityFamily)
        runner = types.FunctionType(function.__code__, namespace, "capacity_cell", function.__defaults__)
        cell, device = output / "cell", torch.device("cpu")
        cell.mkdir(exist_ok=args.resume)
        if args.method == "ar":
            result = runner(data, config, identity, cell, device)
        elif args.method == "replay_bs":
            result = runner(data, counts, config, identity, cell, device)
        else:
            result = runner(data, None, None, metadata, config, identity, cell, device)
            calibrated = []
            for cp in result["checkpoints"]:
                state = ps.checkpoint_load(cell / cp["checkpoint_file"], identity)
                model = CapacityMLP(data.x_train.shape[1], args.width, len(data.class_order))
                model.load_state_dict(state["model_state"], strict=True)
                calibration, points, calibration_seconds, evaluation_seconds = bs.evaluate(
                    model, data, cp["seen_classes"], valid[data.class_order.index("Normal")], counts, config, device)
                calibrated.append(dict(task_index=cp["task_index"], calibration=calibration,
                    operating_points=points, calibration_seconds=calibration_seconds,
                    evaluation_seconds=evaluation_seconds, checkpoint_sha256=cp["checkpoint_sha256"]))
            save_json(output / "calibrated_stages.json", calibrated)
        assert result["completed"] and len(result["checkpoints"]) == len(data.tasks)
        from ncd.analyze_stagewise_forgetting import stage_metrics, aggregate_trajectory
        stages = calibrated if args.method in ("freq", "wacb") else result["checkpoints"]
        metrics = {}
        for alpha in ("0.01", "0.02"):
            trajectory = [dict(task_index=cp["task_index"], metrics=stage_metrics(
                cp["operating_points"][alpha]["confusion_matrix"], data.tasks[0] if cp["task_index"] == 0 else
                [c for task in data.tasks[:cp["task_index"]+1] for c in task], data.class_order)) for cp in stages]
            metrics[alpha] = aggregate_trajectory(trajectory, data.tasks, data.class_order)
        save_json(output / "summary.json", {"completed": True, "scientific_result": args.formal,
            "smoke": args.smoke, "profile": args.profile, "method": args.method,
            "dataset": args.dataset, "width": args.width, "hidden_layers": 3,
            "seed": args.seed, "epochs_per_task": config["epochs_per_task"], "max_rows_per_class": 64 if args.smoke else None,
            "cpu_threads": 4, "checkpoint_count": len(result["checkpoints"]),
            "elapsed_seconds": time.perf_counter() - start,
            "family_parameters": result.get("architecture_capacity"),
            "single_network_parameters": sum(p.numel() for p in CapacityMLP(data.x_train.shape[1], args.width, len(data.class_order)).parameters()),
            "metrics": metrics, "job_id": os.environ.get("SLURM_JOB_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"), "node": os.environ.get("SLURMD_NODENAME"),
            "source_commit": identity["git_commit"],
            "entry_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limitations": "Smoke/profile are not formal five-seed conclusions. Capacity sensitivity is descriptive RQ2 evidence."})


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Supervisor captures the complete console traceback as well as cell logs.
        traceback.print_exc()
        raise
