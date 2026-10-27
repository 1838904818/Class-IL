"""Fresh-seed fixed 2x128 confirmation cells with saved-checkpoint default views."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import torch
from ncd import run_aacil as aa, run_aacil_compact as ar
from ncd import run_problem_scout as ps, run_replay_bs as bs
from ncd.aacil_compact import CompactFamily
from ncd.model import MLP
from ncd.default_decision import evaluate_models
from ncd.prepare_problem_scout_data import prepare_setting
from ncd.source_commit import source_commit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("UNSW-NB15", "CIC-IDS-2017", "CSE-CIC-IDS2018"), required=True)
    p.add_argument("--method", choices=("ar", "replay_bs", "freq", "wacb"), required=True)
    p.add_argument("--seed", type=int, choices=range(10, 15), required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--threads", type=int, choices=(1, 2, 4), default=4)
    a = p.parse_args()
    selection = json.loads((ROOT / "paper_aacil/results/section123_schedule_selection.json").read_text())
    if selection["selected_schedule"] != "original_fixed":
        p.error("Registered original_fixed schedule required")
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    config = json.loads((ROOT / "ncd/configs/problem_scout_section21.json").read_text())
    config.update(problem_scout_data_root=str(a.data_root.resolve(strict=True)),
        hidden=128, hidden_layers=2, device="cpu", cpu_threads=a.threads,
        section27=True, section31=True, section81=True, smoke=a.smoke,
        epochs_per_task=1 if a.smoke else 8, min_steps_per_task=0 if a.smoke else 2000,
        normal_training_rows=50000, lambda_g=0., derpp_alpha=1., derpp_beta=1.,
        learning_rate=.001, batch_size=256, ema_decay=.999, ema_consistency=1.,
        cbf_epochs=1, cbf_lr_multiplier=.1, deadline_iso=None,
        gate_step_policy="matched_s2000", calibration_policy="permanent_normal_holdout_outside_m",
        selection_only=False, selection_validation_policy="Retained old-class validation only; no selection",
        derpp_allocation_adaptation=ar.ADAPTATION,
        derpp_coefficient_source="Registered fixed original AR; no tuning")
    if a.method == "replay_bs":
        config.update(section28=True, section28_method="replay_bs", post_hoc=True,
            calibration_protocol=bs.CALIBRATION_PROTOCOL, deadline="9999-12-31T23:59:59+00:00")
    if a.method in ("freq", "wacb"):
        config.update(section26=True, section26_schema_version=2, section26_access_policy=ps.SECTION26_POLICY)
    output = a.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=a.resume)
    started = time.perf_counter()
    data, _, _, metadata = prepare_setting(a.dataset, config)
    counts = np.bincount(data.y_train, minlength=len(data.class_order))
    if a.smoke:
        data, _, _ = ps.smoke_subset(data, None, None, limit=64)
    train, valid = ps.section26_pools(data, a.seed)
    identity = dict(git_commit=source_commit(ROOT), dataset=a.dataset,
        method=ar.NOGKD_WA if a.method == "ar" else a.method, seed=a.seed,
        effective_config=config, command=[v for v in sys.argv if v != "--resume"], class_order=data.class_order,
        tasks=data.tasks, feature_names=data.feature_names, data_binding=ps.binding(data, None, None),
        data_metadata_sha256=ps.canonical_hash(metadata), original_train_counts=counts.tolist(),
        regularization=0., section26_access_policy=ps.SECTION26_POLICY,
        section27_access_policy=aa.POLICY, section28_access_policy=aa.POLICY,
        offline_joint_reference=False, lambda_file=None, lambda_file_sha256=None, lambda_source_commit=None,
        section26_holdout={str(c): dict(train_count=len(train[c]), validation_count=len(valid[c]),
            validation_indices_sha256=hashlib.sha256(valid[c].tobytes()).hexdigest()) for c in valid})
    cell = output / "cell"
    cell.mkdir(exist_ok=a.resume)
    device = torch.device("cpu")
    if a.method == "ar":
        result = ar.run_cell(data, config, identity, cell, device)
    elif a.method == "replay_bs":
        result = bs.run_cell(data, counts, config, identity, cell, device)
    else:
        result = ps.run_cell(data, None, None, metadata, config, identity, cell, device)
    defaults, calibrated = [], []
    for cp in result["checkpoints"]:
        state = ps.checkpoint_load(cell / cp["checkpoint_file"], identity)
        family, attacks = None, None
        model = MLP(data.x_train.shape[1], 128, 2 if a.method == "ar" else len(data.class_order))
        if a.method == "ar":
            attacks = state["attack_class_ids"]
            model.load_state_dict(state["gate_model_state"], strict=True)
            family = CompactFamily(data.x_train.shape[1], 128, len(attacks), "aacil_kd_wa", .999)
            family.load_state_dict(state["family_model_state"], strict=True)
            family.assert_capacity()
        else:
            model.load_state_dict(state["model_state"], strict=True)
        if a.method in ("freq", "wacb"):
            calibration, points, cal_seconds, eval_seconds = bs.evaluate(
                model, data, cp["seen_classes"], valid[data.class_order.index("Normal")], counts, config, device)
            calibrated.append(dict(task_index=cp["task_index"], calibration=calibration,
                operating_points=points, calibration_seconds=cal_seconds,
                evaluation_seconds=eval_seconds, checkpoint_sha256=cp["checkpoint_sha256"]))
        defaults.append(dict(task_index=cp["task_index"], checkpoint_sha256=cp["checkpoint_sha256"],
            metrics=evaluate_models(data.x_test, data.y_test, cp["seen_classes"], data.class_order,
                model, family=family, attack_ids=attacks)))
    ps.save_json(output / "default_stages.json", defaults)
    if calibrated:
        ps.save_json(output / "calibrated_stages.json", calibrated)
    assert result["completed"] and len(defaults) == len(data.tasks)
    ps.save_json(output / "summary.json", dict(completed=True, scientific_result=not a.smoke,
        smoke=a.smoke, dataset=a.dataset, method=a.method, seed=a.seed,
        checkpoint_count=len(defaults), elapsed_seconds=time.perf_counter()-started,
        source_commit=identity["git_commit"]))


if __name__ == "__main__":
    main()
