"""Section156 local screening/smoke entrypoint; unsupported switches fail closed."""
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


def main():
    from ncd.section156_config import configurations
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configuration", required=True, choices=[c.identifier for c in configurations()])
    p.add_argument("--dataset", required=True, choices=("UNSW-NB15", "CIC-IDS-2017"))
    p.add_argument("--seed", type=int, choices=(1, 2), required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--threads", type=int, default=4, choices=(1, 2, 4))
    a = p.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    os.environ["MKL_NUM_THREADS"] = str(a.threads)
    import torch
    from ncd.prepare_problem_scout_data import prepare_setting
    from ncd.run_problem_scout import binding, save_json, smoke_subset
    from ncd.section138_cell import run_cell
    from ncd.section156_models import build_models, train_gate
    from ncd.source_commit import source_commit
    cfg = next(c for c in configurations() if c.identifier == a.configuration)
    if cfg.special in ("T3", "T4", "T5") or cfg.family == 3:
        p.error("f3 and T3-T5 are not integrated yet")
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    config = json.loads((ROOT / "ncd/configs/problem_scout_section21.json").read_text())
    config.update(problem_scout_data_root=str(a.data_root.resolve(strict=True)),
        device="cpu", cpu_threads=a.threads, smoke=a.smoke, lambda_g=0.,
        epochs_per_task=1 if a.smoke else 8, min_steps_per_task=0 if a.smoke else 2000,
        batch_size=256, eval_batch_size=4096, learning_rate=.001,
        derpp_alpha=1., derpp_beta=1., deadline_iso=None,
        normal_training_rows=cfg.switches()["normal_training_rows"],
        ema_decay=.999, ema_consistency=1., hidden=128, section156=cfg.switches())
    output = a.output_root.resolve() / ("smoke" if a.smoke else "screen") / a.dataset.lower() / cfg.identifier / f"seed{a.seed}"
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    data, _, _, _ = prepare_setting(a.dataset, config)
    if a.smoke:
        data, _, _ = smoke_subset(data, None, None, limit=64)
    identity = dict(git_commit=source_commit(ROOT), dataset=a.dataset, seed=a.seed,
        method=cfg.identifier, effective_config=config, data_binding=binding(data, None, None),
        command=sys.argv, output_role="smoke" if a.smoke else "screen", smoke=a.smoke)
    def trainer(model, x, y, *, schedule, **kwargs):
        kind = "histgb" if cfg.switches()["control"] else cfg.switches()["gate"]
        return train_gate(model, x, y, kind=kind, **kwargs)
    result = run_cell(data, config, identity, output / "cell", cfg,
                      model_builder=build_models, gate_trainer=trainer)
    save_json(output / "summary.json", dict(completed=True, scientific_result=not a.smoke,
        source_commit=identity["git_commit"], dataset=a.dataset, seed=a.seed,
        method=cfg.identifier, checkpoints=len(result["checkpoints"]),
        elapsed_seconds=time.perf_counter()-started))


if __name__ == "__main__":
    main()
