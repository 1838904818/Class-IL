"""Section165 confirmation: literature memory or Section174 exact-C1 memory."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True,
                   choices=('UNSW-NB15', 'CIC-IDS-2017', 'CSE-CIC-IDS2018'))
    p.add_argument('--method', choices=('R3-L', 'R3-A'), default='R3-L')
    p.add_argument('--seed', required=True, type=int, choices=(1, 10, 11, 12, 13, 14))
    p.add_argument('--data-root', required=True, type=Path)
    p.add_argument('--output-root', required=True, type=Path)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--threads', type=int, choices=(1, 2), default=2)
    a = p.parse_args()
    if (a.seed == 1) != a.smoke:
        p.error('Smoke uses seed 1; confirmation uses seeds 10-14')
    os.environ['OMP_NUM_THREADS'] = str(a.threads)
    os.environ['MKL_NUM_THREADS'] = str(a.threads)
    import torch
    from ncd.prepare_problem_scout_data import prepare_setting
    from ncd.run_problem_scout import binding, save_json, smoke_subset
    from ncd.section138_cell import run_cell
    from ncd.section156_config import configurations
    from ncd.section156_models import build_models, train_gate
    from ncd.source_commit import source_commit
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    cfg = next(c for c in configurations() if c.identifier == 'C1')
    config = json.loads((ROOT / 'ncd/configs/problem_scout_section21.json').read_text())
    config.update(problem_scout_data_root=str(a.data_root.resolve(strict=True)),
        device='cpu', cpu_threads=a.threads, smoke=a.smoke, lambda_g=0.,
        epochs_per_task=1 if a.smoke else 8, min_steps_per_task=0 if a.smoke else 2000,
        batch_size=256, eval_batch_size=4096, learning_rate=.001,
        derpp_alpha=1., derpp_beta=1., deadline_iso=None,
        normal_training_rows=50000, ema_decay=.999, ema_consistency=1., hidden=128,
        section156=cfg.switches(),
        section165_memory='literature' if a.method == 'R3-L' else 'exact_c1')
    output = a.output_root.resolve() / ('smoke' if a.smoke else 'confirmation') / a.dataset.lower() / a.method / f'seed{a.seed}'
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    data, _, _, _ = prepare_setting(a.dataset, config)
    if a.smoke:
        data, _, _ = smoke_subset(data, None, None, limit=64)
    identity = dict(git_commit=source_commit(ROOT), dataset=a.dataset, seed=a.seed,
        method=a.method, effective_config=config, data_binding=binding(data, None, None),
        command=sys.argv, output_role='smoke' if a.smoke else 'confirmation', smoke=a.smoke)
    def trainer(model, x, y, *, schedule, **kwargs):
        return train_gate(model, x, y, kind='histgb', **kwargs)
    result = run_cell(data, config, identity, output / 'cell', cfg,
                      model_builder=build_models, gate_trainer=trainer)
    save_json(output / 'summary.json', dict(completed=True, scientific_result=not a.smoke,
        source_commit=identity['git_commit'], dataset=a.dataset, seed=a.seed,
        method=a.method, checkpoints=len(result['checkpoints']),
        elapsed_seconds=time.perf_counter()-started))


if __name__ == '__main__':
    main()
