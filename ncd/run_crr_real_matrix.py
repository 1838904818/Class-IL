"""Shared local/Slurm real-sequence matrix entry with final aggregation."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ncd.merge_crr_section15 import METHODS, SEEDS, merge, _write_csv
from ncd.run_crr_real_sequence import run_cell
from ncd.source_commit import source_commit

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['NSL-KDD', 'UNSW-NB15'])
    p.add_argument('--lambda-file', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    import torch
    torch.set_num_threads(2)
    config = json.loads((ROOT / 'ncd/configs/crr_section15.json').read_text())
    config['device'] = 'cuda'
    selection = json.loads(a.lambda_file.read_text())
    assert selection['dataset'] == a.dataset
    assert set(selection['lambdas']) == set(METHODS)
    commit = source_commit(ROOT)
    seeds = (1,) if a.smoke else SEEDS
    if a.smoke:
        config.update(epochs_per_task=1, smoke_rows_per_class=64, buffer_per_class=16)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / 'lambda_selections.json').write_text(json.dumps(selection, indent=2))
    paths = []
    for method in METHODS:
        for seed in seeds:
            path = a.output_dir / 'shards' / f'{method}_seed{seed}' / 'real_result.json'
            if path.exists():
                old = json.loads(path.read_text())
                assert old['git_commit'] == commit and old['effective_config'] == config
            else:
                print(f'CELL_START {a.dataset} {method} seed={seed}', flush=True)
                result = run_cell(config, a.dataset, method, seed, selection['lambdas'][method])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(result, indent=2, sort_keys=True))
                _write_csv(path.parent / 'real_rows.csv', result['rows'])
                _write_csv(path.parent / 'real_pi_drift_rows.csv', result['pi_drift_rows'])
                print(f'CELL_COMPLETE {a.dataset} {method} seed={seed}', flush=True)
            paths.append(path)
    result = merge(paths, 'real', a.dataset, commit, seeds)
    result['smoke'] = a.smoke
    result['lambda_source_commit'] = selection['git_commit']
    (a.output_dir / 'real_result.json').write_text(json.dumps(result, indent=2, sort_keys=True))
    _write_csv(a.output_dir / 'real_result.csv', result['rows'])
    _write_csv(a.output_dir / 'real_pi_drift_rows.csv', result['pi_drift_rows'])
    (a.output_dir / 'COMPLETED.json').write_text(json.dumps({'source_commit': commit, 'dataset': a.dataset, 'rows': len(result['rows']), 'smoke': a.smoke}))
    print(f'MATRIX_COMPLETE rows={len(result["rows"])} output={a.output_dir}', flush=True)

if __name__ == '__main__':
    main()
