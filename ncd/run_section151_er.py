"""Run the unmodified official ReplayIDS B2 entry with Windows I/O adapters.

Only loader workers and file locations change. The official eight-epoch config,
global vocabulary, per-experience scaling, replay and oracle are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--dependencies', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scenario', type=int, choices=(1, 2), required=True)
    p.add_argument('--seed', type=int, choices=(42, 1, 2, 3, 4), required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    upstream, data, output = (v.resolve() for v in (a.upstream, a.data, a.output))
    commit = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != 'b154c79aaf7c422cc5dc35c9763f2ab1d5775342':
        raise ValueError('Official source revision changed; inspect before use')
    if subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Official source must be unmodified')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'configs').mkdir()
    (output / 'model').mkdir()
    shutil.copy2(upstream / 'configs/config.yaml', output / 'configs/config.yaml')
    os.environ.update(WANDB_MODE='disabled', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                      TORCH_BLAS_PREFER_CUBLASLT='0', CUDA_LAUNCH_BLOCKING='1')
    sys.path[:0] = [str(a.dependencies.resolve()), str(a.runtime.resolve()), str(upstream)]
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.preferred_blas_library('cublas')
    if not torch.cuda.is_available():
        raise RuntimeError('B2 ER requires local CUDA')
    report = json.loads((data / 'data_report.json').read_text())
    for name, expected in report['artifacts'].items():
        with (data / name).open('rb') as f:
            digest = hashlib.sha256()
            for block in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(block)
            actual = digest.hexdigest()
        if actual != expected:
            raise ValueError('Prepared data hash differs: ' + name)
    original_load = np.load

    def load(path, *args, **kwargs):
        s = str(path).replace('\\', '/')
        if s.startswith('dataset/CICIDS2017/'):
            path = data / Path(s).name
        return original_load(path, *args, **kwargs)

    np.load = load
    original_loader = torch.utils.data.DataLoader

    def loader(*args, **kwargs):
        kwargs.update(num_workers=0, persistent_workers=False)
        return original_loader(*args, **kwargs)

    torch.utils.data.DataLoader = loader
    from src.training import train
    original_test = train.test_and_report
    evaluations = []

    def evaluate(*args, **kwargs):
        result, labels, predictions = original_test(*args, **kwargs)
        y, pred = np.asarray(labels), np.asarray(predictions)
        normal = y == 0
        record = dict(report=result, rows=len(y), benign_rows=int(normal.sum()),
                      argmax_fpr=float((pred[normal] != 0).mean()) if normal.any() else None)
        evaluations.append(record)
        with (output / 'evaluations.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        return result, labels, predictions

    train.test_and_report = evaluate
    config = __import__('yaml').safe_load((output / 'configs/config.yaml').read_text())
    provenance = dict(section=151, method='ER-Bal-TabT', scenario=a.scenario, seed=a.seed,
                      git_commit=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
                      upstream_commit=commit, config=config, data_artifacts=report['artifacts'],
                      command=f'python -I -B ncd/run_section151_er.py --upstream UPSTREAM --dependencies DEPENDENCIES --runtime RUNTIME --data DATA --output OUTPUT --scenario {a.scenario} --seed {a.seed}',
                      log_path='stdout.log; stderr.log', memory_percent=10,
                      adaptations=['Windows DataLoader workers=0; unchanged scientific code',
                                   'External prepared arrays, per-cell config and checkpoint directory'],
                      limitations=['Official preprocessing retains duplicates and zero-fills invalid values.',
                                   'Official vocabulary includes train/validation/test feature maxima.',
                                   'Official config uses eight epochs, not the B1 100-epoch schedule.',
                                   'Official per-task report renames micro-F1 as accuracy when labels are restricted.',
                                   'Official intransigence_f1 broadcasts oracle vector against the task matrix; corrected cumulative value is separate.'])
    (output / 'run.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    os.chdir(output)
    sys.argv = [str(upstream / 'main.py'), '--er', '--balanced', 'True', '--mem', '10',
                '--scenario', str(a.scenario), '--seed', str(a.seed), '--oracle']
    ns = dict(__name__='__main__', __file__=str(upstream / 'main.py'))
    started = time.time()
    try:
        try:
            exec(compile((upstream / 'main.py').read_text(), str(upstream / 'main.py'), 'exec'), ns)
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
        if 'oracle_acc' not in ns or len(ns['overall_accuracy_matrix']) != 4:
            raise RuntimeError('Incomplete official run')
        result = dict(provenance, completed=True, elapsed_seconds=time.time()-started,
                      evaluations=evaluations)
        for key in ('accuracy_matrix', 'f1_matrix', 'overall_accuracy_matrix',
                    'overall_f1_matrix', 'oracle_acc', 'oracle_f1'):
            result[key] = ns[key].tolist()
        result['intransigence_accuracy'] = (ns['oracle_acc']-ns['overall_accuracy_matrix']).tolist()
        result['intransigence_macro_f1_cumulative'] = (ns['oracle_f1']-ns['overall_f1_matrix']).tolist()
        result['official_intransigence_f1_broadcast_mean'] = float(ns['overall_intransigence_f1'])
        for key in ('accuracy_matrix', 'f1_matrix'):
            matrix = ns[key]
            result['final_forgetting_' + key] = float(np.mean([matrix[:3, j].max()-matrix[3, j] for j in range(3)]))
        (output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    except BaseException:
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
