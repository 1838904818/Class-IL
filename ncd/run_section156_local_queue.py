"""Bounded section162 local workers; GPU and sealed cells are never launched."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ncd.section156_config import configurations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6, choices=range(1, 7))
    a = p.parse_args()
    data, output = a.data_root.resolve(strict=True), a.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Existing attempts, including failures, are never silently overwritten.
    configs = [c for c in configurations() if c.family != 3 and c.special not in ('T3', 'T4', 'T5')]
    tasks = []
    for dataset, abundance in (('UNSW-NB15', 1), ('CIC-IDS-2017', 1),
                               ('CIC-IDS-2017', 2), ('UNSW-NB15', 2)):
        for cfg in configs:
            level = 2 if cfg.switches()['normal_training_rows'] == 500000 else 1
            if level != abundance:
                continue
            for seed in (1, 2):
                cell = output/'screen'/dataset.lower()/cfg.identifier/f'seed{seed}'
                if cell.exists():
                    print(f'SKIP_EXISTING {dataset} {cfg.identifier} {seed}', flush=True)
                    continue
                tasks.append((dataset, cfg.identifier, seed))
    def run(task):
        dataset, cfg, seed = task
        prefix = output/f'{cfg}-{dataset.lower()}-seed{seed}'
        command = [sys.executable, '-I', '-B', '-u', str(ROOT/'ncd/run_section156.py'),
            '--configuration', cfg, '--dataset', dataset, '--seed', str(seed),
            '--data-root', str(data), '--output-root', str(output), '--threads', '2']
        print(f'START {task}', flush=True)
        with Path(str(prefix)+'.stdout.log').open('xb') as stdout, Path(str(prefix)+'.stderr.log').open('xb') as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr)
        print(f'EXIT {task} code={result.returncode}', flush=True)
        return result.returncode
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(run, task) for task in tasks]
        failures = sum(f.result() != 0 for f in as_completed(futures))
    print(f'SUPPORTED_QUEUE_FINISHED failures={failures}; f3 and GPU variants remain', flush=True)


if __name__ == '__main__':
    main()
