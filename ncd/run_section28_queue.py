"""Sequential local Section28 queue with an absolute, child-only hard cutoff.

This launcher changes no training routine. Parent logs and each cell output
must be new. Use separate invocations only for disjoint method/seed cells.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='+', default=['ts_mem', 'aa_noattmem', 'replay_bs'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 42])
    ap.add_argument('--deadline', required=True)
    ap.add_argument('--output-root', required=True)
    ap.add_argument('--log-root', required=True)
    args = ap.parse_args()
    deadline = dt.datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        raise ValueError('Deadline needs a timezone')
    root = Path(__file__).resolve().parent.parent
    output_root = (root / args.output_root).resolve()
    log_root = Path(args.log_root).resolve()
    if output_root.exists() or log_root.exists():
        raise FileExistsError('A new queue output and log root are required')
    output_root.mkdir(parents=True)
    log_root.mkdir(parents=True)
    ledger = {'deadline': deadline.isoformat(), 'started_at': dt.datetime.now().astimezone().isoformat(),
              'cells': [], 'status': 'running'}
    def save():
        (output_root / 'queue.json').write_text(json.dumps(ledger, indent=2), encoding='utf-8')
    save()
    for seed in args.seeds:
        for method in args.methods:
            left = (deadline - dt.datetime.now().astimezone()).total_seconds()
            if left <= 0:
                ledger['status'] = 'hard_deadline'
                save()
                return
            cell_name = f'{method}_seed{seed}'
            out = output_root / cell_name
            env = os.environ.copy()
            env.pop('CUDA_VISIBLE_DEVICES', None)
            if method == 'replay_bs':
                cmd = [sys.executable, '-I', '-u', 'ncd/run_replay_bs.py', '--datasets', 'CIC-IDS-2017',
                       '--seeds', str(seed), '--output-dir', str(out), '--deadline', deadline.isoformat()]
                env.pop('NCD_SOURCE_COMMIT', None)
            else:
                cmd = [sys.executable, '-I', '-u', 'ncd/run_aacil.py', '--datasets', 'CIC-IDS-2017',
                       '--methods', method, '--seeds', str(seed), '--gate-step-policy', 'matched_s2000',
                       '--calibration-policy', 'permanent_normal_holdout_outside_m', '--output-dir', str(out)]
                env['NCD_SOURCE_COMMIT'] = '414136b418cbb5f682875309b00e4f8151240bfe'
            begin = time.perf_counter()
            entry = {'method': method, 'seed': seed, 'started_at': dt.datetime.now().astimezone().isoformat(),
                     'output': out.relative_to(root).as_posix(), 'status': 'running'}
            with (log_root / (cell_name + '.console.log')).open('xb') as stdout, \
                 (log_root / (cell_name + '.stderr.log')).open('xb') as stderr:
                child = subprocess.Popen(cmd, cwd=root, env=env, stdout=stdout, stderr=stderr,
                                         creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                entry['pid'] = child.pid
                ledger['cells'].append(entry)
                save()
                print(json.dumps(entry), flush=True)
                try:
                    rc = child.wait(timeout=max(0.1, left))
                    entry['status'] = 'process_exited'
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:
                        rc = child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        rc = child.wait()
                    entry['status'] = 'hard_deadline_terminated_owned_child'
                entry['exit_code'] = rc
                entry['elapsed_seconds'] = time.perf_counter() - begin
                entry['finished_at'] = dt.datetime.now().astimezone().isoformat()
                # Exit zero alone is not evidence of a complete scientific cell.
                results = list(out.glob('*/*/result.json')) if out.exists() else []
                entry['complete_results'] = sum(bool(json.loads(p.read_text(encoding='utf-8')).get('completed')) for p in results)
                save()
                print(json.dumps(entry), flush=True)
                if rc != 0 or entry['complete_results'] != 1:
                    ledger['status'] = 'stopped_on_incomplete_child'
                    save()
                    return
    ledger['status'] = 'complete'
    ledger['finished_at'] = dt.datetime.now().astimezone().isoformat()
    save()
    print('QUEUE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
