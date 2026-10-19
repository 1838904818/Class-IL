"""Bounded Section29 selection and formal queue; one owned GPU child at a time."""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deadline', required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--output-root', default='ncd/results/problem_scout/section29/queue_v1')
    p.add_argument('--formal-root', default='ncd/results/problem_scout/section29/local_v1')
    p.add_argument('--log-root', required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output_root).resolve()
    formal = (root / args.formal_root).resolve()
    logs = Path(args.log_root).resolve()
    deadline = dt.datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        raise ValueError('Deadline must include timezone')
    if any(path.exists() for path in (output, formal, logs)):
        raise FileExistsError('New output, formal and log roots required; never overwrite a prior attempt')
    for path in (output, formal, logs):
        path.mkdir(parents=True)
    clock = lambda: dt.datetime.now().astimezone().isoformat()
    state = {'started_at': clock(), 'deadline': deadline.isoformat(),
             'source_commit': args.source_commit, 'status': 'running', 'attempts': [],
             'selection_rule': 'UNSW seeds5/6 mean over stages of validation BAcc at alpha=.02; permanent Normal holdout plus legal retained attack validation; no test selection',
             'tie_rule': 'Higher validation mean, then lower lambda_G, alpha, beta',
             'formal_seed_order': [1, 2, 3, 4, 42]}

    def save():
        (output / 'queue.json').write_text(json.dumps(state, indent=2), encoding='utf-8')

    def run(name, child_output, extra):
        remaining = (deadline - dt.datetime.now().astimezone()).total_seconds()
        if remaining <= 0:
            state['status'] = 'hard_deadline'; save(); return None
        env = os.environ.copy()
        env.pop('CUDA_VISIBLE_DEVICES', None)
        env['NCD_SOURCE_COMMIT'] = args.source_commit
        cmd = [sys.executable, '-I', '-B', '-u', 'ncd/run_aacil_kd.py',
               '--output-dir', str(child_output), *extra]
        attempt = {'name': name, 'started_at': clock(), 'status': 'running',
                   'output': child_output.relative_to(root).as_posix()}
        start = time.perf_counter()
        with (logs / (name + '.console.log')).open('xb') as stdout, (logs / (name + '.stderr.log')).open('xb') as stderr:
            child = subprocess.Popen(cmd, cwd=root, env=env, stdout=stdout, stderr=stderr,
                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            attempt['pid'] = child.pid
            state['attempts'].append(attempt); save()
            print(json.dumps(attempt), flush=True)
            try:
                code = child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    code = child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill(); code = child.wait()
                attempt['status'] = 'hard_deadline_terminated_owned_child'
                state['status'] = 'hard_deadline'
            else:
                attempt['status'] = 'process_exited'
            attempt.update(exit_code=code, finished_at=clock(), elapsed_seconds=time.perf_counter()-start)
            save(); print(json.dumps(attempt), flush=True)
        result_path = child_output / 'result.json'
        if code != 0 or not result_path.exists():
            state['status'] = state['status'] if state['status'] == 'hard_deadline' else 'stopped_on_child_failure'
            save(); return None
        result = json.loads(result_path.read_text(encoding='utf-8'))
        if result.get('execution_status') != 'complete':
            state['status'] = 'stopped_on_incomplete_child'; save(); return None
        return result

    save()
    candidates = []
    for lam, alpha, beta in itertools.product((.5, 1., 2.), repeat=3):
        name = f'lg{lam:g}_a{alpha:g}_b{beta:g}'.replace('.', 'p')
        result = run(name, output / 'tuning' / name,
                     ['--datasets', 'UNSW-NB15', '--methods', 'aacil_kd', '--seeds', '5', '6',
                      '--selection-only', '--lambda-g', str(lam), '--der-alpha', str(alpha), '--der-beta', str(beta)])
        if result is None:
            return
        if not result.get('selection_complete'):
            state['status'] = 'stopped_on_incomplete_selection'; save(); return
        candidates.append({'lambda_g': lam, 'der_alpha': alpha, 'der_beta': beta,
                           'validation_score': result['selection_score'],
                           'source_result': (output/'tuning'/name/'result.json').relative_to(root).as_posix()})
        state['completed_candidates'] = candidates; save()
    selected = min(candidates, key=lambda c: (-c['validation_score'], c['lambda_g'], c['der_alpha'], c['der_beta']))
    state['selection_complete'] = True
    state['selected'] = selected
    state['selection_completed_at'] = clock(); save()
    print('SELECTION_COMPLETE ' + json.dumps(selected), flush=True)
    for dataset in ('UNSW-NB15', 'NSL-KDD', 'CIC-IDS-2017'):
        result = run(dataset.lower().replace('-', '_'), formal/dataset.lower().replace('-', '_'),
                     ['--datasets', dataset, '--methods', 'aacil_kd', 'aacil_nogkd', 'aacil_iso',
                      '--seeds', '1', '2', '3', '4', '42', '--lambda-g', str(selected['lambda_g']),
                      '--der-alpha', str(selected['der_alpha']), '--der-beta', str(selected['der_beta'])])
        if result is None:
            return
        print('DATASET_COMPLETE ' + dataset, flush=True)
    state.update(status='complete', finished_at=clock()); save()
    print('QUEUE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
