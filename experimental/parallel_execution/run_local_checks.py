"""Bounded software verification only: never launch a real experiment or network client."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
TRACKS = ('fair_comparison', 'prospective_sampling', 'repair_controls', 'mechanism', 'operations')


def source_hashes():
    names = json.loads((ROOT / 'PUBLIC_FILES.json').read_text(encoding='utf-8'))['files']
    assert len(names) == len(set(names)) and 'PUBLIC_FILES.json' in names
    hashes = {}
    for name in names:
        p = Path(name)
        assert not p.is_absolute() and '..' not in p.parts
        path = ROOT / p
        assert path.is_file() and not path.is_symlink() and (path.suffix in ('.py', '.md', '.json') or path.name == 'requirements.txt')
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    v = p.parse_args()
    output = v.output.resolve()
    if output.exists() or output == ROOT or ROOT in output.parents:
        raise SystemExit('Use a new report path outside the source package')
    start_hashes = source_hashes()
    report = {'schema': 'local-executable-checks-v1',
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'python_version': sys.version.split()[0],
              'scope': 'synthetic CPU training, source-array derivation, score export and R1 software tests',
              'remote_actions': False, 'real_data_integration_verified': False, 'experimental_results': False,
              'source_sha256': start_hashes, 'tracks': []}
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', WANDB_MODE='disabled')
    for track in TRACKS:
        folder = ROOT / track
        tests = folder / 'tests' if (folder / 'tests').exists() else folder
        command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(tests), '-p', 'test_*.py', '-v']
        try:
            run = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=180, env=env)
            log = run.stdout + run.stderr
            count = re.search(r'Ran (\d+) tests? in', log)
            skipped = re.search(r'OK \(skipped=(\d+)\)', log)
            n, skip = int(count.group(1)) if count else 0, int(skipped.group(1)) if skipped else 0
            report['tracks'].append({'id': track, 'status': 'PASS' if run.returncode == 0 and n > 0 else 'FAIL',
                                      'test_count': n, 'skipped_count': skip, 'executed_count': n - skip,
                                      'exit_code': run.returncode, 'log': log})
        except subprocess.TimeoutExpired:
            report['tracks'].append({'id': track, 'status': 'TIMEOUT', 'test_count': 0, 'skipped_count': 0,
                                     'executed_count': 0, 'exit_code': None})
    report['source_unchanged_during_checks'] = source_hashes() == start_hashes
    report['status'] = 'PASS' if report['source_unchanged_during_checks'] and all(x['status'] == 'PASS' for x in report['tracks']) else 'FAIL'
    report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('status', 'scope', 'source_unchanged_during_checks')}))
    print(json.dumps([{k: row[k] for k in ('id', 'status', 'test_count', 'executed_count', 'skipped_count')} for row in report['tracks']]))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
