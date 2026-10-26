"""Publish completed screening evidence, without inference or sealed reads."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_section81 import public


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cell', type=Path, required=True)
    p.add_argument('--log-prefix', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--confirmation-r3l', action='store_true')
    a = p.parse_args()
    seeds = (10, 11, 12, 13, 14) if a.confirmation_r3l else (1, 2)
    role = 'confirmation' if a.confirmation_r3l else 'screen'
    summary = json.loads((a.cell/'summary.json').read_text())
    if summary['seed'] not in seeds or not summary['completed'] or not summary['scientific_result']:
        raise ValueError('Completed formal screening cell required')
    path = a.cell/'cell/result.json'
    raw = json.loads(path.read_text())
    identity = raw['identity']
    if identity['output_role'] != role or identity['smoke'] or not raw['completed']:
        raise ValueError('Not a formal screening result')
    if a.confirmation_r3l and (identity['method'] != 'R3-L' or
            identity['effective_config'].get('section165_memory') != 'literature'):
        raise ValueError('Not an R3-L confirmation')
    for key in ('dataset', 'seed', 'method'):
        if identity[key] != summary[key]:
            raise ValueError('Summary identity mismatch')
    if identity['git_commit'] != summary['source_commit'] or len(raw['checkpoints']) != len(raw['tasks']):
        raise ValueError('Source or completeness mismatch')
    for i, cp in enumerate(raw['checkpoints']):
        rel = f'checkpoints/checkpoint_{i:03d}.pt'
        if cp['checkpoint_file'] != rel or hashlib.sha256((a.cell/'cell'/rel).read_bytes()).hexdigest() != cp['checkpoint_sha256']:
            raise ValueError('Checkpoint changed')
    body = public(dict(raw, execution_summary=summary), hashlib.sha256(path.read_bytes()).hexdigest(), 'result.json')
    body['publication_derivation']['original_json'] = ('section165/confirmation/cell/result.json'
        if a.confirmation_r3l else 'section156/screen/cell/result.json')
    body['execution_cost'] = dict(elapsed_seconds=summary['elapsed_seconds'],
        sampled_peak_rss_bytes=None, process_exit_code=None,
        limitation='Standalone run: completion summary verified; OS exit code and peak RSS were not recorded.')
    destination = a.destination/summary['dataset'].lower()/f"{summary['method']}_seed{summary['seed']}"
    logs = {}
    body['log_artifacts'] = {}
    for stream in ('stdout', 'stderr'):
        content = Path(str(a.log_prefix)+f'.{stream}.log').read_bytes()
        text = content.decode('utf-8', errors='replace')
        text = re.sub(r'[A-Za-z]:[/\\]Users[/\\][^\s"\n]+', 'LOCAL_PATH', text)
        if re.search(r'github_pat_|ghp_|BEGIN .*PRIVATE KEY', text):
            raise ValueError('Sensitive log pattern')
        logs[stream] = gzip.compress(text.encode(), mtime=0)
        body['log_artifacts'][stream] = dict(path=f'{stream}.log.gz',
            raw_sha256=hashlib.sha256(content).hexdigest(),
            published_sha256=hashlib.sha256(logs[stream]).hexdigest())
    def strip_local(value):
        if isinstance(value, dict):
            return {k: strip_local(v) for k, v in value.items()}
        if isinstance(value, list):
            return [strip_local(v) for v in value]
        if isinstance(value, str) and re.search(r'[A-Za-z]:[/\\]', value):
            return 'LOCAL_PATH_REDACTED'
        return value
    body = strip_local(body)
    body['publication_derivation']['additional_redaction'] = 'Windows absolute-path string values removed'
    from export_section81 import canonical
    body['publication_derivation']['published_identity_canonical_sha256'] = canonical(body['identity'])
    text = json.dumps(body, indent=2, allow_nan=False)
    if re.search(r'[A-Za-z]:[/\\]|github_pat_|ghp_|BEGIN .*PRIVATE KEY', text):
        raise ValueError('Sensitive result pattern')
    destination.mkdir(parents=True, exist_ok=False)
    (destination/'result.json').write_text(text+'\n', encoding='utf-8')
    for stream, content in logs.items():
        (destination/f'{stream}.log.gz').write_bytes(content)
    print(destination)


if __name__ == '__main__':
    main()
