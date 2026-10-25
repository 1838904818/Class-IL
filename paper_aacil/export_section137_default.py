"""Validate and publish one completed inference-only ER default evaluation."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--training', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists(), 'Never overwrite an existing published evaluation'
    job = read(a.evaluation/'job_exit.json')
    result_path = a.evaluation/'default_result.json'
    r = read(result_path)
    source_path = a.training/'run/cell/result.json'
    source = read(source_path)
    assert job['status'] == 'complete' and job['exit_code'] == 0 and job['inference_only']
    assert r['completed'] and r['inference_only'] and r['retraining'] is False
    assert sha(result_path) == job['result_sha256']
    assert sha(source_path) == r['source_result_sha256']
    assert (r['dataset'],r['seed']) == (source['identity']['dataset'],source['identity']['seed'])
    assert r['training_source_commit'] == source['identity']['git_commit']
    assert len(r['checkpoints']) == len(source['checkpoints'])
    for cp, old in zip(r['checkpoints'],source['checkpoints']):
        assert cp['task_index'] == old['task_index']
        assert cp['checkpoint_sha256'] == old['checkpoint_sha256'] == sha(a.training/'run/cell'/old['checkpoint_file'])
        assert cp['default_decision']['seen_classes'] == old['seen_classes']
    for line in (a.evaluation/'resources.jsonl').read_text().splitlines():
        assert json.loads(line).get('guard_trigger') is None
    text = json.dumps(r,indent=2,allow_nan=False)+'\n'
    for token in ('C:\\', '/home/', '/scr/', 'password', 'api_key', 'BEGIN PRIVATE'):
        assert token not in text
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(text)
    print(f'Published verified default evaluation: {r["dataset"]} seed {r["seed"]}')


if __name__ == '__main__':
    main()
