"""Freeze only four historical Windows cells for original-prior comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--base', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert not a.output.exists(), 'Never overwrite an existing input snapshot'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

for method, seed in (('icarl', 10), ('cer-bic-common', 10),
                     ('cer-bic-common', 11), ('cer-bic-common', 12)):
    relative = ('icarl-scoped-npy/seed10' if method == 'icarl' else
                'cer-bic-common/seed10' if seed == 10 else
                f'cer-bic-common-scoped/seed{seed}')
    root = a.base / 'runs/section178/cic-ids-2018' / relative
    rp = root / ('cell/result.json' if method == 'icarl' else 'result.json')
    r = json.loads(rp.read_text(encoding='utf-8'))
    assert r['completed'] and r['identity']['seed'] == seed
    dest = a.output / 'cic-ids-2018' / method / f'seed{seed}'
    dest.mkdir(parents=True)
    files = [(rp, dest / 'result.json', digest(rp))]
    if method == 'icarl':
        for name in ('default_stages.json', 'calibrated_stages.json'):
            files.append((root / name, dest / name, digest(root / name)))
    for cp in r['checkpoints']:
        files.append((rp.parent / cp['checkpoint_file'], dest / cp['checkpoint_file'], cp['checkpoint_sha256']))
    for source, target, expected in files:
        assert digest(source) == expected
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        assert digest(target) == expected
    print(json.dumps(dict(method=method, seed=seed, checkpoints=len(r['checkpoints']), copied_files=len(files))), flush=True)
