"""Copy trusted own BiC partial states into a distinct portable run identity."""
from copy import deepcopy
import json


def import_partial(source, destination, identity, ps, code_root):
    old = json.loads((source/'identity.json').read_text())
    assert old['seed'] == identity['seed'] == 13
    assert old['dataset'] == identity['dataset'] == 'CSE-CIC-IDS2018'
    assert len(old['source_sha256']) == 403
    assert old['source_sha256'] == identity['source_sha256']
    for name, digest in old['source_sha256'].items():
        assert ps.digest(code_root/name) == digest, name
    excluded = {'git_commit','command','log_paths','entry_adaptation','portable_entry_sha256','effective_config'}
    assert {k:v for k,v in old.items() if k not in excluded} == {
        k:v for k,v in identity.items() if k not in excluded}
    assert {k:v for k,v in old['effective_config'].items() if k != 'problem_scout_data_root'} == {
        k:v for k,v in identity['effective_config'].items() if k != 'problem_scout_data_root'}
    for folder in ('checkpoints','progress'):
        for path in sorted((source/folder).glob('*.pt')):
            before = ps.digest(path)
            target = destination/folder/path.name
            if target.exists():
                ps.checkpoint_load(target, identity)
                continue
            original = ps.checkpoint_load(path, old)
            imported = deepcopy(original)
            imported['identity'] = identity
            imported['portable_import'] = dict(source_sha256=before, original_identity=old,
                source_file=f'{folder}/{path.name}', payload_unchanged=True)
            ps.checkpoint_save(target, imported)
            restored = ps.checkpoint_load(target, identity)
            import torch
            def equal(a,b):
                if isinstance(a,torch.Tensor):
                    assert isinstance(b,torch.Tensor) and a.dtype == b.dtype and torch.equal(a,b)
                elif isinstance(a,dict):
                    assert a.keys() == b.keys()
                    for k in a: equal(a[k],b[k])
                elif isinstance(a,(list,tuple)):
                    assert type(a) is type(b) and len(a)==len(b)
                    for x,y in zip(a,b): equal(x,y)
                else: assert a == b
            equal({k:v for k,v in original.items() if k!='identity'},
                  {k:v for k,v in restored.items() if k not in ('identity','portable_import')})
            assert ps.digest(path) == before
            print(f'PARTIAL_IMPORTED_EXACT {folder}/{path.name} {before}',flush=True)
