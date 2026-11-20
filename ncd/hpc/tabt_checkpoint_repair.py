"""Preserve trained task-0 tensors while rebinding a scoped evaluation repair."""
import copy
import hashlib
import json
from pathlib import Path

CHECKPOINTS = {
    10: 'a77687bab564682b9aede9e2284ffd798fac70327c9fe24c84324f34dc524d06',
    11: '31368c8f7fd6c915054da0eeee243860fc00f90fa95d17c81caeba1572b532a8',
    12: '69e625f8682f845d3ecb9b337cc7bdb2c75188aba7824a85ffc0a3b47b2aa695',
    13: '8d2061cd8bf5383830c137b0ebea5fc37695cf12d5e846f255f89c6160b01a25',
    14: 'c4d5af9789b347c4bc54873fbc0c2dfa30c224121d5379f56dccf280e6517a36',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def equal(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def prepare(source_root, output, identity, seed):
    import torch
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    assert source_root != output and identity['dataset'] == 'CSE-CIC-IDS2018'
    assert not identity['effective_config']['smoke'] and seed in CHECKPOINTS
    source = source_root / 'cell/checkpoints/checkpoint_000.pt'
    assert digest(source) == CHECKPOINTS[seed]
    assert list(source.parent.glob('*.pt')) == [source]
    assert not (source_root / 'cell/result.json').exists()
    attempt = json.loads((source_root / 'attempt.json').read_text())
    assert attempt['status'] == 'failed' and attempt['seed'] == seed
    expected_error = 'Observed Normal FP violates tau monotonicity' if seed == 12 else 'Default-decision worker requires CPU models'
    assert expected_error in attempt['error']
    state = torch.load(source, map_location='cpu', weights_only=True)
    old = state['identity']
    ignored = {'portable_sources'}
    assert {k: v for k, v in old.items() if k not in ignored} == {k: v for k, v in identity.items() if k not in ignored}, 'Repair changes training/data/runtime identity'
    assert old['portable_sources']['tabt_extension_cell.py'] == '7f232a68c46473a08f981a13e56fe8222b374caf5573c9eba75d707cfce8c3bb'
    assert old['portable_sources']['tabt_cpu_gpu_portable.py'] == '15d611c4fb46ca5a659c32bd6f782851afef5e5c05501aeb54e06cdc04b57d67'
    assert identity['portable_sources']['tabt_feature_extension.py'] == old['portable_sources']['tabt_feature_extension.py']
    assert set(identity['portable_sources']) == set(old['portable_sources']) | {'tabt_checkpoint_repair.py'}
    assert state['task_index'] == 0
    target = output / 'cell/checkpoints/checkpoint_000.pt'
    if target.exists():
        recovered = torch.load(target, map_location='cpu', weights_only=True)
        assert recovered['identity'] == identity
        assert recovered['recovered_from']['checkpoint_sha256'] == CHECKPOINTS[seed]
    else:
        recovered = copy.copy(state)
        recovered['identity'] = identity
        recovered['recovered_from'] = dict(checkpoint_sha256=CHECKPOINTS[seed],
                                           identity=old, reason='Scoped default-evaluation source repair; no retraining or tensor changes')
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.pt.tmp')
        torch.save(recovered, temporary)
        temporary.replace(target)
        recovered = torch.load(target, map_location='cpu', weights_only=True)
    for key in state:
        if key != 'identity':
            assert equal(state[key], recovered[key]), 'Repair changes trained state: ' + key
    assert digest(source) == CHECKPOINTS[seed]
    return dict(original_checkpoint_sha256=CHECKPOINTS[seed], repaired_checkpoint_sha256=digest(target),
                training_tensors_unchanged=True)
