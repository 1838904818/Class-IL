"""Freeze completed original-runtime comparisons with path-redacted evidence."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--base', type=Path, required=True)
a = p.parse_args()
B = a.base
target = B / 'admin/cse-original-runtime-publication-v1'
assert not target.exists(), 'Frozen archives are never overwritten'
snapshot = B / 'validation/hpc-493931-20261103'
protected = snapshot / 'protected-cse-original-runtime-20261103'
report = json.loads((B / 'logs/cse-original-runtime-independent-20261103.stdout.log').read_text())

def redact(value):
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r'(?i)[A-Z]:[\\/][^\r\n\t "\']+', '@@LOCAL_PATH@@', value)
        value = re.sub(r'/(?:home|scr)/user/[^\s"\']+', '@@OWNED_PATH@@', value)
        return value.replace('leon12138', '@@USER@@')
    return value

def numeric(value, path=()):
    if isinstance(value, dict):
        return {k: v for key, item in value.items() for k, v in numeric(item, path + (key,)).items()}
    if isinstance(value, list):
        return {k: v for key, item in enumerate(value) for k, v in numeric(item, path + (key,)).items()}
    return {path: value} if isinstance(value, (int, float, bool)) else {}

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')

target.mkdir()
for path in sorted((protected / 'results').glob('*.json')):
    original = json.loads(path.read_text())
    clean = redact(original)
    assert numeric(original) == numeric(clean)
    clean['original_bytes_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(target / path.name, clean)
logs = [protected / 'diagnosis.stdout.log', protected / 'diagnosis.stderr.log',
        snapshot / 'slurm-cse-original-runtime-493931.out', snapshot / 'slurm-cse-original-runtime-493931.err']
for name in ('stage', 'software-sample', 'submit', 'download', 'independent'):
    for stream in ('stdout', 'stderr'):
        logs.append(B / f'logs/cse-original-runtime-{name}-20261103.{stream}.log')
receipts = []
for path in logs:
    raw = path.read_bytes()
    clean = redact(raw.decode('utf-8'))
    destination = target / 'logs' / (path.name + '.gz')
    destination.parent.mkdir(exist_ok=True)
    destination.write_bytes(gzip.compress(clean.encode('utf-8'), mtime=0))
    receipts.append(dict(file='logs/' + destination.name, original_bytes=len(raw),
                         original_sha256=hashlib.sha256(raw).hexdigest(),
                         published_gzip_sha256=hashlib.sha256(destination.read_bytes()).hexdigest()))
aggregate = dict(report, job=dict(id=493931, state='COMPLETED', exit_code='0:0',
                                elapsed_seconds=694, total_cpu_seconds=1304.862, step_max_rss_kib=11238544),
                 logs=receipts, source_evaluation_sha256='316e8795a90b9848bfea6c79315c8853ed593ba73c6dfbad746b038f24fc17a6',
                 caveat='Saved-CM arithmetic independently checked; no independent per-logit reconstruction. Original primary results and thresholds are unchanged.')
write_json(target / 'aggregate.json', aggregate)
readme = '''# Historical Windows checkpoints evaluated on Linux (493931)

Scope: CSE iCaRL seed10 and BiC-common seeds10–12, four checkpoints each.
No training, threshold search, rounding adjustment, or original-result mutation.
The same saved models and original prior-shift thresholds were evaluated on Linux.
This is a derived runtime comparison, not a replacement registered primary result.

| Method | Seed | Windows final1% BAcc (%) | Linux final1% BAcc (%) | Delta (pp) |
|---|---:|---:|---:|---:|
'''
for row in report['final_one_percent']:
    readme += f"| {row['method']} | {row['seed']} | {row['windows_bacc_percent']:.11f} | {row['linux_bacc_percent']:.11f} | {row['delta_pp']:.12f} |\n"
readme += '''
The iCaRL final1% change is one additional Normal false positive; exact BAcc delta
is -4/752507 percentage points. BiC final1% confusion matrices agree exactly for
all three seeds, but this does not imply equality for all stages or thresholds.
13 of64 TEST confusion matrices differ;18 of48 holdout FP counts differ. The
largest holdout difference is961 (BiC10/task0/5%). All exact cell-level differences
are retained in aggregate.json and the four result files. No tolerance was used
to relabel a discrepancy as exact agreement. These counts do not prove a cause.

Independent validation checks integer CM deltas, unchanged supports, original
reference/identity/checkpoint/tau bindings, recalls and rational final BAcc. It
does not independently rerun model predictions or establish raw-logit bitwise
identity. The independent forward functions were reused from their fixed source.
iCaRL uses default batch4096, calibrated batch2048 and original NME prototype
construction; BiC uses batch512. Platform versions remain in each result file.

Job493931 completed0:0:694s wall,1304.862s CPU,step RSS11238544KiB. This is job cost,
not an end-to-end workflow speedup. CPU-progress stopping was not triggered.
Both completed diagnostic stderr and Slurm stderr are empty. Frozen logs are
path-redacted gzip; aggregate.json retains original-byte SHA receipts. Export and
publication logs produced after this freeze remain working records, not archive
contents. Original data, checkpoints and registered results remain unchanged.

RQ1/RQ3, substantive innovation, the41-item requirements, manuscript current-source
visual QA and backup completion remain open. This group does not close them.
'''
(target / 'README.md').write_text(readme, encoding='utf-8', newline='\n')
assert len([v for v in target.rglob('*') if v.is_file()]) == 21
print('FROZEN_21_ARTIFACTS_NUMERIC_LEAVES_UNCHANGED_14_REDACTED_LOGS')
