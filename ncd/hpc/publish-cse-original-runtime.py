"""Dual-publish completed derived runtime comparisons; preserve original science."""
import argparse
import gzip
import hashlib
import msvcrt
import os
from pathlib import Path
import re
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--base', type=Path, required=True)
p.add_argument('--workspace', type=Path, required=True)
a = p.parse_args()
B, W = a.base, a.workspace
archive = B / 'admin/cse-original-runtime-publication-v1'
prefix = 'paper_aacil/results/section194_diagnostics/cic-ids-2018/original_runtime_493931'
files = {prefix + '/' + v.relative_to(archive).as_posix(): v.read_bytes()
         for v in archive.rglob('*') if v.is_file()}
assert len(files) == 21
for name in ('stage-cse-original-runtime.py', 'reevaluate-cse-original-runtime.py',
             'reevaluate-cse-original-runtime.sbatch', 'submit-cse-original-runtime.sh',
             'verify-cse-original-runtime.py', 'export-cse-original-runtime.py',
             'publish-cse-original-runtime.py'):
    files['ncd/hpc/' + name] = (B / 'admin' / name).read_bytes()

def scan(name, data):
    if name.endswith('.gz'):
        data = gzip.decompress(data)
    assert not re.search(rb'/home/user/[A-Za-z0-9]|/scr/user/[A-Za-z0-9]|(?<![A-Za-z])[CE]:[/\\]|ghp_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY', data), name

for name, data in files.items():
    scan(name, data)
stderr = (B / 'logs/cse-original-runtime-publication-20261103.git.stderr.log').open('ab')
def git(repo, *args):
    r = subprocess.run(['git', '-c', 'core.longpaths=true', '-C', str(repo), *args],
                       stdout=subprocess.PIPE, stderr=stderr)
    assert r.returncode == 0, args
    return r.stdout.decode().strip()

private = W / 'work/section165_paper_publish'
public = W / 'work/github_supervisor_sync_20261010'
wo = 'docs/WORK_ORDER_2026-10-09.md'
readme = 'paper_aacil/results/section178_literature/cic-ids-2018/README.md'
note = '''

### Windows/Linux provenance: original-prior reevaluation completed (493931)

CSE iCaRL10 and BiC-common10–12 saved checkpoints were evaluated read-only on
Linux, retaining their original thresholds. Derived results are under
../../section194_diagnostics/cic-ids-2018/original_runtime_493931.
Final1% BAcc delta: iCaRL10 -0.00000531556517pp (one additional Normal false
positive); BiC10/11/12 zero, with exact final1% CM agreement. This is not equality
of all outputs:13/64 TEST CM and18/48 holdout FP counts differ; maximum holdout
delta is961. All differences are retained, with no tolerance or threshold change.
The differences alone do not establish a numerical mechanism or feature cause.

Original Windows results remain immutable; these are separately identified Linux
comparisons, not silent primary replacements. Independent saved-CM/reference and
rational BAcc checks do not independently reconstruct raw logits. iCaRL default
batch4096/calibrated2048 and BiC512; fixed forward-source provenance is retained.
Job493931 COMPLETED0:0,694s wall/1304.862s CPU/11238544KiB stepRSS,4cells16cp.
No training. Current-source manuscript visual QA, RQ1/RQ3, substantive innovation,
the41-item requirements and backup completion remain open.
'''
with (W / 'work/section151_local/publication.lock').open('r+b') as lock:
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    try:
        heads = []
        common = None
        for index, (repo, url, branch) in enumerate((
                (private, 'https://github.com/1838904818/Class-IL-in-IDS.git', ''),
                (public, 'https://github.com/1838904818/Class-IL.git', 'main'))):
            assert git(repo, 'remote', 'get-url', 'origin') == url
            assert git(repo, 'branch', '--show-current') == branch
            assert not git(repo, 'diff', '--name-only') and not git(repo, 'diff', '--cached', '--name-only')
            git(repo, 'fetch', 'origin')
            git(repo, 'merge', '--ff-only', 'origin/main')
            if index == 0:
                files[readme] = (git(repo, 'show', 'HEAD:' + readme) + note).encode()
                files['.gitattributes'] = (git(repo, 'show', 'HEAD:.gitattributes') +
                    '\n\n# Exact completed original-runtime comparison bytes\n' +
                    '\n'.join('/' + n + ' -text' for n in sorted(files)) + '\n').encode()
                common = dict(files)
                assert len(common) == 30
                for name, data in common.items():
                    scan(name, data)
            own = dict(common)
            if index == 0:
                own[wo] = (git(repo, 'show', 'HEAD:' + wo) + note + '''
sacct -j493535 -X --format=JobID,State,Reason,Elapsed: PENDING,00:00:00 at01:10;
squeue reason Priority,0/10. No ETA, idle-node probing or queue migration.
DICC job 493931 submitted:2026-11-03 00:42:10 Asia/Kuala_Lumpur;COMPLETED0:0,
4/4 cells,failures empty,software_only=false. Submission and terminal evidence
are delivered together here; helper receipt/lock must never be reused. Simulation
493930 was not an actual job. Read-only Linux original-prior comparison under
sections351–353 is completed; original registered results are not overwritten.
''').encode()
            for name, data in own.items():
                if name not in (wo, readme, '.gitattributes'):
                    assert not git(repo, 'ls-tree', '--name-only', 'HEAD', '--', name), name
                dest = repo / name
                if os.name == 'nt':
                    dest = Path('\\\\?\\' + str(dest.resolve()))
                if name not in (wo, readme, '.gitattributes') and dest.exists():
                    assert dest.read_bytes() == data, ('Existing partial differs', name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
            names = list(own)
            for off in range(0, len(names), 15):
                git(repo, 'add', '--sparse', '-f', '--', *names[off:off + 15])
            assert set(git(repo, 'diff', '--cached', '--name-only').splitlines()) == set(own)
            git(repo, '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check')
            git(repo, 'commit', '-m', 'sacct493535 PENDING;493931 completed original-runtime comparison',
                '-m', 'Retain exact stage and holdout discrepancies for four historical Windows cells evaluated read-only on Linux. Preserve original thresholds, primary results and runtime limitations.')
            git(repo, 'push', 'origin', 'HEAD:main')
            head = git(repo, 'rev-parse', 'HEAD')
            assert git(repo, 'ls-remote', 'origin', 'refs/heads/main').split()[0] == head
            for name, data in own.items():
                expected = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
                assert git(repo, 'rev-parse', head + ':' + name) == expected, name
            heads.append(head)
            print(str(repo) + ' ' + head, flush=True)
        for name in common:
            assert git(private, 'rev-parse', heads[0] + ':' + name) == git(public, 'rev-parse', heads[1] + ':' + name)
        print('ALL30_COMMON_RAW_BLOBS_BOTH_REMOTE_HEADS_CROSSREPO_VERIFIED', flush=True)
    finally:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
