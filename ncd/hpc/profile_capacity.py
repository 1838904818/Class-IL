"""One bounded local CPU resource profile; never a formal result or retry."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ncd.hpc.section81_capacity_job import ProcessSample, protect, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--protected-dir', type=Path, required=True)
    args = p.parse_args()
    out, safe = args.output_dir.resolve(), args.protected_dir.resolve()
    if out == safe or out in safe.parents or safe in out.parents:
        p.error('Separate roots required')
    if out.exists() or safe.exists():
        p.error('Existing profile must not be repeated')
    out.mkdir(parents=True)
    safe.mkdir(parents=True)
    command = [sys.executable, '-I', '-B', str(ROOT / 'ncd/run_section81_capacity.py'),
               '--dataset', 'CIC-IDS-2017', '--width', '1024', '--method', 'ar',
               '--seed', '1', '--profile', '--data-root', str(args.data_root.resolve()),
               '--output-dir', str(out / 'run')]
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4', CUDA_VISIBLE_DEVICES='')
    start, peak, reason = time.monotonic(), 0, None
    with (out / 'stdout.log').open('x') as stdout, (out / 'stderr.log').open('x') as stderr:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        monitor = ProcessSample(child.pid)
        try:
            while child.poll() is None:
                elapsed = time.monotonic() - start
                try:
                    cpu, rss = monitor.sample()
                except (ProcessLookupError, FileNotFoundError):
                    continue
                peak = max(peak, rss)
                status = dict(pid=child.pid, elapsed_seconds=elapsed, rss_bytes=rss,
                              peak_rss_bytes=peak, cpu_percent_one_core=cpu)
                write(out / 'status.json', status)
                with (out / 'resources.jsonl').open('a') as stream:
                    stream.write(json.dumps(status, allow_nan=False) + '\n')
                protect(out, safe)
                if elapsed >= 900 or rss > 8 * 1024**3:
                    reason = '15-minute profile bound' if elapsed >= 900 else '8-GiB profile guard'
                    break
                time.sleep(2)
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
            code = child.wait()
            write(out / 'profile_exit.json', dict(exit_code=code, reason=reason,
                  scientific_result=False, elapsed_seconds=time.monotonic()-start,
                  peak_rss_bytes=peak, completed=code == 0 and reason is None))
    protect(out, safe)
    return code if reason is None else 2


if __name__ == '__main__':
    raise SystemExit(main())
