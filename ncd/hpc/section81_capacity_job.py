"""Bounded four-worker CPU capacity matrix; no scheduler submission or retry."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("ar", "replay_bs", "freq", "wacb")
SEEDS = (1, 2, 3, 4, 42)


class ProcessSample:
    """Linux procfs sampling requires no additional cluster package."""
    def __init__(self, pid):
        self.pid = pid
        self.previous = None

    def cpu_affinity(self, cpus):
        os.sched_setaffinity(self.pid, cpus)

    def sample(self):
        if os.name == "nt":
            import ctypes as ct
            from ctypes import wintypes as wt
            class Memory(ct.Structure):
                _fields_ = [("cb", wt.DWORD), ("faults", wt.DWORD)] + [(name, ct.c_size_t) for name in
                    ("peak", "rss", "qpp", "qp", "qnp", "qn", "page", "pagepeak")]
            kernel = ct.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.restype = wt.HANDLE
            handle = kernel.OpenProcess(0x410, False, self.pid)
            if not handle:
                raise ProcessLookupError(self.pid)
            try:
                times = [ct.c_uint64() for _ in range(4)]
                if not kernel.GetProcessTimes(wt.HANDLE(handle), *(ct.byref(v) for v in times)):
                    raise ProcessLookupError(self.pid)
                mem = Memory(); mem.cb = ct.sizeof(mem)
                if not ct.WinDLL("psapi").GetProcessMemoryInfo(wt.HANDLE(handle), ct.byref(mem), mem.cb):
                    raise ProcessLookupError(self.pid)
                cpu, rss = (times[2].value + times[3].value) / 1e7, mem.rss
            finally:
                kernel.CloseHandle(wt.HANDLE(handle))
        else:
            fields = Path(f"/proc/{self.pid}/stat").read_text().rsplit(")", 1)[1].split()
            cpu = (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
            rss = int(Path(f"/proc/{self.pid}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        now = time.monotonic()
        usage = 0. if self.previous is None else 100 * (cpu-self.previous[0]) / (now-self.previous[1])
        self.previous = (cpu, now)
        return usage, rss


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def protect(source, target):
    for path in source.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".log", ".jsonl"):
            dest = target / path.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("UNSW-NB15", "CIC-IDS-2017"))
    parser.add_argument("--width", required=True, type=int, choices=(512, 1024))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protected-dir", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.smoke:
        if not os.environ.get("SLURM_JOB_ID") or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != 16:
            parser.error("Formal matrix requires a 16-CPU Slurm allocation")
        if os.environ.get("SLURM_JOB_PARTITION") not in ("cpu-epyc-genoa", "cpu-epyc"):
            parser.error("CPU partition required")
    output, protected = args.output_dir.resolve(), args.protected_dir.resolve()
    if output == protected or output in protected.parents or protected in output.parents:
        parser.error("Active and protected roots must be separate")
    output.mkdir(parents=True, exist_ok=args.resume)
    protected.mkdir(parents=True, exist_ok=args.resume)
    queue = [(method, seed) for seed in ((1,) if args.smoke else SEEDS) for method in METHODS]
    live, finished = {}, []
    start = time.monotonic()
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    affinity = list(range(os.cpu_count())) if os.name == "nt" else sorted(os.sched_getaffinity(0))
    if not args.smoke and len(affinity) < 16:
        raise RuntimeError("Slurm CPU affinity has fewer than sixteen CPUs")
    low_since = None
    error = None
    try:
        while queue or live:
            elapsed = time.monotonic() - start
            if stopping or elapsed >= args.wall_seconds - (5 if args.smoke else 120):
                raise RuntimeError("Termination signal or walltime checkpoint margin reached")
            for slot in range(4):
                if slot in live or not queue:
                    continue
                method, seed = queue.pop(0)
                cell = output / f"{method}_seed{seed}"
                if args.resume and (cell / "summary.json").exists():
                    summary = json.loads((cell / "summary.json").read_text(encoding="utf-8"))
                    entry_sha = hashlib.sha256((ROOT / "ncd/run_section81_capacity.py").read_bytes()).hexdigest()
                    if not summary.get("completed") or summary["entry_sha256"] != entry_sha:
                        raise RuntimeError("Resume summary incomplete or entry source changed")
                    finished.append(dict(method=method, seed=seed, exit_code=0, resumed_completed=True, summary=summary))
                    continue
                cmd = [sys.executable, "-I", "-B", str(ROOT / "ncd/run_section81_capacity.py"),
                    "--dataset", args.dataset, "--width", str(args.width), "--seed", str(seed),
                    "--method", method, "--data-root", str(args.data_root.resolve()),
                    "--output-dir", str(cell), "--smoke" if args.smoke else "--formal"]
                if args.resume and cell.exists():
                    cmd.append("--resume")
                out = (output / f"{method}_seed{seed}.stdout.log").open("a", encoding="utf-8")
                err = (output / f"{method}_seed{seed}.stderr.log").open("a", encoding="utf-8")
                env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                    NUMEXPR_NUM_THREADS="4", CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
                process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=out, stderr=err)
                monitor = ProcessSample(process.pid)
                if not args.smoke:
                    monitor.cpu_affinity(affinity[slot * 4:(slot + 1) * 4])
                monitor.sample()
                live[slot] = dict(process=process, monitor=monitor, method=method, seed=seed,
                    start=time.monotonic(), out=out, err=err, cell=cell)
            time.sleep(1 if args.smoke else 10)
            cpu, rss, rows = 0., 0, []
            for slot, row in list(live.items()):
                code = row["process"].poll()
                if code is not None:
                    row["out"].close(); row["err"].close()
                    summary_path = row["cell"] / "summary.json"
                    if code != 0 or not summary_path.exists():
                        raise RuntimeError(f"Cell {row['method']} seed {row['seed']} exited {code}; logs retained")
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    if not summary.get("completed"):
                        raise RuntimeError("Cell summary is incomplete")
                    finished.append(dict(method=row["method"], seed=row["seed"], exit_code=code,
                        elapsed_seconds=time.monotonic()-row["start"], summary=summary))
                    del live[slot]
                else:
                    try:
                        usage, memory = row["monitor"].sample()
                    except (ProcessLookupError, FileNotFoundError):
                        continue
                    cpu += usage; rss += memory
                    rows.append(dict(method=row["method"], seed=row["seed"], pid=row["process"].pid,
                        cpu_percent_one_core=usage, rss_bytes=memory))
            now = time.monotonic()
            utilization = cpu / 16
            if live and utilization < 10 and not args.smoke:
                low_since = now if low_since is None else low_since
            else:
                low_since = None
            if low_since is not None and now-low_since >= args.wall_seconds/3:
                raise RuntimeError("Project CPU utilization guard: below 10% for one third of walltime")
            if rss > 30 * 1024**3:
                raise RuntimeError("Memory guard: child RSS exceeds 30 GiB")
            status = dict(state="running", utc=datetime.now(timezone.utc).isoformat(),
                job_id=os.environ.get("SLURM_JOB_ID"), partition=os.environ.get("SLURM_JOB_PARTITION"),
                node=os.environ.get("SLURMD_NODENAME"), elapsed_seconds=now-start,
                completed_cells=len(finished), pending_cells=len(queue), children=rows,
                cpu_allocation_percent=utilization, rss_bytes=rss)
            write(output / "status.json", status)
            with (output / "resources.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(status)+"\n")
            protect(output, protected)
        write(output / "summary.json", dict(completed=True, smoke=args.smoke,
            dataset=args.dataset, width=args.width, hidden_layers=3, cell_count=len(finished),
            job_id=os.environ.get("SLURM_JOB_ID"), partition=os.environ.get("SLURM_JOB_PARTITION"),
            node=os.environ.get("SLURMD_NODENAME"), elapsed_seconds=time.monotonic()-start, cells=finished))
    except BaseException as exc:
        error = str(exc)
        raise
    finally:
        for row in live.values():
            if row["process"].poll() is None:
                row["process"].terminate()
        for row in live.values():
            try:
                row["process"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                row["process"].kill(); row["process"].wait()
            row["out"].close(); row["err"].close()
        write(output / "job_exit.json", dict(completed=error is None, error=error,
            elapsed_seconds=time.monotonic()-start, completed_cells=len(finished)))
        protect(output, protected)


if __name__ == "__main__":
    main()
