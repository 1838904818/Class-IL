"""Bounded CPU capacity matrices; no scheduler submission or automatic retry."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import hashlib
import math
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
    parser.add_argument("--dataset", choices=("UNSW-NB15", "CIC-IDS-2017"))
    parser.add_argument("--width", type=int, choices=(512, 1024))
    parser.add_argument("--remaining", action="store_true", help="Only UNSW1024, CIC512 and CIC1024; excludes completed UNSW512")
    parser.add_argument("--workers", type=int, default=4, choices=(4, 16))
    parser.add_argument("--memory-limit-gib", type=float, default=30.)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protected-dir", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--exclude-cell", action="append", default=[], help="Exact dataset/method/seed already assigned or complete")
    args = parser.parse_args()
    if not args.confirmation and args.remaining == (args.dataset is not None or args.width is not None):
        parser.error("Choose --remaining or both --dataset and --width")
    if not args.confirmation and not args.remaining and (args.dataset is None or args.width is None):
        parser.error("Single matrix requires dataset and width")
    if not math.isfinite(args.memory_limit_gib) or args.memory_limit_gib <= 0:
        parser.error("Memory guard must be positive and based on measurement")
    allocated_cpus = args.workers * 4
    if not args.smoke:
        if not os.environ.get("SLURM_JOB_ID") or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != allocated_cpus:
            parser.error("Formal matrix requires exactly four allocated CPUs per worker")
        if os.environ.get("SLURM_JOB_PARTITION") not in ("cpu-epyc-genoa", "cpu-epyc"):
            parser.error("CPU partition required")
    output, protected = args.output_dir.resolve(), args.protected_dir.resolve()
    if output == protected or output in protected.parents or protected in output.parents:
        parser.error("Active and protected roots must be separate")
    output.mkdir(parents=True, exist_ok=args.resume)
    protected.mkdir(parents=True, exist_ok=args.resume)
    pairs = [("UNSW-NB15", 1024), ("CIC-IDS-2017", 512), ("CIC-IDS-2017", 1024)] if args.remaining else [(args.dataset, args.width)]
    queue = [(dataset, width, method, seed) for dataset, width in pairs
             for seed in ((1,) if args.smoke else SEEDS) for method in METHODS]
    if args.confirmation:
        pairs = [("UNSW-NB15", 128), ("CIC-IDS-2017", 128)]
        queue = [(dataset, width, method, seed) for dataset, width in pairs[:1 if args.smoke else 2]
                 for seed in ((10,) if args.smoke else (10, 11, 12, 13, 14))
                 for method in ("ar_joint", "ar", "replay_bs", "freq", "wacb")
                 if f"{dataset}/{method}/{seed}" not in args.exclude_cell]
    live, finished = {}, []
    start = time.monotonic()
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    affinity = list(range(os.cpu_count())) if os.name == "nt" else sorted(os.sched_getaffinity(0))
    if not args.smoke and len(affinity) < allocated_cpus:
        raise RuntimeError("Slurm CPU affinity is smaller than the worker allocation")
    low_since = None
    error = None
    try:
        while queue or live:
            elapsed = time.monotonic() - start
            if stopping or elapsed >= args.wall_seconds - (5 if args.smoke else 120):
                raise RuntimeError("Termination signal or walltime checkpoint margin reached")
            for slot in range(args.workers):
                if slot in live or not queue:
                    continue
                dataset, width, method, seed = queue.pop(0)
                stem = f"{dataset.lower()}_w{width}_{method}_seed{seed}" if args.remaining or args.confirmation else f"{method}_seed{seed}"
                cell = output / stem
                if args.confirmation and args.resume and (cell / "summary.json").exists():
                    summary = json.loads((cell / "summary.json").read_text())
                    expected = dict(dataset=dataset, method=method, seed=seed, smoke=args.smoke,
                                    scientific_result=not args.smoke)
                    if not summary.get("completed") or any(summary.get(k) != v for k, v in expected.items()):
                        raise RuntimeError("Confirmation completed marker identity differs")
                    if summary.get("source_commit") != os.environ.get("NCD_SOURCE_COMMIT"):
                        raise RuntimeError("Resume must use the original explicit source commit")
                    finished.append(dict(dataset=dataset, method=method, seed=seed,
                                         exit_code=0, resumed_completed=True, summary=summary))
                    continue
                if args.resume and (cell / "summary.json").exists():
                    summary = json.loads((cell / "summary.json").read_text(encoding="utf-8"))
                    entry_sha = hashlib.sha256((ROOT / "ncd/run_section81_capacity.py").read_bytes()).hexdigest()
                    expected = dict(dataset=dataset, width=width, method=method, seed=seed,
                                    smoke=args.smoke, scientific_result=not args.smoke)
                    if (not summary.get("completed") or summary["entry_sha256"] != entry_sha
                            or any(summary.get(k) != v for k, v in expected.items())):
                        raise RuntimeError("Resume summary incomplete or entry source changed")
                    finished.append(dict(dataset=dataset, width=width, method=method, seed=seed, exit_code=0, resumed_completed=True, summary=summary))
                    continue
                cmd = [sys.executable, "-I", "-B", str(ROOT / "ncd/run_section81_capacity.py"),
                    "--dataset", dataset, "--width", str(width), "--seed", str(seed),
                    "--method", method, "--data-root", str(args.data_root.resolve()),
                    "--output-dir", str(cell), "--smoke" if args.smoke else "--formal"]
                if args.resume and cell.exists():
                    cmd.append("--resume")
                if args.confirmation:
                    entry = "run_section123_joint.py" if method == "ar_joint" else "run_section123_fixed.py"
                    cmd = [sys.executable, "-I", "-B", str(ROOT / "ncd" / entry),
                           "--dataset", dataset, "--seed", str(seed),
                           "--data-root", str(args.data_root.resolve()), "--output-dir", str(cell)]
                    if method != "ar_joint":
                        cmd += ["--method", method]
                    if args.smoke:
                        cmd.append("--smoke")
                    if args.resume and cell.exists():
                        cmd.append("--resume")
                out = (output / f"{stem}.stdout.log").open("a", encoding="utf-8")
                err = (output / f"{stem}.stderr.log").open("a", encoding="utf-8")
                env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                    NUMEXPR_NUM_THREADS="4", CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
                process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=out, stderr=err)
                monitor = ProcessSample(process.pid)
                if not args.smoke:
                    monitor.cpu_affinity(affinity[slot * 4:(slot + 1) * 4])
                monitor.sample()
                live[slot] = dict(process=process, monitor=monitor, dataset=dataset, width=width, method=method, seed=seed,
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
                    finished.append(dict(dataset=row["dataset"], width=row["width"], method=row["method"], seed=row["seed"], exit_code=code,
                        elapsed_seconds=time.monotonic()-row["start"], summary=summary))
                    del live[slot]
                else:
                    try:
                        usage, memory = row["monitor"].sample()
                    except (ProcessLookupError, FileNotFoundError):
                        continue
                    cpu += usage; rss += memory
                    rows.append(dict(dataset=row["dataset"], width=row["width"], method=row["method"], seed=row["seed"], pid=row["process"].pid,
                        cpu_percent_one_core=usage, rss_bytes=memory))
            now = time.monotonic()
            utilization = cpu / allocated_cpus
            if live and utilization < 10 and not args.smoke:
                low_since = now if low_since is None else low_since
            else:
                low_since = None
            if low_since is not None and now-low_since >= args.wall_seconds/3:
                raise RuntimeError("Project CPU utilization guard: below 10% for one third of walltime")
            if rss > args.memory_limit_gib * 1024**3:
                raise RuntimeError("Memory guard: child RSS exceeds configured measured limit")
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
            dataset=args.dataset, width=args.width, matrix=pairs, workers=args.workers,
            hidden_layers=2 if args.confirmation else 3, cell_count=len(finished),
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
