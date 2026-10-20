"""Evaluate existing new-seed Freq/WACB weights at 1%; never train or tune.

Two threads, exact default-CM reproduction. A mismatch fails closed;
CUDA recovery is explicit and supervised, never a relaxed tolerance. Published decision releases
must cover all 25 cells before any model results are opened.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import json
import signal
import subprocess
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = "0" if "--cuda-recovery" in sys.argv else "-1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from ncd import analyze_farq_operating_curves as old
from ncd import calibrate_aacil_baselines as cal
from ncd.analyze_stagewise_forgetting import logits, stage_metrics, aggregate_trajectory

SOURCE = "359102b09f389a8c1d59b60d973dd8129c1dd5a7"
DECISION = ("aacil_nogkd_wa", "replay_bs", "aacil", "freq", "wacb")
METHODS = ("freq", "wacb")
ORIGINAL_CPU_SOURCE = "f4f55927b277582de179b0c9211c272ec3997de1337375d2a72fa8778e594548"
RELEASE = ROOT / "ncd/results/problem_scout/section64/interim_local_v1"
RUNS = {
    5: "section64_local_cse_seed5_v1",
    6: "section66_local_cse_seed6_decision_v1",
    7: "section67_local_cse_seed7_decision_v1",
    8: "section69_local_cse_seed8_decision_v1",
    9: "section69_local_cse_seed9_decision_v1",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def compatible_binding(saved, expected):
    """Reuse exact completed CPU stages, retaining their original provenance."""
    if saved == expected:
        return True
    previous = dict(expected, evaluation_source_sha256=ORIGINAL_CPU_SOURCE, device="cpu")
    return saved == previous


def supervise(args):
    from ncd import run_section35_local as helper
    require(old.digest(Path(helper.__file__)) ==
            "bf8a5aa83b033927966d26dcf0d834ad27008fa4bfc1622a79c8a2dd5721239b", "Resource helper differs")
    work = args.experiment_root.resolve().parent
    ref = helper.load_reference(SimpleNamespace(code_root=work / "section35_reference_code_2701212",
        reference_package=args.experiment_root.resolve() / "section31_hpc_code_2701212_full_dependencies.tar"))
    identifiers = ref.visible_gpu_uuids()
    require(len(identifiers) == 1 and ref.gpu_sample(identifiers)["gpu_telemetry_valid"], "One valid CUDA UUID required")
    output = args.output_root.resolve()
    require(args.resume and output.is_dir(), "CUDA recovery requires explicit existing-output resume")
    protected = output.with_name(output.name + "_small")
    protected.mkdir(exist_ok=True)
    lock = work / ".section35_local_cuda.lock"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(dict(pid=os.getpid(), started_at=cal.now()), stream)
    prefix = output / f"cuda_supervisor_{os.getpid()}_{time.time_ns()}"
    child, copied, stopped = None, {}, []
    start = time.monotonic()
    record = dict(started_at=cal.now(), pid=os.getpid(), gpu_uuids=identifiers,
                  source_sha256=old.digest(Path(__file__)), status="running", training_performed=False)

    def stop(number, _frame):
        stopped.append(number)

    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, stop)
    try:
        command = [sys.executable, "-I", "-B", "-u", str(Path(__file__).resolve()), *sys.argv[1:], "--guarded-child"]
        record["command"] = command
        with prefix.with_suffix(".stdout.log").open("x") as stdout, prefix.with_suffix(".stderr.log").open("x") as stderr:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                env=dict(os.environ, SECTION71_PARENT_PID=str(os.getpid())), creationflags=subprocess.CREATE_NO_WINDOW)
            record["child_pid"] = child.pid
            old.save(prefix.with_suffix(".json"), record)
            guard, next_sample = ref.TelemetryGuard(), 0.
            while child.poll() is None:
                require(not stopped and not (output / "STOP_REQUESTED").exists(), "Owned stop requested")
                elapsed = time.monotonic() - start
                if elapsed >= next_sample:
                    sample = helper.own_process_sample(ref, child.pid, output, identifiers)
                    reason = guard.observe(sample, elapsed)
                    with prefix.with_suffix(".jsonl").open("a") as stream:
                        stream.write(json.dumps(dict(resource=sample, elapsed_seconds=elapsed, guard_trigger=reason)) + "\n")
                    next_sample = elapsed + 60
                    require(reason is None, reason)
                ref.copy_small(output, protected, copied)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            record["exit_code"] = child.returncode
            require(child.returncode == 0, "Evaluation failed; inspect retained child stderr")
        record["status"] = "complete"
    except BaseException as exc:
        record.update(status="failed", error=str(exc))
        raise
    finally:
        ref.terminate(child)
        record.update(ended_at=cal.now(), elapsed_seconds=time.monotonic()-start,
                      exit_code=child.returncode if child is not None else None)
        old.save(prefix.with_suffix(".json"), record)
        ref.copy_small(output, protected, copied)
        lock.unlink()


def release_set():
    releases = {}
    for seed in RUNS:
        path = RELEASE / f"seed_{seed}/result.json"
        release = old.read(path)
        require(release["seed"] == seed and release["physical_cells"] == 5
                and [r["method"] for r in release["rows"]] == list(DECISION),
                "Decision release identity mismatch")
        for row in release["rows"]:
            c = row["cell_attempt"]
            require(row["seed"] == c["seed"] == seed and c["method"] == row["method"]
                    and c["exit_code"] == 0 and c["ended_at"], "Incomplete decision release")
        releases[seed] = old.digest(path)
    return releases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cuda-recovery", action="store_true")
    parser.add_argument("--guarded-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(not args.guarded_child or args.cuda_recovery, "Guarded child requires CUDA mode")
    if args.cuda_recovery and not args.guarded_child:
        supervise(args)
        return
    device = "cuda" if args.cuda_recovery else "cpu"
    if args.cuda_recovery:
        parent = int(os.environ.get("SECTION71_PARENT_PID", "0"))
        require(parent == os.getppid() and parent > 0 and
                old.read(args.experiment_root.resolve().parent / ".section35_local_cuda.lock")["pid"] == parent,
                "CUDA recovery must run under its lock-holding supervisor")
        from ncd.run_replay_bs import logits_for
        from ncd.train_cil import seed_everything
    releases = release_set()
    old.TRAINING_SOURCE = SOURCE
    cal.ALPHAS = (.01,)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    output = args.output_root.resolve()
    require(args.resume or not output.exists(), "Existing output requires explicit resume")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".evaluation.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    start = time.perf_counter()
    stamp = time.time_ns()
    attempt_path = output / f"attempt_{os.getpid()}_{stamp}.json"
    attempt = dict(pid=os.getpid(), started_at=cal.now(), status="running", current=None,
                   completed_cells=[], resume=args.resume, device=device, cpu_threads=2,
                   source_sha256=old.digest(Path(__file__)), training_performed=False)
    attempt.update(command=[sys.executable, *sys.argv], runtime_versions=dict(
        python=sys.version.split()[0], torch=str(torch.__version__), numpy=np.__version__, cuda=torch.version.cuda),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        cuda_helper_sha256=old.digest(ROOT / "ncd/run_replay_bs.py") if args.cuda_recovery else None)
    cache = {}
    try:
        for seed, run in RUNS.items():
            for method in METHODS:
                attempt["current"] = dict(seed=seed, method=method, task_index=None)
                old.save(attempt_path, attempt)
                published = RELEASE / f"seed_{seed}/{method}/raw/result.json"
                prior = old.read(published)
                source = (args.experiment_root.resolve() / run / "cse_cic_ids2018/cells"
                          / method / f"seed_{seed}/raw/result.json")
                require(old.digest(source) == prior["publication_derivation"]["original_result_sha256"],
                        "Original training result differs from released source")
                ident = prior["identity"]
                cell = old.verify_source(source, ("CSE-CIC-IDS2018", method, seed),
                                         ident["source_hashes"], ident["config_sha256"])
                require(cell is not None and cell["identity"] == ident, "Released identity differs")
                binding = dict(training_result=published.relative_to(ROOT).as_posix(),
                               training_result_sha256=old.digest(source),
                               publication_sha256=old.digest(published), release_sha256=releases[seed],
                               training_identity_sha256=old.canonical(ident), training_source_commit=SOURCE,
                               evaluation_source_sha256=attempt["source_sha256"],
                               helper_hashes={p: old.digest(ROOT / p) for p in (
                                   "ncd/analyze_farq_operating_curves.py", "ncd/calibrate_aacil_baselines.py",
                                   "ncd/analyze_stagewise_forgetting.py", "ncd/model.py")},
                               alpha=.01, device=device, test_used_for_threshold_selection=False)
                folder = output / f"{method}_seed{seed}"
                folder.mkdir(exist_ok=True)
                final = folder / "result.json"
                if final.exists():
                    saved = old.read(final)
                    require(saved["completed"] and compatible_binding(saved["binding"], binding), "Completed binding differs")
                    attempt["completed_cells"].append(dict(seed=seed, method=method, reused=True))
                    continue
                data, data_seconds = old.load_data(dict(ident, dataset="CIC-IDS-2018"),
                                                    args.data_root.resolve(), cache)
                normal, ids, pool = cal.normal_pool(ident, data)
                names, tasks = ident["class_order"], ident["tasks"]
                stages = []
                costs = []
                for cp in cell["result"]["checkpoints"]:
                    tick = time.perf_counter()
                    t, seen = cp["task_index"], cp["seen_classes"]
                    attempt["current"]["task_index"] = t
                    old.save(attempt_path, attempt)
                    payload = (source.parent / cp["checkpoint_file"]).resolve()
                    require(payload.is_relative_to(source.parent.resolve())
                            and old.read(payload.with_suffix(".json")) == cp
                            and old.digest(payload) == cp["checkpoint_sha256"], "Checkpoint/sidecar mismatch")
                    stage_binding = dict(source=binding, task_index=t, seen_classes=seen,
                                         checkpoint_sha256=cp["checkpoint_sha256"])
                    target = folder / f"task_{t:03d}.json"
                    if target.exists():
                        stage = old.read(target)
                        require(stage["completed"] and
                                {k: v for k, v in stage["binding"].items() if k != "source"} ==
                                {k: v for k, v in stage_binding.items() if k != "source"} and
                                compatible_binding(stage["binding"]["source"], binding), "Saved stage differs")
                    else:
                        state = torch.load(payload, map_location="cpu", weights_only=True)
                        require(state["identity"] == ident and state["task_index"] == t
                                and state["seen_classes"] == seen and state["allocation"] == cp["allocation"],
                                "Checkpoint identity mismatch")
                        model = old.MLP(data["x_test"].shape[1], ident["effective_config"]["hidden"], len(names))
                        model.load_state_dict(state["model_state"], strict=True)
                        model.eval()
                        if args.cuda_recovery:
                            seed_everything(seed * 10000 + t)
                            model.to(device)
                        batch = ident["effective_config"]["eval_batch_size"]
                        prior_logits = torch.from_numpy(data["log_prior"][seen])
                        def forward(x, selected):
                            if args.cuda_recovery:
                                return logits_for(model, x, selected, seen, ident["effective_config"], torch.device(device))
                            return logits(model, x, selected, seen, batch)
                        holdout = forward(data["x_train"], ids)
                        grid, points, queries, gap = cal.select_taus(holdout, prior_logits, seen.index(normal))
                        rows = np.flatnonzero(np.isin(data["y_test"], seen))
                        scores = forward(data["x_test"], rows)

                        def cm(tau):
                            pred = np.asarray(seen)[(scores + tau * prior_logits).argmax(1).numpy()]
                            n = len(names)
                            return np.bincount(data["y_test"][rows] * n + pred, minlength=n*n).reshape(n, n)

                        difference = old.cm_difference(cp["id"]["confusion_matrix"], cm(0.))
                        attempt["last_default_cm_difference"] = difference
                        require(np.array_equal(cm(0.), np.asarray(cp["id"]["confusion_matrix"])),
                                "Exact default CM reproduction failed; no implicit CUDA retry")
                        point = points["0.01"]
                        stage = dict(completed=True, binding=stage_binding, task_index=t,
                                     calibration=point, metrics=stage_metrics(cm(point["total_tau"]), seen, names),
                                     normal_pool=pool, calibration_grid=grid, normal_only_queries=queries,
                                     minimum_normal_prior_gap=gap, default_cm_reproduced=True,
                                     execution_device=device, elapsed_seconds=time.perf_counter()-tick,
                                     completed_at=cal.now(), training_performed=False,
                                     test_used_for_threshold_selection=False)
                        old.save(target, stage)
                        del model, state, scores, holdout
                    require(stage_metrics(stage["metrics"]["confusion_matrix"], seen, names) == stage["metrics"],
                            "Stage metric consistency failed")
                    stages.append(dict(task_index=t, metrics=stage["metrics"]))
                    costs.append(stage["elapsed_seconds"])
                old.save(final, dict(completed=True, completed_at=cal.now(), dataset="CSE-CIC-IDS2018",
                         method=method, seed=seed, binding=binding, class_order=names, tasks=tasks,
                         alpha=.01, stages=stages, metrics=aggregate_trajectory(stages, tasks, names),
                         cost_seconds=dict(data_validation_this_attempt=data_seconds, stages_total=sum(costs)),
                         cost_scope="Stage totals nested within retained attempt wall times; do not add twice.",
                         training_performed=False, selection_changed=False, final_tests_executed=False,
                         independent_dataset_confirmation=False, original_40_still_required=True,
                         limitation="Descriptive exposed-data evaluation; target FPR is not a TEST guarantee."))
                attempt["completed_cells"].append(dict(seed=seed, method=method, reused=False))
                print(f"cell_complete {method} seed={seed}", flush=True)
        attempt.update(status="complete", current=None)
    except BaseException as exc:
        attempt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        attempt.update(ended_at=cal.now(), elapsed_seconds=time.perf_counter()-start)
        old.save(attempt_path, attempt)
        lock.unlink()


if __name__ == "__main__":
    main()
