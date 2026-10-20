"""Evaluate saved external checkpoints at 0.5/1% FPR; never train a model.

This entry uses the established Section27 calibration bindings for the three
development datasets. CSE18 requires a separately verified checkpoint inventory.
Every stage is evaluated on seen TEST classes after Normal-only calibration.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from ncd import calibrate_aacil_baselines as cal
from ncd import analyze_farq_operating_curves as old
from ncd.analyze_stagewise_forgetting import stage_metrics, aggregate_trajectory, logits
from ncd.run_replay_bs import logits_for

METHODS = ("replay", "derpp", "wacb", "freq", "joint")
SEEDS = (1, 2, 3, 4, 42)
ALPHAS = (.005, .01)
cal.ALPHAS = ALPHAS
PROTOCOL = dict(cal.CALIBRATION_PROTOCOL, section="46.2", targets=list(ALPHAS),
                stages="all saved tasks; seen-class TEST population",
                selection_changed=False, training_performed=False,
                exploratory_on_exposed_datasets=True)


def require(value, message):
    if not value:
        raise ValueError(message)


def same_binding(saved, expected):
    """Reuse the validated CPU stages from the pre-fallback implementation."""
    saved, expected = dict(saved), dict(expected)
    allowed = {"3ed52b3c1749c663b385c43546c426f0ca3691b23a5bd3358981c79557ee8a82",
               old.digest(Path(__file__))}
    require(saved.pop("evaluation_source_sha256") in allowed
            and expected.pop("evaluation_source_sha256") in allowed, "Unrecognized evaluator source")
    return saved == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=old.DATASETS, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    slug = args.dataset.lower().replace("-", "_")
    output = args.output_root.resolve() / slug
    require(args.resume or not output.exists(), "Existing output requires explicit resume")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".evaluation.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    attempt_start = time.perf_counter()
    attempt = {"pid": os.getpid(), "started_at": cal.now(), "dataset": args.dataset,
               "resume": args.resume, "completed_cells": [], "current": None,
               "analysis_source_sha256": old.digest(Path(__file__)),
               "status": "running", "training_performed": False, "cpu_threads": 2,
               "device": "cpu"}
    attempt_file = output / f"attempt_{os.getpid()}.json"
    cache = {}
    try:
        for method in METHODS:
            for seed in SEEDS:
                attempt["current"] = {"method": method, "seed": seed}
                old.save(attempt_file, attempt)
                published = ROOT / "ncd/results/problem_scout/section27/baseline_calibration_v2" / slug / f"{method}_seed{seed}/result.json"
                previous = old.read(published)
                b = previous["binding"]
                require(previous["completed"] and previous["verification"]["all_requested_checks_passed"],
                        "Existing calibration is incomplete")
                source = (ROOT / b["source_result"]).resolve()
                require(source.is_relative_to(ROOT) and old.digest(source) == b["source_result_sha256"],
                        "Original source differs from published calibration binding")
                cell = old.verify_source(source, (args.dataset, method, seed),
                                         b["source_hashes"], b["config_sha256"])
                require(cell is not None, "Incomplete training source")
                ident = cell["identity"]
                require(old.canonical(ident) == b["identity_sha256"], "Canonical source identity differs")
                data, data_seconds = old.load_data(ident, args.data_root.resolve(), cache)
                normal, ids, pool = cal.normal_pool(ident, data)
                names, tasks = ident["class_order"], ident["tasks"]
                binding = {"training_result": b["source_result"],
                           "training_result_sha256": cell["result_sha256"],
                           "training_identity_sha256": old.canonical(ident),
                           "training_source_commit": ident["git_commit"],
                           "training_source_hashes": ident["source_hashes"],
                           "effective_config_sha256": ident["effective_config_sha256"],
                           "data_binding": ident["data_binding"],
                           "previous_calibration_sha256": old.digest(published),
                           "evaluation_source_sha256": attempt["analysis_source_sha256"],
                           "helper_hashes": {p: old.digest(ROOT / p) for p in
                               ("ncd/calibrate_aacil_baselines.py", "ncd/analyze_farq_operating_curves.py",
                                "ncd/analyze_stagewise_forgetting.py", "ncd/model.py")},
                           "protocol": PROTOCOL}
                folder = output / f"{method}_seed{seed}"
                folder.mkdir(exist_ok=True)
                final_path = folder / "result.json"
                if final_path.exists():
                    prior_result = old.read(final_path)
                    require(prior_result["completed"] and same_binding(prior_result["binding"], binding),
                            "Completed evaluation binding changed")
                    attempt["completed_cells"].append({"method": method, "seed": seed, "reused": True})
                    continue
                trajectories = {str(a): [] for a in ALPHAS}
                costs = []
                for cp in cell["result"]["checkpoints"]:
                    tick = time.perf_counter()
                    t, seen = cp["task_index"], cp["seen_classes"]
                    payload = (source.parent / cp["checkpoint_file"]).resolve()
                    require(payload.is_relative_to(source.parent)
                            and old.read(payload.with_suffix(".json")) == cp
                            and old.digest(payload) == cp["checkpoint_sha256"],
                            "Stage checkpoint payload/sidecar mismatch")
                    stage_binding = {"source": binding, "checkpoint_sha256": cp["checkpoint_sha256"],
                                     "task_index": t, "seen_classes": seen}
                    target = folder / f"task_{t:03d}.json"
                    if target.exists():
                        stage = old.read(target)
                        saved_binding = dict(stage["binding"])
                        saved_source = saved_binding.pop("source")
                        require(stage["completed"] and same_binding(saved_source, binding)
                                and saved_binding == {k: v for k, v in stage_binding.items() if k != "source"},
                                "Stage cache binding differs")
                    else:
                        state = torch.load(payload, map_location="cpu", weights_only=True)
                        require(state["identity"] == ident and state["task_index"] == t
                                and state["seen_classes"] == seen and state["allocation"] == cp["allocation"],
                                "Loaded checkpoint identity differs")
                        model = old.MLP(data["x_test"].shape[1], ident["effective_config"]["hidden"], len(names))
                        model.load_state_dict(state["model_state"], strict=True)
                        model.eval()
                        prior = torch.from_numpy(data["log_prior"][seen])
                        holdout = logits(model, data["x_train"], ids, seen, ident["effective_config"]["eval_batch_size"])
                        grid, points, queries, gap = cal.select_taus(holdout, prior, seen.index(normal))
                        rows = np.flatnonzero(np.isin(data["y_test"], seen))
                        scores = logits(model, data["x_test"], rows, seen, ident["effective_config"]["eval_batch_size"])
                        n = len(names)
                        def confusion(tau):
                            pred = np.asarray(seen)[(scores + tau * prior).argmax(1).numpy()]
                            return np.bincount(data["y_test"][rows] * n + pred, minlength=n*n).reshape(n, n)
                        cpu_difference = old.cm_difference(cp["id"]["confusion_matrix"], confusion(0.))
                        execution_device = "cpu"
                        if cpu_difference["different_entries"]:
                            # The original model was trained/evaluated on CUDA.
                            # Repeat only this stage using its original device;
                            # keep the exact reproduction gate, never a tolerance.
                            require(torch.cuda.is_available(), "CPU reproduction differs; original CUDA device unavailable")
                            execution_device = "cuda"
                            model.to("cuda")
                            holdout = logits_for(model, data["x_train"], ids, seen,
                                                 ident["effective_config"], torch.device("cuda"))
                            grid, points, queries, gap = cal.select_taus(holdout, prior, seen.index(normal))
                            scores = logits_for(model, data["x_test"], rows, seen,
                                                ident["effective_config"], torch.device("cuda"))
                        require(np.array_equal(confusion(0.), np.asarray(cp["id"]["confusion_matrix"])),
                                "Saved checkpoint default prediction reproduction failed")
                        evaluated = {}
                        for alpha in ALPHAS:
                            point = points[str(alpha)]
                            evaluated[str(alpha)] = {"calibration": point,
                                "metrics": stage_metrics(confusion(point["total_tau"]), seen, names)}
                        stage = {"completed": True, "binding": stage_binding, "task_index": t,
                                 "operating_points": evaluated, "normal_pool": pool,
                                 "calibration_grid": grid, "normal_only_queries": queries,
                                 "minimum_normal_prior_gap": gap, "default_cm_reproduced": True,
                                 "execution_device": execution_device,
                                 "cpu_reproduction_difference": cpu_difference,
                                 "elapsed_seconds": time.perf_counter()-tick, "completed_at": cal.now(),
                                 "test_used_for_threshold_selection": False, "training_performed": False}
                        old.save(target, stage)
                        del model, state, scores, holdout
                    for alpha in ALPHAS:
                        metric = stage["operating_points"][str(alpha)]["metrics"]
                        require(stage_metrics(metric["confusion_matrix"], seen, names) == metric,
                                "Stage CM metrics inconsistent")
                        trajectories[str(alpha)].append({"task_index": t, "metrics": metric})
                    costs.append(stage["elapsed_seconds"])
                result = {"completed": True, "completed_at": cal.now(), "dataset": args.dataset,
                          "method": method, "seed": seed, "class_order": names, "tasks": tasks,
                          "binding": binding, "normal_holdout": pool,
                          "operating_points": {a: {"stages": ss, "metrics": aggregate_trajectory(ss, tasks, names)}
                                               for a, ss in trajectories.items()},
                          "cost_seconds": {"data_validation_this_attempt": data_seconds,
                                           "stage_evaluation_total": sum(costs)},
                          "training_performed": False, "selection_changed": False,
                          "limitations": ["Exposed development data; not independent confirmation.",
                              "Calibration target is not a test FPR guarantee.",
                              "Joint is an offline non-memory-matched reference.",
                              "Historical training and historical tests are unchanged."]}
                old.save(final_path, result)
                attempt["completed_cells"].append({"method": method, "seed": seed, "reused": False})
                print(json.dumps({"event": "cell_complete", "dataset": args.dataset, "method": method, "seed": seed}), flush=True)
        attempt.update(status="complete", current=None)
    except BaseException as exc:
        attempt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        attempt.update(ended_at=cal.now(), elapsed_seconds=time.perf_counter()-attempt_start)
        old.save(attempt_file, attempt)
        lock.unlink()


if __name__ == "__main__":
    main()
