"""Section25 D1--D3 Replay cells; historical S2000 training stays unchanged.

Only replay allocation/selection changes. Normal coverage is selected from the
prepared TRAINING feature space, never from the test split or model predictions.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import sklearn
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
import torch

from ncd.crr_benchmarks import _train
from ncd.model import MLP
from ncd.prepare_problem_scout_data import prepare_setting
from ncd.run_problem_scout import (
    Tee, binding, canonical_hash, checkpoint_load, checkpoint_save, digest,
    evaluate, memory_size, path_label, require_finite, save_json,
    seed_everything, smoke_subset,
)
from ncd.source_commit import source_commit

SOURCE_PATHS = [
    "ncd/run_benign_coverage.py", "ncd/run_problem_scout.py",
    "ncd/crr_benchmarks.py", "ncd/model.py", "ncd/prepare_problem_scout_data.py",
    "ncd/train_cil.py", "ncd/data.py", "ncd/source_commit.py",
    "ncd/run_crr_real_sequence.py",
]
KMEANS = {"n_clusters": 200, "init": "random", "n_init": 1,
          "max_iter": 100, "tol": 1e-4, "random_state": 42,
          "algorithm": "lloyd"}
POLICIES = {
    "training": "Unchanged S2000 Replay: max(8 epochs,2000 optimizer steps), batch256, MLP128, Adam, minibatch-class-balanced CE.",
    "D1": "Normal capacity2000; each attack capacity200. New classes selected with historical task-local random rule.",
    "D2": "Explicit --d2-budget-mode required: staged_archive uses capacity200*seen classes and replenishes growing Normal from original TRAINING pool; fixed_final uses capacity200*final classes from the first task, so early memory exceeds S2000 but never revisits old pools. Normal gets50%; remainder split equally over currently seen attack classes, lowest class IDs receive integer remainders. With no attack, Normal receives the entire budget. Missing rows leave unused capacity, never duplicated.",
    "D2_archive_access": "Only staged_archive replenishes growing Normal capacity from original TRAINING rows using one fixed random priority. No hidden candidate buffer is retained. Old attacks are only reduced from stored rows. This mode changes archive access as well as allocation and is not a strictly streaming memory-only mechanism.",
    "D3": "Full-batch KMeans on all original prepared Normal TRAINING rows, fixed k200/random seed42/n_init1/max_iter100/tol1e-4. Each occupied cluster contributes its nearest actual member, tie broken by original row index; empty clusters are not filled. Attack memory uses unchanged random200/class.",
    "D3_space": "Prepared input feature space, no extra scaling or embedding fit, no test selection, no hyperparameter search.",
    "cost": "Synchronized optimization wall time plus separate selection/evaluation; original kmeans fit time and cache reuse explicit. GPU peaks cover optimization only. Host RAM peak not measured.",
    "scope": "Benign-error diagnostic under RQ2; no new mechanism, RQ1/RQ3 or overall research completion claim.",
}


@contextmanager
def cell_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0"); stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def capacities(treatment, seen, normal, total_classes, d2_mode):
    if treatment != "D2":
        return {c: 2000 if treatment == "D1" and c == normal else 200 for c in seen}
    total = 200 * (total_classes if d2_mode == "fixed_final" else len(seen))
    attacks = sorted(c for c in seen if c != normal)
    if not attacks:
        return {normal: total}
    half = total // 2
    base, remainder = divmod(total - half, len(attacks))
    return {normal: half, **{c: base + (i < remainder) for i, c in enumerate(attacks)}}


def coverage_rows(data, identity, cache_path):
    normal = data.class_order.index("Normal")
    rows = np.flatnonzero(data.y_train == normal)
    params = dict(KMEANS, n_clusters=min(KMEANS["n_clusters"], len(rows)))
    cache_identity = {"dataset": identity["dataset"], "smoke": identity["smoke"],
                      "git_commit": identity["git_commit"],
                      "source_hashes": identity["source_hashes"],
                      "x_train": identity["data_binding"]["x_train"],
                      "y_train": identity["data_binding"]["y_train"],
                      "kmeans": params, "sklearn": sklearn.__version__}
    if cache_path.exists():
        result = json.loads(cache_path.read_text(encoding="utf-8"))
        if result["identity"] != cache_identity or not result.get("completed"):
            raise ValueError("Normal coverage cache has a different source, dataset or configuration")
        selected = np.asarray(result["selected_training_indices"], dtype=np.int64)
        if (len(np.unique(selected)) != len(selected) or np.any(selected < 0)
                or np.any(selected >= len(data.y_train))
                or np.any(data.y_train[selected] != normal)
                or len(selected) != result["occupied_clusters"]):
            raise ValueError("Normal coverage cache has invalid training indices")
        print("KMEANS_CACHE_REUSE", len(selected), flush=True)
        return selected, {"cache_reused": True, "cache_path": path_label(cache_path),
                          "cache_sha256": digest(cache_path), **result["statistics"]}
    started = time.perf_counter()
    print(f"KMEANS_START rows={len(rows)} clusters={params['n_clusters']}", flush=True)
    x = np.asarray(data.x_train[rows], dtype=np.float32)
    with threadpool_limits(limits=int(identity["effective_config"]["cpu_threads"])):
        fit = KMeans(**params).fit(x)
    # Membership, then nearest-centroid actual sample. Do not choose a point
    # belonging to a different cluster, or emit duplicate sample indices.
    selected, cluster_counts = [], []
    for c in range(params["n_clusters"]):
        members = np.flatnonzero(fit.labels_ == c)
        if not len(members):
            cluster_counts.append(0)
            continue
        delta = x[members] - fit.cluster_centers_[c]
        squared = np.einsum("ij,ij->i", delta, delta)
        selected.append(int(rows[members[int(np.argmin(squared))]]))
        cluster_counts.append(int(len(members)))
    selected = np.asarray(sorted(selected), dtype=np.int64)
    if not len(selected) or len(np.unique(selected)) != len(selected):
        raise ValueError("Invalid KMeans representatives")
    stats = {"kmeans_parameters": params, "fit_and_representative_seconds": time.perf_counter() - started,
             "normal_training_rows": len(rows), "n_iter": int(fit.n_iter_),
             "inertia": float(fit.inertia_), "occupied_clusters": len(selected),
             "cluster_counts": cluster_counts, "fit_reached_max_iter": int(fit.n_iter_) == params["max_iter"]}
    result = {"completed": True, "identity": cache_identity,
              "selected_training_indices": selected.tolist(), "occupied_clusters": len(selected),
              "statistics": stats}
    save_json(cache_path, result)
    print(f"KMEANS_COMPLETE representatives={len(selected)} iterations={fit.n_iter_} seconds={stats['fit_and_representative_seconds']:.3f}", flush=True)
    return selected, {"cache_reused": False, "cache_path": path_label(cache_path),
                      "cache_sha256": digest(cache_path), **stats}


def select_memory(data, treatment, seed, t, task, seen, stored, identity, cache_path):
    normal = data.class_order.index("Normal")
    mode = identity["d2_budget_mode"]
    cap = capacities(treatment, seen, normal, len(data.class_order), mode)
    selected = {int(c): np.asarray(v, dtype=np.int64).copy() for c, v in stored.items()}
    detail = {"capacities": {str(c): int(n) for c, n in cap.items()},
              "normal_archive_revisited": treatment == "D2" and mode == "staged_archive" and normal not in task,
              "normal_coverage": None}
    for c in seen:
        count = cap[c]
        if treatment == "D2" and c == normal and (mode == "staged_archive" or c in task):
            rows = np.flatnonzero(data.y_train == c)
            # One fixed priority makes larger Normal memories nested; this
            # transient permutation is not retained as an uncounted replay set.
            priority = np.random.default_rng(seed * 10000 * 10007 + c).permutation(len(rows))
            selected[c] = np.sort(rows[priority[:min(count, len(rows))]])
        elif c in task:
            if treatment == "D3" and c == normal:
                selected[c], detail["normal_coverage"] = coverage_rows(data, identity, cache_path)
            else:
                rows = np.flatnonzero(data.y_train == c)
                local = np.random.default_rng((seed * 10000 + t) * 10007 + c).choice(
                    len(rows), min(count, len(rows)), replace=False)
                selected[c] = np.sort(rows[local])
        elif len(selected[c]) > count:
            local = np.random.default_rng(seed * 27011 + c * 103 + t).choice(
                len(selected[c]), count, replace=False)
            selected[c] = np.sort(selected[c][local])
        if len(selected[c]) > count or len(np.unique(selected[c])) != len(selected[c]):
            raise ValueError("Replay memory exceeds capacity or has repeated row indices")
    detail["actual_counts"] = {str(c): len(selected[c]) for c in seen}
    detail["capacity_total"] = sum(cap.values())
    detail["actual_total"] = sum(len(selected[c]) for c in seen)
    return selected, detail


def materialize_memory(data, selected, stored, bx, task, identity):
    """Old unchanged/subsampled rows come from memory, not the old train pool."""
    old_arrays = {c: x for c, x in zip(sorted(stored), bx)}
    normal = data.class_order.index("Normal")
    result = []
    for c in sorted(selected):
        archive_normal = (identity["treatment"] == "D2" and identity["d2_budget_mode"] == "staged_archive" and c == normal)
        if c in task or archive_normal:
            result.append(data.x_train[selected[c]].copy())
        else:
            positions = np.searchsorted(stored[c], selected[c])
            if np.any(positions >= len(stored[c])) or not np.array_equal(stored[c][positions], selected[c]):
                raise ValueError("Old memory would require unauthorized pool expansion")
            result.append(old_arrays[c][positions].copy())
    return result


def run_cell(data, metadata, identity, output, config, device, cache_path):
    final_path = output / "result.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        if result.get("identity") != identity or not result.get("completed") or len(result["checkpoints"]) != len(data.tasks):
            raise ValueError("Completed cell identity/completeness differs")
        for cp in result["checkpoints"]:
            path = output / cp["checkpoint_file"]
            if digest(path) != cp["checkpoint_sha256"] or json.loads(path.with_suffix(".json").read_text(encoding="utf-8")) != cp:
                raise ValueError("Completed checkpoint or metrics changed")
        print("CELL_RESUME", identity["dataset"], identity["treatment"], identity["seed"], flush=True)
        return result
    seed, treatment = identity["seed"], identity["treatment"]
    model = MLP(data.x_train.shape[1], config["hidden"], len(data.class_order)).to(device)
    stored, bx, by, records, old, cumulative = {}, [], [], [], (), 0.0
    for t, task in enumerate(data.tasks):
        seen = tuple(c for group in data.tasks[:t + 1] for c in group)
        task_seed = seed * 10000 + t
        seed_everything(task_seed)
        cp_path = output / "checkpoints" / f"checkpoint_{t:03d}.pt"
        metric_path = cp_path.with_suffix(".json")
        if cp_path.exists():
            state = checkpoint_load(cp_path, identity)
            if state["task_index"] != t or state["seen_classes"] != list(seen):
                raise ValueError("Checkpoint task order mismatch")
            model.load_state_dict(state["model_state"])
            stored = {int(c): a.numpy().copy() for c, a in state["buffer_indices"].items()}
            bx = [a.numpy().copy() for a in state["buffer_x"]]
            by = [a.numpy().copy() for a in state["buffer_y"]]
            cumulative = state["training_seconds_cumulative"]
            print(f"CHECKPOINT_RESUME task={t}", flush=True)
        else:
            if t == 0:
                model = MLP(data.x_train.shape[1], config["hidden"], len(data.class_order)).to(device)
            mask = np.isin(data.y_train, task)
            cx, cy = data.x_train[mask], data.y_train[mask]
            tx = np.concatenate([cx, *bx]) if bx else cx
            ty = np.concatenate([cy, *by]) if by else cy
            before_n, before_bytes = memory_size(bx, by)
            schedule = {}
            if device.type == "cuda":
                torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            print(f"TASK_TRAIN task={t} current_rows={len(cy)} replay_rows={before_n}", flush=True)
            _train(model, tx, ty, method="replay", old_model=None, old_classes=old,
                   seen_classes=seen, epochs=config["epochs_per_task"], batch_size=config["batch_size"],
                   learning_rate=config["learning_rate"], seed=task_seed, regularization=0.0,
                   temperature=config["temperature"], min_steps=config["min_steps_per_task"],
                   training_stats=schedule)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            cumulative += elapsed
            target = max(config["epochs_per_task"] * ((len(ty) + config["batch_size"] - 1) // config["batch_size"]), config["min_steps_per_task"])
            if schedule.get("optimizer_steps") != target or schedule.get("target_steps") != target:
                raise ValueError("S2000 optimizer-step budget mismatch")
            peak_alloc = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
            started = time.perf_counter()
            selected, selection = select_memory(data, treatment, seed, t, task, seen, stored, identity, cache_path)
            bx = materialize_memory(data, selected, stored, bx, task, identity)
            stored = selected
            by = [np.full(len(stored[c]), c, dtype=np.int64) for c in sorted(stored)]
            selection_seconds = time.perf_counter() - started
            after_n, after_bytes = memory_size(bx, by)
            state = {"identity": identity, "task_index": t, "seen_classes": list(seen), "new_classes": list(task),
                     "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                     "buffer_x": [torch.from_numpy(a.copy()) for a in bx],
                     "buffer_y": [torch.from_numpy(a.copy()) for a in by],
                     "buffer_indices": {str(c): torch.from_numpy(a.copy()) for c, a in stored.items()},
                     "training_seconds": elapsed, "training_seconds_cumulative": cumulative,
                     "buffer_selection_seconds": selection_seconds, "training_schedule": schedule,
                     "memory_selection": selection, "replay_examples_before": before_n,
                     "replay_examples_after": after_n, "replay_bytes_before": before_bytes,
                     "replay_bytes_after": after_bytes, "replay_index_bytes_after": sum(v.nbytes for v in stored.values()),
                     "train_rows": len(ty), "new_training_rows": len(cy),
                     "train_class_counts": {str(c): int(np.sum(ty == c)) for c in seen},
                     "peak_cuda_allocated_bytes": peak_alloc, "peak_cuda_reserved_bytes": peak_reserved}
            checkpoint_save(cp_path, state)
            del cx, cy, tx, ty
            print(f"TASK_TRAINED task={t} seconds={elapsed:.3f} selection_seconds={selection_seconds:.3f}", flush=True)
        cp_hash = digest(cp_path)
        if metric_path.exists():
            record = json.loads(metric_path.read_text(encoding="utf-8"))
            if record["identity_hash"] != canonical_hash(identity) or record["checkpoint_sha256"] != cp_hash:
                raise ValueError("Checkpoint metrics differ")
        else:
            started = time.perf_counter()
            excluded = {"identity", "model_state", "buffer_x", "buffer_y", "buffer_indices"}
            record = {k: v for k, v in state.items() if k not in excluded}
            record.update(checkpoint_file=cp_path.relative_to(output).as_posix(), checkpoint_sha256=cp_hash,
                          identity_hash=canonical_hash(identity),
                          id=evaluate(model, data.x_test, data.y_test, seen, data, config, device), shift=None)
            record["evaluation_seconds"] = time.perf_counter() - started
            save_json(metric_path, record)
        records.append(record); old = seen
        print(f"TASK_COMPLETE task={t} macro_f1={record['id']['macro_f1']:.6f}", flush=True)
    result = {"schema_version": 1, "completed": True, "identity": identity,
              "class_order": data.class_order, "tasks": data.tasks,
              "train_counts": {str(c): int(np.sum(data.y_train == c)) for c in range(len(data.class_order))},
              "checkpoints": records, "training_seconds_total": cumulative,
              "selection_seconds_total": sum(c["buffer_selection_seconds"] for c in records),
              "evaluation_seconds_total": sum(c["evaluation_seconds"] for c in records),
              "setting_metadata": metadata, "policies": POLICIES,
              "command": identity["command"], "log_path": identity["log_path"]}
    save_json(final_path, result)
    print("CELL_COMPLETE", identity["dataset"], treatment, seed, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("NSL-KDD", "CIC-IDS-2017"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--treatment", choices=("D1", "D2", "D3"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--coverage-cache", type=Path)
    parser.add_argument("--d2-budget-mode", choices=("staged_archive", "fixed_final"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    seeds = (1, 2, 3) if args.dataset == "CIC-IDS-2017" else (1, 2, 3, 4, 42)
    if args.seed not in seeds:
        raise ValueError("Seed outside Section25 reporting scope")
    if (args.treatment == "D2") != (args.d2_budget_mode is not None):
        raise ValueError("D2 requires an explicit allocation/archive interpretation; other treatments do not accept it")
    config_path = ROOT / "ncd/configs/problem_scout_section21.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(min_steps_per_task=2000, epochs_per_task=8, batch_size=256, smoke=args.smoke)
    if args.smoke:
        config.update(min_steps_per_task=0, epochs_per_task=1)
    if args.device:
        config["device"] = args.device
    if args.data_root:
        config["problem_scout_data_root"] = path_label(args.data_root)
    torch.set_num_threads(config["cpu_threads"])
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_path = args.coverage_cache.resolve() if args.coverage_cache else output / "normal_coverage.json"
    with cell_lock(output / ".execution.lock"), (output / "run.log").open("a", encoding="utf-8") as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        data, _, _, metadata = prepare_setting(args.dataset, config)
        if args.smoke:
            data, _, _ = smoke_subset(data, None, None)
        command = subprocess.list2cmdline(["python", "-I", "ncd/run_benign_coverage.py", *sys.argv[1:]])
        for path in (ROOT, args.data_root, args.output_dir, args.coverage_cache):
            if path is not None and path.is_absolute():
                command = command.replace(str(path), path_label(path)).replace(path.as_posix(), path_label(path))
        identity = {"dataset": args.dataset, "method": "replay", "treatment": args.treatment,
                    "seed": args.seed, "smoke": args.smoke, "git_commit": source_commit(ROOT),
                    "d2_budget_mode": args.d2_budget_mode,
                    "effective_config": config, "config_sha256": digest(config_path),
                    "source_hashes": {p: digest(ROOT / p) for p in SOURCE_PATHS},
                    "class_order": data.class_order, "tasks": data.tasks,
                    "data_binding": binding(data, None, None), "kmeans_parameters": KMEANS,
                    "policies": POLICIES, "evaluation_role": "official_ID_test_never_used_for_selection",
                    "runtime_versions": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                                         "numpy": np.__version__, "sklearn": sklearn.__version__, "cuda": torch.version.cuda},
                    "command": command, "log_path": path_label(output / "run.log")}
        run_cell(data, metadata, identity, output, config, device, cache_path)


if __name__ == "__main__":
    main()
