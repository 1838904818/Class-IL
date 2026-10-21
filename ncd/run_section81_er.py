"""B1 ER executable integration. Formal execution awaits the resource supervisor.

Only the explicit CPU smoke is currently enabled; this is not a GPU launcher.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("UNSW-NB15", "CIC-IDS-2017"), required=True)
    parser.add_argument("--memory-setting", choices=("fixed200", "percent1"), required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2, 3, 4, 42), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dependency-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--recover-smoke-from", type=Path,
                        help="Diagnosed failed CPU smoke: reuse Task 0, preserve original output")
    args = parser.parse_args()
    if not args.smoke:
        parser.error("Formal B1 execution is disabled until the CUDA/resource supervisor is integrated")
    # Isolated pinned dependencies, without modifying the research environment.
    dependency_root = args.dependency_root.resolve(strict=True)
    sys.path.insert(1, str(dependency_root))
    import numpy as np
    import torch
    from ncd import run_aacil as aa
    from ncd.prepare_problem_scout_data import prepare_setting
    from ncd.run_aacil_retention import SOURCE_PATHS
    from ncd.run_problem_scout import binding, canonical_hash, digest, save_json, smoke_subset
    from ncd.section81_er_cell import run_er_cell
    from ncd.section81_exposure import ERExposure
    from ncd.source_commit import source_commit
    from ncd.train_cil import Tee

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    config_path = ROOT / "ncd/configs/problem_scout_section21.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(problem_scout_data_root=str(args.data_root.resolve(strict=True)),
                  device="cpu", cpu_threads=2, cpu_interop_threads=1,
                  section81=True, smoke=True, epochs_per_task=1,
                  min_steps_per_task=0, deadline_iso=None,
                  batch_size=256, eval_batch_size=256,
                  optimizer="AdamW", learning_rate=5e-4, weight_decay=1e-5,
                  memory_setting=args.memory_setting,
                  stopping_policy="TRAIN validation CE; best checkpoint; patience10; plateau factor0.1 patience5",
                  formal_epoch_ceiling=100, normal_training_rows=50000)
    sources = sorted(set(SOURCE_PATHS) | {
        "ncd/run_replay_bs.py", "ncd/run_section81_er.py",
        "ncd/section81_er_cell.py", "ncd/section81_er_training.py",
        "ncd/section81_exposure.py", "ncd/section81_memory.py",
        "ncd/section81_tabtransformer.py",
    })
    provenance = {"git_commit": source_commit(ROOT), "effective_config": config,
                  "config_sha256": digest(config_path),
                  "source_hashes": {p: digest(ROOT / p) for p in sources},
                  "runtime_versions": {"python": sys.version.split()[0],
                                       "torch": str(torch.__version__), "numpy": np.__version__}}
    output = args.output_dir.resolve()
    recovery = None
    if args.recover_smoke_from:
        prior_root = args.recover_smoke_from.resolve(strict=True)
        prior_attempt = json.loads((prior_root / "attempt.json").read_text(encoding="utf-8"))
        if (prior_attempt["status"] != "failed" or not prior_attempt["smoke"]
                or not any(message in prior_attempt.get("error", "") for message in (
                    "'dict' object has no attribute 'sum'",
                    "Normal float32 log-prior must be strictly largest"))
                or prior_attempt["dataset"] != args.dataset or prior_attempt["seed"] != args.seed
                or prior_attempt["memory_setting"] != args.memory_setting):
            raise ValueError("Not the diagnosed B1 smoke calibration failure")
        saved = list((prior_root / "cell/checkpoints").glob("*"))
        if len(saved) != 1 or saved[0].name != "checkpoint_000.pt":
            raise ValueError("Recovery requires only the original Task 0 checkpoint")
        prior_state = torch.load(saved[0], map_location="cpu", weights_only=True)
        recovery = {"checkpoint": saved[0], "checkpoint_sha256": digest(saved[0]),
                    "identity": prior_state["identity"]}
        del prior_state
    # No implicit retry or overwrite. Failed attempt output remains for diagnosis.
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    attempt = {"status": "running", "started_at": aa.now(), "smoke": True,
               "scientific_result": False, "pid": os.getpid(), "provenance": provenance,
               "dataset": args.dataset, "seed": args.seed,
               "memory_setting": args.memory_setting, "command": sys.argv}
    if recovery:
        attempt["prior_attempt"] = prior_attempt
        attempt["reused_checkpoint_sha256"] = recovery["checkpoint_sha256"]
    save_json(output / "attempt.json", attempt)
    with (output / "run.log").open("x", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                data, _, _, metadata = prepare_setting(args.dataset, config)
                # The capped smoke sample is class-balanced and cannot supply the
                # population prior required by the unchanged monotone calibrator.
                population_counts = {c: int(np.count_nonzero(data.y_train == c))
                                     for c in range(len(data.class_order))}
                data, _, _ = smoke_subset(data, None, None, limit=64)
                exposure = ERExposure(data, args.seed)
                identity = provenance | {
                    "section": 81, "method": "er_bal_tabt", "dataset": args.dataset,
                    "seed": args.seed, "memory_setting": args.memory_setting,
                    "class_order": data.class_order, "tasks": data.tasks,
                    "feature_names": data.feature_names, "data_binding": binding(data, None, None),
                    "data_metadata_sha256": canonical_hash(metadata),
                    "exposure_policy": exposure.description(),
                    "calibration_ids_sha256": hashlib.sha256(exposure.calibration_ids.tobytes()).hexdigest(),
                    "normal_stopping_ids_sha256": hashlib.sha256(exposure.normal_stopping_ids.tobytes()).hexdigest(),
                    "admission_training_counts": {str(c): n for c, n in exposure.admission_training_counts.items()},
                    "calibration_prior_counts": {str(c): n for c, n in population_counts.items()},
                    "calibration_prior_scope": "Full TRAIN labels before non-scientific smoke subsampling; no TEST",
                    "scientific_result": False,
                }
                result = run_er_cell(
                    data, config, identity, output / "cell", torch.device("cpu"),
                    calibration_ids=exposure.calibration_ids,
                    original_counts=population_counts,
                    admission_training_counts=exposure.admission_training_counts,
                    current_exposure=exposure, smoke_recovery=recovery)
                attempt.update(status="complete", checkpoints=len(result["checkpoints"]),
                               result_sha256=digest(output / "cell/result.json"))
            except BaseException:
                attempt.update(status="failed", error=traceback.format_exc())
                traceback.print_exc()
                raise
            finally:
                attempt.update(ended_at=aa.now(), elapsed_seconds=time.perf_counter() - started)
                save_json(output / "attempt.json", attempt)


if __name__ == "__main__":
    main()
