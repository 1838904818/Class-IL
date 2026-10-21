"""Independent B1 fixed200 AR-TabT entry; CPU smoke until supervision is wired."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr

os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("UNSW-NB15", "CIC-IDS-2017"), required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2, 3, 4, 42), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dependency-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.smoke:
        parser.error("Formal AR requires its independent resource supervisor")
    sys.path.insert(1, str(args.dependency_root.resolve(strict=True)))
    import numpy as np
    import torch
    from ncd import run_aacil as aa
    from ncd import run_aacil_compact as original
    from ncd.prepare_problem_scout_data import prepare_setting
    from ncd.run_problem_scout import (binding, canonical_hash, digest, save_json,
                                       section26_pools, smoke_subset)
    from ncd.section81_ar_cell import run_ar_cell
    from ncd.section81_tabtransformer import fit_feature_spec
    from ncd.source_commit import source_commit
    from ncd.train_cil import Tee

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    config_path = ROOT / "ncd/configs/problem_scout_section21.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(problem_scout_data_root=str(args.data_root.resolve(strict=True)),
                  device="cpu", cpu_threads=2, cpu_interop_threads=1, section81=True,
                  section27=True, section27_schema_version=1, section27_access_policy=aa.POLICY,
                  section31=True, section31_schema_version=1, smoke=True,
                  epochs_per_task=1, min_steps_per_task=0, batch_size=256,
                  learning_rate=0.001, lambda_g=0.0, derpp_alpha=1.0, derpp_beta=1.0,
                  derpp_allocation_adaptation=original.ADAPTATION,
                  derpp_coefficient_source="Section81 fixed original AR; no tuning",
                  ema_decay=0.999, ema_consistency=1.0, cbf_epochs=1, cbf_lr_multiplier=0.1,
                  deadline_iso=None, gate_step_policy="matched_s2000",
                  calibration_policy="permanent_normal_holdout_outside_m",
                  normal_training_rows=50000, selection_only=False,
                  selection_validation_policy="Retained old-class validation only; no selection",
                  memory_setting="fixed200",
                  family_parameter_constraint="Fixed TabTransformer; no original MLP size bound",
                  future_output_policy="Full attack output head; unseen rows masked")
    sources = sorted(set(original.SOURCE_PATHS) | {
        "ncd/run_section81_ar.py", "ncd/section81_ar_cell.py", "ncd/section81_tabtransformer.py"})
    provenance = {"git_commit": source_commit(ROOT), "effective_config": config,
                  "config_sha256": digest(config_path),
                  "source_hashes": {p: digest(ROOT / p) for p in sources},
                  "runtime_versions": {"python": sys.version.split()[0],
                                       "torch": str(torch.__version__), "numpy": np.__version__}}
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    attempt = {"status": "running", "started_at": aa.now(), "pid": os.getpid(),
               "smoke": True, "scientific_result": False, "dataset": args.dataset,
               "seed": args.seed, "method": "ar_tabt", "memory_setting": "fixed200",
               "command": sys.argv, "provenance": provenance}
    save_json(output / "attempt.json", attempt)
    with (output / "run.log").open("a", encoding="utf-8") as log, \
            redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        try:
            data, _, _, metadata = prepare_setting(args.dataset, config)
            data, _, _ = smoke_subset(data, None, None)
            train, valid = section26_pools(data, args.seed)
            normal = data.class_order.index("Normal")
            rows = [aa.select_rows(train[normal], 50000, args.seed * 10000 * 10007 + normal)]
            rows.extend(train[c] for c in data.tasks[0] if c != normal)
            spec = fit_feature_spec(args.dataset, data.feature_names, data.x_train[np.concatenate(rows)])
            identity = provenance | {"section": 81, "dataset": args.dataset,
                "method": "ar_tabt", "seed": args.seed, "memory_setting": "fixed200",
                "command": sys.argv, "class_order": data.class_order, "tasks": data.tasks,
                "feature_names": data.feature_names, "feature_spec": spec,
                "data_binding": binding(data, None, None), "data_metadata_sha256": canonical_hash(metadata),
                "section27_access_policy": aa.POLICY, "effective_lambda_g": 0.0,
                "section26_holdout": {str(c): {"train_count": len(train[c]),
                    "validation_count": len(valid[c]),
                    "validation_indices_sha256": hashlib.sha256(valid[c].tobytes()).hexdigest()} for c in valid}}
            result = run_ar_cell(data, config, identity, output / "cell", torch.device("cpu"))
            attempt.update(status="complete", checkpoint_count=len(result["checkpoints"]),
                           result_sha256=digest(output / "cell/result.json"))
        except BaseException as exc:
            attempt.update(status="failed", error=str(exc), traceback=traceback.format_exc())
            raise
        finally:
            attempt.update(ended_at=aa.now(), elapsed_seconds=time.perf_counter() - start)
            save_json(output / "attempt.json", attempt)


if __name__ == "__main__":
    main()
