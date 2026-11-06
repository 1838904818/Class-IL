"""Run one Section 448 AR-Stateless screen cell or one bounded software smoke."""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("AR-Stateless", "V1a", "V1b", "V2", "V3"), required=True)
    p.add_argument("--dataset", choices=("UNSW-NB15", "CIC-IDS-2017"), required=True)
    p.add_argument("--seed", choices=(1, 2), type=int, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--threads", choices=(1, 2, 4), type=int, default=4)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    os.environ["MKL_NUM_THREADS"] = str(a.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(a.threads)
    import numpy as np
    import sklearn
    import torch
    from ncd.prepare_problem_scout_data import prepare_setting
    from ncd.run_problem_scout import binding, save_json, smoke_subset
    from ncd.section138_cell import run_cell
    from ncd.section156_config import configurations
    from ncd.section156_models import build_models, train_gate
    from ncd.source_commit import source_commit

    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    config = json.loads((ROOT / "ncd/configs/problem_scout_section21.json").read_text(encoding="utf-8"))
    config.update(problem_scout_data_root=str(a.data_root.resolve(strict=True)),
                  device="cpu", cpu_threads=a.threads, smoke=a.smoke, lambda_g=0.,
                  epochs_per_task=1 if a.smoke else 8,
                  min_steps_per_task=0 if a.smoke else 2000,
                  batch_size=256, eval_batch_size=4096, learning_rate=.001,
                  derpp_alpha=1., derpp_beta=1., deadline_iso=None,
                  normal_training_rows=50000, ema_decay=.999, ema_consistency=1.,
                  hidden=128,
                  section156={"gate": "histgb", "family": "histgb", "control": True,
                              "normal_training_rows": 50000, "attack_memory_per_class": 200,
                              "gate_retention": False, "gate_schedule": "section123_es",
                              "last_layer_cbf_epochs": 0, "store_logits": False,
                              "weight_alignment": False, "benign_hard": False})
    data, _, _, _ = prepare_setting(a.dataset, config)
    if a.smoke:
        data, _, _ = smoke_subset(data, None, None, limit=64)

    cfg = next(c for c in configurations() if c.identifier == "C1")
    output = a.output_root.resolve() / ("smoke" if a.smoke else "screen") / a.dataset.lower() / a.arm / f"seed{a.seed}"
    output.mkdir(parents=True, exist_ok=False)
    source_paths = ("ncd/run_section448_screen.py", "ncd/section138_cell.py",
                    "ncd/section448_variants.py", "ncd/section156_models.py",
                    "ncd/section156_config.py", "ncd/section81_exposure.py",
                    "ncd/run_aacil.py", "ncd/prepare_problem_scout_data.py",
                    "ncd/run_problem_scout.py", "ncd/default_decision.py")
    identity = dict(
        git_commit=source_commit(ROOT), dataset=a.dataset, seed=a.seed,
        method=a.arm, reference_configuration="C1" if a.arm == "AR-Stateless" else "C1+Section448",
        effective_config={**config, "problem_scout_data_root": "<DATA_ROOT>"},
        engineering_defaults={"kmeans_n_init": "auto",
                              "kmeans_n_init_scope": "engineering_default_not_method_hyperparameter"},
        data_binding=binding(data, None, None),
        source_sha256={name: sha256(ROOT / name) for name in source_paths},
        command=[Path(sys.argv[0]).name, "--arm", a.arm, "--dataset", a.dataset,
                 "--seed", str(a.seed), "--threads", str(a.threads),
                 "--data-root", "<DATA_ROOT>", "--output-root", "<OUTPUT_ROOT>"]
                + (["--smoke"] if a.smoke else []),
        output_role="software_smoke" if a.smoke else "section448_development_screen",
        smoke=a.smoke,
        protocol={"normal_training_rows": 50000, "threshold_source": "10% Normal TRAIN holdout; full TRAIN prior",
                  "memory_rule": "attack-only bounded memory; stateless HistGB retrained each task",
                  "variant_quota": "aa.attack_targets('aacil', seen, normal); capacity includes unchanged train/validation slots",
                  "V1_cap": "new-class optimization rows capped at class capacity for V1a and 4x capacity for V1b; validation slots unchanged",
                  "V2_kmeans": "k equals the class optimization-memory slot quota; old classes recluster retained rows only",
                  "scope": "development-only screening; does not modify registered S1 or locked RQ1/RQ3"})
    def trainer(model, x, y, *, schedule, **kwargs):
        return train_gate(model, x, y, kind="histgb", **kwargs)

    start = time.perf_counter()
    result = run_cell(data, config, identity, output / "cell", cfg,
                      model_builder=build_models, gate_trainer=trainer,
                      variant=None if a.arm == "AR-Stateless" else a.arm)
    runtime = {"elapsed_seconds": time.perf_counter() - start,
               "platform": platform.platform(), "python": platform.python_version(),
               "pytorch": torch.__version__, "numpy": np.__version__,
               "scikit_learn": sklearn.__version__, "cpu_threads": a.threads}
    result["runtime"] = runtime
    save_json(output / "cell" / "result.json", result)
    save_json(output / "summary.json", dict(completed=True, scientific_result=not a.smoke,
              source_commit=identity["git_commit"], dataset=a.dataset, seed=a.seed,
              method=a.arm, checkpoints=len(result["checkpoints"]), runtime=runtime))
    print(json.dumps({"completed": True, "scientific_result": not a.smoke,
                      "dataset": a.dataset, "seed": a.seed, "arm": a.arm,
                      "elapsed_seconds": runtime["elapsed_seconds"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
