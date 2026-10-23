"""One explicit section138 MLP cell; confirmation output is sealed."""
import argparse
import atexit
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"


def main():
    from ncd.section138_config import configurations
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configuration", choices=[c.identifier for c in configurations()], required=True)
    p.add_argument("--dataset", choices=("UNSW-NB15", "CIC-IDS-2017"), required=True)
    p.add_argument("--seed", type=int, choices=(1, 2, 10, 11, 12, 13, 14), required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true", help="Resume only an identity-matched existing cell")
    p.add_argument("--claim-root", type=Path, help="Existing central shared-filesystem claim directory")
    p.add_argument("--lane", default="local")
    a = p.parse_args()
    if not a.smoke:
        local_cell = a.dataset == "CIC-IDS-2017" and a.seed in (1, 2)
        if a.lane not in ("local", "dicc") or local_cell != (a.lane == "local"):
            p.error("Cell is outside the fixed section143 lane partition")
        if a.claim_root is not None:
            p.error("Section143 uses disjoint lanes, not cross-machine claims")
    if a.resume and a.claim_root is not None:
        p.error("Claimed-cell recovery requires owner reconciliation; automatic reclaim is disabled")
    import torch
    from ncd.prepare_problem_scout_data import prepare_setting
    from ncd.run_problem_scout import binding, save_json, smoke_subset
    from ncd.section138_cell import run_cell
    from ncd.source_commit import source_commit
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cfg = next(c for c in configurations() if c.identifier == a.configuration)
    config = json.loads((ROOT / "ncd/configs/problem_scout_section21.json").read_text())
    config.update(problem_scout_data_root=str(a.data_root.resolve(strict=True)),
        device="cpu", cpu_threads=4, smoke=a.smoke, lambda_g=0.,
        epochs_per_task=1 if a.smoke else 8, min_steps_per_task=0 if a.smoke else 2000,
        batch_size=256, eval_batch_size=4096, learning_rate=.001,
        derpp_alpha=1., derpp_beta=1., deadline_iso=None, normal_training_rows=50000,
        ema_decay=.999, ema_consistency=1., hidden=128, section138=cfg.switches())
    role = "screen" if a.seed in (1, 2) else "s138_sealed"
    output = a.output_root.resolve() / role / a.dataset.lower() / cfg.identifier / f"seed{a.seed}"
    commit = source_commit(ROOT)
    owner = None
    cell_key = f"{cfg.identifier}-{a.dataset}-seed{a.seed}-{'smoke' if a.smoke else 'formal'}"
    if a.claim_root is not None:
        from ncd.section138_claim import claim
        owner = claim(a.claim_root, cell_key, lane=a.lane, source_commit=commit)
        if owner is None:
            print(json.dumps(dict(skipped=True, reason="Cell already claimed", cell=cell_key)))
            return
    if a.resume:
        if not output.is_dir():
            raise ValueError("Resume requires an existing cell directory")
    else:
        output.mkdir(parents=True, exist_ok=False)
    # Exclusive on this filesystem. A killed worker leaves the marker in place;
    # recovery must verify it is no longer running before removing that marker.
    # This does not coordinate independent local and HPC filesystems.
    lock = output / "worker.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump({"pid": os.getpid(), "started": time.time()}, handle)
    atexit.register(lock.unlink)
    started = time.perf_counter()
    data, _, _, _ = prepare_setting(a.dataset, config)
    if a.smoke:
        data, _, _ = smoke_subset(data, None, None, limit=64)
    identity = dict(git_commit=commit, dataset=a.dataset, seed=a.seed,
                    method=cfg.identifier, effective_config=config, data_binding=binding(data, None, None),
                    command=sys.argv, output_role=role, smoke=a.smoke)
    if a.resume:
        final = output / "cell/result.json"
        checkpoints = sorted((output / "cell/checkpoints").glob("checkpoint_*.pt"))
        if final.exists():
            previous = json.loads(final.read_text())["identity"]
        elif checkpoints:
            previous = torch.load(checkpoints[0], map_location="cpu", weights_only=True)["identity"]
        else:
            raise ValueError("No checkpoint or result to resume; preserve this failed attempt")
        # Invocation changes on resume; scientific inputs and source must not.
        if {k: v for k, v in previous.items() if k != "command"} != {
                k: v for k, v in identity.items() if k != "command"}:
            raise ValueError("Resume source, data or configuration differs")
        identity = previous
    result = run_cell(data, config, identity, output / "cell", cfg)
    summary = output / "summary.json"
    if a.resume and summary.exists():
        return
    save_json(summary, dict(completed=True, scientific_result=not a.smoke,
        source_commit=identity["git_commit"], dataset=a.dataset, seed=a.seed, method=cfg.identifier,
        output_role=role, checkpoints=len(result["checkpoints"]), elapsed_seconds=time.perf_counter()-started))
    if owner is not None:
        from ncd.run_problem_scout import digest
        from ncd.section138_claim import finish
        finish(a.claim_root, cell_key, owner,
               result_sha256=digest(output / "cell/result.json"), exit_code=0)


if __name__ == "__main__":
    main()
