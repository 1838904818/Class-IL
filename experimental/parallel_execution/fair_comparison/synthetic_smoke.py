"""Generate 18 synthetic rows, run the actual CPU trainer, and audit it."""
import argparse
import hashlib
from pathlib import Path

import l1_runner as runner
import numpy as np


def make_inputs(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    config = {"seed": 17, "epochs": 2, "batch_size": 6, "eval_batch_size": 6, "rank": 2,
              "lora_alpha": 4.0, "learning_rate": 0.01, "weight_decay": 0.0, "negative_ratio": 1,
              "minority_threshold": 5, "focal_alpha": 0.75, "focal_gamma": 2.0, "exemplar_capacity": 2,
              "exemplar_selection": "uniform_without_replacement", "checkpoint_policy": "fixed_last_epoch",
              "profile_label": "prospective_pilot_defaults_not_optimized_not_preregistered"}
    def reference(name):
        return {"path": name, "sha256": runner.sha(root / name)}
    generator = np.random.default_rng(11)
    splits = {}
    for split, repeats in (("train", 4), ("test", 2)):
        y = np.repeat(np.arange(3, dtype=np.int64), repeats)
        x = (generator.normal(size=(len(y), 4)) + np.eye(4)[y] * 2).astype(np.float32)
        ids = np.array([hashlib.sha256(f"synthetic-{split}-{i}".encode()).hexdigest().encode() for i in range(len(y))], dtype="S64")
        values = {"embeddings": x, "labels": y, "row_ids": ids,
                  "available_tasks": (y == 2).astype(np.int64), "group_ids": ids.copy()}
        splits[split] = {}
        for key, value in values.items():
            name = f"{split}-{key}.npy"
            np.save(root / name, value, allow_pickle=False)
            splits[split][key] = reference(name)
    # A synthetic upstream stage receipt, not evidence about a trained real encoder.
    np.savez(root / "synthetic-encoder.npz", weight=np.eye(4, dtype=np.float32))
    encoder_hash = runner.tensor_state_hash({"weight": runner.torch.from_numpy(np.eye(4, dtype=np.float32))})
    runner.atomic_json(root / "preprocessing.json", {"synthetic": True, "transform": "identity"})
    runner.atomic_json(root / "pretrain-cost.json", {"status": "HISTORICAL_UNMEASURED_DISCLOSED", "evidence_kind": "synthetic",
                       "encoder_state_sha256": encoder_hash, "cost_scope": "shared_encoder_pretraining_separate_from_L1", "measurements": None})
    receipt = {"schema_version": 1, "status": "COMPLETE", "evidence_kind": "synthetic", "encoder_state_sha256": encoder_hash,
               "checkpoint_sha256": runner.sha(root / "synthetic-encoder.npz"), "preprocessing_sha256": runner.sha(root / "preprocessing.json"),
               "split_array_sha256": {s: {k: ref["sha256"] for k, ref in refs.items()} for s, refs in splits.items()},
               "export_environment": {"producer": "synthetic_smoke", "numpy": np.__version__},
               "cost_scope": "separate_embedding_materialization_stage"}
    runner.atomic_json(root / "export-receipt.json", receipt)
    manifest = {"schema_version": 1, "dataset": "synthetic_18_rows", "evidence_kind": "synthetic", "tasks": [[0, 1], [2]],
                "group_mode": "row_identity_proxy", "encoder_binding": {"checkpoint": reference("synthetic-encoder.npz"),
                 "preprocessing": reference("preprocessing.json"), "export_receipt": reference("export-receipt.json"),
                 "pretrain_cost_receipt": reference("pretrain-cost.json"), "state_sha256": encoder_hash,
                 "trained_tasks": [0], "embedding_dimension": 4}, "splits": splits, "config": config, "allow_aggregate_wandb": False}
    runner.atomic_json(root / "manifest.json", manifest)
    return root / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = make_inputs(args.output / "inputs")
    result = runner.run(manifest, args.output / "run", device_name="cpu")
    print(runner.json_bytes({"status": result["status"], "synthetic_rows": 18,
                            "audit": runner.audit(args.output / "run", manifest)}).decode())


if __name__ == "__main__":
    main()
