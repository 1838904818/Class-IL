"""Independent, measured embedding-export stage for a bound Task-0 OFRA encoder."""
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import shutil
import sys
import time

import l1_runner as r
import numpy as np
import torch
from torch import nn

PINNED_RUNTIME = {
    "streaming_full/models.py": "455c516b3da1fdc373263698ba5484050b4edde349b02ae2e29effae6e0d21c2",
    "ofra_encoders/__init__.py": "c66e35b2752cc4b8959ef97fc16ee0a4c26f45a3200c0e985940ad0ca70fd086",
    "ofra_encoders/ft_transformer.py": "3ed138ae19ac0874d5ea037b91c044a1b1355d65c27a9cec8f50ef095bc98fec",
}


def verify_producer_completion(manifest_path):
    """Marked sampling exports require both immutable arm and pair completion."""
    path = Path(manifest_path).absolute()
    root = path.parent
    parent_start = root.parent / "STARTED.json"
    pair_marked = parent_start.exists() and r.read_json(parent_start).get("stage") == "offline_O_P_export_pair"
    marked = pair_marked or any((root / name).exists() for name in ("STARTED.json", "LINEAGE.json"))
    if not marked:
        return None
    def no_links(entry):
        # Include Windows directory junctions, not just POSIX symbolic links.
        for item in (entry, *entry.parents):
            r.require(not item.is_symlink() and not (getattr(item.lstat(), "st_file_attributes", 0) & 0x400),
                      "linked producer artifact")
    no_links(path)
    r.require(path.name == "export-input.json" and root.name in ("P", "O"), "unexpected producer input location")
    complete = r.read_json(root / "COMPLETE.json")
    r.exact(complete, {"status", "evidence_kind", "manifest_sha256", "files"}, "producer completion")
    r.require(complete["status"] == "RAW_EXPORT_INPUT_COMPLETE" and isinstance(complete["files"], dict),
              "raw export input incomplete")
    actual = set()
    for entry in root.rglob("*"):
        no_links(entry)
        if entry.is_file() and entry.name != "COMPLETE.json":
            actual.add(entry.relative_to(root).as_posix())
        elif entry.is_file() and entry.parent != root:
            actual.add(entry.relative_to(root).as_posix())
    r.require(actual == set(complete["files"]), "producer file closure changed")
    for name, digest in complete["files"].items():
        r.relative_file(root, {"path": name, "sha256": digest})
    r.require(complete["manifest_sha256"] == complete["files"].get(path.name), "producer manifest binding changed")
    parent_complete = root.parent / "COMPLETE.json"
    no_links(parent_complete)
    parent = r.read_json(parent_complete)
    r.require(parent["status"] == "RAW_EXPORT_PAIR_COMPLETE" and parent["evidence_kind"] == complete["evidence_kind"]
              and parent[root.name + "_export_manifest_sha256"] == complete["manifest_sha256"],
              "raw export pair incomplete or mismatched")
    return complete


class MLPEncoder(nn.Module):
    def __init__(self, features, width, layers):
        super().__init__()
        modules = []
        for _ in range(layers):
            modules.extend([nn.Linear(features, width), nn.ReLU()])
            features = width
        self.feat = nn.Sequential(*modules)

    def forward(self, value):
        return self.feat(value)


def verify_runtime_sources(runtime_root):
    r.require(runtime_root is not None, "FT export requires the pinned runtime tree")
    root = Path(runtime_root).resolve()
    for path, expected in PINNED_RUNTIME.items():
        r.require(r.sha(root / path) == expected, "OFRA runtime source drift")
    return root


def build_bound_encoder(metadata, runtime_root=None):
    architecture = metadata["architecture"]
    for key in ("feature_dim",):
        r.integer(metadata[key], 1)
    for key in ("d_model", "n_layers"):
        r.integer(architecture[key], 1)
    if architecture["encoder_type"] == "mlp":
        return MLPEncoder(metadata["feature_dim"], architecture["d_model"], architecture["n_layers"])
    r.require(architecture["encoder_type"] == "ft_transformer" and runtime_root is not None,
              "FT export requires the pinned runtime tree")
    root = verify_runtime_sources(runtime_root)
    # The already hash-verified adapter also verifies upstream FT version/source.
    old_flag = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    try:
        if "ofra_encoders" in sys.modules:
            r.require(Path(sys.modules["ofra_encoders"].__file__).resolve() == root / "ofra_encoders/__init__.py",
                      "conflicting imported encoder module")
        spec = importlib.util.spec_from_file_location("l1_pinned_ofra_models", root / "streaming_full/models.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.build_encoder(encoder_type="ft_transformer", n_features=metadata["feature_dim"],
                                    **{key: architecture[key] for key in ("d_model", "n_layers", "ft_heads", "ft_dim_head",
                                       "ft_attn_dropout", "ft_ff_dropout", "ft_num_residual_streams")})
    finally:
        sys.path.pop(0)
        sys.dont_write_bytecode = old_flag


def export(manifest_path, output, *, runtime_root=None, device_name="cpu", batch_size=512):
    r.integer(batch_size, 1)
    started = time.perf_counter()
    code_hashes = {"exporter": r.sha(__file__), "shared_helpers": r.sha(r.__file__)}
    producer = verify_producer_completion(manifest_path)
    source_path = Path(manifest_path).resolve()
    source_hash = r.sha(source_path)
    root = source_path.parent
    spec = r.read_json(source_path)
    r.exact(spec, {"schema_version", "dataset", "evidence_kind", "tasks", "group_mode", "config", "allow_aggregate_wandb",
                   "checkpoint_metadata", "checkpoint_state", "pretrain_cost_receipt", "raw_splits"}, "export manifest")
    r.require(type(spec["schema_version"]) is int and spec["schema_version"] == 1, "export schema")
    r.config_check(spec["config"])
    r.safe_name(spec["dataset"])
    r.require(spec["evidence_kind"] in ("synthetic", "real"), "evidence kind")
    r.require(producer is None or producer["evidence_kind"] == spec["evidence_kind"], "producer evidence kind mismatch")
    metadata = r.read_json(r.relative_file(root, spec["checkpoint_metadata"]))
    r.require(metadata.get("dataset", spec["dataset"]) == spec["dataset"], "checkpoint dataset mismatch")
    state_path = r.relative_file(root, spec["checkpoint_state"])
    r.require(metadata["checkpoint"] == 0 and metadata["seen_classes"] == spec["tasks"][0], "only Task-0 encoder checkpoint is accepted")
    r.require(metadata["inference_state_sha256"] == spec["checkpoint_state"]["sha256"], "checkpoint state mismatch")
    if "canonical_sha256" in metadata:
        without = {k: v for k, v in metadata.items() if k != "canonical_sha256"}
        r.require(r.object_hash(without) == metadata["canonical_sha256"], "checkpoint metadata canonical hash mismatch")
    device = torch.device(device_name)
    r.require(device.type in ("cpu", "cuda") and (device.type != "cuda" or torch.cuda.is_available()), "export device unavailable")
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        r.require(r.os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"), "deterministic CUBLAS config required")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    encoder = build_bound_encoder(metadata, runtime_root).to(device)
    with np.load(state_path, allow_pickle=False) as archive:
        state = {name: torch.from_numpy(np.array(archive[key], copy=True))
                 for name, key in metadata["state_schema"]["encoder"].items()}
        r.strict_model_load(encoder, state)
        normal = metadata["state_schema"]["normalization"]
        mean = np.array(archive[normal["mean"]], copy=True)
        scale = np.array(archive[normal["scale"]], copy=True)
        r.require(mean.dtype == scale.dtype == np.float64, "normalization must be native float64 before conversion")
    state_hash = r.tensor_state_hash(encoder.state_dict())
    r.require(mean.shape == scale.shape == (metadata["feature_dim"],) and np.isfinite(mean).all()
              and np.isfinite(scale).all() and (scale > 0).all(), "invalid frozen normalization")
    del state
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # A failed stage has no COMPLETE receipt and is never reused or overwritten.
    r.atomic_json(output / "STAGE_STARTED.json", {"stage": "embedding_export", "input_sha256": r.sha(source_path)})
    def ref(name):
        return {"path": name, "sha256": r.sha(output / name)}
    try:
        np.savez(output / "encoder.npz", **{name: value.detach().cpu().numpy() for name, value in encoder.state_dict().items()})
        np.savez(output / "normalization.npz", mean=mean, scale=scale)
        if spec["pretrain_cost_receipt"] is None:
            pretrain = {"status": "HISTORICAL_UNMEASURED_DISCLOSED", "evidence_kind": spec["evidence_kind"],
                        "encoder_state_sha256": state_hash, "cost_scope": "shared_encoder_pretraining_separate_from_L1", "measurements": None}
        else:
            pretrain = r.read_json(r.relative_file(root, spec["pretrain_cost_receipt"]))
            r.require(pretrain["encoder_state_sha256"] == state_hash, "pretrain cost is bound to a different encoder")
        r.atomic_json(output / "pretrain-cost.json", pretrain)
        r.exact(spec["raw_splits"], {"train", "test"}, "raw splits")
        splits = {}
        profile = {"stage": "embedding_export", "evidence_kind": spec["evidence_kind"], "rows_forwarded": 0,
                   "batches_forwarded": 0, "encoder_forward_seconds": 0.0, "batch_size": batch_size,
                   "encoder_state_sha256_before": state_hash, "environment": r.environment(device),
                   "runtime_source_sha256": PINNED_RUNTIME if metadata["architecture"]["encoder_type"] == "ft_transformer" else {},
                   "exporter_sha256": r.sha(Path(__file__)), "input_manifest_sha256": r.sha(source_path)}
        profile["code_sha256"] = code_hashes
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        for split, descriptors in spec["raw_splits"].items():
            r.exact(descriptors, (r.SPLIT_KEYS - {"embeddings"}) | {"raw_features"}, "raw split descriptor")
            x = np.load(r.relative_file(root, descriptors["raw_features"]), mmap_mode="r", allow_pickle=False)
            r.require(x.ndim == 2 and x.shape[1] == metadata["feature_dim"] and x.dtype == np.float32 and len(x) > 0,
                      "raw feature shape/dtype")
            name = split + "-embeddings.npy"
            destination = np.lib.format.open_memmap(output / name, mode="w+", dtype=np.float32,
                                                   shape=(len(x), metadata["architecture"]["d_model"]))
            for begin in range(0, len(x), batch_size):
                raw = np.asarray(x[begin:begin + batch_size], dtype=np.float64)
                normalized = ((raw - mean) / scale).astype(np.float32)
                r.require(np.isfinite(normalized).all(), "nonfinite normalized feature")
                tensor = torch.from_numpy(normalized).to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                clock = time.perf_counter()
                with torch.no_grad():
                    embedded = encoder(tensor).cpu().numpy().astype(np.float32)
                r.require(np.isfinite(embedded).all(), "nonfinite embedding")
                profile["encoder_forward_seconds"] += time.perf_counter() - clock
                profile["rows_forwarded"] += len(embedded)
                profile["batches_forwarded"] += 1
                destination[begin:begin + len(embedded)] = embedded
            destination.flush()
            del destination
            splits[split] = {"embeddings": ref(name)}
            for key in r.SPLIT_KEYS - {"embeddings"}:
                source = r.relative_file(root, descriptors[key])
                name = split + "-" + key + ".npy"
                shutil.copyfile(source, output / name)
                splits[split][key] = ref(name)
        profile["encoder_state_sha256_after"] = r.tensor_state_hash(encoder.state_dict())
        r.require(profile["encoder_state_sha256_after"] == state_hash, "export mutated frozen encoder")
        profile.update(process_memory=r.process_memory(), wall_seconds=time.perf_counter() - started,
                       cuda_allocated_peak_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                       cuda_reserved_peak_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                       encoder_storage=r.storage_inventory({"encoder_parameters": list(encoder.parameters()), "encoder_buffers": list(encoder.buffers())}))
        r.atomic_json(output / "EXPORT_PROFILE.json", profile)
        receipt = {"schema_version": 1, "status": "COMPLETE", "evidence_kind": spec["evidence_kind"], "encoder_state_sha256": state_hash,
                   "checkpoint_sha256": ref("encoder.npz")["sha256"], "preprocessing_sha256": ref("normalization.npz")["sha256"],
                   "split_array_sha256": {s: {k: descriptor["sha256"] for k, descriptor in fields.items()} for s, fields in splits.items()},
                   "export_environment": {**r.environment(device), "profile_sha256": ref("EXPORT_PROFILE.json")["sha256"],
                                          "source_checkpoint_metadata_sha256": spec["checkpoint_metadata"]["sha256"],
                                          "source_checkpoint_state_sha256": spec["checkpoint_state"]["sha256"]},
                   "cost_scope": "separate_embedding_materialization_stage"}
        r.atomic_json(output / "export-receipt.json", receipt)
        manifest = {k: spec[k] for k in ("schema_version", "dataset", "evidence_kind", "tasks", "group_mode", "config", "allow_aggregate_wandb")}
        manifest.update(splits=splits, encoder_binding={"checkpoint": ref("encoder.npz"), "preprocessing": ref("normalization.npz"),
                        "export_receipt": ref("export-receipt.json"), "pretrain_cost_receipt": ref("pretrain-cost.json"),
                        "state_sha256": state_hash, "trained_tasks": [0], "embedding_dimension": metadata["architecture"]["d_model"]})
        r.atomic_json(output / "manifest.json", manifest)
        r.load_inputs(output / "manifest.json")
        r.require(r.sha(source_path) == source_hash, "export input manifest changed")
        for descriptor in (spec["checkpoint_metadata"], spec["checkpoint_state"],
                           *[d for split in spec["raw_splits"].values() for d in split.values()]):
            r.relative_file(root, descriptor)
        if spec["pretrain_cost_receipt"] is not None:
            r.relative_file(root, spec["pretrain_cost_receipt"])
        r.require(verify_producer_completion(source_path) == producer, "producer completion changed")
        r.require(code_hashes == {"exporter": r.sha(__file__), "shared_helpers": r.sha(r.__file__)}, "export code changed")
        if metadata["architecture"]["encoder_type"] == "ft_transformer":
            verify_runtime_sources(runtime_root)
        r.atomic_json(output / "COMPLETE.json", {"status": "COMPLETE", "evidence_kind": spec["evidence_kind"],
                      "manifest": ref("manifest.json"), "profile": ref("EXPORT_PROFILE.json")})
        return output / "manifest.json"
    except BaseException as exc:
        r.atomic_json(output / "FAILED.json", {"error_type": type(exc).__name__, "elapsed_seconds": time.perf_counter() - started,
                      "scope": "failed_embedding_stage_cost_no_complete_receipt", "process_memory": r.process_memory()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    try:
        export(args.manifest, args.output, runtime_root=args.runtime_root, device_name=args.device, batch_size=args.batch_size)
        print('{"status":"EMBEDDINGS_COMPLETE","training_performed":false}')
        return 0
    except (r.Invalid, ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        print(r.json_bytes({"status": "BLOCKED_OR_FAILED", "error_type": type(exc).__name__}).decode())
        return 2


if __name__ == "__main__":
    sys.exit(main())
