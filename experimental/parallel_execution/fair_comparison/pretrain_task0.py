"""Train one shared Task-0 encoder on the P transform-fit cohort only."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time

import l1_runner as r
import materialize_embeddings as export
import numpy as np
import torch
from torch import nn

CONFIG_FIELDS = {"seed", "epochs", "batch_size", "learning_rate", "weight_decay", "checkpoint_every_steps",
                 "architecture", "checkpoint_policy", "profile_label"}
ARRAYS = {"x", "labels", "row_ids", "group_ids", "available_tasks", "source_index"}


def no_links(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        r.require(not item.is_symlink() and not (getattr(item.lstat(), "st_file_attributes", 0) & 0x400), "linked input/output")
    return path


def config_check(config):
    r.exact(config, CONFIG_FIELDS, "Task-0 configuration")
    r.integer(config["seed"])
    for key in ("epochs", "batch_size", "checkpoint_every_steps"):
        r.integer(config[key], 1)
    r.number(config["learning_rate"], 1e-15)
    r.number(config["weight_decay"])
    r.require(config["checkpoint_policy"] == "fixed_last_epoch", "held-out checkpoint selection forbidden")
    r.require(config["profile_label"] == "prospective_pilot_defaults_not_optimized_not_preregistered", "pilot disclosure missing")
    architecture = config["architecture"]
    keys = {"encoder_type", "d_model", "n_layers"}
    if architecture.get("encoder_type") == "ft_transformer":
        keys |= {"ft_heads", "ft_dim_head", "ft_attn_dropout", "ft_ff_dropout", "ft_num_residual_streams"}
        for key in ("ft_heads", "ft_dim_head", "ft_num_residual_streams"):
            r.integer(architecture[key], 1)
        for key in ("ft_attn_dropout", "ft_ff_dropout"):
            r.number(architecture[key], 0, 1)
    r.exact(architecture, keys, "encoder architecture")
    r.require(architecture["encoder_type"] in ("mlp", "ft_transformer"), "unsupported encoder")
    for key in ("d_model", "n_layers"):
        r.integer(architecture[key], 1)


def close_arrays(arrays):
    for array in arrays.values():
        mapping = getattr(array, "_mmap", None)
        if mapping is not None and not mapping.closed:
            mapping.close()


def load_cohort(directory, expected_manifest_sha256):
    """Verify completion metadata and only transform-fit bytes; never open other partitions."""
    root = no_links(directory)
    r.check_sha(expected_manifest_sha256)
    path = no_links(root / "manifest.json")
    r.require(r.sha(path) == expected_manifest_sha256, "Task-0 manifest identity mismatch")
    marker = r.read_json(no_links(root / "COMPLETE.json"))
    r.exact(marker, {"schema_version", "status", "manifest_sha256", "files"}, "sampling completion")
    r.require(marker["schema_version"] == 1 and marker["status"] == "TRAIN_INCREMENT_COMPLETE"
              and marker["manifest_sha256"] == expected_manifest_sha256
              and marker["files"].get("manifest.json") == expected_manifest_sha256, "incomplete sampling stage")
    manifest = r.read_json(path)
    r.require(manifest["kind"] == "derived_train_increment" and manifest["status"] == "COMPLETE"
              and type(manifest["increment"]) is int and manifest["increment"] == 0
              and manifest["policy"]["arm"] == "P"
              and manifest["policy"]["transform_fit_policy"] == "common_task0_prospective_fit"
              and manifest["official_test_used_for_selection"] is False, "P Task-0 train-only cohort required")
    r.safe_name(manifest["dataset_id"])
    r.integer(manifest["feature_dim"], 1)
    r.require(manifest["feature_dtype"] == "float32", "no silent feature precision conversion")
    collection = manifest["collections"]["transform_fit"]
    r.exact(collection, {"rows", *ARRAYS}, "transform-fit collection")
    r.integer(collection["rows"], 2)
    r.require(r.object_hash(collection) == manifest["common_transform_fit_sha256"], "transform cohort hash mismatch")
    classes = [c["class_id"] for c in manifest["classes"]]
    r.require(classes == sorted(set(classes)) and len(classes) >= 2, "Task-0 class axis")
    for c in classes:
        r.integer(c)
    arrays = {}
    try:
        for key in sorted(ARRAYS):
            descriptor = collection[key]
            r.exact(descriptor, {"path", "sha256", "rows", "shape", "dtype"}, "cohort descriptor")
            r.require(descriptor["path"] == "transform_fit/" + key + ".npy"
                      and marker["files"].get(descriptor["path"]) == descriptor["sha256"], "cohort not bound to completion")
            array_path = no_links(root / descriptor["path"])
            r.relative_file(root, {k: descriptor[k] for k in ("path", "sha256")})
            array = np.load(array_path, mmap_mode="r", allow_pickle=False)
            arrays[key] = array
            shape = (collection["rows"], manifest["feature_dim"]) if key == "x" else (
                (collection["rows"], 2) if key == "source_index" else (collection["rows"],))
            dtype = "float32" if key == "x" else ("S64" if key in ("row_ids", "group_ids") else "int64")
            r.require(array.shape == shape and list(shape) == descriptor["shape"] and array.dtype == np.dtype(dtype)
                      and array.dtype.str == descriptor["dtype"] and descriptor["rows"] == collection["rows"], "cohort array contract mismatch")
        r.require(set(arrays["labels"].tolist()) == set(classes), "Task-0 labels/support mismatch")
        r.require((arrays["available_tasks"] == 0).all() and (arrays["source_index"] >= 0).all(), "future/invalid cohort provenance")
        for key in ("row_ids", "group_ids"):
            r.require(all(r.re.fullmatch(rb"[0-9a-f]{64}", bytes(v)) for v in arrays[key]), "noncanonical cohort identity")
        r.require(len(set(arrays["row_ids"].tolist())) == collection["rows"], "duplicate training identity")
        return manifest, arrays
    except BaseException:
        close_arrays(arrays)
        raise


def fit_normalization(x, batch_size, journal):
    count, mean, m2 = 0, np.zeros(x.shape[1], np.float64), np.zeros(x.shape[1], np.float64)
    started = time.perf_counter()
    for begin in range(0, len(x), batch_size):
        values = np.array(x[begin:begin + batch_size], dtype=np.float64)
        r.require(np.isfinite(values).all(), "nonfinite training feature")
        batch_mean = values.mean(0)
        batch_m2 = np.square(values - batch_mean).sum(0)
        delta, size = batch_mean - mean, len(values)
        total = count + size
        m2 += batch_m2 + delta * delta * (count * size / total)
        mean += delta * (size / total)
        count = total
        journal.emit({"event": "normalization_batch_completed", "rows": size})
    scale = np.sqrt(m2 / count)
    scale[scale == 0] = 1.0
    r.require(np.isfinite(mean).all() and np.isfinite(scale).all() and (scale > 0).all(), "invalid training normalization")
    return {"mean": torch.from_numpy(mean), "scale": torch.from_numpy(scale), "fit_rows": count,
            "fit_seconds": time.perf_counter() - started, "policy": "train_only_population_std_zero_variance_scale_one"}


def batch_indices(n, config, step):
    count = (n + config["batch_size"] - 1) // config["batch_size"]
    epoch, batch = divmod(step, count)
    order = np.random.default_rng(r.seed_for(config["seed"], "common_task0_shuffle", epoch)).permutation(n)
    begin = batch * config["batch_size"]
    return order[begin:begin + config["batch_size"]]


def pretrain_cost(output, state):
    ledger = r.cost_ledger(output, state["cursor"], state["commits"])
    ledger["normalization_row_presentations"] = 0
    for attempt in ledger["attempts"]:
        with (output / attempt["journal"]).open("rb") as stream:
            for line in stream:
                try:
                    event = r.json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    break
                if event.get("event") == "normalization_batch_completed":
                    ledger["normalization_row_presentations"] += event["rows"]
    return ledger


def run(directory, expected_manifest_sha256, config_path, output, *, evidence_kind, runtime_root=None,
        device_name="cpu", resume=False, max_steps=None, fault_after_steps=None, metric_send=None):
    import pretrain_metrics
    emitter = pretrain_metrics.AttemptEmitter(r.uuid.uuid4().hex, metric_send) if metric_send is not None else None
    started = time.perf_counter()
    r.require(evidence_kind in ("synthetic", "real"), "explicit evidence kind required")
    r.require(fault_after_steps is None or evidence_kind == "synthetic", "synthetic-only fault injection")
    if max_steps is not None:
        r.integer(max_steps, 1)
    config = r.read_json(config_path)
    config_check(config)
    manifest, arrays = load_cohort(directory, expected_manifest_sha256)
    output = Path(output).absolute()
    lock, journal, state = None, None, {"cursor": 0, "commits": []}
    try:
        r.require(Path(directory).resolve() not in output.resolve().parents, "output must be separate from immutable cohort")
        if not resume:
            output.mkdir(parents=True, exist_ok=False)
        no_links(output)
        lock_path = output / "WRITER.lock"
        lock = lock_path.open("xb")
        lock.write(r.json_bytes({"scope": "exclusive_task0_writer"}))
        lock.flush()
        r.os.fsync(lock.fileno())
        r.require(not (output / "COMPLETE.json").exists(), "completed stage cannot be overwritten")
        r.require(device_name in ("cpu", "cuda"), "device type")
        device = torch.device(device_name)
        r.require(device.type != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
        torch.use_deterministic_algorithms(True)
        if device.type == "cuda":
            r.require(r.os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"), "deterministic CUDA configuration missing")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        binding = {"manifest_sha256": expected_manifest_sha256, "config_sha256": r.sha(config_path),
                   "config_value_sha256": r.object_hash(config),
                   "implementation_sha256": r.sha(__file__), "shared_helpers_sha256": r.sha(r.__file__),
                   "encoder_adapter_sha256": r.sha(export.__file__), "evidence_kind": evidence_kind,
                   "environment": r.environment(device), "runtime_source_sha256": export.PINNED_RUNTIME
                   if config["architecture"]["encoder_type"] == "ft_transformer" else {}}
        if resume:
            r.require(r.read_json(output / "INPUT_BINDING.json") == binding, "resume input/code/environment drift")
            if (output / "LATEST.json").exists():
                latest = r.read_json(output / "LATEST.json")
                state = torch.load(r.relative_file(output, {k: latest[k] for k in ("path", "sha256")}),
                                   map_location="cpu", weights_only=True)
                r.require(state["cursor"] == latest["cursor"] and state["binding"] == binding, "checkpoint binding mismatch")
        else:
            r.atomic_json(output / "INPUT_BINDING.json", binding)
        journal = r.Journal(output)
        journal.emit({"event": "attempt_started", "start_cursor": state["cursor"], "input_verify_seconds": time.perf_counter() - started})
        torch.manual_seed(r.seed_for(config["seed"], "common_task0_initialization"))
        encoder = export.build_bound_encoder({"feature_dim": manifest["feature_dim"], "architecture": config["architecture"]}, runtime_root)
        classes = [item["class_id"] for item in manifest["classes"]]
        model = nn.ModuleDict({"encoder": encoder, "classifier": nn.Linear(config["architecture"]["d_model"], len(classes))}).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        if "model" in state:
            r.strict_model_load(model, state.pop("model"))
            r.strict_adam_load(optimizer, state.pop("optimizer"), has_updates=state["cursor"] > 0)
            normal = state["normalization"]
            r.require(normal["fit_rows"] == len(arrays["x"]) and all(isinstance(normal[k], torch.Tensor)
                      and normal[k].dtype == torch.float64 and normal[k].shape == (manifest["feature_dim"],)
                      and torch.isfinite(normal[k]).all().item() for k in ("mean", "scale"))
                      and (normal["scale"] > 0).all().item(), "normalization recovery dtype/shape/value mismatch")
        else:
            state.update(binding=binding, initial_state_sha256=r.tensor_state_hash(model.state_dict()),
                         initial_encoder_sha256=r.tensor_state_hash(encoder.state_dict()),
                         normalization=fit_normalization(arrays["x"], config["batch_size"], journal),
                         measurements=[], sampled_storage_peak_bytes=0, observed_optimizer_peak_bytes=0)
        def checkpoint():
            r.require(r.sha(__file__) == binding["implementation_sha256"] and r.sha(r.__file__) == binding["shared_helpers_sha256"]
                      and r.sha(export.__file__) == binding["encoder_adapter_sha256"], "pretraining code changed")
            if config["architecture"]["encoder_type"] == "ft_transformer":
                export.verify_runtime_sources(runtime_root)
            state["model"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            state["optimizer"] = optimizer.state_dict()
            state["commits"].append({"journal": journal.path.name, "last_sequence": journal.sequence, "cursor": state["cursor"]})
            result = r.save_checkpoint(output, state)
            state.pop("model")
            state.pop("optimizer")
            return result
        latest_ref = checkpoint()
        mean, scale = (state["normalization"][key].numpy() for key in ("mean", "scale"))
        lookup = {c: i for i, c in enumerate(classes)}
        batches_per_epoch = (len(arrays["x"]) + config["batch_size"] - 1) // config["batch_size"]
        total_steps = batches_per_epoch * config["epochs"]
        r.require(state["cursor"] <= total_steps, "checkpoint beyond registered training")
        start_cursor = state["cursor"]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        current_epoch, order = None, None
        for step in range(state["cursor"], total_steps):
            epoch, batch = divmod(step, batches_per_epoch)
            if epoch != current_epoch:
                current_epoch = epoch
                order = np.random.default_rng(r.seed_for(config["seed"], "common_task0_shuffle", epoch)).permutation(len(arrays["x"]))
            begin = batch * config["batch_size"]
            idx = order[begin:begin + config["batch_size"]]
            raw = np.array(arrays["x"][idx], dtype=np.float64)
            normalized = ((raw - mean) / scale).astype(np.float32)
            r.require(np.isfinite(normalized).all(), "nonfinite normalized training feature")
            features = torch.from_numpy(normalized).to(device)
            targets = torch.tensor([lookup[int(c)] for c in arrays["labels"][idx]], dtype=torch.int64, device=device)
            ids = [bytes(c).decode("ascii") for c in arrays["row_ids"][idx]]
            torch.manual_seed(r.seed_for(config["seed"], "common_task0_dropout", step))
            optimizer.zero_grad(set_to_none=True)
            journal.emit({"event": "batch_consumed_step_started", "unit": step, "row_indices": idx.tolist(), "row_ids": ids,
                          "class_targets": [int(c) for c in arrays["labels"][idx]], "target_scope": "multiclass_not_binary"})
            batch_start = time.perf_counter()
            logits = model["classifier"](model["encoder"](features))
            loss = nn.functional.cross_entropy(logits, targets)
            r.require(torch.isfinite(loss).item(), "nonfinite training loss")
            loss.backward()
            r.require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()), "nonfinite gradient")
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            r.require(all(torch.isfinite(p).all().item() for p in model.parameters()), "nonfinite updated state")
            journal.emit({"event": "step_completed", "unit": step})
            inventory = r.storage_inventory({"encoder_parameters": list(encoder.parameters()),
                        "temporary_classifier_parameters": list(model["classifier"].parameters()),
                        "gradients": [p.grad for p in model.parameters() if p.grad is not None],
                        "optimizer_state": optimizer.state, "current_batch": [features, targets, logits, loss]})
            state["sampled_storage_peak_bytes"] = max(state["sampled_storage_peak_bytes"], inventory["total_bytes"])
            state["observed_optimizer_peak_bytes"] = max(state["observed_optimizer_peak_bytes"], inventory["by_category_bytes"]["optimizer_state"])
            state["measurements"].append({"step": step, "epoch": step // batches_per_epoch, "raw_row_presentations": len(idx),
                    "multiclass_targets": len(idx), "binary_targets": 0, "loss": float(loss.detach().cpu()),
                    "batch_seconds": time.perf_counter() - batch_start, "batch_identity_sha256": r.object_hash([ids, arrays["labels"][idx].tolist()]),
                    "storage": inventory, "process_memory": r.process_memory(),
                    "cuda_allocated_peak_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                    "cuda_reserved_peak_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None})
            state["cursor"] = step + 1
            if emitter is not None:
                try:
                    emitter.emit(state["measurements"][-1])
                except Exception:
                    # Preserve completed computation; an uncertain remote send is not retried.
                    latest_ref = checkpoint()
                    raise
            if fault_after_steps is not None and state["cursor"] - start_cursor >= fault_after_steps:
                raise RuntimeError("synthetic_injected_task0_failure")
            if state["cursor"] % config["checkpoint_every_steps"] == 0 or state["cursor"] == total_steps or (
                    max_steps is not None and state["cursor"] - start_cursor >= max_steps):
                latest_ref = checkpoint()
            if max_steps is not None and state["cursor"] - start_cursor >= max_steps and state["cursor"] < total_steps:
                journal.emit({"event": "attempt_completed", "elapsed_seconds": time.perf_counter() - started})
                journal.close()
                journal = None
                r.atomic_json(output / "COST_LEDGER.json", pretrain_cost(output, state))
                paused = {"status": "PAUSED", "evidence_kind": evidence_kind, "cursor": state["cursor"], "checkpoint": latest_ref}
                r.atomic_json(output / "PAUSED.json", paused)
                return paused
        # Revalidate selected inputs, without opening calibration, omitted or test arrays.
        verified, other = load_cohort(directory, expected_manifest_sha256)
        close_arrays(other)
        r.require(verified == manifest and r.sha(config_path) == binding["config_sha256"], "training input changed")
        r.require(r.sha(__file__) == binding["implementation_sha256"] and r.sha(r.__file__) == binding["shared_helpers_sha256"]
                  and r.sha(export.__file__) == binding["encoder_adapter_sha256"], "pretraining code changed")
        if config["architecture"]["encoder_type"] == "ft_transformer":
            export.verify_runtime_sources(runtime_root)
        journal.emit({"event": "attempt_completed", "elapsed_seconds": time.perf_counter() - started})
        journal.close()
        journal = None
        ledger = pretrain_cost(output, state)
        r.atomic_json(output / "COST_LEDGER.json", ledger)
        encoder_hash = r.tensor_state_hash(encoder.state_dict())
        schema = {k: "encoder_" + str(i) for i, k in enumerate(encoder.state_dict())}
        np.savez(output / "inference-state.npz", **{schema[k]: t.detach().cpu().numpy() for k, t in encoder.state_dict().items()},
                 normalization_mean=mean, normalization_scale=scale)
        observed_rows = sum(a["observed_consumed_rows"] for a in ledger["attempts"])
        unknown = ledger["has_unknown_tail"] or any(a["step_completion_unknown_upper_bound"] for a in ledger["attempts"])
        cost = {"status": "MEASURED", "evidence_kind": evidence_kind, "encoder_state_sha256": encoder_hash,
                "cost_scope": "shared_encoder_pretraining_separate_from_L1", "measurements": {
                    "optimizer_steps": ledger["total_observed_completed_steps"], "raw_row_presentations": observed_rows,
                    "elapsed_seconds": sum(a["elapsed_seconds"] or 0.0 for a in ledger["attempts"]),
                    "measurement_scope": "all_attempt_observed_compute_lower_bound_if_uncertain", "unknown_tail": bool(unknown)}}
        r.atomic_json(output / "pretrain-cost.json", cost)
        profile = {"status": "COMPLETE", "evidence_kind": evidence_kind, "binding": binding, "config": config,
                   "training_objective": "temporary_shared_multiclass_cross_entropy", "checkpoint_policy": "fixed_last_epoch",
                   "initial_state_sha256": state["initial_state_sha256"], "initial_encoder_sha256": state["initial_encoder_sha256"],
                   "final_encoder_sha256": encoder_hash, "final_classifier_sha256": r.tensor_state_hash(model["classifier"].state_dict()),
                   "productive_optimizer_steps": state["cursor"], "productive_raw_row_presentations": sum(v["raw_row_presentations"] for v in state["measurements"]),
                   "productive_unique_raw_rows": len(arrays["row_ids"]), "normalization_fit_rows": state["normalization"]["fit_rows"],
                   "normalization_policy": state["normalization"]["policy"], "measurements": state["measurements"],
                   "sampled_storage_peak_bytes": state["sampled_storage_peak_bytes"], "observed_optimizer_peak_bytes": state["observed_optimizer_peak_bytes"],
                   "process_memory": r.process_memory(), "input_logical_array_bytes": {k: int(a.nbytes) for k, a in arrays.items()},
                   "shuffle_index_bytes_per_epoch": len(arrays["x"]) * np.dtype(np.int64).itemsize,
                   "replay_bytes": 0, "centroid_bytes": 0, "evaluation_performed": False, "held_out_arrays_opened": False,
                   "limitations": ["Training-only shared encoder stage, not full OFRA or a baseline reproduction.",
                                   "Tensor peaks are step-boundary samples; OS peaks are process-lifetime, not isolated allocations.",
                                   "Checkpoint I/O and temporary snapshots contribute to process memory and elapsed cost.",
                                   "Observed cost is a lower bound if an interrupted attempt has an unknown tail."]}
        r.atomic_json(output / "PRETRAIN_PROFILE.json", profile)
        metadata = {"dataset": manifest["dataset_id"], "seed": config["seed"], "evidence_kind": evidence_kind,
                    "checkpoint": 0, "seen_classes": classes, "feature_dim": manifest["feature_dim"], "architecture": config["architecture"],
                    "state_schema": {"encoder": schema, "normalization": {"mean": "normalization_mean", "scale": "normalization_scale"}},
                    "inference_state_sha256": r.sha(output / "inference-state.npz"),
                    "common_transform_fit_sha256": manifest["common_transform_fit_sha256"],
                    "normalization_fit_row_ids_sha256": manifest["collections"]["transform_fit"]["row_ids"]["sha256"],
                    "encoder_training_row_ids_sha256": manifest["collections"]["transform_fit"]["row_ids"]["sha256"],
                    "pretrain_profile_sha256": r.sha(output / "PRETRAIN_PROFILE.json"), "scope": "encoder_only_common_task0_not_full_OFRA_checkpoint"}
        metadata["canonical_sha256"] = r.object_hash(metadata)
        r.atomic_json(output / "checkpoint.json", metadata)
        complete = {"status": "COMPLETE", "stage": "common_task0_pretraining", "evidence_kind": evidence_kind,
                    "manifest_sha256": expected_manifest_sha256, "checkpoint": latest_ref,
                    "files": {name: r.sha(output / name) for name in ("checkpoint.json", "inference-state.npz", "pretrain-cost.json",
                                                                     "PRETRAIN_PROFILE.json", "COST_LEDGER.json", "INPUT_BINDING.json")}}
        r.atomic_json(output / "COMPLETE.json", complete)
        return complete
    except BaseException as exc:
        if journal is not None:
            journal.emit({"event": "attempt_failed", "failure_type": type(exc).__name__, "elapsed_seconds": time.perf_counter() - started})
            journal.close()
            journal = None
            # Only the immutable checkpoint's cursor/commit map can call a step productive.
            committed = {"cursor": 0, "commits": []}
            if (output / "LATEST.json").exists():
                latest = r.read_json(output / "LATEST.json")
                committed = torch.load(r.relative_file(output, {k: latest[k] for k in ("path", "sha256")}), map_location="cpu", weights_only=True)
            r.atomic_json(output / "COST_LEDGER.json", pretrain_cost(output, committed))
            r.atomic_json(output / ("FAILED-" + r.uuid.uuid4().hex + ".json"), {"error_type": type(exc).__name__, "status": "FAILED_RESUMABLE"})
        raise
    finally:
        if journal is not None:
            journal.close()
        if lock is not None:
            lock.close()
            lock_path.unlink()
        close_arrays(arrays)


def audit(output, directory, expected_manifest_sha256):
    """Check final state, productive raw-row streams and all-attempt cost identities."""
    output = no_links(output)
    complete = r.read_json(output / "COMPLETE.json")
    r.require(complete["status"] == "COMPLETE" and complete["stage"] == "common_task0_pretraining"
              and complete["manifest_sha256"] == expected_manifest_sha256, "pretraining completion mismatch")
    for name, digest in complete["files"].items():
        r.relative_file(output, {"path": name, "sha256": digest})
    state = torch.load(r.relative_file(output, complete["checkpoint"]), map_location="cpu", weights_only=True)
    profile = r.read_json(output / "PRETRAIN_PROFILE.json")
    metadata = r.read_json(output / "checkpoint.json")
    cost = r.read_json(output / "pretrain-cost.json")
    r.require(r.object_hash({k: v for k, v in metadata.items() if k != "canonical_sha256"}) == metadata["canonical_sha256"], "metadata changed")
    r.require(metadata["inference_state_sha256"] == r.sha(output / "inference-state.npz")
              and metadata["pretrain_profile_sha256"] == r.sha(output / "PRETRAIN_PROFILE.json"), "state/profile binding mismatch")
    with np.load(output / "inference-state.npz", allow_pickle=False) as archive:
        tensors = {k: torch.from_numpy(np.array(archive[v], copy=True)) for k, v in metadata["state_schema"]["encoder"].items()}
        normal = {k: np.array(archive[v], copy=True) for k, v in metadata["state_schema"]["normalization"].items()}
    encoder_hash = r.tensor_state_hash(tensors)
    checkpoint_encoder = {k.removeprefix("encoder."): v for k, v in state["model"].items() if k.startswith("encoder.")}
    r.require(encoder_hash == r.tensor_state_hash(checkpoint_encoder) == profile["final_encoder_sha256"] == cost["encoder_state_sha256"], "exported encoder differs from trained checkpoint")
    r.require(all(np.array_equal(normal[k], state["normalization"][k].numpy()) for k in normal), "exported normalization mismatch")
    r.require(profile["binding"] == state["binding"] == r.read_json(output / "INPUT_BINDING.json"), "profile/checkpoint input mismatch")
    r.require(profile["measurements"] == state["measurements"] and profile["productive_optimizer_steps"] == state["cursor"], "productive accounting mismatch")
    ledger = pretrain_cost(output, state)
    r.require(ledger == r.read_json(output / "COST_LEDGER.json"), "attempt cost ledger changed")
    measurements = cost["measurements"]
    unknown = ledger["has_unknown_tail"] or any(a["step_completion_unknown_upper_bound"] for a in ledger["attempts"])
    r.require(measurements["optimizer_steps"] == ledger["total_observed_completed_steps"]
              and measurements["raw_row_presentations"] == sum(a["observed_consumed_rows"] for a in ledger["attempts"])
              and measurements["elapsed_seconds"] == sum(a["elapsed_seconds"] or 0.0 for a in ledger["attempts"])
              and measurements["unknown_tail"] == bool(unknown), "observed cost differs from journals")
    manifest, arrays = load_cohort(directory, expected_manifest_sha256)
    try:
        expected_ids = manifest["collections"]["transform_fit"]["row_ids"]["sha256"]
        r.require(metadata["normalization_fit_row_ids_sha256"] == metadata["encoder_training_row_ids_sha256"] == expected_ids
                  and metadata["common_transform_fit_sha256"] == manifest["common_transform_fit_sha256"], "training cohort identity mismatch")
        limits = {}
        for commit in state["commits"]:
            limits[commit["journal"]] = max(limits.get(commit["journal"], 0), commit["last_sequence"])
        seen, done = set(), set()
        config = profile["config"]
        config_check(config)
        r.require(r.object_hash(config) == profile["binding"]["config_value_sha256"], "configuration mismatch")
        expected_steps = ((len(arrays["x"]) + config["batch_size"] - 1) // config["batch_size"]) * config["epochs"]
        r.require(state["cursor"] == expected_steps, "incomplete registered training")
        cached_epoch, cached_order = None, None
        per_epoch = (len(arrays["x"]) + config["batch_size"] - 1) // config["batch_size"]
        for attempt in ledger["attempts"]:
            with (output / attempt["journal"]).open("rb") as stream:
                for line in stream:
                    try:
                        event = r.json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        break
                    if event["sequence"] > limits.get(attempt["journal"], 0):
                        continue
                    if event["event"] == "batch_consumed_step_started":
                        step = event["unit"]
                        r.require(type(step) is int and 0 <= step < expected_steps and step not in seen, "repeated/invalid committed step")
                        epoch, batch = divmod(step, per_epoch)
                        if epoch != cached_epoch:
                            cached_epoch = epoch
                            cached_order = np.random.default_rng(r.seed_for(config["seed"], "common_task0_shuffle", epoch)).permutation(len(arrays["x"]))
                        begin = batch * config["batch_size"]
                        idx = cached_order[begin:begin + config["batch_size"]]
                        ids = [bytes(v).decode("ascii") for v in arrays["row_ids"][idx]]
                        labels = arrays["labels"][idx].tolist()
                        r.require(event["row_indices"] == idx.tolist() and event["row_ids"] == ids and event["class_targets"] == labels, "committed raw-row/target identity mismatch")
                        item = profile["measurements"][step]
                        r.require(item["raw_row_presentations"] == item["multiclass_targets"] == len(idx) and item["binary_targets"] == 0
                                  and item["batch_identity_sha256"] == r.object_hash([ids, labels]), "training presentation counters mismatch")
                        seen.add(step)
                    elif event["event"] == "step_completed":
                        r.require(event["unit"] in seen and event["unit"] not in done, "unmatched step completion")
                        done.add(event["unit"])
        r.require(seen == done == set(range(expected_steps)), "missing committed training receipt")
        r.require(profile["productive_raw_row_presentations"] == len(arrays["x"]) * config["epochs"], "epoch exposure mismatch")
        return {"status": "AUDIT_PASS", "evidence_kind": complete["evidence_kind"], "productive_steps": expected_steps,
                "scope": "local_training_state_and_observed_accounting_not_scientific_reproduction"}
    finally:
        close_arrays(arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task0_directory", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-kind", choices=("synthetic", "real"), required=True)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.audit_only:
            result = audit(args.output, args.task0_directory, args.manifest_sha256)
        else:
            result = run(args.task0_directory, args.manifest_sha256, args.config, args.output, evidence_kind=args.evidence_kind,
                         runtime_root=args.runtime_root, device_name=args.device, resume=args.resume, max_steps=args.max_steps)
        print(r.json_bytes({"status": result["status"], "evidence_kind": args.evidence_kind}).decode())
        return 0
    except (r.Invalid, OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        print(r.json_bytes({"status": "BLOCKED_OR_FAILED", "error_type": type(exc).__name__}).decode())
        return 2


if __name__ == "__main__":
    sys.exit(main())
