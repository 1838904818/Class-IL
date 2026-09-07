"""Executable frozen-embedding CE versus conditional-focal objective control."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
import time
import uuid

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)
torch.set_default_dtype(torch.float32)
ARMS = ("ce", "conditional_focal")
CONFIG_KEYS = {"seed", "epochs", "batch_size", "eval_batch_size", "rank", "lora_alpha",
               "learning_rate", "weight_decay", "negative_ratio", "minority_threshold",
               "focal_alpha", "focal_gamma", "exemplar_capacity", "exemplar_selection",
               "checkpoint_policy", "profile_label"}
SPLIT_KEYS = {"embeddings", "labels", "row_ids", "available_tasks", "group_ids"}


class Invalid(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise Invalid(message)


def exact(value, fields, context):
    require(isinstance(value, dict) and set(value) == set(fields), context + ": unexpected fields")


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, "invalid integer")
    return value


def number(value, minimum=0, maximum=None):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum
            and (maximum is None or value <= maximum), "invalid finite number")
    return value


def safe_name(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", value), "unsafe public identifier")
    return value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_sha(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "invalid SHA256")
    return value


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def object_hash(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def read_json(path):
    def pairs(values):
        out = {}
        for key, value in values:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    def floating(value):
        result = float(value)
        require(math.isfinite(result), "nonfinite JSON")
        return result
    def invalid(value):
        raise Invalid("nonfinite JSON")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=pairs,
                      parse_float=floating, parse_constant=invalid)


def relative_file(root, ref):
    exact(ref, {"path", "sha256"}, "file reference")
    check_sha(ref["sha256"])
    name = ref["path"]
    require(isinstance(name, str) and name and ":" not in name and "\\" not in name,
            "unsafe relative path")
    child = Path(name)
    require(not child.is_absolute() and ".." not in child.parts, "unsafe relative path")
    base = Path(root).resolve()
    path = (base / child).resolve()
    require(path != base and path.is_relative_to(base), "artifact escapes root")
    require(sha(path) == ref["sha256"], "artifact digest mismatch")
    return path


def atomic_bytes(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path, value):
    atomic_bytes(path, json_bytes(value))


def seed_for(seed, *parts):
    return int.from_bytes(hashlib.sha256(json_bytes([seed, *parts])).digest()[:8], "little") % (2**63 - 1)


def tensor_state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(json_bytes([name, value.dtype.str, list(value.shape)]))
        digest.update(value.tobytes())
    return digest.hexdigest()


def environment(device):
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__),
            "device": str(device), "torch_threads": torch.get_num_threads(),
            "cuda_runtime": torch.version.cuda if device.type == "cuda" else None,
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def process_memory():
    """Actual current RSS and OS process-lifetime high water; no child processes."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                 "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                 "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        record = Counters()
        record.cb = ctypes.sizeof(record)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        require(psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(record), record.cb),
                "process memory measurement failed")
        return {"rss_bytes": int(record.WorkingSetSize), "process_lifetime_peak_rss_bytes": int(record.PeakWorkingSetSize)}
    if sys.platform.startswith("linux"):
        values = {}
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, kb, _ = line.split()
                values[key] = int(kb) * 1024
        require(set(values) == {"VmRSS:", "VmHWM:"}, "RSS counters unavailable")
        return {"rss_bytes": values["VmRSS:"], "process_lifetime_peak_rss_bytes": values["VmHWM:"]}
    raise Invalid("supported measured-memory platforms: Windows and Linux")


def tensors_in(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors_in(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors_in(item)


def storage_inventory(groups):
    """Count unique live torch storages, not tensor views or numel sums."""
    seen = set()
    categories = {}
    entries = []
    for category, values in groups.items():
        categories[category] = 0
        for tensor in tensors_in(values):
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
            if key in seen:
                continue
            seen.add(key)
            size = storage.nbytes()
            categories[category] += size
            entries.append({"category": category, "device": str(tensor.device), "storage_bytes": size,
                            "representative_shape": list(tensor.shape), "dtype": str(tensor.dtype)})
    return {"scope": "unique_live_torch_storage_not_process_memory", "by_category_bytes": categories,
            "total_bytes": sum(categories.values()), "storages": entries}


def strict_model_load(model, state):
    expected = model.state_dict()
    exact(state, set(expected), "model tensor state")
    require(all(isinstance(state[k], torch.Tensor) and state[k].layout == torch.strided
                and state[k].shape == value.shape and state[k].dtype == value.dtype
                and torch.isfinite(state[k]).all().item() for k, value in expected.items()),
            "model tensor dtype/shape/value mismatch")
    model.load_state_dict(state, strict=True)


def strict_adam_load(optimizer, saved, *, has_updates):
    """Validate before PyTorch can coerce precision or replace registered options."""
    exact(saved, {"state", "param_groups"}, "Adam state")
    expected = optimizer.state_dict()
    require(isinstance(saved["param_groups"], list) and len(saved["param_groups"]) == len(expected["param_groups"]), "Adam group count")
    parameters = {}
    for actual, wanted, live in zip(saved["param_groups"], expected["param_groups"], optimizer.param_groups):
        exact(actual, set(wanted), "Adam parameter group")
        require(all(type(actual[k]) is type(v) and actual[k] == v for k, v in wanted.items()), "Adam options/parameter order mismatch")
        require(all(type(k) is int for k in actual["params"]), "Adam parameter identity type")
        parameters.update(zip(wanted["params"], live["params"]))
    require(isinstance(saved["state"], dict) and all(type(k) is int for k in saved["state"]), "Adam parameter identity type")
    require(set(saved["state"]) == (set(parameters) if has_updates else set()), "Adam missing/unexpected parameter state")
    for key, values in saved["state"].items():
        exact(values, {"step", "exp_avg", "exp_avg_sq"}, "Adam tensor fields")
        param = parameters[key]
        for field in ("exp_avg", "exp_avg_sq"):
            tensor = values[field]
            require(isinstance(tensor, torch.Tensor) and tensor.layout == torch.strided and tensor.shape == param.shape
                    and tensor.dtype == param.dtype and torch.isfinite(tensor).all().item(), "Adam tensor dtype/shape/value mismatch")
        step = values["step"]
        require((values["exp_avg_sq"] >= 0).all().item(), "negative Adam second moment")
        require(isinstance(step, torch.Tensor) and step.shape == torch.Size([]) and step.dtype == torch.float32
                and torch.isfinite(step).item() and step.item() >= 1 and step.item().is_integer(), "Adam step scalar mismatch")
    optimizer.load_state_dict(saved)


class FamilyHead(nn.Module):
    """Same output-space residual low-rank two-logit form; not a router."""
    def __init__(self, width, rank, alpha):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, width))
        self.B = nn.Parameter(torch.zeros(width, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.classifier = nn.Linear(width, 2)
        self.scaling = alpha / rank

    def forward(self, embedding):
        return self.classifier(embedding + self.scaling * (embedding @ self.A.T @ self.B.T))


def objective(logits, labels, arm, positive_population, config):
    ce = F.cross_entropy(logits, labels, reduction="none")
    use_focal = arm == "conditional_focal" and positive_population < config["minority_threshold"]
    if not use_focal:
        return ce.mean(), "binary_cross_entropy"
    probability = torch.softmax(logits, dim=1).gather(1, labels[:, None]).squeeze(1)
    weight = torch.where(labels == 1, config["focal_alpha"], 1.0 - config["focal_alpha"])
    return (weight * (1.0 - probability).pow(config["focal_gamma"]) * ce).mean(), "conditional_focal"


def config_check(config):
    exact(config, CONFIG_KEYS, "config")
    for key in ("seed", "weight_decay", "focal_gamma"):
        number(config[key])
    integer(config["seed"])
    for key in ("epochs", "batch_size", "eval_batch_size", "rank", "negative_ratio", "minority_threshold", "exemplar_capacity"):
        integer(config[key], 1)
    require(config["batch_size"] >= config["negative_ratio"] + 1, "batch size must accommodate one positive and the negative ratio")
    for key in ("learning_rate", "lora_alpha"):
        number(config[key], 1e-15)
    number(config["focal_alpha"], 0, 1)
    require(config["exemplar_selection"] == "uniform_without_replacement", "unsupported exemplar selector")
    require(config["checkpoint_policy"] == "fixed_last_epoch", "test/calibration selection forbidden in L1")
    require(config["profile_label"] == "prospective_pilot_defaults_not_optimized_not_preregistered", "pilot disclosure required")


def load_inputs(manifest_path):
    started = time.perf_counter()
    path = Path(manifest_path).resolve()
    root = path.parent
    manifest_hash = sha(path)
    manifest = read_json(path)
    exact(manifest, {"schema_version", "dataset", "evidence_kind", "tasks", "group_mode", "encoder_binding",
                     "splits", "config", "allow_aggregate_wandb"}, "manifest")
    require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1, "manifest schema")
    safe_name(manifest["dataset"])
    require(manifest["evidence_kind"] in ("synthetic", "real"), "evidence kind")
    require(type(manifest["allow_aggregate_wandb"]) is bool, "W&B governance flag")
    require(manifest["group_mode"] in ("provided_disjoint", "row_identity_proxy"), "group mode")
    config_check(manifest["config"])
    tasks = manifest["tasks"]
    require(isinstance(tasks, list) and tasks and all(isinstance(t, list) and t for t in tasks), "empty tasks")
    introduction = {}
    for t, task in enumerate(tasks):
        for c in task:
            integer(c)
            require(c not in introduction, "duplicate task class")
            introduction[c] = t
    require(len(tasks[0]) >= 2, "Task 0 needs at least two classes for binary negatives")
    binding = manifest["encoder_binding"]
    exact(binding, {"checkpoint", "preprocessing", "export_receipt", "pretrain_cost_receipt", "state_sha256",
                    "trained_tasks", "embedding_dimension"}, "encoder binding")
    require(binding["trained_tasks"] == [0], "encoder must be trained on Task 0 only")
    integer(binding["embedding_dimension"], 1)
    check_sha(binding["state_sha256"])
    verified = {key: relative_file(root, binding[key]) for key in
                ("checkpoint", "preprocessing", "export_receipt", "pretrain_cost_receipt")}
    with np.load(verified["checkpoint"], allow_pickle=False) as checkpoint:
        encoder_tensors = {name: torch.from_numpy(np.array(checkpoint[name], copy=True)) for name in checkpoint.files}
    require(encoder_tensors and all(torch.isfinite(value).all().item() for value in encoder_tensors.values()), "invalid encoder state arrays")
    require(tensor_state_hash(encoder_tensors) == binding["state_sha256"], "actual encoder tensor hash mismatch")
    del encoder_tensors
    receipt = read_json(verified["export_receipt"])
    exact(receipt, {"schema_version", "status", "evidence_kind", "encoder_state_sha256", "checkpoint_sha256",
                   "preprocessing_sha256", "split_array_sha256", "export_environment", "cost_scope"}, "export receipt")
    require(receipt["schema_version"] == 1 and receipt["status"] == "COMPLETE"
            and receipt["evidence_kind"] == manifest["evidence_kind"], "unfinished or mismatched embedding export")
    require(receipt["encoder_state_sha256"] == binding["state_sha256"]
            and receipt["checkpoint_sha256"] == binding["checkpoint"]["sha256"]
            and receipt["preprocessing_sha256"] == binding["preprocessing"]["sha256"], "encoder receipt binding mismatch")
    require(isinstance(receipt["export_environment"], dict) and receipt["export_environment"], "export environment missing")
    if "profile_sha256" in receipt["export_environment"]:
        relative_file(root, {"path": "EXPORT_PROFILE.json", "sha256": receipt["export_environment"]["profile_sha256"]})
    require(receipt["cost_scope"] == "separate_embedding_materialization_stage", "embedding cost scope")
    pretrain = read_json(verified["pretrain_cost_receipt"])
    exact(pretrain, {"status", "evidence_kind", "encoder_state_sha256", "cost_scope", "measurements"}, "pretrain cost receipt")
    require(pretrain["status"] in ("MEASURED", "HISTORICAL_UNMEASURED_DISCLOSED"), "pretrain cost status")
    require(pretrain["evidence_kind"] == manifest["evidence_kind"]
            and pretrain["encoder_state_sha256"] == binding["state_sha256"], "pretrain cost binding")
    require(pretrain["cost_scope"] == "shared_encoder_pretraining_separate_from_L1", "pretrain cost scope")
    if pretrain["status"] == "MEASURED":
        measurement_fields = {"optimizer_steps", "raw_row_presentations", "elapsed_seconds"}
        if "measurement_scope" in pretrain["measurements"]:
            measurement_fields |= {"measurement_scope", "unknown_tail"}
            require(pretrain["measurements"]["measurement_scope"] == "all_attempt_observed_compute_lower_bound_if_uncertain"
                    and type(pretrain["measurements"]["unknown_tail"]) is bool, "pretrain uncertainty disclosure")
        exact(pretrain["measurements"], measurement_fields, "pretrain measurements")
        integer(pretrain["measurements"]["optimizer_steps"], 1)
        integer(pretrain["measurements"]["raw_row_presentations"], 1)
        number(pretrain["measurements"]["elapsed_seconds"], 0)
    else:
        require(pretrain["measurements"] is None, "unknown historical pretraining cost cannot be invented")
    exact(manifest["splits"], {"train", "test"}, "splits")
    exact(receipt["split_array_sha256"], {"train", "test"}, "receipt splits")
    arrays = {}
    identity = {}
    groups = {}
    for split, descriptors in manifest["splits"].items():
        exact(descriptors, SPLIT_KEYS, "split arrays")
        exact(receipt["split_array_sha256"][split], SPLIT_KEYS, "receipt array hashes")
        data = {}
        for key, ref in descriptors.items():
            require(receipt["split_array_sha256"][split][key] == ref["sha256"], "embedding/label export binding mismatch")
            data[key] = np.load(relative_file(root, ref), mmap_mode="r", allow_pickle=False)
        x, y = data["embeddings"], data["labels"]
        require(x.dtype == np.float32 and x.ndim == 2 and x.shape[1] == binding["embedding_dimension"], "embedding shape/dtype")
        require(y.dtype == np.int64 and y.shape == (len(x),) and len(x) > 0, "label shape/dtype")
        require(data["available_tasks"].dtype == np.int64 and data["available_tasks"].shape == y.shape, "availability shape/dtype")
        for key in ("row_ids", "group_ids"):
            require(data[key].dtype == np.dtype("S64") and data[key].shape == y.shape, "identity shape/dtype")
            require(all(re.fullmatch(rb"[0-9a-f]{64}", bytes(value)) for value in data[key]), "noncanonical raw/group id")
        identity[split] = set(data["row_ids"].tolist())
        groups[split] = set(data["group_ids"].tolist())
        require(len(identity[split]) == len(x), "duplicate original raw row")
        require(set(y.tolist()) == set(introduction), "all classes need nonzero fitting/evaluation support")
        expected = np.array([introduction[int(c)] for c in y], dtype=np.int64)
        require(np.array_equal(expected, data["available_tasks"]), "label/task availability mismatch")
        for start in range(0, len(x), manifest["config"]["eval_batch_size"]):
            require(np.isfinite(x[start:start + manifest["config"]["eval_batch_size"]]).all(), "nonfinite embedding")
        arrays[split] = data
    require(not (identity["train"] & identity["test"]), "train/test raw-row leakage")
    require(not (groups["train"] & groups["test"]), "train/test group leakage")
    require(sha(path) == manifest_hash, "input manifest changed during validation")
    return manifest, arrays, {"manifest_sha256": manifest_hash, "pretrain_cost": pretrain,
                             "load_verify_seconds": time.perf_counter() - started}


def verify_input_bindings(manifest_path, manifest, expected_sha256):
    root = Path(manifest_path).resolve().parent
    require(sha(manifest_path) == expected_sha256, "input manifest changed during execution")
    for fields in manifest["splits"].values():
        for descriptor in fields.values():
            relative_file(root, descriptor)
    for key in ("checkpoint", "preprocessing", "export_receipt", "pretrain_cost_receipt"):
        relative_file(root, manifest["encoder_binding"][key])
    receipt = read_json(root / manifest["encoder_binding"]["export_receipt"]["path"])
    if "profile_sha256" in receipt["export_environment"]:
        relative_file(root, {"path": "EXPORT_PROFILE.json", "sha256": receipt["export_environment"]["profile_sha256"]})


def buffers_for(train, tasks, config):
    buffers = {}
    for task in tasks:
        for c in task:
            indices = np.flatnonzero(train["labels"] == c)
            rng = np.random.default_rng(seed_for(config["seed"], "exemplar", c))
            buffers[c] = np.sort(rng.choice(indices, size=min(config["exemplar_capacity"], len(indices)), replace=False))
    return buffers


def batches_for(train, tasks, buffers, config, task, class_id, epoch):
    """All current positives; fresh bounded nonreplacement negatives per epoch."""
    positive = np.flatnonzero((train["labels"] == class_id) & (train["available_tasks"] == task))
    current = np.flatnonzero((train["available_tasks"] == task) & (train["labels"] != class_id))
    old = [buffers[c] for old_task in tasks[:task] for c in old_task]
    negative = np.concatenate([current, *old]) if old else current
    require(len(positive) and len(negative), "empty binary positive/negative pool")
    rng = np.random.default_rng(seed_for(config["seed"], "batch", task, class_id, epoch))
    positive = rng.permutation(positive)
    negative = rng.choice(negative, size=min(len(negative), len(positive) * config["negative_ratio"]), replace=False)
    width = max(1, config["batch_size"] // (config["negative_ratio"] + 1))
    before = 0
    for start in range(0, len(positive), width):
        pos = positive[start:start + width]
        after = len(negative) * (start + len(pos)) // len(positive)
        neg = negative[before:after]
        before = after
        indices = np.concatenate([pos, neg])
        labels = np.concatenate([np.ones(len(pos), np.int64), np.zeros(len(neg), np.int64)])
        order = rng.permutation(len(indices))
        yield indices[order], labels[order], len(positive)


def confusion_metrics(confusion, axis):
    cm = np.asarray(confusion, dtype=np.int64)
    support = cm.sum(1)
    predicted = cm.sum(0)
    correct = np.diag(cm)
    recall = np.divide(correct, support, out=np.zeros(len(axis), float), where=support > 0)
    precision = np.divide(correct, predicted, out=np.zeros(len(axis), float), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(axis), float), where=precision + recall > 0)
    return {"class_axis": list(axis), "confusion": cm.tolist(), "accuracy": float(correct.sum() / support.sum()),
            "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall.mean()),
            "per_class": [{"class_id": int(c), "support": int(support[i]), "precision": float(precision[i]),
                           "recall": float(recall[i]), "f1": float(f1[i])} for i, c in enumerate(axis)]}


@torch.no_grad()
def evaluate(heads, test, task, config, device, prior):
    started = time.perf_counter()
    axis = sorted(heads)
    lookup = {c: i for i, c in enumerate(axis)}
    indices = np.flatnonzero(test["available_tasks"] <= task)
    confusion = np.zeros((len(axis), len(axis)), np.int64)
    prediction_digest = hashlib.sha256()
    for start in range(0, len(indices), config["eval_batch_size"]):
        selected = indices[start:start + config["eval_batch_size"]]
        embedding = torch.from_numpy(np.array(test["embeddings"][selected], copy=True)).to(device)
        scores = torch.stack([torch.softmax(heads[c](embedding), dim=1)[:, 1] for c in axis], dim=1)
        pred = scores.argmax(1).cpu().numpy()
        true = np.array([lookup[int(c)] for c in test["labels"][selected]], np.int64)
        np.add.at(confusion, (true, pred), 1)
        prediction_digest.update(np.asarray(axis, np.int64)[pred].tobytes())
    result = confusion_metrics(confusion, axis)
    previous = {}
    for old in prior:
        for row in old["per_class"]:
            c = row["class_id"]
            previous[c] = max(previous.get(c, -math.inf), row["recall"])
    forgetting = [{"class_id": row["class_id"], "signed_recall_forgetting": previous[row["class_id"]] - row["recall"]}
                  for row in result["per_class"] if row["class_id"] in previous]
    result.update(task=task, signed_forgetting=forgetting,
                  mean_signed_forgetting=float(np.mean([x["signed_recall_forgetting"] for x in forgetting])) if forgetting else None,
                  prediction_sha256=prediction_digest.hexdigest(), prediction_seconds=time.perf_counter() - started,
                  selection_role="report_only_never_checkpoint_or_hyperparameter_selection")
    return result


class Journal:
    def __init__(self, output):
        self.path = output / ("attempt-" + uuid.uuid4().hex + ".jsonl")
        self.stream = self.path.open("xb")
        self.sequence = 0

    def emit(self, event):
        self.sequence += 1
        value = {"sequence": self.sequence, **event}
        self.stream.write(json_bytes(value) + b"\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self):
        self.stream.close()


def save_checkpoint(output, state):
    name = "checkpoint-" + f'{state["cursor"]:06d}-' + uuid.uuid4().hex + ".pt"
    path = output / name
    with path.open("xb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    atomic_json(output / "LATEST.json", {"path": name, "sha256": sha(path), "cursor": state["cursor"]})
    return {"path": name, "sha256": sha(path)}


def units_for(tasks, epochs):
    return [unit for t, task in enumerate(tasks) for unit in
            ([{"kind": "train", "task": t, "class_id": c, "epoch": e} for c in task for e in range(epochs)]
             + [{"kind": "evaluate", "task": t}])]


def load_heads(state, width, config, device):
    heads = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for c, tensors in state["heads"][arm].items():
            model = FamilyHead(width, config["rank"], config["lora_alpha"]).to(device)
            strict_model_load(model, tensors)
            heads[arm][int(c)] = model
    return heads


def snapshot(heads, optimizers, batch_tensors=()):
    groups = {}
    for arm in ARMS:
        groups[arm + "/head_parameters"] = [p for model in heads[arm].values() for p in model.parameters()]
        groups[arm + "/gradients"] = [p.grad for model in heads[arm].values() for p in model.parameters() if p.grad is not None]
        groups[arm + "/optimizer_state"] = optimizers[arm].state if arm in optimizers else {}
    groups["current_batch"] = batch_tensors
    return storage_inventory(groups)


def train_unit(unit, heads, state, train, tasks, buffers, config, device, journal, fault_after_steps=None):
    c, epoch = unit["class_id"], unit["epoch"]
    width = train["embeddings"].shape[1]
    if c not in heads["ce"]:
        torch.manual_seed(seed_for(config["seed"], "head", c))
        initial = FamilyHead(width, config["rank"], config["lora_alpha"])
        initial_state = copy.deepcopy(initial.state_dict())
        initial_hash = tensor_state_hash(initial_state)
        state["initial_head_sha256"][str(c)] = initial_hash
        for arm in ARMS:
            heads[arm][c] = FamilyHead(width, config["rank"], config["lora_alpha"]).to(device)
            heads[arm][c].load_state_dict(initial_state)
            require(tensor_state_hash(heads[arm][c].state_dict()) == initial_hash, "initial state mismatch")
        del initial, initial_state
    optimizers = {arm: torch.optim.Adam(heads[arm][c].parameters(), lr=config["learning_rate"],
                                      weight_decay=config["weight_decay"]) for arm in ARMS}
    if state["active_optimizer"] is not None:
        require(state["active_optimizer"]["class_id"] == c, "optimizer recovery class mismatch")
        for arm in ARMS:
            strict_adam_load(optimizers[arm], state["active_optimizer"][arm], has_updates=True)
        state["active_optimizer"] = None
    else:
        require(epoch == 0, "missing optimizer state for resumed epoch")
    result = {}
    for arm in ARMS:
        model = heads[arm][c]
        model.train()
        started = time.perf_counter()
        counter = {"optimizer_steps": 0, "raw_row_presentations": 0, "binary_targets": 0,
                   "positive_targets": 0, "negative_targets": 0, "old_row_presentations": 0,
                   "new_row_presentations": 0, "zero_negative_steps": 0}
        used = set()
        digest = hashlib.sha256()
        peak_storage = 0
        peak_optimizer = 0
        sampled_rss_peak = 0
        loss_sum = 0.0
        last_inventory = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        for batch_index, (indices, labels, positive_population) in enumerate(batches_for(train, tasks, buffers, config, unit["task"], c, epoch)):
            embedding = torch.from_numpy(np.array(train["embeddings"][indices], copy=True)).to(device)
            target = torch.from_numpy(labels).to(device)
            ids = [bytes(x).decode("ascii") for x in train["row_ids"][indices]]
            event_identity = {"unit": state["cursor"], "arm": arm, "task": unit["task"], "class_id": c,
                              "epoch": epoch, "batch_index": batch_index}
            journal.emit({"event": "batch_consumed_step_started", **event_identity, "row_ids": ids,
                          "row_indices": indices.tolist(), "row_available_tasks": train["available_tasks"][indices].tolist(),
                          "binary_targets": labels.tolist()})
            optimizers[arm].zero_grad(set_to_none=True)
            logits = model(embedding)
            loss, loss_kind = objective(logits, target, arm, positive_population, config)
            require(torch.isfinite(loss).item(), "nonfinite objective")
            loss.backward()
            require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()), "nonfinite gradient")
            optimizers[arm].step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            journal.emit({"event": "step_completed", **event_identity, "rows": len(indices)})
            require(all(torch.isfinite(p).all().item() for p in model.parameters()), "nonfinite updated parameters")
            counter["optimizer_steps"] += 1
            counter["raw_row_presentations"] += len(indices)
            counter["binary_targets"] += len(labels)
            counter["positive_targets"] += int(labels.sum())
            counter["negative_targets"] += int((labels == 0).sum())
            old = int((train["available_tasks"][indices] < unit["task"]).sum())
            counter["old_row_presentations"] += old
            counter["new_row_presentations"] += len(indices) - old
            counter["zero_negative_steps"] += int(not (labels == 0).any())
            used.update(ids)
            digest.update(json_bytes([ids, labels.tolist()]))
            loss_sum += float(loss.detach().cpu()) * len(indices)
            inventory = snapshot(heads, optimizers, [embedding, target, logits, loss])
            last_inventory = inventory
            peak_storage = max(peak_storage, inventory["total_bytes"])
            peak_optimizer = max(peak_optimizer, inventory["by_category_bytes"][arm + "/optimizer_state"])
            sampled_rss_peak = max(sampled_rss_peak, process_memory()["rss_bytes"])
            if fault_after_steps is not None and counter["optimizer_steps"] >= fault_after_steps:
                raise RuntimeError("synthetic_injected_interruption")
        counter["unique_raw_rows"] = len(used)
        memory = process_memory()
        result[arm] = {"counters": counter, "batch_stream_sha256": digest.hexdigest(),
                       "mean_training_loss": loss_sum / counter["raw_row_presentations"], "loss_kind": loss_kind,
                       "train_seconds": time.perf_counter() - started, "head_state_sha256": tensor_state_hash(model.state_dict()),
                       "last_storage_inventory": last_inventory, "sampled_live_tensor_peak_bytes": peak_storage,
                       "observed_optimizer_peak_bytes": peak_optimizer, "sampled_process_rss_peak_bytes": sampled_rss_peak,
                       "process_lifetime_peak_rss_bytes": memory["process_lifetime_peak_rss_bytes"],
                       "cuda_allocated_peak_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                       "cuda_reserved_peak_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                       "scope": "paired_harness_process_including_both_arms_not_single_method_deployment"}
        model.eval()
    require(result["ce"]["counters"] == result["conditional_focal"]["counters"]
            and result["ce"]["batch_stream_sha256"] == result["conditional_focal"]["batch_stream_sha256"],
            "objective arms have unequal exposure or step count")
    state["active_optimizer"] = {"class_id": c, **{arm: optimizers[arm].state_dict() for arm in ARMS}} if epoch + 1 < config["epochs"] else None
    if state["active_optimizer"] is None:
        for arm in ARMS:
            for parameter in heads[arm][c].parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
    return result


def cost_ledger(output, committed_cursor, commits):
    attempts = []
    for path in sorted(output.glob("attempt-*.jsonl")):
        require(path.resolve().is_relative_to(output.resolve()), "journal escapes output root")
        records = []
        complete_lines = True
        with path.open("rb") as stream:
            for line in stream:
                try:
                    records.append(json.loads(line))
                except (ValueError, UnicodeDecodeError):
                    complete_lines = False
                    break
        started = [x for x in records if x.get("event") == "batch_consumed_step_started"]
        done = [x for x in records if x.get("event") == "step_completed"]
        completed_attempt = any(x.get("event") == "attempt_completed" for x in records)
        failure = next((x for x in records if x.get("event") == "attempt_failed"), None)
        committed_here = [x["cursor"] for x in commits if x["journal"] == path.name]
        # Per-attempt commit receipts distinguish retried work from another attempt's later cursor.
        this_cursor = max(committed_here, default=next((x["start_cursor"] for x in records if x.get("event") == "attempt_started"), 0))
        discarded = [x for x in done if x["unit"] >= this_cursor]
        consumed_discarded = [x for x in started if x["unit"] >= this_cursor]
        attempts.append({"journal": path.name, "sha256": sha(path), "completed_attempt": completed_attempt,
                         "failure_type": failure["failure_type"] if failure else None,
                         "elapsed_seconds": failure["elapsed_seconds"] if failure else next((x["elapsed_seconds"] for x in records if x.get("event") == "attempt_completed"), None),
                         "observed_completed_steps": len(done), "observed_consumed_rows": sum(len(x["row_ids"]) for x in started),
                         "uncommitted_completed_steps": len(discarded),
                         "uncommitted_consumed_rows": sum(len(x["row_ids"]) for x in consumed_discarded),
                         "step_completion_unknown_upper_bound": len(started) - len(done),
                         "journal_complete_lines": complete_lines,
                         "hard_failure_tail_unknown": not (completed_attempt or failure) or not complete_lines})
    return {"scope": "all_attempt_compute_including_discarded_or_uncertain_work", "committed_cursor": committed_cursor,
            "attempts": attempts, "total_observed_completed_steps": sum(x["observed_completed_steps"] for x in attempts),
            "total_uncommitted_completed_steps": sum(x["uncommitted_completed_steps"] for x in attempts),
            "has_unknown_tail": any(x["hard_failure_tail_unknown"] for x in attempts)}


def aggregate_wandb(result, project, allowed):
    """Only called by explicit online opt-in; never logs rows, files or source."""
    require(allowed, "dataset governance does not allow aggregate W&B")
    safe_name(project)
    import wandb
    settings = wandb.Settings(console="off", disable_code=True, disable_git=True, disable_job_creation=True,
                              save_code=False, x_disable_meta=True, x_disable_stats=True, x_disable_machine_info=True,
                              x_save_requirements=False, x_stats_track_process_tree=False, silent=True)
    run = wandb.init(project=project, name="l1-objective-control", mode="online", settings=settings,
                     id=object_hash(result["binding"])[:16], resume="allow",
                     config={"stage": "L1", "evidence_kind": result["evidence_kind"]})
    try:
        for arm in ARMS:
            for checkpoint in result["metrics"][arm]:
                payload = {"task": checkpoint["task"], arm + "/accuracy": checkpoint["accuracy"],
                           arm + "/macro_f1": checkpoint["macro_f1"],
                           arm + "/balanced_accuracy": checkpoint["balanced_accuracy"]}
                if checkpoint["mean_signed_forgetting"] is not None:
                    payload[arm + "/mean_signed_forgetting"] = checkpoint["mean_signed_forgetting"]
                per_class = wandb.Table(columns=["task", "class_id", "support", "precision", "recall", "f1"])
                for row in checkpoint["per_class"]:
                    per_class.add_data(checkpoint["task"], *[row[k] for k in ("class_id", "support", "precision", "recall", "f1")])
                confusion = wandb.Table(columns=["task", "true_class_id", "predicted_class_id", "count"])
                for i, true in enumerate(checkpoint["class_axis"]):
                    for j, predicted in enumerate(checkpoint["class_axis"]):
                        confusion.add_data(checkpoint["task"], true, predicted, checkpoint["confusion"][i][j])
                payload[arm + "/per_class_task_" + str(checkpoint["task"])] = per_class
                payload[arm + "/confusion_task_" + str(checkpoint["task"])] = confusion
                run.log(payload)
    finally:
        run.finish()
    return {"status": "client_finished_remote_verification_not_performed", "run_id": str(run.id), "url": str(run.url)}


def run(manifest_path, output, *, resume=False, device_name="cpu", wandb_project=None,
        allow_online=False, fault_after_steps=None, fault_after_unit=None, max_units=None):
    manifest, arrays, input_record = load_inputs(manifest_path)
    config = manifest["config"]
    require((fault_after_steps is None and fault_after_unit is None) or manifest["evidence_kind"] == "synthetic",
            "fault injection is synthetic-test-only")
    if max_units is not None:
        integer(max_units, 1)
    require(device_name == "cpu" or device_name == "cuda", "device must be cpu or cuda")
    device = torch.device(device_name)
    require(device.type != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    require(not wandb_project or (allow_online and manifest["allow_aggregate_wandb"]), "W&B requires explicit online opt-in and governance")
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"), "set deterministic CUBLAS workspace before CUDA launch")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    output = Path(output).resolve()
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    require(output.is_dir(), "resume directory absent")
    lock_path = output / "WRITER.lock"
    with lock_path.open("xb") as stream:
        stream.write(json_bytes({"nonce": uuid.uuid4().hex, "scope": "exclusive_local_writer"}))
        stream.flush()
        os.fsync(stream.fileno())
    journal = None
    started = time.perf_counter()
    try:
        env = environment(device)
        binding = {"manifest_sha256": input_record["manifest_sha256"], "runner_sha256": sha(Path(__file__)),
                   "environment": env, "wandb_project": wandb_project, "allow_online": bool(allow_online)}
        if resume:
            require(not (output / "COMPLETE.json").exists(), "completed run cannot be resumed or overwritten")
            latest = read_json(output / "LATEST.json")
            exact(latest, {"path", "sha256", "cursor"}, "latest checkpoint")
            state = torch.load(relative_file(output, {k: latest[k] for k in ("path", "sha256")}),
                               map_location="cpu", weights_only=True)
            require(state["binding"] == binding and state["cursor"] == latest["cursor"], "resume input/code/environment drift")
            checkpoint = {k: latest[k] for k in ("path", "sha256")}
        else:
            state = {"binding": binding, "cursor": 0, "heads": {a: {} for a in ARMS}, "active_optimizer": None,
                     "initial_head_sha256": {}, "measurements": [], "metrics": {a: [] for a in ARMS}, "commits": []}
            atomic_json(output / "INPUT_BINDING.json", {**binding, "dataset": manifest["dataset"],
                         "evidence_kind": manifest["evidence_kind"], "encoder_binding_sha256": object_hash(manifest["encoder_binding"]),
                         "pretrain_cost": input_record["pretrain_cost"], "scope": "frozen_embedding_head_objective_only"})
            save_checkpoint(output, state)
        journal = Journal(output)
        journal.emit({"event": "attempt_started", "start_cursor": state["cursor"], "load_verify_seconds": input_record["load_verify_seconds"]})
        heads = load_heads(state, arrays["train"]["embeddings"].shape[1], config, device)
        # Recovered tensor snapshots are not retained alongside live models in memory.
        state["heads"] = {a: {} for a in ARMS}
        buffers = {}
        units = units_for(manifest["tasks"], config["epochs"])
        start_cursor = state["cursor"]
        for index in range(state["cursor"], len(units)):
            unit = units[index]
            if unit["kind"] == "train":
                # Old-task replay is materialized only after those tasks have arrived.
                buffers = buffers_for(arrays["train"], manifest["tasks"][:unit["task"]], config)
                # Heads are frozen after their own last epoch; only the active class updates.
                for arm in ARMS:
                    if unit["class_id"] in heads[arm]:
                        for parameter in heads[arm][unit["class_id"]].parameters():
                            parameter.requires_grad_(True)
                measured = train_unit(unit, heads, state, arrays["train"], manifest["tasks"], buffers,
                                      config, device, journal, fault_after_steps)
                state["measurements"].append({"unit": index, **unit, "arms": measured})
            else:
                for arm in ARMS:
                    for model in heads[arm].values():
                        model.eval()
                    state["metrics"][arm].append(evaluate(heads[arm], arrays["test"], unit["task"], config, device, state["metrics"][arm]))
                buffers = buffers_for(arrays["train"], manifest["tasks"][:unit["task"] + 1], config)
            state["cursor"] = index + 1
            state["commits"].append({"journal": journal.path.name, "last_sequence": journal.sequence, "cursor": state["cursor"]})
            state["heads"] = {arm: {str(c): {name: t.detach().cpu().clone() for name, t in model.state_dict().items()}
                                   for c, model in heads[arm].items()} for arm in ARMS}
            checkpoint = save_checkpoint(output, state)
            journal.emit({"event": "checkpoint_committed", "cursor": state["cursor"], "checkpoint_sha256": checkpoint["sha256"]})
            state["heads"] = {a: {} for a in ARMS}
            if fault_after_unit is not None and state["cursor"] >= fault_after_unit:
                raise RuntimeError("synthetic_injected_checkpoint_boundary_interruption")
            if max_units is not None and state["cursor"] - start_cursor >= max_units and state["cursor"] < len(units):
                verify_input_bindings(manifest_path, manifest, binding["manifest_sha256"])
                require(sha(Path(__file__)) == binding["runner_sha256"], "runner changed during execution")
                journal.emit({"event": "attempt_completed", "elapsed_seconds": time.perf_counter() - started})
                journal.close()
                journal = None
                atomic_json(output / "COST_LEDGER.json", cost_ledger(output, state["cursor"], state["commits"]))
                paused = {"status": "PAUSED", "evidence_kind": manifest["evidence_kind"], "cursor": state["cursor"],
                          "scope": "bounded_committed_prefix_not_completed_study", "checkpoint": checkpoint,
                          "measurements": state["measurements"], "binding": binding}
                atomic_json(output / "PAUSED.json", paused)
                return paused
        verify_input_bindings(manifest_path, manifest, binding["manifest_sha256"])
        require(sha(Path(__file__)) == binding["runner_sha256"], "runner changed during execution")
        final_hashes = {arm: {str(c): tensor_state_hash(model.state_dict()) for c, model in heads[arm].items()} for arm in ARMS}
        result = {"schema_version": 1, "status": "COMPLETE", "evidence_kind": manifest["evidence_kind"],
                  "scope": "L1_binary_head_objective_control_not_full_OFRA_architecture_comparison",
                  "binding": binding, "dataset": manifest["dataset"], "config": config,
                  "group_mode": manifest["group_mode"], "initial_head_sha256": state["initial_head_sha256"],
                  "final_head_sha256": final_hashes, "measurements": state["measurements"], "metrics": state["metrics"],
                  "final_checkpoint": checkpoint, "shared_pretrain_cost": input_record["pretrain_cost"],
                  "embedding_stage_cost_included": False,
                  "common_input_logical_array_bytes": {s: {k: int(v.nbytes) for k, v in data.items()} for s, data in arrays.items()},
                  "common_replay_index_bytes": sum(int(x.nbytes) for x in buffers.values()),
                  "encoder_resident_bytes": 0, "encoder_residency_reason": "precomputed_embedding_stage",
                  "centroid_bytes": 0, "centroid_reason": "L1_head_only_no_router",
                  "process_memory": process_memory(), "wall_seconds_this_attempt": time.perf_counter() - started,
                  "wandb": "disabled" if not wandb_project else "requested_aggregate_only",
                  "limitations": ["No full-training memory equivalence: encoder/export costs are separate.",
                                  "Tensor storage is sampled at step boundaries; process peaks are OS lifetime peaks.",
                                  "Both heads coexist in this harness; process peaks and timing are not isolated per-method profiles.",
                                  "Whole-manifest integrity validation sees future-file metadata; fitting/sampling uses only arrived classes."]}
        journal.emit({"event": "attempt_completed", "elapsed_seconds": time.perf_counter() - started})
        journal.close()
        journal = None
        atomic_json(output / "COST_LEDGER.json", cost_ledger(output, state["cursor"], state["commits"]))
        if wandb_project:
            result["wandb"] = aggregate_wandb(result, wandb_project, manifest["allow_aggregate_wandb"])
        atomic_json(output / "result.json", result)
        atomic_json(output / "COMPLETE.json", {"result": {"path": "result.json", "sha256": sha(output / "result.json")},
                     "cost_ledger": {"path": "COST_LEDGER.json", "sha256": sha(output / "COST_LEDGER.json")},
                     "final_checkpoint": checkpoint, "status": "COMPLETE", "evidence_kind": manifest["evidence_kind"]})
        return result
    except BaseException as exc:
        if journal is not None:
            journal.emit({"event": "attempt_failed", "failure_type": type(exc).__name__,
                          "elapsed_seconds": time.perf_counter() - started})
            journal.close()
        if (output / "LATEST.json").exists():
            latest = read_json(output / "LATEST.json")
            saved = torch.load(relative_file(output, {k: latest[k] for k in ("path", "sha256")}), map_location="cpu", weights_only=True)
            atomic_json(output / "COST_LEDGER.json", cost_ledger(output, saved["cursor"], saved["commits"]))
        raise
    finally:
        lock_path.unlink()


def audit(output, manifest_path):
    output = Path(output).resolve()
    complete = read_json(output / "COMPLETE.json")
    exact(complete, {"result", "cost_ledger", "final_checkpoint", "status", "evidence_kind"}, "completion receipt")
    require(complete["status"] == "COMPLETE", "run incomplete")
    result = read_json(relative_file(output, complete["result"]))
    ledger = read_json(relative_file(output, complete["cost_ledger"]))
    checkpoint_path = relative_file(output, complete["final_checkpoint"])
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    manifest, arrays, inputs = load_inputs(manifest_path)
    require(inputs["manifest_sha256"] == result["binding"]["manifest_sha256"], "audit input drift")
    require(state["binding"] == result["binding"] and state["metrics"] == result["metrics"]
            and state["measurements"] == result["measurements"], "checkpoint/result content mismatch")
    require(state["initial_head_sha256"] == result["initial_head_sha256"], "initial state record mismatch")
    for arm in ARMS:
        require({c: tensor_state_hash(tensors) for c, tensors in state["heads"][arm].items()} == result["final_head_sha256"][arm],
                "final model hash mismatch")
        for row in result["metrics"][arm]:
            recomputed = confusion_metrics(row["confusion"], row["class_axis"])
            require(all(row[k] == value for k, value in recomputed.items()), "confusion metric mismatch")
    for unit in result["measurements"]:
        a, b = (unit["arms"][arm] for arm in ARMS)
        require(a["counters"] == b["counters"] and a["batch_stream_sha256"] == b["batch_stream_sha256"], "unequal L1 accounting")
        commit = next(x for x in state["commits"] if x["cursor"] == unit["unit"] + 1)
        safe_name(commit["journal"])
        require((output / commit["journal"]).resolve().is_relative_to(output), "committed journal escapes root")
        events = [json.loads(line) for line in (output / commit["journal"]).read_bytes().splitlines()]
        events = [x for x in events if x["sequence"] <= commit["last_sequence"] and x.get("unit") == unit["unit"]]
        for arm in ARMS:
            starts = [x for x in events if x.get("arm") == arm and x["event"] == "batch_consumed_step_started"]
            ends = [x for x in events if x.get("arm") == arm and x["event"] == "step_completed"]
            require([x["batch_index"] for x in starts] == [x["batch_index"] for x in ends], "committed batch/step receipt mismatch")
            totals = {"optimizer_steps": len(ends), "raw_row_presentations": 0, "binary_targets": 0,
                      "positive_targets": 0, "negative_targets": 0, "old_row_presentations": 0,
                      "new_row_presentations": 0, "zero_negative_steps": 0}
            used = set()
            digest = hashlib.sha256()
            planned = list(batches_for(arrays["train"], manifest["tasks"], buffers_for(arrays["train"], manifest["tasks"][:unit["task"]], manifest["config"]),
                                       manifest["config"], unit["task"], unit["class_id"], unit["epoch"]))
            require(len(planned) == len(starts), "sampler batch count mismatch")
            for event, (indices, labels, _) in zip(starts, planned):
                ids = [bytes(x).decode("ascii") for x in arrays["train"]["row_ids"][indices]]
                available = arrays["train"]["available_tasks"][indices].tolist()
                require(event["row_indices"] == indices.tolist() and event["row_ids"] == ids
                        and event["binary_targets"] == labels.tolist() and event["row_available_tasks"] == available,
                        "consumption receipt differs from bound inputs/sampler")
                require(all(t <= unit["task"] for t in available), "future exposure")
                totals["raw_row_presentations"] += len(ids)
                totals["binary_targets"] += len(labels)
                totals["positive_targets"] += int(labels.sum())
                totals["negative_targets"] += int((labels == 0).sum())
                old = sum(t < unit["task"] for t in available)
                totals["old_row_presentations"] += old
                totals["new_row_presentations"] += len(ids) - old
                totals["zero_negative_steps"] += int(not (labels == 0).any())
                used.update(ids)
                digest.update(json_bytes([ids, labels.tolist()]))
            totals["unique_raw_rows"] = len(used)
            require(totals == unit["arms"][arm]["counters"] and digest.hexdigest() == unit["arms"][arm]["batch_stream_sha256"],
                    "scientific counters differ from durable step receipts")
    require(cost_ledger(output, state["cursor"], state["commits"]) == ledger, "cost journal mismatch")
    require(result["evidence_kind"] == complete["evidence_kind"], "evidence label mismatch")
    return {"status": "AUDIT_PASS", "evidence_kind": result["evidence_kind"], "training_units": len(result["measurements"]),
            "scope": "local_artifact_integrity_and_accounting_not_scientific_reproduction"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("run")
    command.add_argument("manifest", type=Path)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--resume", action="store_true")
    command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    command.add_argument("--wandb-project")
    command.add_argument("--allow-online", action="store_true")
    command.add_argument("--max-units", type=int)
    command = commands.add_parser("audit")
    command.add_argument("output", type=Path)
    command.add_argument("--manifest", type=Path, required=True)
    command = commands.add_parser("validate-input")
    command.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            result = run(args.manifest, args.output, resume=args.resume, device_name=args.device,
                         wandb_project=args.wandb_project, allow_online=args.allow_online, max_units=args.max_units)
            answer = {"status": result["status"], "evidence_kind": result["evidence_kind"], "scope": result["scope"]}
        elif args.command == "audit":
            answer = audit(args.output, args.manifest)
        else:
            manifest, _, _ = load_inputs(args.manifest)
            answer = {"status": "INPUT_VALID", "evidence_kind": manifest["evidence_kind"]}
        print(json.dumps(answer))
        return 0
    except (Invalid, OSError, RuntimeError, ValueError, KeyError, TypeError, IndexError) as exc:
        # Never emit raw exception text: upstream errors may contain user paths.
        print(json.dumps({"status": "BLOCKED_OR_FAILED", "error_type": type(exc).__name__}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
