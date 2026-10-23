"""Section138 standard/embedding MLP building blocks, not an execution entrypoint."""
import copy

import torch
from torch import nn

from ncd.section81_tabtransformer import TabularClassifier, TabularFamily


class EmbeddedFeatures(nn.Module):
    """Reuse the registered Task-0 vocabulary and unknown-token encoding."""
    encode = TabularClassifier.encode

    def __init__(self, spec, hidden, layers):
        super().__init__()
        self.spec = copy.deepcopy(spec)
        if spec.get("test_used_for_fitting") is not False or spec.get("unknown_id") != 0:
            raise ValueError("Task-0 TRAIN vocabulary required")
        self.register_buffer("continuous_columns", torch.tensor(spec["continuous_columns"], dtype=torch.long))
        for i, group in enumerate(spec["groups"]):
            self.register_buffer(f"columns_{i}", torch.tensor(group["columns"], dtype=torch.long))
            dtype = torch.long if group["kind"] == "onehot" else torch.float32
            self.register_buffer(f"vocabulary_{i}", torch.tensor(group["values"], dtype=dtype))
        self.embeddings = nn.ModuleList(nn.Embedding(len(g["values"]) + 1, 8)
                                       for g in spec["groups"])
        width = len(spec["continuous_columns"]) + 8 * len(spec["groups"])
        self.network = dense_features(width, hidden, layers)

    def forward(self, x):
        tokens, continuous = self.encode(x)
        encoded = [embedding(tokens[:, i]) for i, embedding in enumerate(self.embeddings)]
        return self.network(torch.cat([continuous, *encoded], dim=1))


def dense_features(inputs, hidden, layers):
    if (hidden, layers) not in ((128, 2), (512, 3)):
        raise ValueError("Unregistered section138 capacity")
    modules = []
    for index in range(layers):
        modules.extend((nn.Linear(inputs if index == 0 else hidden, hidden), nn.ReLU()))
    return nn.Sequential(*modules)


class Classifier(nn.Module):
    """Expose features and head for the existing retention/WA interfaces."""
    def __init__(self, inputs, outputs, hidden=128, layers=2, *, spec=None):
        super().__init__()
        self.features = (dense_features(inputs, hidden, layers) if spec is None
                         else EmbeddedFeatures(spec, hidden, layers))
        self.head = nn.Linear(hidden, outputs)

    def forward(self, x):
        return self.head(self.features(x))


class Family(TabularFamily):
    """Fixed attack output mask and DER interface with an independent MLP."""
    def __init__(self, inputs, outputs, hidden=128, layers=2, *, spec=None):
        nn.Module.__init__(self)
        self.method = "aacil_kd_wa"  # DER dispatch only; postprocessing is explicit.
        self.student = Classifier(inputs, outputs, hidden, layers, spec=spec)
        self.register_buffer("seen_mask", torch.zeros(outputs, dtype=torch.bool))
        self.initial_parameter_count = sum(p.numel() for p in self.parameters())

    def metadata(self):
        return dict(family_type="section138_mlp", total_parameters=self.assert_capacity(),
                    output_attack_classes=self.head.out_features,
                    seen_attack_columns=torch.where(self.seen_mask)[0].cpu().tolist(),
                    constant_across_tasks=True, extra_components=0,
                    head_policy="fixed full attack head; unseen columns masked")


def last_layer_cbf(model, memory, mapping, config, seed, device):
    """Section138 D3: call after WA; exactly one retained-memory epoch, head only."""
    from ncd.aacil_compact import compact_balanced_finetune
    parameters = list(model.parameters())
    original_flags = [p.requires_grad for p in parameters]
    head_ids = {id(p) for p in model.head.parameters()}
    try:
        for p in parameters:
            p.grad = None
            p.requires_grad_(id(p) in head_ids)
        elapsed, stats = compact_balanced_finetune(model, memory, mapping, config, seed, device)
        stats.update(trainable_scope="last_layer_only", required_order="after_weight_alignment")
        return elapsed, stats
    finally:
        for p, flag in zip(parameters, original_flags):
            p.grad = None
            p.requires_grad_(flag)


def postprocess_family(model, retention, old_columns, new_columns,
                       memory, mapping, config, seed, device):
    """Explicit D1/D2/D3 ordering; does not alter stored admission logits."""
    from ncd.aacil_retention import apply_weight_alignment
    if retention not in (1, 2, 3):
        raise ValueError("Unregistered retention")
    result = {"retention_level": retention, "operations": [],
              "weight_alignment_seconds": 0.0, "cbf_seconds": 0.0}
    if retention >= 2:
        result["weight_alignment"] = apply_weight_alignment(model, old_columns, new_columns)
        result["weight_alignment_seconds"] = result["weight_alignment"]["elapsed_seconds"]
        result["operations"].append("weight_alignment")
    if retention == 3:
        elapsed, stats = last_layer_cbf(model, memory, mapping, config, seed, device)
        result["cbf"] = stats
        result["cbf_seconds"] = elapsed
        result["operations"].append("last_layer_cbf")
    return result
