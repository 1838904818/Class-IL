"""Fixed B1 TabTransformer interface; vocabulary fitting takes Task-0 TRAIN only.

Uses the external implementation pinned by ReplayIDS b154c79a, not a substitute
FT-Transformer. Training, calibration and replay are supplied by the B1 runner.
"""
from __future__ import annotations

import copy
from importlib.metadata import version

import numpy as np
import torch
from torch import nn


DEPENDENCIES = {"tab-transformer-pytorch": "0.4.2", "hyper-connections": "0.2.1",
                "einops": "0.8.1"}
CIC_CATEGORICAL = (
    "Destination Port", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count",
    "ECE Flag Count",
)


def dependency_versions():
    actual = {name: version(name) for name in DEPENDENCIES}
    if actual != DEPENDENCIES:
        raise RuntimeError(f"B1 dependency versions differ: {actual}")
    return actual


def fit_feature_spec(dataset, feature_names, task0_training_x):
    """Caller must pass only optimization-eligible Task-0 TRAIN rows, never TEST."""
    names = list(feature_names)
    x = np.asarray(task0_training_x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != len(names) or not len(x):
        raise ValueError("Nonempty Task-0 training matrix and unique schema required")
    if len(set(names)) != len(names) or not np.isfinite(x).all():
        raise ValueError("Invalid feature schema or nonfinite Task-0 values")
    groups = []
    if dataset == "UNSW-NB15":
        for field in ("proto", "service", "state"):
            columns = [i for i, name in enumerate(names)
                       if name.startswith(f"categorical__{field}_")]
            if not columns:
                raise ValueError(f"Missing one-hot block: {field}")
            block = x[:, columns]
            if not np.isin(block, [0., 1.]).all() or np.any(block.sum(1) > 1):
                raise ValueError(f"Invalid one-hot block: {field}")
            # Only categories actually present in these TRAIN rows enter the vocabulary.
            observed = np.flatnonzero(block.sum(0) > 0).tolist()
            groups.append({"name": field, "columns": columns, "kind": "onehot",
                           "values": observed})
    elif dataset == "CIC-IDS-2017":
        for field in CIC_CATEGORICAL:
            column = names.index(f"numeric__{field}")
            groups.append({"name": field, "columns": [column], "kind": "numeric",
                           "values": np.unique(x[:, column]).tolist()})
    else:
        raise ValueError("B1 supports only UNSW-NB15 and CIC-IDS-2017")
    categorical = {i for group in groups for i in group["columns"]}
    return {"dataset": dataset, "feature_names": names, "groups": groups,
            "continuous_columns": [i for i in range(len(names)) if i not in categorical],
            "fit_scope": "Task-0 optimization-eligible TRAIN only", "unknown_id": 0,
            "fit_rows": len(x), "test_used_for_fitting": False}


class TabularClassifier(nn.Module):
    """Original cache tensors in, logits out; final Linear exposed for AR WA."""

    def __init__(self, spec, outputs):
        super().__init__()
        self.versions = dependency_versions()
        from tab_transformer_pytorch import TabTransformer

        self.spec = copy.deepcopy(spec)
        self.register_buffer("continuous_columns", torch.tensor(spec["continuous_columns"], dtype=torch.long))
        for i, group in enumerate(spec["groups"]):
            self.register_buffer(f"columns_{i}", torch.tensor(group["columns"], dtype=torch.long))
            dtype = torch.long if group["kind"] == "onehot" else torch.float32
            self.register_buffer(f"vocabulary_{i}", torch.tensor(group["values"], dtype=dtype))
        self.model = TabTransformer(
            categories=tuple(len(g["values"]) + 1 for g in spec["groups"]),
            num_continuous=len(spec["continuous_columns"]), dim=32, depth=6,
            heads=10, dim_head=16, dim_out=int(outputs), mlp_hidden_mults=(4, 2),
            mlp_act=nn.GELU(), attn_dropout=0.1, ff_dropout=0.1,
        )

    @property
    def head(self):
        return self.model.mlp.mlp[-1]

    def encode(self, x):
        tokens = []
        for i, group in enumerate(self.spec["groups"]):
            block = x.index_select(1, getattr(self, f"columns_{i}"))
            vocab = getattr(self, f"vocabulary_{i}")
            if group["kind"] == "onehot":
                values = block.argmax(1)
                present = block.sum(1) == 1
            else:
                values = block[:, 0].contiguous()
                present = torch.isfinite(values)
            if vocab.numel():
                indices = torch.searchsorted(vocab, values)
                safe = indices.clamp(max=vocab.numel() - 1)
                known = present & (indices < vocab.numel()) & (vocab[safe] == values)
                tokens.append(torch.where(known, indices + 1, 0))
            else:
                tokens.append(torch.zeros(len(x), dtype=torch.long, device=x.device))
        return torch.stack(tokens, dim=1), x.index_select(1, self.continuous_columns)

    def forward(self, x):
        return self.model(*self.encode(x))


class TabularFamily(nn.Module):
    """AR's fixed attack head and DER/WA interface on the registered backbone."""

    def __init__(self, spec, outputs):
        super().__init__()
        self.method = "aacil_kd_wa"  # Existing DER/WA dispatch, not result identity.
        self.student = TabularClassifier(spec, outputs)
        self.register_buffer("seen_mask", torch.zeros(outputs, dtype=torch.bool))
        self.initial_parameter_count = sum(p.numel() for p in self.parameters())

    @property
    def head(self):
        return self.student.head

    def set_seen(self, columns):
        columns = sorted(set(int(c) for c in columns))
        if not columns or columns[0] < 0 or columns[-1] >= len(self.seen_mask):
            raise ValueError("Invalid seen attack columns")
        updated = torch.zeros_like(self.seen_mask)
        updated[columns] = True
        if torch.any(self.seen_mask & ~updated):
            raise ValueError("Seen attack columns cannot shrink")
        self.seen_mask.copy_(updated)

    def mask(self, logits):
        return logits.masked_fill(~self.seen_mask.unsqueeze(0), -1e6)

    def student_logits(self, x):
        return self.student(x)

    def entry_logits(self, x):
        return self.student(x)

    def forward(self, x):
        return self.mask(self.student(x))

    def update_ema(self):
        pass  # AR has no EMA component.

    def assert_capacity(self):
        count = sum(p.numel() for p in self.parameters())
        if count != self.initial_parameter_count:
            raise AssertionError("B1 fixed TabTransformer parameter count changed")
        return count

    def metadata(self):
        return {"family_type": "ar_tabtransformer", "total_parameters": self.assert_capacity(),
                "output_attack_classes": self.head.out_features,
                "seen_attack_columns": torch.where(self.seen_mask)[0].cpu().tolist(),
                "constant_across_tasks": True, "extra_components": 0,
                "backbone": "tab-transformer-pytorch", "dependencies": self.student.versions,
                "head_policy": "fixed full attack head; unseen columns masked",
                "weight_alignment_target": "final output Linear weight rows",
                "mlp_capacity_bound_applied": False}
